"""
UNIFIED Model Architecture
=========================

Fixes:
1. No more train/inference mode switching
2. Proper gradient flow
3. Consistent forward pass
"""

from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Optional, Tuple
import torchxrayvision as xrv
import torchvision
from .constraint_projection import DykstraConstraintProjection


class ConstrainedModel(nn.Module):
    """
    UNIFIED Model Architecture
    
    Fixes:
    1. No more train/inference mode switching
    2. Proper gradient flow
    3. Consistent forward pass
    """
    
    def __init__(
        self,
        backbone: str = 'densenet121',
        pretrained: bool = True,
        num_classes: int = 14,
        implications: Optional[list] = None,
        exclusions: Optional[list] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Initialize backbone
        if backbone == 'densenet121':
            if pretrained:
                self.backbone = xrv.models.DenseNet(weights="densenet121-res224-nih")
            else:
                self.backbone = torchvision.models.densenet121(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Get backbone output dimension
        if hasattr(self.backbone, 'classifier'):
            backbone_dim = self.backbone.classifier.in_features
            # Replace classifier
            self.backbone.classifier = nn.Linear(backbone_dim, num_classes)
        else:
            # Fallback: use a dummy input to determine dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                backbone_output = self.backbone(dummy_input)
                backbone_dim = backbone_output.shape[-1]
            # Add classifier
            self.classifier = nn.Linear(backbone_dim, num_classes)
        
        # Disable op_norm (output normalization) that causes dimension mismatch
        if hasattr(self.backbone, 'op_threshs'):
            self.backbone.op_threshs = None
        
        # Constraint projection
        self.constraint_projection = DykstraConstraintProjection(
            implications=implications,
            exclusions=exclusions
        )
        
        # Initialize classifier weights
        if hasattr(self, 'classifier'):
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Unified forward pass for both training and inference
        
        Returns:
            raw_logits: For supervised loss computation
            constrained_probs: For final predictions
        """
        # Feature extraction
        if hasattr(self, 'classifier'):
            features = self.backbone(x)
            raw_logits = self.classifier(features)
        else:
            raw_logits = self.backbone(x)
        
        # Apply constraints (always, for consistent gradient flow)
        constrained_probs = self.constraint_projection(raw_logits)
        
        return raw_logits, constrained_probs
    
    def predict(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        """Get final predictions"""
        _, constrained_probs = self.forward(x)
        return (constrained_probs > threshold).float()


# Backward compatibility wrapper
class ConstrainedChestXRayClassifier(ConstrainedModel):
    """Backward compatibility wrapper"""
    
    def __init__(
        self,
        backbone: str = 'densenet121',
        pretrained: bool = True,
        num_classes: int = 14,
        constraint_projection=None,
        projection_mode: str = 'integrated',
        **kwargs
    ):
        # Extract constraint definitions from constraint_projection if provided
        implications = None
        exclusions = None
        
        if constraint_projection is not None:
            implications = getattr(constraint_projection, 'implications', None)
            exclusions = getattr(constraint_projection, 'exclusions', None)
        
        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            implications=implications,
            exclusions=exclusions
        )
        
        # Store for backward compatibility
        self.projection_mode = projection_mode
    
    def forward(self, x: Tensor, apply_constraints: Optional[bool] = None):
        """Backward compatibility forward method"""
        raw_logits, constrained_probs = super().forward(x)
        
        # Maintain backward compatibility with mode switching
        if apply_constraints is None:
            mode = self.projection_mode
        else:
            mode = 'integrated' if apply_constraints else 'post_process'
        
        if mode == 'integrated':
            return constrained_probs
        else:
            return raw_logits
    
    @torch.no_grad()
    def project(self, logits: Tensor) -> Tensor:
        """Backward compatibility project method"""
        return self.constraint_projection(logits)
