from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Optional
import torchxrayvision as xrv
import torchvision
from .constraint_projection import ConstraintProjection


class ConstrainedChestXRayClassifier(nn.Module):
    """Wrap a backbone and (optionally) apply constraint projection.

    projection_mode:
      - 'integrated': constraints applied to sigmoid probabilities and returned
      - 'post_process': forward returns raw logits; a helper method applies constraints
    """

    def __init__(
        self,
        backbone: str = 'densenet121',
        pretrained: bool = True,
        num_classes: int = 14,
        constraint_projection: Optional[ConstraintProjection] = None,
        projection_mode: str = 'integrated',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.projection_mode = projection_mode

        if backbone == 'densenet121':
            if pretrained:
                self.backbone = xrv.models.DenseNet(weights="densenet121-res224-nih")
            else:
                self.backbone = torchvision.models.densenet121(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Replace classifier to match our number of classes
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
        
        # Disable op_norm (output normalization) that causes dimension mismatch
        if hasattr(self.backbone, 'op_threshs'):
            self.backbone.op_threshs = None

        self.constraint_projection = constraint_projection

    def forward(self, x: Tensor, apply_constraints: Optional[bool] = None):
        logits = self.backbone(x)
        if self.constraint_projection is None:
            return logits
        # Decide mode
        if apply_constraints is None:
            mode = self.projection_mode
        else:
            mode = 'integrated' if apply_constraints else 'post_process'
        if mode == 'integrated':
            return self.constraint_projection(logits)
        # post_process mode returns raw logits
        return logits

    @torch.no_grad()
    def project(self, logits: Tensor) -> Tensor:
        if self.constraint_projection is None:
            return torch.sigmoid(logits)
        return self.constraint_projection(logits)
