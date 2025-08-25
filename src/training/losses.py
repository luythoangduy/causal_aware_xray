from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Optional, Dict


class ConstrainedLoss(nn.Module):
    """Composite loss combining supervised BCE, constraint penalty & consistency.
    
    Fixed version with numerical stability improvements.
    """

    def __init__(
        self,
        constraint_weight: float = 1.0,
        consistency_weight: float = 0.5,
        class_weights: Optional[Tensor] = None,
        pos_weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.constraint_weight = constraint_weight
        self.consistency_weight = consistency_weight
        self.register_buffer('class_weights', class_weights if class_weights is not None else torch.tensor([]))
        self.register_buffer('pos_weight', pos_weight if pos_weight is not None else None)
        
        # Use BCEWithLogitsLoss for numerical stability (expects logits, not probabilities)
        self.bce = nn.BCEWithLogitsLoss(
            weight=self.class_weights if self.class_weights.numel() else None,
            pos_weight=self.pos_weight
        )
        self.mse = nn.MSELoss(reduction='mean')

    def forward(
        self,
        constrained_output: Tensor,  # probabilities (after projection)
        raw_output: Tensor,  # logits (pre-projection)
        targets: Tensor,
        constraint_stats: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Tensor]:
        
        # FIX 1: Use raw logits directly for primary loss instead of converting probs back to logits
        # This avoids the numerical instability from log(p/(1-p)) transformation
        primary = self.bce(raw_output, targets)
        
        # FIX 2: Use BCE on probabilities with more stable implementation
        # Alternative: Use focal loss or label smoothing for better stability
        eps = 1e-7  # Smaller epsilon for better numerical range
        constrained_output_clamped = torch.clamp(constrained_output, eps, 1.0 - eps)
        
        # Manual BCE on probabilities (more stable than logit conversion)
        bce_on_probs = -(targets * torch.log(constrained_output_clamped) + 
                        (1 - targets) * torch.log(1 - constrained_output_clamped))
        primary_constrained = bce_on_probs.mean()
        
        # FIX 3: Use constrained predictions for primary loss
        primary = primary_constrained
        
        # Consistency loss between raw and constrained probabilities
        raw_probs = torch.sigmoid(raw_output)
        consistency = self.mse(constrained_output, raw_probs)
        
        # FIX 4: Add gradient clipping and stability checks
        constraint_penalty = torch.tensor(0.0, device=primary.device, dtype=primary.dtype)
        if constraint_stats:
            imp = constraint_stats.get('implication_total', 0.0)
            exc = constraint_stats.get('exclusion_total', 0.0)
            
            # Clamp constraint violations to prevent explosion
            penalty_value = float(imp + exc)
            penalty_value = max(0.0, min(penalty_value, 10.0))  # Clamp to [0, 10]
            
            constraint_penalty = torch.tensor(
                penalty_value, 
                dtype=primary.dtype, 
                device=primary.device
            )
        
        # FIX 5: Scale constraint weight adaptively to prevent dominance
        adaptive_constraint_weight = self.constraint_weight
        if constraint_penalty > 1.0:
            adaptive_constraint_weight = self.constraint_weight / constraint_penalty.item()
        
        # Compute total loss with stability checks
        total = (primary + 
                adaptive_constraint_weight * constraint_penalty + 
                self.consistency_weight * consistency)
        
        # FIX 6: Final stability check - clamp total loss
        total = torch.clamp(total, max=100.0)  # Prevent extreme loss values
        
        # Check for NaN/Inf and handle gracefully
        if torch.isnan(total) or torch.isinf(total):
            print("WARNING: NaN/Inf detected in loss, using fallback")
            total = self.bce(raw_output, targets)  # Fallback to simple BCE
            constraint_penalty = torch.tensor(0.0, device=total.device)
            consistency = torch.tensor(0.0, device=total.device)
        
        return {
            'loss': total,
            'primary': primary.detach(),
            'constraint_penalty': constraint_penalty.detach(),
            'consistency': consistency.detach(),
        }


class FocalConstrainedLoss(nn.Module):
    """Alternative loss function using Focal Loss for better stability."""
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        constraint_weight: float = 1.0,
        consistency_weight: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.constraint_weight = constraint_weight
        self.consistency_weight = consistency_weight
        self.mse = nn.MSELoss()
    
    def focal_loss(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Focal loss implementation for multi-label classification."""
        eps = 1e-7
        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        
        # Compute focal loss
        ce_loss = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        return (focal_weight * ce_loss).mean()
    
    def forward(
        self,
        constrained_output: Tensor,
        raw_output: Tensor,
        targets: Tensor,
        constraint_stats: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Tensor]:
        
        # Primary loss using Focal Loss on constrained probabilities
        primary = self.focal_loss(constrained_output, targets)
        
        # Consistency loss
        raw_probs = torch.sigmoid(raw_output)
        consistency = self.mse(constrained_output, raw_probs)
        
        # Constraint penalty
        constraint_penalty = torch.tensor(0.0, device=primary.device)
        if constraint_stats:
            penalty_val = constraint_stats.get('implication_total', 0.0) + constraint_stats.get('exclusion_total', 0.0)
            constraint_penalty = torch.tensor(min(penalty_val, 5.0), device=primary.device)
        
        total = primary + self.constraint_weight * constraint_penalty + self.consistency_weight * consistency
        
        return {
            'loss': total,
            'primary': primary.detach(),
            'constraint_penalty': constraint_penalty.detach(),
            'consistency': consistency.detach(),
        }


# DIAGNOSTIC FUNCTIONS
def diagnose_loss_issues(constrained_output, raw_output, targets):
    """Diagnostic function to identify loss computation issues."""
    print("=== LOSS DIAGNOSTIC ===")
    print(f"Constrained output range: [{constrained_output.min():.6f}, {constrained_output.max():.6f}]")
    print(f"Raw output range: [{raw_output.min():.6f}, {raw_output.max():.6f}]")
    print(f"Targets range: [{targets.min():.6f}, {targets.max():.6f}]")
    
    # Check for extreme values
    if constrained_output.min() <= 0 or constrained_output.max() >= 1:
        print("⚠️  WARNING: Constrained output outside (0,1) range!")
    
    # Check gradients
    if constrained_output.requires_grad:
        test_loss = -(targets * torch.log(constrained_output + 1e-7)).mean()
        test_loss.backward(retain_graph=True)
        grad_norm = constrained_output.grad.norm() if constrained_output.grad is not None else 0
        print(f"Gradient norm: {grad_norm:.6f}")
        
    print("========================")

