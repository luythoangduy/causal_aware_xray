from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Optional, Dict


class ConstrainedLoss(nn.Module):
    """Composite loss combining supervised BCE, constraint penalty & consistency.

    Args:
        constraint_weight: scales constraint violation penalty
        consistency_weight: scales MSE between raw & constrained outputs
        class_weights: optional per-class weights for BCE
        pos_weight: optional positive class weight (passed to BCEWithLogitsLoss)
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
        self.bce = nn.BCEWithLogitsLoss(weight=self.class_weights if self.class_weights.numel() else None,
                                        pos_weight=self.pos_weight)
        self.mse = nn.MSELoss(reduction='mean')

    def forward(
        self,
        constrained_output: Tensor,  # probabilities (after projection)
        raw_output: Tensor,  # logits (pre-projection)
        targets: Tensor,
        constraint_stats: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Tensor]:
        # Convert constrained probs to logits for BCE stability
        eps = 1e-6
        constrained_logits = torch.log(torch.clamp(constrained_output, eps, 1 - eps) / (1 - torch.clamp(constrained_output, eps, 1 - eps)))
        primary = self.bce(constrained_logits, targets)
        # Consistency
        raw_probs = torch.sigmoid(raw_output)
        consistency = self.mse(constrained_output, raw_probs)
        # Constraint penalty (sum of violation magnitudes pre-projection if provided)
        if constraint_stats:
            imp = constraint_stats.get('implication_total', 0.0)
            exc = constraint_stats.get('exclusion_total', 0.0)
            constraint_penalty = torch.as_tensor(imp + exc, dtype=primary.dtype, device=primary.device)
        else:
            constraint_penalty = torch.tensor(0.0, device=primary.device)

        total = primary + self.constraint_weight * constraint_penalty + self.consistency_weight * consistency
        return {
            'loss': total,
            'primary': primary.detach(),
            'constraint_penalty': constraint_penalty.detach(),
            'consistency': consistency.detach(),
        }
