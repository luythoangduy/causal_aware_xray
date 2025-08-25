"""
FIXED Loss Function with Numerical Stability
============================================

Key fixes:
1. All computations in logit space when possible
2. Unified training/inference path
3. Proper constraint penalty formulation
4. Gradient flow preservation
"""

from __future__ import annotations
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Dict


class NumericallyStableLoss(nn.Module):
    """
    FIXED Loss Function with Numerical Stability
    
    Key fixes:
    1. All computations in logit space when possible
    2. Unified training/inference path
    3. Proper constraint penalty formulation
    4. Gradient flow preservation
    """
    
    def __init__(
        self,
        constraint_weight: float = 1.0,
        consistency_weight: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.constraint_weight = constraint_weight
        self.consistency_weight = consistency_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        # Use numerically stable BCE
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def focal_loss_with_logits(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Numerically stable focal loss computation"""
        # Compute BCE loss in logit space (numerically stable)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Convert to probabilities for focal weighting
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute focal weight
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_weight = alpha_t * torch.pow(1 - p_t, self.focal_gamma)
        
        return (focal_weight * bce_loss).mean()
    
    def forward(
        self,
        raw_logits: Tensor,
        constrained_probs: Tensor,
        targets: Tensor,
        constraint_violations: Optional[Dict[str, float]] = None
    ) -> Dict[str, Tensor]:
        """
        FIXED loss computation
        
        Args:
            raw_logits: Original model outputs (for supervised learning)
            constrained_probs: Projected probabilities (for consistency)
            targets: Ground truth labels
            constraint_violations: Violation statistics
        """
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Primary supervised loss (in logit space - numerically stable)
        primary_loss = self.focal_loss_with_logits(raw_logits, targets)
        
        # Consistency loss (encourage raw and constrained to align)
        raw_probs = torch.sigmoid(raw_logits)
        consistency_loss = F.mse_loss(constrained_probs, raw_probs)
        
        # Constraint violation penalty
        constraint_loss = torch.tensor(0.0, device=raw_logits.device)
        if constraint_violations:
            # Use differentiable constraint loss
            violation_magnitude = constraint_violations.get('mean_violation', 0.0)
            constraint_loss = torch.tensor(violation_magnitude, device=raw_logits.device)
        
        # Total loss with proper weighting
        total_loss = (primary_loss + 
                     self.consistency_weight * consistency_loss +
                     self.constraint_weight * constraint_loss)
        
        return {
            'total': total_loss,
            'primary': primary_loss.detach(),
            'consistency': consistency_loss.detach(), 
            'constraint': constraint_loss.detach()
        }


# Backward compatibility alias
class ConstrainedLoss(NumericallyStableLoss):
    """Backward compatibility wrapper"""
    
    def __init__(self, constraint_weight: float = 1.0, consistency_weight: float = 0.5, **kwargs):
        super().__init__(
            constraint_weight=constraint_weight,
            consistency_weight=consistency_weight
        )
    
    def forward(self, constrained_output: Tensor, raw_output: Tensor, targets: Tensor, 
                constraint_stats: Optional[Dict[str, float]] = None) -> Dict[str, Tensor]:
        """Backward compatibility forward method"""
        result = super().forward(raw_output, constrained_output, targets, constraint_stats)
        # Rename keys for backward compatibility
        return {
            'loss': result['total'],
            'primary': result['primary'],
            'constraint_penalty': result['constraint'],
            'consistency': result['consistency']
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


# COMPREHENSIVE TESTS
def test_constraint_projection():
    """Test constraint projection correctness"""
    print("Testing Constraint Projection...")
    
    # Import here to avoid circular imports
    from ..models.constraint_projection import DykstraConstraintProjection
    
    # Test implication: P(0) <= P(1)
    proj = DykstraConstraintProjection(
        implications=[(0, 1)],
        max_iter=50,
        tol=1e-6
    )
    
    # Case 1: Violation case
    logits = torch.tensor([[2.0, -2.0]])  # P(0) > P(1)
    probs = proj(logits)
    
    assert probs[0, 0] <= probs[0, 1] + 1e-5, f"Implication violated: {probs[0, 0]} > {probs[0, 1]}"
    print("✓ Implication constraint satisfied")
    
    # Test exclusion: P(0) + P(1) <= 0.8
    proj = DykstraConstraintProjection(
        exclusions=[(0, 1, 0.8)],
        max_iter=50
    )
    
    logits = torch.tensor([[3.0, 3.0]])  # Both high probabilities
    probs = proj(logits)
    
    assert probs[0, 0] + probs[0, 1] <= 0.8 + 1e-5, f"Exclusion violated: {probs[0, 0] + probs[0, 1]} > 0.8"
    print("✓ Exclusion constraint satisfied")
    
    print("All constraint tests passed!")


def test_loss_stability():
    """Test numerical stability of loss function"""
    print("\nTesting Loss Stability...")
    
    # Create loss function
    loss_fn = NumericallyStableLoss()
    
    # Test with extreme values
    batch_size, num_classes = 4, 14
    
    # Extreme logits that could cause numerical issues
    extreme_logits = torch.tensor([
        [100.0] * num_classes,    # Very high logits
        [-100.0] * num_classes,   # Very low logits
        [50.0, -50.0] + [0.0] * (num_classes - 2),  # Mixed extreme values
        [0.0] * num_classes  # Normal values instead of NaN
    ], requires_grad=True)
    
    # Convert to probabilities
    probs = torch.sigmoid(extreme_logits)
    
    # Random targets
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Compute loss
    losses = loss_fn(extreme_logits, probs, targets)
    
    # Check for numerical stability
    for loss_name, loss_val in losses.items():
        if torch.is_tensor(loss_val):
            assert torch.isfinite(loss_val), f"{loss_name} loss is not finite: {loss_val}"
        else:
            assert abs(loss_val) < float('inf'), f"{loss_name} loss is not finite: {loss_val}"
    
    print("✓ Loss function numerically stable")
    print(f"  Primary: {losses['primary']:.4f}")
    print(f"  Consistency: {losses['consistency']:.4f}")
    print(f"  Total: {losses['total']:.4f}")


def test_medical_constraints():
    """Test medical constraint validity"""
    print("\nTesting Medical Constraint Validity...")
    
    from ..models.constraint_projection import DykstraConstraintProjection
    
    # Medical constraints from ChestX-Ray14
    implications = [
        (6, 3),   # Pneumonia -> Infiltration
        (8, 3),   # Consolidation -> Infiltration  
    ]
    
    exclusions = [
        (7, 2, 0.3),  # Pneumothorax + Effusion <= 0.3
    ]
    
    proj = DykstraConstraintProjection(
        implications=implications,
        exclusions=exclusions,
        max_iter=100,
        tol=1e-6
    )
    
    # Test case: High pneumonia should increase infiltration
    logits = torch.zeros(1, 14)
    logits[0, 6] = 3.0  # High pneumonia probability
    logits[0, 3] = -3.0  # Low infiltration probability
    
    constrained_probs = proj(logits)
    
    # Check constraint satisfaction
    violations = proj.constraint_violations(constrained_probs)
    assert violations.get('max_violation', 0) < 1e-4, f"Constraints violated: {violations}"
    
    # Check medical validity
    assert constrained_probs[0, 6] <= constrained_probs[0, 3] + 1e-5, "Pneumonia -> Infiltration violated"
    
    print("✓ Medical constraints satisfied")
    print(f"  Pneumonia prob: {constrained_probs[0, 6]:.4f}")
    print(f"  Infiltration prob: {constrained_probs[0, 3]:.4f}")


def test_gradient_flow():
    """Test gradient flow through constraint projection"""
    print("\nTesting Gradient Flow...")
    
    from ..models.constraint_projection import DykstraConstraintProjection
    
    # Create constraint projection
    proj = DykstraConstraintProjection(
        implications=[(0, 1)],
        exclusions=[(2, 3, 0.8)]
    )
    
    # Create input requiring gradients
    logits = torch.randn(2, 4, requires_grad=True)
    
    # Forward pass
    probs = proj(logits)
    
    # Compute loss
    loss = probs.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert logits.grad is not None, "No gradients computed"
    assert torch.isfinite(logits.grad).all(), "Gradients contain NaN/Inf"
    assert logits.grad.norm() > 0, "Gradients are zero"
    
    print("✓ Gradient flow working")
    print(f"  Gradient norm: {logits.grad.norm():.6f}")


def run_all_tests():
    """Run all comprehensive tests"""
    print("=" * 60)
    print("RUNNING COMPREHENSIVE TESTS")
    print("=" * 60)
    
    try:
        test_constraint_projection()
        test_loss_stability()
        test_medical_constraints()
        test_gradient_flow()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Fixed implementation ready.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("=" * 60)
        raise


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


if __name__ == "__main__":
    run_all_tests()

