from __future__ import annotations
import pytest
import torch

from src.training.losses import ConstrainedLoss


def test_constrained_loss_initialization():
    """Test loss function initializes correctly."""
    loss_fn = ConstrainedLoss(
        constraint_weight=1.0,
        consistency_weight=0.5
    )
    
    assert loss_fn.constraint_weight == 1.0
    assert loss_fn.consistency_weight == 0.5


def test_constrained_loss_forward():
    """Test loss forward pass works correctly."""
    loss_fn = ConstrainedLoss()
    
    # Mock data
    batch_size, num_classes = 4, 14
    constrained_output = torch.rand(batch_size, num_classes)  # Probabilities
    raw_output = torch.randn(batch_size, num_classes)  # Logits
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    constraint_stats = {
        'implication_total': 0.1,
        'exclusion_total': 0.05
    }
    
    losses = loss_fn(constrained_output, raw_output, targets, constraint_stats)
    
    # Check output structure
    assert 'loss' in losses
    assert 'primary' in losses
    assert 'constraint_penalty' in losses
    assert 'consistency' in losses
    
    # Check tensor shapes and properties
    assert losses['loss'].shape == ()  # Scalar
    assert losses['loss'].item() > 0  # Loss should be positive
    
    # Check components are positive
    assert losses['primary'].item() >= 0
    assert losses['constraint_penalty'].item() >= 0
    assert losses['consistency'].item() >= 0


def test_constrained_loss_no_constraint_stats():
    """Test loss works without constraint statistics."""
    loss_fn = ConstrainedLoss()
    
    batch_size, num_classes = 2, 14
    constrained_output = torch.rand(batch_size, num_classes)
    raw_output = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    losses = loss_fn(constrained_output, raw_output, targets)
    
    assert losses['constraint_penalty'].item() == 0.0  # Should be zero without stats


def test_constrained_loss_with_class_weights():
    """Test loss with class weights."""
    class_weights = torch.rand(14)
    loss_fn = ConstrainedLoss(class_weights=class_weights)
    
    batch_size, num_classes = 2, 14
    constrained_output = torch.rand(batch_size, num_classes)
    raw_output = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    losses = loss_fn(constrained_output, raw_output, targets)
    
    assert losses['loss'].item() > 0


def test_loss_components_scaling():
    """Test that loss component weights work correctly."""
    # Test with high constraint weight
    loss_fn_high_constraint = ConstrainedLoss(
        constraint_weight=10.0,
        consistency_weight=0.1
    )
    
    # Test with high consistency weight
    loss_fn_high_consistency = ConstrainedLoss(
        constraint_weight=0.1,
        consistency_weight=10.0
    )
    
    batch_size, num_classes = 2, 14
    constrained_output = torch.rand(batch_size, num_classes)
    raw_output = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    constraint_stats = {
        'implication_total': 1.0,
        'exclusion_total': 1.0
    }
    
    losses_1 = loss_fn_high_constraint(constrained_output, raw_output, targets, constraint_stats)
    losses_2 = loss_fn_high_consistency(constrained_output, raw_output, targets, constraint_stats)
    
    # Losses should be different due to different weightings
    assert losses_1['loss'].item() != losses_2['loss'].item()