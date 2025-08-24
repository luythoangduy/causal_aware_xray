from __future__ import annotations
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.models.classifier import ConstrainedChestXRayClassifier
from src.models.constraint_projection import ConstraintProjection


@pytest.fixture
def mock_xrv_model():
    """Mock TorchXRayVision model to avoid loading actual weights."""
    mock_model = MagicMock()
    mock_model.classifier = MagicMock()
    mock_model.classifier.in_features = 1024
    mock_model.classifier.out_features = 18  # Original XRV output
    mock_model.op_threshs = torch.randn(18)  # Mock thresholds
    
    # Mock forward pass
    def mock_forward(x):
        batch_size = x.shape[0]
        return torch.randn(batch_size, 14)  # Return 14 classes after our modification
    mock_model.side_effect = mock_forward
    
    return mock_model


@patch('src.models.classifier.xrv.models.DenseNet')
def test_classifier_initialization(mock_densenet_class):
    """Test classifier initializes correctly."""
    mock_densenet_class.return_value = MagicMock()
    mock_densenet_class.return_value.classifier = torch.nn.Linear(1024, 18)
    mock_densenet_class.return_value.op_threshs = torch.randn(18)
    
    classifier = ConstrainedChestXRayClassifier(
        backbone='densenet121',
        pretrained=True,
        num_classes=14
    )
    
    assert classifier.num_classes == 14
    assert classifier.projection_mode == 'integrated'
    assert mock_densenet_class.called
    # Check classifier was replaced
    assert classifier.backbone.classifier.out_features == 14


@patch('src.models.classifier.xrv.models.DenseNet')
def test_classifier_forward_no_constraints(mock_densenet_class):
    """Test forward pass without constraints."""
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(2, 14)
    mock_densenet_class.return_value = mock_model
    
    classifier = ConstrainedChestXRayClassifier(
        backbone='densenet121',
        num_classes=14,
        constraint_projection=None
    )
    
    x = torch.randn(2, 1, 224, 224)
    output = classifier(x)
    
    assert output.shape == (2, 14)
    mock_model.assert_called_once()


@patch('src.models.classifier.xrv.models.DenseNet')
def test_classifier_with_constraints(mock_densenet_class):
    """Test forward pass with constraints."""
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(2, 14)
    mock_densenet_class.return_value = mock_model
    
    # Create constraint projection
    constraint_proj = ConstraintProjection(
        implications=[(0, 1, 0.0)],
        exclusions=[(2, 3, 1.0)],
        max_iter=5
    )
    
    classifier = ConstrainedChestXRayClassifier(
        backbone='densenet121',
        num_classes=14,
        constraint_projection=constraint_proj,
        projection_mode='integrated'
    )
    
    x = torch.randn(2, 1, 224, 224)
    output = classifier(x)
    
    assert output.shape == (2, 14)
    # Output should be probabilities (after constraint projection)
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


@patch('src.models.classifier.xrv.models.DenseNet')
def test_classifier_post_process_mode(mock_densenet_class):
    """Test post-process mode returns logits."""
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(2, 14)
    mock_densenet_class.return_value = mock_model
    
    constraint_proj = ConstraintProjection(implications=[(0, 1, 0.0)])
    
    classifier = ConstrainedChestXRayClassifier(
        backbone='densenet121',
        num_classes=14,
        constraint_projection=constraint_proj,
        projection_mode='post_process'
    )
    
    x = torch.randn(2, 1, 224, 224)
    output = classifier(x, apply_constraints=False)
    
    # Should return raw logits (can be negative)
    assert output.shape == (2, 14)


@patch('src.models.classifier.xrv.models.DenseNet')
def test_classifier_project_method(mock_densenet_class):
    """Test the project method works correctly."""
    mock_model = MagicMock()
    mock_densenet_class.return_value = mock_model
    
    constraint_proj = ConstraintProjection(implications=[(0, 1, 0.0)])
    
    classifier = ConstrainedChestXRayClassifier(
        backbone='densenet121',
        num_classes=14,
        constraint_projection=constraint_proj
    )
    
    logits = torch.randn(2, 14)
    projected = classifier.project(logits)
    
    assert projected.shape == (2, 14)
    assert torch.all(projected >= 0.0)
    assert torch.all(projected <= 1.0)


def test_unsupported_backbone():
    """Test error handling for unsupported backbone."""
    with pytest.raises(ValueError, match="Unsupported backbone"):
        ConstrainedChestXRayClassifier(backbone='resnet50')