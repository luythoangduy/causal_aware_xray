from __future__ import annotations
import pytest
import torch
import tempfile
import os
import pandas as pd
from PIL import Image
import numpy as np
from unittest.mock import patch

from src.data.chestxray14 import ChestXRay14Dataset
from src.models.constraint_projection import ConstraintProjection
from src.models.classifier import ConstrainedChestXRayClassifier
from src.training.losses import ConstrainedLoss


@pytest.fixture
def mini_dataset():
    """Create a minimal dataset for integration testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSV
        csv_data = pd.DataFrame({
            'Image Index': ['img_001.png', 'img_002.png'],
            'Finding Labels': ['Pneumonia', 'Effusion|Pneumonia'],
            'Patient ID': [1, 2],
            'Patient Age': [50, 60],
            'Patient Gender': ['M', 'F']
        })
        csv_path = os.path.join(temp_dir, 'Data_Entry_2017.csv')
        csv_data.to_csv(csv_path, index=False)
        
        # Create split file
        train_split = os.path.join(temp_dir, 'train_val_list.txt')
        with open(train_split, 'w') as f:
            f.write('img_001.png\nimg_002.png\n')
        
        # Create image structure
        img_dir = os.path.join(temp_dir, 'images_001', 'images')
        os.makedirs(img_dir)
        
        # Create dummy images
        for img_name in ['img_001.png', 'img_002.png']:
            img_path = os.path.join(img_dir, img_name)
            # Create grayscale image
            img_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            img.save(img_path)
        
        yield temp_dir


@patch('src.models.classifier.xrv.models.DenseNet')
def test_end_to_end_training_step(mock_densenet, mini_dataset):
    """Test a complete training step with real data flow."""
    # Mock the model to avoid loading actual weights
    mock_model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, 7, padding=3),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 14)
    )
    mock_densenet.return_value = mock_model
    
    # Create dataset
    dataset = ChestXRay14Dataset(
        data_dir=mini_dataset,
        split='train',
        image_size=224,
        data_augmentation=False
    )
    
    # Create constraint projection
    constraint_layer = ConstraintProjection(
        implications=[(8, 2, 0.0)],  # Pneumonia -> Infiltration
        exclusions=[(7, 3, 1.0)],    # Effusion vs Pneumothorax
        max_iter=10
    )
    
    # Create model
    model = ConstrainedChestXRayClassifier(
        backbone='densenet121',
        pretrained=False,  # Use mock instead
        num_classes=14,
        constraint_projection=constraint_layer,
        projection_mode='integrated'
    )
    
    # Create loss function
    loss_fn = ConstrainedLoss(
        constraint_weight=1.0,
        consistency_weight=0.5
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Single training step
    model.train()
    optimizer.zero_grad()
    
    # Get batch
    batch = [dataset[i] for i in range(len(dataset))]
    imgs = torch.stack([item['img'] for item in batch])
    labs = torch.stack([item['lab'] for item in batch])
    
    # Forward pass
    raw_logits = model(imgs, apply_constraints=False)
    
    # Get constraint stats
    with torch.no_grad():
        raw_probs = torch.sigmoid(raw_logits)
        constraint_stats = constraint_layer.constraint_violations(raw_probs)
    
    # Get constrained output
    constrained_probs = model.project(raw_logits)
    
    # Compute loss
    losses = loss_fn(constrained_probs, raw_logits, labs, constraint_stats)
    
    # Backward pass
    losses['loss'].backward()
    optimizer.step()
    
    # Assertions
    assert imgs.shape == (2, 1, 224, 224)
    assert labs.shape == (2, 14)
    assert raw_logits.shape == (2, 14)
    assert constrained_probs.shape == (2, 14)
    assert torch.all(constrained_probs >= 0.0)
    assert torch.all(constrained_probs <= 1.0)
    assert losses['loss'].item() > 0


def test_constraint_enforcement_integration():
    """Test that constraints are properly enforced in integration."""
    # Create constraint layer with strict implication
    constraint_layer = ConstraintProjection(
        implications=[(0, 1, 0.0)],  # Class 0 -> Class 1 with tau=0
        max_iter=50
    )
    
    # Create logits that violate constraint
    # Class 0 high, Class 1 low (violation)
    logits = torch.tensor([[2.0, -2.0], [1.0, -1.0]])  # [0.88, 0.12], [0.73, 0.27]
    
    projected_probs = constraint_layer(logits)
    
    # Check constraint satisfaction: P(1) >= P(0) + tau
    for i in range(projected_probs.shape[0]):
        assert projected_probs[i, 1] >= projected_probs[i, 0] - 1e-5


def test_medical_constraints_realistic():
    """Test with realistic medical constraints."""
    from src.training.train import MEDICAL_CONSTRAINTS
    
    # Simulate ChestX-ray14 pathology indices
    pathologies = [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
        'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
        'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
    ]
    label_to_idx = {p: i for i, p in enumerate(pathologies)}
    
    # Build constraints
    implications = []
    for a, b, tau in MEDICAL_CONSTRAINTS['implications']:
        if a in label_to_idx and b in label_to_idx:
            implications.append((label_to_idx[a], label_to_idx[b], tau))
    
    exclusions = []
    for a, b, kappa in MEDICAL_CONSTRAINTS['exclusions']:
        if a in label_to_idx and b in label_to_idx:
            exclusions.append((label_to_idx[a], label_to_idx[b], kappa))
    
    constraint_layer = ConstraintProjection(
        implications=implications,
        exclusions=exclusions,
        max_iter=20
    )
    
    # Test with random logits
    batch_size = 4
    logits = torch.randn(batch_size, 14)
    projected_probs = constraint_layer(logits)
    
    assert projected_probs.shape == (batch_size, 14)
    assert torch.all(projected_probs >= 0.0)
    assert torch.all(projected_probs <= 1.0)
    
    # Check some constraints are enforced
    constraint_violations = constraint_layer.constraint_violations(projected_probs)
    
    # Violations should be minimal after projection
    if 'implication_total' in constraint_violations:
        assert constraint_violations['implication_total'] < 0.1
    if 'exclusion_total' in constraint_violations:
        assert constraint_violations['exclusion_total'] < 0.1