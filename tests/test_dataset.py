from __future__ import annotations
import pytest
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock
import pandas as pd
from PIL import Image
import numpy as np

from src.data.chestxray14 import ChestXRay14Dataset, PATHOLOGIES


@pytest.fixture
def mock_dataset_structure():
    """Create a temporary dataset structure for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSV file
        csv_data = pd.DataFrame({
            'Image Index': ['test_001.png', 'test_002.png', 'test_003.png'],
            'Finding Labels': ['Pneumonia', 'No Finding', 'Pneumonia|Effusion'],
            'Patient ID': [1, 2, 3],
            'Patient Age': [50, 60, 70],
            'Patient Gender': ['M', 'F', 'M']
        })
        csv_path = os.path.join(temp_dir, 'Data_Entry_2017.csv')
        csv_data.to_csv(csv_path, index=False)
        
        # Create split files
        train_split = os.path.join(temp_dir, 'train_val_list.txt')
        with open(train_split, 'w') as f:
            f.write('test_001.png\ntest_002.png\n')
        
        test_split = os.path.join(temp_dir, 'test_list.txt')
        with open(test_split, 'w') as f:
            f.write('test_003.png\n')
        
        # Create image directories
        img_dir = os.path.join(temp_dir, 'images_001', 'images')
        os.makedirs(img_dir)
        
        # Create dummy images
        for img_name in ['test_001.png', 'test_002.png', 'test_003.png']:
            img_path = os.path.join(img_dir, img_name)
            img = Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8), mode='L')
            img.save(img_path)
        
        yield temp_dir


def test_dataset_initialization(mock_dataset_structure):
    """Test dataset can be initialized correctly."""
    dataset = ChestXRay14Dataset(
        data_dir=mock_dataset_structure,
        split='train',
        image_size=224
    )
    
    assert len(dataset) == 2  # train split has 2 images
    assert len(dataset.pathologies) == 14
    assert dataset.pathologies == PATHOLOGIES


def test_dataset_image_loading(mock_dataset_structure):
    """Test dataset can load images correctly."""
    dataset = ChestXRay14Dataset(
        data_dir=mock_dataset_structure,
        split='train',
        image_size=224,
        data_augmentation=False
    )
    
    sample = dataset[0]
    assert 'img' in sample
    assert 'lab' in sample
    assert 'idx' in sample
    
    # Check tensor shapes
    assert sample['img'].shape == (1, 224, 224)  # Grayscale
    assert sample['lab'].shape == (14,)  # 14 pathologies
    assert sample['idx'] == 0


def test_dataset_label_encoding(mock_dataset_structure):
    """Test label encoding is correct."""
    dataset = ChestXRay14Dataset(
        data_dir=mock_dataset_structure,
        split='train',
        image_size=224
    )
    
    # First sample: 'Pneumonia'
    sample_0 = dataset[0]
    pneumonia_idx = PATHOLOGIES.index('Pneumonia')
    assert sample_0['lab'][pneumonia_idx] == 1.0
    
    # Second sample: 'No Finding' (should be all zeros since include_no_finding=False)
    sample_1 = dataset[1]
    assert torch.sum(sample_1['lab']) == 0.0


def test_dataset_missing_images():
    """Test handling of missing images."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSV with non-existent image
        csv_data = pd.DataFrame({
            'Image Index': ['missing.png'],
            'Finding Labels': ['Pneumonia'],
            'Patient ID': [1],
            'Patient Age': [50],
            'Patient Gender': ['M']
        })
        csv_path = os.path.join(temp_dir, 'Data_Entry_2017.csv')
        csv_data.to_csv(csv_path, index=False)
        
        # Create split file
        train_split = os.path.join(temp_dir, 'train_val_list.txt')
        with open(train_split, 'w') as f:
            f.write('missing.png\n')
        
        # Don't create images directory
        
        with pytest.raises(FileNotFoundError):
            ChestXRay14Dataset(data_dir=temp_dir, split='train')


def test_dataset_rgb_mode(mock_dataset_structure):
    """Test RGB mode works correctly."""
    dataset = ChestXRay14Dataset(
        data_dir=mock_dataset_structure,
        split='train',
        normalization='imagenet',  # Should use RGB
        image_size=224
    )
    
    sample = dataset[0]
    assert sample['img'].shape == (3, 224, 224)  # RGB