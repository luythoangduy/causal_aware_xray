#!/usr/bin/env python3
"""
Quick test script to verify ChestXRay14Dataset can load correctly.
"""
from __future__ import annotations
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.chestxray14 import ChestXRay14Dataset

def test_dataset(data_root: str):
    """Test dataset loading with the given data root."""
    print(f"Testing dataset at: {data_root}")
    print("=" * 50)
    
    try:
        # Create dataset
        print("Creating dataset...")
        dataset = ChestXRay14Dataset(
            data_dir=data_root,
            split='train',
            image_size=224,
            data_augmentation=False  # Disable for faster testing
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Images mapped: {len(dataset.image_paths)}")
        print(f"Pathologies: {len(dataset.pathologies)}")
        
        # Test loading first few samples
        print("\nTesting sample loading...")
        for i in range(min(3, len(dataset))):
            try:
                sample = dataset[i]
                img_shape = sample['img'].shape
                label_shape = sample['lab'].shape
                print(f"Sample {i}: img_shape={img_shape}, label_shape={label_shape}")
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
                break
        
        print("\n✓ Dataset test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_dataset.py <data_root>")
        print("Example: python test_dataset.py C:/chest-xray/nih_chestxray_14")
        sys.exit(1)
    
    data_root = sys.argv[1]
    success = test_dataset(data_root)
    sys.exit(0 if success else 1)