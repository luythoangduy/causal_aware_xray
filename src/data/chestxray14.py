from __future__ import annotations
import os
from typing import Dict
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

PATHOLOGIES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

class ChestXRay14Dataset(Dataset):
    """ChestX-ray14 dataset loader with multi-label encoding.

    Expected directory content (data_dir):
      - images/ (all image files)
      - Data_Entry_2017_v2020.csv
      - train_val_list.txt
      - test_list.txt
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: int = 224,
        normalization: str = 'torchxrayvision',
        data_augmentation: bool = True,
        include_no_finding: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.include_no_finding = include_no_finding

        meta_path = os.path.join(data_dir, 'Data_Entry_2017.csv')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")
        self.meta = pd.read_csv(meta_path)
        
        # Build file list from split lists
        list_file = 'test_list.txt' if split == 'test' else 'train_val_list.txt'
        list_path = os.path.join(data_dir, list_file)
        names = set([l.strip() for l in open(list_path)])
        
        self.meta = self.meta[self.meta['Image Index'].isin(names)].reset_index(drop=True)

        # Precompute label matrix
        self.pathologies = PATHOLOGIES.copy()
        if include_no_finding and 'No Finding' not in self.pathologies:
            self.pathologies.append('No Finding')

        self.label_map: Dict[str, int] = {p: i for i, p in enumerate(self.pathologies)}
        self.labels = torch.zeros((len(self.meta), len(self.pathologies)), dtype=torch.float32)
        
        # Build labels
        for idx, row in self.meta.iterrows():
            findings = [f.strip() for f in row['Finding Labels'].split('|')]
            if 'No Finding' in findings and include_no_finding:
                self.labels[idx, self.label_map['No Finding']] = 1.0
            for f in findings:
                if f != 'No Finding' and f in self.label_map:
                    self.labels[idx, self.label_map[f]] = 1.0

        # Build image directory mapping for efficient lookup
        self._build_image_mapping()

        # Use ImageNet normalization for grayscale (convert to 3-channel)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
        
        # Simple transforms: resize to grayscale, convert to RGB, CenterCrop
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        
        # Always use RGB mode for ImageNet normalization
        self.rgb_mode = True

    def _build_image_mapping(self):
        """Build mapping from image names to their full paths for efficient lookup."""
        self.image_paths = {}
        
        # Find all images_* directories
        image_dirs = []
        if os.path.exists(self.data_dir):
            for subdir in os.listdir(self.data_dir):
                if subdir.startswith('images_') and os.path.isdir(os.path.join(self.data_dir, subdir)):
                    image_dirs.append(subdir)
        
        if not image_dirs:
            raise FileNotFoundError(f"No image directories found in {self.data_dir}. Expected directories like 'images_001', 'images_002', etc.")
        
        # Build mapping from each images_*/images/ subdirectory
        for img_dir in image_dirs:
            inner_images_path = os.path.join(self.data_dir, img_dir, 'images')
            if os.path.exists(inner_images_path):
                for img_file in os.listdir(inner_images_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths[img_file] = os.path.join(inner_images_path, img_file)
        
        # Simple validation
        missing_count = sum(1 for _, row in self.meta.iterrows() if row['Image Index'] not in self.image_paths)
        if missing_count > 0:
            print(f"WARNING: {missing_count} images from metadata not found")
        else:
            print(f"✓ Mapped {len(self.image_paths)} images successfully")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_name = row['Image Index']
        
        if img_name not in self.image_paths:
            raise FileNotFoundError(f"Image '{img_name}' not found")
        
        img_path = self.image_paths[img_name]
        img = Image.open(img_path)
        # Convert grayscale to RGB for ImageNet normalization
        img = img.convert('RGB')
        
        # Apply transforms (resize + CenterCrop + ToTensor)
        img = self.transform(img)  # Returns tensor: (3, 224, 224)
        
        # Apply normalization
        img = self.normalize(img)
        label = self.labels[idx]
        return {"img": img, "lab": label, "idx": idx}
