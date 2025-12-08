"""
Data preprocessing and augmentation module for malaria detection.
Handles data loading, preprocessing, and augmentation transformations.
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np


def get_transforms(split='train', image_size=224):
    """
    Get data augmentation transforms for training, validation, or test.
    
    Args:
        split: 'train', 'val', or 'test'
        image_size: Target image size (default 224)
    
    Returns:
        torchvision.transforms.Compose object
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


class TransformedSubset(torch.utils.data.Dataset):
    """Wrapper to apply different transforms to a subset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)


def get_data_loaders(data_dir, batch_size=32, image_size=224, 
                     train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                     num_workers=4, random_seed=42, auto_download=True):
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Path to the cell_images directory
        batch_size: Batch size for data loaders
        image_size: Target image size
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducibility
        auto_download: If True, automatically download dataset if missing (default: True)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Check if dataset exists, download if missing
    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        if auto_download:
            print(f"Dataset not found at {data_dir}. Attempting to download...")
            from utils import download_dataset
            downloaded_path = download_dataset(data_dir)
            if downloaded_path is None:
                raise FileNotFoundError(
                    f"Dataset not found at {data_dir} and automatic download failed.\n"
                    "Please download the dataset manually from:\n"
                    "https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria\n"
                    "Or set up Kaggle API authentication for automatic download."
                )
            data_dir = downloaded_path
        else:
            raise FileNotFoundError(
                f"Dataset not found at {data_dir}.\n"
                "Please download the dataset from:\n"
                "https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria\n"
                "Extract it and place the 'cell_images' folder in the 'data' directory."
            )
    
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Use ImageFolder which automatically handles the directory structure
    # It expects: data_dir/class1/, data_dir/class2/, etc.
    full_dataset = ImageFolder(data_dir, transform=None)
    
    # Print dataset info
    print(f"Loaded {len(full_dataset)} images")
    print(f"  Classes: {full_dataset.classes}")
    print(f"  Class to index: {full_dataset.class_to_idx}")
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Apply appropriate transforms to each subset
    train_dataset = TransformedSubset(train_subset, transform=get_transforms('train', image_size))
    val_dataset = TransformedSubset(val_subset, transform=get_transforms('val', image_size))
    test_dataset = TransformedSubset(test_subset, transform=get_transforms('test', image_size))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

