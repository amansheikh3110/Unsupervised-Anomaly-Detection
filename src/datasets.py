"""
MVTec Anomaly Detection Dataset Loader

This module provides data loading utilities for the MVTec AD dataset
used in unsupervised anomaly detection with I-JEPA.

Key Rule: Training uses ONLY normal (good) images.
          Test set contains both normal and anomalous images.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional, Dict


# =============================================================================
# Image Transforms (ImageNet normalization for ViT compatibility)
# =============================================================================

IMG_SIZE = 224
PATCH_SIZE = 16

# ImageNet normalization (standard for pre-trained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Evaluation transforms (no augmentation)
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Transform to convert back to displayable image
inverse_transform = transforms.Compose([
    transforms.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD]
    )
])


# =============================================================================
# MVTec Dataset Class
# =============================================================================

class MVTecDataset(Dataset):
    """
    MVTec Anomaly Detection Dataset
    
    Args:
        root_dir: Path to MVTec data folder (e.g., 'data')
        category: Product category (e.g., 'bottle', 'cable', 'leather')
        split: 'train' for training (good only) or 'test' for evaluation
        transform: Image transformations to apply
        return_label: If True, returns (image, label) where label=0 for good, 1 for defect
    
    Note: Training split ONLY contains normal (good) samples.
          Test split contains both normal and defective samples.
    """
    
    # All available categories in MVTec AD
    CATEGORIES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    
    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        return_label: bool = False
    ):
        assert category in self.CATEGORIES, f"Category must be one of {self.CATEGORIES}"
        assert split in ['train', 'test'], "Split must be 'train' or 'test'"
        
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.return_label = return_label
        
        # Set default transforms
        if transform is None:
            self.transform = train_transform if split == 'train' else eval_transform
        else:
            self.transform = transform
        
        # Collect image paths and labels
        self.samples = []  # List of (image_path, label, defect_type)
        self._load_samples()
    
    def _load_samples(self):
        """Load all sample paths and their labels."""
        category_path = os.path.join(self.root_dir, self.category)
        
        if self.split == 'train':
            # Training: only good/normal images
            good_path = os.path.join(category_path, 'train', 'good')
            if os.path.exists(good_path):
                for img_name in sorted(os.listdir(good_path)):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(good_path, img_name)
                        self.samples.append((img_path, 0, 'good'))
        else:
            # Test: both good and defective images
            test_path = os.path.join(category_path, 'test')
            if os.path.exists(test_path):
                for defect_type in sorted(os.listdir(test_path)):
                    defect_path = os.path.join(test_path, defect_type)
                    if os.path.isdir(defect_path):
                        label = 0 if defect_type == 'good' else 1
                        for img_name in sorted(os.listdir(defect_path)):
                            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                                img_path = os.path.join(defect_path, img_name)
                                self.samples.append((img_path, label, defect_type))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        img_path, label, defect_type = self.samples[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        if self.return_label:
            return image, label
        return image
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get detailed information about a sample."""
        img_path, label, defect_type = self.samples[idx]
        return {
            'path': img_path,
            'label': label,
            'defect_type': defect_type,
            'is_anomaly': label == 1
        }


# =============================================================================
# Patchify Utilities (for ViT / I-JEPA)
# =============================================================================

def patchify(images: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """
    Convert images to non-overlapping patches.
    
    Args:
        images: Tensor of shape [B, C, H, W]
        patch_size: Size of each square patch (default: 16)
    
    Returns:
        patches: Tensor of shape [B, num_patches, patch_dim]
                 where num_patches = (H/patch_size) * (W/patch_size)
                 and patch_dim = C * patch_size * patch_size
    """
    B, C, H, W = images.shape
    ph = pw = patch_size
    
    assert H % ph == 0 and W % pw == 0, \
        f"Image dimensions ({H}x{W}) must be divisible by patch size ({ph})"
    
    # Reshape to patches
    # [B, C, H, W] -> [B, C, H//ph, ph, W//pw, pw]
    images = images.view(B, C, H // ph, ph, W // pw, pw)
    
    # Permute to [B, H//ph, W//pw, C, ph, pw]
    images = images.permute(0, 2, 4, 1, 3, 5).contiguous()
    
    # Flatten to [B, num_patches, patch_dim]
    num_patches = (H // ph) * (W // pw)
    patch_dim = C * ph * pw
    patches = images.view(B, num_patches, patch_dim)
    
    return patches


def unpatchify(patches: torch.Tensor, patch_size: int = PATCH_SIZE, 
               channels: int = 3) -> torch.Tensor:
    """
    Convert patches back to images.
    
    Args:
        patches: Tensor of shape [B, num_patches, patch_dim]
        patch_size: Size of each square patch
        channels: Number of image channels
    
    Returns:
        images: Tensor of shape [B, C, H, W]
    """
    B, num_patches, patch_dim = patches.shape
    ph = pw = patch_size
    
    # Calculate grid size
    grid_size = int(num_patches ** 0.5)
    H = W = grid_size * patch_size
    
    # Reshape patches
    patches = patches.view(B, grid_size, grid_size, channels, ph, pw)
    
    # Permute back to image format
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    
    # Reshape to [B, C, H, W]
    images = patches.view(B, channels, H, W)
    
    return images


def get_patch_positions(img_size: int = IMG_SIZE, 
                        patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """
    Get 2D position indices for each patch.
    
    Returns:
        positions: Tensor of shape [num_patches, 2] with (row, col) for each patch
    """
    grid_size = img_size // patch_size
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            positions.append([i, j])
    return torch.tensor(positions)


# =============================================================================
# Data Loader Utilities
# =============================================================================

def get_dataloader(
    root_dir: str,
    category: str,
    split: str = 'train',
    batch_size: int = 32,
    shuffle: bool = None,
    num_workers: int = 0,
    return_label: bool = False,
    transform: Optional[transforms.Compose] = None
) -> DataLoader:
    """
    Create a DataLoader for MVTec dataset.
    
    Args:
        root_dir: Path to MVTec data folder
        category: Product category
        split: 'train' or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle (default: True for train, False for test)
        num_workers: Number of data loading workers
        return_label: Whether to return labels
        transform: Custom transforms (uses defaults if None)
    
    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = MVTecDataset(
        root_dir=root_dir,
        category=category,
        split=split,
        transform=transform,
        return_label=return_label
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def get_all_categories_dataloader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 32,
    shuffle: bool = None,
    num_workers: int = 0,
    return_label: bool = False
) -> DataLoader:
    """
    Create a DataLoader combining all MVTec categories.
    Useful for training a single model on all categories.
    """
    from torch.utils.data import ConcatDataset
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    datasets = []
    for category in MVTecDataset.CATEGORIES:
        ds = MVTecDataset(
            root_dir=root_dir,
            category=category,
            split=split,
            return_label=return_label
        )
        datasets.append(ds)
    
    combined = ConcatDataset(datasets)
    
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


# =============================================================================
# Dataset Statistics
# =============================================================================

def get_dataset_stats(root_dir: str) -> Dict:
    """Get statistics about the MVTec dataset."""
    stats = {
        'categories': {},
        'total_train': 0,
        'total_test': 0,
        'total_test_good': 0,
        'total_test_defect': 0
    }
    
    for category in MVTecDataset.CATEGORIES:
        train_ds = MVTecDataset(root_dir, category, 'train', return_label=True)
        test_ds = MVTecDataset(root_dir, category, 'test', return_label=True)
        
        test_good = sum(1 for _, label, _ in test_ds.samples if label == 0)
        test_defect = sum(1 for _, label, _ in test_ds.samples if label == 1)
        
        stats['categories'][category] = {
            'train': len(train_ds),
            'test': len(test_ds),
            'test_good': test_good,
            'test_defect': test_defect
        }
        
        stats['total_train'] += len(train_ds)
        stats['total_test'] += len(test_ds)
        stats['total_test_good'] += test_good
        stats['total_test_defect'] += test_defect
    
    return stats


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == '__main__':
    # Quick test of the dataset
    print("Testing MVTec Dataset Loader...")
    print("=" * 60)
    
    # Test with leather category
    dataset = MVTecDataset(
        root_dir='data',
        category='leather',
        split='train',
        return_label=True
    )
    
    print(f"Category: leather")
    print(f"Train samples: {len(dataset)}")
    
    # Test loading an image
    img, label = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Label: {label} (0=good, 1=defect)")
    
    # Test patchify
    img_batch = img.unsqueeze(0)  # Add batch dimension
    patches = patchify(img_batch)
    print(f"Patches shape: {patches.shape}")
    
    # Test unpatchify
    reconstructed = unpatchify(patches)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Verify reconstruction
    diff = (img_batch - reconstructed).abs().max()
    print(f"Reconstruction error: {diff:.6f}")
    
    print("=" * 60)
    print("Dataset loader test passed!")
