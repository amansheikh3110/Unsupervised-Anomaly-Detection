"""
Data Exploration Script for MVTec AD Dataset

This script helps you understand the dataset structure, visualize samples,
and observe the nature of anomalies (texture, structural, object defects).

Run this script to:
1. See dataset statistics
2. Visualize normal training samples
3. Visualize test samples (normal vs defective)
4. Understand different defect types per category
"""

import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import (
    MVTecDataset, 
    get_dataset_stats,
    eval_transform,
    IMG_SIZE
)
from src.utils import (
    set_seed,
    tensor_to_numpy,
    show_image_grid,
    ensure_dir
)


def print_dataset_statistics(data_root: str = 'data'):
    """Print comprehensive statistics about the MVTec dataset."""
    
    print("\n" + "=" * 70)
    print("MVTec Anomaly Detection Dataset Statistics")
    print("=" * 70)
    
    stats = get_dataset_stats(data_root)
    
    print(f"\n{'Category':<15} {'Train':<10} {'Test':<10} {'Good':<10} {'Defect':<10}")
    print("-" * 55)
    
    for cat, cat_stats in stats['categories'].items():
        print(f"{cat:<15} {cat_stats['train']:<10} {cat_stats['test']:<10} "
              f"{cat_stats['test_good']:<10} {cat_stats['test_defect']:<10}")
    
    print("-" * 55)
    print(f"{'TOTAL':<15} {stats['total_train']:<10} {stats['total_test']:<10} "
          f"{stats['total_test_good']:<10} {stats['total_test_defect']:<10}")
    
    print("\n" + "=" * 70)
    print(f"Total Categories: {len(stats['categories'])}")
    print(f"Total Training Images (Normal): {stats['total_train']}")
    print(f"Total Test Images: {stats['total_test']}")
    print(f"  - Normal (Good): {stats['total_test_good']}")
    print(f"  - Anomalous (Defect): {stats['total_test_defect']}")
    print("=" * 70)


def get_defect_types(data_root: str, category: str) -> list:
    """Get all defect types for a category."""
    test_path = os.path.join(data_root, category, 'test')
    if os.path.exists(test_path):
        return [d for d in sorted(os.listdir(test_path)) 
                if os.path.isdir(os.path.join(test_path, d))]
    return []


def visualize_category_samples(data_root: str, category: str, 
                                num_samples: int = 4,
                                save_dir: str = None):
    """
    Visualize samples from a specific category.
    Shows both normal training samples and various defect types.
    """
    
    print(f"\n{'='*60}")
    print(f"Category: {category.upper()}")
    print(f"{'='*60}")
    
    # Get defect types
    defect_types = get_defect_types(data_root, category)
    print(f"Defect types: {defect_types}")
    
    # Load training samples
    train_ds = MVTecDataset(data_root, category, 'train', 
                            transform=eval_transform, return_label=True)
    
    # Load test samples
    test_ds = MVTecDataset(data_root, category, 'test',
                           transform=eval_transform, return_label=True)
    
    # Collect images by type
    images_by_type = {'good (train)': []}
    
    # Get random training samples
    train_indices = random.sample(range(len(train_ds)), min(num_samples, len(train_ds)))
    for idx in train_indices:
        img, _ = train_ds[idx]
        images_by_type['good (train)'].append(img)
    
    # Get test samples by defect type
    for defect in defect_types:
        images_by_type[defect] = []
    
    for idx in range(len(test_ds)):
        info = test_ds.get_sample_info(idx)
        defect = info['defect_type']
        if len(images_by_type[defect]) < num_samples:
            img, _ = test_ds[idx]
            images_by_type[defect].append(img)
    
    # Create visualization
    num_types = len(images_by_type)
    fig, axes = plt.subplots(num_types, num_samples, 
                              figsize=(num_samples * 3, num_types * 3))
    
    if num_types == 1:
        axes = axes.reshape(1, -1)
    
    for row, (defect_type, images) in enumerate(images_by_type.items()):
        for col in range(num_samples):
            ax = axes[row, col]
            if col < len(images):
                img = tensor_to_numpy(images[col])
                ax.imshow(img)
                if col == 0:
                    label = "NORMAL" if defect_type in ['good', 'good (train)'] else "DEFECT"
                    ax.set_ylabel(f"{defect_type}\n({label})", fontsize=10)
            ax.axis('off')
    
    plt.suptitle(f"MVTec AD - {category.upper()}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        ensure_dir(save_dir)
        save_path = os.path.join(save_dir, f"{category}_samples.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def visualize_all_categories_overview(data_root: str, save_dir: str = None):
    """
    Create an overview showing one normal and one defect sample from each category.
    """
    
    categories = MVTecDataset.CATEGORIES
    
    fig, axes = plt.subplots(len(categories), 2, figsize=(6, len(categories) * 2.5))
    
    for row, category in enumerate(categories):
        # Get one normal training sample
        train_ds = MVTecDataset(data_root, category, 'train',
                                transform=eval_transform, return_label=True)
        
        # Get one defect test sample
        test_ds = MVTecDataset(data_root, category, 'test',
                               transform=eval_transform, return_label=True)
        
        # Find a defect sample
        defect_idx = None
        for idx in range(len(test_ds)):
            info = test_ds.get_sample_info(idx)
            if info['label'] == 1:
                defect_idx = idx
                break
        
        # Normal sample
        normal_img, _ = train_ds[0]
        axes[row, 0].imshow(tensor_to_numpy(normal_img))
        axes[row, 0].set_ylabel(category, fontsize=10, fontweight='bold')
        if row == 0:
            axes[row, 0].set_title("Normal", fontsize=12)
        axes[row, 0].axis('off')
        
        # Defect sample
        if defect_idx is not None:
            defect_img, _ = test_ds[defect_idx]
            defect_type = test_ds.get_sample_info(defect_idx)['defect_type']
            axes[row, 1].imshow(tensor_to_numpy(defect_img))
            if row == 0:
                axes[row, 1].set_title("Defect", fontsize=12)
            axes[row, 1].set_xlabel(defect_type, fontsize=8)
        axes[row, 1].axis('off')
    
    plt.suptitle("MVTec AD Dataset Overview\nNormal vs Defect Samples", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        ensure_dir(save_dir)
        save_path = os.path.join(save_dir, "dataset_overview.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved overview to {save_path}")
    
    plt.show()


def visualize_patchification(data_root: str, category: str = 'leather'):
    """
    Visualize how images are divided into patches for I-JEPA.
    """
    from src.datasets import patchify, unpatchify, PATCH_SIZE
    
    # Load a sample image
    train_ds = MVTecDataset(data_root, category, 'train',
                            transform=eval_transform, return_label=False)
    
    img = train_ds[0].unsqueeze(0)  # Add batch dimension
    
    # Patchify
    patches = patchify(img)
    print(f"Original image shape: {img.shape}")
    print(f"Patches shape: {patches.shape}")
    print(f"Number of patches: {patches.shape[1]}")
    print(f"Patch dimension: {patches.shape[2]}")
    
    # Visualize patch grid
    grid_size = int(patches.shape[1] ** 0.5)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(tensor_to_numpy(img[0]))
    axes[0].set_title(f"Original Image ({IMG_SIZE}x{IMG_SIZE})")
    axes[0].axis('off')
    
    # Image with patch grid overlay
    img_np = tensor_to_numpy(img[0])
    axes[1].imshow(img_np)
    for i in range(0, IMG_SIZE + 1, PATCH_SIZE):
        axes[1].axhline(y=i, color='red', linewidth=0.5)
        axes[1].axvline(x=i, color='red', linewidth=0.5)
    axes[1].set_title(f"Patch Grid ({grid_size}x{grid_size} = {grid_size**2} patches)")
    axes[1].axis('off')
    
    # Reconstructed from patches
    reconstructed = unpatchify(patches)
    axes[2].imshow(tensor_to_numpy(reconstructed[0]))
    axes[2].set_title("Reconstructed from Patches")
    axes[2].axis('off')
    
    plt.suptitle(f"Image Patchification (patch_size={PATCH_SIZE})", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    """Main exploration function."""
    
    set_seed(42)
    data_root = 'data'
    results_dir = 'results/exploration'
    
    print("\n" + "=" * 70)
    print("MVTec AD Dataset Exploration")
    print("=" * 70)
    
    # 1. Print statistics
    print("\n[1/4] Dataset Statistics")
    print_dataset_statistics(data_root)
    
    # 2. Overview of all categories
    print("\n[2/4] Creating dataset overview...")
    visualize_all_categories_overview(data_root, save_dir=results_dir)
    
    # 3. Detailed view of select categories
    print("\n[3/4] Detailed category visualization...")
    sample_categories = ['bottle', 'leather', 'tile']  # Different types: object, texture
    for cat in sample_categories:
        visualize_category_samples(data_root, cat, num_samples=4, save_dir=results_dir)
    
    # 4. Patchification visualization
    print("\n[4/4] Patchification visualization...")
    visualize_patchification(data_root, category='leather')
    
    print("\n" + "=" * 70)
    print("Exploration complete!")
    print(f"Visualizations saved to: {results_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
