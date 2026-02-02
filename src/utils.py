"""
Utility functions for I-JEPA Anomaly Detection Project

Contains common helper functions used across the project.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from datetime import datetime


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Device Management
# =============================================================================

def get_device() -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def print_device_info():
    """Print information about available computing devices."""
    print("=" * 50)
    print("Device Information")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA Available: No (using CPU)")
    
    print("=" * 50)


# =============================================================================
# Path Management
# =============================================================================

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def get_timestamp() -> str:
    """Get current timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# Visualization Helpers
# =============================================================================

def denormalize_image(tensor: torch.Tensor, 
                      mean: List[float] = [0.485, 0.456, 0.406],
                      std: List[float] = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """
    Denormalize a tensor image with mean and standard deviation.
    
    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W]
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return tensor * std + mean


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for visualization."""
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image if batch
    
    # Denormalize
    tensor = denormalize_image(tensor)
    
    # Clamp to valid range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose to [H, W, C]
    return tensor.permute(1, 2, 0).cpu().numpy()


def show_image(tensor: torch.Tensor, title: str = "", ax=None):
    """Display a tensor image."""
    img = tensor_to_numpy(tensor)
    
    if ax is None:
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')


def show_image_grid(tensors: List[torch.Tensor], 
                    titles: List[str] = None,
                    ncols: int = 4,
                    figsize: Tuple[int, int] = None,
                    save_path: str = None):
    """
    Display a grid of tensor images.
    
    Args:
        tensors: List of image tensors
        titles: List of titles for each image
        ncols: Number of columns in grid
        figsize: Figure size (auto-calculated if None)
        save_path: Path to save the figure (optional)
    """
    n = len(tensors)
    nrows = (n + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (ncols * 3, nrows * 3)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()
    
    for i, (ax, tensor) in enumerate(zip(axes, tensors)):
        img = tensor_to_numpy(tensor)
        ax.imshow(img)
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(model, optimizer, epoch: int, loss: float, 
                    path: str, **kwargs):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, path: str, device: torch.device = None):
    """Load model checkpoint."""
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('loss', None)


# =============================================================================
# Metrics Helpers
# =============================================================================

class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print_device_info()
    
    # Test seed setting
    set_seed(42)
    print("\nRandom seed set to 42")
    print(f"Random int: {random.randint(0, 100)}")
    print(f"Numpy random: {np.random.rand():.4f}")
    print(f"Torch random: {torch.rand(1).item():.4f}")
