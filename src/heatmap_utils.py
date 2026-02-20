"""
Heatmap generation for anomaly detection.

Creates per-pixel reconstruction error heatmaps from autoencoder input vs reconstruction.
Used for visualization and analysis.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, Tuple

from src.utils import tensor_to_numpy


def reconstruction_error_map(input_tensor: torch.Tensor, recon_tensor: torch.Tensor) -> np.ndarray:
    """
    Compute per-pixel reconstruction error (MSE over channels).

    Args:
        input_tensor: [1, 3, H, W] normalized input
        recon_tensor: [1, 3, H, W] reconstruction

    Returns:
        error_map: [H, W] numpy array, higher = more error (more anomalous)
    """
    with torch.no_grad():
        diff = (input_tensor - recon_tensor) ** 2
        # Mean over channels: [1, 3, H, W] -> [1, H, W]
        error_map = diff.mean(dim=1).squeeze(0).cpu().numpy()
    return error_map


def heatmap_filename_from_path(image_path: str, category: str) -> str:
    """
    Build a descriptive filename for the heatmap from the image path.

    Example: data/leather/test/cut/002.png -> heatmap_leather_test_cut_002
    """
    path = os.path.normpath(image_path).replace("\\", "/")
    parts = [p for p in path.split("/") if p and p != "data"]
    if not parts:
        parts = ["image"]
    # Remove extension from last part
    parts[-1] = os.path.splitext(parts[-1])[0]
    return "heatmap_" + "_".join(parts)


def create_and_save_heatmap(
    input_tensor: torch.Tensor,
    recon_tensor: torch.Tensor,
    error_map: np.ndarray,
    image_path: str,
    category: str,
    anomaly_score: float,
    result_label: str,
    threshold: float,
    save_dir: str = "results/heatmaps",
    show: bool = True,
) -> Tuple[str, str]:
    """
    Create heatmap figure, save as PNG and PDF, optionally display.

    Returns:
        (path_to_png, path_to_pdf)
    """
    os.makedirs(save_dir, exist_ok=True)
    if "upload" in image_path.lower() or os.path.basename(image_path).startswith("upload_"):
        base_name = "heatmap_" + os.path.splitext(os.path.basename(image_path))[0]
    else:
        base_name = heatmap_filename_from_path(image_path, category)

    # Original image for display (denormalized)
    img_np = tensor_to_numpy(input_tensor)

    # Normalize error map for colormap (0-1) so structure is visible
    em = error_map
    if em.max() > em.min():
        em_norm = (em - em.min()) / (em.max() - em.min())
    else:
        em_norm = np.zeros_like(em)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 1) Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis("off")

    # 2) Heatmap only (reconstruction error)
    im = axes[1].imshow(em_norm, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Reconstruction Error (Anomaly Heatmap)", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Normalized error")

    # 3) Overlay: image with heatmap on top
    axes[2].imshow(img_np)
    axes[2].imshow(em_norm, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("Overlay (Image + Heatmap)", fontsize=12)
    axes[2].axis("off")

    # Global title with result
    fig.suptitle(
        f"Category: {category}  |  Score: {anomaly_score:.4f}  |  Threshold: {threshold:.4f}  |  Result: {result_label}",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()

    png_path = os.path.join(save_dir, base_name + ".png")
    pdf_path = os.path.join(save_dir, base_name + ".pdf")

    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return png_path, pdf_path
