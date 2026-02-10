"""
Check a single image for anomaly using the trained baseline autoencoder.

INPUT:  Path to one image file (e.g. a leather image).
OUTPUT: Anomaly score, answer (Normal / Anomaly), and optional heatmap (PNG + PDF).

Usage:
  python check_image.py path/to/image.png
  python check_image.py data/leather/test/color/000.png --category leather
  python check_image.py data/leather/test/cut/002.png --category leather
  python check_image.py data/leather/test/cut/002.png --category leather --no-heatmap
"""

import os
import sys
import argparse
import json
import torch
from PIL import Image

# Project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets import eval_transform
from src.model_autoencoder import get_autoencoder
from src.utils import get_device
from src.heatmap_utils import (
    reconstruction_error_map,
    create_and_save_heatmap,
)


def load_image(path: str):
    """Load and preprocess one image (same as dataset)."""
    img = Image.open(path).convert("RGB")
    return eval_transform(img).unsqueeze(0)  # [1, 3, 224, 224]


def main():
    parser = argparse.ArgumentParser(
        description="Check one image: is it Normal or Anomaly? (baseline autoencoder)"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to check (e.g. data/leather/test/color/000.png)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="leather",
        help="Category the model was trained on (default: leather)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: checkpoints/autoencoder_<category>.pth)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Anomaly threshold (default: load from results/baseline/<category>/metrics_baseline.json if exists, else 0.03)",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip generating and saving heatmap",
    )
    parser.add_argument(
        "--heatmap-dir",
        type=str,
        default="results/heatmaps",
        help="Directory to save heatmap PNG and PDF (default: results/heatmaps)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display heatmap window (only save PNG and PDF)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)

    checkpoint_path = args.checkpoint or os.path.join(
        "checkpoints", f"autoencoder_{args.category}.pth"
    )
    if not os.path.isfile(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Train first with: python run_baseline.py --category", args.category, "--epochs 50")
        sys.exit(1)

    device = get_device()
    print("=" * 60)
    print("Single-image anomaly check (baseline autoencoder)")
    print("=" * 60)
    print("Input image:", args.image_path)
    print("Category:  ", args.category)
    print("Checkpoint:", checkpoint_path)
    print("Device:   ", device)
    print("=" * 60)

    # Load image
    img_tensor = load_image(args.image_path).to(device)

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device)
    latent_dim = ckpt.get("latent_dim", 256)
    model = get_autoencoder(in_channels=3, latent_dim=latent_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Anomaly score = reconstruction error (MSE)
    with torch.no_grad():
        recon, _ = model(img_tensor)
        score = model.reconstruction_loss(img_tensor, recon, reduction="mean").item()

    # Threshold: from saved metrics, or argument, or default
    threshold = args.threshold
    if threshold is None:
        metrics_path = os.path.join(
            "results", "baseline", args.category, "metrics_baseline.json"
        )
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            threshold = metrics.get("best_threshold", 0.03)
            print("(Using threshold from", metrics_path + ")")
        else:
            threshold = 0.03
            print("(Using default threshold 0.03; run evaluation first for a better threshold)")

    is_anomaly = score >= threshold
    answer = "Anomaly (Defect)" if is_anomaly else "Normal"

    print()
    print("Anomaly score:", round(score, 6))
    print("Threshold:    ", round(threshold, 6))
    print("Result:      ", answer)
    print("=" * 60)

    # Heatmap: per-pixel reconstruction error, save PNG + PDF, display
    if not args.no_heatmap:
        error_map = reconstruction_error_map(img_tensor, recon)
        png_path, pdf_path = create_and_save_heatmap(
            img_tensor.cpu(),
            recon.cpu(),
            error_map,
            args.image_path,
            args.category,
            anomaly_score=score,
            result_label=answer,
            threshold=threshold,
            save_dir=args.heatmap_dir,
            show=not args.no_show,
        )
        print("Heatmap saved:")
        print("  PNG:", png_path)
        print("  PDF:", pdf_path)
        print("=" * 60)


if __name__ == "__main__":
    main()
