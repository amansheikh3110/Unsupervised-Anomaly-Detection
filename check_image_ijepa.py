"""
check_image_ijepa.py — Single-image I-JEPA anomaly check

Usage:
    python check_image_ijepa.py data/leather/test/cut/002.png --category leather

Requires:
    - Trained I-JEPA checkpoint in checkpoints/
    - Fitted k-NN detector in checkpoints/ijepa_detector_<category>.pkl
    Both are created by: python run_ijepa.py --categories <category> --epochs 100
"""

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets import eval_transform
from src.ijepa_anomaly_detector import IJEPAAnomalyDetector
from src.model_ijepa import load_ijepa_from_checkpoint
from src.utils import get_device


def parse_args():
    p = argparse.ArgumentParser(description="I-JEPA single-image anomaly check")
    p.add_argument("image", type=str, help="Path to image file")
    p.add_argument("--category", type=str, default="leather",
                   help="MVTec category (must match trained detector)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="I-JEPA checkpoint path (auto-detect if not given)")
    p.add_argument("--detector", type=str, default=None,
                   help="k-NN detector path (auto-detect if not given)")
    p.add_argument("--heatmap_dir", type=str, default="results/heatmaps",
                   help="Where to save heatmap images")
    return p.parse_args()


def find_checkpoint(category: str) -> str | None:
    """Auto-find the best I-JEPA checkpoint for this category."""
    ckpt_dir = "checkpoints"
    if not os.path.isdir(ckpt_dir):
        return None
    for fname in os.listdir(ckpt_dir):
        if (fname.startswith("ijepa_") and fname.endswith(".pth")
                and "detector" not in fname and "final" not in fname
                and f"_{category}" in fname):
            return os.path.join(ckpt_dir, fname)
    # Try loading any ijepa checkpoint and checking 'categories' field
    for fname in os.listdir(ckpt_dir):
        if fname.startswith("ijepa_") and fname.endswith(".pth") and "detector" not in fname:
            full = os.path.join(ckpt_dir, fname)
            try:
                ckpt = torch.load(full, map_location="cpu")
                if category in ckpt.get("categories", []):
                    return full
            except Exception:
                pass
    return None


def main():
    args = parse_args()
    device = get_device()

    # Resolve checkpoint
    ckpt_path = args.checkpoint or find_checkpoint(args.category)
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        print(f"ERROR: No I-JEPA checkpoint found for category '{args.category}'.")
        print(f"Train first: python run_ijepa.py --categories {args.category} --epochs 100")
        sys.exit(1)

    # Resolve detector
    det_path = args.detector or os.path.join(
        "checkpoints", f"ijepa_detector_{args.category}.pkl"
    )
    if not os.path.isfile(det_path):
        print(f"ERROR: No k-NN detector found at {det_path}.")
        print(f"Build it: python run_ijepa.py --categories {args.category} --no_train")
        sys.exit(1)

    print(f"Image:      {args.image}")
    print(f"Category:   {args.category}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Detector:   {det_path}")
    print(f"Device:     {device}")
    print("-" * 50)

    # Load model and detector
    model = load_ijepa_from_checkpoint(ckpt_path, device=device)
    model.eval()
    detector = IJEPAAnomalyDetector.load(det_path)

    # Load image
    if not os.path.isfile(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)
    img = Image.open(args.image).convert("RGB")
    img_tensor = eval_transform(img).unsqueeze(0).to(device)

    # Score
    score, patch_scores = detector.score(model, img_tensor, device)

    # Try to load saved threshold
    import json
    metrics_path = os.path.join("results", "ijepa", args.category, "metrics_ijepa.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        threshold = m.get("best_threshold", score * 0.8)
        roc_auc = m.get("roc_auc", None)
    else:
        threshold = None
        roc_auc = None

    is_anomaly = (score >= threshold) if threshold is not None else None

    print(f"Anomaly Score: {score:.6f}")
    if threshold is not None:
        print(f"Threshold:     {threshold:.6f}")
        print(f"Result:        {'*** ANOMALY (Defect) ***' if is_anomaly else 'Normal'}")
    else:
        print("Threshold:     not available (run evaluate first)")
    if roc_auc:
        print(f"Category ROC-AUC: {roc_auc:.4f}")

    # Generate and save heatmap
    heatmap = detector.patch_scores_to_heatmap(patch_scores)

    import matplotlib.pyplot as plt
    import numpy as np
    from src.utils import tensor_to_numpy

    os.makedirs(args.heatmap_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    hmap_path = os.path.join(args.heatmap_dir, f"ijepa_{args.category}_{base}.png")

    img_np = tensor_to_numpy(img_tensor.cpu())
    hmap_norm = heatmap.copy()
    if hmap_norm.max() > hmap_norm.min():
        hmap_norm = (hmap_norm - hmap_norm.min()) / (hmap_norm.max() - hmap_norm.min())

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis("off")

    im = axes[1].imshow(hmap_norm, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("I-JEPA Patch Anomaly Scores", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_np)
    axes[2].imshow(hmap_norm, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis("off")

    result_str = "ANOMALY" if is_anomaly else ("NORMAL" if is_anomaly is not None else "?")
    fig.suptitle(
        f"I-JEPA | {args.category} | Score: {score:.4f} | {result_str}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(hmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved: {hmap_path}")


if __name__ == "__main__":
    main()
