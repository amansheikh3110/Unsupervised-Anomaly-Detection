"""
Backend logic for web UI: run anomaly check and return result + heatmap.
Uses the same model and heatmap code as check_image.py.
"""

import os
import json
import torch
from PIL import Image

# Project root
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets import eval_transform
from src.model_autoencoder import get_autoencoder
from src.utils import get_device
from src.heatmap_utils import (
    reconstruction_error_map,
    heatmap_filename_from_path,
    create_and_save_heatmap,
)

DATA_ROOT = "data"
CHECKPOINT_DIR = "checkpoints"
HEATMAP_DIR = "results/heatmaps"
METRICS_DIR = "results/baseline"


def _load_image_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return eval_transform(img).unsqueeze(0)


def run_anomaly_check(
    image_path: str,
    category: str,
    heatmap_dir: str = None,
) -> dict:
    """
    Run anomaly detection on one image. Returns dict with score, result, threshold, heatmap paths, etc.
    """
    heatmap_dir = heatmap_dir or HEATMAP_DIR
    os.makedirs(heatmap_dir, exist_ok=True)

    device = get_device()
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"autoencoder_{category}.pth")

    if not os.path.isfile(checkpoint_path):
        return {
            "error": f"No checkpoint for category '{category}'. Train with: python run_baseline.py --category {category} --epochs 50",
        }

    # Load image and model
    img_tensor = _load_image_tensor(image_path).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    latent_dim = ckpt.get("latent_dim", 256)
    model = get_autoencoder(in_channels=3, latent_dim=latent_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        recon, _ = model(img_tensor)
        score = model.reconstruction_loss(img_tensor, recon, reduction="mean").item()

    metrics_path = os.path.join(METRICS_DIR, category, "metrics_baseline.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        threshold = metrics.get("best_threshold", 0.03)
        roc_auc = metrics.get("roc_auc")
        avg_precision = metrics.get("average_precision")
        f1 = metrics.get("f1_at_best_threshold")
    else:
        threshold = 0.03
        roc_auc = avg_precision = f1 = None

    is_anomaly = score >= threshold
    result_label = "Anomaly (Defect)" if is_anomaly else "Normal"

    # Generate heatmap (no display)
    error_map = reconstruction_error_map(img_tensor, recon)
    png_path, pdf_path = create_and_save_heatmap(
        img_tensor.cpu(),
        recon.cpu(),
        error_map,
        image_path,
        category,
        anomaly_score=score,
        result_label=result_label,
        threshold=threshold,
        save_dir=heatmap_dir,
        show=False,
    )
    heatmap_basename = os.path.basename(png_path)

    return {
        "score": round(score, 6),
        "threshold": round(threshold, 6),
        "result": result_label,
        "is_anomaly": is_anomaly,
        "heatmap_png": heatmap_basename,
        "heatmap_pdf": os.path.basename(pdf_path),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "average_precision": round(avg_precision, 4) if avg_precision is not None else None,
        "f1_score": round(f1, 4) if f1 is not None else None,
    }


def get_category_metrics(category: str) -> dict:
    """Load saved metrics for a category (ROC-AUC, etc.) for display in UI."""
    path = os.path.join(METRICS_DIR, category, "metrics_baseline.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def list_categories():
    """List MVTec categories that have data and optionally checkpoint."""
    from src.datasets import MVTecDataset
    out = []
    for cat in MVTecDataset.CATEGORIES:
        data_path = os.path.join(DATA_ROOT, cat)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"autoencoder_{cat}.pth")
        out.append({
            "id": cat,
            "name": cat.replace("_", " ").title(),
            "has_data": os.path.isdir(data_path),
            "has_model": os.path.isfile(ckpt_path),
        })
    return out


def list_dataset_images(category: str, split: str = None):
    """
    List image paths for a category. split: 'train' | 'test' | None (both).
    Returns list of { path, label, defect_type } for frontend to build gallery.
    """
    base = os.path.join(DATA_ROOT, category)
    items = []
    if split in (None, "train"):
        good_path = os.path.join(base, "train", "good")
        if os.path.isdir(good_path):
            for f in sorted(os.listdir(good_path)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    items.append({
                        "path": f"data/{category}/train/good/{f}",
                        "label": "good",
                        "defect_type": "good",
                        "split": "train",
                    })
    if split in (None, "test"):
        test_base = os.path.join(base, "test")
        if os.path.isdir(test_base):
            for defect_type in sorted(os.listdir(test_base)):
                sub = os.path.join(test_base, defect_type)
                if os.path.isdir(sub):
                    for f in sorted(os.listdir(sub)):
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
                            items.append({
                                "path": f"data/{category}/test/{defect_type}/{f}",
                                "label": "good" if defect_type == "good" else "defect",
                                "defect_type": defect_type,
                                "split": "test",
                            })
    return items
