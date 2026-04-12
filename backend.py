"""
Backend logic for web UI: run anomaly check and return result + heatmap.

Supports two models:
  1. Autoencoder baseline (Phase 2): checkpoint = checkpoints/autoencoder_<cat>.pth
  2. I-JEPA (Phase 3+):    checkpoint = checkpoints/ijepa_*_<cat>.pth
                           detector   = checkpoints/ijepa_detector_<cat>.pkl

Priority: I-JEPA detector is used if available; autoencoder is the fallback.
"""

import os
import json
import pickle
import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets import eval_transform, MVTecDataset
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
METRICS_BASELINE_DIR = "results/baseline"
METRICS_IJEPA_DIR = "results/ijepa"


# =============================================================================
# Image loading
# =============================================================================

def _load_image_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return eval_transform(img).unsqueeze(0)


# =============================================================================
# I-JEPA model + detector cache (singleton per category)
# =============================================================================

_ijepa_model_cache: dict = {}   # checkpoint_path → IJEPA model
_detector_cache: dict = {}      # category → IJEPAAnomalyDetector

PRETRAINED_MARKER = os.path.join(CHECKPOINT_DIR, "pretrained_vit_used.json")
_PRETRAINED_SENTINEL = "__pretrained_vit__"


def _using_pretrained_vit() -> bool:
    return os.path.isfile(PRETRAINED_MARKER)


def _get_ijepa_model(checkpoint_path: str, device: torch.device):
    """Load (and cache) an I-JEPA model or the pretrained ViT wrapper."""
    # Pretrained ViT path — single shared instance
    if checkpoint_path == _PRETRAINED_SENTINEL or _using_pretrained_vit():
        if _PRETRAINED_SENTINEL not in _ijepa_model_cache:
            from src.pretrained_vit import PretrainedViTWrapper
            _ijepa_model_cache[_PRETRAINED_SENTINEL] = PretrainedViTWrapper(device=device)
        return _ijepa_model_cache[_PRETRAINED_SENTINEL]

    if checkpoint_path not in _ijepa_model_cache:
        from src.model_ijepa import load_ijepa_from_checkpoint
        model = load_ijepa_from_checkpoint(checkpoint_path, device=device)
        model.eval()
        _ijepa_model_cache[checkpoint_path] = model
    return _ijepa_model_cache[checkpoint_path]


def _get_detector(category: str):
    """Load (and cache) the k-NN detector for a category."""
    if category not in _detector_cache:
        detector_path = os.path.join(CHECKPOINT_DIR, f"ijepa_detector_{category}.pkl")
        if not os.path.isfile(detector_path):
            return None
        from src.ijepa_anomaly_detector import IJEPAAnomalyDetector
        _detector_cache[category] = IJEPAAnomalyDetector.load(detector_path)
    return _detector_cache[category]


def _find_ijepa_checkpoint(category: str) -> str | None:
    """
    Find an I-JEPA checkpoint (or pretrained ViT sentinel) for this category.
    If rebuild_detectors.py has been run, returns the pretrained sentinel so
    _get_ijepa_model() loads the pretrained ViT-B/16 instead.
    """
    # If detector was rebuilt with pretrained features, use that instead
    if _using_pretrained_vit():
        det = os.path.join(CHECKPOINT_DIR, f"ijepa_detector_{category}.pkl")
        if os.path.isfile(det):
            return _PRETRAINED_SENTINEL

    if not os.path.isdir(CHECKPOINT_DIR):
        return None

    candidates = []
    for fname in os.listdir(CHECKPOINT_DIR):
        if not (fname.startswith("ijepa_") and fname.endswith(".pth")
                and "detector" not in fname and "final" not in fname):
            continue
        full = os.path.join(CHECKPOINT_DIR, fname)

        # Fast check by filename
        if f"_{category}" in fname:
            candidates.insert(0, full)   # highest priority
            continue

        # Slower check: load metadata from checkpoint
        try:
            ckpt = torch.load(full, map_location="cpu")
            cats = ckpt.get("categories", [])
            if category in cats:
                candidates.append(full)
        except Exception:
            pass

    return candidates[0] if candidates else None


# =============================================================================
# I-JEPA heatmap (patch score → image heatmap)
# =============================================================================

def _ijepa_heatmap_and_save(
    patch_scores: np.ndarray,
    img_tensor: torch.Tensor,
    image_path: str,
    category: str,
    anomaly_score: float,
    result_label: str,
    threshold: float,
    heatmap_dir: str,
) -> tuple:
    """Generate heatmap from I-JEPA patch scores and save."""
    import matplotlib.pyplot as plt
    from src.utils import tensor_to_numpy
    import torch.nn.functional as F

    os.makedirs(heatmap_dir, exist_ok=True)

    # Build spatial heatmap [14,14] → upsample to [224,224]
    grid_size = 14
    hmap_small = patch_scores.reshape(grid_size, grid_size).astype(np.float32)
    hmap_t = torch.tensor(hmap_small).unsqueeze(0).unsqueeze(0)  # [1,1,14,14]
    hmap_up = F.interpolate(hmap_t, size=(224, 224), mode="bilinear", align_corners=False)
    error_map = hmap_up.squeeze().numpy()  # [224, 224]

    if "upload" in image_path.lower() or os.path.basename(image_path).startswith("upload_"):
        base_name = "ijepa_heatmap_" + os.path.splitext(os.path.basename(image_path))[0]
    else:
        from src.heatmap_utils import heatmap_filename_from_path
        base_name = "ijepa_" + heatmap_filename_from_path(image_path, category)

    img_np = tensor_to_numpy(img_tensor)
    em_norm = error_map
    if em_norm.max() > em_norm.min():
        em_norm = (em_norm - em_norm.min()) / (em_norm.max() - em_norm.min())

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis("off")

    im = axes[1].imshow(em_norm, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("I-JEPA Patch Anomaly Score", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Patch score")

    axes[2].imshow(img_np)
    axes[2].imshow(em_norm, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("Overlay (Image + Heatmap)", fontsize=12)
    axes[2].axis("off")

    fig.suptitle(
        f"I-JEPA | Category: {category}  Score: {anomaly_score:.4f}  "
        f"Threshold: {threshold:.4f}  Result: {result_label}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    png_path = os.path.join(heatmap_dir, base_name + ".png")
    pdf_path = os.path.join(heatmap_dir, base_name + ".pdf")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


# =============================================================================
# Public: run_anomaly_check
# =============================================================================

def run_anomaly_check(
    image_path: str,
    category: str,
    heatmap_dir: str = None,
) -> dict:
    """
    Run anomaly detection on one image.

    Tries I-JEPA first; falls back to autoencoder if not available.
    Returns dict with score, result, threshold, heatmap paths, metrics.
    """
    heatmap_dir = heatmap_dir or HEATMAP_DIR
    os.makedirs(heatmap_dir, exist_ok=True)
    device = get_device()

    # -------------------------------------------------------------------------
    # Try I-JEPA path first
    # -------------------------------------------------------------------------
    ijepa_ckpt = _find_ijepa_checkpoint(category)
    detector = _get_detector(category)

    if ijepa_ckpt and detector is not None:
        return _run_ijepa_check(
            image_path, category, ijepa_ckpt, detector, device, heatmap_dir
        )

    # -------------------------------------------------------------------------
    # Fallback: autoencoder baseline
    # -------------------------------------------------------------------------
    return _run_autoencoder_check(image_path, category, device, heatmap_dir)


def _run_ijepa_check(
    image_path, category, ckpt_path, detector, device, heatmap_dir
) -> dict:
    """Anomaly check using I-JEPA + k-NN detector."""
    img_tensor = _load_image_tensor(image_path).to(device)
    model = _get_ijepa_model(ckpt_path, device)

    score, patch_scores = detector.score(model, img_tensor, device)

    # Load I-JEPA metrics for threshold
    metrics_path = os.path.join(METRICS_IJEPA_DIR, category, "metrics_ijepa.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            saved_metrics = json.load(f)
        threshold = saved_metrics.get("best_threshold", _default_ijepa_threshold(detector))
        roc_auc = saved_metrics.get("roc_auc")
        avg_precision = saved_metrics.get("average_precision")
        f1 = saved_metrics.get("f1_at_best_threshold")
    else:
        threshold = _default_ijepa_threshold(detector)
        roc_auc = avg_precision = f1 = None

    is_anomaly = score >= threshold
    result_label = "Anomaly (Defect)" if is_anomaly else "Normal"

    png_path, pdf_path = _ijepa_heatmap_and_save(
        patch_scores=patch_scores,
        img_tensor=img_tensor.cpu(),
        image_path=image_path,
        category=category,
        anomaly_score=score,
        result_label=result_label,
        threshold=threshold,
        heatmap_dir=heatmap_dir,
    )

    return {
        "score": round(float(score), 6),
        "threshold": round(float(threshold), 6),
        "result": result_label,
        "is_anomaly": bool(is_anomaly),
        "model": "ijepa",
        "heatmap_png": os.path.basename(png_path),
        "heatmap_pdf": os.path.basename(pdf_path),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "average_precision": round(avg_precision, 4) if avg_precision is not None else None,
        "f1_score": round(f1, 4) if f1 is not None else None,
    }


def _default_ijepa_threshold(detector) -> float:
    """Estimate a threshold from the memory bank statistics (fallback)."""
    if detector.memory_bank is None:
        return 0.5
    # 95th percentile self-distance as a rough threshold estimate
    try:
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree", metric="euclidean")
        knn.fit(detector.memory_bank)
        dists, _ = knn.kneighbors(detector.memory_bank[:500])
        self_dists = dists[:, 1]  # skip self (dist=0)
        return float(np.percentile(self_dists, 95))
    except Exception:
        return 0.5


def _run_autoencoder_check(image_path, category, device, heatmap_dir) -> dict:
    """Anomaly check using autoencoder baseline (Phase 2)."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"autoencoder_{category}.pth")
    if not os.path.isfile(checkpoint_path):
        return {
            "error": (
                f"No model for '{category}'. "
                f"Train autoencoder: python run_baseline.py --category {category} --epochs 50  "
                f"OR train I-JEPA: python run_ijepa.py --categories {category} --epochs 100"
            )
        }

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

    metrics_path = os.path.join(METRICS_BASELINE_DIR, category, "metrics_baseline.json")
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

    error_map = reconstruction_error_map(img_tensor, recon)
    png_path, pdf_path = create_and_save_heatmap(
        img_tensor.cpu(), recon.cpu(), error_map,
        image_path, category,
        anomaly_score=score,
        result_label=result_label,
        threshold=threshold,
        save_dir=heatmap_dir,
        show=False,
    )

    return {
        "score": round(score, 6),
        "threshold": round(threshold, 6),
        "result": result_label,
        "is_anomaly": bool(is_anomaly),
        "model": "autoencoder",
        "heatmap_png": os.path.basename(png_path),
        "heatmap_pdf": os.path.basename(pdf_path),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "average_precision": round(avg_precision, 4) if avg_precision is not None else None,
        "f1_score": round(f1, 4) if f1 is not None else None,
    }


# =============================================================================
# Metrics
# =============================================================================

def get_category_metrics(category: str) -> dict:
    """
    Load saved evaluation metrics for a category.
    Prefers I-JEPA metrics; falls back to baseline.
    """
    ijepa_path = os.path.join(METRICS_IJEPA_DIR, category, "metrics_ijepa.json")
    if os.path.isfile(ijepa_path):
        with open(ijepa_path) as f:
            m = json.load(f)
        m["model"] = "ijepa"
        return m

    baseline_path = os.path.join(METRICS_BASELINE_DIR, category, "metrics_baseline.json")
    if os.path.isfile(baseline_path):
        with open(baseline_path) as f:
            m = json.load(f)
        m["model"] = "autoencoder"
        return m

    return {}


# =============================================================================
# Category listing
# =============================================================================

def list_categories():
    """List MVTec categories with data / model availability status."""
    out = []
    for cat in MVTecDataset.CATEGORIES:
        data_path = os.path.join(DATA_ROOT, cat)
        ae_ckpt = os.path.join(CHECKPOINT_DIR, f"autoencoder_{cat}.pth")
        det_path = os.path.join(CHECKPOINT_DIR, f"ijepa_detector_{cat}.pkl")
        ijepa_ckpt = _find_ijepa_checkpoint(cat)

        out.append({
            "id": cat,
            "name": cat.replace("_", " ").title(),
            "has_data": os.path.isdir(data_path),
            "has_model": os.path.isfile(ae_ckpt) or (ijepa_ckpt is not None and os.path.isfile(det_path)),
            "has_autoencoder": os.path.isfile(ae_ckpt),
            "has_ijepa": ijepa_ckpt is not None and os.path.isfile(det_path),
        })
    return out


# =============================================================================
# Dataset image listing
# =============================================================================

def list_dataset_images(category: str, split: str = None):
    """List image paths for a category for frontend gallery."""
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
