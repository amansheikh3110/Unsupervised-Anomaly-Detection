"""
rebuild_detectors.py — Rebuild k-NN detectors using a pretrained ViT-B/16 backbone.

Run this ONCE to replace the 50-epoch I-JEPA detectors with much more accurate
pretrained-feature detectors. No training required — takes ~2-5 minutes total.

Usage:
    python rebuild_detectors.py

    # Only specific categories:
    python rebuild_detectors.py --categories hazelnut cable

    # Keep all patches (no subsampling) for best accuracy:
    python rebuild_detectors.py --coreset 1.0

After this, run python app_ijepa.py — the web app automatically picks up the
new detectors. Accuracy should jump to 85-99% ROC-AUC.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets import eval_transform, get_dataloader
from src.ijepa_anomaly_detector import IJEPAAnomalyDetector
from src.pretrained_vit import PretrainedViTWrapper
from src.utils import get_device

CHECKPOINT_DIR = "checkpoints"
DATA_ROOT      = "data"
RESULTS_DIR    = "results/ijepa"
MARKER_FILE    = os.path.join(CHECKPOINT_DIR, "pretrained_vit_used.json")

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def has_training_data(category: str) -> bool:
    train_dir = os.path.join(DATA_ROOT, category, "train", "good")
    return os.path.isdir(train_dir) and len(os.listdir(train_dir)) > 0


def evaluate_detector(detector, model, category, device):
    """Quick evaluation: ROC-AUC on the test set."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.metrics import f1_score

    test_loader = get_dataloader(
        root_dir=DATA_ROOT,
        category=category,
        split="test",
        batch_size=16,
        shuffle=False,
        num_workers=0,
        return_label=True,
        transform=eval_transform,
    )
    scores, labels = detector.evaluate(model, test_loader, device)

    if labels.sum() == 0 or labels.sum() == len(labels):
        print(f"  [{category}] WARNING: degenerate label set, skipping metrics.")
        return {}

    roc_auc  = roc_auc_score(labels, scores)
    avg_prec = average_precision_score(labels, scores)

    # Find best F1 threshold
    thresholds = np.percentile(scores, np.arange(1, 100, 1))
    best_f1, best_thr = 0.0, thresholds[0]
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return {
        "roc_auc": round(float(roc_auc), 4),
        "average_precision": round(float(avg_prec), 4),
        "f1_at_best_threshold": round(float(best_f1), 4),
        "best_threshold": round(float(best_thr), 6),
        "backbone": "pretrained_vit_b16",
    }


def rebuild(categories, coreset_ratio, k, batch_size, device):
    print("\n" + "=" * 60)
    print("  Rebuilding detectors with pretrained ViT-B/16")
    print(f"  coreset={coreset_ratio*100:.0f}%  k={k}  device={device}")
    print("=" * 60 + "\n")

    print("Loading pretrained ViT-B/16 (ImageNet)...")
    model = PretrainedViTWrapper(device=device)
    print(f"  embed_dim={model.encoder_embed_dim}, patches={model.num_patches}\n")

    results = {}
    for cat in categories:
        print(f"[{cat}]")

        if not has_training_data(cat):
            print(f"  No training data at data/{cat}/train/good — skipping.\n")
            continue

        # Load normal training images
        loader = get_dataloader(
            root_dir=DATA_ROOT,
            category=cat,
            split="train",
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            return_label=False,
            transform=eval_transform,
        )

        # Build detector
        detector = IJEPAAnomalyDetector(
            k=k,
            coreset_ratio=coreset_ratio,
            normalize_feats=True,
            batch_size=batch_size,
        )
        detector.fit(model, loader, device, category=cat)

        # Save detector (overwrites old one)
        save_path = os.path.join(CHECKPOINT_DIR, f"ijepa_detector_{cat}.pkl")
        detector.save(save_path)

        # Evaluate if test data exists
        test_dir = os.path.join(DATA_ROOT, cat, "test")
        if os.path.isdir(test_dir):
            print(f"  Evaluating on test set...")
            metrics = evaluate_detector(detector, model, cat, device)
            if metrics:
                print(f"  ROC-AUC: {metrics['roc_auc']:.4f}  "
                      f"F1: {metrics['f1_at_best_threshold']:.4f}  "
                      f"AP: {metrics['average_precision']:.4f}")
                # Save metrics
                out_dir = os.path.join(RESULTS_DIR, cat)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "metrics_ijepa.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
                results[cat] = metrics
        print()

    # Write marker so backend knows to use pretrained ViT
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(MARKER_FILE, "w") as f:
        json.dump({
            "backbone": "vit_b_16",
            "embed_dim": 768,
            "coreset_ratio": coreset_ratio,
            "k": k,
            "categories": list(results.keys()),
        }, f, indent=2)
    print(f"Marker saved: {MARKER_FILE}")

    # Summary
    if results:
        print("\n" + "=" * 60)
        print(f"  {'Category':<14} {'ROC-AUC':>8} {'F1':>7} {'AP':>8}")
        print("  " + "-" * 40)
        aucs = []
        for cat, m in results.items():
            auc = m.get('roc_auc', 0)
            f1  = m.get('f1_at_best_threshold', 0)
            ap  = m.get('average_precision', 0)
            aucs.append(auc)
            print(f"  {cat:<14} {auc:>8.4f} {f1:>7.4f} {ap:>8.4f}")
        print("  " + "-" * 40)
        print(f"  {'MEAN':<14} {np.mean(aucs):>8.4f}")
        print("=" * 60)

    print("\nDone. Restart app_ijepa.py to use the improved detectors.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="+", default=None,
                   help="Which categories to rebuild (default: all with data)")
    p.add_argument("--coreset", type=float, default=1.0,
                   help="Coreset ratio: 1.0 = keep all patches (best accuracy)")
    p.add_argument("--k", type=int, default=5,
                   help="k for k-NN anomaly scoring (default: 5)")
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_device()

    if args.categories:
        cats = args.categories
    else:
        # Auto-detect: rebuild all categories that have a saved detector
        cats = []
        for cat in ALL_CATEGORIES:
            det = os.path.join(CHECKPOINT_DIR, f"ijepa_detector_{cat}.pkl")
            if os.path.isfile(det):
                cats.append(cat)
        if not cats:
            print("No existing detectors found. Run run_ijepa.py first, or specify --categories.")
            sys.exit(1)
        print(f"Auto-detected categories with detectors: {cats}")

    rebuild(cats, coreset_ratio=args.coreset, k=args.k,
            batch_size=args.batch_size, device=device)
