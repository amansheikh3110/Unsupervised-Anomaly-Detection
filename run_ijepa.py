"""
run_ijepa.py — I-JEPA Full Pipeline (Phase 3 + 4)

One-command script that:
  1. Trains I-JEPA on normal images (or loads existing checkpoint)
  2. For each target category:
       a. Builds a k-NN anomaly detector from normal training features
       b. Evaluates on the test set (ROC-AUC, AP, F1, confusion matrix)
       c. Saves results to results/ijepa/<category>/metrics_ijepa.json
  3. Prints a comparison table: I-JEPA vs Autoencoder baseline

Usage:

  # Quick test on leather (10 epochs — not accurate, just to verify pipeline)
  python run_ijepa.py --categories leather --epochs 10

  # Full single-category run (recommended: 100 epochs)
  python run_ijepa.py --categories leather --epochs 100

  # Train on 10 categories simultaneously + evaluate each one
  python run_ijepa.py \\
      --categories bottle cable capsule carpet grid hazelnut leather metal_nut pill screw \\
      --epochs 100 --batch_size 16 --model_size small

  # Evaluate only (skip training, use existing checkpoint)
  python run_ijepa.py --categories leather --no_train --checkpoint checkpoints/ijepa_small_leather.pth

  # Use tiny model if GPU memory is tight
  python run_ijepa.py --categories leather --model_size tiny --epochs 100
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets import MVTecDataset, eval_transform, get_dataloader
from src.ijepa_anomaly_detector import IJEPAAnomalyDetector, build_detector_for_category
from src.model_ijepa import load_ijepa_from_checkpoint
from src.train_ijepa import train as train_ijepa
from src.utils import ensure_dir, get_device, set_seed

# Default 10 categories to run
DEFAULT_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
]


# =============================================================================
# Argument parser
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="I-JEPA full pipeline: train + build detectors + evaluate",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Categories
    p.add_argument(
        "--categories", type=str, nargs="+", default=["leather"],
        help="MVTec categories to train on and evaluate.\n"
             "E.g.: --categories leather bottle cable",
    )

    # Training
    p.add_argument("--no_train", action="store_true",
                   help="Skip training; load existing checkpoint instead.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to existing I-JEPA checkpoint (used when --no_train).")
    p.add_argument("--model_size", type=str, default="small",
                   choices=["tiny", "small", "base"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--amp", type=lambda x: x.lower() != "false", default=True)

    # Anomaly detection
    p.add_argument("--k", type=int, default=9,
                   help="k for k-NN anomaly scoring.")
    p.add_argument("--coreset_ratio", type=float, default=0.25,
                   help="Fraction of normal patches to keep in memory bank.\n"
                        "Lower = faster but less accurate. 1.0 = keep all.")

    # Paths
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--results_dir", type=str, default="results")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--eval_batch_size", type=int, default=32)

    return p.parse_args()


# =============================================================================
# Evaluation helpers
# =============================================================================

def evaluate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    results_dir: str,
    category: str,
    prefix: str = "",
) -> dict:
    """
    Compute ROC-AUC, AP, F1 from anomaly scores and ground-truth labels.
    Saves metrics JSON and score/label numpy arrays.
    """
    y_true = labels.astype(int)
    y_score = scores.astype(np.float64)

    roc_auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # Best F1 threshold from PR curve
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    f1_vals = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = int(np.argmax(f1_vals[:-1]))
    best_thr = float(thresholds[best_idx]) if len(thresholds) > 0 else float(np.median(y_score))
    y_pred = (y_score >= best_thr).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "f1_at_best_threshold": float(f1),
        "best_threshold": float(best_thr),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "n_normal": int((y_true == 0).sum()),
        "n_anomaly": int((y_true == 1).sum()),
    }

    ensure_dir(results_dir)
    json_path = os.path.join(results_dir, f"{prefix}metrics_ijepa.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(os.path.join(results_dir, f"{prefix}test_scores_ijepa.npy"), scores)
    np.save(os.path.join(results_dir, f"{prefix}test_labels_ijepa.npy"), labels)

    return metrics


def load_baseline_metrics(results_dir: str, category: str) -> dict:
    """Load autoencoder baseline metrics for comparison (may not exist)."""
    path = os.path.join(results_dir, "baseline", category, "metrics_baseline.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print("=" * 75)
    print("I-JEPA Anomaly Detection Pipeline — Phase 3 + 4")
    print("=" * 75)
    print(f"Categories:     {args.categories}")
    print(f"Model size:     {args.model_size}")
    print(f"Device:         {device}")
    print("=" * 75)

    # =========================================================================
    # STEP 1: Train I-JEPA (or load existing checkpoint)
    # =========================================================================

    cats_tag = (
        "_".join(args.categories)
        if len(args.categories) <= 3
        else f"{len(args.categories)}cats"
    )
    auto_ckpt = os.path.join(
        args.checkpoint_dir, f"ijepa_{args.model_size}_{cats_tag}.pth"
    )

    if args.no_train and args.checkpoint:
        ckpt_path = args.checkpoint
        print(f"[SKIP TRAINING] Loading checkpoint: {ckpt_path}")
    elif args.no_train and os.path.isfile(auto_ckpt):
        ckpt_path = auto_ckpt
        print(f"[SKIP TRAINING] Using existing checkpoint: {ckpt_path}")
    else:
        print("\n[STEP 1] Training I-JEPA self-supervised model...")

        # Build argparse namespace for train function
        train_args = argparse.Namespace(
            data_root=args.data_root,
            categories=args.categories,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=0.05,
            warmup_epochs=args.warmup_epochs,
            min_lr=1e-6,
            grad_clip=1.0,
            amp=args.amp,
            ema_start=0.996,
            ema_end=1.0,
            num_targets=4,
            target_scale_min=0.15,
            target_scale_max=0.20,
            checkpoint_dir=args.checkpoint_dir,
            save_name=None,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        ckpt_path = train_ijepa(train_args)

    # =========================================================================
    # STEP 2: Load trained model
    # =========================================================================

    print(f"\n[STEP 2] Loading I-JEPA model from {ckpt_path}")
    model = load_ijepa_from_checkpoint(ckpt_path, device=device)
    model.eval()
    print("Model loaded.")

    # =========================================================================
    # STEP 3: Per-category: build detector + evaluate
    # =========================================================================

    all_results = {}

    for category in args.categories:
        print(f"\n{'=' * 60}")
        print(f"Category: {category}")
        print(f"{'=' * 60}")

        cat_results_dir = os.path.join(args.results_dir, "ijepa", category)
        ensure_dir(cat_results_dir)

        # Check if category data exists
        cat_data_path = os.path.join(args.data_root, category)
        if not os.path.isdir(cat_data_path):
            print(f"  WARNING: data not found at {cat_data_path} — skipping.")
            continue

        detector_path = os.path.join(
            args.checkpoint_dir, f"ijepa_detector_{category}.pkl"
        )

        # --- Build memory bank ---
        print(f"\n[3a] Building k-NN detector for '{category}'...")
        detector = build_detector_for_category(
            model=model,
            category=category,
            data_root=args.data_root,
            save_dir=args.checkpoint_dir,
            device=device,
            k=args.k,
            coreset_ratio=args.coreset_ratio,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )

        # --- Evaluate on test set ---
        print(f"\n[3b] Evaluating on test set for '{category}'...")

        test_loader = get_dataloader(
            root_dir=args.data_root,
            category=category,
            split="test",
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            return_label=True,
            transform=eval_transform,
        )

        scores, labels = detector.evaluate(model, test_loader, device)

        # --- Compute metrics ---
        metrics = evaluate_scores(
            scores, labels,
            results_dir=cat_results_dir,
            category=category,
        )

        # Load baseline for comparison
        baseline = load_baseline_metrics(args.results_dir, category)

        # Print results
        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  Results for '{category}'                    ")
        print(f"  ├─────────────────────────────────────────────┤")
        print(f"  │  I-JEPA  ROC-AUC: {metrics['roc_auc']:.4f}"
              f"  AP: {metrics['average_precision']:.4f}"
              f"  F1: {metrics['f1_at_best_threshold']:.4f}")
        if baseline:
            print(f"  │  Baseline ROC-AUC: {baseline.get('roc_auc', '?'):.4f}"
                  f"  AP: {baseline.get('average_precision', '?'):.4f}"
                  f"  F1: {baseline.get('f1_at_best_threshold', '?'):.4f}")
            delta = metrics["roc_auc"] - baseline.get("roc_auc", 0)
            print(f"  │  Improvement: {delta:+.4f} AUC over baseline")
        print(f"  │  n_normal={metrics['n_normal']}  n_anomaly={metrics['n_anomaly']}")
        print(f"  │  Confusion: TN={metrics['confusion_matrix']['tn']}  "
              f"FP={metrics['confusion_matrix']['fp']}  "
              f"FN={metrics['confusion_matrix']['fn']}  "
              f"TP={metrics['confusion_matrix']['tp']}")
        print(f"  └─────────────────────────────────────────────┘")

        all_results[category] = metrics

    # =========================================================================
    # STEP 4: Summary table
    # =========================================================================

    print(f"\n\n{'=' * 75}")
    print("SUMMARY — I-JEPA Anomaly Detection Results")
    print(f"{'=' * 75}")
    print(f"{'Category':<15}  {'ROC-AUC':>8}  {'AP':>8}  {'F1':>8}  {'vs Baseline':>12}")
    print("-" * 60)

    aucs = []
    for cat, m in all_results.items():
        baseline = load_baseline_metrics(args.results_dir, cat)
        delta_str = ""
        if baseline:
            delta = m["roc_auc"] - baseline.get("roc_auc", 0.0)
            delta_str = f"{delta:+.4f}"
        print(
            f"{cat:<15}  {m['roc_auc']:>8.4f}  "
            f"{m['average_precision']:>8.4f}  "
            f"{m['f1_at_best_threshold']:>8.4f}  "
            f"{delta_str:>12}"
        )
        aucs.append(m["roc_auc"])

    if aucs:
        print("-" * 60)
        print(f"{'MEAN':<15}  {np.mean(aucs):>8.4f}")

    print(f"\nResults saved in: {os.path.join(args.results_dir, 'ijepa', '<category>')}/")
    print(f"Detectors saved in: {args.checkpoint_dir}/ijepa_detector_<category>.pkl")
    print("=" * 75)

    return all_results


if __name__ == "__main__":
    main()
