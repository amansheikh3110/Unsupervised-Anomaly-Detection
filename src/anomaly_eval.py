"""
Anomaly Evaluation: Reconstruction-based baseline

Compute anomaly scores (reconstruction error) and metrics (ROC-AUC, etc.).
Use ground truth ONLY for evaluation.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import MVTecDataset, get_dataloader, eval_transform
from src.model_autoencoder import get_autoencoder
from src.utils import set_seed, get_device, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline autoencoder anomaly detection")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--category", type=str, default="leather")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to autoencoder checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def compute_anomaly_scores(model, loader, device, reduction="mean"):
    """
    For each image, anomaly score = reconstruction error (MSE).
    reduction: 'mean' = mean over pixels (one score per image)
    """
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing scores"):
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
                y = batch[1]
            else:
                images = batch.to(device)
                y = None

            recon, _ = model(images)
            # Per-sample MSE: (B,) when reduction='none' then mean over C,H,W
            loss_per_sample = model.reconstruction_loss(images, recon, reduction="none")
            if loss_per_sample.dim() > 1:
                loss_per_sample = loss_per_sample.view(loss_per_sample.size(0), -1).mean(dim=1)
            scores.append(loss_per_sample.cpu().numpy())
            if y is not None:
                labels.append(y.numpy())

    scores = np.concatenate(scores, axis=0)
    if labels:
        labels = np.concatenate(labels, axis=0)
    else:
        labels = None

    return scores, labels


def evaluate(scores, labels, results_dir=None, prefix=""):
    """Compute ROC-AUC, AP, and optional PR curve / confusion matrix."""
    # Binary: 0 = normal, 1 = anomaly
    y_true = labels.astype(int)
    # Higher score = more anomalous
    y_score = scores.astype(np.float64)

    roc_auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # Best F1 threshold from precision-recall curve
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])  # last is 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else float(np.median(y_score))
    y_pred = (y_score >= best_threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "f1_at_best_threshold": float(f1),
        "best_threshold": float(best_threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_normal": int((y_true == 0).sum()),
        "n_anomaly": int((y_true == 1).sum()),
    }

    if results_dir:
        ensure_dir(results_dir)
        path = os.path.join(results_dir, f"{prefix}metrics_baseline.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {path}")

    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    latent_dim = ckpt.get("latent_dim", 256)
    model = get_autoencoder(in_channels=3, latent_dim=latent_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Test loader with labels
    test_loader = get_dataloader(
        root_dir=args.data_root,
        category=args.category,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        return_label=True,
        transform=eval_transform,
    )

    print("=" * 60)
    print("Baseline Autoencoder - Anomaly Evaluation")
    print("=" * 60)
    print(f"Category:   {args.category}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 60)

    scores, labels = compute_anomaly_scores(model, test_loader, device)
    if labels is None:
        print("No labels in loader; cannot compute metrics.")
        return

    results_dir = os.path.join(args.results_dir, "baseline", args.category)
    prefix = ""
    metrics = evaluate(scores, labels, results_dir=results_dir, prefix=prefix)

    print("\nMetrics:")
    print(f"  ROC-AUC:              {metrics['roc_auc']:.4f}")
    print(f"  Average Precision:    {metrics['average_precision']:.4f}")
    print(f"  F1 (best threshold):  {metrics['f1_at_best_threshold']:.4f}")
    print(f"  Best threshold:       {metrics['best_threshold']:.4f}")
    print(f"  Confusion matrix:     TN={metrics['confusion_matrix']['tn']} FP={metrics['confusion_matrix']['fp']} "
          f"FN={metrics['confusion_matrix']['fn']} TP={metrics['confusion_matrix']['tp']}")
    print("=" * 60)

    # Save scores and labels for later comparison
    np.save(os.path.join(results_dir, "test_scores.npy"), scores)
    np.save(os.path.join(results_dir, "test_labels.npy"), labels)
    print(f"Saved scores and labels to {results_dir}")

    return metrics


if __name__ == "__main__":
    main()
