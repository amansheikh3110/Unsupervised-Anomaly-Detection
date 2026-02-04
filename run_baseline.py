"""
Run Phase 2: Train baseline autoencoder and evaluate.

Usage:
  python run_baseline.py --category leather
  python run_baseline.py --category bottle --epochs 30
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--category", type=str, default="leather")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--skip_train", action="store_true", help="Only evaluate (use existing checkpoint)")
    args = parser.parse_args()

    ckpt_name = f"autoencoder_{args.category}.pth"
    ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)

    if not args.skip_train:
        print("Step 1: Training baseline autoencoder (normal images only)...")
        code = subprocess.call([
            sys.executable, "-m", "src.train_baseline",
            "--data_root", args.data_root,
            "--category", args.category,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--checkpoint_dir", args.checkpoint_dir,
        ])
        if code != 0:
            sys.exit(code)

    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print("\nStep 2: Evaluating anomaly detection (ROC-AUC)...")
    code = subprocess.call([
        sys.executable, "-m", "src.anomaly_eval",
        "--data_root", args.data_root,
        "--category", args.category,
        "--checkpoint", ckpt_path,
        "--batch_size", str(args.batch_size),
        "--results_dir", args.results_dir,
    ])
    if code != 0:
        sys.exit(code)

    print("\nPhase 2 complete. Results saved under results/baseline/<category>/")


if __name__ == "__main__":
    main()
