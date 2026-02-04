"""
Train Baseline Autoencoder (Phase 2)

Train on NORMAL images only. Saves checkpoint for later evaluation.
Anomaly detection uses reconstruction error at inference.
"""

import os
import sys
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import MVTecDataset, get_dataloader, train_transform
from src.model_autoencoder import get_autoencoder
from src.utils import set_seed, get_device, ensure_dir, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline autoencoder on normal images")
    parser.add_argument("--data_root", type=str, default="data", help="Path to MVTec data")
    parser.add_argument("--category", type=str, default="leather", help="MVTec category")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=256, help="Autoencoder latent dimension")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Where to save checkpoints")
    parser.add_argument("--save_name", type=str, default=None, help="Checkpoint filename (default: autoencoder_<category>.pth)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_meter = AverageMeter("loss")
    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        images = images.to(device, non_blocking=True)

        recon, _ = model(images)
        loss = model.reconstruction_loss(images, recon, reduction="mean")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    return loss_meter.avg


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    ensure_dir(args.checkpoint_dir)

    # Data: train on good only
    train_loader = get_dataloader(
        root_dir=args.data_root,
        category=args.category,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        return_label=False,
        transform=train_transform,
    )

    model = get_autoencoder(in_channels=3, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_name = args.save_name or f"autoencoder_{args.category}.pth"
    save_path = os.path.join(args.checkpoint_dir, save_name)

    print("=" * 60)
    print("Baseline Autoencoder Training (Phase 2)")
    print("=" * 60)
    print(f"Category:      {args.category}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device:        {device}")
    print(f"Checkpoint:    {save_path}")
    print("=" * 60)

    history = {"train_loss": []}

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        print(f"Epoch {epoch + 1}/{args.epochs}  Train Loss: {train_loss:.4f}")

    # Save checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": args.epochs,
            "train_loss": history["train_loss"][-1],
            "category": args.category,
            "latent_dim": args.latent_dim,
            "config": vars(args),
        },
        save_path,
    )
    print(f"Saved checkpoint to {save_path}")

    # Save training history for plotting
    history_path = save_path.replace(".pth", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history to {history_path}")

    return save_path


if __name__ == "__main__":
    main()
