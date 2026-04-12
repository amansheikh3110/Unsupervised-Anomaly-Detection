"""
I-JEPA Self-Supervised Training — Phase 3

Train I-JEPA on NORMAL (good) images only using the joint-embedding
predictive objective:
  - Context encoder processes visible (context) patches
  - Target encoder (EMA) processes all patches
  - Predictor maps context → predicted target representations
  - Loss: MSE(predicted, stop_grad(target)) in embedding space

Key training details (from Assran et al. 2023):
  - Optimizer: AdamW (lr=1.5e-4, wd=0.05, betas=(0.9, 0.95))
  - LR schedule: linear warmup (10 epochs) + cosine decay
  - EMA momentum: linear schedule from 0.996 → 1.0
  - AMP (float16) for memory efficiency on GTX 1650 4GB
  - Only context_encoder + predictor params receive gradient
    (target_encoder is a pure EMA copy, never receives gradient)

Usage:
    # Single category
    python -m src.train_ijepa --category leather --epochs 100

    # Multiple categories (trains one shared model)
    python -m src.train_ijepa --categories leather bottle cable --epochs 100

    # All 10 recommended categories
    python -m src.train_ijepa --categories bottle cable capsule carpet grid hazelnut leather metal_nut pill screw --epochs 100
"""

import math
import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import MVTecDataset, get_dataloader, train_transform
from src.model_ijepa import get_ijepa_tiny, get_ijepa_small, get_ijepa_base, IJEPA
from src.masks import BlockMaskGenerator
from src.utils import set_seed, get_device, ensure_dir, AverageMeter


# =============================================================================
# Training hyperparameter helpers
# =============================================================================

def cosine_lr(
    epoch: int,
    total_epochs: int,
    base_lr: float,
    warmup_epochs: int,
    min_lr: float = 1e-6,
) -> float:
    """Linear warm-up then cosine decay to min_lr."""
    if epoch < warmup_epochs:
        return base_lr * epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def ema_schedule(
    step: int,
    total_steps: int,
    ema_start: float = 0.996,
    ema_end: float = 1.0,
) -> float:
    """Cosine schedule: ema_start → ema_end over training."""
    progress = step / max(total_steps, 1)
    return ema_end - (ema_end - ema_start) * (math.cos(math.pi * progress) + 1) / 2


# =============================================================================
# One training epoch
# =============================================================================

def train_one_epoch(
    model: IJEPA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mask_gen: BlockMaskGenerator,
    device: torch.device,
    scaler: GradScaler,
    epoch: int,
    total_epochs: int,
    total_steps: int,
    step_offset: int,
    ema_start: float,
    ema_end: float,
    grad_clip: float = 1.0,
) -> tuple:
    """
    Run one epoch of I-JEPA training.

    Returns:
        avg_loss:  float — mean loss for this epoch
        step:      int   — global step counter after this epoch
    """
    model.context_encoder.train()
    model.predictor.train()
    model.target_encoder.eval()  # Target encoder never trains directly

    loss_meter = AverageMeter("loss")
    step = step_offset
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False)

    for batch in pbar:
        # Unpack batch
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        images = images.to(device, non_blocking=True)
        B = images.shape[0]

        # Generate masks for this batch
        ctx_indices, tgt_indices_list = mask_gen(B, device=device)

        if not tgt_indices_list:
            # Edge case: no valid target blocks; skip
            continue

        optimizer.zero_grad(set_to_none=True)

        # Forward + loss (with optional AMP)
        if scaler is not None:
            with autocast():
                loss = model(images, ctx_indices, tgt_indices_list)
            scaler.scale(loss).backward()
            # Unscale before clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(model.context_encoder.parameters()) +
                list(model.predictor.parameters()),
                grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(images, ctx_indices, tgt_indices_list)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.context_encoder.parameters()) +
                list(model.predictor.parameters()),
                grad_clip,
            )
            optimizer.step()

        # EMA update — momentum increases as training progresses
        momentum = ema_schedule(step, total_steps, ema_start, ema_end)
        model.update_target_encoder(momentum=momentum)

        step += 1
        loss_val = loss.item()
        loss_meter.update(loss_val, B)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", mom=f"{momentum:.5f}")

    return loss_meter.avg, step


# =============================================================================
# Argument parser
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train I-JEPA on MVTec normal images (Phase 3)"
    )

    # Data
    p.add_argument("--data_root", type=str, default="data",
                   help="Path to MVTec AD dataset root")
    p.add_argument(
        "--categories", type=str, nargs="+",
        default=["leather"],
        help=(
            "Categories to train on. Use multiple for a shared model. "
            "E.g.: --categories bottle cable capsule carpet grid hazelnut "
            "leather metal_nut pill screw"
        ),
    )

    # Model
    p.add_argument(
        "--model_size", type=str, default="small",
        choices=["tiny", "small", "base"],
        help=(
            "Encoder size. 'small' (384-dim, 22M params) recommended for GTX 1650. "
            "'tiny' (192-dim) if VRAM is insufficient. 'base' (768-dim) for higher accuracy."
        ),
    )

    # Training
    p.add_argument("--epochs", type=int, default=100,
                   help="Training epochs. 100 recommended; 200 for higher accuracy.")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size. Reduce to 8 if CUDA out of memory.")
    p.add_argument("--lr", type=float, default=1.5e-4,
                   help="Peak learning rate (AdamW).")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=10,
                   help="Linear LR warmup epochs.")
    p.add_argument("--min_lr", type=float, default=1e-6,
                   help="Minimum LR after cosine decay.")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clipping max norm.")
    p.add_argument("--amp", type=lambda x: x.lower() != "false", default=True,
                   help="Mixed precision training (AMP). Set false to disable.")

    # EMA
    p.add_argument("--ema_start", type=float, default=0.996,
                   help="Initial EMA momentum (slowly rises to ema_end).")
    p.add_argument("--ema_end", type=float, default=1.0,
                   help="Final EMA momentum.")

    # Masking
    p.add_argument("--num_targets", type=int, default=4,
                   help="Number of target blocks per image.")
    p.add_argument("--target_scale_min", type=float, default=0.15,
                   help="Minimum target block area fraction.")
    p.add_argument("--target_scale_max", type=float, default=0.20,
                   help="Maximum target block area fraction.")

    # Output
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                   help="Directory to save model checkpoints.")
    p.add_argument("--save_name", type=str, default=None,
                   help="Checkpoint filename (auto-generated if not provided).")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)

    return p.parse_args()


# =============================================================================
# Main training function
# =============================================================================

def train(args):
    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.checkpoint_dir)

    # Determine checkpoint filename
    cats_tag = (
        "_".join(args.categories)
        if len(args.categories) <= 3
        else f"{len(args.categories)}cats"
    )
    save_name = args.save_name or f"ijepa_{args.model_size}_{cats_tag}.pth"
    save_path = os.path.join(args.checkpoint_dir, save_name)

    print("=" * 70)
    print("I-JEPA Self-Supervised Training — Phase 3")
    print("=" * 70)
    print(f"Categories:     {args.categories}")
    print(f"Model size:     {args.model_size}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"LR:             {args.lr}  (warmup {args.warmup_epochs} epochs)")
    print(f"EMA:            {args.ema_start} → {args.ema_end}")
    print(f"Target blocks:  {args.num_targets}  scale={args.target_scale_min}-{args.target_scale_max}")
    print(f"AMP:            {args.amp}")
    print(f"Device:         {device}")
    print(f"Checkpoint:     {save_path}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    if len(args.categories) == 1:
        train_loader = get_dataloader(
            root_dir=args.data_root,
            category=args.categories[0],
            split="train",
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            return_label=False,
            transform=train_transform,
        )
        print(f"Training on {len(train_loader.dataset)} normal images")
    else:
        # Combine multiple categories
        datasets = []
        for cat in args.categories:
            ds = MVTecDataset(
                args.data_root, cat, "train", transform=train_transform
            )
            if len(ds) == 0:
                print(f"  WARNING: category '{cat}' has no training images — skipping.")
                continue
            datasets.append(ds)
            print(f"  {cat}: {len(ds)} images")

        if not datasets:
            raise RuntimeError("No training images found for any category.")

        combined = ConcatDataset(datasets)
        train_loader = DataLoader(
            combined,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,   # ensures uniform batch size
        )
        print(f"Combined: {len(combined)} normal images")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    if args.model_size == "tiny":
        model = get_ijepa_tiny().to(device)
    elif args.model_size == "base":
        model = get_ijepa_base().to(device)
    else:
        model = get_ijepa_small().to(device)

    n_ctx = sum(p.numel() for p in model.context_encoder.parameters())
    n_pred = sum(p.numel() for p in model.predictor.parameters())
    print(f"Context encoder: {n_ctx/1e6:.1f}M params")
    print(f"Predictor:       {n_pred/1e6:.1f}M params")

    # -------------------------------------------------------------------------
    # Mask generator
    # -------------------------------------------------------------------------
    mask_gen = BlockMaskGenerator(
        grid_size=14,  # 224 / 16
        num_targets=args.num_targets,
        target_scale=(args.target_scale_min, args.target_scale_max),
        target_aspect_ratio=(0.75, 1.50),
        context_scale=(0.85, 1.00),
    )
    print(f"Mask generator: {mask_gen}")

    # -------------------------------------------------------------------------
    # Optimizer (only context encoder + predictor parameters)
    # -------------------------------------------------------------------------
    trainable_params = (
        list(model.context_encoder.parameters())
        + list(model.predictor.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # AMP scaler
    use_amp = args.amp and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision (AMP float16) enabled.")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    global_step = 0
    history = {"train_loss": [], "lr": []}
    best_loss = float("inf")

    for epoch in range(args.epochs):
        # Update learning rate
        lr = cosine_lr(epoch, args.epochs, args.lr, args.warmup_epochs, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Train one epoch
        avg_loss, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            mask_gen=mask_gen,
            device=device,
            scaler=scaler,
            epoch=epoch,
            total_epochs=args.epochs,
            total_steps=total_steps,
            step_offset=global_step,
            ema_start=args.ema_start,
            ema_end=args.ema_end,
            grad_clip=args.grad_clip,
        )

        history["train_loss"].append(avg_loss)
        history["lr"].append(lr)

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs}  "
            f"Loss: {avg_loss:.5f}  LR: {lr:.2e}"
        )

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_checkpoint(model, optimizer, epoch, avg_loss, save_path, args)
            print(f"  → Best loss {best_loss:.5f} — saved checkpoint.")

    # Save final checkpoint (regardless of whether it is best)
    final_path = save_path.replace(".pth", "_final.pth")
    _save_checkpoint(model, optimizer, args.epochs - 1, history["train_loss"][-1],
                     final_path, args)

    # Save training history
    hist_path = save_path.replace(".pth", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 70)
    print(f"Training complete! Best loss: {best_loss:.5f}")
    print(f"Best checkpoint:  {save_path}")
    print(f"Final checkpoint: {final_path}")
    print(f"History:          {hist_path}")
    print("=" * 70)

    return save_path


def _save_checkpoint(model, optimizer, epoch, loss, path, args):
    torch.save(
        {
            "epoch": epoch + 1,
            "context_encoder_state": model.context_encoder.state_dict(),
            "target_encoder_state": model.target_encoder.state_dict(),
            "predictor_state": model.predictor.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": loss,
            "categories": args.categories,
            "model_size": args.model_size,
            "config": vars(args),
        },
        path,
    )


# =============================================================================
# Entry point
# =============================================================================

def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
