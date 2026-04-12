"""
I-JEPA Block Masking Strategy

Implements the semantic-scale block masking from Assran et al. (2023).

Key design choices from the paper:
  - Target blocks: large rectangular regions (15-20% of image area each)
    with random aspect ratio. Using LARGE blocks forces the model to predict
    semantic (not just nearby) patches — producing better representations.
  - Context block: all patches NOT in any target block, optionally subsampled.
  - Multiple target blocks (4 by default) per image.

Usage:
    gen = BlockMaskGenerator(grid_size=14)   # 224/16 = 14x14 patch grid
    ctx_indices, tgt_list = gen(batch_size=16, device=device)
    # ctx_indices:  [B, N_ctx]    (same N_ctx for all samples in batch)
    # tgt_list:     list of [B, N_tgt_i]   (one per target block)
"""

import math
import random
from typing import List, Tuple

import numpy as np
import torch


class BlockMaskGenerator:
    """
    Generates I-JEPA context and target block masks for a batch of images.

    For each image:
      1. Sample `num_targets` rectangular blocks (target blocks).
         Each block has area = scale * num_patches and an aspect ratio
         sampled from target_aspect_ratio range.
      2. Context = all patches NOT in any target block.
         Optionally sub-sample context to keep context_scale fraction.

    For batching efficiency, all samples in the batch get the SAME N_ctx
    (minimum across the batch) and the SAME N_tgt per block (minimum across batch).
    This avoids variable-length padding.

    Args:
        grid_size:           Number of patches per spatial dimension (H = W = grid_size)
        num_targets:         Number of target blocks per image (default: 4)
        target_scale:        (min, max) fraction of total patches for each target block
        target_aspect_ratio: (min, max) height/width aspect ratio of target blocks
        context_scale:       (min, max) fraction of non-target patches to keep as context
                             Set max=1.0 to keep all non-target patches.
    """

    def __init__(
        self,
        grid_size: int = 14,
        num_targets: int = 4,
        target_scale: Tuple[float, float] = (0.15, 0.20),
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.50),
        context_scale: Tuple[float, float] = (0.85, 1.00),
    ):
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.num_targets = num_targets
        self.target_scale = target_scale
        self.target_aspect_ratio = target_aspect_ratio
        self.context_scale = context_scale

    def _sample_block(self) -> List[int]:
        """
        Sample one rectangular block of patches.

        Returns a sorted list of patch indices (row-major in 14×14 grid).
        """
        for _ in range(100):  # retry loop (in case clamp produces empty block)
            scale = random.uniform(*self.target_scale)
            aspect = random.uniform(*self.target_aspect_ratio)

            # Height and width in patch units
            area = scale * self.num_patches
            h = max(1, min(int(round(math.sqrt(area * aspect))), self.grid_size))
            w = max(1, min(int(round(math.sqrt(area / aspect))), self.grid_size))

            # Random top-left corner
            top = random.randint(0, self.grid_size - h)
            left = random.randint(0, self.grid_size - w)

            indices = [
                (top + r) * self.grid_size + (left + c)
                for r in range(h)
                for c in range(w)
            ]
            if len(indices) > 0:
                return indices

        # Fallback: return a random single patch
        return [random.randint(0, self.num_patches - 1)]

    def _sample_masks_for_one_image(self) -> Tuple[List[int], List[List[int]]]:
        """
        Generate masks for a single image.

        Returns:
            ctx_indices:  sorted list of context patch indices
            per_target:   list of num_targets sorted patch index lists
        """
        all_target_set: set = set()
        per_target: List[List[int]] = []

        for _ in range(self.num_targets):
            block = self._sample_block()
            per_target.append(sorted(set(block)))
            all_target_set.update(block)

        # Context = all patches not in any target block
        all_patches = set(range(self.num_patches))
        ctx_set = sorted(all_patches - all_target_set)

        # Optionally sub-sample context
        ctx_frac = random.uniform(*self.context_scale)
        n_keep = max(1, int(len(ctx_set) * ctx_frac))
        if n_keep < len(ctx_set):
            ctx_set = sorted(
                np.random.choice(ctx_set, n_keep, replace=False).tolist()
            )

        return ctx_set, per_target

    def __call__(
        self,
        batch_size: int,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate context and target masks for a full batch.

        For batching, we use the MINIMUM valid length across the batch
        for context and each target block separately — no padding needed.

        Args:
            batch_size: number of images in the batch
            device:     target device for returned tensors

        Returns:
            ctx_tensor:  [B, N_ctx]      context patch indices (long)
            tgt_tensors: list of T tensors, each [B, N_tgt_t] (long)
                         T = number of valid target blocks (may be < num_targets
                         if some blocks end up empty after de-duplication)
        """
        # 1. Sample masks for each image
        all_ctx: List[List[int]] = []
        all_per_target: List[List[List[int]]] = []  # [B][T][patch_list]

        for _ in range(batch_size):
            ctx, per_tgt = self._sample_masks_for_one_image()
            all_ctx.append(ctx)
            all_per_target.append(per_tgt)

        # 2. Find minimum context length across batch → trim to that
        min_ctx = min(len(c) for c in all_ctx)
        min_ctx = max(min_ctx, 1)

        ctx_tensor = torch.zeros(batch_size, min_ctx, dtype=torch.long)
        for b, ci in enumerate(all_ctx):
            # Use first min_ctx indices (already sorted by patch index)
            ctx_tensor[b] = torch.tensor(ci[:min_ctx], dtype=torch.long)

        # 3. Build target tensors — one per target block
        tgt_tensors: List[torch.Tensor] = []
        for t in range(self.num_targets):
            # Check all samples have non-empty block t
            sizes = [len(all_per_target[b][t]) for b in range(batch_size)]
            min_tgt = min(sizes)
            if min_tgt == 0:
                continue  # skip this block (edge case)

            tgt_t = torch.zeros(batch_size, min_tgt, dtype=torch.long)
            for b in range(batch_size):
                idxs = all_per_target[b][t][:min_tgt]
                tgt_t[b] = torch.tensor(idxs, dtype=torch.long)
            tgt_tensors.append(tgt_t)

        # 4. Move to device
        if device is not None:
            ctx_tensor = ctx_tensor.to(device)
            tgt_tensors = [t.to(device) for t in tgt_tensors]

        return ctx_tensor, tgt_tensors

    def __repr__(self) -> str:
        return (
            f"BlockMaskGenerator("
            f"grid={self.grid_size}x{self.grid_size}, "
            f"num_targets={self.num_targets}, "
            f"scale={self.target_scale}, "
            f"aspect={self.target_aspect_ratio}, "
            f"ctx_scale={self.context_scale})"
        )


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    gen = BlockMaskGenerator(grid_size=14)
    print(gen)

    B = 4
    ctx, tgts = gen(batch_size=B)
    print(f"Context shape:    {ctx.shape}      (values 0-{gen.num_patches-1})")
    print(f"Num target blocks: {len(tgts)}")
    for i, t in enumerate(tgts):
        print(f"  Target {i}: {t.shape}")

    # Verify no overlap between context and each target
    for b in range(B):
        ctx_set = set(ctx[b].tolist())
        for i, t in enumerate(tgts):
            tgt_set = set(t[b].tolist())
            overlap = ctx_set & tgt_set
            if overlap:
                print(f"  WARNING sample {b} target {i}: overlap = {overlap}")
            else:
                print(f"  Sample {b} target {i}: no overlap (OK)")
