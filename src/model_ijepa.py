"""
I-JEPA Model Implementation — Phase 3

Image-based Joint-Embedding Predictive Architecture
Based on: Assran et al., "Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture", CVPR 2023.

Architecture:
  Context Encoder  — ViT that processes visible (context) patches
  Target Encoder   — EMA copy of context encoder (no gradient)
  Predictor        — Narrow ViT: maps context reps → predicted target reps

Anomaly Detection (after training):
  - Extract all 196 patch features from context encoder (full image, no masking)
  - Build k-NN memory bank from normal training images
  - Score = max patch-level distance to nearest normal neighbors
"""

import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Drop Path (Stochastic Depth)
# =============================================================================

class DropPath(nn.Module):
    """Stochastic depth regularization (per-sample drop)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device)
        noise.floor_()
        return x * noise / keep_prob


# =============================================================================
# Attention + MLP + Transformer Block
# =============================================================================

class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each [B, heads, N, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward MLP block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    """Standard Transformer block: LayerNorm → Attention → LayerNorm → MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =============================================================================
# Patch Embedding
# =============================================================================

class PatchEmbed(nn.Module):
    """Divide image into non-overlapping patches and project to embedding dim."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → [B, num_patches, embed_dim]
        x = self.proj(x)            # [B, D, H//P, W//P]
        x = x.flatten(2)            # [B, D, num_patches]
        x = x.transpose(1, 2)       # [B, num_patches, D]
        return x


# =============================================================================
# Vision Transformer — Context & Target Encoder
# =============================================================================

class VisionTransformer(nn.Module):
    """
    Vision Transformer used as both context encoder and target encoder in I-JEPA.

    forward(x, mask_indices=None):
        If mask_indices is provided, only the specified patches are processed
        (context encoder mode — processes only visible/context patches).
        If None, all patches are processed (target encoder mode / inference).

    Returns: (patch_tokens [B, N, D], cls_token [B, 1, D])
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=None,
        init_std: float = 0.02,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # pos_embed: [1, num_patches+1, D]  (position 0 = CLS)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth schedule (linearly increases from 0 to drop_path_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self._init_weights(init_std)

    def _init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:            [B, C, H, W] input images
            mask_indices: [B, N_keep] indices of patches to keep (context encoder).
                          If None, all 196 patches are used (target encoder / inference).

        Returns:
            patch_tokens: [B, N_keep, D]   (N_keep = num_patches if mask_indices is None)
            cls_token:    [B, 1, D]
        """
        B = x.shape[0]
        D = self.embed_dim

        # 1. Patch embedding (all patches)
        x = self.patch_embed(x)  # [B, num_patches, D]

        # 2. Add absolute patch positional embeddings (positions 1..num_patches+1)
        x = x + self.pos_embed[:, 1:, :]  # [B, num_patches, D]

        # 3. Optionally select only context patches
        if mask_indices is not None:
            # mask_indices: [B, N_ctx]
            idx = mask_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, N_ctx, D]
            x = torch.gather(x, 1, idx)  # [B, N_ctx, D]

        # 4. Prepend CLS token (with its positional embedding)
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, x], dim=1)  # [B, N+1, D]
        x = self.pos_drop(x)

        # 5. Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_token = x[:, :1, :]    # [B, 1, D]
        patch_tokens = x[:, 1:, :] # [B, N, D]
        return patch_tokens, cls_token

    @torch.no_grad()
    def get_all_patch_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full image inference — extract all 196 patch tokens + CLS.
        Used during anomaly detection (memory bank building and scoring).
        """
        return self.forward(x, mask_indices=None)


# =============================================================================
# I-JEPA Predictor (Narrow ViT)
# =============================================================================

class IJEPAPredictor(nn.Module):
    """
    Narrow Vision Transformer that predicts target patch representations
    from context patch representations.

    Architecture:
      - Project context tokens (encoder_dim → predictor_dim)
      - Add predictor positional embeddings to context tokens
      - Add learnable mask tokens + target positional embeddings for each target
      - Process all tokens jointly through predictor transformer blocks
      - Extract target-position outputs and project back (predictor_dim → encoder_dim)

    This allows the predictor to know WHERE each token is (via positional embeddings)
    and predict what the target encoder would produce at target positions.
    """

    def __init__(
        self,
        encoder_dim: int = 384,
        predictor_dim: int = 192,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=None,
        num_patches: int = 196,
        init_std: float = 0.02,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.predictor_dim = predictor_dim
        self.encoder_dim = encoder_dim
        self.num_patches = num_patches

        # Linear projections
        self.input_proj = nn.Linear(encoder_dim, predictor_dim, bias=True)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim, bias=True)

        # Learnable mask token (one shared token, position-conditioned via pos_embed)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        # Position embeddings for all possible patch positions (in predictor dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(predictor_dim)
        self._init_weights(init_std)

    def _init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        context_tokens: torch.Tensor,    # [B, N_ctx, encoder_dim]
        context_indices: torch.Tensor,   # [B, N_ctx]  patch positions of context
        target_indices: torch.Tensor,    # [B, N_tgt]  patch positions of targets
    ) -> torch.Tensor:
        """
        Returns:
            predicted_targets: [B, N_tgt, encoder_dim]
        """
        B = context_tokens.shape[0]
        N_ctx = context_tokens.shape[1]
        N_tgt = target_indices.shape[1]
        P = self.predictor_dim

        # 1. Project context tokens to predictor dim
        ctx = self.input_proj(context_tokens)  # [B, N_ctx, P]

        # 2. Add positional embeddings for context positions
        pos_full = self.pos_embed.expand(B, -1, -1)  # [B, num_patches, P]
        ctx_pos = torch.gather(
            pos_full, 1,
            context_indices.unsqueeze(-1).expand(-1, -1, P)
        )  # [B, N_ctx, P]
        ctx = ctx + ctx_pos

        # 3. Create mask tokens for target positions
        mask_tokens = self.mask_token.expand(B, N_tgt, -1)  # [B, N_tgt, P]
        tgt_pos = torch.gather(
            pos_full, 1,
            target_indices.unsqueeze(-1).expand(-1, -1, P)
        )  # [B, N_tgt, P]
        mask_tokens = mask_tokens + tgt_pos  # positional conditioning

        # 4. Concatenate context + mask tokens
        x = torch.cat([ctx, mask_tokens], dim=1)  # [B, N_ctx + N_tgt, P]

        # 5. Process through predictor transformer
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # 6. Extract target-position outputs (last N_tgt tokens)
        pred = x[:, N_ctx:, :]  # [B, N_tgt, P]

        # 7. Project back to encoder dim
        pred = self.output_proj(pred)  # [B, N_tgt, encoder_dim]
        return pred


# =============================================================================
# Full I-JEPA Model
# =============================================================================

class IJEPA(nn.Module):
    """
    I-JEPA: Image-based Joint-Embedding Predictive Architecture.

    Training forward():
        1. Context encoder processes context patches only (with gradient)
        2. Target encoder processes full image (no gradient, EMA weights)
        3. Predictor maps context features → predicted target features
        4. Loss: MSE(predicted, stop_grad(actual target features))

    Inference extract_features():
        - Context encoder processes ALL 196 patches (no masking)
        - Returns patch tokens for k-NN anomaly scoring
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        # Encoder (context & target share architecture, different weights)
        encoder_embed_dim: int = 384,
        encoder_depth: int = 12,
        encoder_num_heads: int = 6,
        # Predictor (narrower)
        predictor_embed_dim: int = 192,
        predictor_depth: int = 6,
        predictor_num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        # EMA momentum (will be updated externally during training)
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.ema_momentum = ema_momentum
        self.encoder_embed_dim = encoder_embed_dim

        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        self.grid_size = img_size // patch_size

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # --- Context Encoder (trained with gradient) ---
        self.context_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )

        # --- Target Encoder (EMA of context encoder — NO gradient) ---
        self.target_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=0.0,  # No stochastic depth in target encoder
            norm_layer=norm_layer,
        )

        # Initialize target encoder = copy of context encoder
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_tgt.data.copy_(p_ctx.data)
            p_tgt.requires_grad_(False)  # Target encoder never receives gradient

        # --- Predictor ---
        self.predictor = IJEPAPredictor(
            encoder_dim=encoder_embed_dim,
            predictor_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio,
            num_patches=num_patches,
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None):
        """
        EMA update: target_param ← m * target_param + (1-m) * context_param
        Called after every training step.
        """
        m = momentum if momentum is not None else self.ema_momentum
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_tgt.data.mul_(m).add_((1.0 - m) * p_ctx.data)

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            images:              [B, C, H, W]
            context_indices:     [B, N_ctx]  — indices of context patches
            target_indices_list: list of [B, N_tgt_i] — one tensor per target block

        Returns:
            loss: scalar (mean MSE across all target blocks)
        """
        B = images.shape[0]

        # 1. Context encoder (with gradient) — processes only context patches
        ctx_tokens, _ = self.context_encoder(images, mask_indices=context_indices)
        # ctx_tokens: [B, N_ctx, D]

        # 2. Target encoder (no gradient) — processes all patches
        with torch.no_grad():
            tgt_tokens_full, _ = self.target_encoder(images, mask_indices=None)
            # tgt_tokens_full: [B, num_patches, D]
            # Normalize target representations (stabilizes training, from I-JEPA paper)
            tgt_tokens_full = F.layer_norm(
                tgt_tokens_full, (tgt_tokens_full.shape[-1],)
            )

        # 3. For each target block: predict and compute loss
        total_loss = 0.0
        n_targets = 0

        for tgt_indices in target_indices_list:
            # Gather actual target representations from target encoder output
            tgt_reps = torch.gather(
                tgt_tokens_full, 1,
                tgt_indices.unsqueeze(-1).expand(-1, -1, tgt_tokens_full.shape[-1]),
            )  # [B, N_tgt, D]

            # Predict target representations (stop gradient on tgt_reps)
            pred_reps = self.predictor(ctx_tokens, context_indices, tgt_indices)
            # pred_reps: [B, N_tgt, D]

            # L2 loss in embedding space
            loss = F.mse_loss(pred_reps, tgt_reps.detach())
            total_loss += loss
            n_targets += 1

        return total_loss / max(n_targets, 1)

    @torch.no_grad()
    def extract_features(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference: extract all patch features + CLS from context encoder.
        Used for building memory bank and scoring during anomaly detection.

        Args:
            images: [B, C, H, W]

        Returns:
            patch_features: [B, num_patches, embed_dim]  — local patch tokens
            cls_features:   [B, embed_dim]               — global image token
        """
        self.context_encoder.eval()
        patch_tokens, cls_token = self.context_encoder.get_all_patch_features(images)
        cls_features = cls_token.squeeze(1)  # [B, D]
        return patch_tokens, cls_features


# =============================================================================
# Model Factories
# =============================================================================

def get_ijepa_tiny(img_size: int = 224, patch_size: int = 16, **kwargs) -> IJEPA:
    """
    ViT-Tiny/16 encoder: 192-dim, 12 layers, 3 heads.
    Very fast — useful if GTX 1650 runs out of memory with Small.
    ~5M params, ~20MB per encoder.
    """
    return IJEPA(
        img_size=img_size,
        patch_size=patch_size,
        encoder_embed_dim=192,
        encoder_depth=12,
        encoder_num_heads=3,
        predictor_embed_dim=96,
        predictor_depth=4,
        predictor_num_heads=3,
        **kwargs,
    )


def get_ijepa_small(img_size: int = 224, patch_size: int = 16, **kwargs) -> IJEPA:
    """
    ViT-Small/16 encoder: 384-dim, 12 layers, 6 heads.
    Recommended for GTX 1650 4GB: good balance of capacity and speed.
    ~22M params per encoder, ~88MB in float32 (~44MB in float16).
    """
    return IJEPA(
        img_size=img_size,
        patch_size=patch_size,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        predictor_embed_dim=192,
        predictor_depth=6,
        predictor_num_heads=4,
        **kwargs,
    )


def get_ijepa_base(img_size: int = 224, patch_size: int = 16, **kwargs) -> IJEPA:
    """
    ViT-Base/16 encoder: 768-dim, 12 layers, 12 heads.
    Higher capacity — requires more VRAM (may be tight on 4GB).
    ~86M params per encoder. Use batch_size=4 or 8 with AMP.
    """
    return IJEPA(
        img_size=img_size,
        patch_size=patch_size,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        predictor_embed_dim=384,
        predictor_depth=6,
        predictor_num_heads=6,
        **kwargs,
    )


def load_ijepa_from_checkpoint(
    checkpoint_path: str,
    device: torch.device = None,
) -> IJEPA:
    """Load a trained I-JEPA model from a checkpoint file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_size = ckpt.get("model_size", "small")
    config = ckpt.get("config", {})

    if model_size == "tiny":
        model = get_ijepa_tiny()
    elif model_size == "base":
        model = get_ijepa_base()
    else:
        model = get_ijepa_small()

    model.context_encoder.load_state_dict(ckpt["context_encoder_state"])
    model.target_encoder.load_state_dict(ckpt["target_encoder_state"])
    if "predictor_state" in ckpt:
        model.predictor.load_state_dict(ckpt["predictor_state"])

    model = model.to(device)
    return model


# =============================================================================
# Quick architecture test
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing I-JEPA on {device}")

    model = get_ijepa_small().to(device)

    B, C, H, W = 2, 3, 224, 224
    images = torch.randn(B, C, H, W, device=device)

    # Simulate masking
    num_patches = 196
    n_ctx = 150
    n_tgt = 30

    all_idx = torch.randperm(num_patches)
    ctx_idx = all_idx[:n_ctx].unsqueeze(0).expand(B, -1).to(device)
    tgt_idx = all_idx[n_ctx : n_ctx + n_tgt].unsqueeze(0).expand(B, -1).to(device)

    loss = model(images, ctx_idx, [tgt_idx])
    print(f"Training loss: {loss.item():.4f}")

    patch_feat, cls_feat = model.extract_features(images)
    print(f"Patch features: {patch_feat.shape}")   # [2, 196, 384]
    print(f"CLS features:   {cls_feat.shape}")     # [2, 384]

    n_ctx_p = sum(p.numel() for p in model.context_encoder.parameters())
    n_pred_p = sum(p.numel() for p in model.predictor.parameters())
    print(f"Context encoder: {n_ctx_p/1e6:.1f}M params")
    print(f"Predictor:       {n_pred_p/1e6:.1f}M params")
    print("Model test PASSED.")
