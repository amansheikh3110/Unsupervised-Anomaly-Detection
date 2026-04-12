"""
PretrainedViTWrapper — drop-in replacement for the I-JEPA model.

Uses torchvision's ViT-B/16 pretrained on ImageNet (no training required).
Exposes the same extract_features() interface as the IJEPA class so that
the existing IJEPAAnomalyDetector works without any changes.

Why this gives better accuracy:
  - ViT-B/16 was trained on 1.2M ImageNet images.
  - Its patch features already encode semantic shape, texture, and structure.
  - Even for industrial images (hazelnut, cable, ...) these features
    generalise well — normal patches cluster tightly, anomaly patches
    appear as outliers in the k-NN distance space.
  - 50-epoch I-JEPA trained from scratch on ~400 images cannot match
    the representational quality of an ImageNet-pretrained backbone.
"""

import torch
import torch.nn as nn


class _ContextEncoderProxy:
    """Thin proxy so that detector.context_encoder.eval() calls work."""

    def __init__(self, model: nn.Module):
        self._m = model

    def eval(self):
        self._m.eval()
        return self


class PretrainedViTWrapper:
    """
    Wraps torchvision's pretrained ViT-B/16 to match the IJEPA interface.

    Attributes mirrored from IJEPA:
        encoder_embed_dim  — 768 for ViT-B
        context_encoder    — proxy with .eval() so detector code doesn't break
        num_patches        — 196  (14×14 for 224×224 / patch16)

    Methods mirrored from IJEPA:
        extract_features(images) → (patch_tokens [B,196,768], cls [B,768])
    """

    def __init__(self, device: torch.device = None):
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self._vit = vit_b_16(weights=weights)
        except Exception:
            # Fallback for older torchvision API
            import torchvision.models as models
            self._vit = models.vit_b_16(pretrained=True)

        self._vit.eval()
        if device is not None:
            self._vit = self._vit.to(device)
        self._device = device

        self.encoder_embed_dim = 768  # ViT-B hidden dim
        self.num_patches = 196        # 14×14 for 224×224 / patch16
        self.context_encoder = _ContextEncoderProxy(self._vit)

    @torch.no_grad()
    def extract_features(
        self, images: torch.Tensor
    ):
        """
        Extract patch-level features from the pretrained ViT.

        Args:
            images: [B, C, 224, 224]

        Returns:
            patch_tokens: [B, 196, 768]
            cls_features: [B, 768]
        """
        self._vit.eval()
        images = images.to(next(self._vit.parameters()).device)

        # Step through ViT internals to get patch tokens before the head
        # (torchvision VisionTransformer internal API — stable since 0.13)
        x = self._vit._process_input(images)   # [B, 196, 768]
        B = x.shape[0]
        cls = self._vit.class_token.expand(B, -1, -1)  # [B, 1, 768]
        x = torch.cat([cls, x], dim=1)          # [B, 197, 768]
        x = self._vit.encoder(x)                # [B, 197, 768]

        patch_tokens = x[:, 1:, :]              # [B, 196, 768]
        cls_features = x[:, 0, :]               # [B, 768]
        return patch_tokens, cls_features

    # ------------------------------------------------------------------
    # Multi-layer feature extraction (optional; better spatial resolution)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_multilayer_features(
        self, images: torch.Tensor, layers: tuple = (-1, -2, -3)
    ):
        """
        Average features from the last N transformer blocks.
        Captures both fine-grained (early) and semantic (late) information.

        Args:
            images: [B, C, 224, 224]
            layers: which block indices to average (default: last 3)

        Returns:
            patch_tokens: [B, 196, 768]
            cls_features: [B, 768]
        """
        self._vit.eval()
        images = images.to(next(self._vit.parameters()).device)

        x = self._vit._process_input(images)
        B = x.shape[0]
        cls = self._vit.class_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        block_outputs = []
        encoder_blocks = self._vit.encoder.layers
        total = len(encoder_blocks)
        target_indices = set(i % total for i in layers)

        for i, block in enumerate(encoder_blocks):
            x = block(x)
            if i in target_indices:
                block_outputs.append(x.clone())

        # Average patch tokens across selected layers
        stacked = torch.stack([o[:, 1:, :] for o in block_outputs], dim=0)  # [L, B, 196, D]
        patch_tokens = stacked.mean(dim=0)  # [B, 196, D]
        cls_features = x[:, 0, :]          # [B, D]  (from final layer)
        return patch_tokens, cls_features
