"""
Baseline Autoencoder for Anomaly Detection

Reconstruction-based baseline: train on normal images only.
Anomaly score = reconstruction error (MSE). Higher error → more likely anomaly.

Used in Phase 2 to establish a baseline; I-JEPA will be compared against this.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> (optional) MaxPool."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        pool: bool = False,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class DeconvBlock(nn.Module):
    """Upsample -> Conv -> BatchNorm -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, 1, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Autoencoder(nn.Module):
    """
    Convolutional Autoencoder for 224x224 RGB images.

    Encoder: 224x224x3 -> ... -> 7x7xlatent_dim
    Decoder: 7x7xlatent_dim -> ... -> 224x224x3

    Anomaly score = mean squared error between input and reconstruction.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        base_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Encoder: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels, pool=True),      # 224 -> 112
            ConvBlock(base_channels, base_channels * 2, pool=True),  # 112 -> 56
            ConvBlock(base_channels * 2, base_channels * 4, pool=True),  # 56 -> 28
            ConvBlock(base_channels * 4, base_channels * 8, pool=True),  # 28 -> 14
            ConvBlock(base_channels * 8, base_channels * 8, pool=True),  # 14 -> 7
        )
        # 7*7*256 = 12544 -> project to latent_dim
        self.enc_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * (base_channels * 8), 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, latent_dim),
        )

        self.dec_proj = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 7 * 7 * (base_channels * 8)),
            nn.ReLU(inplace=True),
        )

        # Decoder: 7 -> 14 -> 28 -> 56 -> 112 -> 224
        self.decoder = nn.Sequential(
            DeconvBlock(base_channels * 8, base_channels * 8),   # 7 -> 14
            DeconvBlock(base_channels * 8, base_channels * 4),   # 14 -> 28
            DeconvBlock(base_channels * 4, base_channels * 2),   # 28 -> 56
            DeconvBlock(base_channels * 2, base_channels),       # 56 -> 112
            nn.Upsample(scale_factor=2, mode="nearest"),        # 112 -> 224
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent vector."""
        h = self.encoder(x)
        return self.enc_proj(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        B = z.shape[0]
        h = self.dec_proj(z)
        h = h.view(B, -1, 7, 7)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            reconstruction: Reconstructed image [B, 3, 224, 224]
            latent: Latent vector [B, latent_dim]
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def reconstruction_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """MSE reconstruction loss. Use 'none' for per-sample loss."""
        return nn.functional.mse_loss(x, recon, reduction=reduction)


def get_autoencoder(
    in_channels: int = 3,
    latent_dim: int = 256,
    base_channels: int = 32,
) -> Autoencoder:
    """Factory for Autoencoder."""
    return Autoencoder(
        in_channels=in_channels,
        latent_dim=latent_dim,
        base_channels=base_channels,
    )


if __name__ == "__main__":
    # Quick test
    model = get_autoencoder()
    x = torch.randn(2, 3, 224, 224)
    recon, z = model(x)
    print("Input shape:", x.shape)
    print("Reconstruction shape:", recon.shape)
    print("Latent shape:", z.shape)
    loss = model.reconstruction_loss(x, recon)
    print("MSE loss:", loss.item())
