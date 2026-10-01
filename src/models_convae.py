from __future__ import annotations
import torch
from torch import nn


class ConvAE(nn.Module):
    """CPU-friendly 2-D convolutional autoencoder for 1-channel SEM images."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
