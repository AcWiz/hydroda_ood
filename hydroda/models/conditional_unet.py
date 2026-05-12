"""FiLM-conditioned SmallResUNet for prompt-conditioned shared backbone.

No-leakage declaration:
    - FiLM modulation uses prompt vector derived from input-side features only
    - Region embeddings learned from source regions during training
    - No target query labels used in prompt construction
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from hydroda.models.resunet import ConvBlock


class FiLMLayer(nn.Module):
    """FiLM modulation: y = gamma * x + beta, initialized near identity.

    Args:
        channels: number of feature channels to modulate
        prompt_dim: dimension of prompt vector z
    """

    def __init__(self, channels: int, prompt_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(prompt_dim, channels)
        self.beta = nn.Linear(prompt_dim, channels)

        # Initialize gamma near 1 (identity), beta near 0 (no shift)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            x: [B, C, H, W] feature map
            z: [B, prompt_dim] prompt vector

        Returns:
            [B, C, H, W] modulated feature map
        """
        # gamma: [B, C], add 1 for identity initialization
        g = self.gamma(z).view(z.shape[0], -1, 1, 1) + 1.0
        # beta: [B, C]
        b = self.beta(z).view(z.shape[0], -1, 1, 1)
        return g * x + b


class FiLMConditionalResUNet(nn.Module):
    """SmallResUNet with FiLM conditioning at encoder outputs and bottleneck.

    Same architecture as SmallResUNet (3 encoder, 2 decoder), but applies
    FiLM modulation after each encoder block output (e1, e2, e3) and the
    bottleneck (b). The decoder receives modulated skip connections.

    Modulated positions:
        - e1 → FiLM → (skip to dec1, pooled to enc2)
        - e2 → FiLM → (skip to dec2, pooled to enc3)
        - e3 → FiLM → (to bottleneck)
        - b  → FiLM → (to dec2 via upsample)

    Args:
        in_channels: input channels (default 12)
        out_channels: output channels (default 2 for surface/rootzone increment)
        width: base width (default 32)
        prompt_dim: prompt vector dimension (default 64)
        zero_raw_increment_init: zero-init output head (default False)
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 2,
        width: int = 32,
        prompt_dim: int = 64,
        zero_raw_increment_init: bool = False,
    ) -> None:
        super().__init__()
        self.prompt_dim = prompt_dim

        # Encoder (same as SmallResUNet)
        self.enc1 = ConvBlock(in_channels, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.bottleneck = ConvBlock(width * 4, width * 4)

        # FiLM layers at each modulated position
        self.film1 = FiLMLayer(width, prompt_dim)
        self.film2 = FiLMLayer(width * 2, prompt_dim)
        self.film3 = FiLMLayer(width * 4, prompt_dim)
        self.film_b = FiLMLayer(width * 4, prompt_dim)

        # Decoder (same as SmallResUNet)
        self.dec2 = ConvBlock(width * 6, width * 2)
        self.dec1 = ConvBlock(width * 3, width)
        self.head = nn.Conv2d(width, out_channels, 1)

        self._zero_raw_increment_init = zero_raw_increment_init
        if zero_raw_increment_init:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with FiLM conditioning.

        Args:
            x: [B, C, H, W] input tensor
            z: [B, prompt_dim] prompt vector. If None, FiLM acts as identity
               (gamma=1, beta=0), equivalent to SmallResUNet forward pass.

        Returns:
            [B, out_channels, H, W] predicted increments
        """
        # Encoder
        e1 = self.enc1(x)
        if z is not None:
            e1 = self.film1(e1, z)

        e2 = self.enc2(F.avg_pool2d(e1, 2))
        if z is not None:
            e2 = self.film2(e2, z)

        e3 = self.enc3(F.avg_pool2d(e2, 2))
        if z is not None:
            e3 = self.film3(e3, z)

        # Bottleneck
        b = self.bottleneck(e3)
        if z is not None:
            b = self.film_b(b, z)

        # Decoder
        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)
