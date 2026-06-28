"""MixStyle regularization for source-side domain generalization."""
from __future__ import annotations

import torch
from torch import nn


class MixStyle2d(nn.Module):
    """Feature-statistic mixing layer for 2D feature maps.

    The layer is stochastic and active only in training mode. It has no
    trainable parameters and preserves the input tensor shape.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError("MixStyle alpha must be positive")
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        if x.ndim != 4:
            raise ValueError(f"MixStyle2d expects [B,C,H,W], got shape={tuple(x.shape)}")
        if torch.rand((), device=x.device).item() > self.p:
            return x

        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        std = (var + self.eps).sqrt()
        normalized = (x - mean) / std

        beta = torch.distributions.Beta(self.alpha, self.alpha)
        lam = beta.sample((x.size(0), 1, 1, 1)).to(device=x.device, dtype=x.dtype)
        perm = torch.randperm(x.size(0), device=x.device)
        mixed_mean = mean * lam + mean[perm] * (1.0 - lam)
        mixed_std = std * lam + std[perm] * (1.0 - lam)
        return normalized * mixed_std + mixed_mean
