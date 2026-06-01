"""Target-domain adaptation modules for HyperDA.

These modules are intentionally small and auditable. They support the V4.3
protocol where source-trained priors are frozen and only target-specific
lightweight variables are optimized on target_train.
"""
from __future__ import annotations

import torch
from torch import nn


class TargetLatentPrompt(nn.Module):
    """Add a trainable target latent shift to a prompt vector."""

    def __init__(self, prompt_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        if prompt_dim < 1:
            raise ValueError("prompt_dim must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        self.prompt_dim = int(prompt_dim)
        self.latent_dim = int(latent_dim)
        self.latent = nn.Parameter(torch.zeros(latent_dim))
        self.proj = nn.Linear(latent_dim, prompt_dim)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.prompt_dim:
            raise ValueError(f"expected prompt dim {self.prompt_dim}, got {z.shape[-1]}")
        shift = self.proj(self.latent).view(*([1] * (z.ndim - 1)), self.prompt_dim)
        return z + shift


class AdapterCoefficientResidual(nn.Module):
    """Trainable residual added to adapter basis logits."""

    def __init__(self, n_basis: int) -> None:
        super().__init__()
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")
        self.n_basis = int(n_basis)
        self.logit_delta = nn.Parameter(torch.zeros(n_basis))

    def forward(self) -> torch.Tensor:
        return self.logit_delta


class MonthlyResidualGain(nn.Module):
    """Per-month output gain and bias initialized as identity."""

    def __init__(self, out_channels: int, n_months: int = 12) -> None:
        super().__init__()
        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        if n_months < 1:
            raise ValueError("n_months must be >= 1")
        self.out_channels = int(out_channels)
        self.n_months = int(n_months)
        self.gain_delta = nn.Parameter(torch.zeros(n_months, out_channels))
        self.bias = nn.Parameter(torch.zeros(n_months, out_channels))

    def forward(self, y: torch.Tensor, month: torch.Tensor) -> torch.Tensor:
        if y.ndim != 4:
            raise ValueError("MonthlyResidualGain expects y with shape [B, C, H, W]")
        if y.shape[1] != self.out_channels:
            raise ValueError(f"expected {self.out_channels} channels, got {y.shape[1]}")
        if month.ndim != 1 or month.shape[0] != y.shape[0]:
            raise ValueError("month must have shape [B]")
        if torch.any((month < 1) | (month > self.n_months)):
            raise ValueError(f"month values must be in [1, {self.n_months}]")

        month_idx = month.to(device=y.device, dtype=torch.long) - 1
        gain = 1.0 + self.gain_delta.to(y.device)[month_idx]
        bias = self.bias.to(y.device)[month_idx]
        return y * gain[:, :, None, None] + bias[:, :, None, None]
