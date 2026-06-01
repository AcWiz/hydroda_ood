"""Basis-generated adapter blocks for HyperDA."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _AdapterBasis(nn.Module):
    """Single lightweight 1x1 bottleneck adapter basis."""

    def __init__(self, channels: int, adapter_bottleneck: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(channels, adapter_bottleneck, kernel_size=1)
        self.up = nn.Conv2d(adapter_bottleneck, channels, kernel_size=1)

        nn.init.normal_(self.up.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(h)))


class BasisHyperAdapter(nn.Module):
    """Prompt-conditioned mixture of learned adapter bases.

    The prompt does not directly generate full convolution tensors. It predicts
    simplex coefficients over a small basis bank, yielding a stable first
    HyperDA implementation with clear parameter-generation semantics.
    """

    def __init__(
        self,
        channels: int,
        prompt_dim: int,
        n_basis: int = 8,
        adapter_bottleneck: int | None = None,
        adapter_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")
        if adapter_bottleneck is None:
            adapter_bottleneck = max(8, channels // 4)
        if adapter_bottleneck < 1:
            raise ValueError("adapter_bottleneck must be >= 1")

        self.channels = int(channels)
        self.prompt_dim = int(prompt_dim)
        self.n_basis = int(n_basis)
        self.adapter_bottleneck = int(adapter_bottleneck)
        self.adapter_scale = float(adapter_scale)

        self.coeff_head = nn.Linear(prompt_dim, n_basis)
        self.bases = nn.ModuleList(
            [_AdapterBasis(channels, self.adapter_bottleneck) for _ in range(n_basis)]
        )

    def coefficients(
        self,
        z: torch.Tensor,
        logit_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-sample basis coefficients with shape [B, n_basis]."""
        logits = self.coeff_head(z)
        if logit_residual is not None:
            if logit_residual.ndim == 1:
                if logit_residual.shape[0] != self.n_basis:
                    raise ValueError("logit_residual length must match n_basis")
                logits = logits + logit_residual.view(1, self.n_basis)
            elif logit_residual.shape == logits.shape:
                logits = logits + logit_residual
            else:
                raise ValueError("logit_residual must have shape [n_basis] or [B, n_basis]")
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        logit_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the prompt-conditioned adapter residual to feature map ``h``."""
        coeffs = self.coefficients(z, logit_residual=logit_residual)
        basis_outputs = torch.stack([basis(h) for basis in self.bases], dim=1)
        mixed = torch.einsum("bm,bmchw->bchw", coeffs, basis_outputs)
        return h + self.adapter_scale * mixed

    def coefficient_entropy(
        self,
        z: torch.Tensor,
        logit_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-sample entropy of the adapter coefficient distribution."""
        coeffs = self.coefficients(z, logit_residual=logit_residual).clamp_min(1e-8)
        return -(coeffs * coeffs.log()).sum(dim=-1)
