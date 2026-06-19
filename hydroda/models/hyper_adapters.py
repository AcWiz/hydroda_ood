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
        adapter_param_style: str = "basis_1x1",
    ) -> None:
        super().__init__()
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")
        if adapter_bottleneck is None:
            adapter_bottleneck = max(8, channels // 4)
        if adapter_bottleneck < 1:
            raise ValueError("adapter_bottleneck must be >= 1")
        if adapter_param_style not in {"basis_1x1", "dora_like_gain", "dora_like_gain_bounded"}:
            raise ValueError(
                "adapter_param_style must be 'basis_1x1', 'dora_like_gain', "
                "or 'dora_like_gain_bounded'"
            )

        self.channels = int(channels)
        self.prompt_dim = int(prompt_dim)
        self.n_basis = int(n_basis)
        self.adapter_bottleneck = int(adapter_bottleneck)
        self.adapter_scale = float(adapter_scale)
        self.adapter_param_style = str(adapter_param_style)

        self.coeff_head = nn.Linear(prompt_dim, n_basis)
        self.bases = nn.ModuleList(
            [_AdapterBasis(channels, self.adapter_bottleneck) for _ in range(n_basis)]
        )
        self.basis_gain_delta = (
            nn.Parameter(torch.zeros(self.n_basis))
            if self.adapter_param_style in {"dora_like_gain", "dora_like_gain_bounded"}
            else None
        )

    def coefficients(
        self,
        z: torch.Tensor,
        logit_residual: torch.Tensor | None = None,
        coeff_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-sample basis coefficients with shape [B, n_basis]."""
        logits = self.coefficient_logits(z) if coeff_logits is None else coeff_logits
        if logits.shape != (z.shape[0], self.n_basis):
            raise ValueError("coeff_logits must have shape [B, n_basis]")
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

    def coefficient_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Return per-sample basis logits from this adapter's own coefficient head."""
        return self.coeff_head(z)

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        logit_residual: torch.Tensor | None = None,
        coeff_logits: torch.Tensor | None = None,
        residual_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the prompt-conditioned adapter residual to feature map ``h``."""
        coeffs = self.coefficients(
            z,
            logit_residual=logit_residual,
            coeff_logits=coeff_logits,
        )
        basis_outputs = torch.stack([basis(h) for basis in self.bases], dim=1)
        if self.basis_gain_delta is not None:
            gain_delta = self.basis_gain_delta.to(
                device=basis_outputs.device,
                dtype=basis_outputs.dtype,
            )
            if self.adapter_param_style == "dora_like_gain_bounded":
                gain = 1.0 + 0.25 * torch.tanh(gain_delta)
            else:
                gain = 1.0 + gain_delta
            basis_outputs = basis_outputs * gain.view(1, self.n_basis, 1, 1, 1)
        mixed = torch.einsum("bm,bmchw->bchw", coeffs, basis_outputs)
        if residual_gate is not None:
            if residual_gate.ndim == 1:
                residual_gate = residual_gate.view(-1, 1)
            if residual_gate.shape != (h.shape[0], 1):
                raise ValueError("residual_gate must have shape [B] or [B, 1]")
            mixed = mixed * residual_gate.to(dtype=mixed.dtype, device=mixed.device).view(-1, 1, 1, 1)
        return h + self.adapter_scale * mixed

    def coefficient_entropy(
        self,
        z: torch.Tensor,
        logit_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-sample entropy of the adapter coefficient distribution."""
        coeffs = self.coefficients(z, logit_residual=logit_residual).clamp_min(1e-8)
        return -(coeffs * coeffs.log()).sum(dim=-1)
