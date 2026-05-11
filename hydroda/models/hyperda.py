"""Minimal HyperDA skeleton.

This file intentionally implements only lightweight parameter generation. It is
not a full training script. The goal is to make the architecture contract
executable and testable before large experiments.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from hydroda.models.parameter_basis import ParameterBasis


class SimplePromptEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, n_layers: int = 2) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.in_proj = nn.Linear(feature_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, prompt_tokens: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        B = prompt_tokens.shape[0]
        x = self.in_proj(prompt_tokens)
        pool = self.pool.expand(B, -1, -1)
        x = torch.cat([pool, x], dim=1)
        src_key_padding_mask = None
        if token_mask is not None:
            # token_mask: True for valid original tokens. Pool token is valid.
            valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=token_mask.device), token_mask], dim=1)
            src_key_padding_mask = ~valid
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory[:, 0]


class HyperDA(nn.Module):
    """Prompt-conditioned basis-factorized hypernetwork."""

    def __init__(
        self,
        prompt_feature_dim: int,
        block_shapes: Dict[str, tuple[int, ...]],
        hidden_dim: int = 128,
        n_basis: int = 8,
    ) -> None:
        super().__init__()
        self.prompt_encoder = SimplePromptEncoder(prompt_feature_dim, hidden_dim=hidden_dim)
        self.block_names = list(block_shapes.keys())
        self.coeff_heads = nn.ModuleDict(
            {name.replace(".", "__"): nn.Linear(hidden_dim, n_basis) for name in self.block_names}
        )
        self.parameter_basis = ParameterBasis(block_shapes, n_basis=n_basis)

    def forward(self, prompt_tokens: torch.Tensor, token_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        pooled = self.prompt_encoder(prompt_tokens, token_mask)
        coeffs: Dict[str, torch.Tensor] = {}
        for name in self.block_names:
            safe = name.replace(".", "__")
            coeffs[name] = self.coeff_heads[safe](pooled)
        return self.parameter_basis.compose(coeffs)
