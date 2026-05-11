"""Basis-factorized parameter generation for HyperDA."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class ParameterBasis(nn.Module):
    """Compose lightweight parameters from learned bases.

    This module stores basis tensors per parameter block. ``coeffs`` should map
    each block name to shape [B, n_basis].
    """

    def __init__(self, block_shapes: Dict[str, tuple[int, ...]], n_basis: int = 8) -> None:
        super().__init__()
        self.n_basis = int(n_basis)
        self.block_names = list(block_shapes.keys())
        self.base = nn.ParameterDict()
        self.bases = nn.ParameterDict()
        for name, shape in block_shapes.items():
            safe = name.replace(".", "__")
            self.base[safe] = nn.Parameter(torch.zeros(*shape))
            self.bases[safe] = nn.Parameter(torch.randn(n_basis, *shape) * 0.01)

    def compose(self, coeffs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        zeta: Dict[str, torch.Tensor] = {}
        for name in self.block_names:
            safe = name.replace(".", "__")
            alpha = coeffs[name]  # [B, M]
            basis = self.bases[safe]  # [M, ...]
            base = self.base[safe]  # [...]
            delta = torch.einsum("bm,m...->b...", alpha, basis)
            zeta[name] = base.unsqueeze(0) + delta
        return zeta
