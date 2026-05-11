"""Lightweight generated adapter blocks for HyperDA."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class BottleneckAdapter(nn.Module):
    """Trainable adapter used by adapter tuning baselines."""

    def __init__(self, channels: int, bottleneck: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.up = nn.Conv2d(bottleneck, channels, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.down(self.norm(h))
        z = F.gelu(z)
        z = self.up(z)
        return h + z


def functional_adapter_forward(h: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Functional generated adapter forward.

    Expected params:
      down_w, down_b, up_w, up_b. Norm is intentionally outside this helper so
      the generated parameter schema stays simple and auditable.
    """
    z = F.conv2d(h, params["down_w"], params.get("down_b"))
    z = F.gelu(z)
    z = F.conv2d(z, params["up_w"], params.get("up_b"))
    return h + z
