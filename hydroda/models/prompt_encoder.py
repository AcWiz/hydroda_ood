"""Region prompt encoder for FiLM-conditioned neural DA increment operators.

No-leakage declaration:
    - Prompt uses input-side statistics only (no target query labels)
    - Region embedding learned from source regions only during training
    - Temporal encoding uses sin/cos of month from sample metadata
    - Target embedding initialized as mean of source embeddings, not learned from target data
"""
from __future__ import annotations

import torch
from torch import nn


class RegionPromptEncoder(nn.Module):
    """Encodes region identity + input statistics + temporal info into a prompt vector.

    Prompt tokens:
        - region_embedding (learned, dim=16 for N regions)
        - input_summary_stats (mean/std of input channels -> 2*C dims)
        - temporal_encoding (sin/cos of month -> 2 dims)

    All concatenated and passed through an MLP to produce prompt vector z (dim=hidden_dim).

    Args:
        num_regions: number of distinct region embeddings (default 6 for US-R1..R6)
        input_channels: number of input channels (default 12)
        hidden_dim: output prompt vector dimension (default 64)
    """

    def __init__(
        self,
        num_regions: int = 6,
        input_channels: int = 12,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_regions = num_regions
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim

        # Per-region learned embedding (region_id 0..num_regions-1)
        self.region_embed = nn.Embedding(num_regions, 16)

        # Input summary statistics projection: (mean, std) per channel -> 16 dims
        self.input_proj = nn.Linear(input_channels * 2, 16)

        # Temporal encoding projection: sin/cos month -> 8 dims
        self.temporal_proj = nn.Linear(2, 8)

        # MLP to combine all prompt tokens into a single vector z
        self.mlp = nn.Sequential(
            nn.Linear(16 + 16 + 8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _compute_input_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-channel mean and std from input tensor.

        Args:
            x: [B, C, H, W] input tensor

        Returns:
            [B, C*2] tensor of (mean, std) per channel
        """
        B, C = x.shape[0], x.shape[1]
        # Compute over spatial dims (H, W)
        x_flat = x.view(B, C, -1)  # [B, C, H*W]
        # Handle NaN/inf by masking
        valid_mask = torch.isfinite(x_flat)
        # Compute mean with valid masking
        valid_count = valid_mask.float().sum(dim=-1).clamp(min=1.0)  # [B, C]
        mean = (x_flat * valid_mask.float()).sum(dim=-1) / valid_count  # [B, C]
        # Compute std
        diff = (x_flat - mean.unsqueeze(-1)) * valid_mask.float()
        var = (diff ** 2).sum(dim=-1) / valid_count
        std = torch.sqrt(var.clamp(min=1e-8))  # [B, C]

        return torch.cat([mean, std], dim=1)  # [B, C*2]

    @staticmethod
    def _temporal_encoding(month: torch.Tensor) -> torch.Tensor:
        """Create sin/cos encoding of month.

        Args:
            month: [B] tensor of month integers (1-12)

        Returns:
            [B, 2] tensor of (sin, cos) encoding
        """
        # Normalize month to [0, 2*pi]
        theta = month.float() / 12.0 * 2.0 * torch.pi
        return torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        region_ids: torch.Tensor,
        month: torch.Tensor,
    ) -> torch.Tensor:
        """Encode prompt from input, region, and temporal info.

        Args:
            x: [B, C, H, W] input tensor
            region_ids: [B] long tensor of region indices (0..num_regions-1)
            month: [B] long tensor of month integers (1-12)

        Returns:
            [B, hidden_dim] prompt vector z
        """
        # Region embedding
        r_emb = self.region_embed(region_ids)  # [B, 16]

        # Input statistics
        input_stats = self._compute_input_stats(x)  # [B, C*2]
        i_emb = self.input_proj(input_stats)  # [B, 16]

        # Temporal encoding
        t_enc = self._temporal_encoding(month)  # [B, 2]
        t_emb = self.temporal_proj(t_enc)  # [B, 8]

        # Concatenate and project
        combined = torch.cat([r_emb, i_emb, t_emb], dim=1)  # [B, 40]
        z = self.mlp(combined)  # [B, hidden_dim]

        return z
