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
        safe_x = torch.where(valid_mask, x_flat, torch.zeros_like(x_flat))
        # Compute mean with valid masking
        valid_count = valid_mask.float().sum(dim=-1).clamp(min=1.0)  # [B, C]
        mean = safe_x.sum(dim=-1) / valid_count  # [B, C]
        # Compute std
        diff = torch.where(valid_mask, x_flat - mean.unsqueeze(-1), torch.zeros_like(x_flat))
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


class RobustInputSideDAPromptEncoder(RegionPromptEncoder):
    """Prompt encoder using DA-aware robust input-side diagnostics only.

    This encoder preserves ``RegionPromptEncoder.forward(x, region_ids, month)``
    and replaces the input summary branch with finite-value diagnostics derived
    from ``x``. Channel 11 is included only as a bounded diagnostic coverage
    feature; it is never used to mask, select, weight, or drop other channels.
    """

    diagnostic_schema = [
        "sm_surface_forecast_median",
        "sm_surface_forecast_iqr",
        "sm_rootzone_forecast_median",
        "sm_rootzone_forecast_iqr",
        "soil_temp_layer1_forecast_median",
        "soil_temp_layer1_forecast_iqr",
        "surface_temp_forecast_median",
        "surface_temp_forecast_iqr",
        "mwrtm_vegopacity_median",
        "mwrtm_vegopacity_iqr",
        "tb_h_innovation_median",
        "tb_h_innovation_iqr",
        "tb_v_innovation_median",
        "tb_v_innovation_iqr",
        "tb_obs_hv_contrast_median",
        "tb_obs_hv_contrast_iqr",
        "tb_assim_hv_contrast_median",
        "tb_assim_hv_contrast_iqr",
        "tb_h_obs_error_confidence",
        "tb_v_obs_error_confidence",
        "tb_h_innovation_normalized_abs_median",
        "tb_v_innovation_normalized_abs_median",
        "finite_input_coverage",
        "base_valid_mask_fraction_diagnostic_only",
    ]

    def __init__(
        self,
        num_regions: int = 6,
        input_channels: int = 12,
        hidden_dim: int = 64,
    ) -> None:
        if int(input_channels) != 12:
            raise ValueError(
                "RobustInputSideDAPromptEncoder requires the audited 12-channel "
                "HydroDA input contract"
            )
        super().__init__(
            num_regions=num_regions,
            input_channels=input_channels,
            hidden_dim=hidden_dim,
        )

    @staticmethod
    def _finite_values(values: torch.Tensor) -> torch.Tensor:
        return values[torch.isfinite(values)].float()

    @classmethod
    def _median_iqr(cls, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        finite = cls._finite_values(values)
        if finite.numel() == 0:
            zero = values.new_tensor(0.0)
            return zero, zero
        quantiles = torch.quantile(finite, finite.new_tensor([0.25, 0.5, 0.75]))
        median = quantiles[1].to(dtype=values.dtype, device=values.device)
        iqr = (quantiles[2] - quantiles[0]).to(dtype=values.dtype, device=values.device)
        return median, iqr

    @classmethod
    def _median(cls, values: torch.Tensor) -> torch.Tensor:
        finite = cls._finite_values(values)
        if finite.numel() == 0:
            return values.new_tensor(0.0)
        return torch.quantile(finite, finite.new_tensor(0.5)).to(
            dtype=values.dtype,
            device=values.device,
        )

    @classmethod
    def _confidence_from_error_std(cls, values: torch.Tensor) -> torch.Tensor:
        finite = cls._finite_values(values.abs())
        if finite.numel() == 0:
            return values.new_tensor(0.0)
        median_abs_err = torch.quantile(finite, finite.new_tensor(0.5)).to(
            dtype=values.dtype,
            device=values.device,
        )
        return 1.0 / (1.0 + median_abs_err.clamp_min(0.0))

    @staticmethod
    def _finite_coverage(values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_tensor(0.0)
        return torch.isfinite(values).float().mean().to(dtype=values.dtype, device=values.device)

    @classmethod
    def _base_valid_fraction(cls, values: torch.Tensor) -> torch.Tensor:
        finite = cls._finite_values(values)
        if finite.numel() == 0:
            return values.new_tensor(0.0)
        return (finite > 0.5).float().mean().to(dtype=values.dtype, device=values.device)

    def _compute_input_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Compute DA-aware robust diagnostics from finite input-side fields.

        The returned shape remains ``[B, input_channels * 2]`` for checkpoint
        compatibility with the existing projection layer. Diagnostics are based
        only on the current input tensor and the audited HydroDA channel order:
        forecasts, temperatures, vegetation opacity, H/V brightness-temperature
        observations, observation error std, assimilated TB proxies, finite
        coverage, and bounded channel-11 coverage.
        """
        batch_size, channels = x.shape[0], x.shape[1]
        if channels != 12:
            raise ValueError(
                "RobustInputSideDAPromptEncoder expects 12 input channels, "
                f"got {channels}"
            )
        x_flat = x.reshape(batch_size, channels, -1)
        rows = []

        for sample_idx in range(batch_size):
            sample = x_flat[sample_idx]
            features = []

            for channel_idx in (0, 1, 2, 3, 4):
                median, iqr = self._median_iqr(sample[channel_idx])
                features.extend([median, iqr])

            tb_h_innovation = sample[5] - sample[9]
            tb_v_innovation = sample[6] - sample[10]
            tb_obs_hv_contrast = sample[6] - sample[5]
            tb_assim_hv_contrast = sample[10] - sample[9]

            for diagnostic in (
                tb_h_innovation,
                tb_v_innovation,
                tb_obs_hv_contrast,
                tb_assim_hv_contrast,
            ):
                median, iqr = self._median_iqr(diagnostic)
                features.extend([median, iqr])

            h_error_conf = self._confidence_from_error_std(sample[7])
            v_error_conf = self._confidence_from_error_std(sample[8])
            h_norm_abs_innov = self._median(tb_h_innovation.abs() / (1.0 + sample[7].abs().clamp_min(0.0)))
            v_norm_abs_innov = self._median(tb_v_innovation.abs() / (1.0 + sample[8].abs().clamp_min(0.0)))
            finite_coverage = self._finite_coverage(sample)
            base_valid_fraction = self._base_valid_fraction(sample[11])

            features.extend(
                [
                    h_error_conf,
                    v_error_conf,
                    h_norm_abs_innov,
                    v_norm_abs_innov,
                    finite_coverage,
                    base_valid_fraction,
                ]
            )
            rows.append(torch.stack(features))

        diagnostics = torch.stack(rows, dim=0)
        if diagnostics.shape[1] != len(self.diagnostic_schema):
            raise RuntimeError(
                "DA diagnostic feature count mismatch: "
                f"got {diagnostics.shape[1]}, expected {len(self.diagnostic_schema)}"
            )
        return torch.nan_to_num(diagnostics, nan=0.0, posinf=0.0, neginf=0.0)
