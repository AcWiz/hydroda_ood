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


class TargetSpatialResidualHead(nn.Module):
    """Small target-domain residual head for spatially local increment correction.

    The final convolution is zero-initialized so enabling the module preserves
    the source prior at step zero. By default only the surface channel is
    refined; RootZone can be enabled explicitly for ablations.
    """

    def __init__(
        self,
        input_channels: int,
        out_channels: int,
        hidden_channels: int = 16,
        refine_rootzone: bool = False,
    ) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be >= 1")
        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1")
        self.input_channels = int(input_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.refine_rootzone = bool(refine_rootzone)
        self.conv = nn.Conv2d(self.input_channels + self.out_channels, self.hidden_channels, 3, padding=1)
        self.act = nn.GELU()
        self.final = nn.Conv2d(self.hidden_channels, self.out_channels, 3, padding=1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

        mask = torch.ones(self.out_channels, dtype=torch.float32)
        if self.out_channels >= 2 and not self.refine_rootzone:
            mask[1] = 0.0
        self.register_buffer("output_mask", mask.view(1, self.out_channels, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_raw: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError("TargetSpatialResidualHead expects x and y with shape [B, C, H, W]")
        if x.shape[0] != y.shape[0] or x.shape[-2:] != y.shape[-2:]:
            raise ValueError("x and y must have matching batch and spatial dimensions")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"expected {self.input_channels} input channels, got {x.shape[1]}")
        if y.shape[1] != self.out_channels:
            raise ValueError(f"expected {self.out_channels} output channels, got {y.shape[1]}")
        residual = self.final(self.act(self.conv(torch.cat([x, y], dim=1))))
        return residual * self.output_mask.to(device=y.device, dtype=y.dtype)


def _da_regime_features(x_raw: torch.Tensor) -> torch.Tensor:
    """Build DA evidence features without using them as hard masks."""
    if x_raw.ndim != 4 or x_raw.shape[1] < 12:
        raise ValueError("DA regime features expect x_raw with shape [B, >=12, H, W]")
    err_h = x_raw[:, 7:8].abs().clamp_min(1e-3)
    err_v = x_raw[:, 8:9].abs().clamp_min(1e-3)
    innov_h = ((x_raw[:, 5:6] - x_raw[:, 9:10]) / err_h).clamp(-10.0, 10.0)
    innov_v = ((x_raw[:, 6:7] - x_raw[:, 10:11]) / err_v).clamp(-10.0, 10.0)
    err_h_feat = torch.log1p(err_h).clamp(0.0, 10.0) / 10.0
    err_v_feat = torch.log1p(err_v).clamp(0.0, 10.0) / 10.0
    veg_opacity = x_raw[:, 4:5].clamp(-5.0, 5.0)
    soil_temp = ((x_raw[:, 2:3] - 273.15) / 20.0).clamp(-10.0, 10.0)
    surface_temp = ((x_raw[:, 3:4] - 273.15) / 20.0).clamp(-10.0, 10.0)
    base_valid = x_raw[:, 11:12].clamp(0.0, 1.0)
    features = torch.cat(
        [
            innov_h / 10.0,
            innov_v / 10.0,
            err_h_feat,
            err_v_feat,
            veg_opacity,
            soil_temp,
            surface_temp,
            base_valid,
        ],
        dim=1,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class DARegimeGainMixer(nn.Module):
    """Soft DA-evidence gain for residual updates.

    The output is a per-pixel, per-channel ``alpha`` in ``[0, 2]``. A zero
    final layer makes ``alpha=1`` at initialization, so wrapping an existing
    residual candidate preserves its initial behavior.
    """

    feature_channels = 8
    month_channels = 2

    def __init__(self, out_channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1")
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        in_channels = self.out_channels * 2 + self.feature_channels + self.month_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.out_channels, 1),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def month_encoding(
        month: torch.Tensor,
        spatial_shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if month.ndim != 1:
            raise ValueError("month must have shape [B]")
        if torch.any((month < 1) | (month > 12)):
            raise ValueError("month values must be in [1, 12]")
        angle = (month.to(device=device, dtype=dtype) - 1.0) * (2.0 * torch.pi / 12.0)
        enc = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)
        return enc[:, :, None, None].expand(month.shape[0], 2, spatial_shape[0], spatial_shape[1])

    def regime_features(self, x_raw: torch.Tensor) -> torch.Tensor:
        return _da_regime_features(x_raw)

    def forward(
        self,
        candidate_residual: torch.Tensor,
        y: torch.Tensor,
        x_raw: torch.Tensor,
        month: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_residual.ndim != 4 or y.ndim != 4:
            raise ValueError("DARegimeGainMixer expects residual and y with shape [B, C, H, W]")
        if candidate_residual.shape != y.shape:
            raise ValueError("candidate_residual and y must have identical shape")
        if candidate_residual.shape[1] != self.out_channels:
            raise ValueError(f"expected {self.out_channels} channels, got {candidate_residual.shape[1]}")
        if month.ndim != 1 or month.shape[0] != candidate_residual.shape[0]:
            raise ValueError("month must have shape [B]")
        features = self.regime_features(x_raw).to(device=candidate_residual.device, dtype=candidate_residual.dtype)
        month_features = self.month_encoding(
            month,
            spatial_shape=candidate_residual.shape[-2:],
            device=candidate_residual.device,
            dtype=candidate_residual.dtype,
        )
        h = torch.cat([candidate_residual, y, features, month_features], dim=1)
        return 2.0 * torch.sigmoid(self.net(h))


class BoundedDARegimeGainMixer(nn.Module):
    """Identity-centered, bounded DA-evidence gain for residual updates.

    The output is ``alpha = 1 + gain_span * tanh(raw_alpha)``. A zero final
    layer therefore initializes exactly to ``alpha=1`` while limiting learned
    departures from the Hydro-MSR residual candidate.
    """

    feature_channels = DARegimeGainMixer.feature_channels
    month_channels = DARegimeGainMixer.month_channels

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int = 16,
        gain_span: float = 0.25,
        learn_rootzone_gain: bool = False,
    ) -> None:
        super().__init__()
        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1")
        if gain_span < 0:
            raise ValueError("gain_span must be non-negative")
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.gain_span = float(gain_span)
        self.learn_rootzone_gain = bool(learn_rootzone_gain)
        in_channels = self.out_channels * 2 + self.feature_channels + self.month_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.out_channels, 1),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        mask = torch.ones(self.out_channels, dtype=torch.float32)
        if self.out_channels >= 2 and not self.learn_rootzone_gain:
            mask[1] = 0.0
        self.register_buffer("alpha_delta_mask", mask.view(1, self.out_channels, 1, 1))

    @staticmethod
    def month_encoding(
        month: torch.Tensor,
        spatial_shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return DARegimeGainMixer.month_encoding(month, spatial_shape, device, dtype)

    def regime_features(self, x_raw: torch.Tensor) -> torch.Tensor:
        return _da_regime_features(x_raw)

    def forward(
        self,
        candidate_residual: torch.Tensor,
        y: torch.Tensor,
        x_raw: torch.Tensor,
        month: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_residual.ndim != 4 or y.ndim != 4:
            raise ValueError("BoundedDARegimeGainMixer expects residual and y with shape [B, C, H, W]")
        if candidate_residual.shape != y.shape:
            raise ValueError("candidate_residual and y must have identical shape")
        if candidate_residual.shape[1] != self.out_channels:
            raise ValueError(f"expected {self.out_channels} channels, got {candidate_residual.shape[1]}")
        if month.ndim != 1 or month.shape[0] != candidate_residual.shape[0]:
            raise ValueError("month must have shape [B]")
        features = self.regime_features(x_raw).to(device=candidate_residual.device, dtype=candidate_residual.dtype)
        month_features = self.month_encoding(
            month,
            spatial_shape=candidate_residual.shape[-2:],
            device=candidate_residual.device,
            dtype=candidate_residual.dtype,
        )
        h = torch.cat([candidate_residual, y, features, month_features], dim=1)
        raw_alpha = self.net(h)
        alpha_delta = torch.tanh(raw_alpha) * self.alpha_delta_mask.to(device=raw_alpha.device, dtype=raw_alpha.dtype)
        return 1.0 + self.gain_span * alpha_delta


def _normalization_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _DAQualityFiLM(nn.Module):
    """Identity-initialized FiLM from DA-side quality features.

    The features use DA diagnostics as context only. They are not used as hard
    output gates, loss masks, or metric masks.
    """

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(7, self.hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels * 2, 1),
        )
        final = self.conv[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def _quality_features(x_raw: torch.Tensor) -> torch.Tensor:
        err_h = x_raw[:, 7:8].abs().clamp_min(1e-3)
        err_v = x_raw[:, 8:9].abs().clamp_min(1e-3)
        innov_h = ((x_raw[:, 5:6] - x_raw[:, 9:10]) / err_h).clamp(-10.0, 10.0)
        innov_v = ((x_raw[:, 6:7] - x_raw[:, 10:11]) / err_v).clamp(-10.0, 10.0)
        innov_mag = torch.sqrt(innov_h.square() + innov_v.square()).clamp(0.0, 10.0)
        veg_opacity = x_raw[:, 4:5].clamp(-5.0, 5.0)
        soil_temp = ((x_raw[:, 2:3] - 273.15) / 20.0).clamp(-10.0, 10.0)
        surface_temp = ((x_raw[:, 3:4] - 273.15) / 20.0).clamp(-10.0, 10.0)
        base_valid = x_raw[:, 11:12].clamp(0.0, 1.0)
        features = torch.cat(
            [
                innov_h / 10.0,
                innov_v / 10.0,
                innov_mag / 10.0,
                veg_opacity,
                soil_temp,
                surface_temp,
                base_valid,
            ],
            dim=1,
        )
        return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, x_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x_raw.ndim != 4 or x_raw.shape[1] < 12:
            raise ValueError("_DAQualityFiLM expects x_raw with shape [B, >=12, H, W]")
        gamma_beta = self.conv(self._quality_features(x_raw))
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return gamma, beta


class RobustObservationSpaceEncoder(nn.Module):
    """Bounded DA-aware observation-space features for target refinement.

    The encoder reads input-side fields only. It summarizes robust innovations,
    polarization contrast, observation error confidence, forecast state, thermal
    state, vegetation opacity, and the diagnostic base-valid channel. It does
    not read analysis, increment labels, model errors, or split-level target
    statistics.
    """

    feature_channels = 14

    def __init__(self) -> None:
        super().__init__()
        self.feature_scale = nn.Parameter(torch.ones(1, self.feature_channels, 1, 1))
        self.feature_bias = nn.Parameter(torch.zeros(1, self.feature_channels, 1, 1))

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        if x_raw.ndim != 4 or x_raw.shape[1] < 12:
            raise ValueError("RobustObservationSpaceEncoder expects x_raw with shape [B, >=12, H, W]")
        x_safe = torch.nan_to_num(x_raw.float(), nan=0.0, posinf=0.0, neginf=0.0)

        sm_surface = ((x_safe[:, 0:1] - 0.25) / 0.25).clamp(-1.0, 1.0)
        sm_rootzone = ((x_safe[:, 1:2] - 0.30) / 0.30).clamp(-1.0, 1.0)
        soil_temp = ((x_safe[:, 2:3] - 273.15) / 40.0).clamp(-1.0, 1.0)
        surface_temp = ((x_safe[:, 3:4] - 273.15) / 40.0).clamp(-1.0, 1.0)
        veg_opacity = torch.tanh(x_safe[:, 4:5] / 2.0)

        err_h = x_safe[:, 7:8].abs().clamp_min(1e-3)
        err_v = x_safe[:, 8:9].abs().clamp_min(1e-3)
        innov_h = ((x_safe[:, 5:6] - x_safe[:, 9:10]) / err_h).clamp(-10.0, 10.0)
        innov_v = ((x_safe[:, 6:7] - x_safe[:, 10:11]) / err_v).clamp(-10.0, 10.0)
        innov_mag = torch.sqrt(innov_h.square() + innov_v.square()).clamp(0.0, 10.0)

        obs_pol = ((x_safe[:, 5:6] - x_safe[:, 6:7]) / 100.0).clamp(-1.0, 1.0)
        assim_pol = ((x_safe[:, 9:10] - x_safe[:, 10:11]) / 100.0).clamp(-1.0, 1.0)
        pol_err = (err_h + err_v).clamp_min(1e-3)
        pol_innov = (((x_safe[:, 5:6] - x_safe[:, 6:7]) - (x_safe[:, 9:10] - x_safe[:, 10:11])) / pol_err).clamp(
            -10.0,
            10.0,
        )

        err_conf_h = (1.0 / (1.0 + err_h)).clamp(0.0, 1.0)
        err_conf_v = (1.0 / (1.0 + err_v)).clamp(0.0, 1.0)
        base_valid = x_safe[:, 11:12].clamp(0.0, 1.0) * 2.0 - 1.0

        base = torch.cat(
            [
                innov_h / 10.0,
                innov_v / 10.0,
                innov_mag / 10.0,
                obs_pol,
                assim_pol,
                pol_innov / 10.0,
                err_conf_h,
                err_conf_v,
                sm_surface,
                sm_rootzone,
                veg_opacity,
                soil_temp,
                surface_temp,
                base_valid,
            ],
            dim=1,
        )
        base = torch.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        return torch.tanh(base * self.feature_scale.to(device=base.device, dtype=base.dtype) + self.feature_bias.to(device=base.device, dtype=base.dtype))


class HydroMSROutputAdapter(nn.Module):
    """Multi-scale target-domain residual adapter for Surface/RootZone outputs.

    The adapter is zero-initialized at the output heads so enabling it preserves
    the current source-prior prediction before target adaptation updates.
    """

    def __init__(
        self,
        input_channels: int,
        out_channels: int,
        hidden_channels: int = 16,
        refine_rootzone: bool = True,
        enable_da_film: bool = False,
    ) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be >= 1")
        if out_channels != 2:
            raise ValueError("HydroMSROutputAdapter currently expects out_channels=2")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1")
        self.input_channels = int(input_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.refine_rootzone = bool(refine_rootzone)
        self.enable_da_film = bool(enable_da_film)

        self.input_proj = nn.Conv2d(self.input_channels + self.out_channels, self.hidden_channels, 1)
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    self.hidden_channels,
                    self.hidden_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    groups=self.hidden_channels,
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.mix = nn.Conv2d(self.hidden_channels * len(self.branches), self.hidden_channels, 1)
        self.norm = nn.GroupNorm(_normalization_groups(self.hidden_channels), self.hidden_channels)
        self.act = nn.GELU()
        self.da_film = _DAQualityFiLM(self.hidden_channels) if self.enable_da_film else None

        self.surface_head = nn.Conv2d(self.hidden_channels, 1, 1)
        self.rootzone_head = nn.Conv2d(self.hidden_channels, 1, 1)
        nn.init.zeros_(self.surface_head.weight)
        nn.init.zeros_(self.surface_head.bias)
        nn.init.zeros_(self.rootzone_head.weight)
        nn.init.zeros_(self.rootzone_head.bias)

        self.surface_to_rootzone = nn.Conv2d(1, 1, 5, padding=2, bias=False)
        nn.init.constant_(self.surface_to_rootzone.weight, 1.0 / 25.0)
        self.surface_to_rootzone_scale = nn.Parameter(torch.zeros(1))

        mask = torch.ones(self.out_channels, dtype=torch.float32)
        if not self.refine_rootzone:
            mask[1] = 0.0
        self.register_buffer("output_mask", mask.view(1, self.out_channels, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_raw: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError("HydroMSROutputAdapter expects x and y with shape [B, C, H, W]")
        if x.shape[0] != y.shape[0] or x.shape[-2:] != y.shape[-2:]:
            raise ValueError("x and y must have matching batch and spatial dimensions")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"expected {self.input_channels} input channels, got {x.shape[1]}")
        if y.shape[1] != self.out_channels:
            raise ValueError(f"expected {self.out_channels} output channels, got {y.shape[1]}")

        h = self.input_proj(torch.cat([x, y], dim=1))
        h = torch.cat([branch(h) for branch in self.branches], dim=1)
        h = self.act(self.norm(self.mix(h)))
        if self.da_film is not None:
            film_x = x_raw if x_raw is not None else x
            gamma, beta = self.da_film(film_x)
            h = h * (1.0 + gamma.to(dtype=h.dtype)) + beta.to(dtype=h.dtype)

        surface = self.surface_head(h)
        rootzone = self.rootzone_head(h)
        rootzone = rootzone + self.surface_to_rootzone_scale.view(1, 1, 1, 1) * self.surface_to_rootzone(surface)
        residual = torch.cat([surface, rootzone], dim=1)
        return residual * self.output_mask.to(device=y.device, dtype=y.dtype)


class HydroMSRROSEOutputAdapter(nn.Module):
    """Hydro-MSR residual adapter with robust observation-space encoding."""

    def __init__(
        self,
        input_channels: int,
        out_channels: int,
        hidden_channels: int = 16,
        refine_rootzone: bool = True,
        enable_da_film: bool = False,
    ) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be >= 1")
        if out_channels != 2:
            raise ValueError("HydroMSRROSEOutputAdapter currently expects out_channels=2")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1")
        self.input_channels = int(input_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.refine_rootzone = bool(refine_rootzone)
        self.enable_da_film = bool(enable_da_film)
        self.rose_encoder = RobustObservationSpaceEncoder()

        rose_channels = self.rose_encoder.feature_channels
        self.input_proj = nn.Conv2d(self.input_channels + self.out_channels + rose_channels, self.hidden_channels, 1)
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    self.hidden_channels,
                    self.hidden_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    groups=self.hidden_channels,
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.mix = nn.Conv2d(self.hidden_channels * len(self.branches), self.hidden_channels, 1)
        self.norm = nn.GroupNorm(_normalization_groups(self.hidden_channels), self.hidden_channels)
        self.act = nn.GELU()
        self.da_film = _DAQualityFiLM(self.hidden_channels) if self.enable_da_film else None

        self.surface_head = nn.Conv2d(self.hidden_channels, 1, 1)
        self.rootzone_head = nn.Conv2d(self.hidden_channels, 1, 1)
        nn.init.zeros_(self.surface_head.weight)
        nn.init.zeros_(self.surface_head.bias)
        nn.init.zeros_(self.rootzone_head.weight)
        nn.init.zeros_(self.rootzone_head.bias)

        self.surface_to_rootzone = nn.Conv2d(1, 1, 5, padding=2, bias=False)
        nn.init.constant_(self.surface_to_rootzone.weight, 1.0 / 25.0)
        self.surface_to_rootzone_scale = nn.Parameter(torch.zeros(1))

        mask = torch.ones(self.out_channels, dtype=torch.float32)
        if not self.refine_rootzone:
            mask[1] = 0.0
        self.register_buffer("output_mask", mask.view(1, self.out_channels, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_raw: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError("HydroMSRROSEOutputAdapter expects x and y with shape [B, C, H, W]")
        if x.shape[0] != y.shape[0] or x.shape[-2:] != y.shape[-2:]:
            raise ValueError("x and y must have matching batch and spatial dimensions")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"expected {self.input_channels} input channels, got {x.shape[1]}")
        if y.shape[1] != self.out_channels:
            raise ValueError(f"expected {self.out_channels} output channels, got {y.shape[1]}")

        rose_x = x_raw if x_raw is not None else x
        rose = self.rose_encoder(rose_x).to(device=x.device, dtype=x.dtype)
        h = self.input_proj(torch.cat([x, y, rose], dim=1))
        h = torch.cat([branch(h) for branch in self.branches], dim=1)
        h = self.act(self.norm(self.mix(h)))
        if self.da_film is not None:
            film_x = x_raw if x_raw is not None else x
            gamma, beta = self.da_film(film_x)
            h = h * (1.0 + gamma.to(dtype=h.dtype)) + beta.to(dtype=h.dtype)

        surface = self.surface_head(h)
        rootzone = self.rootzone_head(h)
        rootzone = rootzone + self.surface_to_rootzone_scale.view(1, 1, 1, 1) * self.surface_to_rootzone(surface)
        residual = torch.cat([surface, rootzone], dim=1)
        return residual * self.output_mask.to(device=y.device, dtype=y.dtype)


class HydroMSRGainOutputAdapter(nn.Module):
    """Hydro-MSR candidate residual with a DA-regime soft gain adapter."""

    def __init__(
        self,
        input_channels: int,
        out_channels: int,
        hidden_channels: int = 16,
        refine_rootzone: bool = True,
        enable_da_film: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.refine_rootzone = bool(refine_rootzone)
        self.enable_da_film = bool(enable_da_film)
        self.candidate = HydroMSROutputAdapter(
            input_channels=input_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            refine_rootzone=refine_rootzone,
            enable_da_film=enable_da_film,
        )
        self.gain_mixer = DARegimeGainMixer(out_channels=out_channels, hidden_channels=hidden_channels)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_raw: torch.Tensor | None = None,
        month: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if month is None:
            raise ValueError("HydroMSRGainOutputAdapter requires month tensor")
        regime_x = x_raw if x_raw is not None else x
        candidate = self.candidate(x, y, x_raw=x_raw)
        alpha = self.gain_mixer(candidate, y, regime_x, month)
        return alpha.to(dtype=candidate.dtype) * candidate


class HydroMSRGainLiteOutputAdapter(nn.Module):
    """Hydro-MSR residual with a small bounded surface gain adjustment."""

    def __init__(
        self,
        input_channels: int,
        out_channels: int,
        hidden_channels: int = 16,
        refine_rootzone: bool = True,
        enable_da_film: bool = False,
        gain_span: float = 0.25,
        learn_rootzone_gain: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.refine_rootzone = bool(refine_rootzone)
        self.enable_da_film = bool(enable_da_film)
        self.gain_span = float(gain_span)
        self.learn_rootzone_gain = bool(learn_rootzone_gain)
        self.candidate = HydroMSROutputAdapter(
            input_channels=input_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            refine_rootzone=refine_rootzone,
            enable_da_film=enable_da_film,
        )
        self.gain_mixer = BoundedDARegimeGainMixer(
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            gain_span=gain_span,
            learn_rootzone_gain=learn_rootzone_gain,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_raw: torch.Tensor | None = None,
        month: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if month is None:
            raise ValueError("HydroMSRGainLiteOutputAdapter requires month tensor")
        regime_x = x_raw if x_raw is not None else x
        candidate = self.candidate(x, y, x_raw=x_raw)
        alpha = self.gain_mixer(candidate, y, regime_x, month)
        return alpha.to(dtype=candidate.dtype) * candidate
