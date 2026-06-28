"""Input-side PhysTrust-D0 diagnostics for HyperDA-TRUST reports.

These helpers intentionally do not participate in model selection or forward
prediction. They summarize raw HydroDA input-side DA fields for explanation:
brightness-temperature innovation, observation-error confidence, vegetation
opacity, forecast surface-rootzone contrast, finite coverage, and bounded
channel-11 diagnostic coverage.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch


PHYS_TRUST_D0_SCHEMA_VERSION = "phys_trust_d0_input_side_diagnostics_v1"
PHYS_TRUST_D0_DIAGNOSTIC_KEYS = (
    "tb_h_normalized_innovation_abs_median",
    "tb_v_normalized_innovation_abs_median",
    "tb_h_obs_error_confidence",
    "tb_v_obs_error_confidence",
    "vegopacity_median",
    "surface_rootzone_forecast_contrast_abs_median",
    "finite_input_coverage",
    "base_valid_mask_fraction_diagnostic_only",
)
PHYS_CONSISTENCY_GUARD_SCHEMA_VERSION = "phys_formula_consistency_guard_v1"
PHYS_CONSISTENCY_GUARD_MODE = "enkf_rt_vertical"
PHYS_CONSISTENCY_GUARD_PRODUCT_MODE = "surface_primary_enkf_rt_vertical_product"
PHYS_CONSISTENCY_GUARD_MODES = (
    PHYS_CONSISTENCY_GUARD_MODE,
    PHYS_CONSISTENCY_GUARD_PRODUCT_MODE,
)
PHYS_CONSISTENCY_SOURCE = "raw_input_side_formula"
PHYS_FORMULA_OPERATOR_SCHEMA_VERSION = "phys_formula_operator_v1"
PHYS_FORMULA_ENHANCED_OPERATOR_SCHEMA_VERSION = "phys_formula_operator_v2_enhanced_input_side"
PHYS_FORMULA_GAIN_OPERATOR_SCHEMA_VERSION = "m3_14_raw_input_side_formula_gain_v1"
PHYS_FORMULA_MODE = "enkf_rt_vertical_temp"
PHYS_FORMULA_SOURCE = "raw_input_side_formula_v2"
PHYS_FORMULA_ENHANCED_SOURCE = "raw_input_side_formula_v3_enhanced"
PHYS_FORMULA_GAIN_SOURCE = "raw_input_side_formula_gain"
PHYS_GAIN_BASIS_SCHEMA_VERSION = "phys_gain_basis_hypertrust_v1"
PHYS_GAIN_BASIS_BANK_SCHEMA_VERSION = "phys_gain_basis_source_gain_bank_v1"
PHYS_GAIN_BASIS_SOURCE = "raw_input_side_phys_gain_basis"
PHYS_GAIN_BASIS_NAMES = ("B_H", "B_V", "B_pol", "B_temp", "B_vert")
PHYS_GAIN_BASIS_VARIABLES = ("surface", "rootzone")
PHYS_GAIN_BASIS_CITATION_KEYS = (
    "SMAP_L4_ATBD_EnKF_gain_innovation",
    "SMAP_L2_L3_ATBD_tau_omega_lband_rtm",
    "Qiu_2021_surface_rootzone_coupling",
    "Willard_2020_physics_guided_ml",
)
PHYS_FORMULA_FEATURE_SCHEMA = (
    "r_enkf",
    "r_rt",
    "r_vert",
    "vod_risk",
    "polarization_mismatch_risk",
    "weak_obs_confidence_risk",
    "finite_input_risk",
    "temperature_contrast_risk",
    "surface_rootzone_forecast_contrast_bounded",
    "base_valid_mask_fraction_diagnostic_only",
)
PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA = (
    "tb_h_normalized_innovation_risk",
    "tb_v_normalized_innovation_risk",
    "tb_innovation_asymmetry_risk",
    "polarization_mismatch_risk",
    "vegetation_opacity_attenuation_risk",
    "weak_observation_confidence_risk",
    "finite_input_risk",
    "soil_surface_temperature_contrast_risk",
    "surface_rootzone_forecast_decoupling_risk",
    "surface_rootzone_hydraulic_gradient_proxy",
    "base_valid_mask_fraction_diagnostic_only",
)
PHYS_FORMULA_GAIN_FEATURE_SCHEMA = (
    "d_H_dry_direction",
    "d_V_dry_direction",
    "m_H_wet_support",
    "m_V_wet_support",
    "gamma",
    "rho_H",
    "rho_V",
    "B_pol",
    "B_temp",
    "B_vert",
    "source_gain_prior_surface_summary",
    "source_gain_prior_rootzone_summary",
    "finite_input_coverage",
    "base_valid_mask_fraction_diagnostic_only",
)
PHYS_FORMULA_SOURCES = (
    PHYS_CONSISTENCY_SOURCE,
    PHYS_FORMULA_SOURCE,
    PHYS_FORMULA_ENHANCED_SOURCE,
    PHYS_FORMULA_GAIN_SOURCE,
)
PHYS_FORMULA_FEATURE_SCHEMAS = {
    PHYS_CONSISTENCY_SOURCE: PHYS_FORMULA_FEATURE_SCHEMA,
    PHYS_FORMULA_SOURCE: PHYS_FORMULA_FEATURE_SCHEMA,
    PHYS_FORMULA_ENHANCED_SOURCE: PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA,
    PHYS_FORMULA_GAIN_SOURCE: PHYS_FORMULA_GAIN_FEATURE_SCHEMA,
}
_FORBIDDEN_PHYS_GAIN_BANK_ROLES = {
    "target_context",
    "target_support",
    "target_val",
    "target_eval",
    "target_query",
    "target_train",
    "target_full_train",
}


def _finite_values(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float32)
    return finite[np.isfinite(finite)]


def _median(values: np.ndarray) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return 0.0
    return float(np.nan_to_num(np.median(finite), nan=0.0, posinf=0.0, neginf=0.0))


def _obs_error_confidence(values: np.ndarray) -> float:
    finite = _finite_values(np.abs(values))
    if finite.size == 0:
        return 0.0
    median_abs_err = max(0.0, float(np.median(finite)))
    return float(1.0 / (1.0 + median_abs_err))


def _finite_coverage(values: np.ndarray, region_mask: np.ndarray | None = None) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    if region_mask is not None:
        mask = np.asarray(region_mask, dtype=bool)
        if mask.sum() <= 0:
            return 0.0
        expanded = np.broadcast_to(mask.reshape(1, *mask.shape), array.shape)
        return float(np.isfinite(array)[expanded].mean())
    return float(np.isfinite(array).mean())


def _base_valid_fraction(values: np.ndarray) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return 0.0
    return float((finite > 0.5).mean())


def _coerce_input_tensor(x: Any) -> np.ndarray:
    array = np.asarray(x, dtype=np.float32)
    if array.ndim != 3 or int(array.shape[0]) != 12:
        raise ValueError(f"PhysTrust-D0 expects raw x with shape [12, H, W], got {array.shape}")
    return array


def _coerce_batched_torch_input(x: Any) -> tuple[torch.Tensor, bool]:
    tensor = x.detach().clone() if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    tensor = tensor.to(dtype=torch.float32)
    squeezed = False
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeezed = True
    if tensor.ndim != 4 or int(tensor.shape[1]) != 12:
        raise ValueError(
            "phys consistency guard expects raw x with shape [B, 12, H, W] "
            f"or [12, H, W], got {tuple(tensor.shape)}"
        )
    return tensor, squeezed


def _coerce_torch_region_mask(
    x: torch.Tensor,
    region_mask: Any | None,
) -> torch.Tensor | None:
    if region_mask is None:
        return None
    mask = region_mask.detach().clone() if isinstance(region_mask, torch.Tensor) else torch.as_tensor(region_mask)
    mask = mask.to(device=x.device)
    if mask.ndim == 2:
        mask = mask.view(1, 1, *mask.shape)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError("region_mask must have shape [H, W], [B, H, W], or [B, 1, H, W]")
    mask = mask.to(dtype=torch.bool)
    if mask.shape[0] == 1 and x.shape[0] > 1:
        mask = mask.expand(x.shape[0], -1, -1, -1)
    if mask.shape[0] != x.shape[0] or mask.shape[-2:] != x.shape[-2:]:
        raise ValueError(
            "region_mask shape must match raw x spatial shape: "
            f"mask={tuple(mask.shape)} x={tuple(x.shape)}"
        )
    return mask


def _masked_torch_values(values: torch.Tensor, mask: torch.Tensor | None, sample_idx: int) -> torch.Tensor:
    row = values[sample_idx]
    if mask is not None:
        row = row[mask[sample_idx, 0]]
    return row[torch.isfinite(row)].float()


def _masked_torch_median(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for sample_idx in range(values.shape[0]):
        finite = _masked_torch_values(values, mask, sample_idx)
        if finite.numel() == 0:
            rows.append(values.new_tensor(0.0, dtype=torch.float32))
        else:
            rows.append(torch.quantile(finite, finite.new_tensor(0.5)))
    return torch.stack(rows, dim=0).to(device=values.device, dtype=values.dtype)


def _masked_torch_finite_coverage(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    finite = torch.isfinite(x)
    for sample_idx in range(x.shape[0]):
        if mask is None:
            denom = int(x[sample_idx].numel())
            rows.append(finite[sample_idx].float().mean() if denom else x.new_tensor(0.0))
            continue
        expanded = mask[sample_idx].expand(x.shape[1], -1, -1)
        denom = int(expanded.sum().detach().cpu().item())
        if denom <= 0:
            rows.append(x.new_tensor(0.0))
        else:
            rows.append((finite[sample_idx] & expanded).float().sum() / float(denom))
    return torch.stack(rows, dim=0).to(device=x.device, dtype=x.dtype)


def _masked_torch_base_valid_fraction(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    base_valid = x[:, 11]
    rows: list[torch.Tensor] = []
    for sample_idx in range(base_valid.shape[0]):
        finite = _masked_torch_values(base_valid, mask, sample_idx)
        if finite.numel() == 0:
            rows.append(base_valid.new_tensor(0.0, dtype=torch.float32))
        else:
            rows.append((finite > 0.5).float().mean())
    return torch.stack(rows, dim=0).to(device=x.device, dtype=x.dtype)


def _coerce_month_values(month: Any | None, batch_size: int) -> list[str]:
    if month is None:
        return ["global"] * batch_size
    if isinstance(month, torch.Tensor):
        values = month.detach().cpu().view(-1).tolist()
    elif isinstance(month, (list, tuple, np.ndarray)):
        values = list(month)
    else:
        values = [month]
    if len(values) == 1 and batch_size > 1:
        values = values * batch_size
    if len(values) != batch_size:
        raise ValueError(f"month must be scalar or length {batch_size}, got length {len(values)}")
    result: list[str] = []
    for value in values:
        try:
            month_int = int(value)
        except Exception:
            result.append("global")
            continue
        if month_int < 1 or month_int > 12:
            result.append("global")
        else:
            result.append(str(month_int))
    return result


def _quantile_row(values: list[float]) -> dict[str, float]:
    finite = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=np.float32)
    if finite.size == 0:
        return {"count": 0, "q50": 0.0, "q75": 0.0, "q90": 0.0, "q95": 0.0, "max": 0.0}
    q50, q75, q90, q95 = np.quantile(finite, [0.50, 0.75, 0.90, 0.95])
    return {
        "count": int(finite.size),
        "q50": float(q50),
        "q75": float(q75),
        "q90": float(q90),
        "q95": float(q95),
        "max": float(finite.max()),
    }


def _vertical_quantiles_from_state(
    source_state: Mapping[str, Any] | None,
    month_key: str,
) -> Mapping[str, Any]:
    if not source_state:
        return {}
    monthly = source_state.get("monthly_vertical_decoupling_quantiles", {})
    if isinstance(monthly, Mapping) and month_key in monthly:
        row = monthly.get(month_key, {})
        return row if isinstance(row, Mapping) else {}
    row = source_state.get("global_vertical_decoupling_quantiles", {})
    return row if isinstance(row, Mapping) else {}


def _vertical_risk_from_quantiles(
    contrast: torch.Tensor,
    *,
    source_state: Mapping[str, Any] | None,
    month: Any | None,
) -> torch.Tensor:
    month_keys = _coerce_month_values(month, int(contrast.shape[0]))
    rows: list[torch.Tensor] = []
    for sample_idx, month_key in enumerate(month_keys):
        q = _vertical_quantiles_from_state(source_state, month_key)
        q50 = float(q.get("q50", 0.0) or 0.0)
        q90 = float(q.get("q90", 0.0) or 0.0)
        value = contrast[sample_idx]
        if np.isfinite(q50) and np.isfinite(q90) and q90 > q50:
            risk = (value - value.new_tensor(q50)) / max(q90 - q50, 1e-6)
        elif np.isfinite(q90) and q90 > 0.0:
            risk = value / max(q90, 1e-6)
        else:
            risk = value / (value + value.new_tensor(0.25))
        rows.append(risk.clamp(0.0, 1.0))
    return torch.stack(rows, dim=0).to(device=contrast.device, dtype=contrast.dtype)


def _stats_from_torch(values: torch.Tensor) -> dict[str, float]:
    detached = values.detach().cpu().float().reshape(-1)
    if detached.numel() == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(detached.mean().item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
    }


def _coerce_gain_basis_tensor(x: Any) -> tuple[torch.Tensor, bool]:
    tensor = x.detach().clone() if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    tensor = tensor.to(dtype=torch.float32)
    squeezed = False
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeezed = True
    if tensor.ndim != 4 or int(tensor.shape[1]) != 12:
        raise ValueError(
            "phys gain basis expects raw x with shape [B, 12, H, W] "
            f"or [12, H, W], got {tuple(tensor.shape)}"
        )
    return tensor, squeezed


def _bounded_signed(values: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.tanh(torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0) / max(float(scale), 1e-12))


def phys_gain_basis_formula_schema() -> dict[str, Any]:
    """Return the M3.12 signed physical gain-basis formula contract."""
    return {
        "schema_version": PHYS_GAIN_BASIS_SCHEMA_VERSION,
        "basis_names": list(PHYS_GAIN_BASIS_NAMES),
        "basis_source": PHYS_GAIN_BASIS_SOURCE,
        "input_channels": {
            "sm_surface_forecast": 0,
            "sm_rootzone_forecast": 1,
            "soil_temp_layer1_forecast": 2,
            "surface_temp_forecast": 3,
            "vegopacity": 4,
            "tb_h_obs": 5,
            "tb_v_obs": 6,
            "tb_h_errstd": 7,
            "tb_v_errstd": 8,
            "tb_h_sim": 9,
            "tb_v_sim": 10,
            "base_valid_mask_diagnostic_only": 11,
        },
        "formula": {
            "d_p": "(TB_p_obs - TB_p_sim) / (TB_p_errstd + eps)",
            "m_p": "-d_p because dTB/dSM < 0; positive m_p means observation supports wetter soil",
            "gamma": "exp(-clip(vegopacity, 0, tau_max) * sec(40deg))",
            "rho_p": "1 / (1 + errstd_p^2)",
            "B_H": "gamma * rho_H * m_H",
            "B_V": "gamma * rho_V * m_V",
            "B_pol": "(TB_V_obs - TB_H_obs)/(TB_V_obs + TB_H_obs + eps) - same_sim",
            "B_temp": "tanh(|soil_temp_layer1_forecast - surface_temp_forecast| / temp_scale)",
            "B_vert": "tanh((sm_surface_forecast - sm_rootzone_forecast) / vert_scale)",
        },
        "citation_keys": list(PHYS_GAIN_BASIS_CITATION_KEYS),
        "label_usage": "none_for_basis_maps",
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "final_eval_only_no_selection",
        "base_valid_mask_usage": "diagnostic_only_not_loss_metric_obs_region_mask_or_gate_mask",
    }


def phys_gain_basis_from_raw_tensor(
    x: Any,
    *,
    tau_max: float = 5.0,
    eps: float = 1e-6,
    temp_scale: float = 5.0,
    vert_scale: float = 0.25,
    return_summary: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return signed M3.12 physics-gain basis maps from raw input-side channels.

    Basis maps are per-pixel and may be used by the source-stage HyperDA
    residual branch. Channel 11 is summarized only as diagnostic coverage.
    """
    raw, squeezed = _coerce_gain_basis_tensor(x)
    eps_value = max(float(eps), 1e-12)
    tau_limit = max(float(tau_max), 0.0)
    sec_40 = 1.0 / float(np.cos(np.deg2rad(40.0)))

    err_h = raw[:, 7].abs()
    err_v = raw[:, 8].abs()
    d_h = (raw[:, 5] - raw[:, 9]) / (err_h + eps_value)
    d_v = (raw[:, 6] - raw[:, 10]) / (err_v + eps_value)
    m_h = -_bounded_signed(d_h)
    m_v = -_bounded_signed(d_v)
    gamma = torch.exp(-raw[:, 4].clamp(0.0, tau_limit) * raw.new_tensor(sec_40))
    rho_h = 1.0 / (1.0 + err_h.square())
    rho_v = 1.0 / (1.0 + err_v.square())
    b_h = gamma * rho_h * m_h
    b_v = gamma * rho_v * m_v
    obs_pol = (raw[:, 6] - raw[:, 5]) / (raw[:, 6] + raw[:, 5] + eps_value)
    sim_pol = (raw[:, 10] - raw[:, 9]) / (raw[:, 10] + raw[:, 9] + eps_value)
    b_pol = _bounded_signed(obs_pol - sim_pol)
    b_temp = torch.tanh((raw[:, 2] - raw[:, 3]).abs() / max(float(temp_scale), eps_value))
    b_vert = _bounded_signed(raw[:, 0] - raw[:, 1], scale=max(float(vert_scale), eps_value))
    basis = torch.stack([b_h, b_v, b_pol, b_temp, b_vert], dim=1)
    basis = torch.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
    if return_summary:
        finite_input_coverage = _masked_torch_finite_coverage(raw, None)
        base_valid_fraction = _masked_torch_base_valid_fraction(raw, None)
        summary = {
            **phys_gain_basis_formula_schema(),
            "enabled": True,
            "basis_stats": {
                name: _stats_from_torch(basis[:, idx])
                for idx, name in enumerate(PHYS_GAIN_BASIS_NAMES)
            },
            "gamma": _stats_from_torch(gamma),
            "rho_H": _stats_from_torch(rho_h),
            "rho_V": _stats_from_torch(rho_v),
            "finite_input_coverage": _stats_from_torch(finite_input_coverage),
            "base_valid_mask_fraction_diagnostic_only": _stats_from_torch(base_valid_fraction),
            "tau_max": float(tau_limit),
            "sec40": float(sec_40),
            "temp_scale": float(temp_scale),
            "vert_scale": float(vert_scale),
        }
    else:
        summary = {"enabled": True, "schema_version": PHYS_GAIN_BASIS_SCHEMA_VERSION}
    if squeezed:
        basis = basis.squeeze(0)
    return basis.to(device=raw.device, dtype=raw.dtype), summary


def _coerce_basis_maps_for_bank(sample: Mapping[str, Any]) -> torch.Tensor:
    if "phys_gain_basis" in sample:
        tensor = sample["phys_gain_basis"]
        tensor = tensor.detach().clone() if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
        tensor = tensor.to(dtype=torch.float32)
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 3 or int(tensor.shape[0]) != len(PHYS_GAIN_BASIS_NAMES):
            raise ValueError(
                "phys_gain_basis sample field must have shape [5, H, W] "
                f"or [1, 5, H, W], got {tuple(tensor.shape)}"
            )
        return tensor
    basis, _ = phys_gain_basis_from_raw_tensor(sample["x"])
    if basis.ndim != 3:
        raise ValueError("single-sample phys gain basis construction did not return [5, H, W]")
    return basis.to(dtype=torch.float32)


def _coerce_increment_map(sample: Mapping[str, Any], key: str) -> torch.Tensor:
    if key not in sample:
        raise ValueError(f"source gain bank sample missing required label field {key!r}")
    tensor = sample[key].detach().clone() if isinstance(sample[key], torch.Tensor) else torch.as_tensor(sample[key])
    return tensor.to(dtype=torch.float32)


def _safe_flat_values(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        values = values[mask]
    values = values.float().reshape(-1)
    return values[torch.isfinite(values)]


def build_phys_gain_source_bank(
    samples: Iterable[Mapping[str, Any]],
    *,
    ridge_lambda: float = 1e-4,
    source_split_roles: Iterable[str] = ("source_fit",),
) -> dict[str, Any]:
    """Build source-only gain priors for M3.12 interpretation/weak regularization."""
    roles = [str(role) for role in source_split_roles]
    forbidden = sorted(set(roles) & _FORBIDDEN_PHYS_GAIN_BANK_ROLES)
    if forbidden:
        raise ValueError(f"phys gain source bank forbids target-side split roles: {forbidden}")
    if not roles or any(role != "source_fit" for role in roles):
        raise ValueError("phys gain source bank must be built from source_fit only")
    def empty_group() -> dict[str, Any]:
        return {
            "sample_count": 0,
            "basis": {
                name: {
                    "count": 0,
                    "sum_x": 0.0,
                    "sum_x2": 0.0,
                    "sum_y_surface": 0.0,
                    "sum_y_rootzone": 0.0,
                    "sum_xy_surface": 0.0,
                    "sum_xy_rootzone": 0.0,
                }
                for name in PHYS_GAIN_BASIS_NAMES
            },
            "coupling": {
                "count": 0,
                "sum_surface": 0.0,
                "sum_rootzone": 0.0,
                "sum_surface2": 0.0,
                "sum_surface_rootzone": 0.0,
            },
        }

    rows_by_group: dict[tuple[str, str], dict[str, Any]] = {}
    n_samples = 0
    for sample in samples:
        role = str(sample.get("split_role", sample.get("source_split", "source_fit")))
        if role in _FORBIDDEN_PHYS_GAIN_BANK_ROLES or role != "source_fit":
            raise ValueError(f"phys gain source bank refuses sample split_role={role!r}")
        region = str(sample.get("sample_region_id") or sample.get("region_id") or sample.get("target_region_id") or "global")
        month = _coerce_month(sample.get("month", 6))
        group = rows_by_group.setdefault((region, month), empty_group())
        group["sample_count"] += int(sample.get("count", 1) or 1)
        n_samples += 1

        basis = _coerce_basis_maps_for_bank(sample)
        inc_s = _coerce_increment_map(sample, "increment_surface")
        inc_r = _coerce_increment_map(sample, "increment_rootzone")
        mask_value = sample.get("loss_mask", sample.get("metric_mask", sample.get("region_mask")))
        mask = None
        if mask_value is not None:
            mask = torch.as_tensor(mask_value, dtype=torch.bool)
            if mask.ndim == 3:
                mask = mask.squeeze(0)
        inc_s_flat = inc_s.float().reshape(-1)
        inc_r_flat = inc_r.float().reshape(-1)
        mask_flat = mask.reshape(-1) if mask is not None else torch.ones_like(inc_s_flat, dtype=torch.bool)
        coupling_valid = mask_flat & torch.isfinite(inc_s_flat) & torch.isfinite(inc_r_flat)
        if bool(coupling_valid.any()):
            s_vals = inc_s_flat[coupling_valid]
            r_vals = inc_r_flat[coupling_valid]
            coupling = group["coupling"]
            coupling["count"] += int(s_vals.numel())
            coupling["sum_surface"] += float(s_vals.sum().item())
            coupling["sum_rootzone"] += float(r_vals.sum().item())
            coupling["sum_surface2"] += float(s_vals.square().sum().item())
            coupling["sum_surface_rootzone"] += float((s_vals * r_vals).sum().item())
        for idx, name in enumerate(PHYS_GAIN_BASIS_NAMES):
            b_flat = basis[idx].float().reshape(-1)
            count = min(int(b_flat.numel()), int(inc_s_flat.numel()), int(inc_r_flat.numel()), int(mask_flat.numel()))
            if count < 1:
                continue
            b_local = b_flat[:count]
            s_local = inc_s_flat[:count]
            r_local = inc_r_flat[:count]
            valid = mask_flat[:count] & torch.isfinite(b_local) & torch.isfinite(s_local) & torch.isfinite(r_local)
            if not bool(valid.any()):
                continue
            b_vals = b_local[valid]
            s_vals = s_local[valid]
            r_vals = r_local[valid]
            stats = group["basis"][name]
            stats["count"] += int(b_vals.numel())
            stats["sum_x"] += float(b_vals.sum().item())
            stats["sum_x2"] += float(b_vals.square().sum().item())
            stats["sum_y_surface"] += float(s_vals.sum().item())
            stats["sum_y_rootzone"] += float(r_vals.sum().item())
            stats["sum_xy_surface"] += float((b_vals * s_vals).sum().item())
            stats["sum_xy_rootzone"] += float((b_vals * r_vals).sum().item())

    lambda_value = max(float(ridge_lambda), 0.0)
    group_priors: dict[str, dict[str, Any]] = {}
    sign_rows = {variable: {name: [] for name in PHYS_GAIN_BASIS_NAMES} for variable in PHYS_GAIN_BASIS_VARIABLES}
    for (region, month), stats_group in sorted(rows_by_group.items()):
        coeffs = {"surface": {}, "rootzone": {}}
        variances = {}
        for name in PHYS_GAIN_BASIS_NAMES:
            stats = stats_group["basis"][name]
            count = int(stats["count"])
            if count <= 1:
                gain_s = 0.0
                gain_r = 0.0
                variance = 0.0
            else:
                inv_n = 1.0 / float(count)
                mean_x = float(stats["sum_x"]) * inv_n
                variance = max(float(stats["sum_x2"]) * inv_n - mean_x * mean_x, 0.0)
                cov_s = float(stats["sum_xy_surface"]) * inv_n - mean_x * (float(stats["sum_y_surface"]) * inv_n)
                cov_r = float(stats["sum_xy_rootzone"]) * inv_n - mean_x * (float(stats["sum_y_rootzone"]) * inv_n)
                gain_s = float(cov_s / (variance + lambda_value))
                gain_r = float(cov_r / (variance + lambda_value))
            coeffs["surface"][name] = gain_s
            coeffs["rootzone"][name] = gain_r
            variances[name] = variance
            sign_rows["surface"][name].append(gain_s)
            sign_rows["rootzone"][name].append(gain_r)
        coupling = stats_group["coupling"]
        coupling_count = int(coupling["count"])
        if coupling_count > 1:
            inv_n = 1.0 / float(coupling_count)
            mean_surface = float(coupling["sum_surface"]) * inv_n
            mean_rootzone = float(coupling["sum_rootzone"]) * inv_n
            var_surface = max(float(coupling["sum_surface2"]) * inv_n - mean_surface * mean_surface, 0.0)
            cov_sr = float(coupling["sum_surface_rootzone"]) * inv_n - mean_surface * mean_rootzone
            c_rz = float(cov_sr / (var_surface + lambda_value))
        else:
            var_surface = 0.0
            c_rz = 0.0
        key = f"{region}|{month}"
        group_priors[key] = {
            "region": region,
            "month": int(month),
            "count": int(stats_group["sample_count"]),
            "G0": coeffs,
            "basis_variance": variances,
            "C0_rootzone_from_surface": c_rz,
            "surface_increment_variance": var_surface,
        }

    sign_agreement = {
        variable: {
            name: float(np.mean(np.asarray(values, dtype=np.float32) >= 0.0)) if values else 0.0
            for name, values in by_basis.items()
        }
        for variable, by_basis in sign_rows.items()
    }
    bank = {
        "schema_version": PHYS_GAIN_BASIS_BANK_SCHEMA_VERSION,
        "formula_schema": phys_gain_basis_formula_schema(),
        "source": "source_fit_labels_only",
        "source_split_roles": {"bank": roles, "forbidden": sorted(_FORBIDDEN_PHYS_GAIN_BANK_ROLES)},
        "label_usage": "source_fit_increments_for_gain_prior_only",
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "final_eval_only_no_selection",
        "ridge_lambda": lambda_value,
        "basis_names": list(PHYS_GAIN_BASIS_NAMES),
        "variables": list(PHYS_GAIN_BASIS_VARIABLES),
        "n_samples": int(n_samples),
        "group_priors": group_priors,
        "sign_agreement_summary": sign_agreement,
        "base_valid_mask_usage": "diagnostic_only_not_loss_metric_obs_region_mask_or_gate_mask",
    }
    bank["source_gain_bank_hash"] = phys_gain_source_bank_hash(bank)
    return bank


def phys_gain_source_bank_hash(bank: Mapping[str, Any] | None) -> str:
    """Stable SHA256 over source gain-bank metadata for checkpoint records."""
    if not bank:
        return ""

    def normalize(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float().tolist()
        if isinstance(value, Mapping):
            return {str(k): normalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    payload = dict(bank)
    payload.pop("source_gain_bank_hash", None)
    encoded = json.dumps(
        normalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def phys_formula_feature_schema_for_source(source: str = PHYS_FORMULA_SOURCE) -> tuple[str, ...]:
    """Return the bounded formula feature schema for a formula source."""
    if source not in PHYS_FORMULA_FEATURE_SCHEMAS:
        raise ValueError(f"Unsupported phys formula source: {source}")
    return tuple(PHYS_FORMULA_FEATURE_SCHEMAS[source])


def _raw_formula_risks_from_raw_tensor(
    x: Any,
    *,
    region_mask: Any | None = None,
    month: Any | None = None,
    source_state: Mapping[str, Any] | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raw, _ = _coerce_batched_torch_input(x)
    mask = _coerce_torch_region_mask(raw, region_mask)
    eps_value = max(float(eps), 1e-12)

    tb_h_innovation = (raw[:, 5] - raw[:, 9]).abs() / raw[:, 7].abs().clamp_min(eps_value)
    tb_v_innovation = (raw[:, 6] - raw[:, 10]).abs() / raw[:, 8].abs().clamp_min(eps_value)
    h_norm = _masked_torch_median(tb_h_innovation, mask)
    v_norm = _masked_torch_median(tb_v_innovation, mask)
    r_enkf = torch.maximum(h_norm, v_norm).clamp(0.0, 1.0)

    vegopacity = _masked_torch_median(raw[:, 4].abs(), mask)
    vod_risk = (vegopacity / (1.0 + vegopacity)).clamp(0.0, 1.0)
    obs_hv = raw[:, 6] - raw[:, 5]
    assim_hv = raw[:, 10] - raw[:, 9]
    pol_delta = _masked_torch_median((obs_hv - assim_hv).abs(), mask)
    pol_scale = (
        _masked_torch_median(obs_hv.abs(), mask)
        + _masked_torch_median(assim_hv.abs(), mask)
        + raw.new_tensor(eps_value)
    )
    pol_risk = (pol_delta / pol_scale).clamp(0.0, 1.0)
    h_err = _masked_torch_median(raw[:, 7].abs(), mask)
    v_err = _masked_torch_median(raw[:, 8].abs(), mask)
    weak_obs_risk = torch.maximum(h_err / (1.0 + h_err), v_err / (1.0 + v_err)).clamp(0.0, 1.0)
    finite_coverage = _masked_torch_finite_coverage(raw, mask)
    finite_risk = (1.0 - finite_coverage).clamp(0.0, 1.0)
    temp_contrast = _masked_torch_median((raw[:, 3] - raw[:, 2]).abs(), mask)
    temp_risk = (temp_contrast / (temp_contrast + raw.new_tensor(5.0))).clamp(0.0, 1.0)
    r_rt = torch.maximum(
        torch.maximum(vod_risk, pol_risk),
        torch.maximum(torch.maximum(weak_obs_risk, finite_risk), temp_risk),
    ).clamp(0.0, 1.0)

    vertical_contrast = _masked_torch_median((raw[:, 1] - raw[:, 0]).abs(), mask)
    r_vert = _vertical_risk_from_quantiles(
        vertical_contrast,
        source_state=source_state,
        month=month,
    )
    vertical_bounded = (vertical_contrast / (vertical_contrast + raw.new_tensor(0.25))).clamp(0.0, 1.0)
    base_valid_fraction = _masked_torch_base_valid_fraction(raw, mask)

    risks = {
        "r_enkf": r_enkf,
        "r_rt": r_rt,
        "r_vert": r_vert,
        "vod_risk": vod_risk,
        "polarization_mismatch_risk": pol_risk,
        "weak_obs_confidence_risk": weak_obs_risk,
        "finite_input_risk": finite_risk,
        "temperature_contrast_risk": temp_risk,
        "surface_rootzone_forecast_contrast_bounded": vertical_bounded,
        "base_valid_mask_fraction_diagnostic_only": base_valid_fraction,
        "finite_input_coverage": finite_coverage,
        "surface_rootzone_forecast_contrast_abs_median": vertical_contrast,
    }
    features = torch.stack(
        [risks[key].clamp(0.0, 1.0) for key in PHYS_FORMULA_FEATURE_SCHEMA],
        dim=1,
    ).to(device=raw.device, dtype=raw.dtype)
    features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return features, risks


def _raw_enhanced_formula_features_from_raw_tensor(
    x: Any,
    *,
    region_mask: Any | None = None,
    month: Any | None = None,
    source_state: Mapping[str, Any] | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raw, _ = _coerce_batched_torch_input(x)
    mask = _coerce_torch_region_mask(raw, region_mask)
    eps_value = max(float(eps), 1e-12)

    tb_h_innovation = (raw[:, 5] - raw[:, 9]).abs() / (raw[:, 7].abs() + eps_value)
    tb_v_innovation = (raw[:, 6] - raw[:, 10]).abs() / (raw[:, 8].abs() + eps_value)
    h_norm = _masked_torch_median(tb_h_innovation, mask)
    v_norm = _masked_torch_median(tb_v_innovation, mask)
    h_risk = h_norm.clamp(0.0, 1.0)
    v_risk = v_norm.clamp(0.0, 1.0)
    innovation_asymmetry = (
        (h_norm - v_norm).abs() / (h_norm + v_norm + raw.new_tensor(eps_value))
    ).clamp(0.0, 1.0)

    obs_hv = raw[:, 6] - raw[:, 5]
    assim_hv = raw[:, 10] - raw[:, 9]
    pol_delta = _masked_torch_median((obs_hv - assim_hv).abs(), mask)
    pol_scale = (
        _masked_torch_median(obs_hv.abs(), mask)
        + _masked_torch_median(assim_hv.abs(), mask)
        + raw.new_tensor(eps_value)
    )
    pol_risk = (pol_delta / pol_scale).clamp(0.0, 1.0)
    vegopacity = _masked_torch_median(raw[:, 4].abs(), mask)
    vod_risk = (vegopacity / (1.0 + vegopacity)).clamp(0.0, 1.0)
    h_err = _masked_torch_median(raw[:, 7].abs(), mask)
    v_err = _masked_torch_median(raw[:, 8].abs(), mask)
    weak_obs_risk = torch.maximum(h_err / (1.0 + h_err), v_err / (1.0 + v_err)).clamp(0.0, 1.0)
    finite_coverage = _masked_torch_finite_coverage(raw, mask)
    finite_risk = (1.0 - finite_coverage).clamp(0.0, 1.0)
    temp_contrast = _masked_torch_median((raw[:, 3] - raw[:, 2]).abs(), mask)
    temp_risk = (temp_contrast / (temp_contrast + raw.new_tensor(5.0))).clamp(0.0, 1.0)

    vertical_contrast = _masked_torch_median((raw[:, 1] - raw[:, 0]).abs(), mask)
    vertical_risk = _vertical_risk_from_quantiles(
        vertical_contrast,
        source_state=source_state,
        month=month,
    )
    hydraulic_gradient = torch.relu(raw[:, 0] - raw[:, 1])
    hydraulic_gradient_median = _masked_torch_median(hydraulic_gradient, mask)
    hydraulic_gradient_proxy = (
        hydraulic_gradient_median / (hydraulic_gradient_median + raw.new_tensor(0.25))
    ).clamp(0.0, 1.0)
    base_valid_fraction = _masked_torch_base_valid_fraction(raw, mask)

    risks = {
        "tb_h_normalized_innovation_risk": h_risk,
        "tb_v_normalized_innovation_risk": v_risk,
        "tb_innovation_asymmetry_risk": innovation_asymmetry,
        "polarization_mismatch_risk": pol_risk,
        "vegetation_opacity_attenuation_risk": vod_risk,
        "weak_observation_confidence_risk": weak_obs_risk,
        "finite_input_risk": finite_risk,
        "soil_surface_temperature_contrast_risk": temp_risk,
        "surface_rootzone_forecast_decoupling_risk": vertical_risk,
        "surface_rootzone_hydraulic_gradient_proxy": hydraulic_gradient_proxy,
        "base_valid_mask_fraction_diagnostic_only": base_valid_fraction,
        "finite_input_coverage": finite_coverage,
        "surface_rootzone_forecast_contrast_abs_median": vertical_contrast,
    }
    features = torch.stack(
        [risks[key].clamp(0.0, 1.0) for key in PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA],
        dim=1,
    ).to(device=raw.device, dtype=raw.dtype)
    features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return features, risks


def _source_gain_prior_summary_value(
    source_state: Mapping[str, Any] | None,
    variable: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    """Return a neutral bounded source-fit gain-prior summary unless provided."""
    neutral = torch.full((batch_size,), 0.5, device=device, dtype=dtype)
    if not source_state:
        return neutral
    container = source_state.get("m3_14_gain_prior_summary", source_state.get("source_gain_prior_summary", {}))
    if not isinstance(container, Mapping):
        return neutral
    value = container.get(variable, container.get(f"{variable}_summary"))
    if value is None:
        return neutral
    try:
        scalar = float(value)
    except Exception:
        return neutral
    if not np.isfinite(scalar):
        return neutral
    return torch.full((batch_size,), float(np.clip(scalar, 0.0, 1.0)), device=device, dtype=dtype)


def _raw_formula_gain_features_from_raw_tensor(
    x: Any,
    *,
    region_mask: Any | None = None,
    month: Any | None = None,
    source_state: Mapping[str, Any] | None = None,
    eps: float = 1e-6,
    tau_max: float = 5.0,
    temp_scale: float = 5.0,
    vert_scale: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raw, _ = _coerce_batched_torch_input(x)
    mask = _coerce_torch_region_mask(raw, region_mask)
    eps_value = max(float(eps), 1e-12)
    tau_limit = max(float(tau_max), 0.0)
    sec_40 = 1.0 / float(np.cos(np.deg2rad(40.0)))

    err_h = raw[:, 7].abs()
    err_v = raw[:, 8].abs()
    d_h_map = (raw[:, 5] - raw[:, 9]) / (err_h + eps_value)
    d_v_map = (raw[:, 6] - raw[:, 10]) / (err_v + eps_value)
    d_h_signed = _masked_torch_median(_bounded_signed(d_h_map), mask)
    d_v_signed = _masked_torch_median(_bounded_signed(d_v_map), mask)
    m_h_map = -_bounded_signed(d_h_map)
    m_v_map = -_bounded_signed(d_v_map)
    m_h_signed = _masked_torch_median(m_h_map, mask)
    m_v_signed = _masked_torch_median(m_v_map, mask)

    gamma_map = torch.exp(-raw[:, 4].clamp(0.0, tau_limit) * raw.new_tensor(sec_40))
    gamma = _masked_torch_median(gamma_map, mask).clamp(0.0, 1.0)
    rho_h = _masked_torch_median(1.0 / (1.0 + err_h.square()), mask).clamp(0.0, 1.0)
    rho_v = _masked_torch_median(1.0 / (1.0 + err_v.square()), mask).clamp(0.0, 1.0)

    obs_pol = (raw[:, 6] - raw[:, 5]) / (raw[:, 6] + raw[:, 5] + eps_value)
    sim_pol = (raw[:, 10] - raw[:, 9]) / (raw[:, 10] + raw[:, 9] + eps_value)
    b_pol_signed = _masked_torch_median(_bounded_signed(obs_pol - sim_pol), mask)
    b_temp = _masked_torch_median(
        torch.tanh((raw[:, 2] - raw[:, 3]).abs() / max(float(temp_scale), eps_value)),
        mask,
    ).clamp(0.0, 1.0)
    b_vert_signed = _masked_torch_median(
        _bounded_signed(raw[:, 0] - raw[:, 1], scale=max(float(vert_scale), eps_value)),
        mask,
    )

    source_gain_prior_surface = _source_gain_prior_summary_value(
        source_state,
        "surface",
        device=raw.device,
        dtype=raw.dtype,
        batch_size=int(raw.shape[0]),
    )
    source_gain_prior_rootzone = _source_gain_prior_summary_value(
        source_state,
        "rootzone",
        device=raw.device,
        dtype=raw.dtype,
        batch_size=int(raw.shape[0]),
    )
    finite_coverage = _masked_torch_finite_coverage(raw, mask)
    base_valid_fraction = _masked_torch_base_valid_fraction(raw, mask)

    risks = {
        "d_H_dry_direction": ((d_h_signed + 1.0) * 0.5).clamp(0.0, 1.0),
        "d_V_dry_direction": ((d_v_signed + 1.0) * 0.5).clamp(0.0, 1.0),
        "m_H_wet_support": ((m_h_signed + 1.0) * 0.5).clamp(0.0, 1.0),
        "m_V_wet_support": ((m_v_signed + 1.0) * 0.5).clamp(0.0, 1.0),
        "gamma": gamma,
        "rho_H": rho_h,
        "rho_V": rho_v,
        "B_pol": ((b_pol_signed + 1.0) * 0.5).clamp(0.0, 1.0),
        "B_temp": b_temp,
        "B_vert": ((b_vert_signed + 1.0) * 0.5).clamp(0.0, 1.0),
        "source_gain_prior_surface_summary": source_gain_prior_surface,
        "source_gain_prior_rootzone_summary": source_gain_prior_rootzone,
        "finite_input_coverage": finite_coverage.clamp(0.0, 1.0),
        "base_valid_mask_fraction_diagnostic_only": base_valid_fraction,
        "d_H_signed": d_h_signed,
        "d_V_signed": d_v_signed,
        "m_H_signed": m_h_signed,
        "m_V_signed": m_v_signed,
        "B_pol_signed": b_pol_signed,
        "B_vert_signed": b_vert_signed,
        "tb_h_normalized_innovation_abs_median": _masked_torch_median(d_h_map.abs(), mask),
        "tb_v_normalized_innovation_abs_median": _masked_torch_median(d_v_map.abs(), mask),
    }
    features = torch.stack(
        [risks[key].clamp(0.0, 1.0) for key in PHYS_FORMULA_GAIN_FEATURE_SCHEMA],
        dim=1,
    ).to(device=raw.device, dtype=raw.dtype)
    features = torch.nan_to_num(features, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return features, risks


def phys_formula_features_from_raw_tensor(
    x: Any,
    *,
    region_mask: Any | None = None,
    month: Any | None = None,
    source_state: Mapping[str, Any] | None = None,
    mode: str = PHYS_FORMULA_MODE,
    source: str = PHYS_FORMULA_SOURCE,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return bounded formula features from raw input-side channels only."""
    if mode != PHYS_FORMULA_MODE:
        raise ValueError(f"Unsupported phys formula mode: {mode}")
    if source not in PHYS_FORMULA_SOURCES:
        raise ValueError(f"Unsupported phys formula source: {source}")
    if source == PHYS_FORMULA_GAIN_SOURCE:
        features, risks = _raw_formula_gain_features_from_raw_tensor(
            x,
            region_mask=region_mask,
            month=month,
            source_state=source_state,
            eps=eps,
        )
        schema_version = PHYS_FORMULA_GAIN_OPERATOR_SCHEMA_VERSION
        feature_schema = PHYS_FORMULA_GAIN_FEATURE_SCHEMA
        formula = {
            "d_p": "(TB_p_obs - TB_p_sim) / (TB_p_errstd + eps); encoded as 0.5*(tanh(d_p)+1), >0.5 dry-direction",
            "m_p": "-tanh(d_p); encoded as 0.5*(m_p+1), >0.5 wet-support",
            "gamma": "exp(-clip(vegopacity, 0, tau_max) * sec(40deg))",
            "rho_p": "1 / (1 + TB_p_errstd^2)",
            "B_pol": "bounded observed-minus-simulated H/V polarization contrast encoded around 0.5",
            "B_temp": "tanh(|soil_temp_layer1_forecast - surface_temp_forecast| / temp_scale)",
            "B_vert": "tanh((sm_surface_forecast - sm_rootzone_forecast) / vert_scale) encoded around 0.5",
            "source_gain_prior_summary": "source_fit-only bounded summary when provided; neutral 0.5 otherwise",
        }
        stats_keys = (
            "d_H_dry_direction",
            "d_V_dry_direction",
            "m_H_wet_support",
            "m_V_wet_support",
            "gamma",
            "rho_H",
            "rho_V",
            "B_pol",
            "B_temp",
            "B_vert",
            "source_gain_prior_surface_summary",
            "source_gain_prior_rootzone_summary",
            "d_H_signed",
            "d_V_signed",
            "m_H_signed",
            "m_V_signed",
        )
    elif source == PHYS_FORMULA_ENHANCED_SOURCE:
        features, risks = _raw_enhanced_formula_features_from_raw_tensor(
            x,
            region_mask=region_mask,
            month=month,
            source_state=source_state,
            eps=eps,
        )
        schema_version = PHYS_FORMULA_ENHANCED_OPERATOR_SCHEMA_VERSION
        feature_schema = PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA
        formula = {
            "tb_h_normalized_innovation_risk": "|TB_H_obs - TB_H_assim| / (obs_err_H + eps)",
            "tb_v_normalized_innovation_risk": "|TB_V_obs - TB_V_assim| / (obs_err_V + eps)",
            "tb_innovation_asymmetry_risk": "|r_H - r_V| / (r_H + r_V + eps)",
            "surface_rootzone_forecast_decoupling_risk": "monthly source-fit quantile-bounded |rootzone-surface forecast|",
            "surface_rootzone_hydraulic_gradient_proxy": "bounded relu(surface_forecast - rootzone_forecast)",
        }
        stats_keys = (
            "tb_h_normalized_innovation_risk",
            "tb_v_normalized_innovation_risk",
            "tb_innovation_asymmetry_risk",
            "polarization_mismatch_risk",
            "surface_rootzone_forecast_decoupling_risk",
            "surface_rootzone_hydraulic_gradient_proxy",
        )
    else:
        features, risks = _raw_formula_risks_from_raw_tensor(
            x,
            region_mask=region_mask,
            month=month,
            source_state=source_state,
            eps=eps,
        )
        schema_version = PHYS_FORMULA_OPERATOR_SCHEMA_VERSION
        feature_schema = PHYS_FORMULA_FEATURE_SCHEMA
        formula = {
            "r_enkf": "median(|TB_obs-TB_assim|/(obs_err+eps)) over H/V",
            "r_rt": "max(VOD risk, H/V polarization mismatch, weak obs confidence, finite risk, temperature contrast risk)",
            "r_vert": "surface-rootzone forecast decoupling bounded by monthly source-side quantiles",
        }
        stats_keys = ("r_enkf", "r_rt", "r_vert")
    summary = {
        "enabled": True,
        "schema_version": schema_version,
        "mode": mode,
        "phys_formula_source": source,
        "source": "x_raw_region_mask_month_only",
        "label_usage": "none",
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "final_eval_only_no_selection",
        "feature_schema": list(feature_schema),
        "base_valid_mask_usage": "diagnostic_only_not_loss_metric_obs_region_mask_or_gate_mask",
        "finite_input_coverage": _stats_from_torch(risks["finite_input_coverage"]),
        "base_valid_mask_fraction_diagnostic_only": _stats_from_torch(
            risks["base_valid_mask_fraction_diagnostic_only"]
        ),
        "formula": formula,
    }
    if source == PHYS_FORMULA_GAIN_SOURCE:
        summary.update(
            {
                "method_id": "M3_16_source_only_phys_m3trust_lite_or_M3_14_formula_gain",
                "feature_encoding": "bounded_0_to_1_signed_features_use_0p5_as_neutral",
                "coefficient_injection_role": "bounded_operator_coefficient_logit_delta_only",
                "final_output_residual_allowed": False,
                "label_usage": "none_for_features_source_fit_only_for_optional_gain_prior_summary",
                "source_fit_regularization_role": "weak_high_confidence_tb_innovation_increment_direction_consistency",
                "source_fit_regularization_lambda_default": 0.01,
                "channel_11_usage": "diagnostic_only_not_hard_mask",
            }
        )
    for key in stats_keys:
        summary[key] = _stats_from_torch(risks[key])
    return features, summary


def phys_consistency_source_state_from_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    mode: str = PHYS_CONSISTENCY_GUARD_MODE,
    source: str | None = None,
) -> dict[str, Any]:
    """Build source-side monthly vertical quantiles from input-side samples only."""
    if mode not in PHYS_CONSISTENCY_GUARD_MODES:
        raise ValueError(f"Unsupported phys consistency guard mode: {mode}")
    resolved_source = source or (
        PHYS_FORMULA_SOURCE
        if mode == PHYS_CONSISTENCY_GUARD_PRODUCT_MODE
        else PHYS_CONSISTENCY_SOURCE
    )
    if resolved_source not in PHYS_FORMULA_SOURCES:
        raise ValueError(f"Unsupported phys consistency source: {resolved_source}")
    rows_by_month: dict[str, list[float]] = {str(month): [] for month in range(1, 13)}
    for sample in samples:
        x = sample["x"]
        month_key = _coerce_month(sample.get("month", 6))
        region_mask = sample.get("region_mask", sample.get("active_region_mask"))
        diagnostics = phys_trust_d0_diagnostics_from_tensor(x, region_mask=region_mask)
        rows_by_month[month_key].append(
            float(diagnostics["surface_rootzone_forecast_contrast_abs_median"])
        )
    all_rows = [value for rows in rows_by_month.values() for value in rows]
    return {
        "schema_version": PHYS_CONSISTENCY_GUARD_SCHEMA_VERSION,
        "mode": mode,
        "source": f"source_fit_input_side_{resolved_source}",
        "phys_consistency_source": resolved_source,
        "phys_formula_source": resolved_source if resolved_source != PHYS_CONSISTENCY_SOURCE else "",
        "label_usage": "none",
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "final_eval_only_no_selection",
        "model_selection_usage": "source_val_gate_only_no_target_eval",
        "monthly_vertical_decoupling_quantiles": {
            str(month): _quantile_row(rows_by_month[str(month)]) for month in range(1, 13)
        },
        "global_vertical_decoupling_quantiles": _quantile_row(all_rows),
        "base_valid_mask_usage": "diagnostic_only_not_loss_metric_obs_region_mask_or_gate_mask",
    }


def phys_consistency_guard_from_raw_tensor(
    x: Any,
    *,
    region_mask: Any | None = None,
    month: Any | None = None,
    source_state: Mapping[str, Any] | None = None,
    source: str | None = None,
    mode: str = PHYS_CONSISTENCY_GUARD_MODE,
    min_surface: float = 0.95,
    min_rootzone: float = 0.90,
    strength_surface: float = 0.10,
    strength_rootzone: float = 0.15,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return M3.7 shrink-or-identity per-variable residual gates.

    The formulas read only the existing raw input-side channels. Channel 11 is
    summarized as diagnostic coverage but is not used as a hard mask.
    """
    if mode not in PHYS_CONSISTENCY_GUARD_MODES:
        raise ValueError(f"Unsupported phys consistency guard mode: {mode}")
    if source is not None and source not in PHYS_FORMULA_SOURCES:
        raise ValueError(f"Unsupported phys consistency source: {source}")
    if not 0.0 <= float(min_surface) <= 1.0:
        raise ValueError("phys_consistency_min_surface must be in [0, 1]")
    if not 0.0 <= float(min_rootzone) <= 1.0:
        raise ValueError("phys_consistency_min_rootzone must be in [0, 1]")
    if float(strength_surface) < 0.0 or float(strength_rootzone) < 0.0:
        raise ValueError("phys consistency guard strengths must be non-negative")

    raw, _ = _coerce_batched_torch_input(x)
    _, risks = _raw_formula_risks_from_raw_tensor(
        raw,
        region_mask=region_mask,
        month=month,
        source_state=source_state,
        eps=eps,
    )
    r_enkf = risks["r_enkf"]
    r_rt = risks["r_rt"]
    r_vert = risks["r_vert"]
    finite_coverage = risks["finite_input_coverage"]
    base_valid_fraction = risks["base_valid_mask_fraction_diagnostic_only"]

    surface_risk = torch.maximum(r_enkf, r_rt).clamp(0.0, 1.0)
    if mode == PHYS_CONSISTENCY_GUARD_PRODUCT_MODE:
        rootzone_risk = (surface_risk * r_vert).clamp(0.0, 1.0)
        source_name = source or PHYS_FORMULA_SOURCE
        guard_action = "surface_primary_product_shrink_or_identity_variable_trust_gate"
        rootzone_formula = "clamp(1 - strength_rootzone * max(r_enkf, r_rt) * r_vert, min_rootzone, 1)"
    else:
        rootzone_risk = torch.maximum(surface_risk, r_vert).clamp(0.0, 1.0)
        source_name = PHYS_CONSISTENCY_SOURCE
        guard_action = "shrink_or_identity_variable_trust_gate"
        rootzone_formula = "clamp(1 - strength_rootzone * max(r_enkf, r_rt, r_vert), min_rootzone, 1)"
    gate_surface = (1.0 - float(strength_surface) * surface_risk).clamp(
        float(min_surface),
        1.0,
    )
    gate_rootzone = (1.0 - float(strength_rootzone) * rootzone_risk).clamp(
        float(min_rootzone),
        1.0,
    )
    gates = torch.stack([gate_surface, gate_rootzone], dim=1)

    summary = {
        "enabled": True,
        "schema_version": PHYS_CONSISTENCY_GUARD_SCHEMA_VERSION,
        "mode": mode,
        "phys_consistency_source": source_name,
        "phys_formula_source": source_name if source_name != PHYS_CONSISTENCY_SOURCE else "",
        "source": "x_raw_region_mask_month_only",
        "label_usage": "none",
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "final_eval_only_no_selection",
        "guard_action": guard_action,
        "base_valid_mask_usage": "diagnostic_only_not_loss_metric_obs_region_mask_or_gate_mask",
        "min_surface": float(min_surface),
        "min_rootzone": float(min_rootzone),
        "strength_surface": float(strength_surface),
        "strength_rootzone": float(strength_rootzone),
        "r_enkf": _stats_from_torch(r_enkf),
        "r_rt": _stats_from_torch(r_rt),
        "r_vert": _stats_from_torch(r_vert),
        "rootzone_risk": _stats_from_torch(rootzone_risk),
        "finite_input_coverage": _stats_from_torch(finite_coverage),
        "base_valid_mask_fraction_diagnostic_only": _stats_from_torch(base_valid_fraction),
        "surface_gate": _stats_from_torch(gate_surface),
        "rootzone_gate": _stats_from_torch(gate_rootzone),
        "formula": {
            "r_enkf": "median(|TB_obs-TB_assim|/(obs_err+eps)) over H/V",
            "r_rt": "max(VOD risk, H/V polarization mismatch, weak obs confidence, finite coverage risk, temperature contrast risk)",
            "r_vert": "surface-rootzone forecast decoupling bounded by monthly source-side quantiles",
            "g_surface": "clamp(1 - strength_surface * max(r_enkf, r_rt), min_surface, 1)",
            "g_rootzone": rootzone_formula,
        },
    }
    return gates, summary


def _coerce_region_mask(array: np.ndarray, region_mask: Any | None) -> np.ndarray | None:
    if region_mask is None:
        return None
    mask = np.asarray(region_mask) > 0.5
    if mask.shape != array.shape[-2:]:
        raise ValueError(
            "PhysTrust-D0 region_mask shape must match x spatial shape: "
            f"mask={mask.shape} x={array.shape[-2:]}"
        )
    return mask


def _apply_region_mask(array: np.ndarray, region_mask: np.ndarray | None) -> np.ndarray:
    if region_mask is None:
        return array
    mask = region_mask
    return np.where(mask.reshape(1, *mask.shape), array, np.nan)


def phys_trust_d0_diagnostics_from_tensor(
    x: Any,
    *,
    region_mask: Any | None = None,
) -> dict[str, float]:
    """Compute PhysTrust-D0 diagnostics from one raw 12-channel input tensor."""
    raw_array = _coerce_input_tensor(x)
    mask = _coerce_region_mask(raw_array, region_mask)
    array = _apply_region_mask(raw_array, mask)
    tb_h_innovation = array[5] - array[9]
    tb_v_innovation = array[6] - array[10]
    h_norm_abs = np.abs(tb_h_innovation) / (1.0 + np.maximum(np.abs(array[7]), 0.0))
    v_norm_abs = np.abs(tb_v_innovation) / (1.0 + np.maximum(np.abs(array[8]), 0.0))
    diagnostics = {
        "tb_h_normalized_innovation_abs_median": _median(h_norm_abs),
        "tb_v_normalized_innovation_abs_median": _median(v_norm_abs),
        "tb_h_obs_error_confidence": _obs_error_confidence(array[7]),
        "tb_v_obs_error_confidence": _obs_error_confidence(array[8]),
        "vegopacity_median": _median(array[4]),
        "surface_rootzone_forecast_contrast_abs_median": _median(np.abs(array[1] - array[0])),
        "finite_input_coverage": _finite_coverage(raw_array, mask),
        "base_valid_mask_fraction_diagnostic_only": _base_valid_fraction(array[11]),
    }
    return {key: float(np.clip(np.nan_to_num(value, nan=0.0), 0.0, np.inf)) for key, value in diagnostics.items()}


def _coerce_month(value: Any) -> str:
    try:
        month = int(value)
    except Exception:
        month = 6
    month = min(12, max(1, month))
    return str(month)


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in PHYS_TRUST_D0_DIAGNOSTIC_KEYS}
    return {
        key: float(np.mean([float(row.get(key, 0.0)) for row in rows]))
        for key in PHYS_TRUST_D0_DIAGNOSTIC_KEYS
    }


def _trust_gate_from_summary(trust_summary: Mapping[str, Any]) -> float:
    """Diagnostic-only gate proxy inferred from trust summary metadata."""
    if not trust_summary or not bool(trust_summary.get("enabled", False)):
        return 1.0
    nearest_bounded = float(trust_summary.get("nearest_distance_bounded", 0.0) or 0.0)
    trust_strength = float(trust_summary.get("trust_strength", 0.0) or 0.0)
    gate = 1.0 - trust_strength * np.clip(nearest_bounded, 0.0, 1.0)
    return float(np.clip(gate, 0.0, 1.0))


def _variable_trust_summary(gate: float) -> dict[str, dict[str, float]]:
    row = {"trust_gate_diagnostic": float(gate)}
    return {"surface": dict(row), "rootzone": dict(row)}


def phys_trust_d0_summary_from_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    hyperda_trust_summary_by_month: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate PhysTrust-D0 diagnostics by month from input-side samples only.

    Only ``x`` and ``month`` are read from each sample. Optional trust summaries
    are metadata already produced from source-side trust-bank routing.
    """
    rows_by_month: dict[str, list[dict[str, float]]] = {str(month): [] for month in range(1, 13)}
    for sample in samples:
        x = sample["x"]
        month_key = _coerce_month(sample.get("month", 6))
        region_mask = sample.get("region_mask", sample.get("active_region_mask"))
        rows_by_month[month_key].append(
            phys_trust_d0_diagnostics_from_tensor(x, region_mask=region_mask)
        )
    return phys_trust_d0_summary_from_monthly_rows(
        rows_by_month,
        hyperda_trust_summary_by_month=hyperda_trust_summary_by_month,
    )


def phys_trust_d0_summary_from_monthly_rows(
    rows_by_month: Mapping[str, Iterable[Mapping[str, float]]],
    *,
    hyperda_trust_summary_by_month: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate precomputed PhysTrust-D0 diagnostic rows by month."""
    normalized_rows_by_month: dict[str, list[dict[str, float]]] = {str(month): [] for month in range(1, 13)}
    for month, rows in rows_by_month.items():
        month_key = _coerce_month(month)
        normalized_rows_by_month[month_key].extend(
            {
                key: float(row.get(key, 0.0))
                for key in PHYS_TRUST_D0_DIAGNOSTIC_KEYS
            }
            for row in rows
        )

    trust_by_month = dict(hyperda_trust_summary_by_month or {})
    monthly: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, float]] = []
    for month in range(1, 13):
        month_key = str(month)
        rows = normalized_rows_by_month[month_key]
        all_rows.extend(rows)
        mean_values = _mean_rows(rows)
        trust_summary = dict(trust_by_month.get(month_key, {}))
        gate = _trust_gate_from_summary(trust_summary)
        monthly[month_key] = {
            "count": int(len(rows)),
            **mean_values,
            "trust_gate_diagnostic": gate,
            "per_variable_trust": _variable_trust_summary(gate),
            "hyperda_trust_summary": trust_summary,
        }

    return {
        "schema_version": PHYS_TRUST_D0_SCHEMA_VERSION,
        "diagnostic_layer": "PhysTrust-D0",
        "source": "input_side_x_month_region_mask_only",
        "label_usage": "none",
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "final_eval_only_no_selection",
        "model_selection_usage": "forbidden_diagnostic_only",
        "base_valid_mask_usage": "bounded_diagnostic_coverage_only_not_loss_metric_obs_or_region_mask",
        "diagnostic_schema": list(PHYS_TRUST_D0_DIAGNOSTIC_KEYS),
        "monthly": monthly,
        "overall": {
            "count": int(len(all_rows)),
            **_mean_rows(all_rows),
        },
    }
