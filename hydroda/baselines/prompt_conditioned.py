"""Prompt-conditioned backbone predictor for HydroDA-OOD / HyperDA V4.

No-leakage declaration:
    - Uses trained FiLMConditionalResUNet + RegionPromptEncoder checkpoint
    - Prompt uses input-side features only (x, region_id, month)
    - No target_eval/query labels used in prompt construction
    - Held-out target unseen-region prompt fallback is retained for split compatibility
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import torch

from hydroda.models.conditional_unet import FiLMConditionalResUNet
from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.hyper_conditional_unet import SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA
from hydroda.models.prompt_encoder import RegionPromptEncoder, RobustInputSideDAPromptEncoder


# Mapping from region name (e.g. "US-R1") to region index (0..5)
_REGION_TO_IDX = {
    "US-R1": 0,
    "US-R2": 1,
    "US-R3": 2,
    "US-R4": 3,
    "US-R5": 4,
    "US-R6": 5,
}

TARGET_CONTEXT_PROMPT_SCHEMA_VERSION = "target_context_prompt_state_v1"
TARGET_CONTEXT_PROMPT_SOURCE = "target_context_monthly_prompt_prototypes"
_MAIN_HYPERDA_METHOD_IDS = {
    "hyperda_zero_shot_context",
    "hyperda_safe_few_shot_k4",
    "hyperda_safe_few_shot_k12",
}
_DIAGNOSTIC_HYPERDA_METHOD_IDS = {
    "hyperda_diagnostic_few_shot_k4",
    "hyperda_diagnostic_few_shot_k12",
}
_LEGACY_HYPERDA_METHOD_ALIASES = {
    "hyperda_few_shot_k4": "hyperda_safe_few_shot_k4",
    "hyperda_few_shot_k12": "hyperda_safe_few_shot_k12",
}
CURRENT_MEAN_STD_CONTEXT_ENCODER = "current_mean_std"
ROBUST_DA_CONTEXT_ENCODER = "robust_input_side_da_diagnostics"
ROBUST_DA_RAW_CONTEXT_ENCODER = "robust_input_side_da_diagnostics_raw"
_CONTEXT_ENCODERS = {
    CURRENT_MEAN_STD_CONTEXT_ENCODER,
    ROBUST_DA_CONTEXT_ENCODER,
    ROBUST_DA_RAW_CONTEXT_ENCODER,
}
RELIABILITY_FEATURE_TRANSFORM = "bounded_v2"
STAGE3_K0_CONTEXT_SHRINKAGE_VARIANT = "M2_4_target_context_conservative_hyperda"
STAGE3_K0_CONTEXT_SHRINKAGE_SCALAR_POLICY = "scalar_reliability_v1"
STAGE3_K0_CONTEXT_SHRINKAGE_VARIABLE_POLICY = "variable_reliability_v1"
STAGE3_K0_CONTEXT_SHRINKAGE_SOURCE_EPISODE_POLICY = "source_episode_calibrated_v1"
STAGE3_K0_CONTEXT_SHRINKAGE_POLICY = STAGE3_K0_CONTEXT_SHRINKAGE_VARIABLE_POLICY
STAGE3_K0_CONTEXT_SHRINKAGE_POLICIES = {
    STAGE3_K0_CONTEXT_SHRINKAGE_SCALAR_POLICY,
    STAGE3_K0_CONTEXT_SHRINKAGE_VARIABLE_POLICY,
    STAGE3_K0_CONTEXT_SHRINKAGE_SOURCE_EPISODE_POLICY,
}


def _build_prompt_encoder(
    *,
    context_encoder: str,
    num_regions: int,
    input_channels: int,
    hidden_dim: int,
) -> RegionPromptEncoder:
    if context_encoder == CURRENT_MEAN_STD_CONTEXT_ENCODER:
        return RegionPromptEncoder(
            num_regions=num_regions,
            input_channels=input_channels,
            hidden_dim=hidden_dim,
        )
    if context_encoder in {ROBUST_DA_CONTEXT_ENCODER, ROBUST_DA_RAW_CONTEXT_ENCODER}:
        return RobustInputSideDAPromptEncoder(
            num_regions=num_regions,
            input_channels=input_channels,
            hidden_dim=hidden_dim,
        )
    raise ValueError(f"Unsupported context_encoder: {context_encoder}")


def is_raw_da_context_encoder(context_encoder: str) -> bool:
    return str(context_encoder) == ROBUST_DA_RAW_CONTEXT_ENCODER


def is_da_context_encoder(context_encoder: str) -> bool:
    return str(context_encoder) in {ROBUST_DA_CONTEXT_ENCODER, ROBUST_DA_RAW_CONTEXT_ENCODER}


def prompt_diagnostic_input_domain(context_encoder: str) -> str:
    """Return the tensor domain used by the prompt input-summary branch."""
    if is_raw_da_context_encoder(context_encoder):
        return "raw_input_side"
    if str(context_encoder) == ROBUST_DA_CONTEXT_ENCODER:
        return "normalized_input_side_legacy"
    return "normalized_input_side"


def prompt_input_feature_source(context_encoder: str) -> str:
    if is_raw_da_context_encoder(context_encoder):
        return "da_aware_raw_input_side_current_context_fields_only"
    if str(context_encoder) == ROBUST_DA_CONTEXT_ENCODER:
        return "da_aware_input_side_current_context_fields_only"
    return "current_input_mean_std"


def prompt_channel_11_usage(context_encoder: str) -> str:
    if is_da_context_encoder(context_encoder):
        return "bounded_base_valid_coverage_diagnostic_only_not_loss_metric_obs_or_region_mask"
    return "mean_std_input_summary"


def prompt_normalized_input_used(context_encoder: str) -> bool:
    return not is_raw_da_context_encoder(context_encoder)


def target_context_input_usage(context_encoder: str) -> str:
    if is_raw_da_context_encoder(context_encoder):
        return "target_context_raw_input_side_da_diagnostics"
    if str(context_encoder) == ROBUST_DA_CONTEXT_ENCODER:
        return "target_context_normalized_input_side_da_diagnostics_legacy"
    return "target_context_normalized_input_summary_only"


def prompt_diagnostic_tensor(
    prompt_encoder: RegionPromptEncoder,
    *,
    context_encoder: str,
    x_norm: torch.Tensor,
    x_raw: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select the tensor domain for prompt diagnostics without changing model input.

    The model backbone continues to consume normalized tensors. Only the prompt
    input-summary branch switches to raw input for the explicit raw DA encoder.
    """
    if context_encoder not in _CONTEXT_ENCODERS:
        raise ValueError(f"Unsupported context_encoder: {context_encoder}")
    if is_raw_da_context_encoder(context_encoder):
        if x_raw is None:
            raise ValueError("raw DA context encoder requires x_raw for prompt diagnostics")
        return x_raw
    return x_norm


def prompt_domain_metadata(context_encoder: str) -> Dict[str, Any]:
    return {
        "prompt_diagnostic_input_domain": prompt_diagnostic_input_domain(context_encoder),
        "prompt_input_feature_source": prompt_input_feature_source(context_encoder),
        "prompt_channel_11_usage": prompt_channel_11_usage(context_encoder),
        "normalized_input_used_for_prompt_diagnostics": prompt_normalized_input_used(context_encoder),
    }


def _hyperda_method_id_from_config(config: Dict[str, Any]) -> Optional[str]:
    method = config.get("method")
    if method in _MAIN_HYPERDA_METHOD_IDS:
        return str(method)
    if method in _DIAGNOSTIC_HYPERDA_METHOD_IDS:
        return str(method)
    if method in _LEGACY_HYPERDA_METHOD_ALIASES:
        return _LEGACY_HYPERDA_METHOD_ALIASES[str(method)]
    setting = config.get("adaptation_setting")
    if config.get("paper_facing_run") is False:
        if setting == "few_shot_k4":
            return "hyperda_diagnostic_few_shot_k4"
        if setting == "few_shot_k12":
            return "hyperda_diagnostic_few_shot_k12"
    if setting == "zero_shot_context":
        return "hyperda_zero_shot_context"
    if setting == "few_shot_k4":
        return "hyperda_safe_few_shot_k4"
    if setting == "few_shot_k12":
        return "hyperda_safe_few_shot_k12"
    return None


def _coerce_month(value: Any, date_str: str = "") -> int:
    try:
        month = int(value)
    except Exception:
        month = int(date_str[5:7]) if date_str and len(date_str) >= 7 else 6
    if month < 1 or month > 12:
        raise ValueError(f"month must be in 1..12, got {month}")
    return month


def _bounded_count_feature(count: float) -> float:
    return float(np.clip(np.log1p(max(0.0, float(count))) / np.log1p(365.0), 0.0, 1.0))


def _bounded_context_count_feature(count: float) -> float:
    return float(np.clip(np.log1p(max(0.0, float(count))) / np.log1p(365.0 * 7.0), 0.0, 1.0))


def _bounded_distance_feature(distance: float) -> float:
    value = max(0.0, float(distance))
    return float(np.clip(value / (1.0 + value), 0.0, 1.0))


def bounded_reliability_features(
    *,
    monthly_count: float,
    has_monthly_prototype: float,
    global_context_count: float,
    finite_input_coverage: float,
    prompt_to_source_manifold_distance: float,
) -> list[float]:
    """Return finite bounded reliability features in the model schema order."""
    features = [
        _bounded_count_feature(monthly_count),
        1.0 if float(has_monthly_prototype) > 0.0 else 0.0,
        _bounded_context_count_feature(global_context_count),
        float(np.clip(finite_input_coverage, 0.0, 1.0)),
        _bounded_distance_feature(prompt_to_source_manifold_distance),
    ]
    return [float(np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)) for value in features]


def compute_stage3_k0_context_shrinkage_rho(
    reliability_features: Sequence[Any] | Dict[str, Any],
    *,
    source_calibrated_rho_cap: float = 1.0,
) -> float:
    """Map target_context input-only reliability features to residual shrinkage."""
    if isinstance(reliability_features, dict):
        values = [
            reliability_features.get("monthly_count", 0.0),
            reliability_features.get("has_monthly_prototype", 0.0),
            reliability_features.get("global_context_count", 0.0),
            reliability_features.get("finite_input_coverage", 0.0),
            reliability_features.get("prompt_to_source_manifold_distance", 1.0),
        ]
    else:
        values = list(reliability_features)
    padded = [*values, *([0.0] * 5)][:5]
    cap = float(np.nan_to_num(source_calibrated_rho_cap, nan=0.0, posinf=1.0, neginf=0.0))
    if cap <= 0.0:
        return 0.0
    cap = min(1.0, cap)
    monthly_count = float(np.clip(np.nan_to_num(float(padded[0]), nan=0.0), 0.0, 1.0))
    has_monthly = float(np.clip(np.nan_to_num(float(padded[1]), nan=0.0), 0.0, 1.0))
    global_count = float(np.clip(np.nan_to_num(float(padded[2]), nan=0.0), 0.0, 1.0))
    coverage = float(np.clip(np.nan_to_num(float(padded[3]), nan=0.0), 0.0, 1.0))
    distance = float(np.clip(np.nan_to_num(float(padded[4]), nan=1.0), 0.0, 1.0))
    reliability = (
        0.30 * monthly_count
        + 0.20 * has_monthly
        + 0.15 * global_count
        + 0.25 * coverage
        + 0.10 * (1.0 - distance)
    )
    return float(np.clip(cap * reliability, 0.0, cap))


def _stage3_clean_rho_cap(value: Any) -> float:
    cap = float(np.nan_to_num(float(value), nan=0.0, posinf=1.0, neginf=0.0))
    return float(np.clip(cap, 0.0, 1.0))


def compute_stage3_k0_context_shrinkage_rhos(
    reliability_features: Sequence[Any] | Dict[str, Any],
    *,
    source_calibrated_rho_cap: float = 1.0,
    surface_rho_cap: float | None = None,
    rootzone_rho_cap: float | None = None,
    policy: str = STAGE3_K0_CONTEXT_SHRINKAGE_VARIABLE_POLICY,
) -> Dict[str, float]:
    """Return variable-specific Stage 3 K=0 residual shrinkage rhos.

    ``scalar_reliability_v1`` is the M2.4-compatible mode: one source-calibrated
    cap is applied to both variables.  Variable policies use the same input-only
    reliability score but permit independent source-episode-calibrated caps.
    """
    policy = str(policy or STAGE3_K0_CONTEXT_SHRINKAGE_VARIABLE_POLICY)
    if policy not in STAGE3_K0_CONTEXT_SHRINKAGE_POLICIES:
        raise ValueError(f"Unsupported Stage 3 K=0 context shrinkage policy: {policy}")
    scalar_cap = _stage3_clean_rho_cap(source_calibrated_rho_cap)
    if policy == STAGE3_K0_CONTEXT_SHRINKAGE_SCALAR_POLICY:
        surface_cap = scalar_cap
        rootzone_cap = scalar_cap
    else:
        surface_cap = scalar_cap if surface_rho_cap is None else _stage3_clean_rho_cap(surface_rho_cap)
        rootzone_cap = scalar_cap if rootzone_rho_cap is None else _stage3_clean_rho_cap(rootzone_rho_cap)
    return {
        "surface": compute_stage3_k0_context_shrinkage_rho(
            reliability_features,
            source_calibrated_rho_cap=surface_cap,
        ),
        "rootzone": compute_stage3_k0_context_shrinkage_rho(
            reliability_features,
            source_calibrated_rho_cap=rootzone_cap,
        ),
    }


def _masked_input_stats_from_tensor(
    prompt_encoder: RegionPromptEncoder,
    x_norm: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Compute prompt input stats inside a spatial mask without reading labels."""
    if mask is None:
        return prompt_encoder._compute_input_stats(x_norm)
    if x_norm.ndim != 4:
        raise ValueError("x_norm must have shape [B, C, H, W]")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError("mask must have shape [H, W], [B, H, W], or [B, 1, H, W]")
    mask = mask.to(device=x_norm.device, dtype=torch.bool)
    if mask.shape[0] == 1 and x_norm.shape[0] > 1:
        mask = mask.expand(x_norm.shape[0], -1, -1, -1)
    if mask.shape[0] != x_norm.shape[0] or mask.shape[-2:] != x_norm.shape[-2:]:
        raise ValueError(
            "mask shape is incompatible with x_norm: "
            f"mask={tuple(mask.shape)} x_norm={tuple(x_norm.shape)}"
        )
    x_masked = x_norm.masked_fill(~mask.expand(-1, x_norm.shape[1], -1, -1), float("nan"))
    return prompt_encoder._compute_input_stats(x_masked)


def masked_input_embedding_and_coverage(
    prompt_encoder: RegionPromptEncoder,
    x_norm: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, float]:
    """Return prompt input embedding and finite coverage inside ``mask``."""
    input_stats = _masked_input_stats_from_tensor(prompt_encoder, x_norm, mask)
    input_emb = prompt_encoder.input_proj(input_stats)
    if mask is None:
        finite = torch.isfinite(x_norm)
        coverage = float(finite.float().mean().detach().cpu().item()) if x_norm.numel() else 0.0
    else:
        if mask.ndim == 2:
            mask_for_cov = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask_for_cov = mask.unsqueeze(1)
        else:
            mask_for_cov = mask
        mask_for_cov = mask_for_cov.to(device=x_norm.device, dtype=torch.bool)
        if mask_for_cov.shape[0] == 1 and x_norm.shape[0] > 1:
            mask_for_cov = mask_for_cov.expand(x_norm.shape[0], -1, -1, -1)
        expanded = mask_for_cov.expand(-1, x_norm.shape[1], -1, -1)
        denom = int(expanded.sum().detach().cpu().item())
        if denom <= 0:
            coverage = 0.0
        else:
            coverage = float((torch.isfinite(x_norm) & expanded).float().sum().detach().cpu().item() / denom)
    return input_emb, float(np.clip(coverage, 0.0, 1.0))


def _prompt_tensor(value: Any, device: torch.device | str | None = None) -> Optional[torch.Tensor]:
    if value is None:
        return None
    tensor = value.detach().clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
    tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def normalize_target_context_prompt_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized target-context prompt state with CPU tensor prototypes."""
    if not state:
        raise ValueError("target_context_prompt_state is empty")
    schema = state.get("schema_version")
    if schema != TARGET_CONTEXT_PROMPT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported target_context_prompt_state schema_version={schema!r}; "
            f"expected {TARGET_CONTEXT_PROMPT_SCHEMA_VERSION!r}"
        )

    monthly_counts_raw = state.get("monthly_counts", {})
    monthly_counts = {str(month): int(monthly_counts_raw.get(str(month), 0)) for month in range(1, 13)}
    reliability_raw = state.get("reliability_features", {})
    reliability_features = {
        str(month): [
            float(value)
            for value in reliability_raw.get(
                str(month),
                [0.0] * len(SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA),
            )
        ]
        for month in range(1, 13)
    }
    monthly_raw = state.get("monthly_prototypes", {})
    monthly_prototypes = {
        str(month): _prompt_tensor(monthly_raw.get(str(month)))
        for month in range(1, 13)
    }
    global_prototype = _prompt_tensor(state.get("global_prototype"))
    if global_prototype is None:
        raise ValueError("target_context_prompt_state missing global_prototype")

    normalized = dict(state)
    normalized["schema_version"] = TARGET_CONTEXT_PROMPT_SCHEMA_VERSION
    normalized["prompt_source"] = TARGET_CONTEXT_PROMPT_SOURCE
    normalized["label_usage"] = "none"
    normalized["monthly_counts"] = monthly_counts
    normalized["reliability_feature_schema"] = list(
        state.get("reliability_feature_schema") or SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA
    )
    normalized["reliability_features"] = reliability_features
    normalized["monthly_prototypes"] = monthly_prototypes
    normalized["global_prototype"] = global_prototype.detach().cpu()
    context_hash = str(normalized.get("context_hash") or normalized.get("context_date_hash") or "")
    normalized["context_hash"] = context_hash
    normalized["context_date_hash"] = context_hash
    normalized["metadata"] = dict(state.get("metadata", {}))
    context_encoder = str(normalized["metadata"].get("context_encoder", CURRENT_MEAN_STD_CONTEXT_ENCODER))
    if context_encoder not in _CONTEXT_ENCODERS:
        context_encoder = CURRENT_MEAN_STD_CONTEXT_ENCODER
    normalized["metadata"].setdefault("context_encoder", context_encoder)
    normalized["metadata"].setdefault("input_usage", target_context_input_usage(context_encoder))
    for key, value in prompt_domain_metadata(context_encoder).items():
        normalized["metadata"].setdefault(key, value)
    normalized["metadata"].setdefault("eval_input_usage", "none_for_prompt_update")
    normalized["metadata"].setdefault("eval_month_usage", "known_seasonal_phase_selector_only")
    normalized["metadata"].setdefault("temporal_usage", "month_of_year_seasonal_phase")
    normalized["metadata"].setdefault("reliability_feature_source", "input_side_context_summary_only")
    normalized["metadata"].setdefault("reliability_feature_transform", RELIABILITY_FEATURE_TRANSFORM)
    normalized["metadata"].setdefault("channel_11_usage", "finite_input_feature_only_not_observation_or_static_mask")
    normalized["metadata"].setdefault("target_val_usage", "unused_in_main_protocol")
    normalized["metadata"].setdefault("target_eval_usage", "final_eval_only_no_selection")
    return normalized


def compose_target_context_prompt_from_state(
    state: Dict[str, Any],
    months: int | Sequence[int] | torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Select monthly target-context prompt prototypes, falling back to global."""
    normalized = normalize_target_context_prompt_state(state)
    if isinstance(months, torch.Tensor):
        month_values = [int(v) for v in months.detach().cpu().view(-1).tolist()]
    elif isinstance(months, int):
        month_values = [int(months)]
    else:
        month_values = [int(v) for v in months]

    prompts = []
    for month in month_values:
        month = _coerce_month(month)
        prompt = normalized["monthly_prototypes"].get(str(month))
        if prompt is None:
            prompt = normalized["global_prototype"]
        prompts.append(prompt.to(device=device) if device is not None else prompt)
    return torch.stack(prompts, dim=0)


def target_context_prompt_metadata(state: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_target_context_prompt_state(state)
    metadata = dict(normalized.get("metadata", {}))
    metadata.update(
        {
            "schema_version": normalized["schema_version"],
            "prompt_source": normalized["prompt_source"],
            "label_usage": normalized["label_usage"],
            "context_hash": normalized.get("context_hash", ""),
            "context_date_hash": normalized.get("context_date_hash", normalized.get("context_hash", "")),
            "n_samples": int(normalized.get("n_samples", sum(normalized["monthly_counts"].values()))),
            "date_start": normalized.get("date_start", ""),
            "date_end": normalized.get("date_end", ""),
            "monthly_counts": dict(normalized["monthly_counts"]),
            "reliability_feature_schema": list(normalized.get("reliability_feature_schema", [])),
        }
    )
    return metadata


def _hash_context_dates(dates: Sequence[str], monthly_counts: Dict[str, int]) -> str:
    payload = json.dumps(
        {"dates": list(dates), "monthly_counts": monthly_counts},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_target_context_prompt_state(
    samples: Iterable[Dict[str, Any]],
    prompt_encoder: RegionPromptEncoder,
    normalize_x: Callable[[torch.Tensor], torch.Tensor],
    target_region_embedding: torch.Tensor,
    device: torch.device | str,
    context_hash: str = "",
    context_encoder: str = CURRENT_MEAN_STD_CONTEXT_ENCODER,
) -> Dict[str, Any]:
    """Build monthly target-context prompt prototypes from input-side fields only.

    Reads only ``x``, ``month``, ``date_str``, and region/input masks from each
    sample. Target labels, increments, residuals, validation scores, and eval
    inputs are not consulted.
    """
    if context_encoder not in _CONTEXT_ENCODERS:
        raise ValueError(f"Unsupported context_encoder: {context_encoder}")
    device = torch.device(device)
    by_month: Dict[int, list[torch.Tensor]] = {month: [] for month in range(1, 13)}
    finite_coverage_by_month: Dict[int, list[float]] = {month: [] for month in range(1, 13)}
    dates: list[str] = []
    all_input_embs: list[torch.Tensor] = []
    all_temporal_embs: list[torch.Tensor] = []

    target_region_embedding = target_region_embedding.to(device=device, dtype=torch.float32)
    if target_region_embedding.ndim == 1:
        target_region_embedding = target_region_embedding.unsqueeze(0)
    elif target_region_embedding.ndim != 2 or target_region_embedding.shape[0] != 1:
        raise ValueError("target_region_embedding must have shape [16] or [1,16]")

    with torch.no_grad():
        for sample in samples:
            x_np = np.asarray(sample["x"], dtype=np.float32)
            region_mask_np = sample.get("region_mask", sample.get("active_region_mask"))
            region_mask_t = None
            if region_mask_np is not None:
                region_mask_t = torch.as_tensor(np.asarray(region_mask_np) > 0.5, dtype=torch.bool, device=device)
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            x_norm = normalize_x(x)
            x_prompt = prompt_diagnostic_tensor(
                prompt_encoder,
                context_encoder=context_encoder,
                x_norm=x_norm,
                x_raw=x,
            )
            input_emb, finite_coverage = masked_input_embedding_and_coverage(
                prompt_encoder,
                x_prompt,
                region_mask_t,
            )

            date_str = str(sample.get("date_str", ""))
            month_value = _coerce_month(sample.get("month", None), date_str)
            month = torch.tensor([month_value], dtype=torch.long, device=device)
            temporal = prompt_encoder._temporal_encoding(month)
            temporal_emb = prompt_encoder.temporal_proj(temporal)

            by_month[month_value].append(input_emb.detach())
            finite_coverage_by_month[month_value].append(finite_coverage)
            all_input_embs.append(input_emb.detach())
            all_temporal_embs.append(temporal_emb.detach())
            if date_str:
                dates.append(date_str)

        if not all_input_embs:
            raise ValueError("Cannot build target_context prompt state from zero samples")

        r_emb = target_region_embedding
        global_i = torch.stack(all_input_embs, dim=0).mean(dim=0)
        global_t = torch.stack(all_temporal_embs, dim=0).mean(dim=0)
        global_prompt = prompt_encoder.mlp(torch.cat([r_emb, global_i, global_t], dim=1)).squeeze(0).detach().cpu()

        monthly_prototypes: Dict[str, Optional[torch.Tensor]] = {}
        monthly_counts: Dict[str, int] = {}
        reliability_features: Dict[str, list[float]] = {}
        for month_value in range(1, 13):
            month_key = str(month_value)
            input_embs = by_month[month_value]
            monthly_counts[month_key] = len(input_embs)
            if not input_embs:
                monthly_prototypes[month_key] = None
                reliability_features[month_key] = bounded_reliability_features(
                    monthly_count=0.0,
                    has_monthly_prototype=0.0,
                    global_context_count=float(len(all_input_embs)),
                    finite_input_coverage=(
                        float(np.mean([v for values in finite_coverage_by_month.values() for v in values]))
                        if all_input_embs
                        else 0.0
                    ),
                    prompt_to_source_manifold_distance=0.0,
                )
                continue
            month_i = torch.stack(input_embs, dim=0).mean(dim=0)
            month_tensor = torch.tensor([month_value], dtype=torch.long, device=device)
            month_t = prompt_encoder.temporal_proj(prompt_encoder._temporal_encoding(month_tensor))
            prompt = prompt_encoder.mlp(torch.cat([r_emb, month_i, month_t], dim=1)).squeeze(0)
            prompt_cpu = prompt.detach().cpu()
            monthly_prototypes[month_key] = prompt_cpu
            reliability_features[month_key] = bounded_reliability_features(
                monthly_count=float(len(input_embs)),
                has_monthly_prototype=1.0,
                global_context_count=float(len(all_input_embs)),
                finite_input_coverage=(
                    float(np.mean(finite_coverage_by_month[month_value]))
                    if finite_coverage_by_month[month_value]
                    else 0.0
                ),
                prompt_to_source_manifold_distance=float(torch.linalg.vector_norm(prompt_cpu - global_prompt).item()),
            )

    return {
        "schema_version": TARGET_CONTEXT_PROMPT_SCHEMA_VERSION,
        "prompt_source": TARGET_CONTEXT_PROMPT_SOURCE,
        "label_usage": "none",
        "context_hash": context_hash or _hash_context_dates(dates, monthly_counts),
        "context_date_hash": context_hash or _hash_context_dates(dates, monthly_counts),
        "date_start": min(dates) if dates else "",
        "date_end": max(dates) if dates else "",
        "n_samples": int(sum(monthly_counts.values())),
        "monthly_counts": monthly_counts,
        "reliability_feature_schema": list(SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA),
        "reliability_features": reliability_features,
        "global_prototype": global_prompt,
        "monthly_prototypes": monthly_prototypes,
        "metadata": {
            "prompt_source": TARGET_CONTEXT_PROMPT_SOURCE,
            "context_encoder": context_encoder,
            "input_usage": target_context_input_usage(context_encoder),
            **prompt_domain_metadata(context_encoder),
            "region_usage": "target_region_embedding_or_source_mean_fallback",
            "temporal_usage": "month_of_year_seasonal_phase",
            "reliability_feature_source": "input_side_context_summary_only",
            "reliability_feature_transform": RELIABILITY_FEATURE_TRANSFORM,
            "input_summary_mask": "target_active_region_mask",
            "channel_11_usage": "finite_input_feature_only_not_observation_or_static_mask",
            "label_usage": "none",
            "target_val_usage": "unused_in_main_protocol",
            "target_eval_usage": "final_eval_only_no_selection",
            "eval_input_usage": "none_for_prompt_update",
            "eval_month_usage": "known_seasonal_phase_selector_only",
        },
    }


class PromptConditionedBackbonePredictor:
    """Neural predictor wrapping trained FiLMConditionalResUNet + RegionPromptEncoder.

    Loads checkpoint, sets model and prompt encoder to eval mode, and predicts
    with region-conditioned prompt.

    Held-out target unseen-region prompt fallback:
        When target region is not in the source region set, uses the mean of all
        source region embeddings as the target region embedding. This is correct
        because the prompt encoder's num_regions only covers source regions,
        so the target region index would otherwise alias a source region embedding.

    Args:
        checkpoint_path: path to trained .pt checkpoint
        device: device string (default "cuda")
        target_region: target region name (e.g. "US-R1")
        target_region_idx: override target region embedding index (default: from _REGION_TO_IDX)
        apply_residual_gain: apply residual gain alpha from calibration (default True)
    """

    method_name = "prompt_conditioned_shared_backbone"

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        target_region: Optional[str] = None,
        target_region_idx: Optional[int] = None,
        apply_residual_gain: bool = True,
    ) -> None:
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)

        if target_region_idx is None and target_region is not None:
            target_region_idx = _REGION_TO_IDX.get(target_region, 0)
        if target_region_idx is None:
            target_region_idx = 0
        self._target_region_idx = target_region_idx

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        saved_config = checkpoint.get("config", {})
        source_config = checkpoint.get("source_checkpoint_config", {})
        stage3_posterior_state_dict = checkpoint.get("stage3_posterior_state_dict", {})
        stage3_posterior_metadata = (
            stage3_posterior_state_dict.get("metadata", {})
            if isinstance(stage3_posterior_state_dict, dict)
            else {}
        )
        if not stage3_posterior_metadata:
            stage3_posterior_metadata = saved_config.get("stage3_posterior_state", {}) or {}
        self.stage3_protocol_metadata: Dict[str, Any] = {
            "posterior_state_schema": (
                stage3_posterior_state_dict.get("schema_version", "")
                if isinstance(stage3_posterior_state_dict, dict)
                else ""
            ),
            "posterior_metadata_schema": stage3_posterior_metadata.get("schema_version", ""),
            "posterior_form": stage3_posterior_metadata.get("posterior_form", ""),
            "K": stage3_posterior_metadata.get("K", saved_config.get("K")),
            "adapt_scope": stage3_posterior_metadata.get("adapt_scope", saved_config.get("adapt_scope", "")),
            "stage3_posterior_policy": stage3_posterior_metadata.get(
                "stage3_posterior_policy",
                saved_config.get("stage3_posterior_policy", ""),
            ),
            "stage3_posterior_decision": stage3_posterior_metadata.get(
                "stage3_posterior_decision",
                saved_config.get("stage3_posterior_decision", ""),
            ),
            "support_gate_status": stage3_posterior_metadata.get(
                "support_gate_status",
                saved_config.get("support_gate_status", ""),
            ),
            "support_gate_enabled": bool(
                stage3_posterior_metadata.get(
                    "support_gate_enabled",
                    saved_config.get("support_gate_enabled", False),
                )
            ),
            "anchor_alpha": stage3_posterior_metadata.get("anchor_alpha", saved_config.get("anchor_alpha")),
            "source_prior_hash_before": stage3_posterior_metadata.get(
                "source_prior_hash_before",
                saved_config.get("stage3_source_prior_hash_before", ""),
            ),
            "source_prior_hash_after": stage3_posterior_metadata.get(
                "source_prior_hash_after",
                saved_config.get("stage3_source_prior_hash_after", ""),
            ),
            "source_prior_unchanged": bool(
                stage3_posterior_state_dict.get(
                    "source_prior_unchanged",
                    saved_config.get("stage3_source_prior_unchanged", False),
                )
                if isinstance(stage3_posterior_state_dict, dict)
                else saved_config.get("stage3_source_prior_unchanged", False)
            ),
            "target_adapter_state_hash": stage3_posterior_metadata.get(
                "target_adapter_state_hash",
                stage3_posterior_state_dict.get("target_adapter_state_hash", "")
                if isinstance(stage3_posterior_state_dict, dict)
                else "",
            ),
            "target_adapter_anchor_hash": stage3_posterior_metadata.get(
                "target_adapter_anchor_hash",
                stage3_posterior_state_dict.get("target_adapter_anchor_hash", "")
                if isinstance(stage3_posterior_state_dict, dict)
                else "",
            ),
            "target_labels_loaded_for_adaptation": bool(
                stage3_posterior_metadata.get("target_labels_loaded_for_adaptation", False)
            ),
            "target_labels_used_for_adaptation": bool(
                stage3_posterior_metadata.get("target_labels_used_for_adaptation", False)
            ),
            "target_val_usage": stage3_posterior_metadata.get(
                "target_val_usage",
                saved_config.get("target_val_usage", ""),
            ),
            "target_eval_usage": stage3_posterior_metadata.get(
                "target_eval_usage",
                saved_config.get("target_eval_usage", ""),
            ),
        }

        def cfg_get(name: str, default: Any = None) -> Any:
            if name in saved_config and saved_config[name] is not None:
                return saved_config[name]
            return source_config.get(name, default)

        if bool(cfg_get("enable_pigo", False)):
            raise ValueError(
                "PIGO target-adaptation checkpoints are no longer supported. "
                "Use a non-PIGO spatial-rootzone Phase 5 checkpoint."
            )

        # Init conditional backbone
        width = cfg_get("width", 32)
        prompt_dim = cfg_get("prompt_dim", 64)
        model_type = saved_config.get("model_type", "prompt_conditioned")
        self.model_type = model_type
        is_hyperda = model_type in {"hyperda_basis_adapter", "hyperda_basis_adapter_target_adapt"}
        is_target_adapt = model_type == "hyperda_basis_adapter_target_adapt"
        if is_target_adapt:
            self.method_name = (
                _hyperda_method_id_from_config(saved_config)
                or _hyperda_method_id_from_config(source_config)
                or "hyperda_target_adapt"
            )
        elif is_hyperda:
            self.method_name = "hyperda_basis_adapter_shared"
        else:
            self.method_name = "prompt_conditioned_shared_backbone"
        if is_hyperda:
            self.model = HyperAdapterConditionalResUNet(
                in_channels=12,
                out_channels=2,
                width=width,
                prompt_dim=prompt_dim,
                hyper_n_basis=cfg_get("hyper_n_basis", 8),
                hyper_adapter_bottleneck=cfg_get("hyper_adapter_bottleneck"),
                hyper_adapter_scale=cfg_get("hyper_adapter_scale", 1.0),
                hyper_coeff_generator=cfg_get("hyper_coeff_generator", "per_adapter"),
                hyper_rank_gate_top_k=cfg_get("hyper_rank_gate_top_k", 4),
                hyper_rank_gate_temperature_init=cfg_get("hyper_rank_gate_temperature_init", 1.0),
                hyper_adapter_param_style=cfg_get("hyper_adapter_param_style", "basis_1x1"),
                hyper_reliability_gate=cfg_get("hyper_reliability_gate", "none"),
                hyper_reliability_init=cfg_get("hyper_reliability_init", 0.95),
                hyper_source_saliency_prior=cfg_get("hyper_source_saliency_prior"),
                hyper_source_saliency_prior_beta=cfg_get("hyper_source_saliency_prior_beta", 0.0),
                hyper_source_saliency_prior_path=cfg_get("hyper_source_saliency_prior_path", ""),
                hyper_source_saliency_prior_application=cfg_get(
                    "hyper_source_saliency_prior_application",
                    "soft_regularization_metadata",
                ),
                hyper_prompt_manifold_reliability=cfg_get("hyper_prompt_manifold_reliability", False),
                hyper_prompt_manifold_reliability_strength=cfg_get(
                    "hyper_prompt_manifold_reliability_strength",
                    0.0,
                ),
                hyper_enable_film=cfg_get("hyper_enable_film", True),
                hyper_enable_adapters=cfg_get("hyper_enable_adapters", True),
                zero_shot_prior_form=cfg_get("zero_shot_prior_form", "direct_hyper"),
                source_residual_rho=cfg_get("source_residual_rho", cfg_get("zero_shot_rho", 1.0)),
                source_residual_gate=cfg_get("source_residual_gate", "prompt_reliability_scalar"),
                source_residual_gate_init=cfg_get("source_residual_gate_init", 0.95),
                source_residual_reliability_dim=cfg_get("source_residual_reliability_dim", 5),
                zero_raw_increment_init=cfg_get("zero_raw_increment_init", False),
                enable_target_adaptation=is_target_adapt,
                target_latent_dim=cfg_get("target_latent_dim", 32),
                enable_target_spatial_refine=cfg_get("enable_target_spatial_refine", False),
                target_spatial_refine_hidden=cfg_get("target_spatial_refine_hidden", 16),
                target_spatial_refine_rootzone=cfg_get("target_spatial_refine_rootzone", False),
                target_spatial_refine_input=cfg_get("target_spatial_refine_input", "normalized"),
                target_spatial_refine_type=cfg_get("target_spatial_refine_type", "simple"),
                target_spatial_refine_gain_span=cfg_get("target_spatial_refine_gain_span", 0.25),
                hydro_msr_hidden=cfg_get("hydro_msr_hidden", cfg_get("target_spatial_refine_hidden", 16)),
                enable_hydro_msr_da_film=cfg_get("enable_hydro_msr_da_film", False),
            )
        else:
            self.model = FiLMConditionalResUNet(
                in_channels=12,
                out_channels=2,
                width=width,
                prompt_dim=prompt_dim,
                zero_raw_increment_init=cfg_get("zero_raw_increment_init", False),
            )
        if is_hyperda:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device).eval()
        self._requires_month = is_target_adapt

        # Init context prompt encoder. Old checkpoints omit this field.
        num_regions = cfg_get("num_regions", 6)
        self.context_encoder = cfg_get("context_encoder", "current_mean_std")
        if self.context_encoder not in _CONTEXT_ENCODERS:
            raise ValueError(f"Unsupported checkpoint context_encoder: {self.context_encoder}")
        self.prompt_encoder = _build_prompt_encoder(
            context_encoder=self.context_encoder,
            num_regions=num_regions,
            input_channels=12,
            hidden_dim=prompt_dim,
        )
        if "prompt_encoder_state_dict" in checkpoint:
            self.prompt_encoder.load_state_dict(checkpoint["prompt_encoder_state_dict"])
        self.prompt_encoder.to(device).eval()

        # Held-out target fallback: target region not in source region set.
        # The prompt_encoder was trained with num_regions = len(source_regions).
        # _REGION_TO_IDX maps global region names to indices 0..5, but the
        # prompt encoder's embedding indices correspond only to source regions.
        # We use source_region_global_indices from the checkpoint to determine
        # if the target is unseen. If so, use the mean of all source embeddings
        # as an "unknown target" embedding.
        self._is_target_unseen = False
        self._target_region_emb: Optional[torch.Tensor] = None
        self._source_global_to_prompt_idx: Dict[int, int] = {}
        source_global_indices = cfg_get("source_region_global_indices")
        if source_global_indices is not None:
            self.source_regions = [f"US-R{int(global_idx) + 1}" for global_idx in source_global_indices]
            self._source_global_to_prompt_idx = {
                int(global_idx): prompt_idx
                for prompt_idx, global_idx in enumerate(source_global_indices)
            }
            source_global_set = set(source_global_indices)
            if self._target_region_idx not in source_global_set:
                self._is_target_unseen = True
                with torch.no_grad():
                    all_emb = self.prompt_encoder.region_embed.weight.data.clone()  # [N, 16]
                    self._target_region_emb = all_emb.mean(dim=0)  # [16]
        elif self._target_region_idx >= num_regions:
            self.source_regions = [f"US-R{i + 1}" for i in range(num_regions)]
            # Fallback for old checkpoints without source_region_global_indices
            self._is_target_unseen = True
            with torch.no_grad():
                all_emb = self.prompt_encoder.region_embed.weight.data.clone()
                self._target_region_emb = all_emb.mean(dim=0)
        else:
            self.source_regions = [f"US-R{i + 1}" for i in range(num_regions)]

        # Normalization params
        ch_mean = cfg_get("ch_mean")
        ch_std = cfg_get("ch_std")
        self._ch_mean = np.array(ch_mean, dtype=np.float32) if ch_mean is not None else None
        self._ch_std = np.array(ch_std, dtype=np.float32) if ch_std is not None else None

        # Increment normalization params
        inc_mean = cfg_get("inc_mean")
        inc_std = cfg_get("inc_std")
        self._inc_mean = np.array(inc_mean, dtype=np.float32) if inc_mean is not None else None
        self._inc_std = np.array(inc_std, dtype=np.float32) if inc_std is not None else None
        self._has_inc_norm = self._inc_mean is not None and self._inc_std is not None

        # Residual gain alphas (from source_val calibration)
        self.alpha_surface = float(checkpoint.get("residual_gain_alpha_surface", 1.0))
        self.alpha_rootzone = float(checkpoint.get("residual_gain_alpha_rootzone", 1.0))
        self.apply_residual_gain = apply_residual_gain
        self._prompt_route_uses_target_fallback = False
        self._fixed_target_prompt: Optional[torch.Tensor] = None
        self._target_context_prompt_state: Optional[Dict[str, Any]] = None
        self._target_prompt_metadata: Dict[str, Any] = {}
        self._stage3_k0_context_shrinkage_enabled = False
        self._stage3_k0_context_shrinkage_rho_cap = 1.0
        self._stage3_k0_context_shrinkage_surface_rho_cap = 1.0
        self._stage3_k0_context_shrinkage_rootzone_rho_cap = 1.0
        self._stage3_k0_context_shrinkage_policy = STAGE3_K0_CONTEXT_SHRINKAGE_VARIABLE_POLICY
        self._stage3_k0_context_shrinkage_last_rho: Optional[float] = None
        self._stage3_k0_context_shrinkage_last_rhos: Dict[str, float] = {}
        self.stage3_k0_context_shrinkage_metadata: Dict[str, Any] = {
            "enabled": False,
            "stage3_variant": STAGE3_K0_CONTEXT_SHRINKAGE_VARIANT,
        }
        state_candidate = checkpoint.get("target_context_prompt_state") or saved_config.get("target_context_prompt_state")
        if state_candidate:
            self.load_target_context_prompt_state(state_candidate)
        elif is_target_adapt and self.method_name in _MAIN_HYPERDA_METHOD_IDS:
            raise ValueError(
                "Paper-facing HyperDA zero/few-shot checkpoints must include "
                "target_context_prompt_state so target_eval inputs cannot update prompts."
            )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise normalization with NaN/Inf guard."""
        if self._ch_mean is None or self._ch_std is None:
            return x
        mean_t = torch.from_numpy(self._ch_mean).to(x.device).view(1, 12, 1, 1)
        std_t = torch.from_numpy(self._ch_std).to(x.device).view(1, 12, 1, 1)
        x_norm = (x - mean_t) / std_t
        if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
            n_nan = torch.isnan(x_norm).sum().item()
            n_inf = torch.isinf(x_norm).sum().item()
            print(f"  WARNING: normalize produced {n_nan} NaN / {n_inf} Inf — returning raw input", flush=True)
            return x
        return x_norm

    def _build_prompt(
        self,
        x_norm: torch.Tensor,
        region_idx: int,
        month_val: int,
        *,
        x_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build prompt vector z, handling unseen target region.

        When the target region is unseen during training (not in source set),
        uses pre-computed mean of source embeddings instead of aliasing a
        wrong source region embedding.
        """
        region_ids = torch.tensor([region_idx], dtype=torch.long, device=x_norm.device)
        month = torch.tensor([month_val], dtype=torch.long, device=x_norm.device)
        x_prompt = prompt_diagnostic_tensor(
            self.prompt_encoder,
            context_encoder=self.context_encoder,
            x_norm=x_norm,
            x_raw=x_raw,
        )

        if not (self._is_target_unseen and self._prompt_route_uses_target_fallback):
            # Target is a source region: use standard prompt encoder forward
            return self.prompt_encoder(x_prompt, region_ids, month)

        # Target region not in source set: manually assemble prompt
        # using mean of source region embeddings
        input_stats = self.prompt_encoder._compute_input_stats(x_prompt)  # [1, C*2]
        i_emb = self.prompt_encoder.input_proj(input_stats)  # [1, 16]
        t_enc = self.prompt_encoder._temporal_encoding(month)  # [1, 2]
        t_emb = self.prompt_encoder.temporal_proj(t_enc)  # [1, 8]

        r_emb = self._target_region_emb.unsqueeze(0).to(x_norm.device)  # [1, 16]

        combined = torch.cat([r_emb, i_emb, t_emb], dim=1)  # [1, 40]
        z = self.prompt_encoder.mlp(combined)  # [1, hidden_dim]
        return z

    @staticmethod
    def _is_source_split(split_role: str) -> bool:
        return split_role in {"source_train", "source_fit", "source_val", "source_test"}

    def _resolve_prompt_region_idx(self, sample: Dict[str, Any]) -> tuple[int, bool]:
        """Return compact prompt id and whether to use held-out target fallback.

        Training uses compact source-region ids (0..Nsource-1). Source split
        evaluation must therefore route by the sample's source region, while
        target splits use the held-out target route.
        """
        split_role = str(sample.get("split_role", ""))
        if self._is_source_split(split_role):
            region_id_str = sample.get("sample_region_id") or sample.get("target_region_id", "")
            global_idx = _REGION_TO_IDX.get(region_id_str, self._target_region_idx)
            if self._source_global_to_prompt_idx:
                if global_idx not in self._source_global_to_prompt_idx:
                    raise ValueError(
                        f"Source split sample_region_id={region_id_str!r} is not in checkpoint "
                        f"source_region_global_indices={sorted(self._source_global_to_prompt_idx)}"
                    )
                return self._source_global_to_prompt_idx[global_idx], False
            return global_idx, False

        region_id_str = sample.get("target_region_id", "")
        return _REGION_TO_IDX.get(region_id_str, self._target_region_idx), True

    @property
    def uses_fixed_target_prompt(self) -> bool:
        return self._target_context_prompt_state is not None or self._fixed_target_prompt is not None

    @property
    def target_context_prompt_state(self) -> Dict[str, Any]:
        if self._target_context_prompt_state is None:
            raise RuntimeError("target_context_prompt_state has not been initialized")
        return normalize_target_context_prompt_state(self._target_context_prompt_state)

    def _target_region_embedding_for_prompt_state(self) -> torch.Tensor:
        if self._is_target_unseen:
            if self._target_region_emb is None:
                raise RuntimeError("Target region fallback embedding was not initialized")
            return self._target_region_emb.unsqueeze(0).to(self.device)
        target_ids = torch.tensor([self._target_region_idx], dtype=torch.long, device=self.device)
        return self.prompt_encoder.region_embed(target_ids)

    def load_target_context_prompt_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        normalized = normalize_target_context_prompt_state(state)
        self._target_context_prompt_state = normalized
        self._fixed_target_prompt = normalized["global_prototype"].unsqueeze(0).to(self.device)
        self._target_prompt_metadata = target_context_prompt_metadata(normalized)
        return dict(self._target_prompt_metadata)

    def compose_target_context_prompt(self, month: int | Sequence[int] | torch.Tensor) -> torch.Tensor:
        if self._target_context_prompt_state is None:
            raise RuntimeError("target_context_prompt_state has not been initialized")
        return compose_target_context_prompt_from_state(self._target_context_prompt_state, month, device=self.device)

    def compose_target_context_reliability_features(self, month: int | Sequence[int] | torch.Tensor) -> torch.Tensor:
        if self._target_context_prompt_state is None:
            raise RuntimeError("target_context_prompt_state has not been initialized")
        normalized = normalize_target_context_prompt_state(self._target_context_prompt_state)
        if isinstance(month, torch.Tensor):
            month_values = [int(v) for v in month.detach().cpu().view(-1).tolist()]
        elif isinstance(month, int):
            month_values = [int(month)]
        else:
            month_values = [int(v) for v in month]
        features = [
            normalized["reliability_features"].get(
                str(_coerce_month(month_value)),
                [0.0] * len(normalized.get("reliability_feature_schema", [])),
            )
            for month_value in month_values
        ]
        return torch.as_tensor(features, dtype=torch.float32, device=self.device)

    def enable_stage3_k0_context_shrinkage(
        self,
        *,
        source_calibrated_rho_cap: float = 1.0,
        policy: str = STAGE3_K0_CONTEXT_SHRINKAGE_VARIABLE_POLICY,
        surface_rho_cap: float | None = None,
        rootzone_rho_cap: float | None = None,
        policy_json_path: str = "",
    ) -> Dict[str, Any]:
        if not hasattr(self.model, "source_base_forward"):
            raise ValueError("Stage 3 K=0 context shrinkage requires a HyperDA model with source_base_forward")
        policy = str(policy or STAGE3_K0_CONTEXT_SHRINKAGE_VARIABLE_POLICY)
        policy_payload: Dict[str, Any] = {}
        if policy_json_path:
            with Path(policy_json_path).open(encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("--stage3_k0_context_shrinkage_policy_json must contain a JSON object")
            policy_payload = dict(loaded)
            policy = str(
                policy_payload.get("policy")
                or policy_payload.get("policy_source")
                or policy
            )
            if policy_payload.get("rho_cap") is not None:
                source_calibrated_rho_cap = float(policy_payload["rho_cap"])
            if policy_payload.get("rho_surface_cap") is not None:
                surface_rho_cap = float(policy_payload["rho_surface_cap"])
            if policy_payload.get("rho_rootzone_cap") is not None:
                rootzone_rho_cap = float(policy_payload["rho_rootzone_cap"])
        if policy not in STAGE3_K0_CONTEXT_SHRINKAGE_POLICIES:
            raise ValueError(f"Unsupported Stage 3 K=0 context shrinkage policy: {policy}")
        cap = _stage3_clean_rho_cap(source_calibrated_rho_cap)
        if float(source_calibrated_rho_cap) != cap:
            raise ValueError("source_calibrated_rho_cap must be in [0, 1]")
        surface_cap = cap if surface_rho_cap is None else _stage3_clean_rho_cap(surface_rho_cap)
        rootzone_cap = cap if rootzone_rho_cap is None else _stage3_clean_rho_cap(rootzone_rho_cap)
        if surface_rho_cap is not None and float(surface_rho_cap) != surface_cap:
            raise ValueError("surface_rho_cap must be in [0, 1]")
        if rootzone_rho_cap is not None and float(rootzone_rho_cap) != rootzone_cap:
            raise ValueError("rootzone_rho_cap must be in [0, 1]")
        if policy == STAGE3_K0_CONTEXT_SHRINKAGE_SCALAR_POLICY:
            surface_cap = cap
            rootzone_cap = cap
        if self._target_context_prompt_state is None:
            raise ValueError("Stage 3 K=0 context shrinkage requires target_context_prompt_state")
        self._stage3_k0_context_shrinkage_enabled = True
        self._stage3_k0_context_shrinkage_rho_cap = cap
        self._stage3_k0_context_shrinkage_surface_rho_cap = surface_cap
        self._stage3_k0_context_shrinkage_rootzone_rho_cap = rootzone_cap
        self._stage3_k0_context_shrinkage_policy = policy
        self._stage3_k0_context_shrinkage_last_rho = None
        self._stage3_k0_context_shrinkage_last_rhos = {}
        policy_source = str(policy_payload.get("policy_source") or policy)
        policy_hash = str(policy_payload.get("policy_hash") or "")
        source_episode_regions = [
            str(region)
            for region in policy_payload.get("source_episode_regions", [])
            if str(region)
        ]
        self.stage3_k0_context_shrinkage_metadata = {
            "enabled": True,
            "stage3_variant": STAGE3_K0_CONTEXT_SHRINKAGE_VARIANT,
            "policy": policy,
            "legacy_policy_alias": "target_context_input_reliability_posthoc_residual_shrinkage",
            "rho_cap": cap,
            "rho_surface_cap": surface_cap,
            "rho_rootzone_cap": rootzone_cap,
            "rho_selection": "target_context_input_reliability_source_calibrated_variable_caps",
            "policy_source": policy_source,
            "policy_hash": policy_hash,
            "source_episode_regions": source_episode_regions,
            "policy_json_path": str(policy_json_path or ""),
            "source_prior": "M2_1_rank_gated_dora_stable",
            "extra_source_finetune": False,
            "target_context_signal": "input_side_monthly_prototype_reliability_only",
            "target_labels_used_for_adaptation": False,
            "target_val_usage": "unused_in_main_protocol",
            "target_eval_usage": "final_eval_only_no_selection",
            "target_eval_input_stats_used_for_update": False,
        }
        return dict(self.stage3_k0_context_shrinkage_metadata)

    @property
    def stage3_k0_context_shrinkage_last_rho(self) -> Optional[float]:
        return self._stage3_k0_context_shrinkage_last_rho

    @property
    def stage3_k0_context_shrinkage_last_rhos(self) -> Dict[str, float]:
        return dict(self._stage3_k0_context_shrinkage_last_rhos)

    def set_target_context_prompt_from_samples(self, samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Build monthly target-context prompt prototypes from input-side fields only.

        The prompt summary reads only ``x``, ``month``, and ``date_str``. It
        deliberately does not read target analysis or increment labels.
        """
        state = build_target_context_prompt_state(
            samples=samples,
            prompt_encoder=self.prompt_encoder,
            normalize_x=self._normalize,
            target_region_embedding=self._target_region_embedding_for_prompt_state(),
            device=self.device,
            context_encoder=self.context_encoder,
        )
        return self.load_target_context_prompt_state(state)

    def set_target_prompt_from_samples(self, samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Legacy alias for older target_train prompt call sites."""
        metadata = self.set_target_context_prompt_from_samples(samples)
        metadata["legacy_alias"] = "set_target_prompt_from_samples"
        self._target_prompt_metadata = dict(metadata)
        return metadata

    def _model_forward(
        self,
        x_norm: torch.Tensor,
        z: torch.Tensor,
        *,
        month: torch.Tensor,
        x_raw: torch.Tensor,
        reliability_features: torch.Tensor | None,
    ) -> torch.Tensor:
        parameters = inspect.signature(self.model.forward).parameters
        kwargs: Dict[str, Any] = {}
        if self._requires_month:
            kwargs["month"] = month
        if self._requires_month and "x_raw" in parameters:
            kwargs["x_raw"] = x_raw
        if "reliability_features" in parameters and reliability_features is not None:
            kwargs["reliability_features"] = reliability_features
        return self.model(x_norm, z, **kwargs)

    def predict(self, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Predict DA increments and analysis for a single sample with prompt conditioning.

        Args:
            sample: dict with keys:
                - x: raw input array [12, H, W]
                - forecast_surface: [H, W]
                - forecast_rootzone: [H, W]
                - target_region_id: str (e.g. "US-R1")
                - month: int (1-12)
                - (optional) date_str, metric_mask, etc.

        Returns:
            dict with pred_increment_*, pred_analysis_*
        """
        x = torch.from_numpy(np.asarray(sample["x"], dtype=np.float32))
        x = x.unsqueeze(0).to(self.device)  # [1, 12, H, W]

        x_norm = self._normalize(x)

        # Build prompt. Source splits route by sample_region_id; target splits
        # route by target_region_id and use held-out target fallback when needed.
        region_idx, use_target_fallback = self._resolve_prompt_region_idx(sample)
        month_val = int(sample.get("month", 6))

        with torch.no_grad():
            if use_target_fallback and self._target_context_prompt_state is not None:
                z = compose_target_context_prompt_from_state(
                    self._target_context_prompt_state,
                    month_val,
                    device=x_norm.device,
                )
                reliability_features = self.compose_target_context_reliability_features(month_val).to(x_norm.device)
            elif use_target_fallback and self._fixed_target_prompt is not None:
                z = self._fixed_target_prompt.to(x_norm.device)
                reliability_features = None
            else:
                try:
                    self._prompt_route_uses_target_fallback = use_target_fallback
                    build_prompt_params = inspect.signature(self._build_prompt).parameters
                    if "x_raw" in build_prompt_params:
                        z = self._build_prompt(x_norm, region_idx, month_val, x_raw=x)
                    else:
                        z = self._build_prompt(x_norm, region_idx, month_val)
                finally:
                    self._prompt_route_uses_target_fallback = False
                reliability_features = None
            month_tensor = torch.tensor([month_val], dtype=torch.long, device=x_norm.device)
            pred = self._model_forward(
                x_norm,
                z,
                month=month_tensor,
                x_raw=x,
                reliability_features=reliability_features,
            )  # [1, 2, H, W]
            if self._stage3_k0_context_shrinkage_enabled:
                if reliability_features is None:
                    raise RuntimeError("Stage 3 K=0 context shrinkage requires target_context reliability features")
                features_np = reliability_features.detach().cpu().view(-1).tolist()
                rhos = compute_stage3_k0_context_shrinkage_rhos(
                    features_np,
                    source_calibrated_rho_cap=self._stage3_k0_context_shrinkage_rho_cap,
                    surface_rho_cap=self._stage3_k0_context_shrinkage_surface_rho_cap,
                    rootzone_rho_cap=self._stage3_k0_context_shrinkage_rootzone_rho_cap,
                    policy=self._stage3_k0_context_shrinkage_policy,
                )
                source_base = self.model.source_base_forward(x_norm)
                rho_tensor = torch.tensor(
                    [float(rhos["surface"]), float(rhos["rootzone"])],
                    dtype=pred.dtype,
                    device=pred.device,
                ).view(1, 2, 1, 1)
                pred = source_base + rho_tensor * (pred - source_base)
                mean_rho = float((float(rhos["surface"]) + float(rhos["rootzone"])) / 2.0)
                self._stage3_k0_context_shrinkage_last_rho = mean_rho
                self._stage3_k0_context_shrinkage_last_rhos = {
                    "surface": float(rhos["surface"]),
                    "rootzone": float(rhos["rootzone"]),
                }
                self.stage3_k0_context_shrinkage_metadata = {
                    **self.stage3_k0_context_shrinkage_metadata,
                    "last_rho": mean_rho,
                    "last_rho_surface": float(rhos["surface"]),
                    "last_rho_rootzone": float(rhos["rootzone"]),
                    "last_month": int(month_val),
                    "last_reliability_features": [float(v) for v in features_np],
                }

        pred_inc_s = pred[0, 0].cpu().numpy().astype(np.float32)
        pred_inc_r = pred[0, 1].cpu().numpy().astype(np.float32)

        forecast_surface = np.asarray(sample["forecast_surface"], dtype=np.float32)
        forecast_rootzone = np.asarray(sample["forecast_rootzone"], dtype=np.float32)

        # Denormalize increments if needed
        if self._has_inc_norm:
            pred_inc_s = pred_inc_s * self._inc_std[0] + self._inc_mean[0]
            pred_inc_r = pred_inc_r * self._inc_std[1] + self._inc_mean[1]

        # Apply residual gain before returning so public outputs satisfy
        # pred_analysis = forecast + pred_increment.
        if self.apply_residual_gain:
            alpha_s = self.alpha_surface
            alpha_r = self.alpha_rootzone
        else:
            alpha_s = 1.0
            alpha_r = 1.0

        pred_inc_s = (alpha_s * pred_inc_s).astype(np.float32)
        pred_inc_r = (alpha_r * pred_inc_r).astype(np.float32)

        pred_analysis_surface = (forecast_surface + pred_inc_s).astype(np.float32)
        pred_analysis_rootzone = (forecast_rootzone + pred_inc_r).astype(np.float32)

        return {
            "pred_increment_surface": pred_inc_s,
            "pred_increment_rootzone": pred_inc_r,
            "pred_analysis_surface": pred_analysis_surface,
            "pred_analysis_rootzone": pred_analysis_rootzone,
            "residual_gain_alpha_surface": alpha_s,
            "residual_gain_alpha_rootzone": alpha_r,
        }
