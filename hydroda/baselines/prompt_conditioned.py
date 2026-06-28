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
from hydroda.models.hyper_conditional_unet import (
    LEGACY_SOURCE_MANIFOLD_DISTANCE_KEY,
    SOURCE_MANIFOLD_DISTANCE_KEY,
    SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA,
)
from hydroda.models.phys_trust import (
    PHYS_CONSISTENCY_GUARD_MODE,
    PHYS_CONSISTENCY_GUARD_PRODUCT_MODE,
    PHYS_CONSISTENCY_SOURCE,
    PHYS_FORMULA_MODE,
    PHYS_FORMULA_SOURCE,
    phys_consistency_guard_from_raw_tensor,
    phys_formula_features_from_raw_tensor,
    phys_trust_d0_diagnostics_from_tensor,
    phys_trust_d0_summary_from_monthly_rows,
)
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
CONTEXT_TTA_NONE = "none"
CONTEXT_TTA_PROMPT_FEATURE_ALIGNMENT = "prompt_feature_alignment"
CONTEXT_TTA_CONTEXT_PROMPT_RESIDUAL_SHIFT = "context_prompt_residual_shift"
CONTEXT_TTA_MODES = {
    CONTEXT_TTA_NONE,
    CONTEXT_TTA_PROMPT_FEATURE_ALIGNMENT,
    CONTEXT_TTA_CONTEXT_PROMPT_RESIDUAL_SHIFT,
}
CONTEXT_TTA_ALIGNMENT_SCHEMA_VERSION = "target_context_prompt_feature_alignment_v1"
CONTEXT_TTA_RESIDUAL_SHIFT_SCHEMA_VERSION = "target_context_prompt_residual_shift_v1"
_FORBIDDEN_CONTEXT_TTA_SAMPLE_KEYS = (
    "analysis",
    "increment",
    "label",
    "residual",
    "loss",
    "metric",
    "score",
)
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
SOURCE_PROMPT_MANIFOLD_GUARD_SCHEMA_VERSION = "source_prompt_manifold_guard_state_v1"
HYPERDA_SOURCE_TRUST_BANK_SCHEMA_VERSION = "hyperda_source_trust_bank_v1"
SOURCE_MANIFOLD_DISTANCE_SCHEMA = {
    "distance_key": SOURCE_MANIFOLD_DISTANCE_KEY,
    "legacy_distance_key": LEGACY_SOURCE_MANIFOLD_DISTANCE_KEY,
    "distance_space": "prompt_encoder_input_embedding",
    "distance": "nearest_source_region_month_input_embedding_l2",
    "normalization": "source_side_quantile_bounded",
    "bounded_range": [0.0, 1.0],
}
HYPERDA_TRUST_BANK_METADATA_DEFAULTS = {
    "source": "source_fit_source_val_only",
    "label_usage": "none",
    "target_eval_usage": "final_eval_only_no_selection",
}
SOURCE_TRUST_QUERY_MODE_PROMPT = "prompt_embedding"
SOURCE_TRUST_QUERY_MODE_RAW_DA = "raw_input_side_da_diagnostics"
SOURCE_TRUST_QUERY_MODE_BLENDED_RAW_DA = "blended_prompt_raw_da_0p25"
SOURCE_TRUST_QUERY_BLEND_LAMBDA = 0.25


def source_trust_query_requires_separate_bank(mode: str) -> bool:
    return str(mode) in {
        SOURCE_TRUST_QUERY_MODE_RAW_DA,
        SOURCE_TRUST_QUERY_MODE_BLENDED_RAW_DA,
    }


def source_trust_query_input_domain(mode: str, context_encoder: str) -> str:
    if str(mode) == SOURCE_TRUST_QUERY_MODE_RAW_DA:
        return "raw_input_side"
    if str(mode) == SOURCE_TRUST_QUERY_MODE_BLENDED_RAW_DA:
        return "blended_prompt_raw_input_side"
    return prompt_diagnostic_input_domain(context_encoder)


def blend_prompt_and_raw_trust_query(
    prompt: torch.Tensor,
    raw_query: torch.Tensor,
    *,
    blend_lambda: float = SOURCE_TRUST_QUERY_BLEND_LAMBDA,
) -> torch.Tensor:
    """Blend only the trust-neighbor query, leaving the main prompt unchanged."""
    lam = float(blend_lambda)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("source trust query blend lambda must be in [0, 1]")
    return (1.0 - lam) * prompt + lam * raw_query


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


def bounded_source_manifold_distance(distance: float, state: Dict[str, Any] | None) -> float:
    """Normalize a raw source-manifold distance with source-side quantiles."""
    value = float(np.nan_to_num(float(distance), nan=0.0, posinf=1.0, neginf=0.0))
    value = max(0.0, value)
    if not state:
        return _bounded_distance_feature(value)
    scale = float(
        state.get("distance_scale")
        or state.get("distance_quantiles", {}).get("q90")
        or state.get("distance_quantiles", {}).get("q75")
        or 1.0
    )
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return float(np.clip(value / scale, 0.0, 1.0))


def bounded_reliability_features(
    *,
    monthly_count: float,
    has_monthly_prototype: float,
    global_context_count: float,
    finite_input_coverage: float,
    source_manifold_distance_bounded: float | None = None,
    prompt_to_source_manifold_distance: float | None = None,
) -> list[float]:
    """Return finite bounded reliability features in the model schema order."""
    if source_manifold_distance_bounded is None:
        source_manifold_distance_bounded = _bounded_distance_feature(
            0.0 if prompt_to_source_manifold_distance is None else prompt_to_source_manifold_distance
        )
    features = [
        _bounded_count_feature(monthly_count),
        1.0 if float(has_monthly_prototype) > 0.0 else 0.0,
        _bounded_context_count_feature(global_context_count),
        float(np.clip(finite_input_coverage, 0.0, 1.0)),
        float(np.clip(source_manifold_distance_bounded, 0.0, 1.0)),
    ]
    return [float(np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)) for value in features]


def _source_manifold_tensor(value: Any) -> Optional[torch.Tensor]:
    tensor = _prompt_tensor(value)
    if tensor is None:
        return None
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.detach().cpu().to(dtype=torch.float32)


def _sha256_update_tensor(digest: "hashlib._Hash", tensor: torch.Tensor) -> None:
    cpu = tensor.detach().cpu().contiguous().to(dtype=torch.float32)
    digest.update(str(tuple(cpu.shape)).encode("utf-8"))
    digest.update(cpu.numpy().tobytes())


def _hash_source_trust_bank_state(state: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(state.get("schema_version", HYPERDA_SOURCE_TRUST_BANK_SCHEMA_VERSION)).encode("utf-8"))
    for key in ("source", "label_usage", "target_eval_usage", "target_val_usage", "coefficient_source"):
        digest.update(str(state.get(key, "")).encode("utf-8"))
    _sha256_update_tensor(digest, state["source_prompt_embeddings"])
    trust_query = state.get("source_trust_query_embeddings")
    if isinstance(trust_query, torch.Tensor) and trust_query.numel() > 0:
        digest.update(str(state.get("source_trust_query_mode", "")).encode("utf-8"))
        _sha256_update_tensor(digest, trust_query)
    for layer_name in sorted(state["layer_consensus_logits"]):
        digest.update(layer_name.encode("utf-8"))
        _sha256_update_tensor(digest, state["layer_consensus_logits"][layer_name])
    digest.update(
        json.dumps(
            state.get("distance_quantiles", {}),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    for key in ("prompt_distance_quantiles", "source_trust_query_distance_quantiles"):
        digest.update(
            json.dumps(
                state.get(key, {}),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
    return digest.hexdigest()


def normalize_hyperda_source_trust_bank_state(state: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Return a CPU tensor source trust bank, or None for old checkpoints."""
    if not state:
        return None
    normalized = dict(state)
    normalized["schema_version"] = str(
        normalized.get("schema_version") or HYPERDA_SOURCE_TRUST_BANK_SCHEMA_VERSION
    )
    source_emb = _source_manifold_tensor(normalized.get("source_prompt_embeddings"))
    if source_emb is None or source_emb.numel() == 0:
        return None
    if source_emb.ndim != 2:
        raise ValueError("source trust bank source_prompt_embeddings must have shape [N, D]")
    layer_raw = normalized.get("layer_consensus_logits", {})
    if not isinstance(layer_raw, dict) or not layer_raw:
        return None
    layer_consensus: Dict[str, torch.Tensor] = {}
    for layer_name, value in layer_raw.items():
        tensor = _source_manifold_tensor(value)
        if tensor is None:
            continue
        if tensor.ndim != 2:
            raise ValueError(f"source trust bank layer {layer_name!r} consensus must have shape [N, M]")
        if tensor.shape[0] != source_emb.shape[0]:
            raise ValueError(
                f"source trust bank layer {layer_name!r} row count {tensor.shape[0]} "
                f"does not match source_prompt_embeddings {source_emb.shape[0]}"
            )
        layer_consensus[str(layer_name)] = tensor
    if not layer_consensus:
        return None
    normalized["source_prompt_embeddings"] = source_emb
    query_mode = str(normalized.get("source_trust_query_mode") or SOURCE_TRUST_QUERY_MODE_PROMPT)
    trust_query_emb = _source_manifold_tensor(normalized.get("source_trust_query_embeddings"))
    if trust_query_emb is not None and trust_query_emb.numel() > 0:
        if trust_query_emb.ndim != 2:
            raise ValueError("source trust bank source_trust_query_embeddings must have shape [N, D]")
        if trust_query_emb.shape != source_emb.shape:
            raise ValueError(
                "source trust bank source_trust_query_embeddings shape must match "
                f"source_prompt_embeddings: query={tuple(trust_query_emb.shape)} source={tuple(source_emb.shape)}"
            )
        normalized["source_trust_query_embeddings"] = trust_query_emb
    else:
        normalized.pop("source_trust_query_embeddings", None)
        if source_trust_query_requires_separate_bank(query_mode):
            raise ValueError(
                f"source_trust_query_mode={query_mode} requires "
                "source_trust_query_embeddings in the source trust bank; refusing prompt-space fallback"
            )
    normalized["layer_consensus_logits"] = layer_consensus
    normalized["distance_quantiles"] = {
        str(key): float(value)
        for key, value in dict(normalized.get("distance_quantiles", {})).items()
    }
    normalized["prompt_distance_quantiles"] = {
        str(key): float(value)
        for key, value in dict(
            normalized.get("prompt_distance_quantiles")
            or normalized.get("distance_quantiles", {})
        ).items()
    }
    normalized["source_trust_query_distance_quantiles"] = {
        str(key): float(value)
        for key, value in dict(
            normalized.get("source_trust_query_distance_quantiles")
            or (
                normalized.get("distance_quantiles", {})
                if trust_query_emb is not None and trust_query_emb.numel() > 0
                else {}
            )
        ).items()
    }
    q75 = float(
        normalized.get("distance_temperature")
        or normalized.get("distance_quantiles", {}).get("q75")
        or normalized.get("distance_quantiles", {}).get("q90")
        or 1.0
    )
    if not np.isfinite(q75) or q75 <= 0.0:
        q75 = 1.0
    normalized["distance_temperature"] = q75
    normalized.setdefault("source_neighbor_top_m", 4)
    normalized.setdefault("trust_strength", 0.0)
    normalized["source_trust_query_mode"] = query_mode
    normalized.setdefault("target_val_usage", "unused_in_main_protocol")
    normalized.setdefault("coefficient_source", "source_neighborhood_consensus")
    for key, value in HYPERDA_TRUST_BANK_METADATA_DEFAULTS.items():
        normalized.setdefault(key, value)
    normalized["source_region_month_count"] = int(source_emb.shape[0])
    normalized["trust_bank_hash"] = str(
        normalized.get("trust_bank_hash") or _hash_source_trust_bank_state(normalized)
    )
    return normalized


def hyperda_trust_bank_summary(state: Dict[str, Any] | None) -> Dict[str, Any]:
    normalized = normalize_hyperda_source_trust_bank_state(state)
    if not normalized:
        return {
            "enabled": False,
            "schema_version": HYPERDA_SOURCE_TRUST_BANK_SCHEMA_VERSION,
            "source": "disabled_or_old_checkpoint",
        }
    return {
        "enabled": True,
        "schema_version": normalized.get("schema_version", HYPERDA_SOURCE_TRUST_BANK_SCHEMA_VERSION),
        "source": normalized.get("source", "source_fit_source_val_only"),
        "label_usage": normalized.get("label_usage", "none"),
        "target_val_usage": normalized.get("target_val_usage", "unused_in_main_protocol"),
        "target_eval_usage": normalized.get("target_eval_usage", "final_eval_only_no_selection"),
        "trust_bank_hash": normalized.get("trust_bank_hash", ""),
        "source_neighbor_top_m": int(normalized.get("source_neighbor_top_m", 4)),
        "trust_strength": float(normalized.get("trust_strength", 0.0)),
        "source_trust_query_mode": normalized.get("source_trust_query_mode", "prompt_embedding"),
        "has_separate_source_trust_query": bool(
            isinstance(normalized.get("source_trust_query_embeddings"), torch.Tensor)
            and normalized["source_trust_query_embeddings"].numel() > 0
        ),
        "distance_temperature": float(normalized.get("distance_temperature", 1.0)),
        "source_region_month_count": int(normalized.get("source_region_month_count", 0)),
        "layers": sorted(normalized.get("layer_consensus_logits", {}).keys()),
    }


def hyperda_trust_summary_for_embedding(
    input_embedding: torch.Tensor,
    trust_bank_state: Dict[str, Any] | None,
    *,
    top_m: int | None = None,
) -> Dict[str, Any]:
    bank = normalize_hyperda_source_trust_bank_state(trust_bank_state)
    if not bank:
        return {
            "enabled": False,
            "nearest_distance": 0.0,
            "nearest_distance_bounded": 0.0,
            "source_neighbor_top_m": 0,
            "nearest_neighbor_indices": [],
        }
    emb = _prompt_tensor(input_embedding)
    if emb is None:
        raise ValueError("input_embedding is required for HyperDA trust summary")
    emb = emb.detach().cpu().to(dtype=torch.float32).view(1, -1)
    source_emb = (
        bank.get("source_trust_query_embeddings")
        if source_trust_query_requires_separate_bank(str(bank.get("source_trust_query_mode")))
        and isinstance(bank.get("source_trust_query_embeddings"), torch.Tensor)
        else bank["source_prompt_embeddings"]
    ).to(dtype=torch.float32)
    if source_emb.shape[1] != emb.shape[1]:
        raise ValueError(
            "HyperDA trust bank embedding dimension mismatch: "
            f"source={tuple(source_emb.shape)} target={tuple(emb.shape)}"
        )
    resolved_top_m = min(int(top_m or bank.get("source_neighbor_top_m", 4)), int(source_emb.shape[0]))
    distances = torch.linalg.vector_norm(source_emb - emb, dim=1)
    nearest_distances, nearest_indices = torch.topk(distances, k=resolved_top_m, largest=False)
    temperature = float(bank.get("distance_temperature", 1.0))
    if not np.isfinite(temperature) or temperature <= 0.0:
        temperature = 1.0
    bounded = float(np.clip(float(nearest_distances[0].item()) / temperature, 0.0, 1.0))
    return {
        "enabled": True,
        "nearest_distance": float(nearest_distances[0].item()),
        "nearest_distance_bounded": bounded,
        "source_neighbor_top_m": int(resolved_top_m),
        "nearest_neighbor_indices": [int(v) for v in nearest_indices.tolist()],
        "trust_strength": float(bank.get("trust_strength", 0.0)),
        "distance_temperature": temperature,
        "trust_bank_hash": str(bank.get("trust_bank_hash", "")),
    }


def normalize_source_prompt_manifold_guard_state(state: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Return a CPU tensor source-manifold guard state, or None for old checkpoints."""
    if not state:
        return None
    normalized = dict(state)
    normalized["schema_version"] = str(
        normalized.get("schema_version") or SOURCE_PROMPT_MANIFOLD_GUARD_SCHEMA_VERSION
    )
    normalized["distance_key"] = str(normalized.get("distance_key") or SOURCE_MANIFOLD_DISTANCE_KEY)
    normalized["distance_schema"] = dict(normalized.get("distance_schema") or SOURCE_MANIFOLD_DISTANCE_SCHEMA)
    source_value = normalized.get("source_region_month_input_embeddings")
    if source_value is None:
        source_value = normalized.get("source_input_embeddings")
    if source_value is None:
        source_value = normalized.get("source_region_month_embeddings")
    source_emb = _source_manifold_tensor(source_value)
    if source_emb is None or source_emb.numel() == 0:
        return None
    normalized["source_region_month_input_embeddings"] = source_emb
    global_emb = _prompt_tensor(normalized.get("global_input_embedding"))
    normalized["global_input_embedding"] = (
        global_emb.detach().cpu().to(dtype=torch.float32)
        if global_emb is not None
        else source_emb.mean(dim=0)
    )
    distance_quantiles = {
        str(key): float(value)
        for key, value in dict(normalized.get("distance_quantiles", {})).items()
    }
    normalized["distance_quantiles"] = distance_quantiles
    distance_scale = float(
        normalized.get("distance_scale")
        or distance_quantiles.get("q90")
        or distance_quantiles.get("q75")
        or 1.0
    )
    if not np.isfinite(distance_scale) or distance_scale <= 0.0:
        distance_scale = 1.0
    normalized["distance_scale"] = distance_scale
    normalized.setdefault("source", "source_fit_source_val_only")
    normalized.setdefault("calibration_source", "source_fit_source_val_only")
    return normalized


def source_manifold_distance_for_embedding(
    input_embedding: torch.Tensor,
    guard_state: Dict[str, Any] | None,
) -> Dict[str, float]:
    """Compute nearest-source raw and bounded manifold distance for one input embedding."""
    state = normalize_source_prompt_manifold_guard_state(guard_state)
    if state is None:
        return {"raw": 0.0, "bounded": 0.0}
    emb = _prompt_tensor(input_embedding)
    if emb is None:
        return {"raw": 0.0, "bounded": 0.0}
    emb = emb.detach().cpu().to(dtype=torch.float32)
    if emb.ndim == 2 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    source_emb = state["source_region_month_input_embeddings"].to(dtype=emb.dtype)
    if source_emb.ndim != 2 or int(source_emb.shape[1]) != int(emb.numel()):
        raise ValueError(
            "source manifold embedding dimension mismatch: "
            f"source={tuple(source_emb.shape)} target={tuple(emb.shape)}"
        )
    distances = torch.linalg.vector_norm(source_emb - emb.view(1, -1), dim=1)
    raw = float(distances.min().item()) if distances.numel() else 0.0
    bounded = bounded_source_manifold_distance(raw, state)
    return {"raw": raw, "bounded": bounded}


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


def compose_prompt_from_input_embedding(
    prompt_encoder: RegionPromptEncoder,
    *,
    region_embedding: torch.Tensor,
    input_embedding: torch.Tensor,
    month: torch.Tensor,
) -> torch.Tensor:
    """Compose a prompt vector from precomputed input-branch embeddings."""
    if region_embedding.ndim == 1:
        region_embedding = region_embedding.unsqueeze(0)
    temporal = prompt_encoder._temporal_encoding(month)
    temporal_emb = prompt_encoder.temporal_proj(temporal)
    return prompt_encoder.mlp(torch.cat([region_embedding, input_embedding, temporal_emb], dim=1))


def raw_da_trust_input_embedding(
    prompt_encoder: RegionPromptEncoder,
    x_raw: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Project raw input-side DA diagnostics through the trained input branch."""
    raw_encoder = RobustInputSideDAPromptEncoder(
        num_regions=int(prompt_encoder.num_regions),
        input_channels=12,
        hidden_dim=int(prompt_encoder.hidden_dim),
    ).to(device=x_raw.device)
    input_stats = _masked_input_stats_from_tensor(raw_encoder, x_raw, mask)
    return prompt_encoder.input_proj(input_stats)


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
    monthly_trust_raw = state.get("monthly_source_trust_query_prototypes", {})
    monthly_trust_query_prototypes = {
        str(month): _prompt_tensor(monthly_trust_raw.get(str(month)))
        for month in range(1, 13)
    }
    monthly_phys_raw = state.get("monthly_phys_context_prototypes", {})
    monthly_phys_context_prototypes = {
        str(month): _prompt_tensor(monthly_phys_raw.get(str(month)))
        for month in range(1, 13)
    }
    global_prototype = _prompt_tensor(state.get("global_prototype"))
    if global_prototype is None:
        raise ValueError("target_context_prompt_state missing global_prototype")
    global_trust_query_prototype = _prompt_tensor(state.get("global_source_trust_query_prototype"))
    global_phys_context_prototype = _prompt_tensor(state.get("global_phys_context_prototype"))

    normalized = dict(state)
    normalized["schema_version"] = TARGET_CONTEXT_PROMPT_SCHEMA_VERSION
    normalized["prompt_source"] = TARGET_CONTEXT_PROMPT_SOURCE
    normalized["label_usage"] = "none"
    normalized["monthly_counts"] = monthly_counts
    normalized["reliability_feature_schema"] = list(
        state.get("reliability_feature_schema") or SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA
    )
    if LEGACY_SOURCE_MANIFOLD_DISTANCE_KEY in normalized["reliability_feature_schema"]:
        normalized["reliability_feature_schema"] = [
            SOURCE_MANIFOLD_DISTANCE_KEY if key == LEGACY_SOURCE_MANIFOLD_DISTANCE_KEY else key
            for key in normalized["reliability_feature_schema"]
        ]
    normalized["reliability_features"] = reliability_features
    normalized["monthly_prototypes"] = monthly_prototypes
    normalized["monthly_source_trust_query_prototypes"] = monthly_trust_query_prototypes
    normalized["monthly_phys_context_prototypes"] = monthly_phys_context_prototypes
    normalized["global_prototype"] = global_prototype.detach().cpu()
    normalized["global_source_trust_query_prototype"] = (
        global_trust_query_prototype.detach().cpu()
        if global_trust_query_prototype is not None
        else None
    )
    normalized["global_phys_context_prototype"] = (
        global_phys_context_prototype.detach().cpu()
        if global_phys_context_prototype is not None
        else None
    )
    if "phys_trust_d0_summary" in state:
        normalized["phys_trust_d0_summary"] = dict(state.get("phys_trust_d0_summary") or {})
    context_hash = str(normalized.get("context_hash") or normalized.get("context_date_hash") or "")
    normalized["context_hash"] = context_hash
    normalized["context_date_hash"] = context_hash
    context_tta_mode = str(normalized.get("context_tta_mode") or CONTEXT_TTA_NONE)
    if context_tta_mode not in CONTEXT_TTA_MODES:
        raise ValueError(f"Unsupported context_tta_mode={context_tta_mode!r}")
    context_tta_state = normalize_context_tta_state(normalized.get("context_tta_state"))
    context_tta_state_hash = str(
        normalized.get("context_tta_state_hash")
        or (context_tta_state.get("state_hash", context_tta_state.get("alignment_hash", "")) if context_tta_state else "")
    )
    context_tta_effective = bool(
        normalized.get(
            "context_tta_effective",
            context_tta_state.get("effective", False) if isinstance(context_tta_state, dict) else False,
        )
    )
    context_tta_source_stat_status = str(
        normalized.get(
            "context_tta_source_stat_status",
            context_tta_state.get("source_stat_status", "not_requested") if isinstance(context_tta_state, dict) else "not_requested",
        )
    )
    prompt_l2_delta_mean = float(normalized.get("prompt_l2_delta_mean", 0.0) or 0.0)
    normalized["context_tta_mode"] = context_tta_mode
    normalized["context_tta_state"] = context_tta_state
    normalized["context_tta_state_hash"] = context_tta_state_hash
    normalized["context_tta_label_usage"] = "none"
    normalized["context_tta_effective"] = context_tta_effective
    normalized["context_tta_source_stat_status"] = context_tta_source_stat_status
    normalized["prompt_l2_delta_mean"] = prompt_l2_delta_mean
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
    normalized["metadata"].setdefault("source_manifold_distance_schema", SOURCE_MANIFOLD_DISTANCE_SCHEMA)
    normalized["metadata"].setdefault("source_manifold_guard_source", "disabled_or_old_checkpoint")
    normalized["metadata"].setdefault("source_trust_query_mode", SOURCE_TRUST_QUERY_MODE_PROMPT)
    normalized["metadata"].setdefault("context_tta_mode", context_tta_mode)
    normalized["metadata"].setdefault("context_tta_state_hash", context_tta_state_hash)
    normalized["metadata"].setdefault("context_tta_label_usage", "none")
    normalized["metadata"].setdefault("context_tta_effective", context_tta_effective)
    normalized["metadata"].setdefault("context_tta_source_stat_status", context_tta_source_stat_status)
    normalized["metadata"].setdefault("prompt_l2_delta_mean", prompt_l2_delta_mean)
    normalized["metadata"].setdefault(
        "source_trust_query_input_domain",
        prompt_diagnostic_input_domain(context_encoder),
    )
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


def compose_target_context_source_trust_query_from_state(
    state: Dict[str, Any],
    months: int | Sequence[int] | torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    """Select frozen monthly target-context trust query prototypes, if present."""
    normalized = normalize_target_context_prompt_state(state)
    metadata = normalized.get("metadata", {})
    if not source_trust_query_requires_separate_bank(str(metadata.get("source_trust_query_mode"))):
        return None
    if isinstance(months, torch.Tensor):
        month_values = [int(v) for v in months.detach().cpu().view(-1).tolist()]
    elif isinstance(months, int):
        month_values = [int(months)]
    else:
        month_values = [int(v) for v in months]

    prompts = []
    for month in month_values:
        month = _coerce_month(month)
        prompt = normalized["monthly_source_trust_query_prototypes"].get(str(month))
        if prompt is None:
            prompt = normalized.get("global_source_trust_query_prototype")
        if prompt is None:
            return None
        prompts.append(prompt.to(device=device) if device is not None else prompt)
    return torch.stack(prompts, dim=0)


def compose_target_context_phys_token_from_state(
    state: Dict[str, Any],
    months: int | Sequence[int] | torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    """Select frozen monthly target-context physical-token prototypes."""
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
        prompt = normalized["monthly_phys_context_prototypes"].get(str(month))
        if prompt is None:
            prompt = normalized.get("global_phys_context_prototype")
        if prompt is None:
            return None
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
            "phys_trust_d0_summary": dict(normalized.get("phys_trust_d0_summary", {})),
            "phys_trust_d0_usage": normalized.get("metadata", {}).get(
                "phys_trust_d0_usage",
                "diagnostic_only_not_model_selection",
            ),
            "context_tta_mode": normalized.get("context_tta_mode", CONTEXT_TTA_NONE),
            "context_tta_state_hash": normalized.get("context_tta_state_hash", ""),
            "context_tta_label_usage": normalized.get("context_tta_label_usage", "none"),
            "context_tta_effective": bool(normalized.get("context_tta_effective", False)),
            "context_tta_source_stat_status": normalized.get("context_tta_source_stat_status", "not_requested"),
            "prompt_l2_delta_mean": float(normalized.get("prompt_l2_delta_mean", 0.0) or 0.0),
        }
    )
    context_tta_state = normalized.get("context_tta_state")
    if isinstance(context_tta_state, dict):
        metadata["context_tta_state"] = {
            "schema_version": context_tta_state.get("schema_version", ""),
            "mode": context_tta_state.get("mode", ""),
            "label_usage": context_tta_state.get("label_usage", "none"),
            "target_val_usage": context_tta_state.get("target_val_usage", "unused_in_main_protocol"),
            "target_eval_usage": context_tta_state.get("target_eval_usage", "final_eval_only_no_selection"),
            "eval_update_policy": context_tta_state.get("eval_update_policy", ""),
            "source_statistics_provenance": context_tta_state.get("source_statistics_provenance", ""),
            "source_stat_status": context_tta_state.get("source_stat_status", ""),
            "effective": bool(context_tta_state.get("effective", False)),
            "target_context_date_hash": context_tta_state.get("target_context_date_hash", ""),
            "alignment_hash": context_tta_state.get("alignment_hash", ""),
            "state_hash": context_tta_state.get("state_hash", context_tta_state.get("alignment_hash", "")),
            "prompt_l2_delta_mean": float(context_tta_state.get("prompt_l2_delta_mean", 0.0) or 0.0),
            "input_embedding_dim": int(context_tta_state.get("input_embedding_dim", 0) or 0),
            "n_samples": int(context_tta_state.get("n_samples", 0) or 0),
        }
    return metadata


def target_context_prompt_state_without_context_tta(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a no-TTA prompt state for residual-shift diagnostics.

    This inverts only the explicit bounded residual added to prompt prototypes.
    It does not use target_eval/query samples or labels.
    """
    normalized = normalize_target_context_prompt_state(state)
    mode = str(normalized.get("context_tta_mode") or CONTEXT_TTA_NONE)
    if mode == CONTEXT_TTA_NONE:
        return normalized
    if mode != CONTEXT_TTA_CONTEXT_PROMPT_RESIDUAL_SHIFT:
        raise ValueError(
            "no-TTA prompt-state reconstruction is only supported for "
            f"{CONTEXT_TTA_CONTEXT_PROMPT_RESIDUAL_SHIFT}; got {mode!r}"
        )
    context_tta_state = normalize_context_tta_residual_shift_state(normalized.get("context_tta_state"))
    if context_tta_state is None:
        raise ValueError("residual-shift context_tta_state is required for no-TTA reconstruction")
    residual = context_tta_state["residual_prompt_shift"].detach().cpu().to(dtype=torch.float32)

    def remove_residual(prompt: torch.Tensor | None) -> torch.Tensor | None:
        if prompt is None:
            return None
        shifted = prompt.detach().cpu().to(dtype=torch.float32)
        local_residual = residual
        while local_residual.ndim < shifted.ndim:
            local_residual = local_residual.unsqueeze(0)
        return shifted - local_residual

    no_tta = dict(normalized)
    no_tta["global_prototype"] = remove_residual(normalized["global_prototype"])
    no_tta["monthly_prototypes"] = {
        str(month): remove_residual(normalized["monthly_prototypes"].get(str(month)))
        for month in range(1, 13)
    }
    no_tta["context_tta_mode"] = CONTEXT_TTA_NONE
    no_tta["context_tta_state"] = None
    no_tta["context_tta_state_hash"] = ""
    no_tta["context_tta_label_usage"] = "none"
    no_tta["context_tta_effective"] = False
    no_tta["context_tta_source_stat_status"] = "not_requested"
    no_tta["prompt_l2_delta_mean"] = 0.0
    no_tta["metadata"] = dict(no_tta.get("metadata", {}))
    no_tta["metadata"].update(
        {
            "context_tta_mode": CONTEXT_TTA_NONE,
            "context_tta_state_hash": "",
            "context_tta_label_usage": "none",
            "context_tta_effective": False,
            "context_tta_source_stat_status": "not_requested",
            "prompt_l2_delta_mean": 0.0,
        }
    )
    return normalize_target_context_prompt_state(no_tta)


def _hash_context_dates(dates: Sequence[str], monthly_counts: Dict[str, int]) -> str:
    payload = json.dumps(
        {"dates": list(dates), "monthly_counts": monthly_counts},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_context_tta_alignment_state(state: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(state.get("schema_version", CONTEXT_TTA_ALIGNMENT_SCHEMA_VERSION)).encode("utf-8"))
    for key in (
        "mode",
        "label_usage",
        "target_val_usage",
        "target_eval_usage",
        "source_statistics_provenance",
        "source_stat_status",
        "target_context_date_hash",
        "input_embedding_dim",
    ):
        digest.update(str(state.get(key, "")).encode("utf-8"))
        digest.update(b"\0")
    for key in (
        "source_embedding_mean",
        "source_embedding_std",
        "target_embedding_mean",
        "target_embedding_std",
        "alignment_shift",
        "alignment_scale",
    ):
        tensor = _prompt_tensor(state.get(key))
        if tensor is not None:
            _sha256_update_tensor(digest, tensor)
    digest.update(
        json.dumps(
            {
                "monthly_counts": state.get("monthly_counts", {}),
                "n_samples": state.get("n_samples", 0),
                "date_start": state.get("date_start", ""),
                "date_end": state.get("date_end", ""),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _hash_context_tta_residual_shift_state(state: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(state.get("schema_version", CONTEXT_TTA_RESIDUAL_SHIFT_SCHEMA_VERSION)).encode("utf-8"))
    for key in (
        "mode",
        "label_usage",
        "target_val_usage",
        "target_eval_usage",
        "target_context_date_hash",
        "input_embedding_dim",
        "prompt_dim",
        "residual_scale",
        "residual_clip_l2",
    ):
        digest.update(str(state.get(key, "")).encode("utf-8"))
        digest.update(b"\0")
    for key in (
        "target_embedding_mean",
        "target_embedding_std",
        "residual_prompt_shift",
    ):
        tensor = _prompt_tensor(state.get(key))
        if tensor is not None:
            _sha256_update_tensor(digest, tensor)
    digest.update(
        json.dumps(
            {
                "monthly_counts": state.get("monthly_counts", {}),
                "n_samples": state.get("n_samples", 0),
                "date_start": state.get("date_start", ""),
                "date_end": state.get("date_end", ""),
                "source_stat_status": state.get("source_stat_status", ""),
                "prompt_l2_delta_mean": state.get("prompt_l2_delta_mean", 0.0),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def normalize_context_tta_alignment_state(state: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not state:
        return None
    normalized = dict(state)
    schema = str(normalized.get("schema_version", CONTEXT_TTA_ALIGNMENT_SCHEMA_VERSION))
    if schema != CONTEXT_TTA_ALIGNMENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported context_tta_state schema_version={schema!r}; "
            f"expected {CONTEXT_TTA_ALIGNMENT_SCHEMA_VERSION!r}"
        )
    mode = str(normalized.get("mode", CONTEXT_TTA_PROMPT_FEATURE_ALIGNMENT))
    if mode != CONTEXT_TTA_PROMPT_FEATURE_ALIGNMENT:
        raise ValueError(f"Unsupported context TTA alignment mode: {mode!r}")
    for key in (
        "source_embedding_mean",
        "source_embedding_std",
        "target_embedding_mean",
        "target_embedding_std",
        "alignment_shift",
        "alignment_scale",
    ):
        tensor = _prompt_tensor(normalized.get(key))
        if tensor is None:
            raise ValueError(f"context_tta_state missing {key}")
        normalized[key] = tensor.detach().cpu()
    normalized["schema_version"] = CONTEXT_TTA_ALIGNMENT_SCHEMA_VERSION
    normalized["mode"] = CONTEXT_TTA_PROMPT_FEATURE_ALIGNMENT
    normalized["label_usage"] = "none"
    normalized.setdefault("target_val_usage", "unused_in_main_protocol")
    normalized.setdefault("target_eval_usage", "final_eval_only_no_selection")
    normalized.setdefault("eval_update_policy", "frozen_state_no_target_eval_updates")
    normalized.setdefault("source_statistics_provenance", "source_checkpoint_or_identity_fallback")
    normalized.setdefault("source_stat_status", str(normalized.get("source_statistics_provenance", "")))
    normalized["effective"] = bool(
        normalized.get("effective", True)
        and "identity_fallback" not in str(normalized.get("source_stat_status", ""))
    )
    normalized.setdefault("prompt_l2_delta_mean", 0.0)
    normalized["monthly_counts"] = {
        str(month): int(dict(normalized.get("monthly_counts", {})).get(str(month), 0))
        for month in range(1, 13)
    }
    normalized["input_embedding_dim"] = int(normalized["target_embedding_mean"].numel())
    normalized["alignment_hash"] = str(
        normalized.get("alignment_hash") or _hash_context_tta_alignment_state(normalized)
    )
    normalized["state_hash"] = normalized["alignment_hash"]
    return normalized


def normalize_context_tta_residual_shift_state(state: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not state:
        return None
    normalized = dict(state)
    schema = str(normalized.get("schema_version", CONTEXT_TTA_RESIDUAL_SHIFT_SCHEMA_VERSION))
    if schema != CONTEXT_TTA_RESIDUAL_SHIFT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported context_tta_state schema_version={schema!r}; "
            f"expected {CONTEXT_TTA_RESIDUAL_SHIFT_SCHEMA_VERSION!r}"
        )
    mode = str(normalized.get("mode", CONTEXT_TTA_CONTEXT_PROMPT_RESIDUAL_SHIFT))
    if mode != CONTEXT_TTA_CONTEXT_PROMPT_RESIDUAL_SHIFT:
        raise ValueError(f"Unsupported context TTA residual-shift mode: {mode!r}")
    for key in ("target_embedding_mean", "target_embedding_std", "residual_prompt_shift"):
        tensor = _prompt_tensor(normalized.get(key))
        if tensor is None:
            raise ValueError(f"context_tta_state missing {key}")
        normalized[key] = tensor.detach().cpu()
    normalized["schema_version"] = CONTEXT_TTA_RESIDUAL_SHIFT_SCHEMA_VERSION
    normalized["mode"] = CONTEXT_TTA_CONTEXT_PROMPT_RESIDUAL_SHIFT
    normalized["label_usage"] = "none"
    normalized.setdefault("target_val_usage", "unused_in_main_protocol")
    normalized.setdefault("target_eval_usage", "final_eval_only_no_selection")
    normalized.setdefault("eval_update_policy", "frozen_state_no_target_eval_updates")
    normalized.setdefault("source_statistics_provenance", "target_context_only_no_source_statistics_required")
    normalized["source_stat_status"] = "target_context_only_no_source_statistics_required"
    normalized["monthly_counts"] = {
        str(month): int(dict(normalized.get("monthly_counts", {})).get(str(month), 0))
        for month in range(1, 13)
    }
    normalized["input_embedding_dim"] = int(normalized["target_embedding_mean"].numel())
    normalized["prompt_dim"] = int(normalized["residual_prompt_shift"].numel())
    normalized["effective"] = bool(float(normalized.get("prompt_l2_delta_mean", 0.0) or 0.0) > 0.0)
    normalized["state_hash"] = str(
        normalized.get("state_hash") or _hash_context_tta_residual_shift_state(normalized)
    )
    return normalized


def normalize_context_tta_state(state: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not state:
        return None
    schema = str(state.get("schema_version", CONTEXT_TTA_ALIGNMENT_SCHEMA_VERSION))
    if schema == CONTEXT_TTA_ALIGNMENT_SCHEMA_VERSION:
        return normalize_context_tta_alignment_state(state)
    if schema == CONTEXT_TTA_RESIDUAL_SHIFT_SCHEMA_VERSION:
        return normalize_context_tta_residual_shift_state(state)
    raise ValueError(f"Unsupported context_tta_state schema_version={schema!r}")


def _reject_forbidden_context_tta_sample_fields(sample: Dict[str, Any]) -> None:
    bad_keys = [
        str(key)
        for key in sample.keys()
        if any(token in str(key).lower() for token in _FORBIDDEN_CONTEXT_TTA_SAMPLE_KEYS)
    ]
    if bad_keys:
        raise ValueError(
            "context_tta_mode found forbidden target label/evaluation fields "
            f"in target_context sample: {bad_keys[:8]}"
        )


def _source_alignment_stats_from_guard_or_trust(
    *,
    guard_state: Dict[str, Any] | None,
    trust_bank_state: Dict[str, Any] | None,
    embedding_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    source_embeddings: torch.Tensor | None = None
    provenance = "identity_fallback_no_source_statistics"
    if guard_state is not None and isinstance(guard_state.get("source_region_month_input_embeddings"), torch.Tensor):
        source_embeddings = guard_state["source_region_month_input_embeddings"].detach().cpu().float()
        provenance = str(guard_state.get("source", "source_prompt_manifold_guard_source_fit_source_val"))
    elif trust_bank_state is not None and isinstance(trust_bank_state.get("source_prompt_embeddings"), torch.Tensor):
        source_embeddings = trust_bank_state["source_prompt_embeddings"].detach().cpu().float()
        provenance = str(trust_bank_state.get("source", "hyperda_source_trust_bank_source_fit_source_val"))

    if source_embeddings is None or source_embeddings.numel() == 0 or source_embeddings.ndim != 2:
        return (
            torch.zeros(int(embedding_dim), dtype=torch.float32),
            torch.ones(int(embedding_dim), dtype=torch.float32),
            provenance,
        )
    if int(source_embeddings.shape[1]) != int(embedding_dim):
        return (
            torch.zeros(int(embedding_dim), dtype=torch.float32),
            torch.ones(int(embedding_dim), dtype=torch.float32),
            f"{provenance}_dimension_mismatch_identity_fallback",
        )
    return (
        source_embeddings.mean(dim=0).to(dtype=torch.float32),
        source_embeddings.std(dim=0, unbiased=False).clamp_min(1e-6).to(dtype=torch.float32),
        provenance,
    )


def build_context_tta_alignment_state(
    *,
    context_tta_mode: str,
    target_input_embeddings: Sequence[torch.Tensor],
    monthly_counts: Dict[str, int],
    dates: Sequence[str],
    target_context_date_hash: str,
    guard_state: Dict[str, Any] | None = None,
    trust_bank_state: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    if context_tta_mode not in CONTEXT_TTA_MODES:
        raise ValueError(f"Unsupported context_tta_mode={context_tta_mode!r}")
    if context_tta_mode == CONTEXT_TTA_NONE:
        return None
    if context_tta_mode != CONTEXT_TTA_PROMPT_FEATURE_ALIGNMENT:
        return None
    if not target_input_embeddings:
        raise ValueError("context_tta_mode=prompt_feature_alignment requires target_context input embeddings")
    target_stack = torch.stack(
        [tensor.detach().cpu().float().view(-1) for tensor in target_input_embeddings],
        dim=0,
    )
    target_mean = target_stack.mean(dim=0)
    target_std = target_stack.std(dim=0, unbiased=False).clamp_min(1e-6)
    source_mean, source_std, source_provenance = _source_alignment_stats_from_guard_or_trust(
        guard_state=guard_state,
        trust_bank_state=trust_bank_state,
        embedding_dim=int(target_stack.shape[1]),
    )
    source_stat_status = source_provenance
    effective = "identity_fallback" not in source_stat_status
    if "identity_fallback" in source_provenance:
        source_mean = target_mean.clone()
        source_std = target_std.clone()
    state = {
        "schema_version": CONTEXT_TTA_ALIGNMENT_SCHEMA_VERSION,
        "mode": CONTEXT_TTA_PROMPT_FEATURE_ALIGNMENT,
        "label_usage": "none",
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "final_eval_only_no_selection",
        "eval_update_policy": "frozen_state_no_target_eval_updates",
        "source_statistics_provenance": source_provenance,
        "source_stat_status": source_stat_status,
        "effective": bool(effective),
        "target_context_date_hash": str(target_context_date_hash or ""),
        "target_context_input_source": "target_context_2015_2021_input_side_embedding",
        "input_embedding_dim": int(target_stack.shape[1]),
        "n_samples": int(target_stack.shape[0]),
        "date_start": min(dates) if dates else "",
        "date_end": max(dates) if dates else "",
        "monthly_counts": {str(month): int(monthly_counts.get(str(month), 0)) for month in range(1, 13)},
        "source_embedding_mean": source_mean,
        "source_embedding_std": source_std,
        "target_embedding_mean": target_mean,
        "target_embedding_std": target_std,
        "alignment_shift": source_mean - target_mean,
        "alignment_scale": source_std / target_std.clamp_min(1e-6),
        "prompt_l2_delta_mean": 0.0,
    }
    state["alignment_hash"] = _hash_context_tta_alignment_state(state)
    state["state_hash"] = state["alignment_hash"]
    return state


def build_context_prompt_residual_shift_state(
    *,
    context_tta_mode: str,
    target_input_embeddings: Sequence[torch.Tensor],
    monthly_counts: Dict[str, int],
    dates: Sequence[str],
    target_context_date_hash: str,
    prompt_dim: int,
    residual_scale: float = 0.05,
    residual_clip_l2: float = 0.0,
) -> Dict[str, Any] | None:
    if context_tta_mode != CONTEXT_TTA_CONTEXT_PROMPT_RESIDUAL_SHIFT:
        return None
    if not target_input_embeddings:
        raise ValueError("context_tta_mode=context_prompt_residual_shift requires target_context input embeddings")
    target_stack = torch.stack(
        [tensor.detach().cpu().float().view(-1) for tensor in target_input_embeddings],
        dim=0,
    )
    target_mean = target_stack.mean(dim=0)
    target_std = target_stack.std(dim=0, unbiased=False).clamp_min(1e-6)
    centered = target_mean - target_mean.mean()
    bounded = torch.tanh(centered / target_std.mean().clamp_min(1e-6))
    prompt_dim = int(prompt_dim)
    if prompt_dim <= 0:
        raise ValueError("context_prompt_residual_shift requires positive prompt_dim")
    if int(bounded.numel()) < prompt_dim:
        repeats = int(np.ceil(prompt_dim / max(1, int(bounded.numel()))))
        residual = bounded.repeat(repeats)[:prompt_dim]
    else:
        residual = bounded[:prompt_dim]
    residual = residual * float(residual_scale)
    residual_clip_l2 = float(residual_clip_l2)
    if residual_clip_l2 > 0.0:
        norm = torch.linalg.vector_norm(residual.float()).clamp_min(1e-12)
        if float(norm.item()) > residual_clip_l2:
            residual = residual * (residual_clip_l2 / float(norm.item()))
    prompt_l2_delta_mean = float(torch.linalg.vector_norm(residual.float()).item())
    state = {
        "schema_version": CONTEXT_TTA_RESIDUAL_SHIFT_SCHEMA_VERSION,
        "mode": CONTEXT_TTA_CONTEXT_PROMPT_RESIDUAL_SHIFT,
        "label_usage": "none",
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "final_eval_only_no_selection",
        "eval_update_policy": "frozen_state_no_target_eval_updates",
        "source_statistics_provenance": "target_context_only_no_source_statistics_required",
        "source_stat_status": "target_context_only_no_source_statistics_required",
        "effective": bool(prompt_l2_delta_mean > 0.0),
        "target_context_date_hash": str(target_context_date_hash or ""),
        "target_context_input_source": "target_context_2015_2021_input_side_embedding",
        "input_embedding_dim": int(target_stack.shape[1]),
        "prompt_dim": prompt_dim,
        "residual_scale": float(residual_scale),
        "residual_clip_l2": residual_clip_l2,
        "n_samples": int(target_stack.shape[0]),
        "date_start": min(dates) if dates else "",
        "date_end": max(dates) if dates else "",
        "monthly_counts": {str(month): int(monthly_counts.get(str(month), 0)) for month in range(1, 13)},
        "target_embedding_mean": target_mean,
        "target_embedding_std": target_std,
        "residual_prompt_shift": residual,
        "prompt_l2_delta_mean": prompt_l2_delta_mean,
    }
    state["state_hash"] = _hash_context_tta_residual_shift_state(state)
    return state


def apply_context_prompt_residual_shift_to_prompt(
    prompt: torch.Tensor,
    context_tta_state: Dict[str, Any] | None,
) -> torch.Tensor:
    state = normalize_context_tta_state(context_tta_state)
    if state is None or str(state.get("mode", "")) != CONTEXT_TTA_CONTEXT_PROMPT_RESIDUAL_SHIFT:
        return prompt
    residual = state["residual_prompt_shift"].to(device=prompt.device, dtype=prompt.dtype)
    while residual.ndim < prompt.ndim:
        residual = residual.unsqueeze(0)
    return prompt + residual


def apply_context_tta_alignment_to_embedding(
    input_embedding: torch.Tensor,
    context_tta_state: Dict[str, Any] | None,
) -> torch.Tensor:
    state = normalize_context_tta_alignment_state(context_tta_state)
    if state is None:
        return input_embedding
    emb = input_embedding.float()
    source_mean = state["source_embedding_mean"].to(device=emb.device, dtype=emb.dtype)
    target_mean = state["target_embedding_mean"].to(device=emb.device, dtype=emb.dtype)
    scale = state["alignment_scale"].to(device=emb.device, dtype=emb.dtype)
    while source_mean.ndim < emb.ndim:
        source_mean = source_mean.unsqueeze(0)
        target_mean = target_mean.unsqueeze(0)
        scale = scale.unsqueeze(0)
    return source_mean + (emb - target_mean) * scale


def maybe_apply_context_tta_alignment_to_embedding(
    input_embedding: torch.Tensor,
    context_tta_state: Dict[str, Any] | None,
) -> torch.Tensor:
    if not context_tta_state:
        return input_embedding
    if str(context_tta_state.get("mode", "")) != CONTEXT_TTA_PROMPT_FEATURE_ALIGNMENT:
        return input_embedding
    return apply_context_tta_alignment_to_embedding(input_embedding, context_tta_state)


def _prompt_l2_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm((left.detach().cpu().float() - right.detach().cpu().float()).view(-1)).item())


def build_target_context_prompt_state(
    samples: Iterable[Dict[str, Any]],
    prompt_encoder: RegionPromptEncoder,
    normalize_x: Callable[[torch.Tensor], torch.Tensor],
    target_region_embedding: torch.Tensor,
    device: torch.device | str,
    context_hash: str = "",
    context_encoder: str = CURRENT_MEAN_STD_CONTEXT_ENCODER,
    source_prompt_manifold_guard_state: Dict[str, Any] | None = None,
    source_trust_bank_state: Dict[str, Any] | None = None,
    context_tta_mode: str = CONTEXT_TTA_NONE,
    context_tta_residual_scale: float = 0.05,
    context_tta_residual_clip_l2: float = 0.0,
) -> Dict[str, Any]:
    """Build monthly target-context prompt prototypes from input-side fields only.

    Reads only ``x``, ``month``, ``date_str``, and region/input masks from each
    sample. Target labels, increments, residuals, validation scores, and eval
    inputs are not consulted.
    """
    if context_encoder not in _CONTEXT_ENCODERS:
        raise ValueError(f"Unsupported context_encoder: {context_encoder}")
    if context_tta_mode not in CONTEXT_TTA_MODES:
        raise ValueError(f"Unsupported context_tta_mode={context_tta_mode!r}")
    device = torch.device(device)
    guard_state = normalize_source_prompt_manifold_guard_state(source_prompt_manifold_guard_state)
    trust_bank_state = normalize_hyperda_source_trust_bank_state(source_trust_bank_state)
    trust_bank_meta = hyperda_trust_bank_summary(trust_bank_state)
    by_month: Dict[int, list[torch.Tensor]] = {month: [] for month in range(1, 13)}
    trust_query_by_month: Dict[int, list[torch.Tensor]] = {month: [] for month in range(1, 13)}
    phys_context_by_month: Dict[int, list[torch.Tensor]] = {month: [] for month in range(1, 13)}
    finite_coverage_by_month: Dict[int, list[float]] = {month: [] for month in range(1, 13)}
    dates: list[str] = []
    phys_trust_rows_by_month: Dict[str, list[Dict[str, float]]] = {
        str(month): [] for month in range(1, 13)
    }
    all_input_embs: list[torch.Tensor] = []
    all_temporal_embs: list[torch.Tensor] = []
    tta_input_embs: list[torch.Tensor] = []

    target_region_embedding = target_region_embedding.to(device=device, dtype=torch.float32)
    if target_region_embedding.ndim == 1:
        target_region_embedding = target_region_embedding.unsqueeze(0)
    elif target_region_embedding.ndim != 2 or target_region_embedding.shape[0] != 1:
        raise ValueError("target_region_embedding must have shape [16] or [1,16]")

    with torch.no_grad():
        for sample in samples:
            if context_tta_mode != CONTEXT_TTA_NONE:
                _reject_forbidden_context_tta_sample_fields(sample)
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
            trust_query_input_emb: torch.Tensor | None = None
            trust_query_mode = str(
                trust_bank_state.get("source_trust_query_mode", SOURCE_TRUST_QUERY_MODE_PROMPT)
                if trust_bank_state
                else SOURCE_TRUST_QUERY_MODE_PROMPT
            )
            if trust_bank_state and source_trust_query_requires_separate_bank(trust_query_mode):
                trust_query_input_emb = raw_da_trust_input_embedding(prompt_encoder, x, region_mask_t)
            phys_context_input_emb = raw_da_trust_input_embedding(prompt_encoder, x, region_mask_t)

            date_str = str(sample.get("date_str", ""))
            month_value = _coerce_month(sample.get("month", None), date_str)
            month = torch.tensor([month_value], dtype=torch.long, device=device)
            temporal = prompt_encoder._temporal_encoding(month)
            temporal_emb = prompt_encoder.temporal_proj(temporal)

            by_month[month_value].append(input_emb.detach())
            if trust_query_input_emb is not None:
                trust_query_by_month[month_value].append(trust_query_input_emb.detach())
            phys_context_by_month[month_value].append(phys_context_input_emb.detach())
            finite_coverage_by_month[month_value].append(finite_coverage)
            all_input_embs.append(input_emb.detach())
            tta_input_embs.append(input_emb.detach().cpu().squeeze(0))
            all_temporal_embs.append(temporal_emb.detach())
            if date_str:
                dates.append(date_str)
            phys_trust_rows_by_month[str(month_value)].append(
                phys_trust_d0_diagnostics_from_tensor(x_np, region_mask=region_mask_np)
            )

        if not all_input_embs:
            raise ValueError("Cannot build target_context prompt state from zero samples")

        preliminary_monthly_counts = {str(month): len(by_month[month]) for month in range(1, 13)}
        resolved_context_hash = context_hash or _hash_context_dates(dates, preliminary_monthly_counts)
        context_tta_state = build_context_tta_alignment_state(
            context_tta_mode=context_tta_mode,
            target_input_embeddings=tta_input_embs,
            monthly_counts=preliminary_monthly_counts,
            dates=dates,
            target_context_date_hash=resolved_context_hash,
            guard_state=guard_state,
            trust_bank_state=trust_bank_state,
        )
        if context_tta_state is None:
            context_tta_state = build_context_prompt_residual_shift_state(
                context_tta_mode=context_tta_mode,
                target_input_embeddings=tta_input_embs,
                monthly_counts=preliminary_monthly_counts,
                dates=dates,
                target_context_date_hash=resolved_context_hash,
                prompt_dim=int(prompt_encoder.hidden_dim),
                residual_scale=float(context_tta_residual_scale),
                residual_clip_l2=float(context_tta_residual_clip_l2),
            )
        aligned_all_input_embs = [
            maybe_apply_context_tta_alignment_to_embedding(input_emb, context_tta_state)
            for input_emb in all_input_embs
        ]

        r_emb = target_region_embedding
        base_global_i = torch.stack(all_input_embs, dim=0).mean(dim=0)
        global_i = torch.stack(aligned_all_input_embs, dim=0).mean(dim=0)
        global_t = torch.stack(all_temporal_embs, dim=0).mean(dim=0)
        base_global_prompt = prompt_encoder.mlp(torch.cat([r_emb, base_global_i, global_t], dim=1)).squeeze(0)
        raw_global_prompt = prompt_encoder.mlp(torch.cat([r_emb, global_i, global_t], dim=1)).squeeze(0)
        shifted_global_prompt = apply_context_prompt_residual_shift_to_prompt(raw_global_prompt, context_tta_state)
        prompt_l2_deltas: list[float] = [_prompt_l2_delta(base_global_prompt, shifted_global_prompt)]
        global_prompt = shifted_global_prompt.detach().cpu()
        trust_query_lists = [row for values in trust_query_by_month.values() for row in values]
        phys_context_lists = [row for values in phys_context_by_month.values() for row in values]
        global_trust_query_prompt: torch.Tensor | None = None
        global_phys_context_prompt: torch.Tensor | None = None
        if trust_query_lists:
            global_trust_query_i = torch.stack(trust_query_lists, dim=0).mean(dim=0)
            raw_global_trust_query_prompt = prompt_encoder.mlp(
                torch.cat([r_emb, global_trust_query_i, global_t], dim=1)
            ).squeeze(0)
            if trust_query_mode == SOURCE_TRUST_QUERY_MODE_BLENDED_RAW_DA:
                raw_global_trust_query_prompt = blend_prompt_and_raw_trust_query(
                    global_prompt.to(device=raw_global_trust_query_prompt.device),
                    raw_global_trust_query_prompt,
                )
            global_trust_query_prompt = raw_global_trust_query_prompt.detach().cpu()
        if phys_context_lists:
            global_phys_context_i = torch.stack(phys_context_lists, dim=0).mean(dim=0)
            global_phys_context_prompt = prompt_encoder.mlp(
                torch.cat([r_emb, global_phys_context_i, global_t], dim=1)
            ).squeeze(0).detach().cpu()

        monthly_prototypes: Dict[str, Optional[torch.Tensor]] = {}
        monthly_trust_query_prototypes: Dict[str, Optional[torch.Tensor]] = {}
        monthly_phys_context_prototypes: Dict[str, Optional[torch.Tensor]] = {}
        monthly_counts: Dict[str, int] = {}
        reliability_features: Dict[str, list[float]] = {}
        source_manifold_distance_by_month: Dict[str, Dict[str, Any]] = {}
        source_manifold_multiplier_by_month: Dict[str, float] = {}
        hyperda_trust_summary_by_month: Dict[str, Dict[str, Any]] = {}
        guard_strength = float(guard_state.get("guard_strength", 0.0)) if guard_state else 0.0
        guard_min_multiplier = float(guard_state.get("min_multiplier", 0.0)) if guard_state else 0.0
        global_distance = source_manifold_distance_for_embedding(global_i, guard_state)
        for month_value in range(1, 13):
            month_key = str(month_value)
            input_embs = by_month[month_value]
            monthly_counts[month_key] = len(input_embs)
            if not input_embs:
                monthly_prototypes[month_key] = None
                monthly_trust_query_prototypes[month_key] = None
                monthly_phys_context_prototypes[month_key] = None
                bounded_distance = float(global_distance["bounded"])
                source_manifold_distance_by_month[month_key] = {
                    "raw": float(global_distance["raw"]),
                    "bounded": bounded_distance,
                    "has_monthly_prototype": 0,
                    "fallback": "global_target_context_prototype",
                }
                source_manifold_multiplier_by_month[month_key] = float(
                    np.clip(1.0 - guard_strength * bounded_distance, guard_min_multiplier, 1.0)
                )
                trust_summary = hyperda_trust_summary_for_embedding(
                    global_trust_query_prompt if global_trust_query_prompt is not None else global_prompt,
                    trust_bank_state,
                    top_m=int(trust_bank_meta.get("source_neighbor_top_m", 4) or 4),
                )
                trust_summary["has_monthly_prototype"] = 0
                trust_summary["fallback"] = "global_target_context_prototype"
                hyperda_trust_summary_by_month[month_key] = trust_summary
                reliability_features[month_key] = bounded_reliability_features(
                    monthly_count=0.0,
                    has_monthly_prototype=0.0,
                    global_context_count=float(len(all_input_embs)),
                    finite_input_coverage=(
                        float(np.mean([v for values in finite_coverage_by_month.values() for v in values]))
                        if all_input_embs
                        else 0.0
                    ),
                    source_manifold_distance_bounded=bounded_distance,
                )
                continue
            aligned_input_embs = [
                maybe_apply_context_tta_alignment_to_embedding(input_emb, context_tta_state)
                for input_emb in input_embs
            ]
            base_month_i = torch.stack(input_embs, dim=0).mean(dim=0)
            month_i = torch.stack(aligned_input_embs, dim=0).mean(dim=0)
            month_tensor = torch.tensor([month_value], dtype=torch.long, device=device)
            month_t = prompt_encoder.temporal_proj(prompt_encoder._temporal_encoding(month_tensor))
            base_prompt = prompt_encoder.mlp(torch.cat([r_emb, base_month_i, month_t], dim=1)).squeeze(0)
            raw_prompt = prompt_encoder.mlp(torch.cat([r_emb, month_i, month_t], dim=1)).squeeze(0)
            prompt = apply_context_prompt_residual_shift_to_prompt(raw_prompt, context_tta_state)
            prompt_l2_deltas.append(_prompt_l2_delta(base_prompt, prompt))
            prompt_cpu = prompt.detach().cpu()
            trust_query_prompt_cpu: torch.Tensor | None = None
            trust_query_inputs = trust_query_by_month[month_value]
            if trust_query_inputs:
                trust_query_i = torch.stack(trust_query_inputs, dim=0).mean(dim=0)
                raw_trust_query_prompt = prompt_encoder.mlp(torch.cat([r_emb, trust_query_i, month_t], dim=1)).squeeze(0)
                if trust_query_mode == SOURCE_TRUST_QUERY_MODE_BLENDED_RAW_DA:
                    raw_trust_query_prompt = blend_prompt_and_raw_trust_query(
                        prompt,
                        raw_trust_query_prompt,
                    )
                trust_query_prompt_cpu = raw_trust_query_prompt.detach().cpu()
            phys_context_prompt_cpu: torch.Tensor | None = None
            phys_context_inputs = phys_context_by_month[month_value]
            if phys_context_inputs:
                phys_context_i = torch.stack(phys_context_inputs, dim=0).mean(dim=0)
                phys_context_prompt_cpu = prompt_encoder.mlp(
                    torch.cat([r_emb, phys_context_i, month_t], dim=1)
                ).squeeze(0).detach().cpu()
            month_distance = source_manifold_distance_for_embedding(month_i, guard_state)
            bounded_distance = float(month_distance["bounded"])
            source_manifold_distance_by_month[month_key] = {
                "raw": float(month_distance["raw"]),
                "bounded": bounded_distance,
                "has_monthly_prototype": 1,
                "fallback": "none",
            }
            source_manifold_multiplier_by_month[month_key] = float(
                np.clip(1.0 - guard_strength * bounded_distance, guard_min_multiplier, 1.0)
            )
            trust_summary = hyperda_trust_summary_for_embedding(
                trust_query_prompt_cpu if trust_query_prompt_cpu is not None else prompt_cpu,
                trust_bank_state,
                top_m=int(trust_bank_meta.get("source_neighbor_top_m", 4) or 4),
            )
            trust_summary["has_monthly_prototype"] = 1
            trust_summary["fallback"] = "none"
            hyperda_trust_summary_by_month[month_key] = trust_summary
            monthly_prototypes[month_key] = prompt_cpu
            monthly_trust_query_prototypes[month_key] = trust_query_prompt_cpu
            monthly_phys_context_prototypes[month_key] = phys_context_prompt_cpu
            reliability_features[month_key] = bounded_reliability_features(
                monthly_count=float(len(input_embs)),
                has_monthly_prototype=1.0,
                global_context_count=float(len(all_input_embs)),
                finite_input_coverage=(
                    float(np.mean(finite_coverage_by_month[month_value]))
                    if finite_coverage_by_month[month_value]
                    else 0.0
                ),
                source_manifold_distance_bounded=bounded_distance,
            )

    multiplier_values = list(source_manifold_multiplier_by_month.values())
    distance_values = [float(row["bounded"]) for row in source_manifold_distance_by_month.values()]
    phys_trust_d0_summary = phys_trust_d0_summary_from_monthly_rows(
        phys_trust_rows_by_month,
        hyperda_trust_summary_by_month=hyperda_trust_summary_by_month,
    )
    prompt_l2_delta_mean = float(np.mean(prompt_l2_deltas)) if "prompt_l2_deltas" in locals() and prompt_l2_deltas else 0.0
    if isinstance(context_tta_state, dict):
        context_tta_state["prompt_l2_delta_mean"] = prompt_l2_delta_mean
        if str(context_tta_state.get("mode", "")) == CONTEXT_TTA_CONTEXT_PROMPT_RESIDUAL_SHIFT:
            context_tta_state["effective"] = bool(prompt_l2_delta_mean > 0.0)
            context_tta_state["state_hash"] = _hash_context_tta_residual_shift_state(context_tta_state)
        elif str(context_tta_state.get("mode", "")) == CONTEXT_TTA_PROMPT_FEATURE_ALIGNMENT:
            context_tta_state["state_hash"] = context_tta_state.get("alignment_hash", "")
    context_tta_state_hash = (
        str(context_tta_state.get("state_hash", context_tta_state.get("alignment_hash", "")))
        if context_tta_state is not None
        else ""
    )
    context_tta_source_stat_status = (
        str(context_tta_state.get("source_stat_status", "not_requested"))
        if context_tta_state is not None
        else "not_requested"
    )
    context_tta_effective = bool(context_tta_state.get("effective", False)) if context_tta_state is not None else False
    return {
        "schema_version": TARGET_CONTEXT_PROMPT_SCHEMA_VERSION,
        "prompt_source": TARGET_CONTEXT_PROMPT_SOURCE,
        "label_usage": "none",
        "context_hash": resolved_context_hash,
        "context_date_hash": resolved_context_hash,
        "context_tta_mode": context_tta_mode,
        "context_tta_state": context_tta_state,
        "context_tta_state_hash": context_tta_state_hash,
        "context_tta_label_usage": "none",
        "context_tta_effective": context_tta_effective,
        "context_tta_source_stat_status": context_tta_source_stat_status,
        "prompt_l2_delta_mean": prompt_l2_delta_mean,
        "date_start": min(dates) if dates else "",
        "date_end": max(dates) if dates else "",
        "n_samples": int(sum(monthly_counts.values())),
        "monthly_counts": monthly_counts,
        "reliability_feature_schema": list(SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA),
        "reliability_features": reliability_features,
        "source_manifold_distance_schema": dict(SOURCE_MANIFOLD_DISTANCE_SCHEMA),
        "target_context_source_manifold_distance_by_month": source_manifold_distance_by_month,
        "source_manifold_guard_multiplier_by_month": source_manifold_multiplier_by_month,
        "hyperda_trust_bank_summary": trust_bank_meta,
        "hyperda_trust_summary_by_month": hyperda_trust_summary_by_month,
        "phys_trust_d0_summary": phys_trust_d0_summary,
        "source_manifold_guard_multiplier_summary": {
            "enabled": bool(guard_state),
            "min": float(np.min(multiplier_values)) if multiplier_values else 1.0,
            "max": float(np.max(multiplier_values)) if multiplier_values else 1.0,
            "mean": float(np.mean(multiplier_values)) if multiplier_values else 1.0,
            "distance_bounded_min": float(np.min(distance_values)) if distance_values else 0.0,
            "distance_bounded_max": float(np.max(distance_values)) if distance_values else 0.0,
            "distance_bounded_mean": float(np.mean(distance_values)) if distance_values else 0.0,
        },
        "global_prototype": global_prompt,
        "global_source_trust_query_prototype": global_trust_query_prompt,
        "global_phys_context_prototype": global_phys_context_prompt,
        "monthly_prototypes": monthly_prototypes,
        "monthly_source_trust_query_prototypes": monthly_trust_query_prototypes,
        "monthly_phys_context_prototypes": monthly_phys_context_prototypes,
        "metadata": {
            "prompt_source": TARGET_CONTEXT_PROMPT_SOURCE,
            "context_encoder": context_encoder,
            "input_usage": target_context_input_usage(context_encoder),
            **prompt_domain_metadata(context_encoder),
            "region_usage": "target_region_embedding_or_source_mean_fallback",
            "temporal_usage": "month_of_year_seasonal_phase",
            "reliability_feature_source": "input_side_context_summary_only",
            "reliability_feature_transform": RELIABILITY_FEATURE_TRANSFORM,
            "source_manifold_distance_schema": dict(SOURCE_MANIFOLD_DISTANCE_SCHEMA),
            "source_manifold_guard_source": (
                str(guard_state.get("source", "source_fit_source_val_only"))
                if guard_state
                else "disabled_or_old_checkpoint"
            ),
            "source_manifold_guard_calibration_source": (
                str(guard_state.get("calibration_source", "source_fit_source_val_only"))
                if guard_state
                else "disabled"
            ),
            "source_manifold_guard_strength": guard_strength,
            "hyperda_trust_bank_source": trust_bank_meta.get("source", "disabled_or_old_checkpoint"),
            "hyperda_trust_bank_label_usage": trust_bank_meta.get("label_usage", "none"),
            "hyperda_trust_bank_target_eval_usage": trust_bank_meta.get(
                "target_eval_usage",
                "final_eval_only_no_selection",
            ),
            "trust_bank_hash": trust_bank_meta.get("trust_bank_hash", ""),
            "source_neighbor_top_m": int(trust_bank_meta.get("source_neighbor_top_m", 0) or 0),
            "trust_strength": float(trust_bank_meta.get("trust_strength", 0.0) or 0.0),
            "phys_trust_d0_schema_version": phys_trust_d0_summary["schema_version"],
            "phys_trust_d0_usage": "diagnostic_only_not_model_selection",
            "source_trust_query_mode": trust_bank_meta.get(
                "source_trust_query_mode",
                SOURCE_TRUST_QUERY_MODE_PROMPT,
            ),
            "source_trust_query_input_domain": source_trust_query_input_domain(
                str(trust_bank_meta.get("source_trust_query_mode", SOURCE_TRUST_QUERY_MODE_PROMPT)),
                context_encoder,
            ),
            "source_trust_query_blend_lambda": (
                SOURCE_TRUST_QUERY_BLEND_LAMBDA
                if trust_bank_meta.get("source_trust_query_mode") == SOURCE_TRUST_QUERY_MODE_BLENDED_RAW_DA
                else 0.0
            ),
            "main_prompt_unchanged_by_blended_query": bool(
                trust_bank_meta.get("source_trust_query_mode") == SOURCE_TRUST_QUERY_MODE_BLENDED_RAW_DA
            ),
            "input_summary_mask": "target_active_region_mask",
            "channel_11_usage": "finite_input_feature_only_not_observation_or_static_mask",
            "label_usage": "none",
            "context_tta_mode": context_tta_mode,
            "context_tta_residual_scale": (
                float(context_tta_residual_scale)
                if context_tta_mode == CONTEXT_TTA_CONTEXT_PROMPT_RESIDUAL_SHIFT
                else 0.0
            ),
            "context_tta_residual_clip_l2": (
                float(context_tta_residual_clip_l2)
                if context_tta_mode == CONTEXT_TTA_CONTEXT_PROMPT_RESIDUAL_SHIFT
                else 0.0
            ),
            "context_tta_state_hash": context_tta_state_hash,
            "context_tta_label_usage": "none",
            "context_tta_effective": context_tta_effective,
            "context_tta_source_stat_status": context_tta_source_stat_status,
            "prompt_l2_delta_mean": prompt_l2_delta_mean,
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
        apply_support_affine_calibration: bool = True,
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
        raw_adapted_state = saved_config.get("raw_adapted_state", {}) or {}
        post_gate_state = saved_config.get("post_gate_state", {}) or {}
        final_eval_mix_state = saved_config.get("final_eval_mix_state", {}) or {}
        self.checkpoint_context_tta_mode = str(saved_config.get("context_tta_mode") or CONTEXT_TTA_NONE)
        self.stage3_protocol_metadata: Dict[str, Any] = {
            "posterior_state_schema": (
                stage3_posterior_state_dict.get("schema_version", "")
                if isinstance(stage3_posterior_state_dict, dict)
                else ""
            ),
            "posterior_metadata_schema": stage3_posterior_metadata.get("schema_version", ""),
            "posterior_form": stage3_posterior_metadata.get("posterior_form", ""),
            "K": stage3_posterior_metadata.get("K", saved_config.get("K")),
            "stage3_kshot_mode": saved_config.get("stage3_kshot_mode", "paper_safe"),
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
            "raw_adapted_state_hash": raw_adapted_state.get(
                "target_adapter_state_hash",
                saved_config.get("raw_adapted_state_hash", ""),
            ),
            "post_gate_state_hash": post_gate_state.get(
                "target_adapter_state_hash",
                saved_config.get("post_gate_state_hash", ""),
            ),
            "raw_adapted_drift_from_prior": raw_adapted_state.get(
                "drift_from_prior",
                saved_config.get("raw_adapted_drift_from_prior", {}),
            ),
            "post_gate_drift_from_prior": post_gate_state.get(
                "drift_from_prior",
                saved_config.get("post_gate_drift_from_prior", {}),
            ),
            "final_eval_mix_state": dict(final_eval_mix_state),
            "context_tta_mode": self.checkpoint_context_tta_mode,
            "context_tta_state_hash": saved_config.get("context_tta_state_hash", ""),
            "context_tta_label_usage": saved_config.get("context_tta_label_usage", "none"),
            "context_tta_effective": bool(saved_config.get("context_tta_effective", False)),
            "context_tta_source_stat_status": saved_config.get("context_tta_source_stat_status", "not_requested"),
            "prompt_l2_delta_mean": float(saved_config.get("prompt_l2_delta_mean", 0.0) or 0.0),
            "target_labels_loaded_for_adaptation": bool(
                stage3_posterior_metadata.get("target_labels_loaded_for_adaptation", False)
            ),
            "target_labels_used_for_adaptation": bool(
                stage3_posterior_metadata.get("target_labels_used_for_adaptation", False)
            ),
            "policy_source": saved_config.get("policy_source", ""),
            "paper_facing_run": bool(saved_config.get("paper_facing_run", False)),
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

        self.hyper_phys_consistency_guard = bool(cfg_get("hyper_phys_consistency_guard", False))
        self.hyper_phys_formula_operator = bool(cfg_get("hyper_phys_formula_operator", False))
        self.phys_formula_mode = str(cfg_get("phys_formula_mode", PHYS_FORMULA_MODE))
        self.phys_formula_source = str(cfg_get("phys_formula_source", PHYS_FORMULA_SOURCE))
        self.phys_consistency_guard_mode = str(
            cfg_get("phys_consistency_guard_mode", PHYS_CONSISTENCY_GUARD_MODE)
        )
        self.phys_consistency_source = str(cfg_get("phys_consistency_source", PHYS_CONSISTENCY_SOURCE))
        self.phys_consistency_min_surface = float(cfg_get("phys_consistency_min_surface", 0.95))
        self.phys_consistency_min_rootzone = float(cfg_get("phys_consistency_min_rootzone", 0.90))
        self.phys_consistency_strength_surface = float(cfg_get("phys_consistency_strength_surface", 0.10))
        self.phys_consistency_strength_rootzone = float(cfg_get("phys_consistency_strength_rootzone", 0.15))
        source_state_candidate = (
            checkpoint.get("phys_consistency_source_state")
            or saved_config.get("phys_consistency_source_state")
            or source_config.get("phys_consistency_source_state")
            or (saved_config.get("phys_consistency_guard_summary", {}) or {}).get("source_state_summary")
            or (source_config.get("phys_consistency_guard_summary", {}) or {}).get("source_state_summary")
        )
        self._phys_consistency_source_state = (
            source_state_candidate if isinstance(source_state_candidate, dict) else None
        )
        self._last_phys_consistency_guard_summary: Dict[str, Any] = {"enabled": False}
        self._last_phys_formula_feature_summary: Dict[str, Any] = {"enabled": False}

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
                hyper_source_manifold_guard=cfg_get("hyper_source_manifold_guard", False),
                hyper_source_manifold_guard_strength=cfg_get("hyper_source_manifold_guard_strength", 0.25),
                hyper_source_manifold_guard_distance_key=cfg_get(
                    "hyper_source_manifold_guard_distance_key",
                    SOURCE_MANIFOLD_DISTANCE_KEY,
                ),
                hyper_source_manifold_guard_min_multiplier=cfg_get(
                    "hyper_source_manifold_guard_min_multiplier",
                    0.0,
                ),
                source_manifold_guard_calibration=cfg_get("source_manifold_guard_calibration", "disabled"),
                hyper_source_trust_routing=cfg_get("hyper_source_trust_routing", False),
                hyper_source_trust_strength=cfg_get("hyper_source_trust_strength", 0.0),
                hyper_source_trust_top_m=cfg_get("hyper_source_trust_top_m", 4),
                hyper_source_trust_variable_gate=cfg_get("hyper_source_trust_variable_gate", False),
                hyper_phys_agreement_guard=cfg_get("hyper_phys_agreement_guard", False),
                hyper_phys_agreement_guard_strength=cfg_get("hyper_phys_agreement_guard_strength", 1.0),
                hyper_phys_agreement_guard_min_multiplier=cfg_get("hyper_phys_agreement_guard_min_multiplier", 0.0),
                hyper_phys_agreement_guard_risk_rule=cfg_get("hyper_phys_agreement_guard_risk_rule", "or"),
                hyper_phys_context_modulation=cfg_get("hyper_phys_context_modulation", False),
                hyper_phys_delta_scale=cfg_get("hyper_phys_delta_scale", 0.25),
                hyper_phys_gate_init=cfg_get("hyper_phys_gate_init", 0.90),
                hyper_operator_droppath_p=cfg_get("hyper_operator_droppath_p", 0.10),
                phys_context_source=cfg_get("phys_context_source", "raw_input_side_da_diagnostics"),
                hyper_phys_gain_basis_residual=cfg_get("hyper_phys_gain_basis_residual", False),
                hyper_phys_gain_basis_coeff_scale=cfg_get("hyper_phys_gain_basis_coeff_scale", 0.05),
                hyper_phys_gain_basis_residual_clip=cfg_get("hyper_phys_gain_basis_residual_clip", 0.25),
                hyper_phys_gain_basis_beta_init=cfg_get("hyper_phys_gain_basis_beta_init", 0.50),
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
        self.support_affine_calibration = (
            checkpoint.get("support_affine_calibration")
            or saved_config.get("support_affine_calibration")
            or {}
        )
        if not isinstance(self.support_affine_calibration, dict):
            self.support_affine_calibration = {}
        self.apply_support_affine_calibration = bool(
            apply_support_affine_calibration
            and
            self.support_affine_calibration
            and self.support_affine_calibration.get("status") == "calibrated"
        )
        self.stage3_protocol_metadata["support_affine_calibration"] = dict(self.support_affine_calibration)
        self._prompt_route_uses_target_fallback = False
        self._fixed_target_prompt: Optional[torch.Tensor] = None
        self._target_context_prompt_state: Optional[Dict[str, Any]] = None
        self._target_prompt_metadata: Dict[str, Any] = {}
        self._source_prompt_manifold_guard_state = normalize_source_prompt_manifold_guard_state(
            checkpoint.get("source_prompt_manifold_guard_state")
            or saved_config.get("source_prompt_manifold_guard_state")
            or saved_config.get("source_prompt_manifold_guard_summary")
        )
        self._source_trust_bank_state = normalize_hyperda_source_trust_bank_state(
            checkpoint.get("source_trust_bank_state")
            or checkpoint.get("hyperda_source_trust_bank_state")
            or saved_config.get("source_trust_bank_state")
            or saved_config.get("hyperda_source_trust_bank_state")
            or saved_config.get("source_trust_bank_summary")
        )
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

    def load_no_tta_target_context_prompt_state_from_current(self) -> Dict[str, Any]:
        if self._target_context_prompt_state is None:
            raise RuntimeError("target_context_prompt_state has not been initialized")
        state = target_context_prompt_state_without_context_tta(self._target_context_prompt_state)
        return self.load_target_context_prompt_state(state)

    def compose_target_context_prompt(self, month: int | Sequence[int] | torch.Tensor) -> torch.Tensor:
        if self._target_context_prompt_state is None:
            raise RuntimeError("target_context_prompt_state has not been initialized")
        return compose_target_context_prompt_from_state(self._target_context_prompt_state, month, device=self.device)

    def compose_target_context_source_trust_query(
        self,
        month: int | Sequence[int] | torch.Tensor,
    ) -> torch.Tensor | None:
        if self._target_context_prompt_state is None:
            return None
        return compose_target_context_source_trust_query_from_state(
            self._target_context_prompt_state,
            month,
            device=self.device,
        )

    def compose_target_context_phys_token(
        self,
        month: int | Sequence[int] | torch.Tensor,
    ) -> torch.Tensor | None:
        if self._target_context_prompt_state is None:
            return None
        return compose_target_context_phys_token_from_state(
            self._target_context_prompt_state,
            month,
            device=self.device,
        )

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

    def set_target_context_prompt_from_samples(
        self,
        samples: Iterable[Dict[str, Any]],
        *,
        context_tta_mode: str | None = None,
    ) -> Dict[str, Any]:
        """Build monthly target-context prompt prototypes from input-side fields only.

        The prompt summary reads only ``x``, ``month``, and ``date_str``. It
        deliberately does not read target analysis or increment labels.
        """
        resolved_context_tta_mode = (
            str(context_tta_mode)
            if context_tta_mode is not None
            else self.checkpoint_context_tta_mode
        )
        state = build_target_context_prompt_state(
            samples=samples,
            prompt_encoder=self.prompt_encoder,
            normalize_x=self._normalize,
            target_region_embedding=self._target_region_embedding_for_prompt_state(),
            device=self.device,
            context_encoder=self.context_encoder,
            source_prompt_manifold_guard_state=self._source_prompt_manifold_guard_state,
            source_trust_bank_state=self._source_trust_bank_state,
            context_tta_mode=resolved_context_tta_mode,
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
        region_mask: torch.Tensor | None,
        reliability_features: torch.Tensor | None,
    ) -> torch.Tensor:
        parameters = inspect.signature(self.model.forward).parameters
        kwargs: Dict[str, Any] = {}
        if self._requires_month:
            kwargs["month"] = month
        if "x_raw" in parameters:
            kwargs["x_raw"] = x_raw
        if "reliability_features" in parameters and reliability_features is not None:
            kwargs["reliability_features"] = reliability_features
        if "source_trust_bank" in parameters and self._source_trust_bank_state is not None:
            kwargs["source_trust_bank"] = self._source_trust_bank_state
            if "source_trust_query" in parameters:
                source_trust_query = self.compose_target_context_source_trust_query(month)
                if source_trust_query is not None:
                    kwargs["source_trust_query"] = source_trust_query
        if "z_phys" in parameters and getattr(self.model, "hyper_phys_context_modulation", False):
            z_phys = None
            if self.hyper_phys_formula_operator:
                formula_encoder = getattr(self.model, "formula_phys_context_encoder", None)
                if formula_encoder is None:
                    raise ValueError(
                        "hyper_phys_formula_operator checkpoint requires "
                        "model.formula_phys_context_encoder"
                    )
                features, summary = phys_formula_features_from_raw_tensor(
                    x_raw,
                    region_mask=region_mask,
                    month=month,
                    source_state=self._phys_consistency_source_state,
                    mode=self.phys_formula_mode,
                    source=self.phys_formula_source,
                )
                self._last_phys_formula_feature_summary = summary
                z_phys = formula_encoder(features.to(device=x_norm.device, dtype=x_norm.dtype))
            else:
                z_phys = self.compose_target_context_phys_token(month)
            if z_phys is not None:
                kwargs["z_phys"] = z_phys
        if "variable_trust_gate" in parameters and self.hyper_phys_consistency_guard:
            variable_gate, summary = phys_consistency_guard_from_raw_tensor(
                x_raw,
                region_mask=region_mask,
                month=month,
                source_state=self._phys_consistency_source_state,
                source=self.phys_consistency_source,
                mode=self.phys_consistency_guard_mode,
                min_surface=self.phys_consistency_min_surface,
                min_rootzone=self.phys_consistency_min_rootzone,
                strength_surface=self.phys_consistency_strength_surface,
                strength_rootzone=self.phys_consistency_strength_rootzone,
            )
            self._last_phys_consistency_guard_summary = summary
            kwargs["variable_trust_gate"] = variable_gate.to(device=x_norm.device, dtype=x_norm.dtype)
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
        region_mask = None
        region_mask_np = sample.get("region_mask", sample.get("active_region_mask"))
        if region_mask_np is not None:
            region_mask = torch.as_tensor(
                np.asarray(region_mask_np) > 0.5,
                dtype=torch.bool,
                device=x_norm.device,
            )

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
                region_mask=region_mask,
                reliability_features=reliability_features,
            )  # [1, 2, H, W]

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
        if self.apply_support_affine_calibration:
            season = str(sample.get("season", "") or "")

            def _affine_coeff(variable: str) -> tuple[float, float]:
                seasonal = self.support_affine_calibration.get("seasonal_affine_coefficients", {})
                if season and isinstance(seasonal, dict):
                    season_block = seasonal.get(season, {})
                    if isinstance(season_block, dict) and variable in season_block:
                        block = season_block.get(variable, {}) or {}
                        return float(block.get("a", 1.0)), float(block.get("b", 0.0))
                block = (
                    self.support_affine_calibration.get("support_affine_coefficients", {})
                    or {}
                ).get(variable, {})
                if not isinstance(block, dict):
                    return 1.0, 0.0
                return float(block.get("a", 1.0)), float(block.get("b", 0.0))

            a_s, b_s = _affine_coeff("surface")
            a_r, b_r = _affine_coeff("rootzone")
            pred_inc_s = (a_s * pred_inc_s + b_s).astype(np.float32)
            pred_inc_r = (a_r * pred_inc_r + b_r).astype(np.float32)

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
