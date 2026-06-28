"""HyperDA-RISE retrieval-informed expert composition.

This module implements the protocol-safe core for HyperDA-RISE:

* source-side router priors are trained from pseudo-target source episodes and
  source_val expert WRMSE;
* K=0 routing uses input-side target_context descriptors only;
* K>0 posterior updates are restricted to low-dimensional mixture/gain/bias
  parameters and never update frozen expert backbones.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


VARIABLES = ("surface", "rootzone")
RISE_METHOD = "HyperDA-RISE"
RISE_ROUTER_SCHEMA = "hyperda_rise_router_prior_v1"
RISE_POSTERIOR_SCHEMA = "hyperda_rise_posterior_config_v1"
SOURCE_REGIONS = ("US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6")
FROZEN_PARAMETER_KEYS = (
    "backbone",
    "forecast_expert",
    "source_pooled_backbone",
    "prompt_encoder_backbone",
    "hyperda_basis",
    "hypernetwork",
    "source_region_specialist_backbones",
)
LOW_DIM_UPDATED_PARAMETER_KEYS = (
    "mixture_logits",
    "mixture_weights",
    "scalar_gain",
    "scalar_bias",
    "monthly_gain",
    "monthly_bias",
)


@dataclass(frozen=True)
class ContextDescriptor:
    """Input-side context descriptor used by the RISE router."""

    vector: np.ndarray
    monthly_vectors: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


def _as_float_array(value: Any, *, name: str = "array") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _softmax(scores: Mapping[str, float], *, temperature: float) -> Dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not scores:
        raise ValueError("cannot softmax an empty score mapping")
    ids = sorted(scores)
    values = np.asarray([float(scores[eid]) for eid in ids], dtype=np.float64)
    values = values / float(temperature)
    values = values - np.max(values)
    exp = np.exp(values)
    total = float(np.sum(exp))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("non-finite softmax normalizer")
    return {eid: float(exp[idx] / total) for idx, eid in enumerate(ids)}


def _normalise_weights(weights: Mapping[str, float], *, expert_ids: Sequence[str]) -> Dict[str, float]:
    out = {eid: max(0.0, float(weights.get(eid, 0.0))) for eid in expert_ids}
    total = float(sum(out.values()))
    if total <= 0.0:
        uniform = 1.0 / max(1, len(expert_ids))
        return {eid: uniform for eid in expert_ids}
    return {eid: float(value / total) for eid, value in out.items()}


def _require_no_label_keys(sample: Mapping[str, Any], *, purpose: str) -> None:
    forbidden = [
        "analysis_surface",
        "analysis_rootzone",
        "increment_surface",
        "increment_rootzone",
        "true_increment_surface",
        "true_increment_rootzone",
        "target",
        "y",
        "metric_mask",
        "loss_mask",
        "label_valid_mask",
    ]
    present = [key for key in forbidden if key in sample]
    if present:
        raise ValueError(f"{purpose} must use input-side samples only; found label/eval keys: {present}")


def _sample_descriptor_vector(sample: Mapping[str, Any]) -> np.ndarray:
    _require_no_label_keys(sample, purpose="RISE context descriptor")
    x = _as_float_array(sample.get("x"), name="sample.x")
    if x.ndim != 3:
        raise ValueError(f"sample.x must have shape [C,H,W], got {x.shape}")
    mask = np.asarray(sample.get("region_mask", sample.get("active_region_mask", np.ones(x.shape[-2:], dtype=np.float32))), dtype=np.float32)
    if mask.shape != x.shape[-2:]:
        raise ValueError(f"region mask shape {mask.shape} does not match x spatial shape {x.shape[-2:]}")
    valid = (mask > 0.5) & np.all(np.isfinite(x), axis=0)
    if not np.any(valid):
        raise ValueError("context descriptor has no finite active pixels")
    channel_means = []
    channel_stds = []
    for channel in range(x.shape[0]):
        values = x[channel][valid]
        channel_means.append(float(np.mean(values)))
        channel_stds.append(float(np.std(values)))

    forecast_surface = np.asarray(sample.get("forecast_surface", x[0]), dtype=np.float32)
    forecast_rootzone = np.asarray(sample.get("forecast_rootzone", x[1] if x.shape[0] > 1 else x[0]), dtype=np.float32)
    surface_vals = forecast_surface[valid]
    rootzone_vals = forecast_rootzone[valid]
    mask_fraction = float(np.mean(mask > 0.5))
    base_valid = sample.get("base_valid_mask")
    base_valid_fraction = float(np.nanmean(np.asarray(base_valid, dtype=np.float32)[mask > 0.5])) if base_valid is not None and np.any(mask > 0.5) else 0.0
    month = int(sample.get("month", 1))
    month_sin = math.sin(2.0 * math.pi * (month - 1) / 12.0)
    month_cos = math.cos(2.0 * math.pi * (month - 1) / 12.0)

    extra = [
        float(np.mean(surface_vals)),
        float(np.std(surface_vals)),
        float(np.mean(rootzone_vals)),
        float(np.std(rootzone_vals)),
        mask_fraction,
        base_valid_fraction,
        month_sin,
        month_cos,
    ]
    return np.asarray(channel_means + channel_stds + extra, dtype=np.float32)


def build_context_descriptor(samples: Iterable[Mapping[str, Any]]) -> ContextDescriptor:
    """Build an input-only global/monthly descriptor from context samples.

    The function refuses samples containing target label or metric-mask fields.
    Use ``HydroDADataset.get_input_side_sample`` for target_context.
    """
    vectors: list[np.ndarray] = []
    by_month: Dict[str, list[np.ndarray]] = {str(month): [] for month in range(1, 13)}
    dates: list[str] = []
    for sample in samples:
        vector = _sample_descriptor_vector(sample)
        vectors.append(vector)
        month = int(sample.get("month", 1))
        if month < 1 or month > 12:
            raise ValueError(f"month must be in 1..12, got {month}")
        by_month[str(month)].append(vector)
        date_str = str(sample.get("date_str", ""))
        if date_str:
            dates.append(date_str)
    if not vectors:
        raise ValueError("Cannot build RISE context descriptor from zero samples")

    matrix = np.stack(vectors, axis=0)
    global_vector = matrix.mean(axis=0).astype(np.float32)
    monthly_vectors = {
        str(month): (
            np.stack(month_vectors, axis=0).mean(axis=0).astype(np.float32)
            if month_vectors
            else global_vector.copy()
        )
        for month, month_vectors in by_month.items()
    }
    monthly_counts = {month: len(month_vectors) for month, month_vectors in by_month.items()}
    context_hash = _stable_hash({"dates": dates, "monthly_counts": monthly_counts, "vector": global_vector})
    metadata = {
        "schema_version": "hyperda_rise_context_descriptor_v1",
        "descriptor_source": "target_context_input_side" if dates else "input_side",
        "label_usage": "none",
        "n_samples": len(vectors),
        "date_start": min(dates) if dates else "",
        "date_end": max(dates) if dates else "",
        "monthly_counts": monthly_counts,
        "context_hash": context_hash,
        "target_eval_used": False,
    }
    return ContextDescriptor(vector=global_vector, monthly_vectors=monthly_vectors, metadata=metadata)


def _expert_ids(candidates: Iterable[Mapping[str, Any]]) -> list[str]:
    ids = [str(candidate["expert_id"]) for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate expert_id values: {ids}")
    if not ids:
        raise ValueError("RISE requires at least one expert candidate")
    return ids


def _validate_source_episode(episode: Mapping[str, Any], *, expert_ids: Sequence[str]) -> None:
    split_role = str(episode.get("metric_split_role", ""))
    if split_role != "source_val":
        raise ValueError(
            "RISE router prior training may use only source_val pseudo-target metrics; "
            f"got metric_split_role={split_role!r}"
        )
    pseudo_target = str(episode.get("pseudo_target_region_id", ""))
    if not pseudo_target or pseudo_target not in SOURCE_REGIONS:
        raise ValueError(f"pseudo_target_region_id must be one of {list(SOURCE_REGIONS)}, got {pseudo_target!r}")
    metrics = episode.get("expert_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("router prior episode missing expert_metrics mapping")
    for variable in VARIABLES:
        var_metrics = metrics.get(variable)
        if not isinstance(var_metrics, Mapping):
            raise ValueError(f"router prior episode missing expert_metrics.{variable}")
        missing = [eid for eid in expert_ids if eid not in var_metrics]
        if missing:
            raise ValueError(f"expert_metrics.{variable} missing expert ids: {missing}")
        for eid in expert_ids:
            value = float(var_metrics[eid])
            if not math.isfinite(value):
                raise ValueError(f"expert metric for {variable}/{eid} is not finite: {value}")


def build_router_prior(
    *,
    episodes: Iterable[Mapping[str, Any]],
    candidates: Iterable[Mapping[str, Any]],
    temperature: float = 0.2,
    uncertainty_floor: float = 0.05,
) -> Dict[str, Any]:
    """Build a source-only RISE router prior from pseudo-target episodes."""
    candidate_list = [dict(candidate) for candidate in candidates]
    ids = _expert_ids(candidate_list)
    episode_list = [dict(episode) for episode in episodes]
    if not episode_list:
        raise ValueError("RISE router prior requires at least one source pseudo-target episode")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    for episode in episode_list:
        _validate_source_episode(episode, expert_ids=ids)

    prototypes: list[Dict[str, Any]] = []
    for episode in episode_list:
        descriptor = np.asarray(episode.get("descriptor"), dtype=np.float32).reshape(-1)
        if descriptor.size == 0 or not np.all(np.isfinite(descriptor)):
            raise ValueError("episode descriptor must be a finite non-empty vector")
        metrics = episode["expert_metrics"]
        weights: Dict[str, Dict[str, float]] = {}
        uncertainty: Dict[str, float] = {}
        for variable in VARIABLES:
            scores = {eid: -float(metrics[variable][eid]) for eid in ids}
            weights[variable] = _softmax(scores, temperature=temperature)
            values = np.asarray([float(metrics[variable][eid]) for eid in ids], dtype=np.float64)
            spread = float(np.std(values))
            mean = float(np.mean(np.abs(values))) + 1e-12
            uncertainty[variable] = max(float(uncertainty_floor), min(1.0, spread / mean))
        prototypes.append(
            {
                "pseudo_target_region_id": str(episode["pseudo_target_region_id"]),
                "descriptor": descriptor.astype(float).tolist(),
                "metric_split_role": "source_val",
                "source_val_metric": str(episode.get("source_val_metric", "increment_rmse_latw")),
                "weights": weights,
                "uncertainty": uncertainty,
            }
        )

    return {
        "schema_version": RISE_ROUTER_SCHEMA,
        "method": RISE_METHOD,
        "method_id": "hyperda_rise_source_side_router_prior",
        "router_name": "Retrieval-Informed Source-Side Expert Router",
        "router_training_source": "source_pseudo_target_episodes",
        "training_label_source": "source_val_2022",
        "model_selection_source": "source_val_2022_preregistered",
        "temperature": float(temperature),
        "uncertainty_floor": float(uncertainty_floor),
        "candidates": candidate_list,
        "expert_ids": ids,
        "prototypes": prototypes,
        "no_leakage_declaration": {
            "target_context_labels_used": False,
            "target_support_labels_used_for_prior": False,
            "target_val_used_for_router_training": False,
            "target_eval_used_for_router_training": False,
            "target_eval_used_for_router_weights": False,
            "target_eval_used_for_expert_selection": False,
            "dynamic_target_eval_gating": False,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def route_weights_from_prior(
    prior: Mapping[str, Any],
    *,
    descriptor: Sequence[float] | np.ndarray,
    retrieval_temperature: float | None = None,
) -> Dict[str, Any]:
    """Retrieve source prototypes and return variable-wise expert weights."""
    validate_rise_metadata_no_target_eval_selection(prior)
    if prior.get("schema_version") != RISE_ROUTER_SCHEMA:
        raise ValueError(f"Unsupported RISE router prior schema_version={prior.get('schema_version')!r}")
    prototypes = list(prior.get("prototypes", []))
    if not prototypes:
        raise ValueError("RISE router prior has no prototypes")
    expert_ids = [str(eid) for eid in prior.get("expert_ids", [])]
    if not expert_ids:
        expert_ids = _expert_ids(prior.get("candidates", []))
    query = np.asarray(descriptor, dtype=np.float32).reshape(-1)
    if query.size == 0 or not np.all(np.isfinite(query)):
        raise ValueError("query descriptor must be a finite non-empty vector")
    temperature = float(retrieval_temperature or prior.get("temperature", 0.2))
    if temperature <= 0:
        raise ValueError("retrieval_temperature must be positive")

    distance_scores: Dict[str, float] = {}
    proto_vectors: list[np.ndarray] = []
    for idx, prototype in enumerate(prototypes):
        vector = np.asarray(prototype["descriptor"], dtype=np.float32).reshape(-1)
        if vector.shape != query.shape:
            raise ValueError(f"prototype descriptor shape {vector.shape} does not match query {query.shape}")
        proto_vectors.append(vector)
        distance = float(np.linalg.norm(query - vector))
        distance_scores[str(idx)] = -distance
    proto_weights = _softmax(distance_scores, temperature=temperature)

    weights: Dict[str, Dict[str, float]] = {}
    uncertainty: Dict[str, float] = {}
    for variable in VARIABLES:
        accum = {eid: 0.0 for eid in expert_ids}
        uncertainty_value = 0.0
        for idx, prototype in enumerate(prototypes):
            retrieval_weight = float(proto_weights[str(idx)])
            proto_var_weights = prototype["weights"][variable]
            for eid in expert_ids:
                accum[eid] += retrieval_weight * float(proto_var_weights.get(eid, 0.0))
            uncertainty_value += retrieval_weight * float(prototype.get("uncertainty", {}).get(variable, 0.5))
        weights[variable] = _normalise_weights(accum, expert_ids=expert_ids)
        uncertainty[variable] = float(uncertainty_value)

    return {
        "schema_version": "hyperda_rise_router_weights_v1",
        "method": RISE_METHOD,
        "weights": weights,
        "uncertainty": uncertainty,
        "retrieval_temperature": temperature,
        "source_prior_hash": _stable_hash(
            {
                "schema_version": prior.get("schema_version"),
                "expert_ids": expert_ids,
                "prototypes": prototypes,
            }
        ),
        "no_leakage_declaration": {
            "target_context_labels_used": False,
            "target_support_labels_used_for_router_weights": False,
            "target_eval_used_for_router_weights": False,
            "target_eval_used_for_expert_selection": False,
        },
    }


def validate_support_budget(support_samples: Sequence[Mapping[str, Any]], *, K: int) -> None:
    """Validate that K counts labeled support DA cycles, not pixels."""
    if int(K) not in (0, 4, 12):
        raise ValueError(f"RISE supports K in {{0,4,12}}, got {K}")
    if int(K) == 0 and len(support_samples) != 0:
        raise ValueError("K=0 RISE path must not receive target support labels")
    if len(support_samples) > int(K):
        raise ValueError(
            f"K={K} allows at most {K} labeled support DA cycles; "
            f"got {len(support_samples)} samples"
        )
    seen_cycles: set[tuple[str, int]] = set()
    for idx, sample in enumerate(support_samples):
        cycle = (str(sample.get("date_str", "")), int(sample.get("time_index", idx)))
        if cycle in seen_cycles:
            raise ValueError(f"duplicate support DA cycle detected: {cycle}")
        seen_cycles.add(cycle)


def _extract_true_increment(sample: Mapping[str, Any], variable: str) -> np.ndarray:
    for key in (f"true_increment_{variable}", f"increment_{variable}"):
        if key in sample:
            return np.asarray(sample[key], dtype=np.float32)
    raise ValueError(f"support sample missing true increment for {variable}")


def _extract_support_prediction(sample: Mapping[str, Any], expert_id: str, variable: str) -> np.ndarray:
    predictions = sample.get("expert_predictions")
    if not isinstance(predictions, Mapping) or expert_id not in predictions:
        raise ValueError(f"support sample missing expert_predictions for expert {expert_id!r}")
    pred = predictions[expert_id]
    if f"pred_increment_{variable}" not in pred:
        raise ValueError(f"expert {expert_id!r} support prediction missing pred_increment_{variable}")
    return np.asarray(pred[f"pred_increment_{variable}"], dtype=np.float32)


def _support_mask(sample: Mapping[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    mask = np.asarray(sample.get("metric_mask", np.ones(shape, dtype=np.float32)), dtype=np.float32)
    if mask.shape != shape:
        raise ValueError(f"support metric_mask shape {mask.shape} does not match increment shape {shape}")
    return (mask > 0.5) & np.isfinite(mask)


def _solve_variable_weights(
    *,
    prior_weights: Mapping[str, float],
    support_samples: Sequence[Mapping[str, Any]],
    expert_ids: Sequence[str],
    variable: str,
    ridge_lambda: float,
) -> tuple[Dict[str, float], float, float, Dict[str, Any]]:
    if not support_samples:
        return _normalise_weights(prior_weights, expert_ids=expert_ids), 1.0, 0.0, {
            "n_support_cycles": 0,
            "n_equations": 0,
        }
    rows = []
    targets = []
    for sample in support_samples:
        target = _extract_true_increment(sample, variable)
        mask = _support_mask(sample, target.shape)
        if not np.any(mask):
            continue
        preds = [_extract_support_prediction(sample, eid, variable) for eid in expert_ids]
        pred_matrix = np.stack([pred[mask] for pred in preds], axis=1).astype(np.float64)
        rows.append(pred_matrix)
        targets.append(target[mask].astype(np.float64))
    if not rows:
        return _normalise_weights(prior_weights, expert_ids=expert_ids), 1.0, 0.0, {
            "n_support_cycles": len(support_samples),
            "n_equations": 0,
        }
    x = np.concatenate(rows, axis=0)
    y = np.concatenate(targets, axis=0)
    if x.shape[0] == 0:
        raise ValueError("support posterior has zero valid equations")
    prior = np.asarray([float(prior_weights.get(eid, 0.0)) for eid in expert_ids], dtype=np.float64)
    prior = prior / max(1e-12, float(np.sum(prior)))
    regularizer = max(0.0, float(ridge_lambda))
    xtx = x.T @ x + regularizer * np.eye(len(expert_ids), dtype=np.float64)
    rhs = x.T @ y + regularizer * prior
    try:
        coef = np.linalg.solve(xtx, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(xtx, rhs, rcond=None)[0]
    coef = np.clip(coef, 0.0, None)
    if float(np.sum(coef)) <= 0.0:
        coef = prior
    weights = {eid: float(coef[idx]) for idx, eid in enumerate(expert_ids)}
    weights = _normalise_weights(weights, expert_ids=expert_ids)

    mixed = x @ np.asarray([weights[eid] for eid in expert_ids], dtype=np.float64)
    denom = float(np.sum(mixed * mixed)) + regularizer
    gain = float((np.sum(mixed * y) + regularizer * 1.0) / denom) if denom > 0 else 1.0
    residual = y - gain * mixed
    bias = float(np.mean(residual)) if residual.size else 0.0
    diagnostics = {
        "n_support_cycles": len(support_samples),
        "n_equations": int(x.shape[0]),
        "ridge_lambda": float(ridge_lambda),
        "prior_weights": {eid: float(prior[idx]) for idx, eid in enumerate(expert_ids)},
    }
    return weights, gain, bias, diagnostics


def solve_support_posterior(
    *,
    prior_weights: Mapping[str, Mapping[str, float]],
    support_samples: Sequence[Mapping[str, Any]],
    K: int,
    ridge_lambda: float = 1.0,
    support_reliability: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Solve a low-dimensional RISE K-shot posterior from support labels."""
    validate_support_budget(support_samples, K=K)
    if int(K) == 0:
        if support_samples:
            raise ValueError("K=0 RISE posterior cannot use support labels")
        return {
            "schema_version": RISE_POSTERIOR_SCHEMA,
            "method": RISE_METHOD,
            "method_id": "hyperda_rise_k0_context_router",
            "K": 0,
            "weights": {variable: dict(prior_weights[variable]) for variable in VARIABLES},
            "gain": {variable: 1.0 for variable in VARIABLES},
            "bias": {variable: 0.0 for variable in VARIABLES},
            "updated_parameter_keys": [],
            "frozen_parameter_keys": list(FROZEN_PARAMETER_KEYS),
            "support_label_usage": "none",
            "support_reliability": dict(support_reliability or {}),
            "no_leakage_declaration": _posterior_no_leakage_declaration(target_support_used=False),
        }
    if len(support_samples) == 0:
        raise ValueError(f"K={K} RISE posterior requires at least one support DA cycle")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be non-negative")
    expert_ids = sorted(set().union(*(set(prior_weights[variable]) for variable in VARIABLES)))
    if not expert_ids:
        raise ValueError("prior_weights contain no experts")

    weights: Dict[str, Dict[str, float]] = {}
    gain: Dict[str, float] = {}
    bias: Dict[str, float] = {}
    diagnostics: Dict[str, Any] = {}
    for variable in VARIABLES:
        w, g, b, diag = _solve_variable_weights(
            prior_weights=prior_weights[variable],
            support_samples=support_samples,
            expert_ids=expert_ids,
            variable=variable,
            ridge_lambda=ridge_lambda,
        )
        weights[variable] = w
        gain[variable] = float(g)
        bias[variable] = float(b)
        diagnostics[variable] = diag

    return {
        "schema_version": RISE_POSTERIOR_SCHEMA,
        "method": RISE_METHOD,
        "method_id": f"hyperda_rise_k{int(K)}_support_posterior",
        "K": int(K),
        "weights": weights,
        "gain": gain,
        "bias": bias,
        "monthly_gain": {},
        "monthly_bias": {},
        "ridge_lambda": float(ridge_lambda),
        "updated_parameter_keys": list(LOW_DIM_UPDATED_PARAMETER_KEYS),
        "frozen_parameter_keys": list(FROZEN_PARAMETER_KEYS),
        "support_label_usage": f"up_to_{int(K)}_labeled_target_DA_cycles",
        "support_reliability": dict(support_reliability or {}),
        "diagnostics": diagnostics,
        "no_leakage_declaration": _posterior_no_leakage_declaration(target_support_used=True),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def _posterior_no_leakage_declaration(*, target_support_used: bool) -> Dict[str, Any]:
    return {
        "target_context_labels_used": False,
        "target_support_labels_used": bool(target_support_used),
        "target_support_used_for_low_dimensional_posterior_only": bool(target_support_used),
        "target_val_used_for_posterior": False,
        "target_eval_used_for_posterior": False,
        "target_eval_used_for_router_weights": False,
        "target_eval_used_for_reliability": False,
        "target_eval_used_for_lambda_temperature_normalization_or_expert_selection": False,
        "backbone_updated": False,
        "hyperda_basis_updated": False,
        "prompt_encoder_backbone_updated": False,
    }


def build_posterior_config(
    *,
    K: int,
    prior_weights: Mapping[str, Mapping[str, float]],
    support_samples: Sequence[Mapping[str, Any]],
    ridge_lambda: float,
    temperature: float,
    support_reliability: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Build the serializable K0/K-shot posterior config."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    posterior = solve_support_posterior(
        prior_weights=prior_weights,
        support_samples=support_samples,
        K=K,
        ridge_lambda=ridge_lambda,
        support_reliability=support_reliability,
    )
    posterior["temperature"] = float(temperature)
    posterior["model_selection_source"] = "source_val_2022_preregistered"
    posterior["normalization_source"] = "source_fit_only"
    return posterior


def _require_prediction_keys(prediction: Mapping[str, Any], expert_id: str) -> None:
    missing = [
        key
        for key in (
            "pred_increment_surface",
            "pred_increment_rootzone",
        )
        if key not in prediction
    ]
    if missing:
        raise KeyError(f"RISE expert {expert_id!r} output missing keys: {missing}")


class ExpertMixturePredictor:
    """Late-fusion variable-wise mixture over frozen expert predictors."""

    method_name = "hyperda_rise_expert_mixture"

    def __init__(
        self,
        *,
        experts: Mapping[str, Any],
        weights: Mapping[str, Mapping[str, float]],
        gain: Mapping[str, float] | None = None,
        bias: Mapping[str, float] | None = None,
        monthly_gain: Mapping[str, Mapping[str, float]] | None = None,
        monthly_bias: Mapping[str, Mapping[str, float]] | None = None,
        method_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.experts = dict(experts)
        if not self.experts:
            raise ValueError("ExpertMixturePredictor requires at least one expert")
        expert_ids = sorted(self.experts)
        self.weights = {
            variable: _normalise_weights(weights.get(variable, {}), expert_ids=expert_ids)
            for variable in VARIABLES
        }
        self.gain = {variable: float((gain or {}).get(variable, 1.0)) for variable in VARIABLES}
        self.bias = {variable: float((bias or {}).get(variable, 0.0)) for variable in VARIABLES}
        self.monthly_gain = {
            variable: {str(month): float(value) for month, value in (monthly_gain or {}).get(variable, {}).items()}
            for variable in VARIABLES
        }
        self.monthly_bias = {
            variable: {str(month): float(value) for month, value in (monthly_bias or {}).get(variable, {}).items()}
            for variable in VARIABLES
        }
        if method_name:
            self.method_name = str(method_name)
        self.metadata = dict(metadata or {})

    def _variable_gain_bias(self, variable: str, sample: Mapping[str, Any]) -> tuple[float, float]:
        month = str(int(sample.get("month", 0))) if sample.get("month", None) is not None else ""
        gain = self.gain[variable] * float(self.monthly_gain[variable].get(month, 1.0))
        bias = self.bias[variable] + float(self.monthly_bias[variable].get(month, 0.0))
        return gain, bias

    def predict(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        expert_predictions = {
            expert_id: predictor.predict(sample)
            for expert_id, predictor in self.experts.items()
        }
        for expert_id, prediction in expert_predictions.items():
            _require_prediction_keys(prediction, expert_id)
        forecast_surface = np.asarray(sample["forecast_surface"], dtype=np.float32)
        forecast_rootzone = np.asarray(sample["forecast_rootzone"], dtype=np.float32)
        output: Dict[str, Any] = {}
        for variable, forecast in (("surface", forecast_surface), ("rootzone", forecast_rootzone)):
            inc = np.zeros_like(forecast, dtype=np.float32)
            for expert_id, weight in self.weights[variable].items():
                pred = expert_predictions[expert_id][f"pred_increment_{variable}"]
                inc = inc + float(weight) * np.asarray(pred, dtype=np.float32)
            gain, bias = self._variable_gain_bias(variable, sample)
            inc = (gain * inc + bias).astype(np.float32)
            output[f"pred_increment_{variable}"] = inc
            output[f"pred_analysis_{variable}"] = (forecast + inc).astype(np.float32)
            output[f"rise_{variable}_weights"] = dict(self.weights[variable])
            output[f"rise_{variable}_gain"] = float(gain)
            output[f"rise_{variable}_bias"] = float(bias)
        return output


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, default=_json_default)
    return out


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_router_prior(path: str | Path) -> Dict[str, Any]:
    prior = load_json(path)
    if prior.get("schema_version") != RISE_ROUTER_SCHEMA:
        raise ValueError(f"{path} is not a {RISE_ROUTER_SCHEMA} router prior")
    validate_rise_metadata_no_target_eval_selection(prior)
    return prior


def load_posterior_config(path: str | Path) -> Dict[str, Any]:
    posterior = load_json(path)
    if posterior.get("schema_version") != RISE_POSTERIOR_SCHEMA:
        raise ValueError(f"{path} is not a {RISE_POSTERIOR_SCHEMA} posterior config")
    validate_rise_metadata_no_target_eval_selection(posterior)
    return posterior


def validate_rise_metadata_no_target_eval_selection(metadata: Mapping[str, Any]) -> None:
    """Reject RISE metadata that declares target_eval-dependent selection."""
    declaration = metadata.get("no_leakage_declaration", {})
    if not isinstance(declaration, Mapping):
        raise ValueError("RISE metadata missing no_leakage_declaration mapping")
    forbidden = {
        "target_eval_used_for_router_training",
        "target_eval_used_for_router_weights",
        "target_eval_used_for_expert_selection",
        "target_eval_used_for_posterior",
        "target_eval_used_for_reliability",
        "target_eval_used_for_lambda_temperature_normalization_or_expert_selection",
        "dynamic_target_eval_gating",
    }
    bad = [key for key in sorted(forbidden) if declaration.get(key) not in (None, False)]
    if bad:
        raise ValueError(f"RISE metadata declares forbidden target_eval usage: {bad}")


def candidate_metrics_to_episodes(
    metrics: Any,
    *,
    descriptor_by_region: Mapping[str, Sequence[float]],
    expert_ids: Sequence[str],
    metric_name: str = "increment_rmse_latw",
    split_role: str = "source_val",
) -> list[Dict[str, Any]]:
    """Convert a long candidate-metrics table into router-prior episodes."""
    import pandas as pd

    df = pd.DataFrame(metrics)
    required = {"pseudo_target_region_id", "candidate_id", "split_role", "variable", "metric", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"candidate metrics missing columns: {sorted(missing)}")
    episodes = []

    def _candidate_metric_map(rows: Any, variable: str) -> Dict[str, float]:
        var_rows = rows[rows["variable"].astype(str) == variable].copy()
        primary = var_rows[var_rows["metric"].astype(str) == metric_name].copy()
        primary["value"] = pd.to_numeric(primary["value"], errors="coerce")
        primary = primary[np.isfinite(primary["value"])]
        grouped = primary.groupby("candidate_id", as_index=False)["value"].mean()
        values = {str(row["candidate_id"]): float(row["value"]) for row in grouped.to_dict("records")}

        if metric_name == "increment_rmse_latw":
            fallback = var_rows[var_rows["metric"].astype(str) == "increment_mse_latw"].copy()
            fallback["value"] = pd.to_numeric(fallback["value"], errors="coerce")
            fallback = fallback[np.isfinite(fallback["value"])]
            fallback_grouped = fallback.groupby("candidate_id", as_index=False)["value"].mean()
            for row in fallback_grouped.to_dict("records"):
                candidate_id = str(row["candidate_id"])
                values.setdefault(candidate_id, float(np.sqrt(max(0.0, float(row["value"])))))
        return values

    for region_id, descriptor in descriptor_by_region.items():
        region_rows = df[
            (df["pseudo_target_region_id"].astype(str) == str(region_id))
            & (df["split_role"].astype(str) == split_role)
        ]
        if region_rows.empty:
            raise ValueError(f"No candidate metrics for pseudo-target region {region_id}")
        expert_metrics: Dict[str, Dict[str, float]] = {}
        for variable in VARIABLES:
            metrics_by_id = _candidate_metric_map(region_rows, variable)
            missing_ids = [eid for eid in expert_ids if eid not in metrics_by_id]
            if missing_ids:
                raise ValueError(f"Missing {variable} metrics for {region_id}: {missing_ids}")
            expert_metrics[variable] = {eid: metrics_by_id[eid] for eid in expert_ids}
        episodes.append(
            {
                "pseudo_target_region_id": str(region_id),
                "descriptor": list(np.asarray(descriptor, dtype=np.float32).reshape(-1).astype(float)),
                "metric_split_role": split_role,
                "source_val_metric": metric_name,
                "expert_metrics": expert_metrics,
            }
        )
    return episodes


def support_predictions_from_predictors(
    *,
    support_dataset: Any,
    experts: Mapping[str, Any],
    max_samples: int | None = None,
) -> list[Dict[str, Any]]:
    """Materialize support labels and frozen expert predictions for posterior solve."""
    n_samples = len(support_dataset) if max_samples is None else min(len(support_dataset), max_samples)
    out = []
    for idx in range(n_samples):
        sample = support_dataset[idx]
        entry = {
            "date_str": sample.get("date_str", ""),
            "time_index": int(sample.get("time_index", idx)),
            "month": sample.get("month", None),
            "metric_mask": np.asarray(sample["metric_mask"], dtype=np.float32),
            "true_increment_surface": np.asarray(sample["increment_surface"], dtype=np.float32),
            "true_increment_rootzone": np.asarray(sample["increment_rootzone"], dtype=np.float32),
            "expert_predictions": {},
        }
        for expert_id, predictor in experts.items():
            pred = predictor.predict(sample)
            _require_prediction_keys(pred, expert_id)
            entry["expert_predictions"][expert_id] = {
                "pred_increment_surface": np.asarray(pred["pred_increment_surface"], dtype=np.float32),
                "pred_increment_rootzone": np.asarray(pred["pred_increment_rootzone"], dtype=np.float32),
            }
        out.append(entry)
    return out


def build_support_reliability_rows(
    support_samples: Sequence[Mapping[str, Any]],
    *,
    K: int,
) -> list[Dict[str, Any]]:
    """Return simple source-preregistered support reliability diagnostics."""
    validate_support_budget(support_samples, K=K)
    rows = []
    for idx, sample in enumerate(support_samples):
        mask = np.asarray(sample.get("metric_mask", []), dtype=np.float32)
        valid_fraction = float(np.mean(mask > 0.5)) if mask.size else 0.0
        rows.append(
            {
                "support_index": idx,
                "date_str": str(sample.get("date_str", "")),
                "time_index": int(sample.get("time_index", idx)),
                "K": int(K),
                "valid_fraction": valid_fraction,
                "reliability_weight": 1.0 if valid_fraction > 0.0 else 0.0,
                "selection_source": "calendar_and_support_valid_mask_only",
                "target_eval_used": False,
            }
        )
    return rows


def posterior_config_for_eval(
    *,
    posterior: Mapping[str, Any],
    method_id: str | None = None,
) -> Dict[str, Any]:
    """Normalize a posterior config for constructing an eval predictor."""
    validate_rise_metadata_no_target_eval_selection(posterior)
    return {
        "method_id": method_id or str(posterior.get("method_id", "hyperda_rise")),
        "weights": {variable: dict(posterior["weights"][variable]) for variable in VARIABLES},
        "gain": {variable: float(posterior.get("gain", {}).get(variable, 1.0)) for variable in VARIABLES},
        "bias": {variable: float(posterior.get("bias", {}).get(variable, 0.0)) for variable in VARIABLES},
        "monthly_gain": {
            variable: dict(posterior.get("monthly_gain", {}).get(variable, {}))
            for variable in VARIABLES
        },
        "monthly_bias": {
            variable: dict(posterior.get("monthly_bias", {}).get(variable, {}))
            for variable in VARIABLES
        },
    }
