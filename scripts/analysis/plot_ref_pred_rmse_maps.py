#!/usr/bin/env python3
"""Plot Ref / Pred / error maps from HydroDA prediction records.

The figure is an evaluation-only paper artifact. Ref and Pred are analysis-space
soil-moisture fields reconstructed from the physical DA analysis and the
forecast plus model-predicted increment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import xarray as xr

from hydroda.evaluation.harness import prediction_record_array

try:  # Cartopy is optional; the script falls back to plain grid panels.
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:  # pragma: no cover - exercised only in environments without cartopy
    ccrs = None
    cfeature = None


DEFAULT_TARGET_REGION = "US-R1"
DEFAULT_CANDIDATE_ID = "M3_1_K0"
DEFAULT_MODEL_LABEL = "M3_1_hyperda_trust_medium / HyperDA-TRUST K=0"
DEFAULT_RUN_DIR = Path(
    "artifacts/runs/stage3_hyperda_posterior/"
    "US-R1_s0_stage2_full_inference_20260620T091256Z"
)
DEFAULT_CHECKPOINT = (
    DEFAULT_RUN_DIR / "K0" / "adapt" / "checkpoints" / "checkpoint_final_preregistered.pt"
)
DEFAULT_SPLITS_JSON = Path("artifacts/splits/US_loro_zero_few_shot_splits.json")
DEFAULT_GEOLOCATION_NC = Path("artifacts/geolocation/US_latlon.nc")
DEFAULT_CROP_MANIFEST = Path("artifacts/region_crops/US/manifest_region_crops_US.json")
DEFAULT_REGION_MASK_NC = Path("artifacts/regions/US_region_masks.nc")
DEFAULT_OUTPUT_DIR = Path("artifacts/figures/ref_pred_rmse_maps/us_r1_m3_1_k0")
DEFAULT_MAX_SAMPLES = 512
VARIABLE_ORDER = ("surface", "rootzone")
VARIABLE_LABELS = {
    "surface": "Surface SM",
    "rootzone": "Rootzone SM",
}
COLORBAR_STYLE = {
    "width": "2.4%",
    "height": "68%",
    "x_offset": 1.03,
    "tick_labelsize": 5.4,
    "tick_width": 0.35,
    "tick_length": 1.2,
    "outline_width": 0.45,
}
VISUAL_SELECTION_WEIGHTS = {
    "model_wrmse_rank": 1.0,
    "edge_concentration_rank": 1.1,
    "local_skill_rank": 0.45,
    "combined_skill_rank": 0.15,
}
CANDIDATE_TABLE_FIELDS = [
    "selection_rank",
    "selected",
    "eligible_for_balanced_visual_selection",
    "positive_two_variable_skill",
    "signal_eligible",
    "skill_eligible",
    "sample_idx",
    "query_date",
    "combined_model_wrmse",
    "combined_relative_skill_vs_forecast",
    "true_increment_strength",
    "balanced_visual_score",
    "left_error_rmse_ratio_vs_full",
    "lower_left_error_rmse_ratio_vs_full",
    "edge_error_concentration",
    "left_skill_vs_forecast",
    "lower_left_skill_vs_forecast",
    "local_skill_floor",
    "model_wrmse_rank",
    "edge_concentration_rank",
    "local_skill_rank",
    "combined_skill_rank",
    "surface_model_wrmse",
    "surface_relative_skill_vs_forecast",
    "surface_true_increment_wrmse",
    "rootzone_model_wrmse",
    "rootzone_relative_skill_vs_forecast",
    "rootzone_true_increment_wrmse",
]


@dataclass
class VariableMap:
    name: str
    ref: np.ndarray
    pred: np.ndarray
    forecast: np.ndarray
    rmse_map: np.ndarray
    model_wrmse: float
    forecast_wrmse: float
    skill_vs_forecast: float
    true_increment_wrmse: float


@dataclass
class SampleMap:
    sample_idx: int
    query_time_index: int
    query_date: str
    month: Any
    season: str
    target_region_id: str
    split_role: str
    method: str
    record_metadata: dict[str, Any]
    mask: np.ndarray
    latitude_weight: np.ndarray
    variables: dict[str, VariableMap]
    selection_reason: str = ""
    selection_score: float = float("nan")
    selection_rank_key: tuple[Any, ...] | None = field(default=None, repr=False)
    selection_metadata: dict[str, Any] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    """Load HydroDA prediction records from JSONL or a JSON ``records`` object."""
    record_path = Path(path)
    if not record_path.exists():
        raise FileNotFoundError(f"Prediction record path not found: {record_path}")
    if record_path.suffix == ".json":
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
            records = payload["records"]
        else:
            raise ValueError(f"JSON prediction record file has no records list: {record_path}")
    else:
        records = [
            json.loads(line)
            for line in record_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not records:
        raise ValueError(f"No prediction records loaded from {record_path}")
    return [dict(record) for record in records]


def _decode_required_array(arrays: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in arrays:
        raise KeyError(f"Prediction record arrays missing required key: {key}")
    return np.asarray(prediction_record_array(dict(arrays[key])), dtype=np.float32)


def _valid_mask(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
    latitude_weight: np.ndarray,
) -> np.ndarray:
    return (
        (np.asarray(mask) > 0.5)
        & np.isfinite(pred)
        & np.isfinite(ref)
        & np.isfinite(latitude_weight)
        & (latitude_weight >= 0)
    )


def weighted_rmse(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
    latitude_weight: np.ndarray,
) -> float:
    """Latitude-weighted RMSE over finite valid pixels."""
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    weights = np.asarray(latitude_weight, dtype=np.float64)
    valid = _valid_mask(pred, ref, mask, weights)
    if int(valid.sum()) == 0:
        return float("nan")
    valid_weights = weights[valid]
    if float(valid_weights.sum()) <= 0.0:
        return float("nan")
    mse = np.average((pred[valid] - ref[valid]) ** 2, weights=valid_weights)
    return float(np.sqrt(mse))


def _masked_array(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.asarray(values, dtype=np.float32).copy()
    masked[~valid] = np.nan
    return masked


def _build_variable_map(
    *,
    name: str,
    ref: np.ndarray,
    pred: np.ndarray,
    forecast: np.ndarray,
    mask: np.ndarray,
    latitude_weight: np.ndarray,
) -> VariableMap:
    valid = _valid_mask(pred, ref, mask, latitude_weight)
    ref_masked = _masked_array(ref, valid)
    pred_masked = _masked_array(pred, valid)
    forecast_masked = _masked_array(forecast, valid)
    rmse_map = _masked_array(np.abs(pred - ref), valid)
    model_wrmse = weighted_rmse(pred, ref, mask, latitude_weight)
    forecast_wrmse = weighted_rmse(forecast, ref, mask, latitude_weight)
    skill = (
        float(1.0 - model_wrmse / forecast_wrmse)
        if np.isfinite(model_wrmse) and np.isfinite(forecast_wrmse) and forecast_wrmse > 0.0
        else float("nan")
    )
    return VariableMap(
        name=name,
        ref=ref_masked,
        pred=pred_masked,
        forecast=forecast_masked,
        rmse_map=rmse_map,
        model_wrmse=model_wrmse,
        forecast_wrmse=forecast_wrmse,
        skill_vs_forecast=skill,
        true_increment_wrmse=forecast_wrmse,
    )


def build_sample_map(record: Mapping[str, Any]) -> SampleMap:
    """Decode one prediction record into masked analysis-space Ref/Pred/error arrays."""
    if record.get("schema_version") != "hydroda_prediction_record_v1":
        raise ValueError(
            "Expected schema_version='hydroda_prediction_record_v1', "
            f"got {record.get('schema_version')!r}"
        )
    arrays = dict(record.get("arrays") or {})
    mask = _decode_required_array(arrays, "metric_mask")
    latitude_weight = _decode_required_array(arrays, "latitude_weight")
    variables: dict[str, VariableMap] = {}
    for variable in VARIABLE_ORDER:
        forecast = _decode_required_array(arrays, f"forecast_{variable}")
        ref = _decode_required_array(arrays, f"analysis_{variable}")
        pred_increment = _decode_required_array(arrays, f"pred_increment_{variable}")
        pred = forecast + pred_increment
        variables[variable] = _build_variable_map(
            name=variable,
            ref=ref,
            pred=pred,
            forecast=forecast,
            mask=mask,
            latitude_weight=latitude_weight,
        )
    metadata_keys = [
        "adaptation_setting",
        "K",
        "seed",
        "protocol_freeze_id",
        "split_file",
        "split_manifest_sha256",
        "target_context_dates_hash",
        "target_support_dates_hash",
        "support_dates_hash",
        "target_train_dates_hash",
        "target_eval_dates_hash",
        "prediction_content_hash",
        "adapt_mix_rho",
    ]
    return SampleMap(
        sample_idx=int(record.get("sample_idx", -1)),
        query_time_index=int(record.get("query_time_index", -1)),
        query_date=str(record.get("query_date", "")),
        month=record.get("month", None),
        season=str(record.get("season", "")),
        target_region_id=str(record.get("target_region_id", "")),
        split_role=str(record.get("split_role", "")),
        method=str(record.get("method", "")),
        record_metadata={key: record.get(key, "") for key in metadata_keys},
        mask=mask,
        latitude_weight=latitude_weight,
        variables=variables,
    )


def _combined_true_increment_strength(sample: SampleMap) -> float:
    strengths = [
        sample.variables[name].true_increment_wrmse
        for name in VARIABLE_ORDER
        if np.isfinite(sample.variables[name].true_increment_wrmse)
    ]
    if not strengths:
        return float("-inf")
    return float(np.mean(strengths))


def _combined_skill(sample: SampleMap) -> float:
    skills = [
        sample.variables[name].skill_vs_forecast
        for name in VARIABLE_ORDER
        if np.isfinite(sample.variables[name].skill_vs_forecast)
    ]
    if not skills:
        return float("-inf")
    return float(np.mean(skills))


def _combined_model_wrmse(sample: SampleMap) -> float:
    wrmses = [
        sample.variables[name].model_wrmse
        for name in VARIABLE_ORDER
        if np.isfinite(sample.variables[name].model_wrmse)
    ]
    if not wrmses:
        return float("inf")
    return float(np.mean(wrmses))


def _has_positive_two_variable_skill(sample: SampleMap) -> bool:
    return all(
        np.isfinite(sample.variables[name].skill_vs_forecast)
        and sample.variables[name].skill_vs_forecast > 0.0
        for name in VARIABLE_ORDER
    )


def _selection_candidate_summary(sample: SampleMap) -> dict[str, Any]:
    return {
        "sample_idx": int(sample.sample_idx),
        "query_date": str(sample.query_date),
        "combined_relative_skill_vs_forecast": _combined_skill(sample),
        "combined_model_wrmse": _combined_model_wrmse(sample),
        "true_increment_strength": _combined_true_increment_strength(sample),
    }


def _legacy_high_true_increment_key(sample: SampleMap) -> tuple[float, float, str, int]:
    return (
        _combined_true_increment_strength(sample),
        _combined_skill(sample),
        str(sample.query_date),
        -int(sample.sample_idx),
    )


def _best_skill_rank_key(sample: SampleMap) -> tuple[float, float, float, str, int]:
    return (
        _combined_skill(sample),
        -_combined_model_wrmse(sample),
        _combined_true_increment_strength(sample),
        str(sample.query_date),
        -int(sample.sample_idx),
    )


def _lowest_rmse_rank_key(sample: SampleMap) -> tuple[float, float, float, str, int]:
    return (
        -_combined_model_wrmse(sample),
        _combined_skill(sample),
        _combined_true_increment_strength(sample),
        str(sample.query_date),
        -int(sample.sample_idx),
    )


def _finite_or_nan(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def _nanmean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _nanmin(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.min(finite)) if finite else float("nan")


def _windowed_error_stats(sample: SampleMap, variable: str, window_mask: np.ndarray) -> dict[str, float]:
    variable_map = sample.variables[variable]
    model_error = np.abs(variable_map.pred - variable_map.ref)
    forecast_error = np.abs(variable_map.forecast - variable_map.ref)
    metric_mask = (np.asarray(sample.mask) > 0.5) & np.asarray(window_mask, dtype=bool)
    return _sample_weighted_error_stats(
        model_error=model_error,
        forecast_error=forecast_error,
        mask=metric_mask.astype(np.float32),
        latitude_weight=sample.latitude_weight,
    )


def _sample_visual_selection_metrics(
    sample: SampleMap,
    *,
    edge_fraction: float = 0.25,
    lower_left_fraction: float = 0.25,
) -> dict[str, float]:
    """Return compact spatial-error diagnostics used only for display-case ranking."""
    if not 0.0 < float(edge_fraction) <= 1.0:
        raise ValueError(f"edge_fraction must be in (0, 1], got {edge_fraction}")
    shape = tuple(np.asarray(sample.mask).shape)
    height, width = shape
    left_width = max(1, int(np.ceil(width * float(edge_fraction))))
    left_mask = np.zeros(shape, dtype=bool)
    left_mask[:, :left_width] = True
    lower_left = _window_mask(shape, _lower_left_window(shape, lower_left_fraction))
    full_mask = np.ones(shape, dtype=bool)

    left_ratios: list[float] = []
    lower_left_ratios: list[float] = []
    left_skills: list[float] = []
    lower_left_skills: list[float] = []
    for variable in VARIABLE_ORDER:
        full = _windowed_error_stats(sample, variable, full_mask)
        left = _windowed_error_stats(sample, variable, left_mask)
        lower = _windowed_error_stats(sample, variable, lower_left)
        full_rmse = _finite_or_nan(full.get("model_rmse"))
        left_rmse = _finite_or_nan(left.get("model_rmse"))
        lower_rmse = _finite_or_nan(lower.get("model_rmse"))
        left_ratios.append(
            float(left_rmse / full_rmse)
            if np.isfinite(left_rmse) and np.isfinite(full_rmse) and full_rmse > 0.0
            else float("nan")
        )
        lower_left_ratios.append(
            float(lower_rmse / full_rmse)
            if np.isfinite(lower_rmse) and np.isfinite(full_rmse) and full_rmse > 0.0
            else float("nan")
        )
        left_skills.append(_finite_or_nan(left.get("mean_skill_vs_forecast")))
        lower_left_skills.append(_finite_or_nan(lower.get("mean_skill_vs_forecast")))

    left_ratio = _nanmean(left_ratios)
    lower_left_ratio = _nanmean(lower_left_ratios)
    edge_concentration = (
        float(np.nanmax([left_ratio, lower_left_ratio]))
        if np.isfinite([left_ratio, lower_left_ratio]).any()
        else float("nan")
    )
    local_skill_floor = _nanmin([*left_skills, *lower_left_skills])
    return {
        "left_error_rmse_ratio_vs_full": left_ratio,
        "lower_left_error_rmse_ratio_vs_full": lower_left_ratio,
        "edge_error_concentration": edge_concentration,
        "left_skill_vs_forecast": _nanmean(left_skills),
        "lower_left_skill_vs_forecast": _nanmean(lower_left_skills),
        "local_skill_floor": local_skill_floor,
    }


def _numeric_ranks(values: Sequence[float], *, higher_is_better: bool) -> list[int]:
    numeric_values = [_finite_or_nan(value) for value in values]

    def sort_key(index: int) -> tuple[int, float, int]:
        value = numeric_values[index]
        if not np.isfinite(value):
            return (1, 0.0, index)
        sortable = -value if higher_is_better else value
        return (0, sortable, index)

    order = sorted(range(len(values)), key=sort_key)
    ranks = [len(values)] * len(values)
    previous_value = float("nan")
    previous_rank = 1
    finite_seen = 0
    for rank, index in enumerate(order, start=1):
        value = numeric_values[index]
        if not np.isfinite(value):
            ranks[index] = len(values)
            continue
        finite_seen += 1
        if finite_seen == 1:
            previous_rank = rank
        elif not np.isclose(value, previous_value, rtol=1e-9, atol=1e-12):
            previous_rank = rank
        ranks[index] = previous_rank
        previous_value = value
    return ranks


def _visual_candidate_rank_rows(samples: Sequence[SampleMap]) -> dict[tuple[int, int, str], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        metrics = _sample_visual_selection_metrics(sample)
        row = {
            "sample_key": _sample_key(sample),
            "sample": sample,
            "combined_model_wrmse": _combined_model_wrmse(sample),
            "combined_relative_skill_vs_forecast": _combined_skill(sample),
            "true_increment_strength": _combined_true_increment_strength(sample),
            **metrics,
        }
        rows.append(row)
    if not rows:
        return {}

    rank_specs = (
        ("model_wrmse_rank", "combined_model_wrmse", False),
        ("edge_concentration_rank", "edge_error_concentration", False),
        ("local_skill_rank", "local_skill_floor", True),
        ("combined_skill_rank", "combined_relative_skill_vs_forecast", True),
    )
    for rank_name, value_name, higher_is_better in rank_specs:
        ranks = _numeric_ranks([row[value_name] for row in rows], higher_is_better=higher_is_better)
        for row, rank in zip(rows, ranks):
            row[rank_name] = int(rank)

    for row in rows:
        row["balanced_visual_score"] = float(
            VISUAL_SELECTION_WEIGHTS["model_wrmse_rank"] * row["model_wrmse_rank"]
            + VISUAL_SELECTION_WEIGHTS["edge_concentration_rank"] * row["edge_concentration_rank"]
            + VISUAL_SELECTION_WEIGHTS["local_skill_rank"] * row["local_skill_rank"]
            - VISUAL_SELECTION_WEIGHTS["combined_skill_rank"] * row["combined_skill_rank"]
        )

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            _finite_or_nan(row["balanced_visual_score"]),
            _finite_or_nan(row["combined_model_wrmse"]),
            -_finite_or_nan(row["combined_relative_skill_vs_forecast"]),
            str(row["sample"].query_date),
            int(row["sample"].sample_idx),
        ),
    )
    for rank, row in enumerate(sorted_rows, start=1):
        row["selection_rank"] = int(rank)
    return {row["sample_key"]: row for row in rows}


def _sample_key(sample: SampleMap) -> tuple[int, int, str]:
    return (int(sample.sample_idx), int(sample.query_time_index), str(sample.query_date))


def _attach_selection_metadata(
    *,
    selected: SampleMap,
    reason: str,
    score: float,
    selection_mode: str,
    rank_key: tuple[Any, ...],
    total_sample_count: int,
    positive_skill_candidate_count: int,
    signal_threshold: float | None,
    min_signal_quantile: float,
    eligible_candidate_count: int,
    filtered_below_signal_threshold: Sequence[SampleMap],
    skill_threshold: float | None = None,
    min_skill_quantile: float | None = None,
    filtered_below_skill_threshold: Sequence[SampleMap] = (),
    extra_selection_metadata: Mapping[str, Any] | None = None,
) -> SampleMap:
    selected.selection_reason = reason
    selected.selection_score = float(score)
    selected.selection_rank_key = rank_key
    selected.selection_metadata = {
        "selection_mode": selection_mode,
        "selection_reason": reason,
        "combined_relative_skill_vs_forecast": _combined_skill(selected),
        "combined_model_wrmse": _combined_model_wrmse(selected),
        "true_increment_strength": _combined_true_increment_strength(selected),
        "candidate_pool_size": int(positive_skill_candidate_count),
        "total_sample_count": int(total_sample_count),
        "positive_skill_candidate_count": int(positive_skill_candidate_count),
        "eligible_candidate_count": int(eligible_candidate_count),
        "signal_quantile": float(min_signal_quantile),
        "signal_threshold": (
            float(signal_threshold) if signal_threshold is not None and np.isfinite(signal_threshold) else None
        ),
        "skill_quantile": (
            float(min_skill_quantile) if min_skill_quantile is not None and np.isfinite(min_skill_quantile) else None
        ),
        "skill_threshold": (
            float(skill_threshold) if skill_threshold is not None and np.isfinite(skill_threshold) else None
        ),
        "filtered_below_signal_threshold": [
            _selection_candidate_summary(sample) for sample in filtered_below_signal_threshold
        ],
        "filtered_below_skill_threshold": [
            _selection_candidate_summary(sample) for sample in filtered_below_skill_threshold
        ],
        "rank_key": list(rank_key),
    }
    if extra_selection_metadata:
        selected.selection_metadata.update(dict(extra_selection_metadata))
    return selected


def select_representative_sample(
    samples: Sequence[SampleMap],
    *,
    selection_mode: str = "balanced_visual_with_signal",
    min_signal_quantile: float = 0.5,
    min_skill_quantile: float = 0.25,
) -> SampleMap:
    """Select a display case without using target_eval for model selection.

    The default paper-figure mode selects a visually readable positive-skill
    case: nontrivial DA increment signal, non-tiny relative skill, low
    two-variable model WRMSE, and no strong left-edge/lower-left error
    concentration. This is label-dependent target_eval visualization selection
    only; it must not be reused for model selection or tuning.
    """
    if not samples:
        raise ValueError("No samples available for representative selection")
    if not 0.0 <= float(min_signal_quantile) <= 1.0:
        raise ValueError(f"min_signal_quantile must be in [0, 1], got {min_signal_quantile}")
    if not 0.0 <= float(min_skill_quantile) <= 1.0:
        raise ValueError(f"min_skill_quantile must be in [0, 1], got {min_skill_quantile}")
    legacy_mode_aliases = {
        "best_display_with_signal": "balanced_visual_with_signal",
    }
    normalized_selection_mode = legacy_mode_aliases.get(selection_mode, selection_mode)

    positives = [sample for sample in samples if _has_positive_two_variable_skill(sample)]
    if not positives:
        selected = max(samples, key=_legacy_high_true_increment_key)
        return _attach_selection_metadata(
            selected=selected,
            reason="fallback_high_true_increment_no_positive_skill",
            score=_combined_true_increment_strength(selected),
            selection_mode=normalized_selection_mode,
            rank_key=_legacy_high_true_increment_key(selected),
            total_sample_count=len(samples),
            positive_skill_candidate_count=0,
            signal_threshold=None,
            min_signal_quantile=float(min_signal_quantile),
            min_skill_quantile=float(min_skill_quantile),
            eligible_candidate_count=len(samples),
            filtered_below_signal_threshold=[],
        )

    if normalized_selection_mode == "positive_skill_high_true_increment":
        selected = max(positives, key=_legacy_high_true_increment_key)
        return _attach_selection_metadata(
            selected=selected,
            reason="positive_skill_high_true_increment",
            score=_combined_true_increment_strength(selected),
            selection_mode=normalized_selection_mode,
            rank_key=_legacy_high_true_increment_key(selected),
            total_sample_count=len(samples),
            positive_skill_candidate_count=len(positives),
            signal_threshold=None,
            min_signal_quantile=float(min_signal_quantile),
            min_skill_quantile=float(min_skill_quantile),
            eligible_candidate_count=len(positives),
            filtered_below_signal_threshold=[],
        )

    allowed_modes = {
        "balanced_visual_with_signal",
        "lowest_rmse_with_signal",
        "best_skill_with_signal",
    }
    if normalized_selection_mode not in allowed_modes:
        raise ValueError(
            "selection_mode must be 'balanced_visual_with_signal', "
            "'lowest_rmse_with_signal', 'best_skill_with_signal', "
            "'best_display_with_signal' (legacy alias), or "
            f"'positive_skill_high_true_increment', got {selection_mode!r}"
        )

    strengths = np.asarray(
        [_combined_true_increment_strength(sample) for sample in positives],
        dtype=np.float64,
    )
    finite_strengths = strengths[np.isfinite(strengths)]
    signal_threshold = (
        float(np.quantile(finite_strengths, float(min_signal_quantile)))
        if finite_strengths.size
        else float("-inf")
    )
    signal_eligible = [
        sample
        for sample in positives
        if np.isfinite(_combined_true_increment_strength(sample))
        and _combined_true_increment_strength(sample) >= signal_threshold
    ]
    filtered = [
        sample
        for sample in positives
        if np.isfinite(_combined_true_increment_strength(sample))
        and _combined_true_increment_strength(sample) < signal_threshold
    ]
    if not signal_eligible:
        signal_eligible = positives

    skill_values = np.asarray([_combined_skill(sample) for sample in positives], dtype=np.float64)
    finite_skills = skill_values[np.isfinite(skill_values)]
    skill_threshold = (
        float(np.quantile(finite_skills, float(min_skill_quantile)))
        if finite_skills.size
        else float("-inf")
    )
    eligible = [
        sample
        for sample in signal_eligible
        if np.isfinite(_combined_skill(sample)) and _combined_skill(sample) >= skill_threshold
    ]
    filtered_skill = [
        sample
        for sample in signal_eligible
        if np.isfinite(_combined_skill(sample)) and _combined_skill(sample) < skill_threshold
    ]
    if not eligible:
        eligible = signal_eligible

    if normalized_selection_mode == "lowest_rmse_with_signal":
        selected = max(eligible, key=_lowest_rmse_rank_key)
        rank_key = _lowest_rmse_rank_key(selected)
        reason = "lowest_rmse_with_nontrivial_increment"
        score = _combined_model_wrmse(selected)
        extra_metadata: dict[str, Any] = {}
    elif normalized_selection_mode == "best_skill_with_signal":
        selected = max(eligible, key=_best_skill_rank_key)
        rank_key = _best_skill_rank_key(selected)
        reason = "best_skill_with_nontrivial_increment"
        score = _combined_skill(selected)
        extra_metadata = {}
    else:
        visual_rows = _visual_candidate_rank_rows(eligible)
        selected = min(
            eligible,
            key=lambda sample: (
                _finite_or_nan(visual_rows[_sample_key(sample)]["balanced_visual_score"]),
                _finite_or_nan(visual_rows[_sample_key(sample)]["combined_model_wrmse"]),
                -_finite_or_nan(visual_rows[_sample_key(sample)]["combined_relative_skill_vs_forecast"]),
                str(sample.query_date),
                int(sample.sample_idx),
            ),
        )
        selected_row = visual_rows[_sample_key(selected)]
        rank_key = (
            -_finite_or_nan(selected_row["balanced_visual_score"]),
            -_combined_model_wrmse(selected),
            _combined_skill(selected),
            _combined_true_increment_strength(selected),
            str(selected.query_date),
            -int(selected.sample_idx),
        )
        reason = "balanced_visual_with_nontrivial_increment"
        score = _finite_or_nan(selected_row["balanced_visual_score"])
        extra_metadata = {
            "visual_selection_weights": dict(VISUAL_SELECTION_WEIGHTS),
            "balanced_visual_score": score,
            "left_error_rmse_ratio_vs_full": selected_row["left_error_rmse_ratio_vs_full"],
            "lower_left_error_rmse_ratio_vs_full": selected_row["lower_left_error_rmse_ratio_vs_full"],
            "edge_error_concentration": selected_row["edge_error_concentration"],
            "left_skill_vs_forecast": selected_row["left_skill_vs_forecast"],
            "lower_left_skill_vs_forecast": selected_row["lower_left_skill_vs_forecast"],
            "local_skill_floor": selected_row["local_skill_floor"],
            "model_wrmse_rank": selected_row["model_wrmse_rank"],
            "edge_concentration_rank": selected_row["edge_concentration_rank"],
            "local_skill_rank": selected_row["local_skill_rank"],
            "combined_skill_rank": selected_row["combined_skill_rank"],
            "selection_rank": selected_row["selection_rank"],
        }

    return _attach_selection_metadata(
        selected=selected,
        reason=reason,
        score=score,
        selection_mode=normalized_selection_mode,
        rank_key=rank_key,
        total_sample_count=len(samples),
        positive_skill_candidate_count=len(positives),
        signal_threshold=signal_threshold,
        min_signal_quantile=float(min_signal_quantile),
        skill_threshold=skill_threshold,
        min_skill_quantile=float(min_skill_quantile),
        eligible_candidate_count=len(eligible),
        filtered_below_signal_threshold=filtered,
        filtered_below_skill_threshold=filtered_skill,
        extra_selection_metadata=extra_metadata,
    )


def build_candidate_ranking(
    samples: Sequence[SampleMap],
    *,
    selected: SampleMap | None = None,
    min_signal_quantile: float = 0.5,
    min_skill_quantile: float = 0.25,
) -> dict[str, Any]:
    """Build a visualization-only candidate table for manual display-case review."""
    if not samples:
        raise ValueError("No samples available for candidate ranking")
    positives = [sample for sample in samples if _has_positive_two_variable_skill(sample)]
    selected_key = _sample_key(selected) if selected is not None else None

    signal_threshold: float | None = None
    skill_threshold: float | None = None
    signal_eligible: list[SampleMap] = []
    eligible: list[SampleMap] = []
    if positives:
        strengths = np.asarray(
            [_combined_true_increment_strength(sample) for sample in positives],
            dtype=np.float64,
        )
        finite_strengths = strengths[np.isfinite(strengths)]
        signal_threshold = (
            float(np.quantile(finite_strengths, float(min_signal_quantile)))
            if finite_strengths.size
            else float("-inf")
        )
        signal_eligible = [
            sample
            for sample in positives
            if np.isfinite(_combined_true_increment_strength(sample))
            and _combined_true_increment_strength(sample) >= signal_threshold
        ]
        if not signal_eligible:
            signal_eligible = positives

        skill_values = np.asarray([_combined_skill(sample) for sample in positives], dtype=np.float64)
        finite_skills = skill_values[np.isfinite(skill_values)]
        skill_threshold = (
            float(np.quantile(finite_skills, float(min_skill_quantile)))
            if finite_skills.size
            else float("-inf")
        )
        eligible = [
            sample
            for sample in signal_eligible
            if np.isfinite(_combined_skill(sample)) and _combined_skill(sample) >= skill_threshold
        ]
        if not eligible:
            eligible = signal_eligible

    positive_keys = {_sample_key(sample) for sample in positives}
    signal_keys = {_sample_key(sample) for sample in signal_eligible}
    eligible_keys = {_sample_key(sample) for sample in eligible}
    visual_rows = _visual_candidate_rank_rows(eligible) if eligible else {}

    rows: list[dict[str, Any]] = []
    for sample in samples:
        key = _sample_key(sample)
        visual = visual_rows.get(key)
        metrics = _sample_visual_selection_metrics(sample)
        row: dict[str, Any] = {
            "sample_key": key,
            "selected": bool(selected_key is not None and key == selected_key),
            "eligible_for_balanced_visual_selection": key in eligible_keys,
            "positive_two_variable_skill": key in positive_keys,
            "signal_eligible": key in signal_keys,
            "skill_eligible": key in eligible_keys,
            "sample_idx": int(sample.sample_idx),
            "query_date": str(sample.query_date),
            "combined_model_wrmse": _combined_model_wrmse(sample),
            "combined_relative_skill_vs_forecast": _combined_skill(sample),
            "true_increment_strength": _combined_true_increment_strength(sample),
            "balanced_visual_score": (
                _finite_or_nan(visual["balanced_visual_score"]) if visual is not None else float("nan")
            ),
            "selection_rank": int(visual["selection_rank"]) if visual is not None else "",
            "model_wrmse_rank": int(visual["model_wrmse_rank"]) if visual is not None else "",
            "edge_concentration_rank": int(visual["edge_concentration_rank"]) if visual is not None else "",
            "local_skill_rank": int(visual["local_skill_rank"]) if visual is not None else "",
            "combined_skill_rank": int(visual["combined_skill_rank"]) if visual is not None else "",
            **metrics,
        }
        for variable in VARIABLE_ORDER:
            variable_map = sample.variables[variable]
            row[f"{variable}_model_wrmse"] = variable_map.model_wrmse
            row[f"{variable}_relative_skill_vs_forecast"] = variable_map.skill_vs_forecast
            row[f"{variable}_true_increment_wrmse"] = variable_map.true_increment_wrmse
        rows.append(row)

    rows.sort(
        key=lambda row: (
            1 if row["selection_rank"] == "" else 0,
            int(row["selection_rank"]) if row["selection_rank"] != "" else 10**9,
            0 if row["positive_two_variable_skill"] else 1,
            _finite_or_nan(row["combined_model_wrmse"]),
            str(row["query_date"]),
            int(row["sample_idx"]),
        )
    )
    return {
        "schema_version": "hydroda_ref_pred_error_map_candidate_ranking_v1",
        "selection_usage": "visualization_only_not_model_selection",
        "total_sample_count": int(len(samples)),
        "positive_skill_candidate_count": int(len(positives)),
        "eligible_candidate_count": int(len(eligible)),
        "signal_quantile": float(min_signal_quantile),
        "signal_threshold": signal_threshold,
        "skill_quantile": float(min_skill_quantile),
        "skill_threshold": skill_threshold,
        "visual_selection_weights": dict(VISUAL_SELECTION_WEIGHTS),
        "rows": rows,
        "leakage_note": (
            "This table ranks target_eval prediction records only for manual "
            "paper-figure display-case review. It is not used for training, "
            "adaptation, model selection, threshold calibration, or tuning."
        ),
    }


def write_candidate_ranking_files(
    ranking: Mapping[str, Any],
    *,
    output_dir: str | Path,
    csv_name: str = "sample_candidate_ranking.csv",
    json_name: str = "sample_candidate_ranking.json",
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / csv_name
    json_path = output_path / json_name

    rows = list(ranking.get("rows") or [])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_TABLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CANDIDATE_TABLE_FIELDS})

    public_ranking = dict(ranking)
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        public_rows.append({key: value for key, value in row.items() if key != "sample_key"})
    public_ranking["rows"] = public_rows
    json_path.write_text(json.dumps(public_ranking, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"candidate_ranking_csv": str(csv_path), "candidate_ranking_json": str(json_path)}


def _gallery_error_limit(samples: Sequence[SampleMap], variable: str, q: float = 99.0) -> float:
    chunks = [
        np.asarray(sample.variables[variable].rmse_map, dtype=np.float32)[
            np.isfinite(sample.variables[variable].rmse_map)
        ].reshape(-1)
        for sample in samples
        if np.isfinite(sample.variables[variable].rmse_map).any()
    ]
    if not chunks:
        return 1.0
    values = np.concatenate(chunks)
    if values.size == 0:
        return 1.0
    limit = float(np.nanpercentile(values, q))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.nanmax(values))
    return limit if np.isfinite(limit) and limit > 0.0 else 1.0


def render_sample_error_gallery(
    samples: Sequence[SampleMap],
    *,
    ranking: Mapping[str, Any],
    output_dir: str | Path,
    output_stem: str,
    page_size: int = 32,
    formats: Sequence[str] = ("png",),
) -> dict[str, list[str]]:
    """Render paginated all-sample error thumbnails for manual figure selection."""
    if not samples:
        return {}
    if int(page_size) <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    configure_matplotlib()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = list(ranking.get("rows") or [])
    sample_by_key = {_sample_key(sample): sample for sample in samples}
    ordered: list[tuple[dict[str, Any], SampleMap]] = [
        (row, sample_by_key[row["sample_key"]])
        for row in rows
        if row.get("sample_key") in sample_by_key
    ]
    seen_keys = {row["sample_key"] for row, _sample in ordered}
    for sample in samples:
        key = _sample_key(sample)
        if key not in seen_keys:
            ordered.append(({"sample_key": key}, sample))

    cmap = morandi_error_cmap()
    limits = {variable: _gallery_error_limit(samples, variable) for variable in VARIABLE_ORDER}
    page_count = int(np.ceil(len(ordered) / int(page_size)))
    outputs: dict[str, list[str]] = {fmt.lower().lstrip("."): [] for fmt in formats}

    for page_idx in range(page_count):
        page_items = ordered[page_idx * int(page_size) : (page_idx + 1) * int(page_size)]
        ncols = 4
        nrows = int(np.ceil(len(page_items) / ncols))
        fig = plt.figure(figsize=(8.7, max(2.0, 1.52 * nrows)))
        outer = fig.add_gridspec(nrows, ncols, hspace=0.42, wspace=0.18)
        first_images: dict[str, Any] = {}
        for item_idx, (row, sample) in enumerate(page_items):
            row_idx = item_idx // ncols
            col_idx = item_idx % ncols
            inner = outer[row_idx, col_idx].subgridspec(2, 1, hspace=0.04)
            rank = row.get("selection_rank", "")
            rank_text = f"r{rank}" if rank != "" else "unranked"
            selected_prefix = "SEL " if row.get("selected") else ""
            title = (
                f"{selected_prefix}{rank_text} | idx {sample.sample_idx} | {sample.query_date}\n"
                f"WRMSE {row.get('combined_model_wrmse', float('nan')):.4g} "
                f"skill {row.get('combined_relative_skill_vs_forecast', float('nan')):.2f} "
                f"edge {row.get('edge_error_concentration', float('nan')):.2f}"
            )
            for variable_idx, variable in enumerate(VARIABLE_ORDER):
                ax = fig.add_subplot(inner[variable_idx, 0])
                image = ax.imshow(
                    sample.variables[variable].rmse_map,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=limits[variable],
                    origin="upper",
                    interpolation="nearest",
                    aspect="auto",
                    rasterized=True,
                )
                first_images.setdefault(variable, image)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(
                    0.02,
                    0.86,
                    "S" if variable == "surface" else "R",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=4.8,
                    color="#222222",
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 0.6},
                )
                for spine in ax.spines.values():
                    spine.set_linewidth(0.55 if row.get("selected") else 0.2)
                    spine.set_edgecolor("#8f3d45" if row.get("selected") else "#777777")
                if variable_idx == 0:
                    ax.set_title(title, fontsize=5.1, pad=2)
        fig.suptitle(
            "US-R1 target_eval sample error thumbnails sorted by visualization-only ranking",
            fontsize=7,
            y=0.995,
        )
        for fmt in outputs:
            path = output_path / f"{output_stem}_page_{page_idx + 1:02d}.{fmt}"
            fig.savefig(path, dpi=260)
            outputs[fmt].append(str(path))
        plt.close(fig)
    return {f"sample_error_gallery_{fmt}": paths for fmt, paths in outputs.items()}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _region_bbox_from_manifest(manifest_path: Path, region_id: str) -> dict[str, int] | None:
    if not manifest_path.exists():
        return None
    manifest = _load_json(manifest_path)
    for row in manifest.get("regions", []):
        if row.get("region_id") == region_id:
            bbox = row.get("resolved_index_bbox") or {}
            required = ("y_start", "y_end", "x_start", "x_end")
            if all(key in bbox for key in required):
                return {key: int(bbox[key]) for key in required}
    return None


def _bbox_slices(bbox: Mapping[str, int]) -> tuple[slice, slice]:
    return (
        slice(int(bbox["y_start"]), int(bbox["y_end"]) + 1),
        slice(int(bbox["x_start"]), int(bbox["x_end"]) + 1),
    )


def crop_sample_to_bbox(sample: SampleMap, bbox: Mapping[str, int]) -> SampleMap:
    """Crop full-grid sample arrays to an inclusive region bbox and recompute metrics."""
    y_slice, x_slice = _bbox_slices(bbox)
    mask = np.asarray(sample.mask[y_slice, x_slice], dtype=np.float32)
    latitude_weight = np.asarray(sample.latitude_weight[y_slice, x_slice], dtype=np.float32)
    variables: dict[str, VariableMap] = {}
    for variable in VARIABLE_ORDER:
        original = sample.variables[variable]
        ref = np.asarray(original.ref[y_slice, x_slice], dtype=np.float32)
        pred = np.asarray(original.pred[y_slice, x_slice], dtype=np.float32)
        forecast = np.asarray(original.forecast[y_slice, x_slice], dtype=np.float32)
        variables[variable] = _build_variable_map(
            name=variable,
            ref=ref,
            pred=pred,
            forecast=forecast,
            mask=mask,
            latitude_weight=latitude_weight,
        )
    cropped = SampleMap(
        sample_idx=sample.sample_idx,
        query_time_index=sample.query_time_index,
        query_date=sample.query_date,
        month=sample.month,
        season=sample.season,
        target_region_id=sample.target_region_id,
        split_role=sample.split_role,
        method=sample.method,
        record_metadata=dict(sample.record_metadata),
        mask=mask,
        latitude_weight=latitude_weight,
        variables=variables,
        selection_reason=sample.selection_reason,
        selection_score=sample.selection_score,
        selection_metadata=dict(sample.selection_metadata),
    )
    cropped.selection_rank_key = sample.selection_rank_key
    return cropped


def crop_sample_to_region_if_full_grid(
    sample: SampleMap,
    *,
    crop_manifest: str | Path,
    region_id: str,
    full_shape: tuple[int, int] = (256, 640),
) -> tuple[SampleMap, dict[str, Any]]:
    """Crop a full-grid record to the region bbox when metadata is available."""
    current_shape = sample.variables["surface"].ref.shape
    metadata: dict[str, Any] = {
        "sample_shape_before_crop": list(current_shape),
        "crop_applied": False,
        "crop_reason": "",
    }
    if current_shape != tuple(full_shape):
        metadata["crop_reason"] = "sample_is_already_crop_or_nonstandard_shape"
        return sample, metadata
    bbox = _region_bbox_from_manifest(Path(crop_manifest), region_id)
    if bbox is None:
        metadata["crop_reason"] = f"missing_crop_bbox_for_{region_id}"
        return sample, metadata
    cropped = crop_sample_to_bbox(sample, bbox)
    metadata.update(
        {
            "crop_applied": True,
            "crop_reason": "full_grid_record_cropped_to_region_bbox",
            "resolved_index_bbox": bbox,
            "sample_shape_after_crop": list(cropped.variables["surface"].ref.shape),
        }
    )
    return cropped, metadata


def _sample_weighted_error_stats(
    *,
    model_error: np.ndarray,
    forecast_error: np.ndarray,
    mask: np.ndarray,
    latitude_weight: np.ndarray,
) -> dict[str, float | int]:
    model_error = np.asarray(model_error, dtype=np.float64)
    forecast_error = np.asarray(forecast_error, dtype=np.float64)
    weights = np.asarray(latitude_weight, dtype=np.float64)
    valid = (
        (np.asarray(mask) > 0.5)
        & np.isfinite(model_error)
        & np.isfinite(forecast_error)
        & np.isfinite(weights)
        & (weights >= 0.0)
    )
    if int(valid.sum()) == 0 or float(weights[valid].sum()) <= 0.0:
        return {
            "valid_pixel_count": int(valid.sum()),
            "model_rmse": float("nan"),
            "forecast_rmse": float("nan"),
            "mean_model_abs_error": float("nan"),
            "mean_forecast_abs_error": float("nan"),
            "mean_skill_vs_forecast": float("nan"),
        }
    valid_weights = weights[valid]
    model_mse = np.average(model_error[valid] ** 2, weights=valid_weights)
    forecast_mse = np.average(forecast_error[valid] ** 2, weights=valid_weights)
    model_rmse = float(np.sqrt(model_mse))
    forecast_rmse = float(np.sqrt(forecast_mse))
    model_mae = float(np.average(model_error[valid], weights=valid_weights))
    forecast_mae = float(np.average(forecast_error[valid], weights=valid_weights))
    skill = (
        float(1.0 - model_rmse / forecast_rmse)
        if np.isfinite(model_rmse) and np.isfinite(forecast_rmse) and forecast_rmse > 0.0
        else float("nan")
    )
    return {
        "valid_pixel_count": int(valid.sum()),
        "model_rmse": model_rmse,
        "forecast_rmse": forecast_rmse,
        "mean_model_abs_error": model_mae,
        "mean_forecast_abs_error": forecast_mae,
        "mean_skill_vs_forecast": skill,
    }


def _pixel_mean_stack(arrays: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not arrays:
        return np.full(shape, np.nan, dtype=np.float32)
    stack = np.stack([np.asarray(array, dtype=np.float32) for array in arrays], axis=0)
    valid_counts = np.sum(np.isfinite(stack), axis=0)
    sums = np.nansum(stack, axis=0)
    mean = np.full(shape, np.nan, dtype=np.float32)
    valid = valid_counts > 0
    with np.errstate(invalid="ignore"):
        mean[valid] = (sums[valid] / valid_counts[valid]).astype(np.float32)
    return mean


def _pixel_mean_error_maps(samples: Sequence[SampleMap], variable: str) -> dict[str, np.ndarray]:
    shape = samples[0].variables[variable].ref.shape
    model_errors: list[np.ndarray] = []
    forecast_errors: list[np.ndarray] = []
    skill_maps: list[np.ndarray] = []
    for sample in samples:
        variable_map = sample.variables[variable]
        if variable_map.ref.shape != shape:
            raise ValueError(
                "All samples must have the same shape for diagnostic aggregation: "
                f"expected={shape} got={variable_map.ref.shape}"
            )
        valid = _valid_mask(variable_map.pred, variable_map.ref, sample.mask, sample.latitude_weight)
        model_error = _masked_array(np.abs(variable_map.pred - variable_map.ref), valid)
        forecast_error = _masked_array(np.abs(variable_map.forecast - variable_map.ref), valid)
        skill = np.full(shape, np.nan, dtype=np.float32)
        finite = valid & np.isfinite(forecast_error) & (forecast_error > 0.0) & np.isfinite(model_error)
        skill[finite] = (1.0 - model_error[finite] / forecast_error[finite]).astype(np.float32)
        model_errors.append(model_error)
        forecast_errors.append(forecast_error)
        skill_maps.append(skill)
    return {
        "mean_model_abs_error": _pixel_mean_stack(model_errors, shape),
        "mean_forecast_abs_error": _pixel_mean_stack(forecast_errors, shape),
        "mean_skill_vs_forecast": _pixel_mean_stack(skill_maps, shape),
    }


def _lower_left_window(shape: tuple[int, int], fraction: float) -> dict[str, int]:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"lower_left_fraction must be in (0, 1], got {fraction}")
    height, width = shape
    window_height = max(1, int(np.ceil(height * float(fraction))))
    window_width = max(1, int(np.ceil(width * float(fraction))))
    return {
        "y_start": int(height - window_height),
        "y_end": int(height),
        "x_start": 0,
        "x_end": int(window_width),
    }


def _window_mask(shape: tuple[int, int], window: Mapping[str, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[
        int(window["y_start"]) : int(window["y_end"]),
        int(window["x_start"]) : int(window["x_end"]),
    ] = True
    return mask


def _summarize_error_region(
    *,
    region_mask: np.ndarray,
    model_error: np.ndarray,
    forecast_error: np.ndarray,
    metric_mask: np.ndarray,
    latitude_weight: np.ndarray,
    panel_valid_pixel_count: int,
) -> dict[str, Any]:
    masked_metric = (np.asarray(metric_mask) > 0.5) & np.asarray(region_mask, dtype=bool)
    stats = _sample_weighted_error_stats(
        model_error=model_error,
        forecast_error=forecast_error,
        mask=masked_metric.astype(np.float32),
        latitude_weight=latitude_weight,
    )
    valid_count = int(stats["valid_pixel_count"])
    stats["valid_pixel_fraction_of_panel"] = (
        float(valid_count / panel_valid_pixel_count) if panel_valid_pixel_count > 0 else float("nan")
    )
    return stats


def build_error_diagnostic_summary(
    samples: Sequence[SampleMap],
    *,
    lower_left_fraction: float = 0.25,
) -> dict[str, Any]:
    """Summarize pixel-level error behavior for the lower-left crop area.

    This diagnostic uses the same already-decoded panel coordinates, metric
    masks, and latitude weights as the figure. It is evaluation evidence only,
    not model selection or tuning input.
    """
    if not samples:
        raise ValueError("No samples available for error diagnostics")
    shape = samples[0].variables["surface"].ref.shape
    mask_shape = np.asarray(samples[0].mask).shape
    if mask_shape != shape:
        raise ValueError(f"Sample mask shape {mask_shape} does not match variable shape {shape}")

    aggregate_metric_mask = np.zeros(shape, dtype=np.float32)
    aggregate_latitude_weight = np.asarray(samples[0].latitude_weight, dtype=np.float32)
    for sample in samples:
        if sample.variables["surface"].ref.shape != shape:
            raise ValueError(
                "All samples must have matching cropped shapes for diagnostics: "
                f"expected={shape} got={sample.variables['surface'].ref.shape}"
            )
        aggregate_metric_mask = np.maximum(aggregate_metric_mask, (np.asarray(sample.mask) > 0.5).astype(np.float32))
    panel_valid_pixel_count = int(aggregate_metric_mask.sum())
    window = _lower_left_window(shape, lower_left_fraction)
    lower_left_mask = _window_mask(shape, window)
    outside_mask = ~lower_left_mask

    variables: dict[str, Any] = {}
    for variable in VARIABLE_ORDER:
        maps = _pixel_mean_error_maps(samples, variable)
        model_error = maps["mean_model_abs_error"]
        forecast_error = maps["mean_forecast_abs_error"]
        full_panel = _summarize_error_region(
            region_mask=np.ones(shape, dtype=bool),
            model_error=model_error,
            forecast_error=forecast_error,
            metric_mask=aggregate_metric_mask,
            latitude_weight=aggregate_latitude_weight,
            panel_valid_pixel_count=panel_valid_pixel_count,
        )
        lower_left = _summarize_error_region(
            region_mask=lower_left_mask,
            model_error=model_error,
            forecast_error=forecast_error,
            metric_mask=aggregate_metric_mask,
            latitude_weight=aggregate_latitude_weight,
            panel_valid_pixel_count=panel_valid_pixel_count,
        )
        outside = _summarize_error_region(
            region_mask=outside_mask,
            model_error=model_error,
            forecast_error=forecast_error,
            metric_mask=aggregate_metric_mask,
            latitude_weight=aggregate_latitude_weight,
            panel_valid_pixel_count=panel_valid_pixel_count,
        )
        lower_left.update(
            {
                "window": dict(window),
                "mean_model_abs_error_rank_from_worst": 1
                if np.isfinite(lower_left["mean_model_abs_error"])
                and (
                    not np.isfinite(outside["mean_model_abs_error"])
                    or lower_left["mean_model_abs_error"] >= outside["mean_model_abs_error"]
                )
                else 2,
                "model_rmse_ratio_vs_full_panel": (
                    float(lower_left["model_rmse"] / full_panel["model_rmse"])
                    if np.isfinite(lower_left["model_rmse"])
                    and np.isfinite(full_panel["model_rmse"])
                    and full_panel["model_rmse"] > 0.0
                    else float("nan")
                ),
                "model_rmse_ratio_vs_outside": (
                    float(lower_left["model_rmse"] / outside["model_rmse"])
                    if np.isfinite(lower_left["model_rmse"])
                    and np.isfinite(outside["model_rmse"])
                    and outside["model_rmse"] > 0.0
                    else float("nan")
                ),
            }
        )
        variables[variable] = {
            "full_panel": full_panel,
            "lower_left_window": lower_left,
            "outside_lower_left_window": outside,
            "pixel_mean_maps": {
                name: {
                    "mean": float(np.nanmean(values)) if np.isfinite(values).any() else float("nan"),
                    "max": float(np.nanmax(values)) if np.isfinite(values).any() else float("nan"),
                }
                for name, values in maps.items()
            },
        }

    return {
        "schema_version": "hydroda_ref_pred_error_diagnostic_v1",
        "diagnostic_usage": "error_diagnostic_only_not_model_selection",
        "sample_count": int(len(samples)),
        "panel_shape": list(shape),
        "lower_left_fraction": float(lower_left_fraction),
        "variables": variables,
        "leakage_note": (
            "This diagnostic summarizes target_eval prediction records only for "
            "post-hoc visualization/error review. It is not used for training, "
            "adaptation, model selection, threshold calibration, or tuning."
        ),
    }


def load_region_latlon(
    *,
    geolocation_nc: str | Path,
    crop_manifest: str | Path,
    region_id: str,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return lon/lat arrays for a sample crop, or grid coordinates as fallback."""
    geolocation_path = Path(geolocation_nc)
    manifest_path = Path(crop_manifest)
    fallback = {
        "coordinate_mode": "grid",
        "geolocation_nc": str(geolocation_path),
        "crop_manifest": str(manifest_path),
        "fallback_reason": "",
    }

    try:
        with xr.open_dataset(geolocation_path) as ds:
            lat_full = np.asarray(ds["latitude"].values)
            lon_full = np.asarray(ds["longitude"].values)
    except Exception as exc:
        fallback["fallback_reason"] = f"failed_to_load_geolocation: {exc}"
        return _grid_coordinates(expected_shape, fallback)

    if lat_full.shape == expected_shape and lon_full.shape == expected_shape:
        return lon_full, lat_full, {
            "coordinate_mode": "latlon_full_shape",
            "geolocation_nc": str(geolocation_path),
            "crop_manifest": str(manifest_path),
            "latlon_shape": list(lat_full.shape),
        }

    bbox = _region_bbox_from_manifest(manifest_path, region_id)
    if bbox is None:
        fallback["fallback_reason"] = f"missing_crop_bbox_for_{region_id}"
        return _grid_coordinates(expected_shape, fallback)

    y_slice, x_slice = _bbox_slices(bbox)
    lat_crop = lat_full[y_slice, x_slice]
    lon_crop = lon_full[y_slice, x_slice]
    if lat_crop.shape != expected_shape or lon_crop.shape != expected_shape:
        fallback["fallback_reason"] = (
            "crop_shape_mismatch: "
            f"expected={expected_shape} lat_crop={lat_crop.shape} lon_crop={lon_crop.shape}"
        )
        fallback["resolved_index_bbox"] = bbox
        return _grid_coordinates(expected_shape, fallback)

    return lon_crop, lat_crop, {
        "coordinate_mode": "latlon_region_crop",
        "geolocation_nc": str(geolocation_path),
        "crop_manifest": str(manifest_path),
        "resolved_index_bbox": bbox,
        "latlon_shape": list(lat_crop.shape),
    }


def _grid_coordinates(
    shape: tuple[int, int],
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = shape
    x = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    y = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))
    return x, y, dict(metadata)


def load_region_mask_crop(
    *,
    region_mask_nc: str | Path,
    crop_manifest: str | Path,
    region_id: str,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load the cropped canonical region mask used only for map outlining."""
    region_mask_path = Path(region_mask_nc)
    metadata: dict[str, Any] = {
        "region_mask_nc": str(region_mask_path),
        "outline_available": False,
        "outline_reason": "",
    }
    if not region_mask_path.exists():
        metadata["outline_reason"] = "missing_region_mask_nc"
        return None, metadata

    bbox = _region_bbox_from_manifest(Path(crop_manifest), region_id)
    if bbox is None:
        metadata["outline_reason"] = f"missing_crop_bbox_for_{region_id}"
        return None, metadata

    try:
        with xr.open_dataset(region_mask_path) as ds:
            if "region_mask_onehot" not in ds:
                metadata["outline_reason"] = "missing_region_mask_onehot"
                return None, metadata
            region_ids = [str(value) for value in ds["region_id"].values]
            if region_id not in region_ids:
                metadata["outline_reason"] = f"missing_region_id_{region_id}"
                return None, metadata
            region_index = region_ids.index(region_id)
            y_slice, x_slice = _bbox_slices(bbox)
            mask = np.asarray(
                ds["region_mask_onehot"].isel(region_id=region_index).values[y_slice, x_slice],
                dtype=np.float32,
            )
    except Exception as exc:
        metadata["outline_reason"] = f"failed_to_load_region_mask: {exc}"
        return None, metadata

    if mask.shape != expected_shape:
        metadata["outline_reason"] = f"mask_shape_mismatch: expected={expected_shape} got={mask.shape}"
        metadata["resolved_index_bbox"] = bbox
        return None, metadata
    metadata.update(
        {
            "outline_available": True,
            "outline_reason": "loaded_region_mask_crop",
            "resolved_index_bbox": bbox,
            "mask_shape": list(mask.shape),
        }
    )
    return mask, metadata


def region_outline_segments_from_mask(
    mask: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> list[np.ndarray]:
    """Build lon/lat line segments along the outer boundary of a region mask."""
    valid = np.asarray(mask) > 0.5
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if valid.shape != lon.shape or valid.shape != lat.shape:
        raise ValueError(
            "mask, lon, and lat must have matching shapes: "
            f"mask={valid.shape} lon={lon.shape} lat={lat.shape}"
        )
    height, width = valid.shape
    if height == 0 or width == 0:
        return []
    x_centers = np.nanmedian(lon, axis=0)
    y_centers = np.nanmedian(lat, axis=1)
    if not np.isfinite(x_centers).all() or not np.isfinite(y_centers).all():
        return []
    x_edges = _coordinate_edges_1d(x_centers)
    y_edges = _coordinate_edges_1d(y_centers)
    segments: list[np.ndarray] = []
    for y in range(height):
        for x in range(width):
            if not valid[y, x]:
                continue
            north_exposed = y == 0 or not valid[y - 1, x]
            south_exposed = y == height - 1 or not valid[y + 1, x]
            west_exposed = x == 0 or not valid[y, x - 1]
            east_exposed = x == width - 1 or not valid[y, x + 1]
            if north_exposed:
                segments.append(
                    np.asarray([[x_edges[x], y_edges[y]], [x_edges[x + 1], y_edges[y]]])
                )
            if south_exposed:
                segments.append(
                    np.asarray([[x_edges[x], y_edges[y + 1]], [x_edges[x + 1], y_edges[y + 1]]])
                )
            if west_exposed:
                segments.append(
                    np.asarray([[x_edges[x], y_edges[y]], [x_edges[x], y_edges[y + 1]]])
                )
            if east_exposed:
                segments.append(
                    np.asarray([[x_edges[x + 1], y_edges[y]], [x_edges[x + 1], y_edges[y + 1]]])
                )
    return segments


def _coordinate_edges_1d(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 1:
        return np.asarray([centers[0] - 0.5, centers[0] + 0.5], dtype=np.float64)
    midpoints = (centers[:-1] + centers[1:]) / 2.0
    first = centers[0] - (midpoints[0] - centers[0])
    last = centers[-1] + (centers[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]]).astype(np.float64)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def morandi_field_cmap() -> LinearSegmentedColormap:
    """Low-saturation dry-to-wet colormap for soil-moisture fields."""
    return LinearSegmentedColormap.from_list(
        "hydroda_morandi_field",
        ["#f2eee7", "#d8d1bd", "#a9b994", "#6f9c92", "#3f7288"],
    )


def morandi_error_cmap() -> LinearSegmentedColormap:
    """Low-saturation sequential colormap for analysis-space error magnitude."""
    return LinearSegmentedColormap.from_list(
        "hydroda_morandi_error",
        ["#f7f4ee", "#e5d8c8", "#d1aa96", "#b97874", "#7f4f5d"],
    )


def _robust_abs_limit(*arrays: np.ndarray, q: float = 98.0) -> float:
    finite_chunks = [
        np.abs(np.asarray(arr)[np.isfinite(arr)]).reshape(-1)
        for arr in arrays
        if np.isfinite(arr).any()
    ]
    if not finite_chunks:
        return 1.0
    values = np.concatenate(finite_chunks)
    if values.size == 0:
        return 1.0
    limit = float(np.nanpercentile(values, q))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.nanmax(values))
    return limit if np.isfinite(limit) and limit > 0.0 else 1.0


def _robust_range(*arrays: np.ndarray, lower_q: float = 2.0, upper_q: float = 98.0) -> tuple[float, float]:
    finite_chunks = [
        np.asarray(arr, dtype=np.float64)[np.isfinite(arr)].reshape(-1)
        for arr in arrays
        if np.isfinite(arr).any()
    ]
    if not finite_chunks:
        return 0.0, 1.0
    values = np.concatenate(finite_chunks)
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(values, [lower_q, upper_q])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return 0.0, 1.0
    return float(vmin), float(vmax)


def _robust_positive_limit(array: np.ndarray, q: float = 98.0) -> float:
    values = np.asarray(array)[np.isfinite(array)].reshape(-1)
    if values.size == 0:
        return 1.0
    limit = float(np.nanpercentile(values, q))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.nanmax(values))
    return limit if np.isfinite(limit) and limit > 0.0 else 1.0


def create_figure_axes(*, use_map_axes: bool) -> tuple[plt.Figure, np.ndarray]:
    """Create 2x3 axes, using Cartopy map axes for every panel when available."""
    fig = plt.figure(figsize=(7.4, 4.55), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, hspace=0.015)
    axes = np.empty((2, 3), dtype=object)
    width_ratios = [1.0, 1.0, 0.12, 1.0]
    panel_grid_columns = (0, 1, 3)
    for row_idx, subfig in enumerate(subfigs):
        row_grid = subfig.add_gridspec(
            1,
            4,
            width_ratios=width_ratios,
            wspace=0.08,
        )
        for col_idx, grid_col_idx in enumerate(panel_grid_columns):
            if use_map_axes and ccrs is not None:
                axes[row_idx, col_idx] = subfig.add_subplot(
                    row_grid[0, grid_col_idx],
                    projection=ccrs.PlateCarree(),
                )
            else:
                axes[row_idx, col_idx] = subfig.add_subplot(row_grid[0, grid_col_idx])
    return fig, axes


def _add_compact_panel_colorbar(
    fig: plt.Figure,
    mesh: Any,
    *,
    anchor_ax: plt.Axes,
) -> Any:
    """Add a compact vertical colorbar sized relative to one map panel."""
    cax = inset_axes(
        anchor_ax,
        width=COLORBAR_STYLE["width"],
        height=COLORBAR_STYLE["height"],
        loc="center left",
        bbox_to_anchor=(COLORBAR_STYLE["x_offset"], 0.0, 1.0, 1.0),
        bbox_transform=anchor_ax.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.ax.tick_params(
        labelsize=COLORBAR_STYLE["tick_labelsize"],
        width=COLORBAR_STYLE["tick_width"],
        length=COLORBAR_STYLE["tick_length"],
    )
    cbar.outline.set_linewidth(COLORBAR_STYLE["outline_width"])
    return cbar


def _axis_extent(lon: np.ndarray, lat: np.ndarray, pad_fraction: float = 0.035) -> list[float]:
    lon_values = np.asarray(lon)[np.isfinite(lon)]
    lat_values = np.asarray(lat)[np.isfinite(lat)]
    if lon_values.size == 0 or lat_values.size == 0:
        return [0.0, 1.0, 0.0, 1.0]
    lon_min, lon_max = float(lon_values.min()), float(lon_values.max())
    lat_min, lat_max = float(lat_values.min()), float(lat_values.max())
    lon_pad = max((lon_max - lon_min) * pad_fraction, 1e-6)
    lat_pad = max((lat_max - lat_min) * pad_fraction, 1e-6)
    return [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad]


def _coarse_degree_ticks(
    axis_min: float,
    axis_max: float,
    *,
    step: float = 2.0,
    max_ticks: int = 4,
) -> list[float]:
    """Return sparse integer-degree ticks inside a compact map extent."""
    if not np.isfinite(axis_min) or not np.isfinite(axis_max) or axis_min >= axis_max:
        return []
    first = float(np.ceil(axis_min / step) * step)
    last = float(np.floor(axis_max / step) * step)
    if first > last:
        center = round((axis_min + axis_max) / 2.0)
        return [float(center)] if axis_min <= center <= axis_max else []
    ticks = np.arange(first, last + step * 0.5, step, dtype=np.float64)
    ticks = [float(tick) for tick in ticks if axis_min < tick < axis_max]
    if len(ticks) <= max_ticks:
        return ticks
    keep_indices = np.linspace(0, len(ticks) - 1, max_ticks, dtype=int)
    return [ticks[int(index)] for index in keep_indices]


def _style_map_axis(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    show_ylabel: bool,
    subtle_boundaries: bool = False,
    show_grid_labels: bool = True,
) -> None:
    if ccrs is None or cfeature is None or not hasattr(ax, "set_extent"):
        return
    coastline_width = 0.25 if subtle_boundaries else 0.35
    state_width = 0.22 if subtle_boundaries else 0.35
    boundary_alpha = 0.48 if subtle_boundaries else 1.0
    grid_alpha = 0.22 if subtle_boundaries else 0.45
    grid_width = 0.18 if subtle_boundaries else 0.25
    extent = _axis_extent(lon, lat)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f4f4f1", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f7fbff", edgecolor="none", zorder=0)
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=coastline_width,
        edgecolor="#5f646b",
        alpha=boundary_alpha,
        zorder=3,
    )
    ax.add_feature(
        cfeature.STATES.with_scale("50m"),
        linewidth=state_width,
        edgecolor="#7a7d82",
        facecolor="none",
        alpha=boundary_alpha,
        zorder=3,
    )
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=show_grid_labels,
        linewidth=grid_width,
        color="#9aa0a6",
        alpha=grid_alpha,
        linestyle=":",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = show_grid_labels
    gl.left_labels = show_grid_labels and show_ylabel
    if show_grid_labels:
        gl.xlocator = FixedLocator(_coarse_degree_ticks(extent[0], extent[1]))
        gl.ylocator = FixedLocator(_coarse_degree_ticks(extent[2], extent[3]))
        gl.xlabel_style = {"size": 6, "color": "#333333"}
        gl.ylabel_style = {"size": 6, "color": "#333333"}
    else:
        ax.set_xticks([], crs=ccrs.PlateCarree())
        ax.set_yticks([], crs=ccrs.PlateCarree())
    ax.set_xlabel("")
    ax.set_ylabel("")


def _draw_region_outline(
    ax: plt.Axes,
    *,
    region_mask: np.ndarray | None,
    lon: np.ndarray,
    lat: np.ndarray,
    transform: Any = None,
    subtle: bool = False,
) -> None:
    if region_mask is None:
        return
    linewidth = 0.42 if subtle else 0.55
    alpha = 0.55 if subtle else 0.85
    for segment in region_outline_segments_from_mask(region_mask, lon, lat):
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color="#111111",
            linewidth=linewidth,
            alpha=alpha,
            transform=transform,
            zorder=4,
        )


def _plot_map_panel(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    region_mask: np.ndarray | None,
    show_ylabel: bool,
    subtle_boundaries: bool = False,
    show_grid_labels: bool = True,
) -> Any:
    transform = ccrs.PlateCarree() if ccrs is not None and hasattr(ax, "projection") else None
    mesh_kwargs: dict[str, Any] = {}
    if transform is not None:
        mesh_kwargs["transform"] = transform
    mesh = ax.pcolormesh(
        lon,
        lat,
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
        zorder=2,
        **mesh_kwargs,
    )
    _style_map_axis(
        ax,
        lon,
        lat,
        show_ylabel=show_ylabel,
        subtle_boundaries=subtle_boundaries,
        show_grid_labels=show_grid_labels,
    )
    _draw_region_outline(
        ax,
        region_mask=region_mask,
        lon=lon,
        lat=lat,
        transform=transform,
        subtle=subtle_boundaries,
    )
    ax.set_title(title, fontsize=8, pad=4)
    ax.tick_params(labelsize=6, width=0.4, length=2)
    return mesh


def _plot_rmse_panel(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> Any:
    mesh = ax.imshow(
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        rasterized=True,
    )
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    return mesh


def _plot_grid_panel(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    coord_mode: str,
) -> Any:
    mesh = ax.pcolormesh(
        lon,
        lat,
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
    )
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(labelsize=7, width=0.5, length=2)
    if coord_mode.startswith("latlon"):
        ax.set_xlabel("Longitude", fontsize=7)
        ax.set_ylabel("Latitude", fontsize=7)
    else:
        ax.set_xlabel("Grid x", fontsize=7)
        ax.set_ylabel("Grid y", fontsize=7)
        ax.invert_yaxis()
    return mesh


def render_figure(
    sample: SampleMap,
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    coordinate_metadata: Mapping[str, Any],
    region_mask: np.ndarray | None = None,
    output_dir: str | Path,
    output_stem: str,
    formats: Sequence[str] = ("png", "pdf", "svg"),
    model_label: str = DEFAULT_MODEL_LABEL,
) -> dict[str, str]:
    configure_matplotlib()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    coord_mode = str(coordinate_metadata.get("coordinate_mode", "grid"))
    use_map_axes = coord_mode.startswith("latlon") and ccrs is not None and cfeature is not None

    fig, axes = create_figure_axes(use_map_axes=use_map_axes)
    column_titles = ("Ref analysis SM", "Pred analysis SM", "|Pred-Ref|")
    field_cmap = morandi_field_cmap()
    error_cmap = morandi_error_cmap()

    for row_idx, variable in enumerate(VARIABLE_ORDER):
        variable_map = sample.variables[variable]
        _, field_vmax = _robust_range(variable_map.ref, variable_map.pred)
        field_vmin = 0.0
        rmse_limit = _robust_positive_limit(variable_map.rmse_map)
        row_label = VARIABLE_LABELS[variable]
        panel_data = (variable_map.ref, variable_map.pred, variable_map.rmse_map)
        panel_specs = (
            (field_cmap, field_vmin, field_vmax),
            (field_cmap, field_vmin, field_vmax),
            (error_cmap, 0.0, rmse_limit),
        )
        for col_idx, (values, spec) in enumerate(zip(panel_data, panel_specs)):
            ax = axes[row_idx, col_idx]
            title = column_titles[col_idx] if row_idx == 0 else ""
            if use_map_axes:
                mesh = _plot_map_panel(
                    ax,
                    lon,
                    lat,
                    values,
                    title=title,
                    cmap=spec[0],
                    vmin=spec[1],
                    vmax=spec[2],
                    region_mask=region_mask,
                    show_ylabel=col_idx == 0,
                    subtle_boundaries=col_idx == 2,
                    show_grid_labels=col_idx < 2,
                )
            elif col_idx == 2:
                mesh = _plot_rmse_panel(
                    ax,
                    values,
                    title=title,
                    cmap=spec[0],
                    vmin=spec[1],
                    vmax=spec[2],
                )
            else:
                mesh = _plot_grid_panel(
                    ax,
                    lon,
                    lat,
                    values,
                    title=title,
                    cmap=spec[0],
                    vmin=spec[1],
                    vmax=spec[2],
                    coord_mode=coord_mode,
                )
            if col_idx == 0:
                ax.text(
                    -0.17,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
            if col_idx == 1:
                _add_compact_panel_colorbar(fig, mesh, anchor_ax=axes[row_idx, 1])
            elif col_idx == 2:
                _add_compact_panel_colorbar(fig, mesh, anchor_ax=axes[row_idx, 2])

    fig.suptitle(
        f"{sample.target_region_id} target_eval | HyperDA-TRUST K=0 | {sample.query_date}",
        fontsize=9,
        y=1.01,
    )

    outputs: dict[str, str] = {}
    for fmt in formats:
        clean_fmt = fmt.lower().lstrip(".")
        path = output_path / f"{output_stem}.{clean_fmt}"
        fig.savefig(path)
        outputs[clean_fmt] = str(path)
    plt.close(fig)
    return outputs


def sample_to_metadata(
    sample: SampleMap,
    *,
    prediction_record_path: Path,
    prediction_record_hash: str,
    checkpoint: Path,
    output_files: Mapping[str, str],
    candidate_review_files: Mapping[str, Any] | None = None,
    coordinate_metadata: Mapping[str, Any],
    crop_metadata: Mapping[str, Any],
    outline_metadata: Mapping[str, Any],
    model_label: str,
    candidate_id: str,
    max_samples: int,
) -> dict[str, Any]:
    variables = {
        name: {
            "model_wrmse": sample.variables[name].model_wrmse,
            "forecast_only_wrmse": sample.variables[name].forecast_wrmse,
            "relative_skill_vs_forecast": sample.variables[name].skill_vs_forecast,
            "true_increment_wrmse": sample.variables[name].true_increment_wrmse,
        }
        for name in VARIABLE_ORDER
    }
    return {
        "schema_version": "hydroda_ref_pred_error_map_selection_v2",
        "candidate_id": candidate_id,
        "model_label": model_label,
        "target_region": sample.target_region_id,
        "split_role": sample.split_role,
        "selection_usage": "visualization_only_not_model_selection",
        "selection_reason": sample.selection_reason,
        "selection_score": sample.selection_score,
        "selection_scores": {
            "selection_mode": sample.selection_metadata.get("selection_mode", ""),
            "combined_relative_skill_vs_forecast": sample.selection_metadata.get(
                "combined_relative_skill_vs_forecast",
                sample.selection_score,
            ),
            "combined_model_wrmse": sample.selection_metadata.get("combined_model_wrmse", float("nan")),
            "true_increment_strength": sample.selection_metadata.get(
                "true_increment_strength",
                _combined_true_increment_strength(sample),
            ),
            "candidate_pool_size": sample.selection_metadata.get("candidate_pool_size", 0),
            "positive_skill_candidate_count": sample.selection_metadata.get(
                "positive_skill_candidate_count",
                0,
            ),
            "eligible_candidate_count": sample.selection_metadata.get("eligible_candidate_count", 0),
            "signal_quantile": sample.selection_metadata.get("signal_quantile", float("nan")),
            "signal_threshold": sample.selection_metadata.get("signal_threshold"),
            "skill_quantile": sample.selection_metadata.get("skill_quantile"),
            "skill_threshold": sample.selection_metadata.get("skill_threshold"),
            "balanced_visual_score": sample.selection_metadata.get("balanced_visual_score"),
            "visual_selection_weights": sample.selection_metadata.get("visual_selection_weights"),
            "left_error_rmse_ratio_vs_full": sample.selection_metadata.get(
                "left_error_rmse_ratio_vs_full"
            ),
            "lower_left_error_rmse_ratio_vs_full": sample.selection_metadata.get(
                "lower_left_error_rmse_ratio_vs_full"
            ),
            "edge_error_concentration": sample.selection_metadata.get("edge_error_concentration"),
            "left_skill_vs_forecast": sample.selection_metadata.get("left_skill_vs_forecast"),
            "lower_left_skill_vs_forecast": sample.selection_metadata.get(
                "lower_left_skill_vs_forecast"
            ),
            "local_skill_floor": sample.selection_metadata.get("local_skill_floor"),
            "model_wrmse_rank": sample.selection_metadata.get("model_wrmse_rank"),
            "edge_concentration_rank": sample.selection_metadata.get("edge_concentration_rank"),
            "local_skill_rank": sample.selection_metadata.get("local_skill_rank"),
            "combined_skill_rank": sample.selection_metadata.get("combined_skill_rank"),
            "selection_rank": sample.selection_metadata.get("selection_rank"),
            "filtered_below_signal_threshold": sample.selection_metadata.get(
                "filtered_below_signal_threshold",
                [],
            ),
            "filtered_below_skill_threshold": sample.selection_metadata.get(
                "filtered_below_skill_threshold",
                [],
            ),
            "rank_key": sample.selection_metadata.get("rank_key", []),
        },
        "sample_idx": sample.sample_idx,
        "query_time_index": sample.query_time_index,
        "query_date": sample.query_date,
        "month": sample.month,
        "season": sample.season,
        "method": sample.method,
        "variables": variables,
        "record_metadata": sample.record_metadata,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.exists() else "",
        "prediction_record_path": str(prediction_record_path),
        "prediction_record_sha256": prediction_record_hash,
        "prediction_record_max_samples": int(max_samples),
        "coordinate_metadata": dict(coordinate_metadata),
        "crop_metadata": dict(crop_metadata),
        "outline_metadata": dict(outline_metadata),
        "figure_style": {
            "ref_pred_panels": "analysis_space_cartopy_map_with_us_state_boundaries_and_target_region_outline",
            "rmse_panels": "analysis_space_absolute_error_cartopy_map",
            "in_figure_text": "minimal",
            "palette": "low_saturation_morandi",
        },
        "caption": (
            "Ref is the physical DA analysis soil-moisture field; Pred is "
            "forecast_soil_moisture + model-predicted DA increment; the error "
            "panel is the per-pixel |Pred-Ref| analysis-space absolute error "
            "map for this single target_eval case, not an aggregate MAE or "
            "WRMSE panel."
        ),
        "leakage_note": (
            "The target_eval label-dependent sample selection is used only for "
            "this final visualization and is not used for training, adaptation, "
            "model selection, threshold calibration, or hyperparameter tuning."
        ),
        "candidate_review_files": dict(candidate_review_files or {}),
        "output_files": dict(output_files),
    }


def export_prediction_records_if_needed(
    *,
    record_path: Path,
    checkpoint: Path,
    target_region: str,
    K: int,
    seed: int,
    splits_json: Path,
    output_dir: Path,
    max_samples: int,
    batch_size: int,
    device: str,
    force: bool,
) -> None:
    if record_path.exists() and not force:
        return
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found for prediction export: {checkpoint}")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    eval_output_dir = output_dir / "eval_record_export"
    cmd = [
        sys.executable,
        "scripts/eval/evaluate_checkpoint.py",
        "--checkpoint",
        str(checkpoint),
        "--target_region",
        target_region,
        "--adaptation_setting",
        "zero_shot_context" if int(K) == 0 else f"few_shot_k{int(K)}",
        "--K",
        str(int(K)),
        "--seed",
        str(int(seed)),
        "--split_type",
        "target_eval",
        "--splits_json",
        str(splits_json),
        "--predictor_type",
        "hyperda_target_adapt",
        "--device",
        device,
        "--output_dir",
        str(eval_output_dir),
        "--max_samples",
        str(int(max_samples)),
        "--batch_size",
        str(int(batch_size)),
        "--adapt_mix_rho",
        "1.0",
        "--prediction_record_path",
        str(record_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "." if not env.get("PYTHONPATH") else f".{os.pathsep}{env['PYTHONPATH']}"
    subprocess.run(cmd, check=True, env=env)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a 2x3 analysis-space Ref / Pred / error target_eval map figure."
    )
    parser.add_argument("--target_region", default=DEFAULT_TARGET_REGION)
    parser.add_argument("--candidate_id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--model_label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--splits_json", type=Path, default=DEFAULT_SPLITS_JSON)
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_stem", default="ref_pred_rmse_map_US-R1_M3_1_K0")
    parser.add_argument(
        "--selection_mode",
        default="balanced_visual_with_signal",
        choices=[
            "balanced_visual_with_signal",
            "lowest_rmse_with_signal",
            "best_display_with_signal",
            "best_skill_with_signal",
            "positive_skill_high_true_increment",
        ],
        help="Visualization-only target_eval sample selection policy.",
    )
    parser.add_argument(
        "--min_signal_quantile",
        type=float,
        default=0.5,
        help=(
            "Minimum combined true-increment strength quantile among positive-skill "
            "candidate samples for lowest_rmse_with_signal selection."
        ),
    )
    parser.add_argument(
        "--min_skill_quantile",
        type=float,
        default=0.25,
        help=(
            "Minimum combined relative-skill quantile among positive-skill candidate "
            "samples before choosing the lowest-WRMSE display case."
        ),
    )
    parser.add_argument(
        "--lower_left_fraction",
        type=float,
        default=0.25,
        help="Panel fraction used for lower-left error diagnostic summary.",
    )
    parser.add_argument(
        "--render_sample_gallery",
        action="store_true",
        help="Render paginated all-sample error thumbnail galleries for manual display-case review.",
    )
    parser.add_argument(
        "--sample_gallery_page_size",
        type=int,
        default=32,
        help="Number of samples per gallery page when --render_sample_gallery is set.",
    )
    parser.add_argument("--prediction_record_path", type=Path, default=None)
    parser.add_argument("--force_export_records", action="store_true")
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--geolocation_nc", type=Path, default=DEFAULT_GEOLOCATION_NC)
    parser.add_argument("--crop_manifest", type=Path, default=DEFAULT_CROP_MANIFEST)
    parser.add_argument("--region_mask_nc", type=Path, default=DEFAULT_REGION_MASK_NC)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        help="Output formats to write.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.prediction_record_path
    if record_path is None:
        record_path = args.output_dir / f"prediction_records_target_eval_max{int(args.max_samples)}.jsonl"

    if not args.skip_export:
        export_prediction_records_if_needed(
            record_path=record_path,
            checkpoint=args.checkpoint,
            target_region=args.target_region,
            K=args.K,
            seed=args.seed,
            splits_json=args.splits_json,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            device=args.device,
            force=args.force_export_records,
        )
    elif not record_path.exists():
        raise FileNotFoundError(f"--skip_export was set but record path does not exist: {record_path}")

    records = load_prediction_records(record_path)
    raw_samples = [build_sample_map(record) for record in records]
    samples: list[SampleMap] = []
    crop_metadata_by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for sample in raw_samples:
        cropped_sample, sample_crop_metadata = crop_sample_to_region_if_full_grid(
            sample,
            crop_manifest=args.crop_manifest,
            region_id=args.target_region,
        )
        samples.append(cropped_sample)
        crop_metadata_by_key[_sample_key(cropped_sample)] = sample_crop_metadata

    selected = select_representative_sample(
        samples,
        selection_mode=args.selection_mode,
        min_signal_quantile=args.min_signal_quantile,
        min_skill_quantile=args.min_skill_quantile,
    )
    crop_metadata = crop_metadata_by_key.get(_sample_key(selected), {"crop_applied": False})
    candidate_ranking = build_candidate_ranking(
        samples,
        selected=selected,
        min_signal_quantile=args.min_signal_quantile,
        min_skill_quantile=args.min_skill_quantile,
    )
    candidate_review_files: dict[str, Any] = write_candidate_ranking_files(
        candidate_ranking,
        output_dir=args.output_dir,
    )
    if args.render_sample_gallery:
        candidate_review_files.update(
            render_sample_error_gallery(
                samples,
                ranking=candidate_ranking,
                output_dir=args.output_dir,
                output_stem="sample_error_gallery",
                page_size=args.sample_gallery_page_size,
                formats=("png",),
            )
        )
    shape = selected.variables["surface"].ref.shape
    lon, lat, coordinate_metadata = load_region_latlon(
        geolocation_nc=args.geolocation_nc,
        crop_manifest=args.crop_manifest,
        region_id=args.target_region,
        expected_shape=shape,
    )
    region_mask, outline_metadata = load_region_mask_crop(
        region_mask_nc=args.region_mask_nc,
        crop_manifest=args.crop_manifest,
        region_id=args.target_region,
        expected_shape=shape,
    )
    output_files = render_figure(
        selected,
        lon=lon,
        lat=lat,
        coordinate_metadata=coordinate_metadata,
        region_mask=region_mask,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        formats=args.formats,
        model_label=args.model_label,
    )
    record_hash = sha256_file(record_path)
    metadata = sample_to_metadata(
        selected,
        prediction_record_path=record_path,
        prediction_record_hash=record_hash,
        checkpoint=args.checkpoint,
        output_files=output_files,
        candidate_review_files=candidate_review_files,
        coordinate_metadata=coordinate_metadata,
        crop_metadata=crop_metadata,
        outline_metadata=outline_metadata,
        model_label=args.model_label,
        candidate_id=args.candidate_id,
        max_samples=args.max_samples,
    )
    metadata_path = args.output_dir / "selected_sample.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    diagnostic_summary = build_error_diagnostic_summary(
        samples,
        lower_left_fraction=args.lower_left_fraction,
    )
    diagnostic_path = args.output_dir / "lower_left_error_diagnostic.json"
    diagnostic_path.write_text(
        json.dumps(diagnostic_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_sample": str(metadata_path),
                "error_diagnostic": str(diagnostic_path),
                "candidate_review_files": candidate_review_files,
                "output_files": output_files,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
