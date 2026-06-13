"""Evaluation harness for HydroDA-OOD / HyperDA V4."""
from __future__ import annotations

import hashlib
import json
import zlib
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from hydroda.metrics.skill import (
    compute_variable_metrics,
    effective_mask_fraction,
    valid_pixel_count,
    weighted_mse,
    weighted_mae,
    weighted_bias,
    weighted_corr,
    weighted_analysis_skill_components,
)

_VARIABLES = {
    "surface": {
        "forecast": "forecast_surface",
        "analysis": "analysis_surface",
        "increment": "increment_surface",
        "pred_increment": "pred_increment_surface",
        "pred_analysis": "pred_analysis_surface",
    },
    "rootzone": {
        "forecast": "forecast_rootzone",
        "analysis": "analysis_rootzone",
        "increment": "increment_rootzone",
        "pred_increment": "pred_increment_rootzone",
        "pred_analysis": "pred_analysis_rootzone",
    },
}


def _json_hash_update(hasher: "hashlib._Hash", payload: Dict[str, Any]) -> None:
    hasher.update(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8"))
    hasher.update(b"\n")


def _array_hash_payload(array: Any) -> Dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def _array_record_payload(array: Any, *, inline_value_limit: int = 4096) -> Dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    payload = _array_hash_payload(arr)
    flat = arr.reshape(-1)
    if flat.size <= inline_value_limit:
        payload["values"] = flat.tolist()
    else:
        compressed = zlib.compress(arr.tobytes())
        payload["encoding"] = "float32_zlib_base64"
        payload["data"] = base64.b64encode(compressed).decode("ascii")
    return payload


def prediction_record_array(payload: Dict[str, Any]) -> np.ndarray:
    """Decode an array payload written by ``evaluate_split`` prediction records."""
    shape = tuple(int(dim) for dim in payload.get("shape", []))
    dtype = np.dtype(str(payload.get("dtype", "float32")))
    if "values" in payload:
        arr = np.asarray(payload["values"], dtype=dtype).reshape(shape)
    elif payload.get("encoding") == "float32_zlib_base64":
        raw = zlib.decompress(base64.b64decode(str(payload.get("data", "")).encode("ascii")))
        arr = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    else:
        raise ValueError("Prediction record array payload lacks values or supported encoding")
    expected = str(payload.get("sha256", ""))
    if expected:
        actual = hashlib.sha256(np.ascontiguousarray(arr.astype(np.float32)).tobytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Prediction record array hash mismatch: expected {expected}, got {actual}")
    return arr.astype(np.float32, copy=False)


def _prediction_hash_payload(sample_idx: int, sample: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_idx": int(sample_idx),
        "query_time_index": int(sample.get("time_index", -1)),
        "query_date": str(sample.get("date_str", "")),
        "pred_increment_surface": _array_hash_payload(pred.get("pred_increment_surface")),
        "pred_increment_rootzone": _array_hash_payload(pred.get("pred_increment_rootzone")),
    }


def _single_prediction_content_hash(payload: Dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    _json_hash_update(hasher, payload)
    return hasher.hexdigest()


def _prediction_record(
    *,
    sample_idx: int,
    sample: Dict[str, Any],
    pred: Dict[str, Any],
    split_role: str,
    experiment_id: str,
    method: str,
    protocol_freeze_id: str,
    split_file: str,
    split_manifest_sha256: str,
    adapt_mix_rho: float,
) -> Dict[str, Any]:
    prediction_payload = _prediction_hash_payload(sample_idx, sample, pred)
    arrays = {
        "forecast_surface": _array_record_payload(sample["forecast_surface"]),
        "forecast_rootzone": _array_record_payload(sample["forecast_rootzone"]),
        "analysis_surface": _array_record_payload(sample["analysis_surface"]),
        "analysis_rootzone": _array_record_payload(sample["analysis_rootzone"]),
        "increment_surface": _array_record_payload(sample["increment_surface"]),
        "increment_rootzone": _array_record_payload(sample["increment_rootzone"]),
        "pred_increment_surface": _array_record_payload(pred["pred_increment_surface"]),
        "pred_increment_rootzone": _array_record_payload(pred["pred_increment_rootzone"]),
        "metric_mask": _array_record_payload(sample["metric_mask"]),
        "latitude_weight": _array_record_payload(sample["latitude_weight"]),
    }
    return {
        "schema_version": "hydroda_prediction_record_v1",
        "sample_idx": int(sample_idx),
        "query_time_index": int(sample.get("time_index", -1)),
        "query_date": str(sample.get("date_str", "")),
        "month": sample.get("month", None),
        "season": sample.get("season", ""),
        "country_id": sample.get("country_id", ""),
        "target_region_id": sample.get("target_region_id", ""),
        "sample_region_id": sample.get("sample_region_id", ""),
        "active_region_ids": list(sample.get("active_region_ids", [])),
        "split_role": split_role,
        "adaptation_setting": sample.get("adaptation_setting", "zero_shot_context"),
        "K": sample.get("K", "legacy_none") if sample.get("K") is not None else "legacy_none",
        "seed": int(sample.get("seed", -1)),
        "experiment_id": experiment_id,
        "method": method,
        "protocol_freeze_id": protocol_freeze_id,
        "split_file": split_file,
        "split_manifest_sha256": sample.get("split_manifest_sha256") or split_manifest_sha256,
        "target_context_dates_hash": sample.get("target_context_dates_hash", ""),
        "target_support_dates_hash": sample.get("target_support_dates_hash", ""),
        "support_dates_hash": sample.get("support_dates_hash", ""),
        "target_train_dates_hash": sample.get("target_train_dates_hash", ""),
        "target_eval_dates_hash": sample.get("target_eval_dates_hash", ""),
        "adapt_mix_rho": float(adapt_mix_rho),
        "prediction_content_hash": _single_prediction_content_hash(prediction_payload),
        "arrays": arrays,
    }


def mix_prediction_with_zero_shot(
    sample: Dict[str, Any],
    adapted_pred: Dict[str, Any],
    zero_shot_pred: Dict[str, Any],
    *,
    rho: float,
) -> Dict[str, np.ndarray]:
    """Mix adapted increments with same-context K0 increments at fixed rho."""
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    mixed: Dict[str, np.ndarray] = dict(adapted_pred)
    for variable in ("surface", "rootzone"):
        inc_key = f"pred_increment_{variable}"
        forecast_key = f"forecast_{variable}"
        analysis_key = f"pred_analysis_{variable}"
        zero_inc = np.asarray(zero_shot_pred[inc_key], dtype=np.float32)
        adapted_inc = np.asarray(adapted_pred[inc_key], dtype=np.float32)
        final_inc = (zero_inc + rho * (adapted_inc - zero_inc)).astype(np.float32)
        forecast = np.asarray(sample[forecast_key], dtype=np.float32)
        mixed[inc_key] = final_inc
        mixed[analysis_key] = (forecast + final_inc).astype(np.float32)
    return mixed


def prediction_pair_delta_summary(
    left_pred: Dict[str, Any],
    right_pred: Dict[str, Any],
) -> Dict[str, float]:
    """Return mean/max absolute increment differences for two prediction dicts."""
    diffs = []
    for key in ("pred_increment_surface", "pred_increment_rootzone"):
        left = np.asarray(left_pred[key], dtype=np.float32)
        right = np.asarray(right_pred[key], dtype=np.float32)
        diffs.append(np.abs(left - right).reshape(-1))
    if not diffs:
        return {"mean_abs": 0.0, "max_abs": 0.0}
    all_diff = np.concatenate(diffs)
    return {
        "mean_abs": float(np.mean(all_diff)),
        "max_abs": float(np.max(all_diff)),
    }


def metric_rows_content_hash(rows: List[Dict[str, Any]]) -> str:
    """Return a deterministic content hash for in-memory metric rows."""
    hasher = hashlib.sha256()
    for row in rows:
        _json_hash_update(hasher, row)
    return hasher.hexdigest()


def metric_values_content_hash(rows: List[Dict[str, Any]]) -> str:
    """Hash metric identities and values while ignoring run/method/K metadata."""
    hasher = hashlib.sha256()
    keys = [
        "query_date",
        "query_time_index",
        "variable",
        "metric",
        "value",
        "n_valid_pixels",
        "n_time_steps",
        "mask_fraction",
    ]
    for row in rows:
        _json_hash_update(hasher, {key: row.get(key) for key in keys})
    return hasher.hexdigest()


def _make_metric_row(
    *,
    experiment_id: str,
    method: str,
    sample: Dict[str, Any],
    sample_idx: int,
    split_role: str,
    split_file: str,
    mask_file: str,
    target_context_dates_hash: str,
    target_support_dates_hash: str,
    support_dates_hash: str,
    target_train_dates_hash: str,
    target_eval_dates_hash: str,
    split_manifest_sha256: str,
    protocol_freeze_id: str,
    n_valid: int,
    mask_frac: float,
    variable: str,
    metric_name: str,
    value: float,
) -> Dict[str, Any]:
    """Build a single evaluation metric row with all protocol metadata."""
    k_value = sample.get("K", "legacy_none")
    if k_value is None:
        k_value = "legacy_none"
    return {
        "experiment_id": experiment_id,
        "run_id": f"{experiment_id}_{method}_{sample.get('time_index', sample_idx)}",
        "method": method,
        "query_date": sample.get("date_str", ""),
        "query_time_index": int(sample.get("time_index", -1)),
        "month": sample.get("month", None),
        "season": sample.get("season", ""),
        "target_context_dates_hash": sample.get("target_context_dates_hash") or target_context_dates_hash,
        "target_support_dates_hash": sample.get("target_support_dates_hash") or target_support_dates_hash,
        "support_dates_hash": sample.get("support_dates_hash") or support_dates_hash,
        "target_train_dates_hash": sample.get("target_train_dates_hash") or target_train_dates_hash,
        "target_eval_dates_hash": sample.get("target_eval_dates_hash") or target_eval_dates_hash,
        "split_manifest_sha256": sample.get("split_manifest_sha256") or split_manifest_sha256,
        "split_file": split_file,
        "mask_file": mask_file,
        "country_id": sample.get("country_id", ""),
        "target_region_id": sample.get("target_region_id", ""),
        "sample_region_id": sample.get("sample_region_id", ""),
        "active_region_ids": "|".join(sample.get("active_region_ids", [])),
        "split_role": split_role,
        "adaptation_setting": sample.get("adaptation_setting", "zero_shot_context"),
        "K": k_value,
        "K_legacy": sample.get("K_legacy", None),
        "seed": int(sample.get("seed", -1)),
        "variable": variable,
        "metric": metric_name,
        "value": value,
        "n_valid_pixels": n_valid,
        "n_time_steps": 1,
        "mask_fraction": mask_frac,
        "protocol_freeze_id": protocol_freeze_id,
    }


def evaluate_split(
    dataset: Any,
    predictor: Any,
    *,
    split_role: str,
    experiment_id: str,
    protocol_freeze_id: str,
    method: str,
    split_file: str = "",
    mask_file: str = "",
    target_context_dates_hash: str = "",
    target_support_dates_hash: str = "",
    support_dates_hash: str = "",
    target_train_dates_hash: str = "",
    target_eval_dates_hash: str = "",
    split_manifest_sha256: str = "",
    deadzone_epsilon: float = 0.005,
    high_update_top_fraction: float = 0.2,
    preloaded: bool = True,
    max_samples: Optional[int] = None,
    return_hashes: bool = False,
    zero_shot_predictor: Any = None,
    adapt_mix_rho: float = 1.0,
    prediction_record_path: Optional[Union[str, Path]] = None,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """Evaluate a predictor over a dataset.

    ``predictor`` must implement ``predict(sample) -> dict`` and return both
    pred_increment_* and pred_analysis_* for surface/rootzone.

    Args:
        max_samples: If set, only evaluate the first max_samples samples.
    """
    all_samples = dataset.preload() if preloaded and hasattr(dataset, "preload") else None
    rows: List[Dict[str, Any]] = []
    prediction_hasher = hashlib.sha256()
    zero_prediction_hasher = hashlib.sha256()
    adapted_prediction_hasher = hashlib.sha256()
    prediction_record_count = 0
    prediction_record_file: Optional[TextIO] = None
    mix_delta_from_zero_abs_sum = 0.0
    mix_delta_from_adapted_abs_sum = 0.0
    mix_delta_element_count = 0
    mix_delta_from_zero_max_abs = 0.0
    mix_delta_from_adapted_max_abs = 0.0

    # Global accumulators for correct skill aggregation (aggregate-then-sqrt,
    # not per-sample mean-of-ratios that produces spuriously negative skills).
    global_accum: Dict[str, Dict[str, float]] = {}
    for var_name in ["surface", "rootzone"]:
        global_accum[var_name] = {
            # Non-latw: accumulate squared errors + pixel count
            "model_sse": 0.0,
            "forecast_sse": 0.0,
            "n_pixels": 0,
            # Latw: accumulate weighted squared errors + weight sum
            "model_sse_latw": 0.0,
            "forecast_sse_latw": 0.0,
            "weight_sum": 0.0,
        }
    first_sample: Optional[Dict[str, Any]] = None

    n_eval = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    if prediction_record_path is not None:
        record_path = Path(prediction_record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_record_file = record_path.open("w", encoding="utf-8")

    try:
        for idx in tqdm(range(n_eval), total=n_eval, desc=f"Evaluating {split_role}", unit="samples"):
            sample = all_samples[idx] if all_samples is not None else dataset[idx]
            if first_sample is None:
                first_sample = sample
            adapted_pred = predictor.predict(sample)
            zero_pred = None
            pred = adapted_pred
            if zero_shot_predictor is not None or float(adapt_mix_rho) != 1.0:
                if zero_shot_predictor is None:
                    raise ValueError("zero_shot_predictor is required when adapt_mix_rho != 1.0")
                zero_pred = zero_shot_predictor.predict(sample)
                pred = mix_prediction_with_zero_shot(sample, adapted_pred, zero_pred, rho=float(adapt_mix_rho))
                delta_zero = prediction_pair_delta_summary(pred, zero_pred)
                delta_adapted = prediction_pair_delta_summary(pred, adapted_pred)
                elem_count = int(
                    np.asarray(pred["pred_increment_surface"]).size
                    + np.asarray(pred["pred_increment_rootzone"]).size
                )
                mix_delta_from_zero_abs_sum += delta_zero["mean_abs"] * elem_count
                mix_delta_from_adapted_abs_sum += delta_adapted["mean_abs"] * elem_count
                mix_delta_element_count += elem_count
                mix_delta_from_zero_max_abs = max(mix_delta_from_zero_max_abs, delta_zero["max_abs"])
                mix_delta_from_adapted_max_abs = max(mix_delta_from_adapted_max_abs, delta_adapted["max_abs"])
            prediction_payload = _prediction_hash_payload(idx, sample, pred)
            if return_hashes:
                if zero_pred is not None:
                    _json_hash_update(zero_prediction_hasher, _prediction_hash_payload(idx, sample, zero_pred))
                    _json_hash_update(adapted_prediction_hasher, _prediction_hash_payload(idx, sample, adapted_pred))
                _json_hash_update(prediction_hasher, prediction_payload)
                prediction_record_count += 1
            if prediction_record_file is not None:
                record = _prediction_record(
                    sample_idx=idx,
                    sample=sample,
                    pred=pred,
                    split_role=split_role,
                    experiment_id=experiment_id,
                    method=method,
                    protocol_freeze_id=protocol_freeze_id,
                    split_file=split_file,
                    split_manifest_sha256=split_manifest_sha256,
                    adapt_mix_rho=float(adapt_mix_rho),
                )
                prediction_record_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            mask = sample["metric_mask"]
            n_valid = valid_pixel_count(mask)
            mask_frac = effective_mask_fraction(mask)

            for variable, keys in _VARIABLES.items():
                missing = [k for k in (keys["pred_increment"], keys["pred_analysis"]) if k not in pred]
                if missing:
                    raise KeyError(f"Predictor output missing keys: {missing}")

                metrics = compute_variable_metrics(
                    pred_analysis=pred[keys["pred_analysis"]],
                    true_analysis=sample[keys["analysis"]],
                    forecast=sample[keys["forecast"]],
                    pred_increment=pred[keys["pred_increment"]],
                    true_increment=sample[keys["increment"]],
                    mask=mask,
                    deadzone_epsilon=deadzone_epsilon,
                    high_update_top_fraction=high_update_top_fraction,
                )

                for metric_name, value in metrics.items():
                    rows.append(_make_metric_row(
                        experiment_id=experiment_id, method=method, sample=sample, sample_idx=idx,
                        split_role=split_role, split_file=split_file, mask_file=mask_file,
                        target_context_dates_hash=target_context_dates_hash,
                        target_support_dates_hash=target_support_dates_hash,
                        support_dates_hash=support_dates_hash, protocol_freeze_id=protocol_freeze_id,
                        target_train_dates_hash=target_train_dates_hash,
                        target_eval_dates_hash=target_eval_dates_hash,
                        split_manifest_sha256=split_manifest_sha256,
                        n_valid=n_valid, mask_frac=mask_frac, variable=variable,
                        metric_name=metric_name, value=value,
                    ))

                # Latitude-weighted metrics (per-sample)
                latw = sample.get("latitude_weight")
                if latw is None:
                    raise ValueError(
                        f"Sample {idx} missing 'latitude_weight'. "
                        f"Dataset must return latitude_weight for evaluation harness."
                    )

                pred_analysis = pred[keys["pred_analysis"]]
                true_analysis = sample[keys["analysis"]]
                forecast = sample[keys["forecast"]]
                pred_increment = pred[keys["pred_increment"]]
                true_increment = sample[keys["increment"]]

                # Lat-weighted metrics
                mse_m, mse_f = weighted_analysis_skill_components(
                    pred_analysis=pred_analysis,
                    true_analysis=true_analysis,
                    forecast=forecast,
                    mask=mask,
                    latitude_weight=latw,
                )
                rmse_model_latw = float(np.sqrt(mse_m)) if np.isfinite(mse_m) and mse_m > 0 else float("nan")
                rmse_forecast_latw = float(np.sqrt(mse_f)) if np.isfinite(mse_f) and mse_f > 0 else float("nan")
                skill_latw = float(1.0 - rmse_model_latw / rmse_forecast_latw) if np.isfinite(rmse_model_latw) and np.isfinite(rmse_forecast_latw) and rmse_forecast_latw > 0 else float("nan")

                inc_rmse_latw_val = weighted_mse(pred_increment, true_increment, mask, latw)
                inc_rmse_latw = float(np.sqrt(inc_rmse_latw_val)) if np.isfinite(inc_rmse_latw_val) and inc_rmse_latw_val > 0 else float("nan")
                inc_mae_latw = weighted_mae(pred_increment, true_increment, mask, latw)
                inc_bias_latw = weighted_bias(pred_increment, true_increment, mask, latw)
                inc_corr_latw = weighted_corr(pred_increment, true_increment, mask, latw)

                latw_metrics = {
                    "analysis_mse_latw": mse_m,
                    "analysis_rmse_latw": rmse_model_latw,
                    "analysis_rmse_sqrt_before_time_avg_latw": rmse_model_latw,
                    "analysis_skill_vs_forecast_latw": skill_latw,
                    "increment_mse_latw": inc_rmse_latw_val,
                    "increment_rmse_latw": inc_rmse_latw,
                    "increment_rmse_sqrt_before_time_avg_latw": inc_rmse_latw,
                    "increment_mae_latw": inc_mae_latw,
                    "increment_bias_latw": inc_bias_latw,
                    "increment_corr_latw": inc_corr_latw,
                }

                for metric_name, value in latw_metrics.items():
                    rows.append(_make_metric_row(
                        experiment_id=experiment_id, method=method, sample=sample, sample_idx=idx,
                        split_role=split_role, split_file=split_file, mask_file=mask_file,
                        target_context_dates_hash=target_context_dates_hash,
                        target_support_dates_hash=target_support_dates_hash,
                        support_dates_hash=support_dates_hash, protocol_freeze_id=protocol_freeze_id,
                        target_train_dates_hash=target_train_dates_hash,
                        target_eval_dates_hash=target_eval_dates_hash,
                        split_manifest_sha256=split_manifest_sha256,
                        n_valid=n_valid, mask_frac=mask_frac, variable=variable,
                        metric_name=metric_name, value=value,
                    ))

                # -- Global (cross-sample) accumulation for correct skill aggregation --
                # Non-latw: accumulate squared errors over all valid pixels
                valid_nolatw = (
                    (mask > 0.5)
                    & np.isfinite(pred_analysis)
                    & np.isfinite(true_analysis)
                    & np.isfinite(forecast)
                )
                if valid_nolatw.sum() > 0:
                    pa_v = pred_analysis[valid_nolatw]
                    ta_v = true_analysis[valid_nolatw]
                    fc_v = forecast[valid_nolatw]
                    global_accum[variable]["model_sse"] += float(np.sum((pa_v - ta_v) ** 2))
                    global_accum[variable]["forecast_sse"] += float(np.sum((fc_v - ta_v) ** 2))
                    global_accum[variable]["n_pixels"] += int(valid_nolatw.sum())

                # Latw: accumulate weighted squared errors
                valid_latw = (
                    (mask > 0.5)
                    & np.isfinite(latw)
                    & (latw >= 0)
                    & np.isfinite(pred_analysis)
                    & np.isfinite(true_analysis)
                    & np.isfinite(forecast)
                )
                if valid_latw.sum() > 0:
                    w = latw[valid_latw]
                    pa_v = pred_analysis[valid_latw]
                    ta_v = true_analysis[valid_latw]
                    fc_v = forecast[valid_latw]
                    global_accum[variable]["model_sse_latw"] += float(np.sum(w * (pa_v - ta_v) ** 2))
                    global_accum[variable]["forecast_sse_latw"] += float(np.sum(w * (fc_v - ta_v) ** 2))
                    global_accum[variable]["weight_sum"] += float(np.sum(w))
    finally:
        if prediction_record_file is not None:
            prediction_record_file.close()

    # -- Post-loop: compute global (cross-sample) skill metrics --
    if first_sample is not None:
        for var_name in ["surface", "rootzone"]:
            acc = global_accum[var_name]
            # Use a summary of n_valid from all samples for the global row
            total_n_pixels = int(acc["n_pixels"])

            # Global non-latw skill
            if acc["n_pixels"] > 0 and acc["forecast_sse"] > 0:
                global_rmse_model = float(np.sqrt(acc["model_sse"] / acc["n_pixels"]))
                global_rmse_forecast = float(np.sqrt(acc["forecast_sse"] / acc["n_pixels"]))
                global_skill = float(1.0 - global_rmse_model / global_rmse_forecast) if global_rmse_forecast > 0 else float("nan")
            else:
                global_skill = float("nan")

            rows.append({
                "experiment_id": experiment_id,
                "run_id": f"{experiment_id}_{method}_global",
                "method": method,
                "query_date": "global",
                "query_time_index": -1,
                "month": None,
                "season": "global",
                "target_context_dates_hash": first_sample.get("target_context_dates_hash") or target_context_dates_hash,
                "target_support_dates_hash": first_sample.get("target_support_dates_hash") or target_support_dates_hash,
                "support_dates_hash": first_sample.get("support_dates_hash") or support_dates_hash,
                "target_train_dates_hash": first_sample.get("target_train_dates_hash") or target_train_dates_hash,
                "target_eval_dates_hash": first_sample.get("target_eval_dates_hash") or target_eval_dates_hash,
                "split_manifest_sha256": first_sample.get("split_manifest_sha256") or split_manifest_sha256,
                "split_file": split_file,
                "mask_file": mask_file,
                "country_id": first_sample.get("country_id", ""),
                "target_region_id": first_sample.get("target_region_id", ""),
                "sample_region_id": first_sample.get("sample_region_id", ""),
                "active_region_ids": "|".join(first_sample.get("active_region_ids", [])),
                "split_role": split_role,
                "adaptation_setting": first_sample.get("adaptation_setting", "zero_shot_context"),
                "K": first_sample.get("K", "legacy_none") if first_sample.get("K") is not None else "legacy_none",
                "K_legacy": first_sample.get("K_legacy", None),
                "seed": int(first_sample.get("seed", -1)),
                "variable": var_name,
                "metric": "analysis_skill_vs_forecast_global",
                "value": global_skill,
                "n_valid_pixels": total_n_pixels,
                "n_time_steps": 1,
                "mask_fraction": float("nan"),
                "protocol_freeze_id": protocol_freeze_id,
            })

            # Global latw skill
            if acc["weight_sum"] > 0 and acc["forecast_sse_latw"] > 0:
                global_model_mse_latw = acc["model_sse_latw"] / acc["weight_sum"]
                global_forecast_mse_latw = acc["forecast_sse_latw"] / acc["weight_sum"]
                if global_forecast_mse_latw > 0:
                    global_skill_latw = float(1.0 - np.sqrt(global_model_mse_latw) / np.sqrt(global_forecast_mse_latw))
                else:
                    global_skill_latw = float("nan")
            else:
                global_skill_latw = float("nan")

            rows.append({
                "experiment_id": experiment_id,
                "run_id": f"{experiment_id}_{method}_global",
                "method": method,
                "query_date": "global",
                "query_time_index": -1,
                "month": None,
                "season": "global",
                "target_context_dates_hash": first_sample.get("target_context_dates_hash") or target_context_dates_hash,
                "target_support_dates_hash": first_sample.get("target_support_dates_hash") or target_support_dates_hash,
                "support_dates_hash": first_sample.get("support_dates_hash") or support_dates_hash,
                "target_train_dates_hash": first_sample.get("target_train_dates_hash") or target_train_dates_hash,
                "target_eval_dates_hash": first_sample.get("target_eval_dates_hash") or target_eval_dates_hash,
                "split_manifest_sha256": first_sample.get("split_manifest_sha256") or split_manifest_sha256,
                "split_file": split_file,
                "mask_file": mask_file,
                "country_id": first_sample.get("country_id", ""),
                "target_region_id": first_sample.get("target_region_id", ""),
                "sample_region_id": first_sample.get("sample_region_id", ""),
                "active_region_ids": "|".join(first_sample.get("active_region_ids", [])),
                "split_role": split_role,
                "adaptation_setting": first_sample.get("adaptation_setting", "zero_shot_context"),
                "K": first_sample.get("K", "legacy_none") if first_sample.get("K") is not None else "legacy_none",
                "K_legacy": first_sample.get("K_legacy", None),
                "seed": int(first_sample.get("seed", -1)),
                "variable": var_name,
                "metric": "analysis_skill_vs_forecast_latw_global",
                "value": global_skill_latw,
                "n_valid_pixels": total_n_pixels,
                "n_time_steps": 1,
                "mask_fraction": float("nan"),
                "protocol_freeze_id": protocol_freeze_id,
            })

    if return_hashes:
        hashes = {
            "prediction_content_hash": prediction_hasher.hexdigest(),
            "prediction_record_count": prediction_record_count,
            "metric_content_hash": metric_rows_content_hash(rows),
            "metric_values_content_hash": metric_values_content_hash(rows),
            "metric_row_count": len(rows),
            "adapt_mix_rho": float(adapt_mix_rho),
            "zero_shot_prediction_content_hash": zero_prediction_hasher.hexdigest() if mix_delta_element_count else "",
            "adapted_pre_mix_prediction_content_hash": adapted_prediction_hasher.hexdigest() if mix_delta_element_count else "",
            "final_mixed_prediction_content_hash": prediction_hasher.hexdigest(),
            "mix_mean_abs_change_from_k0": (
                float(mix_delta_from_zero_abs_sum / mix_delta_element_count) if mix_delta_element_count else 0.0
            ),
            "mix_max_abs_change_from_k0": float(mix_delta_from_zero_max_abs),
            "mix_mean_abs_change_from_adapted": (
                float(mix_delta_from_adapted_abs_sum / mix_delta_element_count) if mix_delta_element_count else 0.0
            ),
            "mix_max_abs_change_from_adapted": float(mix_delta_from_adapted_max_abs),
        }
        return rows, hashes

    return rows


def summarize_metric_rows(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Summarize evaluation rows with aggregate skill as the primary diagnostic.

    Per-sample skill is kept as a diagnostic distribution because averaging
    ratios can be dominated by tiny forecast-RMSE denominators.
    """
    summary: Dict[str, Dict[str, float]] = {}
    if df.empty:
        return summary

    per_sample = df[df["query_date"] != "global"] if "query_date" in df.columns else df

    def _single(metric: str, variable: str) -> float:
        match = df[(df["metric"] == metric) & (df["variable"] == variable)]
        if len(match) > 0:
            return float(match["value"].mean())
        return float("nan")

    def _mean(metric: str, variable: str) -> float:
        match = per_sample[(per_sample["metric"] == metric) & (per_sample["variable"] == variable)]
        if len(match) > 0:
            return float(match["value"].mean())
        return float("nan")

    for variable in sorted(df["variable"].dropna().unique().tolist()):
        skill = per_sample[
            (per_sample["metric"] == "analysis_skill_vs_forecast")
            & (per_sample["variable"] == variable)
        ]["value"].astype(float)
        skill_latw = per_sample[
            (per_sample["metric"] == "analysis_skill_vs_forecast_latw")
            & (per_sample["variable"] == variable)
        ]["value"].astype(float)

        var_summary: Dict[str, float] = {
            "skill_primary": _single("analysis_skill_vs_forecast_global", variable),
            "skill_latw_primary": _single("analysis_skill_vs_forecast_latw_global", variable),
            "skill_global": _single("analysis_skill_vs_forecast_global", variable),
            "skill_latw_global": _single("analysis_skill_vs_forecast_latw_global", variable),
            "rmse_mean": _mean("increment_rmse", variable),
            "corr_mean": _mean("increment_corr", variable),
            # Per-sample latw aggregations
            "rmse_latw_mean": _mean("increment_rmse_latw", variable),
            "corr_latw_mean": _mean("increment_corr_latw", variable),
        }
        if len(skill) > 0:
            var_summary.update({
                "skill_mean": float(skill.mean()),
                "skill_std": float(skill.std()),
                "skill_median": float(skill.median()),
                "skill_p05": float(skill.quantile(0.05)),
                "skill_p95": float(skill.quantile(0.95)),
                "skill_negative_outlier_count": int((skill < -10.0).sum()),
            })
        else:
            var_summary.update({
                "skill_mean": float("nan"),
                "skill_std": float("nan"),
                "skill_median": float("nan"),
                "skill_p05": float("nan"),
                "skill_p95": float("nan"),
                "skill_negative_outlier_count": 0,
            })
        # Per-sample skill latw aggregation
        if len(skill_latw) > 0:
            var_summary.update({
                "skill_latw_mean": float(skill_latw.mean()),
                "skill_latw_std": float(skill_latw.std()),
                "skill_latw_median": float(skill_latw.median()),
            })
        else:
            var_summary.update({
                "skill_latw_mean": float("nan"),
                "skill_latw_std": float("nan"),
                "skill_latw_median": float("nan"),
            })
        summary[str(variable)] = var_summary

    return summary


KEY_METRICS = [
    "analysis_skill_vs_forecast_latw",
    "increment_rmse_latw",
    "increment_corr_latw",
]


def build_per_region_summary(
    df_all: pd.DataFrame,
    results_dir: Path,
) -> dict:
    """Build per-region summary from evaluation rows. Saves CSVs/JSON. Returns summary dict."""
    # Exclude global rows, group by region x variable x metric
    per_sample = df_all[df_all["query_date"] != "global"].copy()
    metrics_by_region = (
        per_sample.groupby(["sample_region_id", "variable", "metric"])
        .agg(
            mean_value=("value", "mean"),
            std_value=("value", "std"),
            n_samples=("value", "count"),
        )
        .reset_index()
        .rename(columns={"mean_value": "mean", "std_value": "std"})
    )

    # Save CSVs
    df_all.to_csv(results_dir / "metrics_long.csv", index=False)
    metrics_by_region.to_csv(results_dir / "metrics_by_region.csv", index=False)

    # Build per_region_summary.json
    summary: dict = {}
    for region_id in sorted(metrics_by_region["sample_region_id"].unique()):
        rdf = metrics_by_region[metrics_by_region["sample_region_id"] == region_id]
        rd: dict = {}
        for _, row in rdf.iterrows():
            var = str(row["variable"])
            metric = str(row["metric"])
            if metric in KEY_METRICS:
                rd.setdefault(var, {})[metric] = {
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "n": int(row["n_samples"]),
                }
        summary[str(region_id)] = rd

    with open(results_dir / "per_region_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
