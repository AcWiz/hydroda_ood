"""Evaluation harness for HydroDA-OOD / HyperDA V4."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _make_metric_row(
    *,
    experiment_id: str,
    method: str,
    sample: Dict[str, Any],
    sample_idx: int,
    split_role: str,
    split_file: str,
    mask_file: str,
    support_dates_hash: str,
    protocol_freeze_id: str,
    n_valid: int,
    mask_frac: float,
    variable: str,
    metric_name: str,
    value: float,
) -> Dict[str, Any]:
    """Build a single evaluation metric row with all protocol metadata."""
    return {
        "experiment_id": experiment_id,
        "run_id": f"{experiment_id}_{method}_{sample.get('time_index', sample_idx)}",
        "method": method,
        "query_date": sample.get("date_str", ""),
        "query_time_index": int(sample.get("time_index", -1)),
        "month": sample.get("month", None),
        "season": sample.get("season", ""),
        "support_dates_hash": sample.get("support_dates_hash", support_dates_hash),
        "split_file": split_file,
        "mask_file": mask_file,
        "country_id": sample.get("country_id", ""),
        "target_region_id": sample.get("target_region_id", ""),
        "sample_region_id": sample.get("sample_region_id", ""),
        "active_region_ids": "|".join(sample.get("active_region_ids", [])),
        "split_role": split_role,
        "K": int(sample.get("K", -1)),
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
    support_dates_hash: str = "",
    deadzone_epsilon: float = 0.005,
    high_update_top_fraction: float = 0.2,
    preloaded: bool = True,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Evaluate a predictor over a dataset.

    ``predictor`` must implement ``predict(sample) -> dict`` and return both
    pred_increment_* and pred_analysis_* for surface/rootzone.

    Args:
        max_samples: If set, only evaluate the first max_samples samples.
    """
    all_samples = dataset.preload() if preloaded and hasattr(dataset, "preload") else None
    rows: List[Dict[str, Any]] = []

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
    for idx in tqdm(range(n_eval), total=n_eval, desc=f"Evaluating {split_role}", unit="samples"):
        sample = all_samples[idx] if all_samples is not None else dataset[idx]
        if first_sample is None:
            first_sample = sample
        pred = predictor.predict(sample)
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
                    support_dates_hash=support_dates_hash, protocol_freeze_id=protocol_freeze_id,
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
                    support_dates_hash=support_dates_hash, protocol_freeze_id=protocol_freeze_id,
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
                "support_dates_hash": support_dates_hash,
                "split_file": split_file,
                "mask_file": mask_file,
                "country_id": "",
                "target_region_id": "",
                "sample_region_id": "",
                "active_region_ids": "",
                "split_role": split_role,
                "K": -1,
                "seed": -1,
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
                "support_dates_hash": support_dates_hash,
                "split_file": split_file,
                "mask_file": mask_file,
                "country_id": "",
                "target_region_id": "",
                "sample_region_id": "",
                "active_region_ids": "",
                "split_role": split_role,
                "K": -1,
                "seed": -1,
                "variable": var_name,
                "metric": "analysis_skill_vs_forecast_latw_global",
                "value": global_skill_latw,
                "n_valid_pixels": total_n_pixels,
                "n_time_steps": 1,
                "mask_fraction": float("nan"),
                "protocol_freeze_id": protocol_freeze_id,
            })

    return rows


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
