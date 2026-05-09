"""Evaluation harness — iterates dataset samples and computes metrics.

No-leakage declaration:
    - No model training
    - metric_mask = base_valid_mask + active_region_mask + finite check only
    - target_query only used for evaluation, not optimization
    - No frozen artifact modification
"""

from __future__ import annotations

from typing import List, Dict, Any, Iterable

import numpy as np

from hydroda.metrics.skill import (
    analysis_rmse,
    analysis_mae,
    analysis_skill_vs_forecast,
    increment_rmse,
    increment_mae,
    increment_bias,
    increment_corr,
    sign_accuracy_deadzone,
    valid_pixel_count,
    effective_mask_fraction,
)


_VARIABLE_PAIRS = [
    ("surface", "forecast_surface", "analysis_surface", "increment_surface", "pred_increment_surface", "pred_analysis_surface"),
    ("rootzone", "forecast_rootzone", "analysis_rootzone", "increment_rootzone", "pred_increment_rootzone", "pred_analysis_rootzone"),
]

_METRIC_FUNCS = [
    ("analysis_rmse", lambda p, t, f, m: analysis_rmse(p, t, m)),
    ("analysis_mae", lambda p, t, f, m: analysis_mae(p, t, m)),
    ("analysis_skill_vs_forecast", lambda p, t, f, m: analysis_skill_vs_forecast(p, t, f, m)),
    ("increment_rmse", lambda p, t, f, m: increment_rmse(p, t, m)),
    ("increment_mae", lambda p, t, f, m: increment_mae(p, t, m)),
    ("increment_bias", lambda p, t, f, m: increment_bias(p, t, m)),
    ("increment_corr", lambda p, t, f, m: increment_corr(p, t, m)),
    ("sign_accuracy_deadzone", lambda p, t, f, m: sign_accuracy_deadzone(p, t, m)),
]


def evaluate_split(
    dataset: Any,
    predictor: Any,
    split_role: str,
    experiment_id: str,
    protocol_freeze_id: str,
    preloaded: bool = True,
) -> List[Dict]:
    """Evaluate a predictor over all samples in a dataset.

    Args:
        dataset: HydroDADataset instance
        predictor: object with predict(sample) -> dict
        split_role: "source_train" | "target_support" | "target_query"
        experiment_id: experiment identifier string
        protocol_freeze_id: freeze manifest freeze_id
        preloaded: if True (default), pre-load entire dataset into memory for speed.
            For large target_query datasets (~4600 samples), this avoids per-sample
            xarray isel overhead and reduces evaluation from ~120s to ~5s.

    Returns:
        List of result dicts, one per (sample, variable, metric).
    """
    # Pre-load all samples for fast evaluation
    if preloaded:
        all_samples = dataset.preload()
    else:
        all_samples = None

    results: List[Dict] = []

    for idx in range(len(dataset)):
        sample = all_samples[idx] if preloaded else dataset[idx]

        pred = predictor.predict(sample)

        metric_mask = sample["metric_mask"]
        n_valid = valid_pixel_count(metric_mask)
        n_time_steps = 1
        H, W = metric_mask.shape
        total_pixels = H * W
        mask_frac = effective_mask_fraction(metric_mask, total_pixels)

        for var_name, fcst_key, analy_key, incr_key, pred_incr_key, pred_analy_key in _VARIABLE_PAIRS:
            forecast = sample[fcst_key]
            true_analysis = sample[analy_key]
            true_increment = sample[incr_key]
            pred_increment = pred[pred_incr_key]
            pred_analysis = pred[pred_analy_key]

            for metric_name, metric_fn in _METRIC_FUNCS:
                value = metric_fn(
                    pred_analysis,
                    true_analysis,
                    forecast,
                    metric_mask,
                )

                results.append({
                    "experiment_id": experiment_id,
                    "run_id": f"{experiment_id}_{sample['time_index']}",
                    "query_date": sample.get("date_str", ""),
                    "query_time_index": int(sample.get("time_index", -1)),
                    "support_dates_hash": sample.get("support_dates_hash", ""),
                    "split_file": SPLITS_JSON,
                    "mask_file": "artifacts/regions/US_region_masks.nc",
                    "method": "forecast",  # overridden by caller when method differs
                    "country_id": sample["country_id"],
                    "target_region_id": sample["target_region_id"],
                    "active_region_ids": "|".join(sample["active_region_ids"]),
                    "split_role": split_role,
                    "K": sample["K"],
                    "seed": sample["seed"],
                    "variable": var_name,
                    "metric": metric_name,
                    "value": value,
                    "n_valid_pixels": n_valid,
                    "n_time_steps": n_time_steps,
                    "mask_fraction": mask_frac,
                    "protocol_freeze_id": protocol_freeze_id,
                })

    return results
