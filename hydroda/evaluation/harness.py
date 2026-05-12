"""Evaluation harness for HydroDA-OOD / HyperDA V4."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from hydroda.metrics.skill import (
    compute_variable_metrics,
    effective_mask_fraction,
    valid_pixel_count,
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

    n_eval = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for idx in range(n_eval):
        sample = all_samples[idx] if all_samples is not None else dataset[idx]
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
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "run_id": f"{experiment_id}_{method}_{sample.get('time_index', idx)}",
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
                )

    return rows
