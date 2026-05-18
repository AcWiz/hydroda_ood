"""Source-fit increment statistics for per-channel loss scale normalization.

No-leakage declaration:
    - Only source_fit dates (2015-2020) are used
    - Only loss_mask valid pixels are used
    - target_context / target_query labels are strictly excluded
    - Stats are saved as JSON artifacts (artifacts/stats/)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def compute_source_fit_increment_stats(
    dataset: Any,
    max_samples: int = 200,
) -> Dict[str, Any]:
    """Compute per-channel increment mean/std from source_fit dates only.

    Args:
        dataset: HydroDADataset with split_type="source_fit".
        max_samples: max number of samples to use (spread across dataset).

    Returns:
        dict with surface_mean, surface_std, rootzone_mean, rootzone_std,
        n_pixels, n_samples, protocol_freeze_id_placeholder

    Raises:
        ValueError: if dataset.split_type is not source_fit.
    """
    split_type = getattr(dataset, "split_type", None)
    if split_type not in ("source_train", "source_fit"):
        raise ValueError(
            f"increment_stats: split_type must be source_fit, got {split_type!r}. "
            f"Target query labels must NOT be used for statistic computation."
        )

    n_total = len(dataset)
    n_samples = min(max_samples, n_total)
    step = max(1, n_total // n_samples)
    indices = list(range(0, n_total, step))[:n_samples]

    inc_s_values = []
    inc_r_values = []

    for idx in indices:
        sample = dataset[idx]
        mask = sample["loss_mask"] > 0.5

        inc_s = sample["increment_surface"]
        inc_r = sample["increment_rootzone"]

        valid_s = mask & np.isfinite(inc_s)
        valid_r = mask & np.isfinite(inc_r)

        if valid_s.sum() > 0:
            inc_s_values.append(inc_s[valid_s].reshape(-1).astype(np.float64))
        if valid_r.sum() > 0:
            inc_r_values.append(inc_r[valid_r].reshape(-1).astype(np.float64))

    if not inc_s_values:
        return {
            "surface_mean": 0.0,
            "surface_std": 1.0,
            "rootzone_mean": 0.0,
            "rootzone_std": 1.0,
            "n_pixels": 0,
            "n_samples": 0,
            "warning": "no_valid_pixels",
        }

    inc_s_all = np.concatenate(inc_s_values)
    inc_r_all = np.concatenate(inc_r_values)

    surface_mean = float(inc_s_all.mean())
    surface_std = float(inc_s_all.std()) if inc_s_all.size > 1 else 1.0
    rootzone_mean = float(inc_r_all.mean())
    rootzone_std = float(inc_r_all.std()) if inc_r_all.size > 1 else 1.0

    # Guard against zero std
    surface_std = max(surface_std, 1e-6)
    rootzone_std = max(rootzone_std, 1e-6)

    return {
        "surface_mean": surface_mean,
        "surface_std": surface_std,
        "rootzone_mean": rootzone_mean,
        "rootzone_std": rootzone_std,
        "n_pixels_surface": int(inc_s_all.size),
        "n_pixels_rootzone": int(inc_r_all.size),
        "n_samples": len(indices),
    }


def save_source_fit_increment_stats(stats: Dict[str, Any], path: str | Path) -> None:
    """Save increment stats to JSON artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_source_fit_increment_stats(path: str | Path) -> Dict[str, Any]:
    """Load increment stats from JSON artifact."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
