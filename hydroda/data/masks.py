"""Mask derivation utilities for HydroDA-OOD / HyperDA V4.

No-leakage declaration:
    - All mask functions are stateless, no training, no target label access
    - metric_mask construction uses only region_mask & label_valid_mask
    - channel 11 (obs_mask) NOT used in metric_mask — only stored as diagnostic field
    - label_valid_mask only checks field finiteness, not external masks
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def derive_label_valid_mask(sample: Dict[str, Any]) -> np.ndarray:
    """label_valid_mask: all 6 SM fields must be finite.

    Used as primary metric_mask component. Does NOT use channel 11.

    Args:
        sample: Dict with forecast_surface, forecast_rootzone, analysis_surface,
                analysis_rootzone, increment_surface, increment_rootzone

    Returns:
        Binary mask (float32) where 1 = all 6 fields are finite at that pixel
    """
    required = [
        "forecast_surface",
        "forecast_rootzone",
        "analysis_surface",
        "analysis_rootzone",
        "increment_surface",
        "increment_rootzone",
    ]
    for key in required:
        if key not in sample:
            raise KeyError(f"derive_label_valid_mask requires {key} in sample")

    finite_mask = np.ones_like(sample["forecast_surface"], dtype=bool)
    for key in required:
        finite_mask = np.logical_and(finite_mask, np.isfinite(sample[key]))

    return finite_mask.astype(np.float32)


def derive_obs_mask(sample: Dict[str, Any]) -> np.ndarray:
    """obs_mask: channel 11 observation availability mask (diagnostic only).

    NOT used in metric_mask computation. SMAP retrieval coverage varies
    spatio-temporally and may have low overlap with some US regions.

    Args:
        sample: Dict with base_valid_mask (channel 11)

    Returns:
        Binary mask (float32) where 1 = SMAP observation available at that pixel
    """
    if "base_valid_mask" not in sample:
        raise KeyError("derive_obs_mask requires base_valid_mask in sample")
    return np.where(sample["base_valid_mask"] > 0.5, np.float32(1.0), np.float32(0.0))


def derive_region_mask(sample: Dict[str, Any]) -> np.ndarray:
    """region_mask: active region pixels (0/1 binary).

    Args:
        sample: Dict with active_region_mask

    Returns:
        Binary mask (float32) where 1 = pixel belongs to active region
    """
    if "active_region_mask" not in sample:
        raise KeyError("derive_region_mask requires active_region_mask in sample")
    return np.where(sample["active_region_mask"] > 0.5, np.float32(1.0), np.float32(0.0))


def derive_metric_mask(sample: Dict[str, Any]) -> np.ndarray:
    """metric_mask = region_mask & label_valid_mask.

    Channel 11 (obs_mask) NOT included. SMAP coverage gaps do NOT block evaluation.

    Args:
        sample: Dict with active_region_mask, forecast_surface, forecast_rootzone,
                analysis_surface, analysis_rootzone, increment_surface, increment_rootzone

    Returns:
        Binary mask (float32) where 1 = pixel is in active region AND all SM fields finite
    """
    region_mask = derive_region_mask(sample)
    label_valid_mask = derive_label_valid_mask(sample)
    return np.logical_and(region_mask, label_valid_mask).astype(np.float32)


def compute_mask_coverage(
    label_valid_mask: np.ndarray,
    obs_mask: np.ndarray,
    region_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute coverage statistics for a single sample's masks.

    Args:
        label_valid_mask: Binary mask of label-valid pixels
        obs_mask: Binary mask of SMAP-observed pixels
        region_mask: Binary mask of region pixels

    Returns:
        Dict with coverage fractions for label_valid, obs, and region
    """
    total_px = region_mask.size
    if total_px == 0:
        return {"label_valid_fraction": 0.0, "obs_fraction": 0.0, "region_fraction": 0.0}

    region_px = int(region_mask.sum())
    if region_px == 0:
        return {"label_valid_fraction": 0.0, "obs_fraction": 0.0, "region_fraction": 0.0}

    return {
        "label_valid_fraction": float(label_valid_mask.sum()) / float(region_px),
        "obs_fraction": float(obs_mask.sum()) / float(region_px),
        "region_fraction": float(region_px) / float(total_px),
    }


def summarize_mask_coverage(
    samples: List[Dict[str, Any]],
    by_region: bool = True,
    by_season: bool = True,
) -> Dict[str, Any]:
    """Audit mask coverage across dataset samples, separating obs vs label validity.

    Args:
        samples: List of dataset sample dicts with mask fields
        by_region: If True, also compute per target_region_id breakdown
        by_season: If True, also compute per season breakdown

    Returns:
        Dict with overall stats, and optionally per-region and per-season breakdowns
    """
    region_stats: Dict[str, Dict[str, List[float]]] = {}
    season_stats: Dict[str, Dict[str, List[float]]] = {}
    overall_label_valid = []
    overall_obs = []

    for sample in samples:
        label_valid = derive_label_valid_mask(sample)
        obs = derive_obs_mask(sample)
        region = derive_region_mask(sample)

        cov = compute_mask_coverage(label_valid, obs, region)
        overall_label_valid.append(cov["label_valid_fraction"])
        overall_obs.append(cov["obs_fraction"])

        region_id = sample.get("target_region_id", "unknown")
        season = sample.get("season", "unknown")

        if by_region:
            if region_id not in region_stats:
                region_stats[region_id] = {"label_valid": [], "obs": []}
            region_stats[region_id]["label_valid"].append(cov["label_valid_fraction"])
            region_stats[region_id]["obs"].append(cov["obs_fraction"])

        if by_season:
            if season not in season_stats:
                season_stats[season] = {"label_valid": [], "obs": []}
            season_stats[season]["label_valid"].append(cov["label_valid_fraction"])
            season_stats[season]["obs"].append(cov["obs_fraction"])

    result: Dict[str, Any] = {
        "overall": {
            "label_valid_mean": float(np.mean(overall_label_valid)) if overall_label_valid else 0.0,
            "label_valid_std": float(np.std(overall_label_valid)) if overall_label_valid else 0.0,
            "obs_mean": float(np.mean(overall_obs)) if overall_obs else 0.0,
            "obs_std": float(np.std(overall_obs)) if overall_obs else 0.0,
            "n_samples": len(samples),
        }
    }

    if by_region:
        result["by_region"] = {
            rid: {
                "label_valid_mean": float(np.mean(vals["label_valid"])) if vals["label_valid"] else 0.0,
                "label_valid_std": float(np.std(vals["label_valid"])) if vals["label_valid"] else 0.0,
                "obs_mean": float(np.mean(vals["obs"])) if vals["obs"] else 0.0,
                "obs_std": float(np.std(vals["obs"])) if vals["obs"] else 0.0,
                "n_samples": len(vals["label_valid"]),
            }
            for rid, vals in region_stats.items()
        }

    if by_season:
        result["by_season"] = {
            season: {
                "label_valid_mean": float(np.mean(vals["label_valid"])) if vals["label_valid"] else 0.0,
                "label_valid_std": float(np.std(vals["label_valid"])) if vals["label_valid"] else 0.0,
                "obs_mean": float(np.mean(vals["obs"])) if vals["obs"] else 0.0,
                "obs_std": float(np.std(vals["obs"])) if vals["obs"] else 0.0,
                "n_samples": len(vals["label_valid"]),
            }
            for season, vals in season_stats.items()
        }

    return result
