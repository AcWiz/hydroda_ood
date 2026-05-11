"""Metric functions for HydroDA-OOD / HyperDA V4.

All functions are stateless and operate on already materialized NumPy arrays.
No normalization, threshold calibration, support selection, or model selection is
performed here. Target query labels may be used only as evaluation labels after
prediction, never to modify a model or protocol artifact.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _valid_flat(*arrays: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Return flattened arrays over finite valid pixels only."""
    valid = mask > 0.5
    for arr in arrays:
        valid &= np.isfinite(arr)
    if valid.sum() == 0:
        return tuple(np.asarray([], dtype=np.float64) for _ in arrays)
    return tuple(np.asarray(arr[valid], dtype=np.float64).reshape(-1) for arr in arrays)


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def analysis_rmse(pred_analysis: np.ndarray, true_analysis: np.ndarray, mask: np.ndarray) -> float:
    p, t = _valid_flat(pred_analysis, true_analysis, mask=mask)
    return _rmse(p, t)


def analysis_mae(pred_analysis: np.ndarray, true_analysis: np.ndarray, mask: np.ndarray) -> float:
    p, t = _valid_flat(pred_analysis, true_analysis, mask=mask)
    if p.size == 0:
        return np.nan
    return float(np.mean(np.abs(p - t)))


def analysis_skill_vs_forecast(
    pred_analysis: np.ndarray,
    true_analysis: np.ndarray,
    forecast: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Forecast-relative skill in analysis space.

    Forecast-only should return exactly 0 up to floating-point noise when
    pred_analysis == forecast.
    """
    p, t, f = _valid_flat(pred_analysis, true_analysis, forecast, mask=mask)
    if p.size == 0:
        return np.nan
    rmse_pred = _rmse(p, t)
    rmse_fcst = _rmse(f, t)
    if not np.isfinite(rmse_fcst) or rmse_fcst <= 0:
        return np.nan
    return float(1.0 - rmse_pred / rmse_fcst)


def increment_rmse(pred_increment: np.ndarray, true_increment: np.ndarray, mask: np.ndarray) -> float:
    p, t = _valid_flat(pred_increment, true_increment, mask=mask)
    return _rmse(p, t)


def increment_mae(pred_increment: np.ndarray, true_increment: np.ndarray, mask: np.ndarray) -> float:
    p, t = _valid_flat(pred_increment, true_increment, mask=mask)
    if p.size == 0:
        return np.nan
    return float(np.mean(np.abs(p - t)))


def increment_bias(pred_increment: np.ndarray, true_increment: np.ndarray, mask: np.ndarray) -> float:
    p, t = _valid_flat(pred_increment, true_increment, mask=mask)
    if p.size == 0:
        return np.nan
    return float(np.mean(p - t))


def increment_corr(pred_increment: np.ndarray, true_increment: np.ndarray, mask: np.ndarray) -> float:
    p, t = _valid_flat(pred_increment, true_increment, mask=mask)
    if p.size < 2:
        return np.nan
    if np.std(p) <= 0 or np.std(t) <= 0:
        return np.nan
    return float(np.corrcoef(p, t)[0, 1])


def sign_accuracy_deadzone(
    pred_increment: np.ndarray,
    true_increment: np.ndarray,
    mask: np.ndarray,
    epsilon: float,
) -> float:
    """Sign accuracy excluding small true increments.

    ``epsilon`` must be fixed from source-train statistics or a config. Do not
    tune it on target query labels.
    """
    p, t = _valid_flat(pred_increment, true_increment, mask=mask)
    if p.size == 0:
        return np.nan
    keep = np.abs(t) >= float(epsilon)
    if keep.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(p[keep]) == np.sign(t[keep])))


def high_update_increment_skill(
    pred_increment: np.ndarray,
    true_increment: np.ndarray,
    mask: np.ndarray,
    top_fraction: float = 0.2,
) -> float:
    """RMSE skill on top-|increment| evaluation pixels.

    This is evaluation-only. It may use target query labels after prediction to
    define the evaluation subset, but must never be called during training,
    support selection, early stopping, or model selection.
    """
    p, t = _valid_flat(pred_increment, true_increment, mask=mask)
    if p.size == 0:
        return np.nan
    top_fraction = float(top_fraction)
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError("top_fraction must be in (0, 1].")
    q = np.quantile(np.abs(t), 1.0 - top_fraction)
    keep = np.abs(t) >= q
    if keep.sum() == 0:
        return np.nan
    rmse_pred = _rmse(p[keep], t[keep])
    rmse_zero = _rmse(np.zeros_like(t[keep]), t[keep])
    if rmse_zero <= 0:
        return np.nan
    return float(1.0 - rmse_pred / rmse_zero)


def valid_pixel_count(mask: np.ndarray) -> int:
    return int(np.sum(mask > 0.5))


def effective_mask_fraction(mask: np.ndarray, total: int | None = None) -> float:
    if total is None:
        total = int(mask.size)
    if total <= 0:
        return np.nan
    return float(valid_pixel_count(mask)) / float(total)


def compute_variable_metrics(
    *,
    pred_analysis: np.ndarray,
    true_analysis: np.ndarray,
    forecast: np.ndarray,
    pred_increment: np.ndarray,
    true_increment: np.ndarray,
    mask: np.ndarray,
    deadzone_epsilon: float = 0.005,
    high_update_top_fraction: float = 0.2,
) -> Dict[str, float]:
    """Compute all V4 metrics for one variable.

    This helper prevents the common bug of routing ``pred_analysis`` into
    increment-space metrics.
    """
    return {
        "analysis_rmse": analysis_rmse(pred_analysis, true_analysis, mask),
        "analysis_mae": analysis_mae(pred_analysis, true_analysis, mask),
        "analysis_skill_vs_forecast": analysis_skill_vs_forecast(
            pred_analysis, true_analysis, forecast, mask
        ),
        "increment_rmse": increment_rmse(pred_increment, true_increment, mask),
        "increment_mae": increment_mae(pred_increment, true_increment, mask),
        "increment_bias": increment_bias(pred_increment, true_increment, mask),
        "increment_corr": increment_corr(pred_increment, true_increment, mask),
        "sign_accuracy_deadzone": sign_accuracy_deadzone(
            pred_increment, true_increment, mask, deadzone_epsilon
        ),
        "high_update_increment_skill": high_update_increment_skill(
            pred_increment, true_increment, mask, high_update_top_fraction
        ),
    }
