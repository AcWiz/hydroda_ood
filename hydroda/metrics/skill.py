"""Stateless metric functions — all use mask > 0.5 before computing.

No-leakage declaration:
    - No target_query labels used in normalization
    - mask applied before all computations
    - Returns np.nan when no valid pixels after masking
"""

from __future__ import annotations

import numpy as np


def _apply_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply metric mask: mask > 0.5, then flatten."""
    m = mask > 0.5
    return arr[m]


def analysis_rmse(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
) -> float:
    """RMSE in analysis space: rmse(pred, true)."""
    p = _apply_mask(pred, mask)
    t = _apply_mask(true, mask)
    if p.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((p - t) ** 2)))


def analysis_mae(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
) -> float:
    """MAE in analysis space: mae(pred, true)."""
    p = _apply_mask(pred, mask)
    t = _apply_mask(true, mask)
    if p.size == 0:
        return np.nan
    return float(np.mean(np.abs(p - t)))


def analysis_skill_vs_forecast(
    pred: np.ndarray,
    true: np.ndarray,
    forecast: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Analysis skill vs forecast: 1 - RMSE(pred,true) / RMSE(forecast,true).

    For forecast-only baseline (pred == forecast): skill = 0.
    """
    p = _apply_mask(pred, mask)
    t = _apply_mask(true, mask)
    f = _apply_mask(forecast, mask)
    if p.size == 0:
        return np.nan
    rmse_pred = np.sqrt(np.mean((p - t) ** 2))
    rmse_fcst = np.sqrt(np.mean((f - t) ** 2))
    if rmse_fcst == 0:
        return np.nan
    return float(1.0 - rmse_pred / rmse_fcst)


def increment_rmse(
    pred_inc: np.ndarray,
    true_inc: np.ndarray,
    mask: np.ndarray,
) -> float:
    """RMSE in increment space: rmse(pred_inc, true_inc)."""
    p = _apply_mask(pred_inc, mask)
    t = _apply_mask(true_inc, mask)
    if p.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((p - t) ** 2)))


def increment_mae(
    pred_inc: np.ndarray,
    true_inc: np.ndarray,
    mask: np.ndarray,
) -> float:
    """MAE in increment space: mae(pred_inc, true_inc)."""
    p = _apply_mask(pred_inc, mask)
    t = _apply_mask(true_inc, mask)
    if p.size == 0:
        return np.nan
    return float(np.mean(np.abs(p - t)))


def increment_bias(
    pred_inc: np.ndarray,
    true_inc: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Increment bias: mean(pred_inc - true_inc)."""
    p = _apply_mask(pred_inc, mask)
    t = _apply_mask(true_inc, mask)
    if p.size == 0:
        return np.nan
    return float(np.mean(p - t))


def increment_corr(
    pred_inc: np.ndarray,
    true_inc: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Pearson correlation of increments. Returns np.nan if variance is zero."""
    p = _apply_mask(pred_inc, mask)
    t = _apply_mask(true_inc, mask)
    if p.size == 0:
        return np.nan
    # Safe: if either variance is zero, return nan instead of raising
    if np.std(p) == 0 or np.std(t) == 0:
        return np.nan
    return float(np.corrcoef(p.flatten(), t.flatten())[0, 1])


def sign_accuracy_deadzone(
    pred_inc: np.ndarray,
    true_inc: np.ndarray,
    mask: np.ndarray,
    epsilon: float = 0.005,
) -> float:
    """Sign accuracy with deadzone: exclude pixels where |true_inc| < epsilon.

    Sign is correct when pred*true > 0.
    """
    m = mask > 0.5
    p = pred_inc[m]
    t = true_inc[m]
    # Exclude deadzone
    alive = np.abs(t) >= epsilon
    if alive.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(p[alive]) == np.sign(t[alive])))


def valid_pixel_count(
    mask: np.ndarray,
) -> int:
    """Count of valid pixels: mask > 0.5."""
    return int((mask > 0.5).sum())


def effective_mask_fraction(
    mask: np.ndarray,
    total: int,
) -> float:
    """Fraction of total pixels that are valid after masking."""
    if total == 0:
        return np.nan
    return float((mask > 0.5).sum()) / float(total)
