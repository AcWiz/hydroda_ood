"""Monthly mean increment baseline — grouped by calendar month.

No-leakage declaration:
    - Fit on target_support only
    - Uses calendar month (not target_query labels)
    - 12 month buckets, each with a mean increment
    - No target_query labels used
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional


class TargetMonthlySupportIncrementBaseline:
    """Monthly grouped mean increment baseline.

    Computes mean increment separately for each calendar month (1-12)
    using target_support data. Predicts using the monthly mean for
    each target_query sample's month.

    Only applicable for K >= 12 (K=4 has insufficient support dates per month).
    """

    def __init__(self):
        self._monthly_mean_surface: Optional[Dict[int, float]] = None
        self._monthly_mean_rootzone: Optional[Dict[int, float]] = None
        self._fitted = False

    def fit(self, samples: list) -> "TargetMonthlySupportIncrementBaseline":
        """Compute per-month mean increment from target_support samples.

        Args:
            samples: List of sample dicts from target_support split.
                Each must contain: increment_surface, increment_rootzone, metric_mask, month.
        """
        # Accumulate per-month: 12 buckets
        month_accum_s: Dict[int, list] = {m: [] for m in range(1, 13)}
        month_accum_r: Dict[int, list] = {m: [] for m in range(1, 13)}

        for s in samples:
            m = s["metric_mask"]
            month = s.get("month")
            if month is None:
                continue
            month = int(month)

            valid = (
                (m > 0.5)
                & np.isfinite(s["increment_surface"])
                & np.isfinite(s["increment_rootzone"])
            )
            inc_s_flat = s["increment_surface"][valid].astype(np.float64)
            inc_r_flat = s["increment_rootzone"][valid].astype(np.float64)

            if inc_s_flat.size > 0:
                month_accum_s[month].append(inc_s_flat)
            if inc_r_flat.size > 0:
                month_accum_r[month].append(inc_r_flat)

        # Compute per-month mean
        self._monthly_mean_surface = {}
        self._monthly_mean_rootzone = {}
        for month in range(1, 13):
            if month_accum_s[month]:
                all_s = np.concatenate(month_accum_s[month])
                self._monthly_mean_surface[month] = float(np.nanmean(all_s))
            else:
                self._monthly_mean_surface[month] = 0.0
            if month_accum_r[month]:
                all_r = np.concatenate(month_accum_r[month])
                self._monthly_mean_rootzone[month] = float(np.nanmean(all_r))
            else:
                self._monthly_mean_rootzone[month] = 0.0

        self._fitted = True
        return self

    def predict(self, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Predict using monthly mean increment.

        Args:
            sample: Dict with forecast_surface, forecast_rootzone, month keys.

        Returns:
            dict with keys:
                pred_increment_surface: monthly mean tiled to shape
                pred_increment_rootzone: monthly mean tiled to shape
                pred_analysis_surface: forecast_surface + pred_increment_surface
                pred_analysis_rootzone: forecast_rootzone + pred_increment_rootzone
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        forecast_surface = sample["forecast_surface"]
        forecast_rootzone = sample["forecast_rootzone"]
        month = int(sample.get("month", 1))
        H, W = forecast_surface.shape

        mean_s = self._monthly_mean_surface.get(month, 0.0)
        mean_r = self._monthly_mean_rootzone.get(month, 0.0)

        pred_inc_s = np.full((H, W), mean_s, dtype=np.float32)
        pred_inc_r = np.full((H, W), mean_r, dtype=np.float32)

        return {
            "pred_increment_surface": pred_inc_s,
            "pred_increment_rootzone": pred_inc_r,
            "pred_analysis_surface": (forecast_surface + pred_inc_s).astype(np.float32),
            "pred_analysis_rootzone": (forecast_rootzone + pred_inc_r).astype(np.float32),
        }
