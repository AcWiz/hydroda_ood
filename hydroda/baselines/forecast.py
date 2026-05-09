"""Forecast-only baseline — stateless, zero increment.

No-leakage declaration:
    - No fitting, no training, no target labels read
    - pred_increment_surface and pred_increment_rootzone are strictly all zeros
    - pred_analysis_surface == forecast_surface exactly
    - pred_analysis_rootzone == forecast_rootzone exactly
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any


class ForecastBaseline:
    """Stateless forecast-only baseline.

    Predicts zero increment everywhere:
        pred_increment = 0
        pred_analysis = forecast

    This is the sanity baseline — skill should be exactly 0.
    """

    def predict(self, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Return prediction dict for a single sample.

        Args:
            sample: Dict with forecast_surface, forecast_rootzone keys.
                Shape: (H, W) each.

        Returns:
            dict with keys:
                pred_increment_surface: zeros_like(forecast_surface)
                pred_increment_rootzone: zeros_like(forecast_rootzone)
                pred_analysis_surface: forecast_surface.copy()
                pred_analysis_rootzone: forecast_rootzone.copy()
        """
        forecast_surface = sample["forecast_surface"]
        forecast_rootzone = sample["forecast_rootzone"]

        return {
            "pred_increment_surface": np.zeros_like(forecast_surface),
            "pred_increment_rootzone": np.zeros_like(forecast_rootzone),
            "pred_analysis_surface": forecast_surface.copy(),
            "pred_analysis_rootzone": forecast_rootzone.copy(),
        }
