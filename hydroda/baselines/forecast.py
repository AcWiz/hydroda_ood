"""Forecast-only baseline for DA increment emulation.

This is the natural zero-correction baseline:
``pred_increment = 0`` and ``pred_analysis = forecast``.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


class ForecastBaseline:
    method_name = "forecast_only"

    def predict(self, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
        forecast_surface = np.asarray(sample["forecast_surface"], dtype=np.float32)
        forecast_rootzone = np.asarray(sample["forecast_rootzone"], dtype=np.float32)
        return {
            "pred_increment_surface": np.zeros_like(forecast_surface, dtype=np.float32),
            "pred_increment_rootzone": np.zeros_like(forecast_rootzone, dtype=np.float32),
            "pred_analysis_surface": forecast_surface.copy(),
            "pred_analysis_rootzone": forecast_rootzone.copy(),
        }
