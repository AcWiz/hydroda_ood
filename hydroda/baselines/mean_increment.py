"""Mean increment baselines — source_mean and target_train_mean.

No-leakage declaration:
    - SourceMeanIncrementBaseline: fit on source_train only, no target labels
    - TargetSupportMeanIncrementBaseline: legacy name; fit on target_train only
    - Both use metric_mask for valid pixel selection before computing mean
    - No target_eval/query labels used in any computation
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional

from hydroda.data.leakage_guard import LeakageGuard
from hydroda.data.protocol import ProtocolConfig


def _check_target_train_samples(samples: list, purpose: str) -> None:
    dates = [s.get("date_str", "") for s in samples if s.get("date_str")]
    if dates:
        LeakageGuard(ProtocolConfig()).check_target_adaptation_scope(
            dates,
            purpose=purpose,
            labels_allowed=True,
        )


class SourceMeanIncrementBaseline:
    """Mean increment baseline fit on source_train data.

    Computes global mean increment from source_train region+dates.
    Predicts this same increment for target_eval/query samples.

    Prediction: pred_increment = mean_increment (fitted from source_train)

    No target labels used.
    """

    def __init__(self):
        self._mean_inc_surface: Optional[np.ndarray] = None
        self._mean_inc_rootzone: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, samples: list) -> "SourceMeanIncrementBaseline":
        """Compute mean increment from source_train samples.

        Args:
            samples: List of sample dicts from source_train split.
                Each must contain: increment_surface, increment_rootzone, metric_mask.
        """
        inc_s_list = []
        inc_r_list = []
        mask_list = []

        for s in samples:
            m = s["metric_mask"]
            # Only use finite, valid pixels
            valid = (m > 0.5) & np.isfinite(s["increment_surface"]) & np.isfinite(s["increment_rootzone"])
            inc_s_list.append(s["increment_surface"][valid])
            inc_r_list.append(s["increment_rootzone"][valid])
            mask_list.append(valid)

        if not inc_s_list:
            raise ValueError("No valid pixels found in source_train samples")

        all_inc_s = np.concatenate(inc_s_list)
        all_inc_r = np.concatenate(inc_r_list)

        if all_inc_s.size == 0 or all_inc_r.size == 0:
            raise ValueError("No valid pixels found in source_train samples")

        # Global scalar mean increment
        self._mean_inc_surface = float(np.mean(all_inc_s))
        self._mean_inc_rootzone = float(np.mean(all_inc_r))
        self._fitted = True
        return self

    def predict(self, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Predict using global mean increment from source_train.

        Args:
            sample: Dict with forecast_surface, forecast_rootzone, increment_surface,
                increment_rootzone keys.

        Returns:
            dict with keys:
                pred_increment_surface: scalar mean tiled to shape
                pred_increment_rootzone: scalar mean tiled to shape
                pred_analysis_surface: forecast_surface + pred_increment_surface
                pred_analysis_rootzone: forecast_rootzone + pred_increment_rootzone
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        forecast_surface = sample["forecast_surface"]
        forecast_rootzone = sample["forecast_rootzone"]
        H, W = forecast_surface.shape

        pred_inc_s = np.full((H, W), self._mean_inc_surface, dtype=np.float32)
        pred_inc_r = np.full((H, W), self._mean_inc_rootzone, dtype=np.float32)

        return {
            "pred_increment_surface": pred_inc_s,
            "pred_increment_rootzone": pred_inc_r,
            "pred_analysis_surface": (forecast_surface + pred_inc_s).astype(np.float32),
            "pred_analysis_rootzone": (forecast_rootzone + pred_inc_r).astype(np.float32),
        }


class TargetSupportMeanIncrementBaseline:
    """Mean increment baseline fit on target_train data.

    This class name is retained for legacy scripts. Under the main protocol it
    computes the mean increment from the full 2015-2021 target_train/adaptation
    period, not a K-shot support set.

    Prediction: pred_increment = mean_increment (fitted from target_train)

    Uses target train labels (for fit), but never sees target_eval/query labels.
    """

    def __init__(self):
        self._mean_inc_surface: Optional[float] = None
        self._mean_inc_rootzone: Optional[float] = None
        self._fitted = False

    def fit(self, samples: list) -> "TargetSupportMeanIncrementBaseline":
        """Compute mean increment from target_train samples.

        Args:
            samples: List of sample dicts from target_train split or legacy target_support split.
                Each must contain: increment_surface, increment_rootzone, metric_mask.
        """
        _check_target_train_samples(samples, "target_train_mean_increment_fit")
        inc_s_list = []
        inc_r_list = []

        for s in samples:
            m = s["metric_mask"]
            valid = (m > 0.5) & np.isfinite(s["increment_surface"]) & np.isfinite(s["increment_rootzone"])
            inc_s_list.append(s["increment_surface"][valid])
            inc_r_list.append(s["increment_rootzone"][valid])

        if not inc_s_list:
            raise ValueError("No valid pixels found in target_train samples")

        all_inc_s = np.concatenate(inc_s_list)
        all_inc_r = np.concatenate(inc_r_list)

        if all_inc_s.size == 0 or all_inc_r.size == 0:
            raise ValueError("No valid pixels found in target_train samples")

        self._mean_inc_surface = float(np.mean(all_inc_s))
        self._mean_inc_rootzone = float(np.mean(all_inc_r))
        self._fitted = True
        return self

    def predict(self, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Predict using global mean increment from target_train.

        Args:
            sample: Dict with forecast_surface, forecast_rootzone keys.

        Returns:
            dict with keys:
                pred_increment_surface: scalar mean tiled to shape
                pred_increment_rootzone: scalar mean tiled to shape
                pred_analysis_surface: forecast_surface + pred_increment_surface
                pred_analysis_rootzone: forecast_rootzone + pred_increment_rootzone
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        forecast_surface = sample["forecast_surface"]
        forecast_rootzone = sample["forecast_rootzone"]
        H, W = forecast_surface.shape

        pred_inc_s = np.full((H, W), self._mean_inc_surface, dtype=np.float32)
        pred_inc_r = np.full((H, W), self._mean_inc_rootzone, dtype=np.float32)

        return {
            "pred_increment_surface": pred_inc_s,
            "pred_increment_rootzone": pred_inc_r,
            "pred_analysis_surface": (forecast_surface + pred_inc_s).astype(np.float32),
            "pred_analysis_rootzone": (forecast_rootzone + pred_inc_r).astype(np.float32),
        }


TargetTrainMeanIncrementBaseline = TargetSupportMeanIncrementBaseline
