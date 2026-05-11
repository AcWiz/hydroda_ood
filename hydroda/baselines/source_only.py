"""Source-only backbone predictor for HydroDA-OOD / HyperDA V4.

No-leakage declaration:
    - Uses trained SmallResUNet checkpoint (source_train only, no target labels)
    - No target prompt used in prediction
    - prediction uses only input features and learned weights
    - metric computation uses target_query labels post-prediction only
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from hydroda.models.resunet import SmallResUNet


class SourceOnlyBackbonePredictor:
    """Neural predictor wrapping a trained SmallResUNet checkpoint.

    Loads checkpoint, sets to eval mode, and provides:
        predict(sample: dict) -> dict:
            - x = sample['x']  # [12, H, W] raw input
            - pred_increment = model(x.unsqueeze(0))  # [1, 2, H, W]
            - forecast_surface = sample['forecast_surface']
            - forecast_rootzone = sample['forecast_rootzone']
            - pred_analysis_surface = forecast_surface + pred_increment[0, 0]
            - pred_analysis_rootzone = forecast_rootzone + pred_increment[0, 1]

    Args:
        checkpoint_path: path to trained .pt checkpoint
        device: device string (default "cuda")
    """

    method_name = "source_only_backbone"

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)

        # Load model
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        config = checkpoint.get("config", {})
        ch_mean = checkpoint.get("config", {}).get("ch_mean")
        ch_std = checkpoint.get("config", {}).get("ch_std")

        # Init model — read width from checkpoint config
        saved_config = checkpoint.get("config", {})
        width = saved_config.get("width", 32)
        self.model = SmallResUNet(in_channels=12, out_channels=2, width=width)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device).eval()

        # Normalization params
        self._ch_mean = np.array(ch_mean, dtype=np.float32) if ch_mean is not None else None
        self._ch_std = np.array(ch_std, dtype=np.float32) if ch_std is not None else None

        # Increment normalization params (when target_increment_normalization was used)
        inc_mean = saved_config.get("inc_mean")
        inc_std = saved_config.get("inc_std")
        self._inc_mean = np.array(inc_mean, dtype=np.float32) if inc_mean is not None else None
        self._inc_std = np.array(inc_std, dtype=np.float32) if inc_std is not None else None
        self._has_inc_norm = self._inc_mean is not None and self._inc_std is not None

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise normalization."""
        if self._ch_mean is None or self._ch_std is None:
            return x
        mean_t = torch.from_numpy(self._ch_mean).to(x.device).view(1, 12, 1, 1)
        std_t = torch.from_numpy(self._ch_std).to(x.device).view(1, 12, 1, 1)
        return (x - mean_t) / std_t

    def predict(self, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Predict DA increments and analysis for a single sample.

        Args:
            sample: dict with keys:
                - x: raw input array [12, H, W]
                - forecast_surface: [H, W]
                - forecast_rootzone: [H, W]
                - (optional) date_str, metric_mask, etc.

        Returns:
            dict with:
                - pred_increment_surface: [H, W]
                - pred_increment_rootzone: [H, W]
                - pred_analysis_surface: [H, W]
                - pred_analysis_rootzone: [H, W]
        """
        x = torch.from_numpy(np.asarray(sample["x"], dtype=np.float32))
        x = x.unsqueeze(0).to(self.device)  # [1, 12, H, W]

        x_norm = self._normalize(x)

        with torch.no_grad():
            pred = self.model(x_norm)  # [1, 2, H, W]

        pred_inc_s = pred[0, 0].cpu().numpy().astype(np.float32)
        pred_inc_r = pred[0, 1].cpu().numpy().astype(np.float32)

        forecast_surface = np.asarray(sample["forecast_surface"], dtype=np.float32)
        forecast_rootzone = np.asarray(sample["forecast_rootzone"], dtype=np.float32)

        # Denormalize increments back to raw m³/m³ if model was trained with increment normalization
        if self._has_inc_norm:
            pred_inc_s = pred_inc_s * self._inc_std[0] + self._inc_mean[0]
            pred_inc_r = pred_inc_r * self._inc_std[1] + self._inc_mean[1]

        pred_analysis_surface = (forecast_surface + pred_inc_s).astype(np.float32)
        pred_analysis_rootzone = (forecast_rootzone + pred_inc_r).astype(np.float32)

        return {
            "pred_increment_surface": pred_inc_s,
            "pred_increment_rootzone": pred_inc_r,
            "pred_analysis_surface": pred_analysis_surface,
            "pred_analysis_rootzone": pred_analysis_rootzone,
        }