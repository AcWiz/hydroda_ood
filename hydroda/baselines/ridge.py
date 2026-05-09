"""Ridge regression baseline with three feature sets.

No-leakage declaration:
    - Fit on target_support only
    - Predict on target_query
    - No target_query labels used in fitting
    - metric_mask applied before computing targets
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.linear_model import Ridge
from typing import Dict, Any, Optional, List


# Feature set definitions
FEATURE_SETS = {
    "core": {
        "features": ["forecast_surface", "forecast_rootzone", "sin_day", "cos_day"],
        "description": "Minimal sanity feature set",
    },
    "input_full": {
        "features": [
            "forecast_surface",
            "forecast_rootzone",
            "tb_h",
            "tb_v",
            "tb_v_minus_tb_h",
            "vegopacity",
            "obs_mask",
            "sin_day",
            "cos_day",
        ],
        "description": "Main method: full input features without geolocation",
    },
    "input_geo_full": {
        "features": [
            "forecast_surface",
            "forecast_rootzone",
            "tb_h",
            "tb_v",
            "tb_v_minus_tb_h",
            "vegopacity",
            "obs_mask",
            "sin_day",
            "cos_day",
            "lat_grid",
            "lon_grid",
        ],
        "description": "Ablation: input_full + geolocation encoding",
    },
}


def _sin_cos_day(day_of_year: int) -> tuple:
    """Return (sin_day, cos_day) for day-of-year cyclical encoding."""
    TWO_PI = 2.0 * np.pi
    return (
        float(np.sin(TWO_PI * day_of_year / 365.0)),
        float(np.cos(TWO_PI * day_of_year / 365.0)),
    )


def _extract_features(sample: Dict[str, Any], feature_names: List[str]) -> np.ndarray:
    """Extract and stack features from a sample into a 1D feature vector.

    Returns:
        Feature vector of shape (n_features,).
    """
    features = []
    x = sample["x"]  # (12, H, W)
    H, W = x.shape[1], x.shape[2]

    date_str = sample.get("date_str", "")
    # Parse YYYY-MM-DD
    if date_str and len(date_str) >= 10:
        day_of_year = int(date_str[5:7]) * 30 + int(date_str[8:10])
    else:
        day_of_year = 180
    sin_day, cos_day = _sin_cos_day(day_of_year)

    for fname in feature_names:
        if fname == "forecast_surface":
            features.append(x[0].flatten())
        elif fname == "forecast_rootzone":
            features.append(x[1].flatten())
        elif fname == "tb_h":
            features.append(x[5].flatten())
        elif fname == "tb_v":
            features.append(x[6].flatten())
        elif fname == "tb_v_minus_tb_h":
            features.append((x[6] - x[5]).flatten())
        elif fname == "vegopacity":
            features.append(x[4].flatten())
        elif fname == "obs_mask":
            # Use base_valid_mask (ch 11) as obs_mask proxy
            features.append((x[11] > 0.5).astype(np.float32).flatten())
        elif fname == "sin_day":
            features.append(np.full(H * W, sin_day, dtype=np.float32))
        elif fname == "cos_day":
            features.append(np.full(H * W, cos_day, dtype=np.float32))
        elif fname == "lat_grid":
            lat = sample.get("lat_grid")
            if lat is None:
                lat = np.linspace(0, 1, H * W).reshape(H, W).astype(np.float32)
            features.append(lat.flatten())
        elif fname == "lon_grid":
            lon = sample.get("lon_grid")
            if lon is None:
                lon = np.linspace(0, 1, H * W).reshape(H, W).astype(np.float32)
            features.append(lon.flatten())
        else:
            raise ValueError(f"Unknown feature: {fname}")

    return np.stack(features, axis=0)  # (n_features, H*W)


def _build_latlon_grids(region_masks_nc: str) -> tuple:
    """Load lat/lon grids from geolocation artifact if available."""
    latlon_path = Path("artifacts/geolocation/US_latlon.nc")
    if latlon_path.exists():
        ds = xr.open_dataset(latlon_path)
        lat = ds["lat"].values.astype(np.float32)
        lon = ds["lon"].values.astype(np.float32)
        ds.close()
        return lat, lon
    return None, None


class RidgeBaseline:
    """Ridge regression baseline with configurable feature sets.

    Fit on target_support samples, predict on target_query samples.

    Args:
        feature_set: "core" | "input_full" | "input_geo_full"
        alpha: Ridge regularization strength (default 1.0)

    No-leakage: trained only on target_support, evaluated on target_query.
    """

    def __init__(self, feature_set: str = "input_full", alpha: float = 1.0):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature_set: {feature_set}. Must be one of {list(FEATURE_SETS.keys())}")
        self.feature_set = feature_set
        self.feature_names = FEATURE_SETS[feature_set]["features"]
        self.alpha = alpha
        self._model_s: Optional[Ridge] = None
        self._model_r: Optional[Ridge] = None
        self._lat_grid: Optional[np.ndarray] = None
        self._lon_grid: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, samples: list) -> "RidgeBaseline":
        """Fit Ridge models for surface and rootzone increments.

        Uses metric_mask to select valid pixels before computing targets.

        Args:
            samples: List of sample dicts from target_support split.
        """
        import xarray as xr

        # Load geolocation grids if needed
        if "lat_grid" in self.feature_names or "lon_grid" in self.feature_names:
            H_sample, W_sample = samples[0]["x"].shape[1], samples[0]["x"].shape[2]
            latlon_path = Path("artifacts/geolocation/US_latlon.nc")
            if latlon_path.exists():
                ds = xr.open_dataset(latlon_path)
                lat_grid = ds["latitude"].values.astype(np.float32)
                lon_grid = ds["longitude"].values.astype(np.float32)
                ds.close()
                # Use artifact only if shape matches sample; otherwise generate uniform
                if lat_grid.shape == (H_sample, W_sample):
                    self._lat_grid = lat_grid
                    self._lon_grid = lon_grid
                else:
                    # Shape mismatch: generate uniform grids based on sample shape
                    self._lat_grid = np.linspace(0, 1, H_sample * W_sample).reshape(H_sample, W_sample).astype(np.float32)
                    self._lon_grid = np.linspace(0, 1, H_sample * W_sample).reshape(H_sample, W_sample).astype(np.float32)
            else:
                # Fallback: uniform grids
                self._lat_grid = np.linspace(0, 1, H_sample * W_sample).reshape(H_sample, W_sample).astype(np.float32)
                self._lon_grid = np.linspace(0, 1, H_sample * W_sample).reshape(H_sample, W_sample).astype(np.float32)

        # Collect feature matrices and target vectors
        X_list = []
        y_s_list = []
        y_r_list = []

        for s in samples:
            X = _extract_features(s, self.feature_names)  # (n_features, H*W)
            # Add lat/lon if needed
            if "lat_grid" in self.feature_names:
                lat_flat = self._lat_grid.flatten()
                lon_flat = self._lon_grid.flatten()
                lat_mean = np.nanmean(lat_flat)
                lon_mean = np.nanmean(lon_flat)
                lat_norm = (lat_flat - lat_mean) / (np.nanstd(lat_flat) + 1e-8)
                lon_norm = (lon_flat - lon_mean) / (np.nanstd(lon_flat) + 1e-8)
                # Reshape to (1, H*W) for vstack with (n_features, H*W)
                X = np.vstack([X, lat_norm[np.newaxis, :], lon_norm[np.newaxis, :]])

            inc_s = s["increment_surface"].flatten()
            inc_r = s["increment_rootzone"].flatten()
            mask = s["metric_mask"].flatten() > 0.5

            # Only use valid, finite pixels
            valid = mask & np.isfinite(inc_s) & np.isfinite(inc_r)
            if valid.sum() == 0:
                continue

            X_valid = X[:, valid].T  # (n_valid, n_features)
            y_s_list.append(inc_s[valid])
            y_r_list.append(inc_r[valid])
            X_list.append(X_valid)

        if not X_list:
            raise ValueError("No valid pixels found for ridge fitting")

        X_all = np.vstack(X_list)  # (n_total, n_features)
        y_s_all = np.concatenate(y_s_list)
        y_r_all = np.concatenate(y_r_list)

        # Impute NaN in features with column mean
        col_means = np.nanmean(X_all, axis=0)
        for j in range(X_all.shape[1]):
            nan_mask = np.isnan(X_all[:, j])
            if nan_mask.any():
                X_all[nan_mask, j] = col_means[j]

        # Fit Ridge models
        self._model_s = Ridge(alpha=self.alpha, fit_intercept=True)
        self._model_r = Ridge(alpha=self.alpha, fit_intercept=True)
        self._model_s.fit(X_all, y_s_all)
        self._model_r.fit(X_all, y_r_all)
        self._n_features = X_all.shape[1]
        self._col_means = col_means
        self._fitted = True
        return self

    def predict(self, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Predict increment using fitted Ridge models.

        Args:
            sample: Dict with x, forecast_surface, forecast_rootzone, date_str,
                and optionally lat_grid/lon_grid.

        Returns:
            dict with pred_increment_surface, pred_increment_rootzone,
            pred_analysis_surface, pred_analysis_rootzone.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        H, W = sample["forecast_surface"].shape

        X = _extract_features(sample, self.feature_names)  # (n_features, H*W)

        # Add lat/lon if needed
        if "lat_grid" in self.feature_names:
            lat_flat = self._lat_grid.flatten()
            lon_flat = self._lon_grid.flatten()
            lat_mean = np.nanmean(lat_flat)
            lon_mean = np.nanmean(lon_flat)
            lat_norm = (lat_flat - lat_mean) / (np.nanstd(lat_flat) + 1e-8)
            lon_norm = (lon_flat - lon_mean) / (np.nanstd(lon_flat) + 1e-8)
            X = np.vstack([X, lat_norm[np.newaxis, :], lon_norm[np.newaxis, :]])

        X_flat = X.T  # (H*W, n_features)

        # Impute NaN with stored column means
        for j in range(X_flat.shape[1]):
            nan_mask = np.isnan(X_flat[:, j])
            if nan_mask.any():
                X_flat[nan_mask, j] = self._col_means[j]

        pred_inc_s_flat = self._model_s.predict(X_flat)
        pred_inc_r_flat = self._model_r.predict(X_flat)

        pred_inc_s = pred_inc_s_flat.reshape(H, W).astype(np.float32)
        pred_inc_r = pred_inc_r_flat.reshape(H, W).astype(np.float32)

        return {
            "pred_increment_surface": pred_inc_s,
            "pred_increment_rootzone": pred_inc_r,
            "pred_analysis_surface": (sample["forecast_surface"] + pred_inc_s).astype(np.float32),
            "pred_analysis_rootzone": (sample["forecast_rootzone"] + pred_inc_r).astype(np.float32),
        }
