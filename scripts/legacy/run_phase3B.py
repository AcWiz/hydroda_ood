#!/usr/bin/env python3
"""Phase 3B — Simple baselines evaluation (source_mean, target_support_mean, monthly_mean, ridge).

Streaming evaluation: processes one split at a time with minimal memory footprint.
Supports source_mean (K=0,4,12,24), target_mean (K=4,12,24), monthly_mean (K=12,24),
and ridge (core, input_full, input_geo_full) at K=4,12,24.
"""

from __future__ import annotations

import sys
import json
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4
import xarray as xr

from hydroda.metrics.skill import (
    analysis_rmse, analysis_mae, analysis_skill_vs_forecast,
    increment_rmse, increment_mae, increment_bias, increment_corr,
    sign_accuracy_deadzone, valid_pixel_count, effective_mask_fraction,
)

DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
GEO_LATLON = "artifacts/geolocation/US_latlon.nc"
OUTPUT_DIR_BASE = Path("artifacts/metrics")
OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

with open(FREEZE_MANIFEST) as f:
    FREEZE_ID = json.load(f)["freeze_id"]

REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
K_VALUES = [0, 4, 12]
SEEDS = [0, 1, 2]
_ALL_REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
CHUNK = 100

_VARIABLE_PAIRS = [
    ("surface", "forecast_surface", "analysis_surface", "increment_surface", "pred_increment_surface", "pred_analysis_surface"),
    ("rootzone", "forecast_rootzone", "analysis_rootzone", "increment_rootzone", "pred_increment_rootzone", "pred_analysis_rootzone"),
]
_METRIC_FUNCS = [
    ("analysis_rmse", lambda p, t, m: analysis_rmse(p, t, m)),
    ("analysis_mae", lambda p, t, m: analysis_mae(p, t, m)),
    ("analysis_skill_vs_forecast", lambda p, t, f, m: analysis_skill_vs_forecast(p, t, f, m)),
    ("increment_rmse", lambda p, t, m: increment_rmse(p, t, m)),
    ("increment_mae", lambda p, t, m: increment_mae(p, t, m)),
    ("increment_bias", lambda p, t, m: increment_bias(p, t, m)),
    ("increment_corr", lambda p, t, m: increment_corr(p, t, m)),
    ("sign_accuracy_deadzone", lambda p, t, m: sign_accuracy_deadzone(p, t, m)),
]


def _parse_month(date_str: str) -> int:
    if date_str and len(date_str) >= 7:
        return int(date_str[5:7])
    return 1


def _sin_cos_day(day_of_year: int) -> tuple:
    TWO_PI = 2.0 * np.pi
    return (
        float(np.sin(TWO_PI * day_of_year / 365.0)),
        float(np.cos(TWO_PI * day_of_year / 365.0)),
    )


def compute_metrics_for_sample(
    pred, sample, experiment_id, split_role, protocol_freeze_id, method_name
) -> list:
    """Compute all metrics for a single sample."""
    metric_mask = sample["metric_mask"]
    H, W = metric_mask.shape
    n_valid = valid_pixel_count(metric_mask)
    mask_frac = effective_mask_fraction(metric_mask, H * W)

    results = []
    for var_name, fcst_key, analy_key, incr_key, pred_incr_key, pred_analy_key in _VARIABLE_PAIRS:
        forecast = sample[fcst_key]
        true_analysis = sample[analy_key]
        pred_analysis = pred[pred_analy_key]

        for metric_name, metric_fn in _METRIC_FUNCS:
            if metric_name == "analysis_skill_vs_forecast":
                value = metric_fn(pred_analysis, true_analysis, forecast, metric_mask)
            else:
                value = metric_fn(pred_analysis, true_analysis, metric_mask)

            results.append({
                "run_id": f"{experiment_id}_{sample.get('time_index', -1)}",
                "query_date": sample.get("date_str", ""),
                "query_time_index": int(sample.get("time_index", -1)),
                "support_dates_hash": "",
                "split_file": SPLITS_JSON,
                "mask_file": "artifacts/regions/US_region_masks.nc",
                "experiment_id": experiment_id,
                "method": method_name,
                "country_id": sample["country_id"],
                "target_region_id": sample["target_region_id"],
                "active_region_ids": "|".join(sample["active_region_ids"]),
                "split_role": split_role,
                "K": sample["K"],
                "seed": sample["seed"],
                "variable": var_name,
                "metric": metric_name,
                "value": value,
                "n_valid_pixels": n_valid,
                "n_time_steps": 1,
                "mask_fraction": mask_frac,
                "protocol_freeze_id": protocol_freeze_id,
            })
    return results


class SourceMeanPredictor:
    """Fitted source mean increment predictor."""
    def __init__(self, mean_inc_s, mean_inc_r):
        self.mean_inc_s = mean_inc_s
        self.mean_inc_r = mean_inc_r

    def predict(self, sample):
        H, W = sample["forecast_surface"].shape
        pred_inc_s = np.full((H, W), self.mean_inc_s, dtype=np.float32)
        pred_inc_r = np.full((H, W), self.mean_inc_r, dtype=np.float32)
        return {
            "pred_increment_surface": pred_inc_s,
            "pred_increment_rootzone": pred_inc_r,
            "pred_analysis_surface": (sample["forecast_surface"] + pred_inc_s).astype(np.float32),
            "pred_analysis_rootzone": (sample["forecast_rootzone"] + pred_inc_r).astype(np.float32),
        }


class TargetMeanPredictor:
    """Fitted target support mean increment predictor."""
    def __init__(self, mean_inc_s, mean_inc_r):
        self.mean_inc_s = mean_inc_s
        self.mean_inc_r = mean_inc_r

    def predict(self, sample):
        H, W = sample["forecast_surface"].shape
        pred_inc_s = np.full((H, W), self.mean_inc_s, dtype=np.float32)
        pred_inc_r = np.full((H, W), self.mean_inc_r, dtype=np.float32)
        return {
            "pred_increment_surface": pred_inc_s,
            "pred_increment_rootzone": pred_inc_r,
            "pred_analysis_surface": (sample["forecast_surface"] + pred_inc_s).astype(np.float32),
            "pred_analysis_rootzone": (sample["forecast_rootzone"] + pred_inc_r).astype(np.float32),
        }


class MonthlyMeanPredictor:
    """Fitted monthly mean increment predictor."""
    def __init__(self, monthly_s, monthly_r):
        self.monthly_s = monthly_s
        self.monthly_r = monthly_r

    def predict(self, sample):
        H, W = sample["forecast_surface"].shape
        month = int(sample.get("month", 1))
        mean_s = self.monthly_s.get(month, 0.0)
        mean_r = self.monthly_r.get(month, 0.0)
        pred_inc_s = np.full((H, W), mean_s, dtype=np.float32)
        pred_inc_r = np.full((H, W), mean_r, dtype=np.float32)
        return {
            "pred_increment_surface": pred_inc_s,
            "pred_increment_rootzone": pred_inc_r,
            "pred_analysis_surface": (sample["forecast_surface"] + pred_inc_s).astype(np.float32),
            "pred_analysis_rootzone": (sample["forecast_rootzone"] + pred_inc_r).astype(np.float32),
        }


class RidgePixelPredictor:
    """Ridge regression predictor operating on pixel-level flattened features."""
    def __init__(self, model_s, model_r, col_means):
        self.model_s = model_s
        self.model_r = model_r
        self.col_means = col_means

    def predict(self, sample):
        H, W = sample["forecast_surface"].shape
        X = self._extract_features(sample)  # (n_features, H*W)
        X_flat = X.T  # (H*W, n_features)

        # Impute NaN
        for j in range(X_flat.shape[1]):
            nan_mask = np.isnan(X_flat[:, j])
            if nan_mask.any():
                X_flat[nan_mask, j] = self.col_means[j]

        pred_inc_s_flat = self.model_s.predict(X_flat)
        pred_inc_r_flat = self.model_r.predict(X_flat)

        pred_inc_s = pred_inc_s_flat.reshape(H, W).astype(np.float32)
        pred_inc_r = pred_inc_r_flat.reshape(H, W).astype(np.float32)

        return {
            "pred_increment_surface": pred_inc_s,
            "pred_increment_rootzone": pred_inc_r,
            "pred_analysis_surface": (sample["forecast_surface"] + pred_inc_s).astype(np.float32),
            "pred_analysis_rootzone": (sample["forecast_rootzone"] + pred_inc_r).astype(np.float32),
        }

    def _extract_features(self, sample):
        """Extract feature matrix from sample."""
        x = sample["x"]  # (12, H, W)
        H, W = x.shape[1], x.shape[2]

        date_str = sample.get("date_str", "")
        if date_str and len(date_str) >= 10:
            day_of_year = int(date_str[5:7]) * 30 + int(date_str[8:10])
        else:
            day_of_year = 180
        sin_day, cos_day = _sin_cos_day(day_of_year)

        features = []
        features.append(x[0].flatten())  # forecast_surface
        features.append(x[1].flatten())  # forecast_rootzone
        features.append(x[5].flatten())  # tb_h
        features.append(x[6].flatten())  # tb_v
        features.append((x[6] - x[5]).flatten())  # tb_v_minus_tb_h
        features.append(x[4].flatten())  # vegopacity
        features.append((x[11] > 0.5).astype(np.float32).flatten())  # obs_mask
        features.append(np.full(H * W, sin_day, dtype=np.float32))
        features.append(np.full(H * W, cos_day, dtype=np.float32))

        return np.stack(features, axis=0)  # (n_features, H*W)


class RidgeGeoPixelPredictor(RidgePixelPredictor):
    """Ridge with geolocation features."""
    def __init__(self, model_s, model_r, col_means, lat_grid, lon_grid):
        super().__init__(model_s, model_r, col_means)
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid

    def _extract_features(self, sample):
        X = super()._extract_features(sample)
        lat_flat = self.lat_grid.flatten()
        lon_flat = self.lon_grid.flatten()
        lat_mean = np.nanmean(lat_flat)
        lon_mean = np.nanmean(lon_flat)
        lat_norm = (lat_flat - lat_mean) / (np.nanstd(lat_flat) + 1e-8)
        lon_norm = (lon_flat - lon_mean) / (np.nanstd(lon_flat) + 1e-8)
        X = np.vstack([X, lat_norm[np.newaxis, :], lon_norm[np.newaxis, :]])
        return X


def fit_source_mean(target_region, K, splits_by_key, region_stats, region_mask_int):
    """Compute source mean increment for a split."""
    split_entry = splits_by_key[(target_region, K, 0)]  # source_train same for all seeds
    active_region_ids = [r for r in _ALL_REGIONS if r != target_region]
    rnum_list = [int(rid.split("-R")[1]) for rid in active_region_ids]
    active_region_mask = np.isin(region_mask_int, rnum_list).astype(np.float32)

    date_list = split_entry["source_train_dates"]
    time_indices = [d["time_index"] for d in date_list]

    inc_s_list = []
    inc_r_list = []

    ds = netCDF4.Dataset(DA_NC, "r")
    inp_var = ds.variables["input"]
    tgt_var = ds.variables["target"]

    for start in range(0, len(time_indices), CHUNK):
        end = min(start + CHUNK, len(time_indices))
        inp = inp_var[time_indices[start:end]].astype(np.float32)
        tgt = tgt_var[time_indices[start:end]].astype(np.float32)

        for i in range(inp.shape[0]):
            base_mask = (inp[i, 11] > 0.5).astype(np.float32)
            loss_mask = (
                active_region_mask.astype(bool)
                & (base_mask > 0.5)
                & np.isfinite(inp[i, 0])
                & np.isfinite(inp[i, 1])
                & np.isfinite(tgt[i, 0])
                & np.isfinite(tgt[i, 1])
            )
            valid = loss_mask.flatten()
            if valid.sum() > 0:
                inc_s_list.append((tgt[i, 0] - inp[i, 0]).flatten()[valid])
                inc_r_list.append((tgt[i, 1] - inp[i, 1]).flatten()[valid])
        del inp, tgt
        gc.collect()
    ds.close()

    mean_s = float(np.mean(np.concatenate(inc_s_list)))
    mean_r = float(np.mean(np.concatenate(inc_r_list)))
    del inc_s_list, inc_r_list
    gc.collect()

    return SourceMeanPredictor(mean_s, mean_r)


def fit_target_mean(target_region, K, splits_by_key, region_mask_int):
    """Compute target support mean increment for a split (K>0)."""
    split_entry = splits_by_key[(target_region, K, 0)]  # same for all seeds
    rnum_list = [int(target_region.split("-R")[1])]
    active_region_mask = np.isin(region_mask_int, rnum_list).astype(np.float32)

    date_list = split_entry["target_support_dates"]
    time_indices = [d["time_index"] for d in date_list]

    inc_s_list = []
    inc_r_list = []

    ds = netCDF4.Dataset(DA_NC, "r")
    inp_var = ds.variables["input"]
    tgt_var = ds.variables["target"]

    for start in range(0, len(time_indices), CHUNK):
        end = min(start + CHUNK, len(time_indices))
        inp = inp_var[time_indices[start:end]].astype(np.float32)
        tgt = tgt_var[time_indices[start:end]].astype(np.float32)

        for i in range(inp.shape[0]):
            base_mask = (inp[i, 11] > 0.5).astype(np.float32)
            loss_mask = (
                active_region_mask.astype(bool)
                & (base_mask > 0.5)
                & np.isfinite(inp[i, 0])
                & np.isfinite(inp[i, 1])
                & np.isfinite(tgt[i, 0])
                & np.isfinite(tgt[i, 1])
            )
            valid = loss_mask.flatten()
            if valid.sum() > 0:
                inc_s_list.append((tgt[i, 0] - inp[i, 0]).flatten()[valid])
                inc_r_list.append((tgt[i, 1] - inp[i, 1]).flatten()[valid])
        del inp, tgt
        gc.collect()
    ds.close()

    if not inc_s_list:
        return None

    mean_s = float(np.mean(np.concatenate(inc_s_list)))
    mean_r = float(np.mean(np.concatenate(inc_r_list)))
    del inc_s_list, inc_r_list
    gc.collect()

    return TargetMeanPredictor(mean_s, mean_r)


def fit_monthly_mean(target_region, K, splits_by_key, region_mask_int):
    """Compute monthly mean increment for a split (K in 12, 24)."""
    split_entry = splits_by_key[(target_region, K, 0)]
    rnum_list = [int(target_region.split("-R")[1])]
    active_region_mask = np.isin(region_mask_int, rnum_list).astype(np.float32)

    date_list = split_entry["target_support_dates"]
    time_indices = [d["time_index"] for d in date_list]
    date_str_map = {d["time_index"]: d["date_str"] for d in date_list}

    month_accum_s = {m: [] for m in range(1, 13)}
    month_accum_r = {m: [] for m in range(1, 13)}

    ds = netCDF4.Dataset(DA_NC, "r")
    inp_var = ds.variables["input"]
    tgt_var = ds.variables["target"]

    for start in range(0, len(time_indices), CHUNK):
        end = min(start + CHUNK, len(time_indices))
        inp = inp_var[time_indices[start:end]].astype(np.float32)
        tgt = tgt_var[time_indices[start:end]].astype(np.float32)

        for i in range(inp.shape[0]):
            ti = time_indices[start + i]
            date_str = date_str_map.get(int(ti), "")
            month = _parse_month(date_str)

            base_mask = (inp[i, 11] > 0.5).astype(np.float32)
            loss_mask = (
                active_region_mask.astype(bool)
                & (base_mask > 0.5)
                & np.isfinite(inp[i, 0])
                & np.isfinite(inp[i, 1])
                & np.isfinite(tgt[i, 0])
                & np.isfinite(tgt[i, 1])
            )
            valid = loss_mask.flatten()
            if valid.sum() > 0:
                inc_s = (tgt[i, 0] - inp[i, 0]).flatten()[valid]
                inc_r = (tgt[i, 1] - inp[i, 1]).flatten()[valid]
                month_accum_s[month].append(inc_s)
                month_accum_r[month].append(inc_r)
        del inp, tgt
        gc.collect()
    ds.close()

    monthly_s = {}
    monthly_r = {}
    for month in range(1, 13):
        if month_accum_s[month]:
            monthly_s[month] = float(np.nanmean(np.concatenate(month_accum_s[month])))
        else:
            monthly_s[month] = 0.0
        if month_accum_r[month]:
            monthly_r[month] = float(np.nanmean(np.concatenate(month_accum_r[month])))
        else:
            monthly_r[month] = 0.0

    return MonthlyMeanPredictor(monthly_s, monthly_r)


def fit_ridge(target_region, K, splits_by_key, region_mask_int, feature_set, geo_available):
    """Fit Ridge regression predictor (K>0)."""
    from sklearn.linear_model import Ridge

    split_entry = splits_by_key[(target_region, K, 0)]
    rnum_list = [int(target_region.split("-R")[1])]
    active_region_mask = np.isin(region_mask_int, rnum_list).astype(np.float32)

    date_list = split_entry["target_support_dates"]
    time_indices = [d["time_index"] for d in date_list]
    date_str_map = {d["time_index"]: d["date_str"] for d in date_list}

    X_list = []
    y_s_list = []
    y_r_list = []

    # Get sample shape
    ds = netCDF4.Dataset(DA_NC, "r")
    inp_var = ds.variables["input"]
    sample_shape = inp_var[time_indices[0:1]].astype(np.float32).shape[1:]
    H, W = sample_shape[1], sample_shape[2]

    # Load lat/lon grids if needed
    lat_grid = None
    lon_grid = None
    if feature_set in ("input_geo_full",) and geo_available:
        try:
            geo_ds = xr.open_dataset(GEO_LATLON)
            lat_grid = geo_ds["latitude"].values.astype(np.float32)
            lon_grid = geo_ds["longitude"].values.astype(np.float32)
            geo_ds.close()
            if lat_grid.shape != (H, W):
                lat_grid = None
                lon_grid = None
        except Exception:
            lat_grid = None
            lon_grid = None

    for start in range(0, len(time_indices), CHUNK):
        end = min(start + CHUNK, len(time_indices))
        inp = inp_var[time_indices[start:end]].astype(np.float32)
        tgt = ds.variables["target"][time_indices[start:end]].astype(np.float32)

        for i in range(inp.shape[0]):
            ti = time_indices[start + i]
            date_str = date_str_map.get(int(ti), "")

            base_mask = (inp[i, 11] > 0.5).astype(np.float32)
            metric_mask = (
                active_region_mask.astype(bool)
                & (base_mask > 0.5)
                & np.isfinite(inp[i, 0])
                & np.isfinite(inp[i, 1])
                & np.isfinite(tgt[i, 0])
                & np.isfinite(tgt[i, 1])
            )
            valid = metric_mask.flatten()

            if valid.sum() == 0:
                continue

            # Extract features
            x_i = inp[i]  # (12, H, W)
            inc_s = (tgt[i, 0] - inp[i, 0]).flatten()
            inc_r = (tgt[i, 1] - inp[i, 1]).flatten()

            # Parse day
            if date_str and len(date_str) >= 10:
                day_of_year = int(date_str[5:7]) * 30 + int(date_str[8:10])
            else:
                day_of_year = 180
            sin_day, cos_day = _sin_cos_day(day_of_year)

            H_i, W_i = H, W
            features = []
            features.append(x_i[0].flatten())  # forecast_surface
            features.append(x_i[1].flatten())  # forecast_rootzone
            features.append(x_i[5].flatten())  # tb_h
            features.append(x_i[6].flatten())  # tb_v
            features.append((x_i[6] - x_i[5]).flatten())
            features.append(x_i[4].flatten())  # vegopacity
            features.append((x_i[11] > 0.5).astype(np.float32).flatten())
            features.append(np.full(H_i * W_i, sin_day, dtype=np.float32))
            features.append(np.full(H_i * W_i, cos_day, dtype=np.float32))

            X = np.stack(features, axis=0)  # (9, H*W)

            if feature_set == "input_geo_full" and lat_grid is not None:
                lat_flat = lat_grid.flatten()
                lon_flat = lon_grid.flatten()
                lat_norm = (lat_flat - np.nanmean(lat_flat)) / (np.nanstd(lat_flat) + 1e-8)
                lon_norm = (lon_flat - np.nanmean(lon_flat)) / (np.nanstd(lon_flat) + 1e-8)
                X = np.vstack([X, lat_norm[np.newaxis, :], lon_norm[np.newaxis, :]])

            X_valid = X[:, valid].T  # (n_valid, n_features)
            X_list.append(X_valid)
            y_s_list.append(inc_s[valid])
            y_r_list.append(inc_r[valid])

        del inp, tgt
        gc.collect()
    ds.close()

    if not X_list:
        return None

    X_all = np.vstack(X_list)
    y_s_all = np.concatenate(y_s_list)
    y_r_all = np.concatenate(y_r_list)
    del X_list, y_s_list, y_r_list
    gc.collect()

    # Impute NaN
    col_means = np.nanmean(X_all, axis=0)
    for j in range(X_all.shape[1]):
        nan_mask = np.isnan(X_all[:, j])
        if nan_mask.any():
            X_all[nan_mask, j] = col_means[j]

    # Fit Ridge
    model_s = Ridge(alpha=1.0, fit_intercept=True)
    model_r = Ridge(alpha=1.0, fit_intercept=True)
    model_s.fit(X_all, y_s_all)
    model_r.fit(X_all, y_r_all)

    del X_all, y_s_all, y_r_all
    gc.collect()

    if feature_set == "input_geo_full" and lat_grid is not None:
        return RidgeGeoPixelPredictor(model_s, model_r, col_means, lat_grid, lon_grid)
    else:
        return RidgePixelPredictor(model_s, model_r, col_means)


def evaluate_on_query(predictor, method_name, target_region, K, seed, splits_by_key, region_mask_int, result_list):
    """Evaluate a fitted predictor on target_query samples."""
    split_entry = splits_by_key[(target_region, K, seed)]
    rnum_list = [int(target_region.split("-R")[1])]
    active_region_mask = np.isin(region_mask_int, rnum_list).astype(np.float32)

    date_list = split_entry["target_query_dates"]
    time_indices = [d["time_index"] for d in date_list]
    date_str_map = {d["time_index"]: d["date_str"] for d in date_list}

    experiment_id = f"phase3B_{method_name}_{target_region}_K{K}_S{seed}_target_query"

    ds = netCDF4.Dataset(DA_NC, "r")
    inp_var = ds.variables["input"]
    tgt_var = ds.variables["target"]

    for start in range(0, len(time_indices), CHUNK):
        end = min(start + CHUNK, len(time_indices))
        inp = inp_var[time_indices[start:end]].astype(np.float32)
        tgt = tgt_var[time_indices[start:end]].astype(np.float32)

        for i in range(inp.shape[0]):
            ti = time_indices[start + i]
            date_str = date_str_map.get(int(ti), "")
            month = _parse_month(date_str)

            forecast_s = inp[i, 0]
            forecast_r = inp[i, 1]
            analysis_s = tgt[i, 0]
            analysis_r = tgt[i, 1]

            base_mask = (inp[i, 11] > 0.5).astype(np.float32)
            metric_mask = (
                active_region_mask.astype(bool)
                & (base_mask > 0.5)
                & np.isfinite(forecast_s)
                & np.isfinite(forecast_r)
                & np.isfinite(analysis_s)
                & np.isfinite(analysis_r)
            ).astype(np.float32)

            if metric_mask.sum() == 0:
                continue

            sample = {
                "forecast_surface": forecast_s,
                "forecast_rootzone": forecast_r,
                "analysis_surface": analysis_s,
                "analysis_rootzone": analysis_r,
                "increment_surface": analysis_s - forecast_s,
                "increment_rootzone": analysis_r - forecast_r,
                "x": inp[i].astype(np.float32),
                "date_str": date_str,
                "month": month,
                "country_id": "US",
                "target_region_id": target_region,
                "active_region_ids": [target_region],
                "K": K,
                "seed": seed,
                "metric_mask": metric_mask,
            }

            pred = predictor.predict(sample)
            metrics = compute_metrics_for_sample(pred, sample, experiment_id, "target_query", FREEZE_ID, method_name)
            result_list.extend(metrics)

        del inp, tgt
        gc.collect()
    ds.close()


def run_phase3B() -> None:
    # Load splits
    with open(SPLITS_JSON) as f:
        splits_data = json.load(f)
    splits_by_key = {
        (s["target_region_id"], s["K"], s["seed"]): s
        for s in splits_data["splits"]
    }

    with open("artifacts/regions/US_region_stats.json") as f:
        region_stats = json.load(f)

    region_ds = xr.open_dataset(REGION_MASKS_NC)
    region_mask_int = region_ds["region_mask_integer"].values.astype(np.int16)
    region_ds.close()

    geo_available = Path(GEO_LATLON).exists()

    # Total evaluations:
    # source_mean: 6*4*10=240, target_mean: 6*3*10=180,
    # monthly_mean: 6*2*10=120, ridge_core/input_full: 6*3*10*3=540
    # Total = 1080
    print(f"\nPhase 3B: 1080 evaluations", flush=True)
    t_start = time.time()

    all_results = {
        "source_mean_increment": [],
        "target_support_mean_increment": [],
        "target_monthly_support_increment": [],
        "ridge_core": [],
        "ridge_input_full": [],
        "ridge_input_geo_full": [],
    }

    # Cache fitted predictors (same for all seeds of a given region+K)
    source_mean_cache = {}
    target_mean_cache = {}
    monthly_mean_cache = {}
    ridge_cache = {}

    eval_count = 0

    for target_region in REGIONS:
        for K in K_VALUES:
            # Fit source_mean (same for all seeds)
            cache_key = (target_region, K)
            if cache_key not in source_mean_cache:
                source_mean_cache[cache_key] = fit_source_mean(target_region, K, splits_by_key, region_stats, region_mask_int)
            predictor_source = source_mean_cache[cache_key]

            # Fit target_mean (K>0, same for all seeds)
            if K > 0 and cache_key not in target_mean_cache:
                target_mean_cache[cache_key] = fit_target_mean(target_region, K, splits_by_key, region_mask_int)
            predictor_target = target_mean_cache.get(cache_key) if K > 0 else None

            # Fit monthly_mean (K in 12,24, same for all seeds)
            if K in (12, 24) and cache_key not in monthly_mean_cache:
                monthly_mean_cache[cache_key] = fit_monthly_mean(target_region, K, splits_by_key, region_mask_int)
            predictor_monthly = monthly_mean_cache.get(cache_key) if K in (12, 24) else None

            # Fit ridge methods (K>0, same for all seeds)
            if K > 0:
                for feat_set in ("core", "input_full", "input_geo_full"):
                    ridge_key = (target_region, K, feat_set)
                    if ridge_key not in ridge_cache:
                        ridge_cache[ridge_key] = fit_ridge(target_region, K, splits_by_key, region_mask_int, feat_set, geo_available)
                predictor_ridge_core = ridge_cache.get((target_region, K, "core"))
                predictor_ridge_full = ridge_cache.get((target_region, K, "input_full"))
                predictor_ridge_geo = ridge_cache.get((target_region, K, "input_geo_full"))
            else:
                predictor_ridge_core = predictor_ridge_full = predictor_ridge_geo = None

            # Evaluate for each seed
            for seed in SEEDS:
                # source_mean
                evaluate_on_query(predictor_source, "source_mean_increment",
                                 target_region, K, seed, splits_by_key, region_mask_int,
                                 all_results["source_mean_increment"])
                eval_count += 1

                # target_mean
                if K > 0 and predictor_target is not None:
                    evaluate_on_query(predictor_target, "target_support_mean_increment",
                                     target_region, K, seed, splits_by_key, region_mask_int,
                                     all_results["target_support_mean_increment"])
                    eval_count += 1

                # monthly_mean
                if K in (12, 24) and predictor_monthly is not None:
                    evaluate_on_query(predictor_monthly, "target_monthly_support_increment",
                                     target_region, K, seed, splits_by_key, region_mask_int,
                                     all_results["target_monthly_support_increment"])
                    eval_count += 1

                # ridge methods
                if K > 0:
                    if predictor_ridge_core is not None:
                        evaluate_on_query(predictor_ridge_core, "ridge_core",
                                         target_region, K, seed, splits_by_key, region_mask_int,
                                         all_results["ridge_core"])
                        eval_count += 1
                    if predictor_ridge_full is not None:
                        evaluate_on_query(predictor_ridge_full, "ridge_input_full",
                                         target_region, K, seed, splits_by_key, region_mask_int,
                                         all_results["ridge_input_full"])
                        eval_count += 1
                    if predictor_ridge_geo is not None:
                        evaluate_on_query(predictor_ridge_geo, "ridge_input_geo_full",
                                         target_region, K, seed, splits_by_key, region_mask_int,
                                         all_results["ridge_input_geo_full"])
                        eval_count += 1

                elapsed = time.time() - t_start
                rate = eval_count / elapsed if elapsed > 0 else 0
                remaining = (1080 - eval_count) / rate if rate > 0 else 0
                if eval_count % 100 == 0 or eval_count == 1080:
                    elapsed = time.time() - t_start
                    rate = eval_count / elapsed if elapsed > 0 else 0
                    remaining = (1080 - eval_count) / rate if rate > 0 else 0
                    _progress = f"  [{eval_count}/1080] elapsed={elapsed:.0f}s ETA={remaining:.0f}s\n"
                    sys.stdout.write(_progress)
                    sys.stdout.flush()

    elapsed_total = time.time() - t_start
    print(f"\nAll done in {elapsed_total:.0f}s")

    for method_name, results in all_results.items():
        if not results:
            continue
        df = pd.DataFrame(results)
        out_path = OUTPUT_DIR_BASE / f"phase3B_{method_name}_US" / "metrics_long.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    run_phase3B()