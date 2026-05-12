"""HydroDADataset for HydroDA-OOD / HyperDA V4.

This replacement keeps the existing lazy xarray design but makes the sample
contract explicit, adds month/season metadata, and avoids hidden modulo or
implicit split behavior.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    class Dataset:  # type: ignore
        pass


_ALL_US_REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
_SPLIT_TYPE_TO_DATES_KEY = {
    "source_train": "source_train_dates",
    "source_fit": "source_train_dates",
    "source_val": "source_val_dates",
    "target_support": "target_support_dates",
    "target_query": "target_query_dates",
}


def _month_and_season(date_str: str) -> tuple[int, str]:
    month = int(date_str[5:7]) if date_str and len(date_str) >= 7 else 1
    if month in (12, 1, 2):
        return month, "DJF"
    if month in (3, 4, 5):
        return month, "MAM"
    if month in (6, 7, 8):
        return month, "JJA"
    return month, "SON"


class HydroDADataset(Dataset):
    """Lazy-loading dataset for DA increment emulation.

    Expected NetCDF fields are still audited externally. The default mapping is:
    ``input[time, channel, y, x]`` and ``target[time, channel, y, x]``.
    """

    def __init__(
        self,
        da_nc_path: str,
        region_masks_nc: str,
        splits_json: str,
        target_region: str,
        split_type: str,
        K: int,
        seed: int,
        freeze_manifest: Optional[str] = "artifacts/protocol/US_region_split_freeze_manifest.json",
        input_var: str = "input",
        target_var: str = "target",
        base_valid_mask_channel: int = 11,
        forecast_surface_channel: int = 0,
        forecast_rootzone_channel: int = 1,
        analysis_surface_channel: int = 0,
        analysis_rootzone_channel: int = 1,
    ) -> None:
        if split_type not in _SPLIT_TYPE_TO_DATES_KEY:
            raise ValueError(f"Unknown split_type={split_type!r}")
        if target_region not in _ALL_US_REGIONS:
            raise ValueError(f"Unsupported target_region={target_region!r} for US development subset")

        self.da_nc_path = da_nc_path
        self.region_masks_nc = region_masks_nc
        self.splits_json = splits_json
        self.target_region = target_region
        self.split_type = split_type
        self.K = int(K)
        self.seed = int(seed)
        self.input_var = input_var
        self.target_var = target_var
        self.base_valid_mask_channel = int(base_valid_mask_channel)
        self.forecast_surface_channel = int(forecast_surface_channel)
        self.forecast_rootzone_channel = int(forecast_rootzone_channel)
        self.analysis_surface_channel = int(analysis_surface_channel)
        self.analysis_rootzone_channel = int(analysis_rootzone_channel)

        self.regime_id = self._load_regime_id(freeze_manifest)
        self._split_entry = self._load_split_entry()
        self._active_region_ids = (
            [r for r in _ALL_US_REGIONS if r != target_region]
            if split_type in ("source_train", "source_fit", "source_val")
            else [target_region]
        )

        date_key = _SPLIT_TYPE_TO_DATES_KEY[split_type]
        all_date_records = list(self._split_entry[date_key])

        # Safety: source_fit must only use 2015-2020 (already enforced by manifest)
        if split_type == "source_fit":
            all_date_records = [
                d for d in all_date_records
                if len(d.get("date_str", "")) >= 4 and 2015 <= int(d["date_str"][:4]) <= 2020
            ]

        self._date_records = all_date_records
        self._time_indices = [int(d["time_index"]) for d in self._date_records]
        self._date_str_map = {int(d["time_index"]): d.get("date_str", "") for d in self._date_records}

        # Validate: source_fit and source_val must have at least one date
        if split_type in ("source_train", "source_fit", "source_val") and len(self._date_records) == 0:
            raise ValueError(
                f"HydroDADataset: {split_type} has zero dates. "
                f"Manifest {date_key} is empty for target={target_region}, K={K}, seed={seed}. "
                f"Check split generation (source_val_dates must be populated from 2021 dates)."
            )

        region_ds = xr.open_dataset(region_masks_nc)
        try:
            self._region_mask_int = region_ds["region_mask_integer"].values.astype(np.int16)
        finally:
            region_ds.close()

        rnum_list = [int(rid.split("-R")[1]) for rid in self._active_region_ids]
        self._active_region_mask = np.isin(self._region_mask_int, rnum_list).astype(np.float32)
        self._da_ds = xr.open_dataset(da_nc_path, chunks={"time": 100})

    def _load_regime_id(self, freeze_manifest: Optional[str]) -> str:
        if not freeze_manifest or not Path(freeze_manifest).exists():
            return self.target_region.split("-")[1]
        with open(freeze_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        region_stats_path = manifest.get("artifacts", {}).get("region_stats")
        if not region_stats_path or not Path(region_stats_path).exists():
            return self.target_region.split("-")[1]
        with open(region_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return str(stats.get(self.target_region, {}).get("regime", self.target_region.split("-")[1]))

    def _load_split_entry(self) -> Dict[str, Any]:
        with open(self.splits_json, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        for entry in splits_data.get("splits", []):
            if (
                entry.get("target_region_id") == self.target_region
                and int(entry.get("K")) == self.K
                and int(entry.get("seed")) == self.seed
            ):
                return entry
        raise KeyError(f"No split entry for target={self.target_region}, K={self.K}, seed={self.seed}")

    def __len__(self) -> int:
        return len(self._time_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        time_index = self._time_indices[idx]

        input_arr = self._da_ds[self.input_var].isel(time=time_index).values.astype(np.float32)
        target_arr = self._da_ds[self.target_var].isel(time=time_index).values.astype(np.float32)

        forecast_surface = input_arr[self.forecast_surface_channel]
        forecast_rootzone = input_arr[self.forecast_rootzone_channel]
        analysis_surface = target_arr[self.analysis_surface_channel]
        analysis_rootzone = target_arr[self.analysis_rootzone_channel]
        increment_surface = (analysis_surface - forecast_surface).astype(np.float32)
        increment_rootzone = (analysis_rootzone - forecast_rootzone).astype(np.float32)

        # base_valid_mask: channel 11 — SMAP observation availability (diagnostic only)
        base_valid_mask = (input_arr[self.base_valid_mask_channel] > 0.5).astype(np.float32)

        # label_valid_mask: all 6 SM fields must be finite — primary quality gate
        label_valid_mask = (
            np.isfinite(forecast_surface)
            & np.isfinite(forecast_rootzone)
            & np.isfinite(analysis_surface)
            & np.isfinite(analysis_rootzone)
            & np.isfinite(increment_surface)
            & np.isfinite(increment_rootzone)
        ).astype(np.float32)

        # region_mask: pixels in active region(s)
        region_mask = (self._active_region_mask > 0.5).astype(np.float32)

        # loss_mask: for training compatibility (still requires obs + finiteness)
        loss_mask = (
            (self._active_region_mask > 0.5)
            & (base_valid_mask > 0.5)
            & np.isfinite(forecast_surface)
            & np.isfinite(forecast_rootzone)
            & np.isfinite(analysis_surface)
            & np.isfinite(analysis_rootzone)
        ).astype(np.float32)

        # metric_mask: for evaluation — region_mask & label_valid_mask ONLY
        # Channel 11 (obs_mask) is NOT included — SMAP coverage gaps do NOT block evaluation
        metric_mask = np.logical_and(region_mask, label_valid_mask).astype(np.float32)

        date_str = self._date_str_map.get(time_index, "")
        month, season = _month_and_season(date_str)
        split_id = f"{self.target_region}-K{self.K}-S{self.seed}-{self.split_type}"

        return {
            "x": input_arr,
            "forecast_surface": forecast_surface,
            "forecast_rootzone": forecast_rootzone,
            "analysis_surface": analysis_surface,
            "analysis_rootzone": analysis_rootzone,
            "increment_surface": increment_surface,
            "increment_rootzone": increment_rootzone,
            "base_valid_mask": base_valid_mask,
            "label_valid_mask": label_valid_mask,
            "region_mask_integer": self._region_mask_int.copy(),
            "active_region_mask": self._active_region_mask.copy(),
            "region_mask": region_mask,
            "loss_mask": loss_mask,
            "metric_mask": metric_mask,
            "date_str": date_str,
            "month": month,
            "season": season,
            "time_index": int(time_index),
            "country_id": "US",
            "target_region_id": self.target_region,
            "active_region_ids": list(self._active_region_ids),
            "split_role": self.split_type,
            "regime_id": self.regime_id,
            "split_id": split_id,
            "K": self.K,
            "seed": self.seed,
        }

    def close(self) -> None:
        self._da_ds.close()

    def preload(self) -> Dict[int, Dict[str, Any]]:
        return {idx: self[idx] for idx in range(len(self))}
