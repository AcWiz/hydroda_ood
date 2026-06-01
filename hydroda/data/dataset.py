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
    "source_test": "source_test_dates",
    "target_train": "target_train_dates",
    "target_adaptation": "target_adaptation_dates",
    "target_val": "target_val_dates",
    "target_eval": "target_eval_dates",
    # Deprecated aliases retained for old K-date artifacts.
    "target_support": "target_support_dates",
    "target_query": "target_query_dates",
}

_DATES_KEY_FALLBACKS = {
    "target_train_dates": "target_support_dates",
    "target_adaptation_dates": "target_train_dates",
    # Existing frozen US manifests predate explicit target_val_dates. Calendar
    # records are shared with source_val_dates, while HydroDADataset's active
    # region mask makes this a held-out target-region validation split.
    "target_val_dates": "source_val_dates",
    "target_eval_dates": "target_query_dates",
    "target_support_dates": "target_train_dates",
    "target_query_dates": "target_eval_dates",
    "source_test_dates": "target_eval_dates",
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
        K: Optional[int],
        seed: int,
        adaptation_setting: Optional[str] = None,
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
        try:
            from hydroda.data.file_hash import compute_sha256

            self.split_manifest_sha256 = compute_sha256(splits_json) if Path(splits_json).exists() else ""
        except Exception:
            self.split_manifest_sha256 = ""
        self.target_region = target_region
        self.split_type = split_type
        self.K = int(K) if K is not None else None
        self.seed = int(seed)
        self.adaptation_setting = adaptation_setting
        self.input_var = input_var
        self.target_var = target_var
        self.base_valid_mask_channel = int(base_valid_mask_channel)
        self.forecast_surface_channel = int(forecast_surface_channel)
        self.forecast_rootzone_channel = int(forecast_rootzone_channel)
        self.analysis_surface_channel = int(analysis_surface_channel)
        self.analysis_rootzone_channel = int(analysis_rootzone_channel)

        self.regime_id = self._load_regime_id(freeze_manifest)
        self._split_entry = self._load_split_entry()
        if self.adaptation_setting is None:
            self.adaptation_setting = self._split_entry.get(
                "adaptation_setting",
                "target_full_train" if self.K is None else f"legacy_few_shot_k{self.K}",
            )
        self._active_region_ids = (
            [r for r in _ALL_US_REGIONS if r != target_region]
            if split_type in ("source_train", "source_fit", "source_val", "source_test")
            else [target_region]
        )

        date_key = _SPLIT_TYPE_TO_DATES_KEY[split_type]
        all_date_records = self._get_date_records(date_key)

        # Safety: source_fit must only use 2015-2021 (already enforced by manifest)
        if split_type == "source_fit":
            all_date_records = [
                d for d in all_date_records
                if len(d.get("date_str", "")) >= 4 and 2015 <= int(d["date_str"][:4]) <= 2021
            ]

        self._date_records = all_date_records
        self._time_indices = [int(d["time_index"]) for d in self._date_records]
        self._date_str_map = {int(d["time_index"]): d.get("date_str", "") for d in self._date_records}

        self._validate_split_dates()

        # Validate: source splits must have at least one date
        if split_type in ("source_train", "source_fit", "source_val", "source_test") and len(self._date_records) == 0:
            raise ValueError(
                f"HydroDADataset: {split_type} has zero dates. "
                f"Manifest {date_key} is empty for target={target_region}, "
                f"adaptation_setting={self.adaptation_setting}, K={K}, seed={seed}. "
                f"Check split generation (source_val_dates must be populated from 2022 dates)."
            )

        region_ds = xr.open_dataset(region_masks_nc)
        try:
            self._region_mask_int = region_ds["region_mask_integer"].values.astype(np.int16)
            # Latitude weight for cos(lat) area weighting (WeatherBench2 convention)
            if "latitude" in region_ds:
                self._latitude = region_ds["latitude"].values.astype(np.float32)
            else:
                self._latitude = None
        finally:
            region_ds.close()

        # Fallback: load latitude from geolocation artifact if not in region masks
        if self._latitude is None:
            latlon_path = Path("artifacts/geolocation/US_latlon.nc")
            if latlon_path.exists():
                lat_ds = xr.open_dataset(latlon_path)
                try:
                    self._latitude = lat_ds["latitude"].values.astype(np.float32)
                finally:
                    lat_ds.close()

        if self._latitude is not None:
            lat_rad = np.deg2rad(self._latitude.astype(np.float64))
            self._latitude_weight = np.cos(lat_rad).clip(min=0.0).astype(np.float32)
        else:
            # No latitude available — uniform weights
            self._latitude_weight = np.ones(self._region_mask_int.shape, dtype=np.float32)

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
            if entry.get("target_region_id") != self.target_region:
                continue
            if int(entry.get("seed", 0)) != self.seed:
                continue
            if self.adaptation_setting is not None:
                if entry.get("adaptation_setting") == self.adaptation_setting:
                    return entry
                continue
            if self.K is None:
                if entry.get("K") is None and entry.get("adaptation_setting", "target_full_train") == "target_full_train":
                    return entry
                continue
            entry_k = entry.get("K", entry.get("K_legacy"))
            if entry_k is not None and int(entry_k) == self.K:
                return entry
        raise KeyError(
            f"No split entry for target={self.target_region}, "
            f"adaptation_setting={self.adaptation_setting}, K={self.K}, seed={self.seed}"
        )

    def _get_date_records(self, date_key: str) -> List[Dict[str, Any]]:
        seen = set()
        current = date_key
        while current and current not in seen:
            seen.add(current)
            if current in self._split_entry:
                return list(self._split_entry[current])
            current = _DATES_KEY_FALLBACKS.get(current, "")
        return []

    def _validate_split_dates(self) -> None:
        dates = [d.get("date_str", "") for d in self._date_records if d.get("date_str")]
        if not dates:
            return
        from hydroda.data.leakage_guard import LeakageGuard
        from hydroda.data.protocol import ProtocolConfig

        guard = LeakageGuard(ProtocolConfig())
        if self.split_type in ("target_query", "target_eval"):
            guard.check_query_evaluation_only(dates)
        elif self.split_type in ("target_support", "target_train", "target_adaptation"):
            guard.check_target_adaptation_scope(
                dates,
                purpose="target_adaptation",
                labels_allowed=True,
            )
        elif self.split_type == "target_val":
            guard.protocol.assert_dates_within(dates, ["target_val"], "target_adaptation_validation")
        elif self.split_type == "source_val":
            guard.check_model_selection_scope(dates, purpose="source_val_dataset")
        elif self.split_type in ("source_train", "source_fit"):
            guard.protocol.assert_dates_within(dates, ["source_fit"], "source_training")
        elif self.split_type == "source_test":
            guard.protocol.assert_dates_within(dates, ["target_eval"], "source_test_dataset")

    def __len__(self) -> int:
        return len(self._time_indices)

    def _resolve_sample_region_id(self, label_valid_mask: np.ndarray) -> str:
        """Resolve the dominant region ID for a sample.

        Finds which region has the most valid (finite, labeled) pixels
        in this spatial sample by intersecting label_valid_mask with region_mask_integer.
        Excludes region 0 (ocean/invalid) from dominance calculation.
        """
        valid_region_ids = self._region_mask_int[label_valid_mask > 0.5]
        if len(valid_region_ids) == 0:
            return self.target_region
        unique, counts = np.unique(valid_region_ids, return_counts=True)
        # Exclude region 0 (ocean/invalid) from dominance calculation
        nonzero_mask = unique != 0
        nonzero_unique = unique[nonzero_mask]
        nonzero_counts = counts[nonzero_mask]
        if len(nonzero_unique) == 0:
            return self.target_region
        dominant = int(nonzero_unique[np.argmax(nonzero_counts)])
        return f"US-R{dominant}"

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

        # loss_mask = metric_mask: training and evaluation use the same mask.
        # Channel 11 (base_valid_mask) is NOT required in training — SMAP coverage
        # gaps do not indicate missing input features and would cause a catastrophic
        # train-eval distribution mismatch (see FINAL_SOURCE_ONLY_DIAGNOSIS.md).
        loss_mask = np.logical_and(region_mask, label_valid_mask).astype(np.float32)
        metric_mask = loss_mask

        # Resolve sample_region_id from the same active, labeled pixels used by
        # training/evaluation so single-region source_test passes route to the
        # matching source prompt.
        sample_region_id = self._resolve_sample_region_id(metric_mask)

        date_str = self._date_str_map.get(time_index, "")
        month, season = _month_and_season(date_str)
        if self.K is None:
            split_id = f"{self.target_region}-{self.adaptation_setting}-S{self.seed}-{self.split_type}"
        else:
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
            "region_mask_integer": self._region_mask_int,
            "active_region_mask": self._active_region_mask,
            "region_mask": region_mask,
            "loss_mask": loss_mask,
            "metric_mask": metric_mask,
            "latitude": self._latitude,
            "latitude_weight": self._latitude_weight,
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
            "sample_region_id": sample_region_id,
            "K": self.K,
            "K_legacy": self.K if str(self.adaptation_setting).startswith("legacy_few_shot") else None,
            "adaptation_setting": self.adaptation_setting,
            "target_train_dates_hash": self._split_entry.get("target_train_dates_hash", self._split_entry.get("support_dates_hash", "")),
            "target_eval_dates_hash": self._split_entry.get("target_eval_dates_hash", self._split_entry.get("target_query_dates_hash", "")),
            "support_dates_hash": self._split_entry.get("support_dates_hash", ""),
            "split_manifest_sha256": self._split_entry.get("split_manifest_sha256", "") or self.split_manifest_sha256,
            "seed": self.seed,
        }

    def set_active_region(self, region_id: str) -> None:
        """Restrict active region mask to a single region's pixels."""
        self._active_region_ids = [region_id]
        rnum = int(region_id.split("-R")[1])
        self._active_region_mask = (self._region_mask_int == rnum).astype(np.float32)

    def set_active_all_regions(self) -> None:
        """Set active region mask to all US regions."""
        self._active_region_ids = list(_ALL_US_REGIONS)
        rnum_list = [int(rid.split("-R")[1]) for rid in _ALL_US_REGIONS]
        self._active_region_mask = np.isin(self._region_mask_int, rnum_list).astype(np.float32)

    def close(self) -> None:
        self._da_ds.close()

    def preload(self) -> Dict[int, Dict[str, Any]]:
        return {idx: self[idx] for idx in range(len(self))}
