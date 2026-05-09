"""HydroDADataset — Lazy-loading PyTorch Dataset for HydroDA-OOD.

No-leakage declaration:
    - Region masks from artifacts/regions/US_region_masks.nc (fixed bbox, not from model results)
    - Split selection uses only calendar rules and base_valid_mask coverage (no analysis increments, no query labels)
    - No normalization by default; normalization stats must be a separate artifact builder
    - target_query statistics are NEVER used in dataset normalization
"""

from __future__ import annotations

import json
import numpy as np
import xarray as xr
from datetime import datetime
from typing import List, Dict, Any


class HydroDADataset:
    """Lazy-loading Dataset for HydroDA-OOD cross-region DA increment emulation.

    Loads from:
    - DA.nc: input[T,12,H,W], target[T,4,H,W], time[seconds since 1970-01-01]
    - US_region_masks.nc: region_mask_integer[H,W] (0..6)
    - US_loro_kdate_splits.json: 240 LORO splits keyed by (target_region, K, seed)

    Args:
        da_nc_path: Path to /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc
        region_masks_nc: Path to artifacts/regions/US_region_masks.nc
        splits_json: Path to artifacts/splits/US_loro_kdate_splits.json
        target_region: Target region ID (e.g., "US-R1".."US-R6")
        split_type: "source_train" | "target_support" | "target_query"
        K: Number of support dates (0, 4, 12, 24)
        seed: Random seed (0..9)
        freeze_manifest: Path to artifacts/protocol/US_region_split_freeze_manifest.json

    Sample dict keys:
        x: np.float32[12, H, W] — all 12 raw input channels (raw passthrough)
        forecast_surface: np.float32[H, W] — input ch 0
        forecast_rootzone: np.float32[H, W] — input ch 1
        analysis_surface: np.float32[H, W] — target ch 0
        analysis_rootzone: np.float32[H, W] — target ch 1
        increment_surface: np.float32[H, W] — analysis_surface - forecast_surface
        increment_rootzone: np.float32[H, W] — analysis_rootzone - forecast_rootzone
        base_valid_mask: np.float32[H, W] — input ch 11 → binary via (raw > 0.5)
        region_mask_integer: np.int16[H, W] — 0..6 from artifact, static
        active_region_mask: np.float32[H, W] — 0/1 from active_region_ids union
        loss_mask: np.float32[H, W] — active_region_mask AND base_valid_mask AND finite
        metric_mask: np.float32[H, W] — same as loss_mask
        date_str: str — "YYYY-MM-DD"
        time_index: int — exact time index
        country_id: str — "US"
        target_region_id: str — "US-R1" etc
        active_region_ids: List[str] — source: 5 regions; target: [target_region]
        split_role: str — "source_train" | "target_support" | "target_query"
        regime_id: str — regime of target_region
        split_id: str — e.g., "US-R1-K4-S0-source_train"
        K: int, seed: int
    """

    _ALL_REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]

    _SPLIT_TYPE_TO_DATES_KEY = {
        "source_train": "source_train_dates",
        "target_support": "target_support_dates",
        "target_query": "target_query_dates",
    }

    def __init__(
        self,
        da_nc_path: str,
        region_masks_nc: str,
        splits_json: str,
        target_region: str,
        split_type: str,
        K: int,
        seed: int,
        freeze_manifest: str = "artifacts/protocol/US_region_split_freeze_manifest.json",
    ) -> None:
        self.target_region = target_region
        self.split_type = split_type
        self.K = K
        self.seed = seed

        # Load freeze manifest for regime_id lookup
        with open(freeze_manifest, "r") as f:
            manifest = json.load(f)
        # Extract regime_id — no need to store full _region_stats
        region_stats_path = manifest["artifacts"]["region_stats"]
        with open(region_stats_path, "r") as f:
            region_stats = json.load(f)
        self.regime_id = region_stats[self.target_region]["regime"]

        # Load splits JSON
        with open(splits_json, "r") as f:
            splits_data = json.load(f)
        splits = splits_data["splits"]

        # Find matching split entry
        self._split_entry = next(
            s for s in splits
            if s["target_region_id"] == target_region
            and s["K"] == K
            and s["seed"] == seed
        )

        # Determine active_region_ids
        if split_type == "source_train":
            # All regions EXCEPT target → 5 source regions
            self._active_region_ids = [r for r in self._ALL_REGIONS if r != target_region]
        else:
            # target_support or target_query → held-out target only
            self._active_region_ids = [target_region]

        # Select date list via dict lookup
        if split_type not in self._SPLIT_TYPE_TO_DATES_KEY:
            raise ValueError(f"Unknown split_type: {split_type!r}")
        date_list = self._split_entry[self._SPLIT_TYPE_TO_DATES_KEY[split_type]]

        # Extract exact time_indices as list of integers
        self._time_indices = [d["time_index"] for d in date_list]

        # Load region_mask_integer[H,W] from artifact
        region_ds = xr.open_dataset(region_masks_nc)
        self._region_mask_int = region_ds["region_mask_integer"].values.astype(np.int16)
        region_ds.close()
        H, W = self._region_mask_int.shape

        # Construct active_region_mask[H,W] via vectorized isin (regions are non-overlapping)
        rnum_list = [int(rid.split("-R")[1]) for rid in self._active_region_ids]
        self._active_region_mask = np.isin(self._region_mask_int, rnum_list).astype(np.float32)

        # Open xarray dataset lazily (no full array preload)
        self._da_ds = xr.open_dataset(
            da_nc_path, chunks={"time": 100}
        )

        # Build date_str lookup from all date lists
        self._date_str_map: Dict[int, str] = {}
        for key in ("source_train_dates", "target_support_dates", "target_query_dates"):
            for d in self._split_entry[key]:
                self._date_str_map[d["time_index"]] = d["date_str"]

    def __len__(self) -> int:
        """Exact count of time indices — NO modulo wrap."""
        return len(self._time_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return sample dict for time index idx.

        Raises:
            IndexError: if idx >= len(time_indices) — NO modulo wrap.
        """
        if idx >= len(self._time_indices):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self._time_indices)}"
            )

        time_index = self._time_indices[idx]

        # Lazy load input[time, 12, H, W] and target[time, 4, H, W]
        input_arr = self._da_ds["input"].isel(time=time_index).values  # (12, H, W)
        target_arr = self._da_ds["target"].isel(time=time_index).values  # (4, H, W)

        forecast_surface = input_arr[0].astype(np.float32)    # (H, W)
        forecast_rootzone = input_arr[1].astype(np.float32)   # (H, W)
        analysis_surface = target_arr[0].astype(np.float32)   # (H, W)
        analysis_rootzone = target_arr[1].astype(np.float32)  # (H, W)

        # Increments
        increment_surface = (analysis_surface - forecast_surface).astype(np.float32)
        increment_rootzone = (analysis_rootzone - forecast_rootzone).astype(np.float32)

        # base_valid_mask: input channel 11 → binary via (raw > 0.5)
        raw_mask = input_arr[11]
        base_valid_mask = (raw_mask > 0.5).astype(np.float32)

        # loss_mask: active_region_mask AND base_valid_mask AND finite conditions
        loss_mask = (
            self._active_region_mask.astype(bool)
            & (base_valid_mask > 0.5)
            & np.isfinite(forecast_surface)
            & np.isfinite(forecast_rootzone)
            & np.isfinite(analysis_surface)
            & np.isfinite(analysis_rootzone)
        ).astype(np.float32)

        date_str = self._date_str_map.get(time_index, "unknown")

        split_id = (
            f"{self.target_region}-K{self.K}-S{self.seed}-{self.split_type}"
        )

        return {
            # Raw data
            "x": input_arr.astype(np.float32),
            "forecast_surface": forecast_surface,
            "forecast_rootzone": forecast_rootzone,
            "analysis_surface": analysis_surface,
            "analysis_rootzone": analysis_rootzone,
            "increment_surface": increment_surface,
            "increment_rootzone": increment_rootzone,
            # Masks
            "base_valid_mask": base_valid_mask,
            "region_mask_integer": self._region_mask_int.copy(),
            "active_region_mask": self._active_region_mask.copy(),
            "loss_mask": loss_mask,
            "metric_mask": loss_mask,  # alias: metric_mask and loss_mask are identical
            # Metadata
            "date_str": date_str,
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
        """Close the xarray dataset."""
        self._da_ds.close()

    def preload(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Pre-load all time indices into memory for fast repeated evaluation.

        Returns:
            Dict mapping time_index -> sample dict (with arrays, no xarray objects).
            The metric_mask and other derived fields are precomputed.

        Warning: For large datasets (4000+ samples), this uses significant RAM.
        """
        samples = {}
        for idx in range(len(self._time_indices)):
            samples[idx] = self[idx]
        return samples