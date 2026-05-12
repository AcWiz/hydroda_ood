#!/usr/bin/env python3
"""NaN and异常值Classification Audit for Phase 3B Verification.

Classifies NaN pixels as:
- expected_nan: base_valid_mask is False OR finite check fails
- unexpected_nan: other NaN (possible data problem)
"""

from __future__ import annotations

import json
import numpy as np
import netCDF4
import xarray as xr
import pandas as pd
from pathlib import Path

DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
OUTPUT_CSV = "artifacts/experiments/phase3_simple_baselines/US/verification/nan_classification_audit.csv"
OUTPUT_DIR = Path(OUTPUT_CSV).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
K_VALUES = [0, 4, 12, 24]


def run_audit():
    with open(SPLITS_JSON) as f:
        splits_data = json.load(f)

    splits_by_key = {
        (s["target_region_id"], s["K"], s["seed"]): s
        for s in splits_data["splits"]
    }

    region_ds = xr.open_dataset(REGION_MASKS_NC)
    region_mask_int = region_ds["region_mask_integer"].values.astype(np.int16)
    region_ds.close()

    rows = []
    ds = netCDF4.Dataset(DA_NC, "r")
    inp_var = ds.variables["input"]
    tgt_var = ds.variables["target"]

    sample_count = 0
    for region in REGIONS:
        for K in K_VALUES:
            split_entry = splits_by_key[(region, K, 0)]
            target_rnum = int(region.split("-R")[1])
            target_mask = np.isin(region_mask_int, [target_rnum]).astype(np.float32)

            date_list = split_entry["target_query_dates"]
            time_indices = [d["time_index"] for d in date_list[:20]]  # sample first 20

            for ti in time_indices:
                inp = inp_var[ti].astype(np.float32)
                tgt = tgt_var[ti].astype(np.float32)

                base_mask = (inp[11] > 0.5).astype(np.float32)
                finite_check = (
                    np.isfinite(inp[0]) & np.isfinite(inp[1])
                    & np.isfinite(tgt[0]) & np.isfinite(tgt[1])
                )

                # Compute where NaN occurs
                nan_forecast_s = np.isnan(inp[0])
                nan_forecast_r = np.isnan(inp[1])
                nan_analysis_s = np.isnan(tgt[0])
                nan_analysis_r = np.isnan(tgt[1])
                nan_any = nan_forecast_s | nan_forecast_r | nan_analysis_s | nan_analysis_r

                # Expected: base_mask is False or finite_check fails
                expected_nan = ~base_mask.astype(bool) | ~finite_check
                unexpected_nan = nan_any & ~expected_nan

                rows.append({
                    "region": region,
                    "K": K,
                    "time_index": int(ti),
                    "total_nan_forecast_s": int(nan_forecast_s.sum()),
                    "total_nan_forecast_r": int(nan_forecast_r.sum()),
                    "total_nan_analysis_s": int(nan_analysis_s.sum()),
                    "total_nan_analysis_r": int(nan_analysis_r.sum()),
                    "expected_nan_count": int(expected_nan.sum()),
                    "unexpected_nan_count": int(unexpected_nan.sum()),
                    "target_mask_pixels": int(target_mask.sum()),
                })

                sample_count += 1
                if sample_count >= 100:
                    break
            if sample_count >= 100:
                break
        if sample_count >= 100:
            break

    ds.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    total_unexpected = df["unexpected_nan_count"].sum()
    print(f"NaN classification audit: {total_unexpected} unexpected NaN pixels in {sample_count} samples")
    print(f"Wrote {OUTPUT_CSV}")
    return df


if __name__ == "__main__":
    run_audit()