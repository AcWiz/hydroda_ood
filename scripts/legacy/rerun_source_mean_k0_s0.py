#!/usr/bin/env python3
"""Minimal re-run: source_mean_increment for K=0, seed=0, 6 US regions.

Uses existing fit/evaluate functions from run_phase3B.
Saves to artifacts/metrics/phase3B_source_mean_increment_US/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_phase3B import (
    fit_source_mean,
    evaluate_on_query,
    SPLITS_JSON,
    REGION_MASKS_NC,
    REGIONS,
)

OUTPUT_DIR = Path("artifacts/metrics/phase3B_source_mean_increment_US")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K = 0
SEED = 0

# Load splits
with open(SPLITS_JSON) as f:
    splits_data = json.load(f)
splits_by_key = {
    (s["target_region_id"], s["K"], s["seed"]): s
    for s in splits_data["splits"]
}

# Load region stats (needed for fit_source_mean signature, but not used inside for source_mean)
with open("artifacts/regions/US_region_stats.json") as f:
    region_stats = json.load(f)

# Load region mask
region_ds = xr.open_dataset(REGION_MASKS_NC)
region_mask_int = region_ds["region_mask_integer"].values.astype(np.int16)
region_ds.close()

all_results = []
t_start = time.time()

for target_region in REGIONS:
    print(f"\n--- {target_region} K={K} seed={SEED} ---", flush=True)

    # Fit source_mean predictor
    predictor = fit_source_mean(target_region, K, splits_by_key, region_stats, region_mask_int)

    # Evaluate on target_query
    evaluate_on_query(
        predictor, "source_mean_increment",
        target_region, K, SEED,
        splits_by_key, region_mask_int,
        all_results,
    )
    print(f"  Done, total samples so far: {len(all_results)}", flush=True)

elapsed = time.time() - t_start
print(f"\nTotal elapsed: {elapsed:.0f}s")

# Save
df = pd.DataFrame(all_results)
out_path = OUTPUT_DIR / "metrics_long.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {out_path} ({len(df)} rows)")

# Quick verification
print("\n=== Verification ===")
for region in REGIONS:
    region_df = df[df["target_region_id"] == region]
    if len(region_df) == 0:
        print(f"  {region}: NO DATA")
        continue
    # Check increment_rmse vs analysis_rmse for surface
    surf = region_df[(region_df["variable"] == "surface")]
    inc_rmse = surf[surf["metric"] == "increment_rmse"]["value"]
    ana_rmse = surf[surf["metric"] == "analysis_rmse"]["value"]
    sign_acc = surf[surf["metric"] == "sign_accuracy_deadzone"]["value"]

    print(f"  {region}:")
    print(f"    increment_rmse mean={inc_rmse.mean():.6f}  (NaN={inc_rmse.isna().sum()})")
    print(f"    analysis_rmse  mean={ana_rmse.mean():.6f}  (NaN={ana_rmse.isna().sum()})")
    print(f"    sign_accuracy  mean={sign_acc.mean():.4f}  (NaN={sign_acc.isna().sum()})")
    print(f"    n_rows={len(region_df)}")
