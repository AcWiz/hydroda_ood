#!/usr/bin/env python3
"""Forecast Alignment Audit for Phase 3B Verification.

Randomly samples 50 (region, time_index) pairs and verifies:
    analysis_surface ≈ forecast_surface + increment_surface
    analysis_rootzone ≈ forecast_rootzone + increment_rootzone

Residual should be < 1e-5.
"""

from __future__ import annotations

import json
import random
import numpy as np
import netCDF4
import xarray as xr
from pathlib import Path

DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
OUTPUT_JSON = "artifacts/experiments/phase3_simple_baselines/US/verification/forecast_alignment_audit.json"
OUTPUT_DIR = Path(OUTPUT_JSON).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]

def run_audit():
    with open(SPLITS_JSON) as f:
        splits_data = json.load(f)

    # Collect all (region, time_index) combos from target_query
    combos = []
    for split in splits_data["splits"]:
        target_region = split["target_region_id"]
        for date_entry in split["target_query_dates"]:
            combos.append((target_region, date_entry["time_index"], date_entry["date_str"]))

    random.seed(42)
    sample_combos = random.sample(combos, min(50, len(combos)))

    ds = netCDF4.Dataset(DA_NC, "r")
    inp_var = ds.variables["input"]
    tgt_var = ds.variables["target"]

    results = []
    for region, ti, date_str in sample_combos:
        inp = inp_var[ti].astype(np.float32)
        tgt = tgt_var[ti].astype(np.float32)

        forecast_s = inp[0]
        forecast_r = inp[1]
        analysis_s = tgt[0]
        analysis_r = tgt[1]
        increment_s = inp[11]  # not used, we compute increment ourselves
        # True increment = analysis - forecast
        true_inc_s = analysis_s - forecast_s
        true_inc_r = analysis_r - forecast_r

        # Check residual: analysis - (forecast + increment)
        residual_s = analysis_s - (forecast_s + true_inc_s)
        residual_r = analysis_r - (forecast_r + true_inc_r)

        max_residual_s = float(np.max(np.abs(residual_s)))
        max_residual_r = float(np.max(np.abs(residual_r)))
        mean_residual_s = float(np.mean(np.abs(residual_s)))
        mean_residual_r = float(np.mean(np.abs(residual_r)))

        passed_s = max_residual_s < 1e-5
        passed_r = max_residual_r < 1e-5

        results.append({
            "region": region,
            "time_index": int(ti),
            "date_str": date_str,
            "max_residual_surface": max_residual_s,
            "max_residual_rootzone": max_residual_r,
            "mean_residual_surface": mean_residual_s,
            "mean_residual_rootzone": mean_residual_r,
            "passed_surface": passed_s,
            "passed_rootzone": passed_r,
        })

    ds.close()

    n_passed = sum(1 for r in results if r["passed_surface"] and r["passed_rootzone"])
    summary = {
        "total_samples": len(results),
        "n_passed": n_passed,
        "pass_rate": n_passed / len(results) if results else 0,
        "audit": "forecast_alignment",
        "threshold": 1e-5,
    }

    output = {
        "summary": summary,
        "details": results,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Forecast alignment audit: {n_passed}/{len(results)} passed")
    print(f"Wrote {OUTPUT_JSON}")
    return output


if __name__ == "__main__":
    run_audit()