#!/usr/bin/env python3
"""Phase 3B — source_mean_increment only re-run (K=0, seed=0, 6 US regions).

Post-fix re-run: increment metrics now correctly computed in increment-space
(previously were accidentally in analysis-space due to metric routing bug).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Allow importing from scripts/ as a package
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import xarray as xr

from scripts.run_phase3B import (
    fit_source_mean,
    evaluate_on_query,
    SPLITS_JSON,
    REGION_MASKS_NC,
    OUTPUT_DIR_BASE,
    REGIONS,
    FREEZE_ID,
    _ALL_REGIONS,
)

OUTPUT_DIR = OUTPUT_DIR_BASE / "phase3B_source_mean_increment_US"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("Phase 3B source_mean_increment re-run")
    print(f"  K=0, seed=0, regions={REGIONS}")
    print(f"  split_role=target_query (2023-2025)")
    print(f"  freeze_id={FREEZE_ID}")
    print("=" * 60)

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

    all_results = []
    t_start = time.time()

    for target_region in REGIONS:
        print(f"\n--- {target_region} ---", flush=True)

        # Fit source_mean predictor (K=0, uses source_train from all other regions)
        print(f"  Fitting source_mean...", flush=True)
        predictor = fit_source_mean(
            target_region, K=0, splits_by_key=splits_by_key,
            region_stats=region_stats, region_mask_int=region_mask_int,
        )

        # Evaluate on target_query (seed=0)
        print(f"  Evaluating on target_query...", flush=True)
        evaluate_on_query(
            predictor, "source_mean_increment",
            target_region, K=0, seed=0,
            splits_by_key=splits_by_key,
            region_mask_int=region_mask_int,
            result_list=all_results,
        )

    elapsed = time.time() - t_start
    print(f"\nAll 6 regions done in {elapsed:.0f}s", flush=True)

    # Save results
    df = pd.DataFrame(all_results)
    out_path = OUTPUT_DIR / "metrics_long.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)", flush=True)

    # Quick summary
    print("\n--- Quick Summary ---")
    for var in ["surface", "rootzone"]:
        sub = df[df["variable"] == var]
        for metric in ["increment_rmse", "analysis_rmse", "sign_accuracy_deadzone"]:
            if metric in sub["metric"].values:
                vals = sub[sub["metric"] == metric]["value"]
                print(f"  {var}/{metric}: mean={vals.mean():.6f}, std={vals.std():.6f}")

    return df


if __name__ == "__main__":
    df = main()
