#!/usr/bin/env python3
"""Build US Leave-One-Region-Out K-Date Splits.

Usage:
    python scripts/data/build_kdate_splits.py \
        --da-nc /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc \
        --region-masks artifacts/regions/US_region_masks.nc \
        --out-json artifacts/splits/US_loro_kdate_splits.json \
        --out-md reports/splits/US_loro_kdate_split_summary.md

No-leakage declaration:
    Support dates are selected ONLY via:
    - Calendar constraints (quarter/month/half-month rules)
    - Time availability in 2022
    - base_valid_mask coverage threshold (via input channel 11)

    NOT via:
    - Analysis increment values
    - Model errors
    - Target query label distribution
    - Future query statistics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr

from hydroda.splits.kdate import (
    dates_to_serializable,
    get_support_dates_for_K,
)
from hydroda.splits.manifest import (
    create_split_manifest,
    generate_split_summary_markdown,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build US LORO K-date splits")
    parser.add_argument("--da-nc", required=True, help="Path to DA.nc SMAP data")
    parser.add_argument("--region-masks", required=True, help="Path to US_region_masks.nc")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument("--out-md", required=True, help="Output markdown path")
    parser.add_argument("--k-values", nargs="+", default=[0, 4, 12], type=int)
    parser.add_argument("--seeds", nargs="+", default=[0, 1, 2], type=int)
    parser.add_argument("--min-coverage", default=0.5, type=float)
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)

    # Load DA.nc
    print(f"Loading DA.nc: {args.da_nc}")
    ds = xr.open_dataset(args.da_nc, decode_times=False)

    # Pre-compute time metadata
    time_vals = ds['time'].values.astype(np.int64)
    n_cycles = len(time_vals)
    print(f"  Total cycles: {n_cycles}")

    # Pre-index by year range to avoid scanning all cycles
    years = np.array([datetime.fromtimestamp(t).year for t in time_vals])
    source_mask = (years >= 2015) & (years <= 2020)
    val_mask = years == 2021
    support_mask = years == 2022
    query_mask = (years >= 2023) & (years <= 2025)

    print(f"  Source train (2015-2020): {source_mask.sum()} cycles")
    print(f"  Source val (2021): {val_mask.sum()} cycles")
    print(f"  Target context/support (2022): {support_mask.sum()} cycles")
    print(f"  Target query (2023-2025): {query_mask.sum()} cycles")

    # Load region masks
    print(f"Loading region masks: {args.region_masks}")
    rm = xr.open_dataset(args.region_masks)
    region_ids = rm["region_id"].values.tolist()
    print(f"  Regions: {region_ids}")

    # Pre-load base_valid_mask (channel 11) into memory
    print("Pre-loading base_valid_mask (channel 11)...")
    base_valid = ds["input"][:, 11, :, :].values.astype(np.float32)  # (T, H, W)
    print(f"  base_valid shape: {base_valid.shape}")

    # Pre-compute region masks as boolean
    region_onehot = rm["region_mask_onehot"].values.astype(bool)  # (6, H, W)
    region_sizes = region_onehot.sum(axis=(1, 2))  # (6,)
    print(f"  Region sizes: {region_sizes}")

    # Pre-compute year-month-day for each cycle (cache dts once)
    dts = [datetime.fromtimestamp(t) for t in time_vals]

    # Pre-compute source_train_dates ONCE outside region loop.
    # It is the same for all 6 targets (union of all source regions).
    # This avoids scanning source cycles 6 times redundantly.
    print("Pre-computing source_train_dates for all source regions...")
    all_source_indices = np.where(source_mask)[0]
    source_dates_all = [
        (int(idx), dts[idx])
        for idx in all_source_indices
        if np.isfinite(base_valid[idx]).sum() > 0
    ]
    print(f"  Total source train cycles (all regions): {len(source_dates_all)}")

    # Pre-compute source_val_dates ONCE outside region loop (2021 only).
    print("Pre-computing source_val_dates for all source regions...")
    all_val_indices = np.where(val_mask)[0]
    val_dates_all = [
        (int(idx), dts[idx])
        for idx in all_val_indices
        if np.isfinite(base_valid[idx]).sum() > 0
    ]
    print(f"  Total source val cycles (all regions): {len(val_dates_all)}")

    # Helper: get target dates for a region
    def get_target_dates(region_idx, target_mask, require_valid=False):
        region_mask_3d = region_onehot[region_idx]
        region_size = region_sizes[region_idx]
        dates = []
        for idx in np.where(target_mask)[0]:
            bv = base_valid[idx]
            if require_valid:
                if region_size == 0 or np.isfinite(bv).sum() == 0:
                    continue
                valid = (region_mask_3d & (bv > 0)).sum()
                if valid == 0:
                    continue
            dates.append((idx, dts[idx]))
        return dates

    # Helper: compute validity mask for support dates
    def compute_valid_support_mask(region_idx, available_dates):
        region_mask_3d = region_onehot[region_idx]
        region_size = region_sizes[region_idx]
        valid = np.zeros(len(available_dates), dtype=bool)
        for i, (idx, dt) in enumerate(available_dates):
            bv = base_valid[idx]
            valid_pixels = (region_mask_3d & (bv > 0)).sum()
            coverage = valid_pixels / region_size if region_size > 0 else 0.0
            valid[i] = coverage >= args.min_coverage
        return valid

    # Build all splits
    splits = []
    k_values = args.k_values
    seeds = args.seeds

    print(f"\nBuilding splits: K={k_values}, seeds={seeds}")

    for target_idx, target_region in enumerate(region_ids):
        print(f"\n=== {target_region} ===")

        # Source regions = all other regions
        source_region_ids = [r for r in region_ids if r != target_region]
        source_region_indices = [region_ids.index(r) for r in source_region_ids]

        # Get source train dates from pre-computed list (same for all targets)
        # Filter to those where the specific source region has valid pixels
        source_dates = []
        for src_idx in source_region_indices:
            region_mask_3d = region_onehot[src_idx]
            region_size = region_sizes[src_idx]
            if region_size == 0:
                continue
            for idx, dt in source_dates_all:
                bv = base_valid[idx]
                valid = (region_mask_3d & (bv > 0)).sum()
                if valid > 0:
                    source_dates.append((idx, dt))
        print(f"  Source train cycles: {len(source_dates)}")

        # Get source val dates from pre-computed val_dates_all (2021 only)
        val_dates = []
        for src_idx in source_region_indices:
            region_mask_3d = region_onehot[src_idx]
            region_size = region_sizes[src_idx]
            if region_size == 0:
                continue
            for idx, dt in val_dates_all:
                bv = base_valid[idx]
                valid = (region_mask_3d & (bv > 0)).sum()
                if valid > 0:
                    val_dates.append((idx, dt))
        print(f"  Source val cycles (2021): {len(val_dates)}")

        # Get available support dates in 2022
        support_available = get_target_dates(target_idx, support_mask, require_valid=False)
        print(f"  Available support dates in 2022: {len(support_available)}")

        # Compute validity mask
        valid_mask = compute_valid_support_mask(target_idx, support_available)
        n_valid = valid_mask.sum()
        print(f"  Valid support dates (coverage >= {args.min_coverage}): {n_valid}")

        # Get target query dates
        query_dates = get_target_dates(target_idx, query_mask, require_valid=False)
        print(f"  Target query cycles: {len(query_dates)}")

        # Generate splits
        for K in k_values:
            for seed in seeds:
                support_selected = get_support_dates_for_K(
                    support_available, valid_mask, K, seed
                )

                manifest = create_split_manifest(
                    target_region=target_region,
                    source_regions=source_region_ids,
                    K=K,
                    seed=seed,
                    source_train_dates=dates_to_serializable(source_dates),
                    source_val_dates=dates_to_serializable(val_dates),
                    support_dates=dates_to_serializable(support_selected),
                    query_dates=dates_to_serializable(query_dates),
                )

                splits.append(manifest)
                print(f"  K={K}, seed={seed}: {len(support_selected)} support, {len(query_dates)} query")

    print(f"\nTotal splits: {len(splits)}")

    # Save JSON
    with open(args.out_json, "w") as f:
        json.dump({"splits": splits}, f, indent=2)
    print(f"Saved: {args.out_json}")

    # Save markdown summary
    generate_split_summary_markdown(splits, args.out_md)

    ds.close()
    rm.close()

    print("\nDone!")


if __name__ == "__main__":
    main()