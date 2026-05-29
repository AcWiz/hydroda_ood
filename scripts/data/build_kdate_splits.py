#!/usr/bin/env python3
"""Build US Leave-One-Region-Out historical target-adaptation splits.

This file is retained as a backward-compatible entrypoint. New main-protocol
workflows should call `scripts/data/build_target_train_splits.py`.

Usage:
    python scripts/data/build_target_train_splits.py \
        --da-nc /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc \
        --region-masks artifacts/regions/US_region_masks.nc \
        --out-json artifacts/splits/US_loro_target_train_splits.json \
        --out-md reports/splits/US_loro_target_train_split_summary.md

No-leakage declaration:
    Main target adaptation uses the full historical target training period:
    - Source fit / target train-adaptation: 2015-2021
    - Source validation / target validation: 2022
    - Target eval/query: 2023-2025

    Legacy few-shot support dates, if requested, are selected ONLY via:
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
    select_target_full_train_dates,
)
from hydroda.splits.manifest import (
    create_split_manifest,
    generate_split_summary_markdown,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build US LORO historical target-adaptation splits")
    parser.add_argument("--da-nc", required=True, help="Path to DA.nc SMAP data")
    parser.add_argument("--region-masks", required=True, help="Path to US_region_masks.nc")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument("--out-md", required=True, help="Output markdown path")
    parser.add_argument(
        "--adaptation-settings",
        nargs="+",
        default=["target_full_train"],
        choices=["target_full_train"],
        help="Main adaptation settings to generate (default: target_full_train)",
    )
    parser.add_argument(
        "--include-legacy-few-shot",
        action="store_true",
        help="Also generate legacy few-shot K-date ablation splits.",
    )
    parser.add_argument("--k-values", nargs="+", default=[0, 4, 12], type=int)
    parser.add_argument("--seeds", nargs="+", default=[0], type=int)
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
    source_mask = (years >= 2015) & (years <= 2021)
    val_mask = years == 2022
    target_train_mask = (years >= 2015) & (years <= 2021)
    query_mask = (years >= 2023) & (years <= 2025)

    print(f"  Source train (2015-2021): {source_mask.sum()} cycles")
    print(f"  Source val (2022): {val_mask.sum()} cycles")
    print(f"  Target train/adaptation (2015-2021): {target_train_mask.sum()} cycles")
    print(f"  Target eval/query (2023-2025): {query_mask.sum()} cycles")

    # Load region masks
    print(f"Loading region masks: {args.region_masks}")
    rm = xr.open_dataset(args.region_masks)
    region_ids = rm["region_id"].values.tolist()
    print(f"  Regions: {region_ids}")

    # Source date availability is filtered by base_valid_mask. Legacy K-shot
    # support sampling also uses this tensor for coverage thresholds.
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

    # Pre-compute source_val_dates ONCE outside region loop (2022 only).
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
            if require_valid:
                bv = base_valid[idx]
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

    print(
        f"\nBuilding splits: adaptation_settings={args.adaptation_settings}, "
        f"legacy_few_shot={args.include_legacy_few_shot}, K={k_values}, seeds={seeds}"
    )

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

        # Get source val dates from pre-computed val_dates_all (2022 only)
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
        print(f"  Source val cycles (2022): {len(val_dates)}")

        # Get available target-train dates in 2015-2021
        target_train_available = get_target_dates(target_idx, target_train_mask, require_valid=False)
        print(f"  Available target-train dates in 2015-2021: {len(target_train_available)}")

        valid_mask = None
        if args.include_legacy_few_shot:
            valid_mask = compute_valid_support_mask(target_idx, target_train_available)
            n_valid = valid_mask.sum()
            print(f"  Valid target-train dates by coverage >= {args.min_coverage}: {n_valid}")

        # Get target query dates
        query_dates = get_target_dates(target_idx, query_mask, require_valid=False)
        print(f"  Target query cycles: {len(query_dates)}")

        # Generate main full-target-train splits. Seeds are retained as run seeds.
        if "target_full_train" in args.adaptation_settings:
            target_train_selected = select_target_full_train_dates(target_train_available)
            for seed in seeds:
                manifest = create_split_manifest(
                    target_region=target_region,
                    source_regions=source_region_ids,
                    K=None,
                    seed=seed,
                    source_train_dates=dates_to_serializable(source_dates),
                    source_val_dates=dates_to_serializable(val_dates),
                    target_train_dates=dates_to_serializable(target_train_selected),
                    query_dates=dates_to_serializable(query_dates),
                    adaptation_setting="target_full_train",
                )

                splits.append(manifest)
                print(
                    f"  target_full_train, seed={seed}: "
                    f"{len(target_train_selected)} target_train, {len(query_dates)} eval"
                )

        # Generate legacy few-shot ablation splits only when requested.
        if args.include_legacy_few_shot:
            for K in k_values:
                for seed in seeds:
                    support_selected = get_support_dates_for_K(
                        target_train_available, valid_mask, K, seed
                    )

                    manifest = create_split_manifest(
                        target_region=target_region,
                        source_regions=source_region_ids,
                        K=K,
                        seed=seed,
                        source_train_dates=dates_to_serializable(source_dates),
                        source_val_dates=dates_to_serializable(val_dates),
                        support_dates=dates_to_serializable(support_selected),
                        target_train_dates=dates_to_serializable(support_selected),
                        query_dates=dates_to_serializable(query_dates),
                        adaptation_setting=f"legacy_few_shot_k{K}",
                        protocol_version="legacy_kdate_protocol_v2",
                    )

                    splits.append(manifest)
                    print(f"  legacy K={K}, seed={seed}: {len(support_selected)} support, {len(query_dates)} query")

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
