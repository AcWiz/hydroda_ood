#!/usr/bin/env python3
"""Build US Leave-One-Region-Out zero/few-shot split manifests.

No-leakage declaration:
    - source_fit and target_context dates come from 2015-2021.
    - source_val dates come from 2022 and are the only main model-selection source.
    - target_support dates are selected only by calendar rules, availability,
      and base_valid coverage.
    - target_eval dates come from 2023-2025 and are final offline evaluation only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import xarray as xr

from hydroda.splits.kdate import dates_to_serializable, get_support_dates_for_K
from hydroda.splits.manifest import create_split_manifest, generate_split_summary_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build US LORO zero/few-shot splits")
    parser.add_argument("--da-nc", required=True, help="Path to DA.nc SMAP data")
    parser.add_argument("--region-masks", required=True, help="Path to US_region_masks.nc")
    parser.add_argument(
        "--out-json",
        default="artifacts/splits/US_loro_zero_few_shot_splits.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--out-md",
        default="reports/splits/US_loro_zero_few_shot_split_summary.md",
        help="Output markdown path",
    )
    parser.add_argument("--k-values", nargs="+", default=[0, 4, 12], type=int)
    parser.add_argument("--seeds", nargs="+", default=[0, 1, 2], type=int)
    parser.add_argument("--min-coverage", default=0.5, type=float)
    return parser.parse_args()


def _has_any_valid_pixels(base_valid: np.ndarray, region_mask: np.ndarray) -> bool:
    return bool((region_mask & (base_valid > 0)).sum() > 0)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)

    ds = xr.open_dataset(args.da_nc, decode_times=False)
    rm = xr.open_dataset(args.region_masks)
    try:
        time_vals = ds["time"].values.astype(np.int64)
        dts = [datetime.fromtimestamp(t) for t in time_vals]
        years = np.array([dt.year for dt in dts])
        source_fit_mask = (years >= 2015) & (years <= 2021)
        source_val_mask = years == 2022
        source_test_mask = (years >= 2023) & (years <= 2025)
        target_context_mask = (years >= 2015) & (years <= 2021)
        target_eval_mask = (years >= 2023) & (years <= 2025)

        region_ids = rm["region_id"].values.tolist()
        region_onehot = rm["region_mask_onehot"].values.astype(bool)
        region_sizes = region_onehot.sum(axis=(1, 2))
        base_valid = ds["input"][:, 11, :, :].values.astype(np.float32)

        def dates_for_region(region_idx: int, mask: np.ndarray, *, require_valid: bool) -> list[tuple[int, datetime]]:
            region_mask = region_onehot[region_idx]
            dates: list[tuple[int, datetime]] = []
            for idx in np.where(mask)[0]:
                if require_valid and not _has_any_valid_pixels(base_valid[idx], region_mask):
                    continue
                dates.append((int(idx), dts[int(idx)]))
            return dates

        def support_valid_mask(region_idx: int, dates: list[tuple[int, datetime]]) -> np.ndarray:
            region_mask = region_onehot[region_idx]
            region_size = int(region_sizes[region_idx])
            valid = np.zeros(len(dates), dtype=bool)
            for i, (idx, _dt) in enumerate(dates):
                valid_pixels = (region_mask & (base_valid[idx] > 0)).sum()
                coverage = float(valid_pixels) / float(region_size) if region_size > 0 else 0.0
                valid[i] = coverage >= args.min_coverage
            return valid

        print("Precomputing per-region date availability...")
        per_region = []
        for region_idx, region_id in enumerate(region_ids):
            per_region.append(
                {
                    "source_fit": dates_for_region(region_idx, source_fit_mask, require_valid=True),
                    "source_val": dates_for_region(region_idx, source_val_mask, require_valid=True),
                    "source_test": dates_for_region(region_idx, source_test_mask, require_valid=True),
                    "target_context": dates_for_region(region_idx, target_context_mask, require_valid=False),
                    "target_eval": dates_for_region(region_idx, target_eval_mask, require_valid=False),
                }
            )
            print(
                f"  {region_id}: context={len(per_region[-1]['target_context'])} "
                f"eval={len(per_region[-1]['target_eval'])}",
                flush=True,
            )

        splits = []
        for target_idx, target_region in enumerate(region_ids):
            source_region_ids = [r for r in region_ids if r != target_region]
            source_region_indices = [region_ids.index(r) for r in source_region_ids]

            source_fit_dates: list[tuple[int, datetime]] = []
            source_val_dates: list[tuple[int, datetime]] = []
            source_test_dates: list[tuple[int, datetime]] = []
            for src_idx in source_region_indices:
                source_fit_dates.extend(per_region[src_idx]["source_fit"])
                source_val_dates.extend(per_region[src_idx]["source_val"])
                source_test_dates.extend(per_region[src_idx]["source_test"])

            target_context_dates = per_region[target_idx]["target_context"]
            target_eval_dates = per_region[target_idx]["target_eval"]
            valid_mask = support_valid_mask(target_idx, target_context_dates)

            for K in args.k_values:
                for seed in args.seeds:
                    support_selected = get_support_dates_for_K(target_context_dates, valid_mask, K, seed)
                    adaptation_setting = "zero_shot_context" if int(K) == 0 else f"few_shot_k{int(K)}"
                    splits.append(
                        create_split_manifest(
                            target_region=target_region,
                            source_regions=source_region_ids,
                            K=int(K),
                            seed=int(seed),
                            source_train_dates=dates_to_serializable(source_fit_dates),
                            source_val_dates=dates_to_serializable(source_val_dates),
                            source_test_dates=dates_to_serializable(source_test_dates),
                            target_context_dates=dates_to_serializable(target_context_dates),
                            support_dates=dates_to_serializable(support_selected),
                            target_train_dates=dates_to_serializable(support_selected),
                            query_dates=dates_to_serializable(target_eval_dates),
                            adaptation_setting=adaptation_setting,
                        )
                    )

        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"splits": splits}, f, indent=2)
        generate_split_summary_markdown(splits, args.out_md)
        print(f"Saved {len(splits)} splits to {args.out_json}")
    finally:
        ds.close()
        rm.close()


if __name__ == "__main__":
    main()
