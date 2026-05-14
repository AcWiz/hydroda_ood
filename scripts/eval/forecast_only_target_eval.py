#!/usr/bin/env python3
"""Forecast-only evaluation for target regions with CLI selection.

No-leakage declaration:
    - ForecastBaseline: no fit(), no training, no target label access
    - Dataset: split from frozen manifest, target_query split only
    - Evaluator: query labels used only as evaluation labels post-prediction

Usage:
    PYTHONPATH=. python scripts/eval/forecast_only_target_eval.py \
        --target_region US-R1 \
        --output_dir artifacts/results/phase3_forecast_only_target

    # Evaluate all US regions
    PYTHONPATH=. python scripts/eval/forecast_only_target_eval.py \
        --target_region US-R1 US-R2 US-R3 US-R4 US-R5 US-R6 \
        --output_dir artifacts/results/phase3_forecast_only_target
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
PROTOCOL_FREEZE_ID = "hyperda_v4_final_2015_2025_context2022_query2023_2025_k0_4_12"


def run_forecast_only_for_region(
    region: str,
    split_type: str = "target_query",
    K: int = 4,
    seed: int = 0,
    max_samples: int = 0,
) -> tuple[list, dict]:
    """Run forecast-only evaluation for a single region."""
    from hydroda.baselines.forecast import ForecastBaseline
    from hydroda.data.dataset import HydroDADataset
    from hydroda.evaluation.harness import evaluate_split

    ds = HydroDADataset(
        da_nc_path=f"{DATA_DIR}/DA.nc",
        region_masks_nc=REGION_MASKS,
        splits_json=SPLITS_JSON,
        target_region=region,
        split_type=split_type,
        K=K, seed=seed,
        freeze_manifest=MANIFEST,
    )

    predictor = ForecastBaseline()
    rows = evaluate_split(
        dataset=ds,
        predictor=predictor,
        split_role=split_type,
        experiment_id=f"phase3_forecast_only_{region}",
        protocol_freeze_id=PROTOCOL_FREEZE_ID,
        method="forecast_only",
        split_file=SPLITS_JSON,
        mask_file=REGION_MASKS,
        preloaded=False,
        max_samples=max_samples if max_samples > 0 else None,
    )
    ds.close()
    return rows, {"region": region, "n_rows": len(rows)}


def aggregate_summary(rows: list) -> dict:
    """Compute summary statistics from metric rows."""
    if not rows:
        return {"status": "no_rows"}
    df = pd.DataFrame(rows)
    summary = {"method": "forecast_only", "n_samples_evaluated": len(df)}

    for variable in ["surface", "rootzone"]:
        var_df = df[df["variable"] == variable]
        if var_df.empty:
            continue
        summary[variable] = {}
        for metric in ["analysis_skill_vs_forecast", "increment_rmse", "analysis_rmse"]:
            metric_df = var_df[var_df["metric"] == metric]
            if metric_df.empty:
                continue
            summary[variable][f"{metric}_mean"] = float(metric_df["value"].mean())
            summary[variable][f"{metric}_std"] = float(metric_df["value"].std())

    summary["n_metric_rows"] = len(df)
    summary["regions"] = sorted(df["target_region_id"].unique().tolist())
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Forecast-only evaluation for target regions"
    )
    parser.add_argument(
        "--target_region", nargs="+", default=["US-R1"],
        help="Region(s) to evaluate (e.g. US-R1 US-R2 ...)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="artifacts/results/phase3_forecast_only_target",
        help="Output directory for results"
    )
    parser.add_argument(
        "--split_type", type=str, default="target_query",
        help="Split type (default: target_query)"
    )
    parser.add_argument(
        "--K", type=int, default=4,
        help="Number of calibration cycles (default: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="Max samples per region (0 = all, default: 0)"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_rows = []
    region_summaries = {}

    for region in args.target_region:
        print(f"\n[{region}] Starting forecast-only evaluation...")
        print(f"  split_type={args.split_type}, K={args.K}, seed={args.seed}, max_samples={args.max_samples}")

        rows, info = run_forecast_only_for_region(
            region=region,
            split_type=args.split_type,
            K=args.K,
            seed=args.seed,
            max_samples=args.max_samples,
        )
        all_rows.extend(rows)
        region_summaries[region] = info

        region_dir = out_dir / region
        region_dir.mkdir(parents=True, exist_ok=True)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(region_dir / "metrics_long.csv", index=False)
            summary = aggregate_summary(rows)
            with open(region_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  -> {len(rows)} rows, summary saved to {region_dir}/")
        else:
            print(f"  -> No rows collected: {info}")

    if all_rows:
        all_df = pd.DataFrame(all_rows)
        all_df.to_csv(out_dir / "metrics_long.csv", index=False)

        overall_summary = aggregate_summary(all_rows)
        overall_summary["regions_evaluated"] = list(region_summaries.keys())
        with open(out_dir / "summary.json", "w") as f:
            json.dump(overall_summary, f, indent=2)
        print(f"\n[OK] Overall: {len(all_rows)} rows across {len(region_summaries)} regions")
        print(f"     Output: {out_dir}/")
    else:
        print(f"\n[WARN] No results collected from any region.")

    print("\n[Region summaries]")
    for region, info in region_summaries.items():
        print(f"  {region}: {info}")


if __name__ == "__main__":
    main()