#!/usr/bin/env python3
"""Rebuild Phase 3 Forecast-Only summary.json files and merged summary table.

Reads existing metrics_long.csv files (no DA.nc I/O) and:
  1. Rebuilds each summary.json with correct regions from active_region_ids
  2. Generates a merged summary table: one row per region, with target_eval
     and source_test metrics side-by-side.

Usage:
    PYTHONPATH=. python scripts/eval/rebuild_phase3_summary_table.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path("artifacts/results/phase3_forecast_only_all_regions")
SPLIT_TYPES = ["target_eval", "source_test"]
REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]

VARIABLES = ["surface", "rootzone"]
METRICS = [
    "analysis_skill_vs_forecast_global",
    "analysis_skill_vs_forecast_latw_global",
    "increment_rmse",
    "analysis_rmse",
    "analysis_rmse_latw",
]


def compute_summary(csv_path: Path) -> dict:
    """Compute summary statistics from a single metrics_long.csv."""
    df = pd.read_csv(str(csv_path))

    # Per-sample rows (exclude global aggregate rows)
    per_sample = df[df["query_date"] != "global"]
    n_dates = per_sample["query_date"].nunique()
    n_valid_pixels_total = int(
        per_sample.groupby("query_date")["n_valid_pixels"].first().sum()
    )

    # Regions from active_region_ids (pipe-separated for source_test)
    regions = sorted(
        set(
            rid
            for aids in per_sample["active_region_ids"].dropna().unique()
            for rid in str(aids).split("|")
        )
    )

    # Target region from column
    target_region_id = sorted(
        df["target_region_id"].dropna().unique().tolist()
    )

    summary = {
        "method": "forecast_only",
        "n_metric_rows": len(df),
        "n_dates": n_dates,
        "n_valid_pixels_total": n_valid_pixels_total,
        "regions": regions,
        "target_region_id": target_region_id,
    }

    for variable in VARIABLES:
        var_df = df[df["variable"] == variable]
        if var_df.empty:
            continue
        summary[variable] = {}
        for metric in METRICS:
            metric_df = var_df[var_df["metric"] == metric]
            if metric_df.empty:
                continue
            if metric.endswith("_global"):
                # Global metrics are single-row per variable (aggregate-then-sqrt)
                summary[variable][metric] = float(metric_df["value"].iloc[0])
            else:
                summary[variable][f"{metric}_mean"] = float(
                    metric_df["value"].mean()
                )
                summary[variable][f"{metric}_std"] = float(
                    metric_df["value"].std()
                )

    return summary


def _fmt(v) -> str:
    """Format a value for markdown table display."""
    if isinstance(v, float):
        return f"{v:.10f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def main():
    # Step 1: Rebuild individual summary.json files
    print("=" * 60)
    print("Step 1: Rebuilding summary.json files")
    print("=" * 60)

    for split_type in SPLIT_TYPES:
        for region in REGIONS:
            csv_path = BASE / split_type / region / "metrics_long.csv"
            if not csv_path.exists():
                print(f"  SKIP (missing): {split_type}/{region}")
                continue

            summary = compute_summary(csv_path)
            summary_json_path = BASE / split_type / region / "summary.json"
            with open(summary_json_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(
                f"  OK: {split_type}/{region}  "
                f"regions={summary['regions']}  "
                f"dates={summary['n_dates']}  "
                f"pixels={summary['n_valid_pixels_total']:,}"
            )

    # Step 2: Build merged summary table
    print()
    print("=" * 60)
    print("Step 2: Building merged summary table")
    print("=" * 60)

    merged = {}
    for split_type in SPLIT_TYPES:
        for region in REGIONS:
            sj_path = BASE / split_type / region / "summary.json"
            if not sj_path.exists():
                continue
            with open(sj_path) as f:
                s = json.load(f)
            merged.setdefault(region, {})[split_type] = s

    # Build markdown table
    header_cols = [
        "Region",
        "N_Dates",
        "N_Px(Tgt)",
        "N_Px(Src)",
        "Surf_WRMSE(Tgt)",
        "Surf_WRMSE(Src)",
        "RZ_WRMSE(Tgt)",
        "RZ_WRMSE(Src)",
    ]

    lines = [
        "# Forecast-Only Baseline — All Regions Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"| {' | '.join(header_cols)} |",
        f"|{'|'.join(['--------'] * len(header_cols))}|",
    ]

    for region in REGIONS:
        data = merged.get(region, {})
        tgt = data.get("target_eval", {})
        src = data.get("source_test", {})

        n_dates = tgt.get("n_dates", 0)

        tgt_px = tgt.get("n_valid_pixels_total", 0)
        src_px = src.get("src_valid_pixels_total", src.get("n_valid_pixels_total", 0))

        tgt_surf = tgt.get("surface", {})
        src_surf = src.get("surface", {})
        tgt_rz = tgt.get("rootzone", {})
        src_rz = src.get("rootzone", {})

        row = [
            region,
            _fmt(n_dates),
            _fmt(tgt_px),
            _fmt(src_px),
            _fmt(tgt_surf.get("analysis_rmse_latw_mean", float("nan"))),
            _fmt(src_surf.get("analysis_rmse_latw_mean", float("nan"))),
            _fmt(tgt_rz.get("analysis_rmse_latw_mean", float("nan"))),
            _fmt(src_rz.get("analysis_rmse_latw_mean", float("nan"))),
        ]
        lines.append(f"| {' | '.join(row)} |")

        print(
            f"  {region}: tgt_px={tgt_px:,}  src_px={src_px:,}  "
            f"tgt_surf_rmse={tgt_surf.get('analysis_rmse_latw_mean', '?'):.10f}  "
            f"src_surf_rmse={src_surf.get('analysis_rmse_latw_mean', '?'):.10f}"
        )

    md_path = BASE / "summary_table.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\n  Table saved to {md_path}")

    # Print table to stdout
    print()
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
