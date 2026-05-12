#!/usr/bin/env python3
"""Generate cross-method comparison tables for Phase 4 results.

Reads metrics_long.csv from forecast-only, source_mean_increment, source_only_backbone,
and prompt_conditioned_shared, and generates comparison markdown tables.

Usage:
    PYTHONPATH=. python scripts/analysis/run_phase4_comparison.py \\
        --region US-R1 --output artifacts/metrics/phase4_comparison_US.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# Expected evaluation output directories
_METRIC_DIRS = {
    "forecast_only": "artifacts/results/phase3A/forecast_only",
    "source_mean_increment": "artifacts/results/phase3A/source_mean_increment",
    "source_only_backbone": "artifacts/results/phase4_source_only",
    "prompt_conditioned_shared": "artifacts/results/phase4_prompt_conditioned",
}


def load_metrics(region: str, method: str, base_dir: str) -> Optional[pd.DataFrame]:
    """Load metrics_long.csv for a given method and region."""
    csv_path = Path(base_dir) / region / "metrics_long.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping {method}")
        return None
    df = pd.read_csv(csv_path)
    df["method"] = method
    return df


def build_summary(df: pd.DataFrame) -> Dict:
    """Build method-level summary from long-format metrics."""
    skill = df[df["metric"] == "analysis_skill_vs_forecast"]
    inc_rmse = df[df["metric"] == "increment_rmse"]
    inc_corr = df[df["metric"] == "increment_corr"]
    inc_bias = df[df["metric"] == "increment_bias"]
    analysis_rmse = df[df["metric"] == "analysis_rmse"]

    def _mean(df_sub: pd.DataFrame, var: str) -> float:
        vals = df_sub[df_sub["variable"] == var]["value"]
        return float(vals.mean()) if len(vals) > 0 else float("nan")

    return {
        "surface_skill": _mean(skill, "surface"),
        "rootzone_skill": _mean(skill, "rootzone"),
        "surface_inc_rmse": _mean(inc_rmse, "surface"),
        "rootzone_inc_rmse": _mean(inc_rmse, "rootzone"),
        "surface_inc_corr": _mean(inc_corr, "surface"),
        "rootzone_inc_corr": _mean(inc_corr, "rootzone"),
        "surface_inc_bias": _mean(inc_bias, "surface"),
        "rootzone_inc_bias": _mean(inc_bias, "rootzone"),
        "surface_analysis_rmse": _mean(analysis_rmse, "surface"),
        "rootzone_analysis_rmse": _mean(analysis_rmse, "rootzone"),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 4 comparison tables")
    parser.add_argument("--region", type=str, default=None,
        help="Single region (e.g. US-R1) or all regions if omitted")
    parser.add_argument("--regions", type=str, nargs="*",
        default=["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"],
        help="List of regions to include")
    parser.add_argument("--output", type=str,
        default="artifacts/metrics/phase4_comparison_US.md",
        help="Output markdown file")
    parser.add_argument("--methods", type=str, nargs="*",
        default=["forecast_only", "source_mean_increment", "source_only_backbone", "prompt_conditioned_shared"],
        help="Methods to include in comparison")
    args = parser.parse_args()

    regions = [args.region] if args.region else args.regions

    all_data: List[pd.DataFrame] = []
    method_summaries: Dict[str, Dict] = {}

    for method in args.methods:
        base_dir = _METRIC_DIRS.get(method)
        if base_dir is None:
            print(f"  WARNING: unknown method {method}, skipping")
            continue

        method_dfs = []
        for region in regions:
            df = load_metrics(region, method, base_dir)
            if df is not None:
                method_dfs.append(df)

        if method_dfs:
            combined = pd.concat(method_dfs, ignore_index=True)
            all_data.append(combined)
            method_summaries[method] = build_summary(combined)
        else:
            print(f"  WARNING: no data for {method}")

    if not all_data:
        print("ERROR: No data loaded for any method. Check paths.")
        return

    # Generate markdown
    lines = []
    lines.append("# Phase 4: Neural Baseline Comparison")
    lines.append("")
    lines.append(f"Regions: {', '.join(regions)}")
    lines.append("")
    lines.append("## Main Comparison Table")
    lines.append("")
    lines.append("| Method | Surf Skill | Root Skill | Surf Inc-RMSE | Root Inc-RMSE | Surf Inc-Corr | Root Inc-Corr | Surf Inc-Bias | Root Inc-Bias |")
    lines.append("|--------|-----------|-----------|--------------|--------------|--------------|--------------|--------------|--------------|")

    for method in args.methods:
        s = method_summaries.get(method)
        if s is None:
            continue
        lines.append(
            f"| {method} | {s['surface_skill']:.4f} | {s['rootzone_skill']:.4f} | "
            f"{s['surface_inc_rmse']:.6f} | {s['rootzone_inc_rmse']:.6f} | "
            f"{s['surface_inc_corr']:.4f} | {s['rootzone_inc_corr']:.4f} | "
            f"{s['surface_inc_bias']:.6f} | {s['rootzone_inc_bias']:.6f} |"
        )

    lines.append("")
    lines.append("## Per-Region Detail")
    lines.append("")

    for region in regions:
        lines.append(f"### {region}")
        lines.append("")
        lines.append("| Method | Surf Skill | Root Skill | Surf Inc-RMSE | Root Inc-RMSE | Surf Inc-Corr | Root Inc-Corr |")
        lines.append("|--------|-----------|-----------|--------------|--------------|--------------|--------------|")

        for method in args.methods:
            base_dir = _METRIC_DIRS.get(method)
            if base_dir is None:
                continue
            csv_path = Path(base_dir) / region / "metrics_long.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            s = build_summary(df)
            lines.append(
                f"| {method} | {s['surface_skill']:.4f} | {s['rootzone_skill']:.4f} | "
                f"{s['surface_inc_rmse']:.6f} | {s['rootzone_inc_rmse']:.6f} | "
                f"{s['surface_inc_corr']:.4f} | {s['rootzone_inc_corr']:.4f} |"
            )
        lines.append("")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Comparison table written to {output_path}")

    # Also write JSON summary
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(method_summaries, f, indent=2)
    print(f"JSON summary written to {json_path}")


if __name__ == "__main__":
    main()
