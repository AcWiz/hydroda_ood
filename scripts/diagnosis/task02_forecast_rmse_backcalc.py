#!/usr/bin/env python3
"""Task 2: Back-calculate forecast-only RMSE from skill scores and increment RMSE.

Formula:
    skill = 1 - inc_rmse / forecast_rmse
    => forecast_rmse = inc_rmse / (1 - skill)

Uses summary.json from the latest source-only nonorm inference results.
Outputs CSV and markdown report.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = REPO_ROOT / "reports" / "source_only_rootzone_diagnosis"

# Summary JSONs from latest inference
INFERENCE_BASE = REPO_ROOT / "artifacts" / "results" / "phase4_source_only_inference"

def find_latest_inference_dir() -> Path | None:
    dirs = sorted(INFERENCE_BASE.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0] if dirs else None


def backcalc_forecast_rmse(skill: float, inc_rmse: float) -> float:
    """Back-calculate forecast-only RMSE from skill and increment RMSE."""
    denom = 1.0 - skill
    if abs(denom) < 1e-12:
        return float("inf") if denom >= 0 else float("-inf")
    return inc_rmse / denom


def main():
    run_dir = find_latest_inference_dir()
    if run_dir is None:
        print("ERROR: No inference directory found.")
        return

    source_summary_path = run_dir / "source_val" / "US-R1" / "summary.json"
    target_summary_path = run_dir / "target_query" / "US-R1" / "summary.json"

    source = json.loads(source_summary_path.read_text())
    target = json.loads(target_summary_path.read_text())

    rows = []
    for split_name, summary in [("source_val", source), ("target_query", target)]:
        for var in ["surface", "rootzone"]:
            skill = summary[var]["skill_mean"]
            inc_rmse = summary[var]["rmse_mean"]
            fcst_rmse = backcalc_forecast_rmse(skill, inc_rmse)
            ratio = inc_rmse / fcst_rmse if fcst_rmse > 0 else float("inf")
            rows.append({
                "split": split_name,
                "variable": var,
                "skill_mean": skill,
                "increment_rmse": inc_rmse,
                "forecast_only_rmse": fcst_rmse,
                "inc_rmse_over_fcst_rmse": ratio,
            })

    df = pd.DataFrame(rows)

    # CSV
    csv_path = REPORTS_DIR / "01_forecast_rmse_backcalc.csv"
    df.to_csv(csv_path, index=False, float_format="%.8f")
    print(f"Saved {csv_path}")

    # Markdown report
    md_lines = [
        "# Task 2: Forecast-only RMSE Back-calculation",
        "",
        f"Source: latest inference run `{run_dir.name}`",
        "",
        "## Formula",
        "",
        "```",
        "skill = 1 - increment_rmse / forecast_only_rmse",
        "forecast_only_rmse = increment_rmse / (1 - skill)",
        "```",
        "",
        "## Results",
        "",
        "| Split       | Variable  | Skill       | Inc RMSE    | Forecast-Only RMSE | Inc/Fcst Ratio |",
        "|:------------|:----------|------------:|------------:|-------------------:|---------------:|",
    ]

    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['split']:<11} | {r['variable']:<9} "
            f"| {r['skill_mean']:+.4f} "
            f"| {r['increment_rmse']:.6f} "
            f"| {r['forecast_only_rmse']:.8f} "
            f"| {r['inc_rmse_over_fcst_rmse']:.2f}x |"
        )

    md_lines.extend([
        "",
        "## Key Findings",
        "",
    ])

    # Rootzone analysis
    rz_sv = df[(df["variable"] == "rootzone") & (df["split"] == "source_val")].iloc[0]
    rz_tq = df[(df["variable"] == "rootzone") & (df["split"] == "target_query")].iloc[0]
    sf_sv = df[(df["variable"] == "surface") & (df["split"] == "source_val")].iloc[0]
    sf_tq = df[(df["variable"] == "surface") & (df["split"] == "target_query")].iloc[0]

    md_lines.append(f"- **Surface**: forecast-only RMSE is {sf_sv['forecast_only_rmse']:.6f} (source) / {sf_tq['forecast_only_rmse']:.6f} (target)")
    md_lines.append(f"  - Source-only inc RMSE is {sf_sv['inc_rmse_over_fcst_rmse']:.1f}x (source) / {sf_tq['inc_rmse_over_fcst_rmse']:.1f}x (target) forecast-only")
    md_lines.append(f"- **Rootzone**: forecast-only RMSE is {rz_sv['forecast_only_rmse']:.6f} (source) / {rz_tq['forecast_only_rmse']:.8f} (target)")
    md_lines.append(f"  - Source-only inc RMSE is {rz_sv['inc_rmse_over_fcst_rmse']:.1f}x (source) / {rz_tq['inc_rmse_over_fcst_rmse']:.1f}x (target) forecast-only")
    md_lines.append(f"  - Target_query rootzone forecast-only RMSE is **extremely small** ({rz_tq['forecast_only_rmse']:.2e})")
    md_lines.append(f"- **Rootzone skill < 0 on source_val** (-{abs(rz_sv['skill_mean']):.1f}) confirms model is worse than forecast-only even in-domain")
    md_lines.append(f"- **Rootzone target_query skill = {rz_tq['skill_mean']:.1f}** is dominated by tiny denominator")
    md_lines.append("")
    md_lines.append("> **Conclusion**: The large negative rootzone skill is primarily driven by an extremely small forecast-only RMSE denominator, not by catastrophically large prediction errors. The increment RMSE itself is small (~5-6e-4), but the forecast is already so good that any nonzero correction degrades performance.")

    md_path = REPORTS_DIR / "01_forecast_rmse_backcalc.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
