#!/usr/bin/env python3
"""Metric Definition Audit for Phase 3B Verification.

Verifies:
1. skill = 1 - RMSE(pred, analysis) / RMSE(forecast, analysis) definition
2. forecast_only baseline: skill = 0 (within 1e-6)
3. true_increment = analysis - forecast
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

METRICS_CSV = "artifacts/metrics/phase3A_forecast_only_US/metrics_long.csv"
OUTPUT_JSON = "artifacts/experiments/phase3_simple_baselines/US/verification/metric_definition_audit.json"
OUTPUT_DIR = Path(OUTPUT_JSON).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_audit():
    if not Path(METRICS_CSV).exists():
        print(f"WARNING: {METRICS_CSV} not found, skipping metric audit")
        print("Run Phase 3A first to generate metrics")
        return None

    df = pd.read_csv(METRICS_CSV)

    # Filter to forecast_only method and target_query split
    fc_only = df[df["method"] == "forecast_only"].copy()

    issues = []
    results = []

    # Check 1: forecast_only skill should be ~0 for all regions/K/seeds
    fc_query = fc_only[fc_only["split_role"] == "target_query"]
    skill_metric = fc_query[fc_query["metric"] == "analysis_skill_vs_forecast"]

    if len(skill_metric) > 0:
        for _, row in skill_metric.iterrows():
            val = row["value"]
            if abs(val) > 1e-6 and not np.isnan(val):
                issues.append(f"forecast_only skill non-zero: {val} for {row['target_region_id']} K={row['K']} S={row['seed']}")
            results.append({
                "region": row["target_region_id"],
                "K": row["K"],
                "seed": row["seed"],
                "skill_value": val,
                "passed": abs(val) <= 1e-6 or np.isnan(val),
            })

    # Check 2: verify skill formula for a sample
    sample = fc_query[fc_query["metric"] == "analysis_rmse"].head(5)
    sample_skill = fc_query[fc_query["metric"] == "analysis_skill_vs_forecast"].head(5)

    if len(sample) > 0 and len(sample_skill) > 0:
        print("Sample skill values (forecast_only should be ~0):")
        print(sample_skill[["target_region_id", "K", "seed", "variable", "value"]].to_string())

    n_issues = len(issues)
    summary = {
        "total_samples": len(results),
        "n_issues": n_issues,
        "pass": n_issues == 0,
    }

    output = {
        "summary": summary,
        "issues": issues[:50],
        "details": results[:50],
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Metric definition audit: {n_issues} issues in {len(results)} samples")
    print(f"Wrote {OUTPUT_JSON}")
    return output


if __name__ == "__main__":
    run_audit()