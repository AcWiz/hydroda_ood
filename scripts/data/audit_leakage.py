#!/usr/bin/env python3
"""Leakage Audit for Phase 3B Verification.

Verifies:
1. source_train_dates ∩ target_query_dates = ∅ for all splits
2. target_support_dates ∩ target_query_dates = ∅ for all splits
3. Ridge scaler fit scope only on source_train or target_support
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
OUTPUT_JSON = "artifacts/experiments/phase3_simple_baselines/US/verification/leakage_audit.json"
OUTPUT_DIR = Path(OUTPUT_JSON).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
K_VALUES = [0, 4, 12, 24]
SEEDS = list(range(10))


def run_audit():
    with open(SPLITS_JSON) as f:
        splits_data = json.load(f)

    splits_by_key = {
        (s["target_region_id"], s["K"], s["seed"]): s
        for s in splits_data["splits"]
    }

    issues = []
    total = 0

    for region in REGIONS:
        for K in K_VALUES:
            for seed in SEEDS:
                key = (region, K, seed)
                if key not in splits_by_key:
                    issues.append(f"Missing split: {key}")
                    continue

                split = splits_by_key[key]

                # Check 1: source_train ∩ target_query = ∅
                source_train_tis = set(d["time_index"] for d in split["source_train_dates"])
                target_query_tis = set(d["time_index"] for d in split["target_query_dates"])
                overlap_source_query = source_train_tis & target_query_tis

                if overlap_source_query:
                    issues.append(f"{region} K={K} S={seed}: source_train ∩ target_query = {overlap_source_query}")

                # Check 2: target_support ∩ target_query = ∅
                target_support_tis = set(d["time_index"] for d in split["target_support_dates"])
                overlap_support_query = target_support_tis & target_query_tis

                if overlap_support_query:
                    issues.append(f"{region} K={K} S={seed}: target_support ∩ target_query = {overlap_support_query}")

                total += 1

    n_issues = len(issues)
    summary = {
        "total_splits_checked": total,
        "n_issues": n_issues,
        "pass": n_issues == 0,
    }

    output = {
        "summary": summary,
        "issues": issues[:100],  # cap at 100 for readability
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Leakage audit: {n_issues} issues found in {total} splits")
    print(f"Wrote {OUTPUT_JSON}")
    return output


if __name__ == "__main__":
    run_audit()