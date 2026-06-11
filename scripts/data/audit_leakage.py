#!/usr/bin/env python3
"""Leakage audit for target-full-train protocol verification.

Verifies:
1. source_train_dates and source_val_dates do not overlap target_eval_dates.
2. target_train_dates / target_adaptation_dates do not overlap target_eval_dates.
3. Target-evaluation labels are not marked as usable for training, prompt,
   normalization, early stopping, model selection, or hyperparameter selection.
"""

from __future__ import annotations

import json
from pathlib import Path

SPLITS_JSON = "artifacts/splits/US_loro_zero_few_shot_splits.json"
OUTPUT_JSON = "artifacts/experiments/phase3_simple_baselines/US/verification/leakage_audit.json"
OUTPUT_DIR = Path(OUTPUT_JSON).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORBIDDEN_EVAL_FLAG_FIELDS = [
    "target_eval_labels_used_for_training",
    "target_eval_labels_used_for_prompt",
    "target_eval_labels_used_for_normalization",
    "target_eval_labels_used_for_model_selection",
    "target_eval_labels_used_for_early_stopping",
    "target_eval_labels_used_for_hyperparameter_selection",
]


def run_audit():
    with open(SPLITS_JSON) as f:
        splits_data = json.load(f)

    issues = []
    total = 0

    for split in splits_data["splits"]:
        region = split.get("target_region_id", "")
        setting = split.get("adaptation_setting", "")
        seed = split.get("seed", "")
        label = f"{region} {setting} S={seed}"

        source_train_tis = {d["time_index"] for d in split.get("source_train_dates", [])}
        source_val_tis = {d["time_index"] for d in split.get("source_val_dates", [])}
        target_train_tis = {
            d["time_index"]
            for d in split.get("target_train_dates", split.get("target_support_dates", []))
        }
        target_adaptation_tis = {
            d["time_index"]
            for d in split.get("target_adaptation_dates", split.get("target_train_dates", []))
        }
        target_eval_tis = {
            d["time_index"]
            for d in split.get("target_eval_dates", split.get("target_query_dates", []))
        }

        if source_train_tis & target_eval_tis:
            issues.append(f"{label}: source_train overlaps target_eval")
        if source_val_tis & target_eval_tis:
            issues.append(f"{label}: source_val overlaps target_eval")
        if target_train_tis & target_eval_tis:
            issues.append(f"{label}: target_train overlaps target_eval")
        if target_adaptation_tis & target_eval_tis:
            issues.append(f"{label}: target_adaptation overlaps target_eval")

        for field in FORBIDDEN_EVAL_FLAG_FIELDS:
            if split.get(field, False) is not False:
                issues.append(f"{label}: forbidden flag {field}={split.get(field)!r}")

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
