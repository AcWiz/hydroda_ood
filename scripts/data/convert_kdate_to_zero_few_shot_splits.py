#!/usr/bin/env python3
"""Convert legacy K-date split envelopes into the V4.4 zero/few-shot schema.

The legacy K-date artifact used older temporal semantics. This converter keeps
only the target-region/K/seed grid from that artifact and sources all date
fields from the corrected historical target-train artifact, then reselects
K-shot support from 2015-2021 target context dates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from hydroda.splits.kdate import get_support_dates_for_K
from hydroda.splits.manifest import create_split_manifest, generate_split_summary_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert legacy K-date splits to zero/few-shot schema")
    parser.add_argument("--in-json", default="artifacts/splits/US_loro_kdate_splits.json")
    parser.add_argument("--target-train-json", default="artifacts/splits/US_loro_target_train_splits.json")
    parser.add_argument("--out-json", default="artifacts/splits/US_loro_zero_few_shot_splits.json")
    parser.add_argument("--out-md", default="reports/splits/US_loro_zero_few_shot_split_summary.md")
    parser.add_argument("--min-coverage", default=0.5, type=float)
    return parser.parse_args()


def _records_to_tuples(records: list[dict]) -> list[tuple[int, object]]:
    from datetime import datetime

    result = []
    for record in records:
        result.append((int(record["time_index"]), datetime.fromisoformat(record["date_str"])))
    return result


def _tuples_to_records(selected: list[tuple[int, object]], records_by_index: dict[int, dict]) -> list[dict]:
    return [records_by_index[int(idx)] for idx, _dt in selected]


def _coverage_valid_mask(records: list[dict], min_coverage: float) -> np.ndarray:
    valid = []
    for record in records:
        coverage = record.get("base_valid_coverage", record.get("base_valid_mask_coverage"))
        if coverage is None:
            coverage = record.get("target_region_base_valid_coverage")
        valid.append(False if coverage is None else float(coverage) >= float(min_coverage))
    return np.asarray(valid, dtype=bool)


def select_support_records_with_coverage(
    context_dates: list[dict],
    K: int,
    seed: int,
    min_coverage: float,
) -> list[dict]:
    """Select support records using calendar buckets and recorded coverage only."""
    candidates = _records_to_tuples(context_dates)
    records_by_index = {int(record["time_index"]): record for record in context_dates}
    valid_mask = _coverage_valid_mask(context_dates, min_coverage=min_coverage)
    return _tuples_to_records(
        get_support_dates_for_K(candidates, valid_mask, int(K), int(seed)),
        records_by_index,
    )


def main() -> None:
    args = parse_args()
    with open(args.in_json, "r", encoding="utf-8") as f:
        kdate_splits = json.load(f)["splits"]

    canonical_by_region: dict[str, dict] = {}
    target_train_path = Path(args.target_train_json)
    if not target_train_path.exists():
        raise FileNotFoundError(
            f"{target_train_path} is required so V4.4 date fields come from the corrected "
            "2015-2021/2022/2023-2025 protocol artifact."
        )
    with open(target_train_path, "r", encoding="utf-8") as f:
        target_train_splits = json.load(f)["splits"]
    for split in target_train_splits:
        canonical_by_region[split["target_region_id"]] = split

    converted = []
    for split in kdate_splits:
        K = int(split["K"])
        if K not in {0, 4, 12}:
            continue
        target_region = split["target_region_id"]
        if target_region not in canonical_by_region:
            raise KeyError(f"target region {target_region} missing from {target_train_path}")
        canonical = canonical_by_region[target_region]
        adaptation_setting = "zero_shot_context" if K == 0 else f"few_shot_k{K}"
        context_dates = canonical.get("target_context_dates") or canonical.get("target_train_dates", [])
        support_dates = select_support_records_with_coverage(
            context_dates=context_dates,
            K=K,
            seed=int(split["seed"]),
            min_coverage=args.min_coverage,
        )
        converted.append(
            create_split_manifest(
                target_region=target_region,
                source_regions=canonical["source_region_ids"],
                K=K,
                seed=int(split["seed"]),
                source_train_dates=canonical.get("source_train_dates", []),
                source_val_dates=canonical.get("source_val_dates", []),
                source_test_dates=canonical.get("source_test_dates", []),
                target_context_dates=context_dates,
                support_dates=support_dates,
                target_train_dates=support_dates,
                query_dates=canonical.get("target_eval_dates", canonical.get("target_query_dates", [])),
                adaptation_setting=adaptation_setting,
            )
        )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"splits": converted}, f, indent=2)
    generate_split_summary_markdown(converted, args.out_md)
    print(f"Saved {len(converted)} splits to {args.out_json}")


if __name__ == "__main__":
    main()
