"""Split manifest generation and validation.

No-leakage declaration:
    Manifests are generated using only:
    - Temporal constraints (date ranges from protocol specs)
    - Region masks from artifacts/regions/US_region_masks.nc
    - Main target-train dates from the full historical target training period
    - Legacy K-date selection via calendar rules (no analysis/model errors)
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np


# Period strings derived from protocol_v4.yaml
# Protocol V4.3 historical target adaptation:
#   source_fit:    2015-01-01 to 2021-12-31
#   source_val:    2022-01-01 to 2022-12-31
#   target_train:  2015-01-01 to 2021-12-31
#   target_eval:   2023-01-01 to 2025-12-31
PERIODS = {
    "source_fit": "2015-01-01 to 2021-12-31",
    "source_val": "2022-01-01 to 2022-12-31",
    "target_train": "2015-01-01 to 2021-12-31",
    "target_adapt": "2015-01-01 to 2021-12-31",
    "target_val": "2022-01-01 to 2022-12-31",
    "target_eval": "2023-01-01 to 2025-12-31",
    # Deprecated aliases retained for older artifacts and scripts.
    "target_support": "2015-01-01 to 2021-12-31",
    "target_query": "2023-01-01 to 2025-12-31",
}

PROTOCOL_FREEZE_ID = "hyperda_v4_3_historical_target_adapt_2015_2025_train2015_2021_val2022_test2023_2025"
TARGET_TRAIN_YEARS = set(range(2015, 2022))
SOURCE_VAL_YEARS = {2022}
TARGET_EVAL_YEARS = {2023, 2024, 2025}


def _records_hash(records: List[dict]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_split_manifest(
    target_region: str,
    source_regions: List[str],
    K: Optional[int] = None,
    seed: int = 0,
    source_train_dates: Optional[List[dict]] = None,
    support_dates: Optional[List[dict]] = None,
    query_dates: Optional[List[dict]] = None,
    source_val_dates: List[dict] | None = None,
    target_train_dates: Optional[List[dict]] = None,
    adaptation_setting: Optional[str] = None,
    country_id: str = "US",
    benchmark_id: str = "hydroda_ood_us_v1",
    protocol_version: str = "target_full_train_protocol_v1",
) -> Dict:
    """Create a split manifest dict for LORO evaluation.

    Args:
        target_region: Target region ID (e.g., "US-R1")
        source_regions: List of source region IDs (e.g., ["US-R2", ..., "US-R6"])
        K: Legacy number of support dates. None for the main full-target-train protocol.
        seed: Random seed/run seed.
        source_train_dates: List of dicts with time_index, date_str, datetime_str
        support_dates: Legacy support dates. For the main protocol this is an alias of target_train_dates.
        query_dates: List of dicts for target eval/query dates
        source_val_dates: List of dicts for source validation (2022). If None, defaults to [].
        target_train_dates: Full target training/adaptation dates (2015-2021). If None, falls back to support_dates.
        adaptation_setting: Main adaptation setting, e.g. "target_full_train".
        country_id: Country identifier (default "US")
        benchmark_id: Benchmark identifier
        protocol_version: Protocol version string

    Returns:
        Manifest dict with all required fields per kdate_protocol.yaml
    """
    source_train_dates = list(source_train_dates or [])
    source_val_dates = list(source_val_dates or [])
    query_dates = list(query_dates or [])

    legacy_K = int(K) if K is not None else None
    if adaptation_setting is None:
        adaptation_setting = "target_full_train" if legacy_K is None else f"legacy_few_shot_k{legacy_K}"

    if target_train_dates is None:
        target_train_dates = list(support_dates or [])
    else:
        target_train_dates = list(target_train_dates)

    if support_dates is None:
        support_dates = list(target_train_dates)
    else:
        support_dates = list(support_dates)

    target_eval_dates = list(query_dates)
    is_legacy_kshot = adaptation_setting.startswith("legacy_few_shot")
    adaptation_protocol = "legacy_few_shot_ablation" if is_legacy_kshot else "target_full_train"

    manifest = {
        "manifest_schema_version": "v3_full_target_train",
        "benchmark_id": benchmark_id,
        "protocol_version": protocol_version,
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "country_id": country_id,
        "source_fit_period": PERIODS["source_fit"],
        "source_val_period": PERIODS["source_val"],
        "target_train_period": PERIODS["target_train"],
        "target_adapt_period": PERIODS["target_adapt"],
        "target_val_period": PERIODS["target_val"],
        "target_eval_period": PERIODS["target_eval"],
        "target_support_period": PERIODS["target_support"],
        "target_query_period": PERIODS["target_query"],
        "target_region_id": target_region,
        "source_region_ids": source_regions,
        "adaptation_protocol": adaptation_protocol,
        "adaptation_setting": adaptation_setting,
        "K": legacy_K,
        "K_legacy": legacy_K,
        "seed": seed,
        "source_train_dates": source_train_dates,
        "source_val_dates": source_val_dates,
        "target_train_dates": target_train_dates,
        "target_adaptation_dates": target_train_dates,
        "target_eval_dates": target_eval_dates,
        # Deprecated aliases retained for existing dataset/evaluation code.
        "target_support_dates": support_dates,
        "target_query_dates": target_eval_dates,
        "source_train_cycle_count": len(source_train_dates),
        "source_val_cycle_count": len(source_val_dates),
        "target_train_cycle_count": len(target_train_dates),
        "target_adaptation_cycle_count": len(target_train_dates),
        "target_eval_cycle_count": len(target_eval_dates),
        "target_support_cycle_count": len(support_dates),
        "target_query_cycle_count": len(target_eval_dates),
        "source_train_dates_hash": _records_hash(source_train_dates),
        "source_val_dates_hash": _records_hash(source_val_dates),
        "target_train_dates_hash": _records_hash(target_train_dates),
        "target_adaptation_dates_hash": _records_hash(target_train_dates),
        "target_eval_dates_hash": _records_hash(target_eval_dates),
        "target_query_dates_hash": _records_hash(target_eval_dates),
        "support_dates_hash": _records_hash(support_dates),
        "selection_uses_analysis": False,
        "selection_uses_query_labels": False,
        "target_eval_labels_used_for_training": False,
        "target_eval_labels_used_for_prompt": False,
        "target_eval_labels_used_for_normalization": False,
        "target_eval_labels_used_for_model_selection": False,
        "normalization_scope": "source_fit_only",
        "model_selection_scope": "source_val_only",
        "source_training_policy": "main_protocol_uses_source_fit_2015_2021_source_val_2022",
        "created_by": "build_kdate_splits.py",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    return manifest


def validate_no_leakage(manifest: Dict) -> Dict[str, bool]:
    """Validate no-leakage constraints in a split manifest.

    Args:
        manifest: Split manifest dict

    Returns:
        Dict of validation results with keys:
        - 'support_in_support_year': target_support_dates are in 2015-2021
        - 'query_in_query_years': target_query_dates are in 2023-2025
        - 'no_support_query_overlap': no date appears in both support and query
        - 'k_matches_or_less_support_count': K >= len(target_support_dates)
        - 'k0_has_empty_support': K=0 implies empty support_dates
        - 'selection_uses_analysis_false': selection_uses_analysis is False
        - 'selection_uses_query_labels_false': selection_uses_query_labels is False
    """
    target_train_dates = manifest.get("target_train_dates", manifest.get("target_support_dates", []))
    support_dates = manifest.get("target_support_dates", target_train_dates)
    query_dates = manifest.get("target_eval_dates", manifest.get("target_query_dates", []))
    K = manifest.get("K")

    # Parse support dates
    target_train_years = set()
    for d in target_train_dates:
        year = int(d["date_str"].split("-")[0])
        target_train_years.add(year)

    # Parse legacy support dates
    support_years = set()
    for d in support_dates:
        year = int(d["date_str"].split("-")[0])
        support_years.add(year)

    # Parse query dates
    query_years = set()
    for d in query_dates:
        year = int(d["date_str"].split("-")[0])
        query_years.add(year)

    target_train_date_strs = set(d["date_str"] for d in target_train_dates)
    support_date_strs = set(d["date_str"] for d in support_dates)
    query_date_strs = set(d["date_str"] for d in query_dates)

    is_legacy = str(manifest.get("adaptation_protocol", "")).startswith("legacy")
    k_match = True
    if is_legacy and K is not None:
        k_match = len(support_dates) <= int(K)

    return {
        "target_train_in_train_year": target_train_years <= TARGET_TRAIN_YEARS,
        "support_in_support_year": support_years <= TARGET_TRAIN_YEARS,
        "query_in_query_years": query_years <= TARGET_EVAL_YEARS,
        "no_target_train_eval_overlap": len(target_train_date_strs & query_date_strs) == 0,
        "no_support_query_overlap": len(support_date_strs & query_date_strs) == 0,
        "k_matches_or_less_support_count": k_match,
        "k0_has_empty_support": K == 0 if is_legacy and len(support_dates) == 0 else True,
        "selection_uses_analysis_false": manifest.get("selection_uses_analysis", False) is False,
        "selection_uses_query_labels_false": manifest.get("selection_uses_query_labels", False) is False,
        "target_eval_labels_not_used_for_training": manifest.get("target_eval_labels_used_for_training", False) is False,
        "target_eval_labels_not_used_for_prompt": manifest.get("target_eval_labels_used_for_prompt", False) is False,
        "target_eval_labels_not_used_for_normalization": manifest.get("target_eval_labels_used_for_normalization", False) is False,
        "target_eval_labels_not_used_for_model_selection": manifest.get("target_eval_labels_used_for_model_selection", False) is False,
    }


def save_split_manifest(manifest: Dict, output_path: str) -> None:
    """Save split manifest to JSON.

    Args:
        manifest: Split manifest dict
        output_path: Output JSON path
    """
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: {output_path}")


def load_split_manifest(input_path: str) -> Dict:
    """Load split manifest from JSON.

    Args:
        input_path: Input JSON path

    Returns:
        Split manifest dict
    """
    with open(input_path, "r") as f:
        return json.load(f)


def generate_split_summary_markdown(splits: List[Dict], output_path: str) -> None:
    """Generate human-readable markdown summary of all splits.

    Args:
        splits: List of split manifest dicts
        output_path: Output markdown path
    """
    lines = [
        "# US Leave-One-Region-Out Target-Train Splits Summary",
        "",
        f"**Benchmark**: {splits[0]['benchmark_id']}",
        f"**Protocol**: {splits[0]['protocol_version']}",
        f"**Country**: {splits[0]['country_id']}",
        f"**Total splits**: {len(splits)}",
        "",
        "## Period Definitions",
        f"- source_fit: {splits[0]['source_fit_period']}",
        f"- source_val: {splits[0]['source_val_period']}",
        f"- target_train/adaptation: {splits[0].get('target_train_period', splits[0].get('target_support_period'))}",
        f"- target_eval: {splits[0].get('target_eval_period', splits[0].get('target_query_period'))}",
        "",
        "## Split Overview",
        "",
        "| Target | Sources | Adaptation | Legacy K | Seed | "
        "Source Cycles | Val Cycles | Target Train Cycles | Eval Cycles | "
        "Uses Analysis | Uses Labels |",
        "|--------|---------|------------|----------|------|"
        "--------------|-----------|---------------------|-------------|"
        "--------------|--------------|",
    ]

    for s in splits:
        src_str = "+".join(s["source_region_ids"])
        lines.append(
            f"| {s['target_region_id']} | {src_str} | "
            f"{s.get('adaptation_setting', 'legacy_few_shot')} | "
            f"{s.get('K', 'legacy_none')} | {s['seed']} | "
            f"{s['source_train_cycle_count']:,} | "
            f"{s.get('source_val_cycle_count', 0):,} | "
            f"{s.get('target_train_cycle_count', s.get('target_support_cycle_count', 0))} | "
            f"{s.get('target_eval_cycle_count', s.get('target_query_cycle_count', 0)):,} | "
            f"{s['selection_uses_analysis']} | "
            f"{s['selection_uses_query_labels']} |"
        )

    lines.append("")
    lines.append("## No-Leakage Declaration")
    lines.append("")
    lines.append("Main target adaptation uses the full frozen 2015-2021 target_train period.")
    lines.append("Legacy support dates, if present, are selected ONLY via calendar/data-availability rules.")
    lines.append("")
    lines.append("**NOT** via:")
    lines.append("- Analysis increment values")
    lines.append("- Model errors")
    lines.append("- Target query label distribution")
    lines.append("- Future query statistics")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


def aggregate_split_statistics(splits: List[Dict]) -> Dict:
    """Aggregate statistics across all splits.

    Args:
        splits: List of split manifest dicts

    Returns:
        Dict with aggregated statistics
    """
    stats = {
        "total_splits": len(splits),
        "K_values": sorted(set(s.get("K") for s in splits), key=lambda v: (-1 if v is None else v)),
        "adaptation_settings": sorted(set(s.get("adaptation_setting", "legacy_few_shot") for s in splits)),
        "seeds": sorted(set(s["seed"] for s in splits)),
        "regions": sorted(set(s["target_region_id"] for s in splits)),
        "total_source_cycles": sum(s["source_train_cycle_count"] for s in splits),
        "total_target_train_cycles": sum(s.get("target_train_cycle_count", s.get("target_support_cycle_count", 0)) for s in splits),
        "total_support_cycles": sum(s.get("target_support_cycle_count", 0) for s in splits),
        "total_eval_cycles": sum(s.get("target_eval_cycle_count", s.get("target_query_cycle_count", 0)) for s in splits),
        "total_query_cycles": sum(s.get("target_query_cycle_count", 0) for s in splits),
    }
    return stats
