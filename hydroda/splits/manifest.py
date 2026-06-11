"""Split manifest generation and validation.

No-leakage declaration:
    Manifests are generated using only:
    - Temporal constraints (date ranges from protocol specs)
    - Region masks from artifacts/regions/US_region_masks.nc
    - Target context dates from 2015-2021 input-side fields
    - K-shot support dates via calendar/data-availability rules
    - No analysis increment values or model errors
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np


# Period strings derived from protocol_v4.yaml
# Protocol V4.4 zero/few-shot generalization:
#   source_fit:    2015-01-01 to 2021-12-31
#   source_val:    2022-01-01 to 2022-12-31
#   target_context/support: 2015-01-01 to 2021-12-31
#   target_eval:   2023-01-01 to 2025-12-31
PERIODS = {
    "source_fit": "2015-01-01 to 2021-12-31",
    "source_val": "2022-01-01 to 2022-12-31",
    "source_test": "2023-01-01 to 2025-12-31",
    "target_train": "2015-01-01 to 2021-12-31",
    "target_adapt": "2015-01-01 to 2021-12-31",
    "target_context": "2015-01-01 to 2021-12-31",
    "target_support": "2015-01-01 to 2021-12-31",
    "target_val": "2022-01-01 to 2022-12-31",
    "target_eval": "2023-01-01 to 2025-12-31",
    # Deprecated aliases retained for older artifacts and scripts.
    "target_support": "2015-01-01 to 2021-12-31",
    "target_query": "2023-01-01 to 2025-12-31",
}

PROTOCOL_FREEZE_ID = "hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025"
TARGET_TRAIN_YEARS = set(range(2015, 2022))
TARGET_CONTEXT_YEARS = set(range(2015, 2022))
SOURCE_VAL_YEARS = {2022}
SOURCE_TEST_YEARS = {2023, 2024, 2025}
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
    source_test_dates: List[dict] | None = None,
    target_context_dates: Optional[List[dict]] = None,
    target_train_dates: Optional[List[dict]] = None,
    adaptation_setting: Optional[str] = None,
    allow_legacy_full_target_train: bool = False,
    country_id: str = "US",
    benchmark_id: str = "hydroda_ood_us_v1",
    protocol_version: str = "v4.4-zero-few-shot-generalization",
) -> Dict:
    """Create a split manifest dict for LORO evaluation.

    Args:
        target_region: Target region ID (e.g., "US-R1")
        source_regions: List of source region IDs (e.g., ["US-R2", ..., "US-R6"])
        K: Main zero/few-shot labeled support budget. Must be 0, 4, or 12.
        seed: Random seed/run seed.
        source_train_dates: List of dicts with time_index, date_str, datetime_str
        support_dates: K labeled target support cycles from 2015-2021.
        query_dates: List of dicts for target eval/query dates
        source_val_dates: List of dicts for source validation (2022). If None, defaults to [].
        source_test_dates: List of dicts for source test (2023-2025). If None, defaults to [].
        target_context_dates: Input-side target context dates from 2015-2021.
        target_train_dates: Legacy alias. In the main protocol this mirrors support_dates.
        adaptation_setting: Main adaptation setting, e.g. "zero_shot_context".
        country_id: Country identifier (default "US")
        benchmark_id: Benchmark identifier
        protocol_version: Protocol version string

    Returns:
        Manifest dict with all required fields per kdate_protocol.yaml
    """
    source_train_dates = list(source_train_dates or [])
    source_val_dates = list(source_val_dates or [])
    source_test_dates = list(source_test_dates or [])
    query_dates = list(query_dates or [])
    raw_support_dates = None if support_dates is None else list(support_dates)
    raw_target_train_dates = None if target_train_dates is None else list(target_train_dates)

    main_K = int(K) if K is not None else None
    if adaptation_setting is None:
        if main_K is None:
            main_K = 0
            adaptation_setting = "zero_shot_context"
        elif main_K == 0:
            adaptation_setting = "zero_shot_context"
        else:
            adaptation_setting = f"few_shot_k{main_K}"

    is_legacy_kshot = str(adaptation_setting).startswith("legacy_few_shot")
    is_legacy_full_target = adaptation_setting == "target_full_train"
    is_main_few_shot = adaptation_setting in {"zero_shot_context", "few_shot_k4", "few_shot_k12"}
    expected_main_setting = {0: "zero_shot_context", 4: "few_shot_k4", 12: "few_shot_k12"}.get(main_K)
    if (
        not is_legacy_kshot
        and not is_legacy_full_target
        and (main_K is None or main_K not in {0, 4, 12} or not is_main_few_shot)
    ):
        raise ValueError(
            "Unsupported main zero/few-shot K/adaptation_setting. "
            "Use K in {0,4,12} with zero_shot_context/few_shot_k4/few_shot_k12, "
            "or pass an explicit legacy_* setting for internal ablations."
        )
    if is_main_few_shot and expected_main_setting is not None and adaptation_setting != expected_main_setting:
        raise ValueError(
            f"K={main_K} does not match adaptation_setting={adaptation_setting!r}; "
            f"expected {expected_main_setting!r}"
        )

    if is_legacy_full_target and not allow_legacy_full_target_train:
        raise ValueError(
            "target_full_train is legacy/internal. Pass allow_legacy_full_target_train=True "
            "only for historical reproduction."
        )

    if target_context_dates is None:
        if is_legacy_full_target:
            target_context_dates = list(raw_target_train_dates or raw_support_dates or [])
        else:
            target_context_dates = list(raw_target_train_dates or raw_support_dates or [])
    else:
        target_context_dates = list(target_context_dates)

    if raw_target_train_dates is None:
        target_train_dates = list(target_context_dates if is_legacy_full_target else (raw_support_dates or []))
    else:
        target_train_dates = raw_target_train_dates

    if raw_support_dates is None:
        support_dates = list(target_train_dates) if is_legacy_full_target else []
    else:
        support_dates = raw_support_dates

    target_eval_dates = list(query_dates)
    if is_main_few_shot:
        adaptation_protocol = "zero_few_shot_generalization"
    elif is_legacy_full_target:
        adaptation_protocol = "legacy_full_target_train"
    elif is_legacy_kshot:
        adaptation_protocol = "legacy_few_shot_ablation"
    else:
        adaptation_protocol = "legacy_internal"

    manifest = {
        "manifest_schema_version": "v4_4_zero_few_shot",
        "benchmark_id": benchmark_id,
        "protocol_version": protocol_version,
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "country_id": country_id,
        "source_fit_period": PERIODS["source_fit"],
        "source_val_period": PERIODS["source_val"],
        "source_test_period": PERIODS["source_test"],
        "target_context_period": PERIODS["target_context"],
        "target_support_period": PERIODS["target_support"],
        "target_train_period": PERIODS["target_train"],
        "target_adapt_period": PERIODS["target_adapt"],
        "target_val_period": PERIODS["target_val"],
        "target_eval_period": PERIODS["target_eval"],
        "target_query_period": PERIODS["target_query"],
        "target_region_id": target_region,
        "source_region_ids": source_regions,
        "adaptation_protocol": adaptation_protocol,
        "adaptation_setting": adaptation_setting,
        "K": main_K,
        "K_legacy": main_K if (is_legacy_kshot or is_legacy_full_target) else None,
        "seed": seed,
        "source_train_dates": source_train_dates,
        "source_val_dates": source_val_dates,
        "source_test_dates": source_test_dates,
        "target_context_dates": target_context_dates,
        "target_support_dates": support_dates,
        "target_train_dates": target_train_dates,
        "target_adaptation_dates": target_train_dates,
        "target_eval_dates": target_eval_dates,
        # Deprecated aliases retained for existing dataset/evaluation code.
        "target_query_dates": target_eval_dates,
        "source_train_cycle_count": len(source_train_dates),
        "source_val_cycle_count": len(source_val_dates),
        "source_test_cycle_count": len(source_test_dates),
        "target_context_cycle_count": len(target_context_dates),
        "target_support_cycle_count": len(support_dates),
        "target_train_cycle_count": len(target_train_dates),
        "target_adaptation_cycle_count": len(target_train_dates),
        "target_eval_cycle_count": len(target_eval_dates),
        "target_query_cycle_count": len(target_eval_dates),
        "source_train_dates_hash": _records_hash(source_train_dates),
        "source_val_dates_hash": _records_hash(source_val_dates),
        "source_test_dates_hash": _records_hash(source_test_dates),
        "target_context_dates_hash": _records_hash(target_context_dates),
        "target_support_dates_hash": _records_hash(support_dates),
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
        "model_selection_scope": "source_val_preregistered",
        "model_selection_source": "source_val_preregistered",
        "target_val_usage": "unused_in_main_protocol",
        "target_full_train_usage": "legacy_internal_only",
        "source_training_policy": "main_protocol_uses_source_fit_2015_2021_source_val_2022",
        "created_by": "build_zero_few_shot_splits.py",
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
    target_context_dates = manifest.get("target_context_dates", [])
    target_train_dates = manifest.get("target_train_dates", manifest.get("target_support_dates", []))
    support_dates = manifest.get("target_support_dates", target_train_dates)
    query_dates = manifest.get("target_eval_dates", manifest.get("target_query_dates", []))
    K = manifest.get("K")

    # Parse context dates
    target_context_years = set()
    for d in target_context_dates:
        year = int(d["date_str"].split("-")[0])
        target_context_years.add(year)

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

    adaptation_protocol = str(manifest.get("adaptation_protocol", ""))
    is_legacy = adaptation_protocol.startswith("legacy")
    is_legacy_full_target = adaptation_protocol == "legacy_full_target_train"
    k_match = True
    k0_empty = True
    if K is not None and not is_legacy_full_target:
        k_match = len(support_dates) <= int(K)
        if int(K) == 0:
            k0_empty = len(support_dates) == 0

    return {
        "target_train_in_train_year": target_train_years <= TARGET_TRAIN_YEARS,
        "target_context_in_context_year": target_context_years <= TARGET_CONTEXT_YEARS,
        "support_in_support_year": support_years <= TARGET_TRAIN_YEARS,
        "query_in_query_years": query_years <= TARGET_EVAL_YEARS,
        "no_target_train_eval_overlap": len(target_train_date_strs & query_date_strs) == 0,
        "no_support_query_overlap": len(support_date_strs & query_date_strs) == 0,
        "k_matches_or_less_support_count": k_match,
        "k0_has_empty_support": k0_empty,
        "selection_uses_analysis_false": manifest.get("selection_uses_analysis", False) is False,
        "selection_uses_query_labels_false": manifest.get("selection_uses_query_labels", False) is False,
        "target_eval_labels_not_used_for_training": manifest.get("target_eval_labels_used_for_training", False) is False,
        "target_eval_labels_not_used_for_prompt": manifest.get("target_eval_labels_used_for_prompt", False) is False,
        "target_eval_labels_not_used_for_normalization": manifest.get("target_eval_labels_used_for_normalization", False) is False,
        "target_eval_labels_not_used_for_model_selection": manifest.get("target_eval_labels_used_for_model_selection", False) is False,
        "target_val_unused_in_main_protocol": manifest.get("target_val_usage") == "unused_in_main_protocol",
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
        "# US Leave-One-Region-Out Zero/Few-Shot Splits Summary",
        "",
        f"**Benchmark**: {splits[0]['benchmark_id']}",
        f"**Protocol**: {splits[0]['protocol_version']}",
        f"**Country**: {splits[0]['country_id']}",
        f"**Total splits**: {len(splits)}",
        "",
        "## Period Definitions",
        f"- source_fit: {splits[0]['source_fit_period']}",
        f"- source_val: {splits[0]['source_val_period']}",
        f"- target_context: {splits[0].get('target_context_period', splits[0].get('target_train_period'))}",
        f"- target_support: {splits[0].get('target_support_period', splits[0].get('target_train_period'))}",
        f"- target_eval: {splits[0].get('target_eval_period', splits[0].get('target_query_period'))}",
        "",
        "## Split Overview",
        "",
        "| Target | Sources | Adaptation | Legacy K | Seed | "
        "Source Cycles | Val Cycles | Context Cycles | Support Cycles | Eval Cycles | "
        "Uses Analysis | Uses Labels |",
        "|--------|---------|------------|----------|------|"
        "--------------|-----------|----------------|----------------|-------------|"
        "--------------|--------------|",
    ]

    for s in splits:
        src_str = "+".join(s["source_region_ids"])
        lines.append(
            f"| {s['target_region_id']} | {src_str} | "
            f"{s.get('adaptation_setting', 'zero_shot_context')} | "
            f"{s.get('K', 'legacy_none')} | {s['seed']} | "
            f"{s['source_train_cycle_count']:,} | "
            f"{s.get('source_val_cycle_count', 0):,} | "
            f"{s.get('target_context_cycle_count', 0):,} | "
            f"{s.get('target_support_cycle_count', 0):,} | "
            f"{s.get('target_eval_cycle_count', s.get('target_query_cycle_count', 0)):,} | "
            f"{s['selection_uses_analysis']} | "
            f"{s['selection_uses_query_labels']} |"
        )

    lines.append("")
    lines.append("## No-Leakage Declaration")
    lines.append("")
    lines.append(
        "Main target prompting uses frozen 2015-2021 input-side target_context dates "
        "to build target-context monthly prompt prototypes."
    )
    lines.append("Eval months are used only as known month-of-year seasonal phase selectors.")
    lines.append("K-shot support dates are selected ONLY via calendar/data-availability rules.")
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
        "adaptation_settings": sorted(set(s.get("adaptation_setting", "zero_shot_context") for s in splits)),
        "seeds": sorted(set(s["seed"] for s in splits)),
        "regions": sorted(set(s["target_region_id"] for s in splits)),
        "total_source_cycles": sum(s["source_train_cycle_count"] for s in splits),
        "total_target_train_cycles": sum(s.get("target_train_cycle_count", s.get("target_support_cycle_count", 0)) for s in splits),
        "total_support_cycles": sum(s.get("target_support_cycle_count", 0) for s in splits),
        "total_eval_cycles": sum(s.get("target_eval_cycle_count", s.get("target_query_cycle_count", 0)) for s in splits),
        "total_query_cycles": sum(s.get("target_query_cycle_count", 0) for s in splits),
    }
    return stats
    # Parse context dates
    target_context_years = set()
    for d in target_context_dates:
        year = int(d["date_str"].split("-")[0])
        target_context_years.add(year)
