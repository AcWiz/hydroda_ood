"""Split Manifest Generation and Validation.

No-leakage declaration:
    Manifests are generated using only:
    - Temporal constraints (date ranges from kdate_protocol.yaml)
    - Region masks from artifacts/regions/US_region_masks.nc
    - K-date selection via calendar rules (no analysis/model errors)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

import numpy as np


# Period strings derived from kdate_protocol.yaml
# Protocol V4-final:
#   source_fit:    2015-01-01 to 2020-12-31
#   source_val:    2021-01-01 to 2021-12-31
#   target_support: 2022-01-01 to 2022-12-31
#   target_query:   2023-01-01 to 2025-12-31
PERIODS = {
    "source_fit": "2015-01-01 to 2020-12-31",
    "source_val": "2021-01-01 to 2021-12-31",
    "target_support": "2022-01-01 to 2022-12-31",
    "target_query": "2023-01-01 to 2025-12-31",
}


def create_split_manifest(
    target_region: str,
    source_regions: List[str],
    K: int,
    seed: int,
    source_train_dates: List[dict],
    support_dates: List[dict],
    query_dates: List[dict],
    country_id: str = "US",
    benchmark_id: str = "hydroda_ood_us_v1",
    protocol_version: str = "kdate_protocol_v2",
) -> Dict:
    """Create a split manifest dict for LORO evaluation.

    Args:
        target_region: Target region ID (e.g., "US-R1")
        source_regions: List of source region IDs (e.g., ["US-R2", ..., "US-R6"])
        K: Number of support dates (0, 4, 12, 24)
        seed: Random seed (0-9)
        source_train_dates: List of dicts with time_index, date_str, datetime_str
        support_dates: List of dicts for support dates (empty for K=0)
        query_dates: List of dicts for query dates
        country_id: Country identifier (default "US")
        benchmark_id: Benchmark identifier
        protocol_version: Protocol version string

    Returns:
        Manifest dict with all required fields per kdate_protocol.yaml
    """
    manifest = {
        "benchmark_id": benchmark_id,
        "protocol_version": protocol_version,
        "country_id": country_id,
        "source_fit_period": PERIODS["source_fit"],
        "source_val_period": PERIODS["source_val"],
        "target_support_period": PERIODS["target_support"],
        "target_query_period": PERIODS["target_query"],
        "target_region_id": target_region,
        "source_region_ids": source_regions,
        "K": K,
        "seed": seed,
        "source_train_dates": source_train_dates,
        "target_support_dates": support_dates,
        "target_query_dates": query_dates,
        "source_train_cycle_count": len(source_train_dates),
        "target_support_cycle_count": len(support_dates),
        "target_query_cycle_count": len(query_dates),
        "selection_uses_analysis": False,
        "selection_uses_query_labels": False,
        "created_by": "build_kdate_splits.py",
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }

    return manifest


def validate_no_leakage(manifest: Dict) -> Dict[str, bool]:
    """Validate no-leakage constraints in a split manifest.

    Args:
        manifest: Split manifest dict

    Returns:
        Dict of validation results with keys:
        - 'support_in_support_year': target_support_dates are in 2022
        - 'query_in_query_years': target_query_dates are in 2023-2025
        - 'no_support_query_overlap': no date appears in both support and query
        - 'k_matches_or_less_support_count': K >= len(target_support_dates)
        - 'k0_has_empty_support': K=0 implies empty support_dates
        - 'selection_uses_analysis_false': selection_uses_analysis is False
        - 'selection_uses_query_labels_false': selection_uses_query_labels is False
    """
    support_dates = manifest["target_support_dates"]
    query_dates = manifest["target_query_dates"]
    K = manifest["K"]

    # Parse support dates
    support_years = set()
    for d in support_dates:
        year = int(d["date_str"].split("-")[0])
        support_years.add(year)

    # Parse query dates
    query_years = set()
    for d in query_dates:
        year = int(d["date_str"].split("-")[0])
        query_years.add(year)

    support_date_strs = set(d["date_str"] for d in support_dates)
    query_date_strs = set(d["date_str"] for d in query_dates)

    return {
        "support_in_support_year": support_years <= {2022},
        "query_in_query_years": query_years <= {2023, 2024, 2025},
        "no_support_query_overlap": len(support_date_strs & query_date_strs) == 0,
        "k_matches_or_less_support_count": len(support_dates) <= K,
        "k0_has_empty_support": K == 0 if len(support_dates) == 0 else True,
        "selection_uses_analysis_false": manifest.get("selection_uses_analysis", False) is False,
        "selection_uses_query_labels_false": manifest.get("selection_uses_query_labels", False) is False,
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
        "# US Leave-One-Region-Out K-Date Splits Summary",
        "",
        f"**Benchmark**: {splits[0]['benchmark_id']}",
        f"**Protocol**: {splits[0]['protocol_version']}",
        f"**Country**: {splits[0]['country_id']}",
        f"**Total splits**: {len(splits)}",
        "",
        "## Period Definitions",
        f"- source_fit: {splits[0]['source_fit_period']}",
        f"- source_val: {splits[0]['source_val_period']}",
        f"- target_support: {splits[0]['target_support_period']}",
        f"- target_query: {splits[0]['target_query_period']}",
        "",
        "## Split Overview",
        "",
        "| Target | Sources | K | Seed | "
        "Source Cycles | Support Cycles | Query Cycles | "
        "Uses Analysis | Uses Labels |",
        "|--------|---------|---|------|"
        "--------------|---------------|--------------|"
        "--------------|--------------|",
    ]

    for s in splits:
        src_str = "+".join(s["source_region_ids"])
        lines.append(
            f"| {s['target_region_id']} | {src_str} | "
            f"{s['K']} | {s['seed']} | "
            f"{s['source_train_cycle_count']:,} | "
            f"{s['target_support_cycle_count']} | "
            f"{s['target_query_cycle_count']:,} | "
            f"{s['selection_uses_analysis']} | "
            f"{s['selection_uses_query_labels']} |"
        )

    lines.append("")
    lines.append("## No-Leakage Declaration")
    lines.append("")
    lines.append("Support dates are selected ONLY via:")
    lines.append("- Calendar constraints (quarter/month/half-month rules)")
    lines.append("- Time availability in 2022")
    lines.append("- base_valid_mask coverage threshold")
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
        "K_values": sorted(set(s["K"] for s in splits)),
        "seeds": sorted(set(s["seed"] for s in splits)),
        "regions": sorted(set(s["target_region_id"] for s in splits)),
        "total_source_cycles": sum(s["source_train_cycle_count"] for s in splits),
        "total_support_cycles": sum(s["target_support_cycle_count"] for s in splits),
        "total_query_cycles": sum(s["target_query_cycle_count"] for s in splits),
    }
    return stats