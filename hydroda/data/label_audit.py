"""
Phase 1C: Label Availability Audit Module

Verifies that frozen split timestamps have valid DA analysis labels.
Report-only: no model training, no auto-resampling.

Scientific definitions:
  labeled_da_cycle: analysis present + active pixels + increment != 0
  forecast_cycle: analysis entirely NaN/missing (analysis_surface finite ratio < 1%)
  no_assimilation_cycle: >=99% of active pixels show analysis ≈ forecast (copy-fill)
  near_zero_increment_cycle: mean_abs_increment < 1e-5 m³/m³
"""

from __future__ import annotations

import json
import numpy as np
import xarray as xr
from dataclasses import dataclass, field, asdict
from datetime import datetime
from hydroda.utils.runtime import get_timestamp
from typing import Dict, List, Any, Set
from collections import defaultdict
from functools import reduce


class _NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def encode(self, o):
        return super().encode(self._sanitize(o))

    def _sanitize(self, o):
        if isinstance(o, dict):
            return {k: self._sanitize(v) for k, v in o.items()}
        elif isinstance(o, (list, tuple)):
            return [self._sanitize(x) for x in o]
        elif isinstance(o, (np.integer,)):
            return int(o)
        elif isinstance(o, (np.floating,)):
            return float(o)
        elif isinstance(o, (np.bool_,)):
            return bool(o)
        elif isinstance(o, np.ndarray):
            return self._sanitize(o.tolist())
        elif isinstance(o, (set, frozenset)):
            return sorted(list(o), key=repr)
        else:
            return o

    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._sanitize(o), _one_shot=_one_shot)


# Channel indices from netcdf_audit_US.json
_SURF_FORECAST_IDX = 0
_ROOT_FORECAST_IDX = 1
_SURF_ANALYSIS_IDX = 0
_ROOT_ANALYSIS_IDX = 1
_BASE_VALID_MASK_IDX = 11

# Thresholds from plan
_FORECAST_ONLY_FINITE_RATIO_THRESHOLD = 0.01  # analysis_surface finite ratio < 1%
_NO_ASSIMILATION_RATIO_THRESHOLD = 0.99      # >=99% of active pixels show analysis ≈ forecast
_NEAR_ZERO_INCREMENT_THRESHOLD = 1e-5        # m³/m³
_BASE_VALID_MASK_COVERAGE_MIN = 0.01         # base_valid_mask_coverage > 1% to be labeled


@dataclass
class TimestampStats:
    """Per-timestamp statistics for label availability audit."""
    time_index: int
    # Finite ratios
    forecast_surface_finite_ratio: float
    forecast_rootzone_finite_ratio: float
    analysis_surface_finite_ratio: float
    analysis_rootzone_finite_ratio: float
    # Mask coverage
    base_valid_mask_coverage: float
    # Increment stats
    increment_surface_mean_abs: float
    increment_surface_rmse: float
    increment_rootzone_mean_abs: float
    increment_rootzone_rmse: float
    # Classification flags
    is_forecast_only: bool
    is_no_assimilation: bool
    is_near_zero_increment: bool
    is_base_valid_mask_near_zero: bool
    # Extra context
    analysis_equals_forecast_ratio: float


@dataclass
class YearMonthStats:
    """Aggregated statistics per (year, month)."""
    year: int
    month: int
    total: int = 0
    forecast_only: int = 0
    labeled: int = 0
    active: int = 0
    no_assimilation: int = 0
    near_zero: int = 0
    median_base_valid_mask_coverage: float = 0.0
    zero_increment_fraction: float = 0.0


@dataclass
class SplitVerificationResult:
    """Verification result for a single frozen split."""
    split_id: str
    target_region_id: str
    K: int
    seed: int
    # Support set stats
    support_total: int = 0
    support_labeled_count: int = 0
    support_forecast_only_count: int = 0
    support_no_assimilation_count: int = 0
    support_near_zero_count: int = 0
    support_all_labeled: bool = False
    # Query set stats
    query_total: int = 0
    query_labeled_count: int = 0
    query_forecast_only_count: int = 0
    query_no_assimilation_count: int = 0
    query_near_zero_count: int = 0
    query_all_labeled: bool = False
    # Summary
    pass_phase3a_gate: bool = False


def compute_timestamp_stats(
    chunk_input: np.ndarray,
    chunk_target: np.ndarray,
    time_indices: List[int],
) -> List[TimestampStats]:
    """Compute per-timestamp statistics for a chunk.

    Args:
        chunk_input: np.ndarray of shape [chunk_size, 12, H, W]
        chunk_target: np.ndarray of shape [chunk_size, 4, H, W]
        time_indices: List of time indices for this chunk

    Returns:
        List of TimestampStats, one per timestamp
    """
    chunk_size = chunk_input.shape[0]
    H, W = chunk_input.shape[2], chunk_input.shape[3]
    total_pixels = H * W

    results = []
    for i in range(chunk_size):
        ti = time_indices[i]

        # Forecast arrays
        forecast_surf = chunk_input[i, _SURF_FORECAST_IDX]
        forecast_root = chunk_input[i, _ROOT_FORECAST_IDX]

        # Analysis arrays
        analysis_surf = chunk_target[i, _SURF_ANALYSIS_IDX]
        analysis_root = chunk_target[i, _ROOT_ANALYSIS_IDX]

        # Base valid mask (channel 11)
        base_valid_mask_raw = chunk_input[i, _BASE_VALID_MASK_IDX]

        # Finite ratios for forecasts
        forecast_surf_finite = np.isfinite(forecast_surf)
        forecast_root_finite = np.isfinite(forecast_root)
        forecast_surf_finite_ratio = forecast_surf_finite.mean()
        forecast_root_finite_ratio = forecast_root_finite.mean()

        # Finite ratios for analysis
        analysis_surf_finite = np.isfinite(analysis_surf)
        analysis_root_finite = np.isfinite(analysis_root)
        analysis_surf_finite_ratio = analysis_surf_finite.mean()
        analysis_root_finite_ratio = analysis_root_finite.mean()

        # base_valid_mask coverage: fraction of mask > 0.5
        base_valid_mask_coverage = (base_valid_mask_raw > 0.5).mean()

        # Active pixels: where base_valid_mask > 0.5 AND forecast is finite
        active_mask = (base_valid_mask_raw > 0.5) & forecast_surf_finite

        # Compute increments for active pixels
        if active_mask.sum() > 0:
            incr_surf = analysis_surf[active_mask] - forecast_surf[active_mask]
            incr_root = analysis_root[active_mask] - forecast_root[active_mask]

            increment_surface_mean_abs = float(np.abs(incr_surf).mean())
            increment_surface_rmse = float(np.sqrt(np.mean(incr_surf ** 2)))
            increment_rootzone_mean_abs = float(np.abs(incr_root).mean())
            increment_rootzone_rmse = float(np.sqrt(np.mean(incr_root ** 2)))

            # analysis ≈ forecast ratio (no assimilation detection)
            diff = np.abs(analysis_surf[active_mask] - forecast_surf[active_mask])
            analysis_equals_forecast_ratio = float((diff < 1e-6).mean())
        else:
            increment_surface_mean_abs = 0.0
            increment_surface_rmse = 0.0
            increment_rootzone_mean_abs = 0.0
            increment_rootzone_rmse = 0.0
            analysis_equals_forecast_ratio = 0.0

        # Classification flags
        is_forecast_only = analysis_surf_finite_ratio < _FORECAST_ONLY_FINITE_RATIO_THRESHOLD
        is_no_assimilation = analysis_equals_forecast_ratio >= _NO_ASSIMILATION_RATIO_THRESHOLD
        is_near_zero_increment = increment_surface_mean_abs < _NEAR_ZERO_INCREMENT_THRESHOLD
        is_base_valid_mask_near_zero = base_valid_mask_coverage < _BASE_VALID_MASK_COVERAGE_MIN

        results.append(TimestampStats(
            time_index=ti,
            forecast_surface_finite_ratio=forecast_surf_finite_ratio,
            forecast_rootzone_finite_ratio=forecast_root_finite_ratio,
            analysis_surface_finite_ratio=analysis_surf_finite_ratio,
            analysis_rootzone_finite_ratio=analysis_root_finite_ratio,
            base_valid_mask_coverage=base_valid_mask_coverage,
            increment_surface_mean_abs=increment_surface_mean_abs,
            increment_surface_rmse=increment_surface_rmse,
            increment_rootzone_mean_abs=increment_rootzone_mean_abs,
            increment_rootzone_rmse=increment_rootzone_rmse,
            is_forecast_only=is_forecast_only,
            is_no_assimilation=is_no_assimilation,
            is_near_zero_increment=is_near_zero_increment,
            is_base_valid_mask_near_zero=is_base_valid_mask_near_zero,
            analysis_equals_forecast_ratio=analysis_equals_forecast_ratio,
        ))

    return results


def accumulate_labeled_cycles(all_stats: List[TimestampStats]) -> Set[int]:
    """Identify labeled DA cycles.

    A cycle is labeled if:
      - NOT forecast_only
      - base_valid_mask_coverage > 0.01
      - NOT no_assimilation

    Args:
        all_stats: List of TimestampStats for all timestamps

    Returns:
        Set of labeled time indices
    """
    labeled = set()
    for stats in all_stats:
        if (
            not stats.is_forecast_only
            and stats.base_valid_mask_coverage > _BASE_VALID_MASK_COVERAGE_MIN
            and not stats.is_no_assimilation
        ):
            labeled.add(stats.time_index)
    return labeled


def compute_year_month_stats(
    all_stats: List[TimestampStats],
    time_coords: np.ndarray,
) -> Dict[tuple, YearMonthStats]:
    """Aggregate timestamp stats by (year, month).

    Args:
        all_stats: List of TimestampStats for all timestamps
        time_coords: array of time values in seconds since 1970-01-01

    Returns:
        Dict mapping (year, month) -> YearMonthStats
    """
    # Build time_index -> datetime mapping
    # time_coords[i] is seconds since 1970-01-01
    # 2015-04-01 00:00:00 UTC = 1427846400
    from datetime import timezone
    ym_stats: Dict[tuple, YearMonthStats] = defaultdict(lambda: YearMonthStats(year=0, month=0))

    # Build lookup
    idx_to_ym = {}
    for i, t in enumerate(time_coords):
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        ym = (dt.year, dt.month)
        idx_to_ym[i] = ym

    # Group stats by year-month
    ym_counts: Dict[tuple, dict] = defaultdict(lambda: defaultdict(list))

    for stats in all_stats:
        ym = idx_to_ym.get(stats.time_index, (0, 0))
        ym_counts[ym]["total"].append(stats.time_index)
        if stats.is_forecast_only:
            ym_counts[ym]["forecast_only"].append(stats.time_index)
        if stats.is_no_assimilation:
            ym_counts[ym]["no_assimilation"].append(stats.time_index)
        if stats.is_near_zero_increment:
            ym_counts[ym]["near_zero"].append(stats.time_index)
        ym_counts[ym]["mask_coverage"].append(stats.base_valid_mask_coverage)

    # Compute labeled for each ym
    labeled = accumulate_labeled_cycles(all_stats)

    for ym, counts in ym_counts.items():
        year, month = ym
        total = len(counts["total"])
        forecast_only = len(counts["forecast_only"])
        no_assimilation = len(counts["no_assimilation"])
        near_zero = len(counts["near_zero"])
        mask_coverages = counts["mask_coverage"]

        labeled_count = sum(1 for ti in counts["total"]
                           if ti in labeled and ti not in counts["forecast_only"])
        active_count = sum(1 for ti in counts["total"]
                          if ti not in counts["forecast_only"]
                          and ti not in counts["no_assimilation"])

        median_mask = float(np.median(mask_coverages)) if mask_coverages else 0.0
        zero_frac = near_zero / total if total > 0 else 0.0

        ym_stats[ym] = YearMonthStats(
            year=year,
            month=month,
            total=total,
            forecast_only=forecast_only,
            labeled=labeled_count,
            active=active_count,
            no_assimilation=no_assimilation,
            near_zero=near_zero,
            median_base_valid_mask_coverage=median_mask,
            zero_increment_fraction=zero_frac,
        )

    return dict(ym_stats)


def verify_frozen_splits(
    splits_data: dict,
    all_stats: List[TimestampStats],
    labeled_cycles: Set[int],
) -> List[SplitVerificationResult]:
    """Verify all 240 frozen splits against label availability.

    Args:
        splits_data: Loaded JSON from US_loro_kdate_splits.json
        all_stats: List of TimestampStats for all timestamps
        labeled_cycles: Set of labeled time indices

    Returns:
        List of SplitVerificationResult for all 240 splits
    """
    # Build time_index -> stats lookup
    stats_map = {s.time_index: s for s in all_stats}

    results = []
    for split in splits_data["splits"]:
        split_id = (
            f"{split['target_region_id']}-K{split['K']}-S{split['seed']}"
        )

        # Support dates
        support_tis = [d["time_index"] for d in split.get("target_support_dates", [])]
        # Query dates
        query_tis = [d["time_index"] for d in split.get("target_query_dates", [])]

        # Count support stats
        support_labeled = sum(1 for ti in support_tis if ti in labeled_cycles)
        support_forecast_only = sum(1 for ti in support_tis
                                   if ti in stats_map and stats_map[ti].is_forecast_only)
        support_no_assim = sum(1 for ti in support_tis
                              if ti in stats_map and stats_map[ti].is_no_assimilation)
        support_near_zero = sum(1 for ti in support_tis
                               if ti in stats_map and stats_map[ti].is_near_zero_increment)

        # Count query stats
        query_labeled = sum(1 for ti in query_tis if ti in labeled_cycles)
        query_forecast_only = sum(1 for ti in query_tis
                                 if ti in stats_map and stats_map[ti].is_forecast_only)
        query_no_assim = sum(1 for ti in query_tis
                            if ti in stats_map and stats_map[ti].is_no_assimilation)
        query_near_zero = sum(1 for ti in query_tis
                              if ti in stats_map and stats_map[ti].is_near_zero_increment)

        support_all_labeled = (support_labeled == len(support_tis)) if support_tis else True
        query_all_labeled = (query_labeled == len(query_tis)) if query_tis else True

        # Phase 3A gate: all support/query dates labeled AND no massive forecast_only in query
        pass_gate = (
            support_all_labeled
            and query_all_labeled
            and query_forecast_only == 0
        )

        results.append(SplitVerificationResult(
            split_id=split_id,
            target_region_id=split["target_region_id"],
            K=split["K"],
            seed=split["seed"],
            support_total=len(support_tis),
            support_labeled_count=support_labeled,
            support_forecast_only_count=support_forecast_only,
            support_no_assimilation_count=support_no_assim,
            support_near_zero_count=support_near_zero,
            support_all_labeled=support_all_labeled,
            query_total=len(query_tis),
            query_labeled_count=query_labeled,
            query_forecast_only_count=query_forecast_only,
            query_no_assimilation_count=query_no_assim,
            query_near_zero_count=query_near_zero,
            query_all_labeled=query_all_labeled,
            pass_phase3a_gate=pass_gate,
        ))

    return results


def generate_json_report(
    all_stats: List[TimestampStats],
    labeled_cycles: Set[int],
    ym_stats: Dict[tuple, YearMonthStats],
    split_results: List[SplitVerificationResult],
    time_coords: np.ndarray,
) -> dict:
    """Generate full JSON report.

    Due to size constraints, per-timestamp stats are summarized for first 1000
    timestamps with full detail, plus summary statistics.
    """
    # Summarize all_stats - only include first 1000 for JSON size
    stats_sample = all_stats[:1000]
    stats_summary = {
        "total_timestamps": len(all_stats),
        "per_timestamp_sample_count": len(stats_sample),
        "stats_sample": [asdict(s) for s in stats_sample],
        "total_labeled_cycles": len(labeled_cycles),
        "forecast_only_count": sum(1 for s in all_stats if s.is_forecast_only),
        "no_assimilation_count": sum(1 for s in all_stats if s.is_no_assimilation),
        "near_zero_increment_count": sum(1 for s in all_stats if s.is_near_zero_increment),
        "base_valid_mask_near_zero_count": sum(1 for s in all_stats if s.is_base_valid_mask_near_zero),
    }

    # Year-month aggregation
    ym_data = {}
    for ym, yms in sorted(ym_stats.items()):
        ym_data[f"{ym[0]}-{ym[1]:02d}"] = asdict(yms)

    # Split verification summary
    total_splits = len(split_results)
    pass_gate_count = sum(1 for r in split_results if r.pass_phase3a_gate)
    support_all_labeled_count = sum(1 for r in split_results if r.support_all_labeled)
    query_all_labeled_count = sum(1 for r in split_results if r.query_all_labeled)

    splits_summary = {
        "total_splits": total_splits,
        "pass_phase3a_gate_count": pass_gate_count,
        "fail_phase3a_gate_count": total_splits - pass_gate_count,
        "support_all_labeled_count": support_all_labeled_count,
        "query_all_labeled_count": query_all_labeled_count,
    }

    report = {
        "audit_type": "label_availability",
        "audit_timestamp": get_timestamp(),
        "total_timestamps": len(all_stats),
        "labeled_cycle_count": len(labeled_cycles),
        "labeled_cycle_fraction": len(labeled_cycles) / len(all_stats) if all_stats else 0.0,
        "stats_summary": stats_summary,
        "year_month_stats": ym_data,
        "split_verification_summary": splits_summary,
        "split_verification_results": [asdict(r) for r in split_results],
        "phase3a_gate_condition": (
            "All support/query dates labeled AND no forecast_only in query periods"
        ),
        "phase3a_gate_pass_rate": pass_gate_count / total_splits if total_splits else 0.0,
    }

    return report


def generate_markdown_report(
    all_stats: List[TimestampStats],
    labeled_cycles: Set[int],
    ym_stats: Dict[tuple, YearMonthStats],
    split_results: List[SplitVerificationResult],
    time_coords: np.ndarray,
) -> str:
    """Generate markdown report with executive summary."""
    total = len(all_stats)
    labeled = len(labeled_cycles)
    forecast_only = sum(1 for s in all_stats if s.is_forecast_only)
    no_assim = sum(1 for s in all_stats if s.is_no_assimilation)
    near_zero = sum(1 for s in all_stats if s.is_near_zero_increment)

    pass_gate = sum(1 for r in split_results if r.pass_phase3a_gate)
    total_splits = len(split_results)

    # Year-month table
    ym_rows = []
    for ym in sorted(ym_stats.keys()):
        yms = ym_stats[ym]
        ym_rows.append(
            f"| {yms.year}-{yms.month:02d} | {yms.total} | {yms.forecast_only} | "
            f"{yms.labeled} | {yms.no_assimilation} | {yms.near_zero} | "
            f"{yms.median_base_valid_mask_coverage:.4f} | {yms.zero_increment_fraction:.4f} |"
        )

    # Split verification table
    split_table = []
    for region in ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]:
        for K in [0, 4, 12, 24]:
            region_results = [r for r in split_results
                            if r.target_region_id == region and r.K == K]
            if not region_results:
                continue
            seeds_pass = sum(1 for r in region_results if r.pass_phase3a_gate)
            support_all_l = all(r.support_all_labeled for r in region_results)
            query_all_l = all(r.query_all_labeled for r in region_results)
            split_table.append(
                f"| {region} | K={K} | {seeds_pass}/10 | "
                f"{'✓' if support_all_l else '✗'} | {'✓' if query_all_l else '✗'} |"
            )

    md = f"""# Phase 1C: Label Availability Audit Report

## Executive Summary

| Metric | Value |
|--------|-------|
| Total timestamps | {total:,} |
| Labeled DA cycles | {labeled:,} ({100*labeled/total:.1f}%) |
| Forecast-only cycles | {forecast_only:,} ({100*forecast_only/total:.1f}%) |
| No-assimilation cycles | {no_assim:,} ({100*no_assim/total:.1f}%) |
| Near-zero increment cycles | {near_zero:,} ({100*near_zero/total:.1f}%) |

## Phase 3A Gate Status

**Pass rate: {pass_gate}/{total_splits} ({100*pass_gate/total_splits:.1f}%)**

Gate condition: All support/query dates labeled AND no forecast_only cycles in query periods.

## Year-by-Year Statistics

| Year-Month | Total | Forecast-only | Labeled | No-Assim | Near-Zero | Median Mask Cov | Zero-Incr Frac |
|------------|-------|---------------|---------|----------|-----------|-----------------|----------------|
{chr(10).join(ym_rows)}

## Split Verification Summary (by Region × K)

| Region | K | Pass Gate | Support All Labeled | Query All Labeled |
|--------|---|-----------|---------------------|-------------------|
{chr(10).join(split_table)}

## Detailed Findings

### Forecast-Only Cycles
Forecast-only cycles are timestamps where analysis data is entirely missing (analysis_surface finite ratio < 1%).
These cycles cannot be used for training or evaluation as they lack DA analysis labels.

### No-Assimilation Cycles
No-assimilation cycles are timestamps where >= 99% of active pixels show analysis ≈ forecast (copy-fill).
These cycles indicate failed or skipped assimilation — no actual data assimilation occurred.

### Near-Zero Increment Cycles
Near-zero increment cycles have mean_abs_increment < 1e-5 m³/m³.
These may indicate stable assimilation conditions or potential assimilation issues.

### base_valid_mask Coverage
The base_valid_mask (channel 11) coverage varies over time. Low coverage (< 1%) indicates
timestamps that cannot be used for effective learning.

## Recommendations

{'**Phase 3A is CLEARED** — all 240 frozen splits pass the label availability gate.' if pass_gate == total_splits else f'**Phase 3A has ISSUES** — {total_splits - pass_gate} splits fail the label availability gate. Human review required before proceeding.'}

---

*Generated: {get_timestamp()}*
"""

    return md


def plot_labeled_cycle_counts_by_year(
    all_stats: List[TimestampStats],
    labeled_cycles: Set[int],
    output_path: str,
) -> None:
    """Generate bar chart: year vs stacked counts (labeled, no_assimilation, forecast_only, near_zero).

    Args:
        all_stats: List of all TimestampStats
        labeled_cycles: Set of labeled time indices
        output_path: Path to save PNG figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("[WARN] matplotlib not available, skipping figure generation")
        return

    from datetime import timezone

    # Group by year
    from collections import defaultdict
    year_data = defaultdict(lambda: {
        "total": 0,
        "labeled": 0,
        "no_assimilation": 0,
        "forecast_only": 0,
        "near_zero": 0,
    })

    # We need time_coords to get years — we'll reconstruct from time_index
    # time_coords[i] gives seconds since 1970-01-01 for time index i
    # But we don't have time_coords here. We'll use a heuristic based on known range.
    # Actually, we need time_coords passed in. Let's check the function signature.

    # Note: This function needs time_coords. Let it be passed separately.
    # For now, just use the stats directly — we track time_index but not year.
    # This is a limitation. Let's make year extraction work from stats if available.

    # For the bar chart we need year. Since we don't store it in TimestampStats,
    # we'll need to compute it externally or add it. For simplicity, we'll do a
    # time-index based chart (index 0..N-1) grouped by year in the plot.

    # Actually, the plan says "year vs stacked counts". Let me re-read the plan.
    # The plan says to use time_coords to get year. Let me refactor this.

    # For now, skip actual plotting since we don't have time_coords in scope.
    # The CLI script will call this with proper arguments.
    pass


def plot_labeled_cycle_counts_by_year_v2(
    all_stats: List[TimestampStats],
    time_coords: np.ndarray,
    labeled_cycles: Set[int],
    output_path: str,
) -> None:
    """Generate bar chart: year vs stacked counts.

    Args:
        all_stats: List of all TimestampStats
        time_coords: array of time values in seconds since 1970-01-01
        labeled_cycles: Set of labeled time indices
        output_path: Path to save PNG figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("[WARN] matplotlib not available, skipping figure generation")
        return

    from datetime import timezone
    from collections import defaultdict

    # Build time_index -> year mapping
    idx_to_year = {}
    for i, t in enumerate(time_coords):
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        idx_to_year[i] = dt.year

    # Group by year
    year_data = defaultdict(lambda: {
        "total": 0,
        "labeled": 0,
        "no_assimilation": 0,
        "forecast_only": 0,
        "near_zero": 0,
    })

    for stats in all_stats:
        year = idx_to_year.get(stats.time_index, 0)
        yd = year_data[year]
        yd["total"] += 1
        if stats.time_index in labeled_cycles:
            yd["labeled"] += 1
        if stats.is_no_assimilation:
            yd["no_assimilation"] += 1
        if stats.is_forecast_only:
            yd["forecast_only"] += 1
        if stats.is_near_zero_increment:
            yd["near_zero"] += 1

    years = sorted(year_data.keys())
    if not years:
        print("[WARN] No year data found, skipping figure generation")
        return

    labeled_vals = [year_data[y]["labeled"] for y in years]
    no_assim_vals = [year_data[y]["no_assimilation"] for y in years]
    forecast_only_vals = [year_data[y]["forecast_only"] for y in years]
    near_zero_vals = [year_data[y]["near_zero"] for y in years]

    x = np.arange(len(years))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, labeled_vals, width, label="Labeled", color="#2ecc71")
    ax.bar(x, no_assim_vals, width, bottom=labeled_vals, label="No-Assimilation", color="#e74c3c")
    ax.bar(x, forecast_only_vals, width,
           bottom=np.array(labeled_vals) + np.array(no_assim_vals),
           label="Forecast-Only", color="#f39c12")
    ax.bar(x, near_zero_vals, width,
           bottom=np.array(labeled_vals) + np.array(no_assim_vals) + np.array(forecast_only_vals),
           label="Near-Zero Increment", color="#9b59b6")

    ax.set_xlabel("Year")
    ax.set_ylabel("Cycle Count")
    ax.set_title("Labeled DA Cycle Counts by Year")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved figure to {output_path}")
