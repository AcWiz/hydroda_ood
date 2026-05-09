"""
Phase 0A: DA.nc Forensic Audit Module

Comprehensive audit of DA.nc multi-channel array structure using chunked reads.
Produces per-channel stats, per-timestamp cycle validity, and Markdown report.

Architecture:
- compute_channel_statistics: chunked per-channel stats (min/max/mean/std/percentiles)
- audit_time_axis: full time-axis audit (timestamps, hour dist, missing days)
- audit_analysis_labels: per-timestamp DA cycle validity + increment stats
- generate_markdown_report: structured Markdown output
- NumpySafeEncoder: JSON serialization for numpy types
"""

import json
import numpy as np
import xarray as xr
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter


# =============================================================================
# JSON Encoder
# =============================================================================

class NumpySafeEncoder(json.JSONEncoder):
    """Handles numpy types in JSON serialization."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# =============================================================================
# Channel Statistics
# =============================================================================

def compute_channel_statistics(
    ds: xr.Dataset, chunk_size: int = 50
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Compute per-channel statistics across ALL time steps using chunked reads.

    For each channel in input and target arrays:
    - finite_ratio, min, max, mean, std, p01, p05, p50, p95, p99
    - mask-like detection: finite_ratio < 0.05 or unique_values <= 2

    Returns:
        channel_stats: list of dicts with per-channel statistics
        mask_like_channels: list of channel names flagged as mask-like
    """
    channel_stats = []
    mask_like_channels = []

    input_coord = ds.coords["variable_input"]
    target_coord = ds.coords["variable_target"]

    # Process input channels
    for arr_name in ["input", "target"]:
        coord = input_coord if arr_name == "input" else target_coord
        channel_names = list(coord.values)
        n_channels = len(channel_names)
        n_time = ds.sizes["time"]

        for ch_idx in range(n_channels):
            ch_name = channel_names[ch_idx]

            # Accumulate stats in chunks
            finite_count = 0
            total_count = 0
            chunk_mins = []
            chunk_maxs = []
            chunk_means = []
            all_values = []  # for percentile calculation

            n_chunks = (n_time + chunk_size - 1) // chunk_size

            for chunk_start in range(0, n_time, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_time)
                slab = ds[arr_name].values[chunk_start:chunk_end, ch_idx, :, :]

                # Finite count
                finite_mask = np.isfinite(slab)
                finite_count += finite_mask.sum()
                total_count += slab.size

                # Min/max per chunk
                chunk_finite = slab[finite_mask]
                if chunk_finite.size > 0:
                    chunk_mins.append(chunk_finite.min())
                    chunk_maxs.append(chunk_finite.max())
                    chunk_means.append(chunk_finite.mean())

                    # Sample for percentile (subsample to avoid memory issues)
                    if chunk_finite.size > 5000:
                        idx = np.random.choice(chunk_finite.size, 5000, replace=False)
                        all_values.append(chunk_finite[idx])
                    else:
                        all_values.append(chunk_finite.flatten())

            finite_ratio = finite_count / total_count if total_count > 0 else 0.0

            # Overall min/max from chunk extremes
            overall_min = min(chunk_mins) if chunk_mins else np.nan
            overall_max = max(chunk_maxs) if chunk_maxs else np.nan

            # Mean and std
            if all_values:
                merged = np.concatenate([v.flatten() for v in all_values])
                overall_mean = float(np.mean(merged))
                overall_std = float(np.std(merged))
                p01 = float(np.percentile(merged, 1))
                p05 = float(np.percentile(merged, 5))
                p50 = float(np.percentile(merged, 50))
                p95 = float(np.percentile(merged, 95))
                p99 = float(np.percentile(merged, 99))
                unique_values = len(np.unique(merged))
            else:
                overall_mean = np.nan
                overall_std = np.nan
                p01 = p05 = p50 = p95 = p99 = np.nan
                unique_values = 0

            # Mask-like detection
            is_mask_like = (finite_ratio < 0.05) or (unique_values <= 2)
            if is_mask_like:
                mask_like_channels.append(ch_name)

            stat = {
                "array": arr_name,
                "channel_index": ch_idx,
                "variable_name": ch_name,
                "finite_ratio": round(finite_ratio, 6),
                "min": round(float(overall_min), 6) if not np.isnan(overall_min) else None,
                "max": round(float(overall_max), 6) if not np.isnan(overall_max) else None,
                "mean": round(overall_mean, 6) if not np.isnan(overall_mean) else None,
                "std": round(overall_std, 6) if not np.isnan(overall_std) else None,
                "p01": round(p01, 6) if not np.isnan(p01) else None,
                "p05": round(p05, 6) if not np.isnan(p05) else None,
                "p50": round(p50, 6) if not np.isnan(p50) else None,
                "p95": round(p95, 6) if not np.isnan(p95) else None,
                "p99": round(p99, 6) if not np.isnan(p99) else None,
                "is_mask_like": is_mask_like,
                "unique_value_count": unique_values,
            }
            channel_stats.append(stat)

    return channel_stats, mask_like_channels


# =============================================================================
# Time Axis Audit
# =============================================================================

def audit_time_axis(ds: xr.Dataset, chunk_size: int = 100) -> Dict[str, Any]:
    """
    Audit time coordinate across ALL time steps using chunked reads.

    Returns:
        dict with timestamps info, hour distribution, year/month counts,
        common time deltas, missing calendar days
    """
    n_time = ds.sizes["time"]
    time_vals = ds["time"].values

    # Basic timestamp info
    time_min = float(np.nanmin(time_vals))
    time_max = float(np.nanmax(time_vals))
    unique_times = len(np.unique(time_vals))
    duplicates = int(n_time - unique_times)

    # Hour distribution
    hours = (time_vals % 86400) // 3600  # hour-of-day from seconds
    hour_counter = Counter(hours.astype(int).tolist())
    hour_distribution = {f"{h:02d}": count for h, count in sorted(hour_counter.items())}

    # Year/month counts
    days = time_vals // 86400  # days since epoch
    year_month_counter = Counter()
    for d in days:
        try:
            dt = datetime(1970, 1, 1) + timedelta(seconds=float(d) * 86400)
            year_month_counter[f"{dt.year}-{dt.month:02d}"] += 1
        except Exception:
            pass
    year_month_counts = dict(sorted(year_month_counter.items()))

    # Common time deltas
    if len(time_vals) > 1:
        sorted_vals = np.sort(time_vals)
        diffs = np.diff(sorted_vals)
        diff_counter = Counter(diffs.tolist())
        common_deltas = [
            {"delta_seconds": int(k), "count": v}
            for k, v in diff_counter.most_common(5)
        ]
    else:
        common_deltas = []

    # Missing calendar days (gaps > 1.5 * median delta)
    missing_days = []
    if len(diffs) > 0:
        median_delta = float(np.median(diffs))
        threshold = median_delta * 1.5
        gap_indices = np.where(diffs > threshold)[0]
        for idx in gap_indices:
            day_before = datetime(1970, 1, 1) + timedelta(seconds=float(sorted_vals[idx]))
            day_after = datetime(1970, 1, 1) + timedelta(seconds=float(sorted_vals[idx + 1]))
            missing_days.append(f"{day_before.date()} to {day_after.date()}")
    else:
        median_delta = 0

    return {
        "start": float(time_min),
        "end": float(time_max),
        "count": int(n_time),
        "unique": int(unique_times),
        "duplicates": duplicates,
        "time_frequency_estimate_hours": round(float(median_delta) / 3600, 2) if median_delta > 0 else None,
        "hour_distribution": hour_distribution,
        "year_month_counts": year_month_counts,
        "common_time_deltas_seconds": common_deltas,
        "missing_calendar_days": missing_days,
    }


# =============================================================================
# Analysis Labels / DA Cycle Audit
# =============================================================================

def audit_analysis_labels(
    ds: xr.Dataset, chunk_size: int = 100
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Per-timestamp stats for target channels + increment computation.

    Returns:
        cycle_df: list of dicts with per-timestamp stats
        valid_da_cycles_by_year: {"YYYY": count}
        valid_da_cycles_by_month: {"YYYY-MM": count}
    """
    input_channels = list(ds.coords["variable_input"].values)
    target_channels = list(ds.coords["variable_target"].values)

    # Find indices for forecast and analysis channels
    surf_forecast_idx = input_channels.index("sm_surface_forecast")
    root_forecast_idx = input_channels.index("sm_rootzone_forecast")
    surf_analysis_idx = target_channels.index("sm_surface_analysis")
    root_analysis_idx = target_channels.index("sm_rootzone_analysis")

    n_time = ds.sizes["time"]
    cycle_rows = []

    valid_by_year: Dict[str, int] = {}
    valid_by_month: Dict[str, int] = {}

    n_chunks = (n_time + chunk_size - 1) // chunk_size

    for chunk_start in range(0, n_time, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_time)

        # Load input and target slabs for this chunk
        input_slab = ds["input"].values[chunk_start:chunk_end]
        target_slab = ds["target"].values[chunk_start:chunk_end]
        time_slab = ds["time"].values[chunk_start:chunk_end]

        for i in range(chunk_end - chunk_start):
            t = chunk_start + i
            t_val = time_slab[i]

            # Forecast: channels 0 and 1 of input
            surf_fc = input_slab[i, surf_forecast_idx, :, :]
            root_fc = input_slab[i, root_forecast_idx, :, :]

            # Analysis: channels 0 and 1 of target
            surf_an = target_slab[i, surf_analysis_idx, :, :]
            root_an = target_slab[i, root_analysis_idx, :, :]

            # Joint finite ratios
            surf_finite = np.isfinite(surf_fc)
            root_finite = np.isfinite(root_fc)
            surf_an_finite = np.isfinite(surf_an)
            root_an_finite = np.isfinite(root_an)

            forecast_joint = surf_finite & root_finite
            analysis_joint = surf_an_finite & root_an_finite
            all_joint = forecast_joint & analysis_joint

            forecast_joint_ratio = float(forecast_joint.sum()) / surf_fc.size
            analysis_joint_ratio = float(analysis_joint.sum()) / surf_an.size
            all_joint_ratio = float(all_joint.sum()) / surf_fc.size

            is_valid = all_joint_ratio > 0.5

            # Increment stats (analysis - forecast)
            incr_surface = surf_an - surf_fc
            incr_rootzone = root_an - root_fc

            valid_incr_surface = incr_surface[all_joint]
            valid_incr_rootzone = incr_rootzone[all_joint]

            # Time decomposition
            try:
                dt = datetime(1970, 1, 1) + timedelta(seconds=float(t_val))
                year = dt.year
                month = dt.month
                day = dt.day
                hour = dt.hour
            except Exception:
                year = month = day = hour = None

            # Accumulate year/month counts
            if is_valid:
                yk = str(year)
                mk = f"{year}-{month:02d}"
                valid_by_year[yk] = valid_by_year.get(yk, 0) + 1
                valid_by_month[mk] = valid_by_month.get(mk, 0) + 1

            row = {
                "time_index": int(t),
                "time_value": float(t_val),
                "year": int(year) if year else None,
                "month": int(month) if month else None,
                "day": int(day) if day else None,
                "hour": int(hour) if hour else None,
                "forecast_joint_finite_ratio": round(forecast_joint_ratio, 6),
                "analysis_joint_finite_ratio": round(analysis_joint_ratio, 6),
                "forecast_analysis_joint_finite_ratio": round(all_joint_ratio, 6),
                "is_valid_da_cycle_global": is_valid,
                "increment_surface_mean": round(float(valid_incr_surface.mean()), 8) if valid_incr_surface.size > 0 else None,
                "increment_surface_std": round(float(valid_incr_surface.std()), 8) if valid_incr_surface.size > 0 else None,
                "increment_surface_min": round(float(valid_incr_surface.min()), 8) if valid_incr_surface.size > 0 else None,
                "increment_surface_p50": round(float(np.percentile(valid_incr_surface, 50)), 8) if valid_incr_surface.size > 0 else None,
                "increment_surface_p95": round(float(np.percentile(valid_incr_surface, 95)), 8) if valid_incr_surface.size > 0 else None,
                "increment_rootzone_mean": round(float(valid_incr_rootzone.mean()), 8) if valid_incr_rootzone.size > 0 else None,
                "increment_rootzone_std": round(float(valid_incr_rootzone.std()), 8) if valid_incr_rootzone.size > 0 else None,
                "increment_rootzone_min": round(float(valid_incr_rootzone.min()), 8) if valid_incr_rootzone.size > 0 else None,
                "increment_rootzone_p50": round(float(np.percentile(valid_incr_rootzone, 50)), 8) if valid_incr_rootzone.size > 0 else None,
                "increment_rootzone_p95": round(float(np.percentile(valid_incr_rootzone, 95)), 8) if valid_incr_rootzone.size > 0 else None,
            }
            cycle_rows.append(row)

    return cycle_rows, valid_by_year, valid_by_month


# =============================================================================
# Markdown Report
# =============================================================================

def generate_markdown_report(
    audit_dict: Dict[str, Any],
    cycle_rows: List[Dict[str, Any]],
    channel_stats: List[Dict[str, Any]],
    config_warning: Optional[str] = None,
) -> str:
    """Generate structured Markdown audit report."""

    lines = []

    # ---- Title ----
    lines.append("# DA.nc Forensic Audit Report")
    lines.append("")
    lines.append(f"**Generated:** {audit_dict.get('audit_timestamp', 'N/A')}")
    lines.append(f"**File:** `{audit_dict.get('file_path', 'N/A')}`")
    lines.append(f"**Country:** {audit_dict.get('country', 'US')}")
    lines.append("")

    # ---- Executive Summary ----
    lines.append("## Executive Summary")
    lines.append("")
    blocking = audit_dict.get("blocking_issues", [])
    if blocking:
        lines.append(f"⚠️ **Blocking Issues ({len(blocking)}):**")
        for b in blocking:
            lines.append(f"  - {b}")
        lines.append("")
    else:
        lines.append("✅ **No blocking issues detected.**")
        lines.append("")

    key_findings = audit_dict.get("key_findings", [])
    if key_findings:
        lines.append("**Key Findings:**")
        for f in key_findings:
            lines.append(f"- {f}")
        lines.append("")

    # ---- Data Structure ----
    lines.append("## Data Structure")
    dims = audit_dict.get("dims", {})
    lines.append(f"- Dimensions: {dims}")
    lines.append("")

    shapes = audit_dict.get("variable_shapes", {})
    dtypes = audit_dict.get("variable_dtypes", {})
    lines.append("### Variables")
    for var in sorted(shapes.keys()):
        lines.append(f"- `{var}`: shape={shapes[var]}, dtype={dtypes.get(var, 'N/A')}")
    lines.append("")

    coords = audit_dict.get("coords", [])
    lines.append(f"### Coordinates: {coords}")
    lines.append("")

    # ---- Channel Summary Table ----
    lines.append("## Channel Summary")
    lines.append("")
    lines.append(
        "| idx | array | variable_name | finite_ratio | min | max | mean | std | "
        "p01 | p05 | p50 | p95 | p99 | is_mask_like | unique_count |"
    )
    lines.append(
        "|---|------|---------------|-------------|-----|-----|------|------|"
        "-----|-----|-----|-----|-----|-------------|------------|"
    )
    for ch in channel_stats:
        lines.append(
            f"| {ch['channel_index']} | {ch['array']} | {ch['variable_name']} | "
            f"{ch['finite_ratio']:.4f} | {ch['min']} | {ch['max']} | "
            f"{ch['mean']} | {ch['std']} | "
            f"{ch['p01']} | {ch['p05']} | {ch['p50']} | {ch['p95']} | {ch['p99']} | "
            f"{ch['is_mask_like']} | {ch['unique_value_count']} |"
        )
    lines.append("")

    # ---- Input/Target Channel Count ----
    lines.append("### Input/Target Channel Mapping")
    input_channels = audit_dict.get("input_channels", [])
    target_channels = audit_dict.get("target_channels", [])
    lines.append(f"- Input channels ({len(input_channels)}): {input_channels}")
    lines.append(f"- Target channels ({len(target_channels)}): {target_channels}")
    lines.append("")

    mask_like = audit_dict.get("mask_like_channels", [])
    if mask_like:
        lines.append(f"⚠️ **Mask-like channels (finite_ratio < 0.05 or unique ≤ 2):** {mask_like}")
        lines.append("")
    else:
        lines.append("✅ No mask-like channels detected.")
        lines.append("")

    # ---- Time Axis ----
    lines.append("## Time Axis")
    ta = audit_dict.get("time_axis", {})
    lines.append(f"- Time range: {ta.get('start')} → {ta.get('end')}")
    lines.append(f"- Count: {ta.get('count')} | Unique: {ta.get('unique')} | Duplicates: {ta.get('duplicates')}")
    lines.append(f"- Estimated frequency: {ta.get('time_frequency_estimate_hours')} hours/step")
    lines.append("")

    # Hour distribution
    hour_dist = ta.get("hour_distribution", {})
    if hour_dist:
        lines.append("### Hour of Day Distribution")
        lines.append("```")
        lines.append("Hour | Count")
        lines.append("-----|------")
        for h in sorted(hour_dist.keys()):
            lines.append(f"{h} | {hour_dist[h]}")
        lines.append("```")
        lines.append("")

    # Year/month counts
    ym_counts = ta.get("year_month_counts", {})
    if ym_counts:
        lines.append("### Year-Month Counts")
        lines.append("```")
        lines.append("Year-Month | Count")
        lines.append("-----------|------")
        for ym in sorted(ym_counts.keys()):
            lines.append(f"{ym} | {ym_counts[ym]}")
        lines.append("```")
        lines.append("")

    # Common time deltas
    deltas = ta.get("common_time_deltas_seconds", [])
    if deltas:
        lines.append("### Common Time Deltas")
        for d in deltas[:5]:
            lines.append(f"- {d['delta_seconds']}s ({d['delta_seconds']/3600:.2f}h): {d['count']} occurrences")
        lines.append("")

    # Missing calendar days
    missing = ta.get("missing_calendar_days", [])
    if missing:
        lines.append(f"⚠️ **Missing calendar days ({len(missing)}):**")
        for m in missing[:10]:
            lines.append(f"  - {m}")
        if len(missing) > 10:
            lines.append(f"  ... and {len(missing) - 10} more")
        lines.append("")
    else:
        lines.append("✅ No missing calendar days detected.")
        lines.append("")

    # ---- Valid DA Cycles ----
    lines.append("## Valid DA Cycles")
    lines.append(f"- **Definition:** forecast_analysis_joint_finite_ratio > 0.5 (global, per-timestamp)")
    lines.append(f"- **Total valid cycles:** {audit_dict.get('valid_da_cycles_total', 'N/A')}")
    lines.append("")

    valid_by_year = audit_dict.get("valid_da_cycles_by_year", {})
    if valid_by_year:
        lines.append("### Valid Cycles by Year")
        lines.append("```")
        lines.append("Year | Valid Cycles")
        lines.append("-----|-------------")
        for y in sorted(valid_by_year.keys()):
            lines.append(f"{y} | {valid_by_year[y]}")
        lines.append("```")
        lines.append("")

    valid_by_month = audit_dict.get("valid_da_cycles_by_month", {})
    if valid_by_month:
        lines.append("### Valid Cycles by Month")
        lines.append("```")
        lines.append("Month | Valid Cycles")
        lines.append("------|-------------")
        for m in sorted(valid_by_month.keys()):
            lines.append(f"{m} | {valid_by_month[m]}")
        lines.append("```")
        lines.append("")

    # ---- Config Warning ----
    if config_warning:
        lines.append(f"⚠️ **Config Warning:** {config_warning}")
        lines.append("")

    # ---- Increment Analysis ----
    lines.append("## Increment Analysis (Δ = analysis - forecast)")
    lines.append("")
    if cycle_rows:
        # Aggregate increment stats
        surf_means = [r["increment_surface_mean"] for r in cycle_rows if r["increment_surface_mean"] is not None]
        rz_means = [r["increment_rootzone_mean"] for r in cycle_rows if r["increment_rootzone_mean"] is not None]
        if surf_means:
            lines.append(f"- Surface increment mean across cycles: {np.mean(surf_means):.6f}")
        if rz_means:
            lines.append(f"- Rootzone increment mean across cycles: {np.mean(rz_means):.6f}")
    lines.append("")

    # ---- Unresolved Ambiguities ----
    unresolved = audit_dict.get("unresolved_ambiguities", [])
    if unresolved:
        lines.append("## Unresolved Ambiguities")
        for u in unresolved:
            lines.append(f"- {u}")
        lines.append("")

    # ---- Next Actions ----
    lines.append("## Next Actions")
    next_actions = audit_dict.get("next_actions", [])
    if next_actions:
        for a in next_actions:
            lines.append(f"- {a}")
    else:
        lines.append("- No immediate actions required.")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main Audit Function
# =============================================================================

def audit_da_nc_structure(
    da_nc_path: str,
    config_vars_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform comprehensive forensic audit of DA.nc.

    Args:
        da_nc_path: Path to DA.nc file
        config_vars_path: Optional path to Config_Vars_DA.json

    Returns:
        audit_dict with all findings
    """
    print(f"Opening: {da_nc_path}")
    ds = xr.open_dataset(da_nc_path, engine="netcdf4", decode_times=False)

    audit = {
        "audit_timestamp": datetime.now().isoformat(),
        "file_path": da_nc_path,
        "country": "US",
    }

    # ---- Basic structure ----
    audit["dims"] = dict(ds.sizes)
    audit["coords"] = list(ds.coords)
    audit["data_vars"] = list(ds.data_vars)
    audit["variable_shapes"] = {v: list(ds[v].shape) for v in ds.data_vars}
    audit["variable_dtypes"] = {v: str(ds[v].dtype) for v in ds.data_vars}

    # ---- Config comparison ----
    config_warning = None
    config_expected_input = None
    config_expected_target = None

    if config_vars_path:
        try:
            with open(config_vars_path) as f:
                config = json.load(f)
            config_expected_input = [v["name"] for v in config.get("input_variables", [])]
            config_expected_target = [v["name"] for v in config.get("target_variables", [])]
        except Exception as e:
            config_warning = f"Could not load config: {e}"
    else:
        config_warning = "Config file not provided; treated as hypothesis only"

    # ---- Channel detection ----
    input_coord = ds.coords["variable_input"]
    target_coord = ds.coords["variable_target"]
    input_channels = list(input_coord.values)
    target_channels = list(target_coord.values)

    audit["input_channels"] = input_channels
    audit["target_channels"] = target_channels
    audit["input_channel_count"] = len(input_channels)
    audit["target_channel_count"] = len(target_channels)

    # Config vs actual comparison
    if config_expected_input:
        if config_expected_input != input_channels:
            config_warning = (
                f"Config input order mismatch. "
                f"Config: {config_expected_input}, Actual: {input_channels}"
            )
        if config_expected_target != target_channels:
            if config_warning:
                config_warning += f"; Target mismatch: Config={config_expected_target}, Actual={target_channels}"
            else:
                config_warning = (
                    f"Config target order mismatch. "
                    f"Config: {config_expected_target}, Actual: {target_channels}"
                )

    audit["config_warning"] = config_warning

    # ---- Time axis ----
    print("Auditing time axis...")
    time_axis = audit_time_axis(ds)
    audit["time_axis"] = time_axis

    # ---- Channel statistics ----
    print("Computing channel statistics (all time steps)...")
    channel_stats, mask_like_channels = compute_channel_statistics(ds)
    audit["channel_statistics"] = channel_stats
    audit["mask_like_channels"] = mask_like_channels

    # ---- Analysis labels / DA cycle audit ----
    print("Auditing DA cycles (all time steps)...")
    cycle_rows, valid_by_year, valid_by_month = audit_analysis_labels(ds)

    total_valid = sum(1 for r in cycle_rows if r["is_valid_da_cycle_global"])
    audit["valid_da_cycles_total"] = total_valid
    audit["valid_da_cycles_by_year"] = dict(sorted(valid_by_year.items()))
    audit["valid_da_cycles_by_month"] = dict(sorted(valid_by_month.items()))

    audit["cycle_rows"] = cycle_rows

    # ---- Blocking issues ----
    blocking_issues = []
    key_findings = []
    unresolved_ambiguities = []
    next_actions = []

    # Check for blocking issues
    if not input_channels:
        blocking_issues.append("No input channels found")
    if not target_channels:
        blocking_issues.append("No target channels found")

    # Check for mask-like channels that are not the expected base_valid_mask
    if mask_like_channels:
        for ch in mask_like_channels:
            if ch not in ["base_valid_mask"]:
                unresolved_ambiguities.append(
                    f"Channel '{ch}' flagged as mask-like but not recognized as base_valid_mask"
                )

    # Check channel 11 status
    if len(input_channels) > 11:
        ch11 = input_channels[11]
        ch11_stats = next((s for s in channel_stats if s["array"] == "input" and s["channel_index"] == 11), None)
        if ch11_stats:
            if ch11_stats["finite_ratio"] < 0.05:
                key_findings.append(
                    f"Channel 11 ('{ch11}') has very low finite_ratio={ch11_stats['finite_ratio']:.4f} — confirmed as mask-like"
                )
            elif ch11_stats["unique_value_count"] <= 2:
                key_findings.append(
                    f"Channel 11 ('{ch11}') has only {ch11_stats['unique_value_count']} unique values — confirmed as mask-like"
                )

    # Valid cycles summary
    total_cycles = len(cycle_rows)
    if total_cycles > 0:
        valid_pct = total_valid / total_cycles * 100
        key_findings.append(f"Valid DA cycles: {total_valid}/{total_cycles} ({valid_pct:.1f}%)")
        key_findings.append(f"Time range: {time_axis['start']:.0f}s to {time_axis['end']:.0f}s")
        key_findings.append(f"Time frequency estimate: {time_axis['time_frequency_estimate_hours']}h/step")

    audit["blocking_issues"] = blocking_issues
    audit["key_findings"] = key_findings
    audit["unresolved_ambiguities"] = unresolved_ambiguities
    audit["next_actions"] = next_actions

    ds.close()

    return audit