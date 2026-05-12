#!/usr/bin/env python
"""
CLI wrapper for NetCDF audit.

Usage:
    python scripts/audit_netcdf.py \
        --data /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc \
        --country US \
        --out-json artifacts/audits/netcdf_audit_US.json \
        --out-md reports/audits/netcdf_audit_US.md
"""

import argparse
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hydroda.data.netcdf_audit import audit_netcdf


def dict_to_markdown(data: dict, level: int = 1) -> str:
    """Convert audit dict to human-readable markdown."""
    lines = []
    title = f"NetCDF Audit Report: {data.get('country', 'Unknown')}"
    lines.append(f"{'#' * level} {title}")
    lines.append("")
    lines.append(f"**Timestamp:** {data.get('audit_timestamp', 'N/A')}")
    lines.append(f"**File:** `{data.get('file_path', 'N/A')}`")
    lines.append(f"**Estimated Memory:** {data.get('estimated_memory_gb', 'N/A'):.2f} GB")
    lines.append("")

    # Dimensions
    lines.append("## Dimensions")
    dims = data.get("dims", {})
    for k, v in dims.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    # Coordinates
    lines.append("## Coordinates")
    coords = data.get("coords", [])
    for c in coords:
        lines.append(f"- `{c}`")
    lines.append("")

    # Coordinate availability
    coord_avail = data.get("coordinate_availability", {})
    lines.append("## Coordinate Availability")
    for k, v in coord_avail.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    # Data Variables
    lines.append("## Data Variables")
    data_vars = data.get("data_vars", [])
    lines.append(f"Total: {len(data_vars)}")
    lines.append("")

    # Variable shapes and dtypes
    lines.append("### Variable Shapes and Dtypes")
    shapes = data.get("variable_shapes", {})
    dtypes = data.get("variable_dtypes", {})
    for var in sorted(data_vars):
        shape = shapes.get(var, "N/A")
        dtype = dtypes.get(var, "N/A")
        lines.append(f"- `{var}`: shape={shape}, dtype={dtype}")
    lines.append("")

    # Time range
    lines.append("## Time Range")
    time_range = data.get("time_range")
    if time_range:
        lines.append(f"- Min: {time_range.get('min', 'N/A')}")
        lines.append(f"- Max: {time_range.get('max', 'N/A')}")
        lines.append(f"- Count: {time_range.get('count', 'N/A')}")
    else:
        lines.append(f"- {time_range}")
    lines.append(f"- Frequency: {data.get('time_frequency', 'N/A')}")
    lines.append("")

    # Channel mapping
    lines.append("## Channel Mapping")
    channel = data.get("channel_names_or_order", {})
    blocking = channel.get("blocking_issue")
    structure = channel.get("structure", "unknown")

    lines.append(f"**Structure:** {structure}")
    if blocking:
        lines.append(f"**BLOCKING ISSUE:** {blocking}")
    lines.append("")

    # List all input channels
    inp_channels = channel.get("input_channels", [])
    lines.append(f"### Input Channels ({len(inp_channels)} total)")
    for i, ch in enumerate(inp_channels):
        lines.append(f"- [{i}] `{ch}`")
    lines.append("")

    # List all target channels
    tgt_channels = channel.get("target_channels", [])
    lines.append(f"### Target Channels ({len(tgt_channels)} total)")
    for i, ch in enumerate(tgt_channels):
        lines.append(f"- [{i}] `{ch}`")
    lines.append("")

    # Contract mapping
    lines.append("### Contract Variable Mapping")
    inp_map = channel.get("input_mapping", {})
    if inp_map:
        for expected, info in inp_map.items():
            lines.append(f"- `{expected}` → channel index {info.get('channel_index')}")
    else:
        lines.append("- No contract inputs mapped")

    tgt_map = channel.get("target_mapping", {})
    if tgt_map:
        for expected, info in tgt_map.items():
            lines.append(f"- `{expected}` → channel index {info.get('channel_index')}")
    else:
        lines.append("- No contract targets mapped")

    missing_inp = channel.get("missing_inputs", [])
    if missing_inp:
        lines.append(f"**Missing Inputs:** {missing_inp}")
    missing_tgt = channel.get("missing_targets", [])
    if missing_tgt:
        lines.append(f"**Missing Targets:** {missing_tgt}")
    lines.append("")

    # NaN/Inf counts
    lines.append("## NaN/Inf Counts (sample)")
    nan_inf = data.get("nan_inf_counts", {})
    for var in sorted(nan_inf.keys())[:10]:
        info = nan_inf[var]
        lines.append(
            f"- `{var}`: nan={info['nan_count']} ({info['nan_ratio']:.2%}), "
            f"inf={info['inf_count']} ({info['inf_ratio']:.2%})"
        )
    if len(nan_inf) > 10:
        lines.append(f"- ... and {len(nan_inf) - 10} more variables")
    lines.append("")

    # Mask info
    lines.append("## Mask Variables")
    masks = data.get("mask_unique_values", {})
    distinct = data.get("mask_keys_distinct", False)
    lines.append(f"Mask keys distinct (obs/region/loss): {distinct}")
    for name, info in masks.items():
        lines.append(
            f"- `{name}`: unique={info['unique_values']}, "
            f"finite_ratio={info['finite_ratio']:.4f}, "
            f"shape={info['shape']}"
        )
    lines.append("")

    # Finite overlap
    lines.append("## Finite Overlap (first 10 variables)")
    finite = data.get("finite_overlap_forecast_analysis", {})
    for var in sorted(finite.keys())[:10]:
        info = finite[var]
        lines.append(
            f"- `{var}`: finite_ratio={info['finite_ratio']:.4f}"
        )
    lines.append("")

    # Increment reconstruction
    lines.append("## Increment Reconstruction")
    inc = data.get("increment_reconstruction", {})
    if "error" in inc:
        lines.append(f"ERROR: {inc['error']}")
    else:
        lines.append(f"**Structure:** {inc.get('structure_note', 'N/A')}")
        lines.append("")

        detected = inc.get("detected_variables", {})
        lines.append("**Channel Indices:**")
        lines.append(f"- surface_forecast_index: {detected.get('surface_forecast_index')}")
        lines.append(f"- surface_analysis_index: {detected.get('surface_analysis_index')}")
        lines.append(f"- rootzone_forecast_index: {detected.get('rootzone_forecast_index')}")
        lines.append(f"- rootzone_analysis_index: {detected.get('rootzone_analysis_index')}")
        lines.append("")

        surf_pass = inc.get("surface_reconstruction_passed")
        root_pass = inc.get("rootzone_reconstruction_passed")
        if surf_pass is not None:
            lines.append(
                f"- Surface: passed={surf_pass}, "
                f"max_error={inc.get('surface_max_error')}, "
                f"mean_error={inc.get('surface_mean_error')}"
            )
        if root_pass is not None:
            lines.append(
                f"- Rootzone: passed={root_pass}, "
                f"max_error={inc.get('rootzone_max_error')}, "
                f"mean_error={inc.get('rootzone_mean_error')}"
            )
        blocking_inc = inc.get("blocking_issue")
        if blocking_inc:
            lines.append(f"**BLOCKING:** {blocking_inc}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit DA.nc NetCDF file")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to DA.nc file",
    )
    parser.add_argument(
        "--country",
        default="US",
        help="Country code (US, CN, AU)",
    )
    parser.add_argument(
        "--out-json",
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--out-md",
        required=True,
        help="Output Markdown path",
    )
    args = parser.parse_args()

    print(f"Auditing: {args.data}")
    print(f"Country: {args.country}")

    audit = audit_netcdf(args.data, args.country)

    # Write JSON
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"JSON written to: {args.out_json}")

    # Write Markdown
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    md_content = dict_to_markdown(audit)
    with open(args.out_md, "w") as f:
        f.write(md_content)
    print(f"Markdown written to: {args.out_md}")

    # Print summary
    blocking_issues = []
    if audit.get("channel_names_or_order", {}).get("blocking_issue"):
        blocking_issues.append("channel_mapping")
    if audit.get("increment_reconstruction", {}).get("blocking_issue"):
        blocking_issues.append("increment_reconstruction")
    coord_avail = audit.get("coordinate_availability", {})
    if not coord_avail.get("lat") or not coord_avail.get("lon"):
        blocking_issues.append("no_lat_lon")

    if blocking_issues:
        print("\n⚠️  BLOCKING ISSUES:")
        for b in blocking_issues:
            print(f"  - {b}")
    else:
        print("\n✅ No blocking issues detected")


if __name__ == "__main__":
    main()
