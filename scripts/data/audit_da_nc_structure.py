#!/usr/bin/env python
"""
Phase 0A: DA.nc Forensic Audit Script

Comprehensive forensic audit of DA.nc using chunked reads.
Generates JSON artifact, Markdown report, cycle CSV, and channel CSV.

Usage:
    python scripts/audit_da_nc_structure.py \\
        --da-nc /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc \\
        --config-vars external/collaborator_code/Config_Vars_DA.json \\
        --out-json artifacts/audits/netcdf_audit_US.json \\
        --out-md reports/audits/netcdf_audit_US.md \\
        --out-cycle-csv artifacts/audits/da_cycle_availability_US.csv \\
        --out-channel-csv artifacts/audits/da_channel_summary_US.csv
"""

import argparse
import csv
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hydroda.data.da_nc_structure_audit import (
    audit_da_nc_structure,
    generate_markdown_report,
    NumpySafeEncoder,
)


def write_cycle_csv(rows: list, path: str) -> None:
    """Write per-timestamp cycle availability CSV."""
    if not rows:
        return

    fieldnames = [
        "time_index", "time_value", "year", "month", "day", "hour",
        "forecast_joint_finite_ratio", "analysis_joint_finite_ratio",
        "forecast_analysis_joint_finite_ratio", "is_valid_da_cycle_global",
        "increment_surface_mean", "increment_surface_std",
        "increment_surface_min", "increment_surface_p50", "increment_surface_p95",
        "increment_rootzone_mean", "increment_rootzone_std",
        "increment_rootzone_min", "increment_rootzone_p50", "increment_rootzone_p95",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Convert numpy types for CSV writer
            csv_row = {}
            for k, v in row.items():
                if v is None:
                    csv_row[k] = ""
                elif isinstance(v, (float, int, bool)):
                    csv_row[k] = v
                else:
                    csv_row[k] = str(v)
            writer.writerow(csv_row)


def write_channel_csv(stats: list, path: str) -> None:
    """Write per-channel summary CSV."""
    if not stats:
        return

    fieldnames = [
        "channel_index", "array", "variable_name", "finite_ratio",
        "min", "max", "mean", "std",
        "p01", "p05", "p50", "p95", "p99",
        "is_mask_like", "unique_value_count",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            csv_row = {k: ("NA" if v is None else v) for k, v in row.items()}
            writer.writerow(csv_row)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0A: Forensic audit of DA.nc multi-channel array"
    )
    parser.add_argument(
        "--da-nc",
        required=True,
        help="Path to DA.nc file (required)",
    )
    parser.add_argument(
        "--config-vars",
        default=None,
        help="Path to Config_Vars_DA.json (auto-search if not given)",
    )
    parser.add_argument(
        "--out-json",
        required=True,
        help="Path for JSON artifact",
    )
    parser.add_argument(
        "--out-md",
        required=True,
        help="Path for Markdown report",
    )
    parser.add_argument(
        "--out-cycle-csv",
        required=True,
        help="Path for per-timestamp cycle CSV",
    )
    parser.add_argument(
        "--out-channel-csv",
        required=True,
        help="Path for per-channel summary CSV",
    )

    args = parser.parse_args()

    # Auto-search for config if not provided
    config_path = args.config_vars
    if not config_path:
        search_paths = [
            "external/collaborator_code/Config_Vars_DA.json",
            os.path.join(os.path.dirname(__file__), "..", "external/collaborator_code/Config_Vars_DA.json"),
        ]
        for p in search_paths:
            if os.path.exists(p):
                config_path = p
                break

    print(f"DA.nc: {args.da_nc}")
    print(f"Config: {config_path or 'none'}")
    print("")

    # Run audit
    audit = audit_da_nc_structure(args.da_nc, config_path)

    # ---- Write JSON artifact ----
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json_dict = {k: v for k, v in audit.items() if k != "cycle_rows"}
    # cycle_rows is written separately as CSV; trim from JSON for size
    with open(args.out_json, "w") as f:
        json.dump(json_dict, f, indent=2, cls=NumpySafeEncoder)
    print(f"JSON: {args.out_json}")

    # ---- Write cycle CSV ----
    write_cycle_csv(audit.get("cycle_rows", []), args.out_cycle_csv)
    print(f"Cycle CSV: {args.out_cycle_csv}")

    # ---- Write channel CSV ----
    write_channel_csv(audit.get("channel_statistics", []), args.out_channel_csv)
    print(f"Channel CSV: {args.out_channel_csv}")

    # ---- Write Markdown report ----
    config_warning = audit.get("config_warning")
    md_content = generate_markdown_report(
        audit,
        audit.get("cycle_rows", []),
        audit.get("channel_statistics", []),
        config_warning=config_warning,
    )
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(md_content)
    print(f"Markdown: {args.out_md}")

    # ---- Summary ----
    print("")
    blocking = audit.get("blocking_issues", [])
    if blocking:
        print(f"⚠️  BLOCKING ISSUES ({len(blocking)}):")
        for b in blocking:
            print(f"  - {b}")
    else:
        print("✅ No blocking issues")

    key = audit.get("key_findings", [])
    if key:
        print("")
        print("Key findings:")
        for f in key:
            print(f"  - {f}")


if __name__ == "__main__":
    main()