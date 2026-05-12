#!/usr/bin/env python3
"""
Phase 1C: Label Availability Audit — CLI Entry Point

Usage:
    python scripts/audit_label_availability.py [--da-nc PATH] [--output-dir DIR]
        [--chunk-size N] [--no-plot]

Default paths:
    DA.nc: /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc
    Output dir: artifacts/audits/
    Chunk size: 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydroda.data.label_audit import (
    compute_timestamp_stats,
    accumulate_labeled_cycles,
    compute_year_month_stats,
    verify_frozen_splits,
    generate_json_report,
    generate_markdown_report,
    plot_labeled_cycle_counts_by_year_v2,
    _NumpySafeEncoder,
)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1C: Label Availability Audit"
    )
    parser.add_argument(
        "--da-nc",
        type=str,
        default="/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc",
        help="Path to DA.nc file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/audits",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--splits-json",
        type=str,
        default="artifacts/splits/US_loro_kdate_splits.json",
        help="Path to frozen splits JSON",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of time steps per chunk (default: 100)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figure generation",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Starting Label Availability Audit")
    print(f"[INFO] DA.nc: {args.da_nc}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Chunk size: {args.chunk_size}")

    # Load frozen splits
    print("[INFO] Loading frozen splits...")
    with open(args.splits_json, "r") as f:
        splits_data = json.load(f)
    print(f"[INFO] Loaded {len(splits_data['splits'])} splits")

    # Open DA.nc with chunking
    print("[INFO] Opening DA.nc with chunked reading...")
    t0 = time.time()
    ds = xr.open_dataset(args.da_nc, engine="netcdf4", decode_times=False, chunks={"time": args.chunk_size})

    n_times = ds.sizes["time"]
    n_chunks = (n_times + args.chunk_size - 1) // args.chunk_size
    print(f"[INFO] Total timestamps: {n_times}, chunks: {n_chunks}")

    # Get time coordinates
    time_coords = ds["time"].values
    input_coords = list(ds.coords["variable_input"].values)
    target_coords = list(ds.coords["variable_target"].values)
    print(f"[INFO] Input channels: {input_coords}")
    print(f"[INFO] Target channels: {target_coords}")

    # Process all chunks
    all_stats = []
    chunk_times = []

    print("[INFO] Processing chunks...")
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * args.chunk_size
        end_idx = min(start_idx + args.chunk_size, n_times)
        chunk_size_actual = end_idx - start_idx

        # Read chunk
        input_chunk = ds["input"].isel(time=slice(start_idx, end_idx)).values
        target_chunk = ds["target"].isel(time=slice(start_idx, end_idx)).values

        time_indices = list(range(start_idx, end_idx))

        # Compute stats
        chunk_stats = compute_timestamp_stats(input_chunk, target_chunk, time_indices)
        all_stats.extend(chunk_stats)

        if (chunk_idx + 1) % 20 == 0 or chunk_idx == n_chunks - 1:
            elapsed = time.time() - t0
            print(f"[INFO] Chunk {chunk_idx + 1}/{n_chunks} done ({elapsed:.1f}s elapsed)")

    ds.close()
    total_time = time.time() - t0
    print(f"[INFO] Total processing time: {total_time:.1f}s")
    print(f"[INFO] Total timestamps processed: {len(all_stats)}")

    # Accumulate labeled cycles
    print("[INFO] Accumulating labeled cycles...")
    labeled_cycles = accumulate_labeled_cycles(all_stats)
    print(f"[INFO] Labeled cycles: {len(labeled_cycles)}")

    # Compute year-month stats
    print("[INFO] Computing year-month aggregation...")
    ym_stats = compute_year_month_stats(all_stats, time_coords)
    print(f"[INFO] Year-month buckets: {len(ym_stats)}")

    # Verify all frozen splits
    print("[INFO] Verifying frozen splits...")
    split_results = verify_frozen_splits(splits_data, all_stats, labeled_cycles)
    print(f"[INFO] Verified {len(split_results)} splits")

    # Generate JSON report
    print("[INFO] Generating JSON report...")
    json_report = generate_json_report(
        all_stats, labeled_cycles, ym_stats, split_results, time_coords
    )
    json_path = output_dir / "label_availability_US.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2, cls=_NumpySafeEncoder)
    print(f"[INFO] Saved JSON report to {json_path}")

    # Generate Markdown report
    print("[INFO] Generating Markdown report...")
    md_report = generate_markdown_report(
        all_stats, labeled_cycles, ym_stats, split_results, time_coords
    )
    md_path = Path("reports/audits/label_availability_US.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"[INFO] Saved Markdown report to {md_path}")

    # Generate figure
    if not args.no_plot:
        print("[INFO] Generating figure...")
        fig_path = Path("figures/audits/labeled_cycle_counts_by_year.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plot_labeled_cycle_counts_by_year_v2(
            all_stats, time_coords, labeled_cycles, str(fig_path)
        )
    else:
        print("[INFO] Skipping figure generation (--no-plot)")

    # Summary
    total = len(all_stats)
    labeled = len(labeled_cycles)
    forecast_only = sum(1 for s in all_stats if s.is_forecast_only)
    no_assim = sum(1 for s in all_stats if s.is_no_assimilation)
    near_zero = sum(1 for s in all_stats if s.is_near_zero_increment)
    pass_gate = sum(1 for r in split_results if r.pass_phase3a_gate)

    print("\n" + "=" * 60)
    print("LABEL AVAILABILITY AUDIT — SUMMARY")
    print("=" * 60)
    print(f"Total timestamps:       {total:,}")
    print(f"Labeled DA cycles:     {labeled:,} ({100*labeled/total:.1f}%)")
    print(f"Forecast-only cycles:  {forecast_only:,} ({100*forecast_only/total:.1f}%)")
    print(f"No-assimilation cycles: {no_assim:,} ({100*no_assim/total:.1f}%)")
    print(f"Near-zero increment:   {near_zero:,} ({100*near_zero/total:.1f}%)")
    print(f"Phase 3A gate passes:   {pass_gate}/{len(split_results)} ({100*pass_gate/len(split_results):.1f}%)")
    print("=" * 60)

    if pass_gate == len(split_results):
        print("\n✓ Phase 3A (forecast-only baseline) is CLEARED")
    else:
        print(f"\n✗ Phase 3A has ISSUES — {len(split_results) - pass_gate} splits fail gate")
        print("  Human review required before proceeding.")


if __name__ == "__main__":
    main()
