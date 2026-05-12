"""Build US Scientific Region Masks from lat/lon grid + bbox definitions.

Usage:
    python scripts/data/build_us_region_masks.py \
        --latlon-nc artifacts/geolocation/US_latlon.nc \
        --regions-spec specs/regions_v2.yaml \
        --out-masks artifacts/regions/US_region_masks.nc \
        --out-stats artifacts/regions/US_region_stats.json \
        --out-md reports/regions/US_region_mask_summary.md

No-leakage: Region masks are built ONLY from:
    - Fixed lat/lon bbox from regions_v2.yaml
    - Lat/lon grids from US_latlon.nc
NOT from: DA analysis increments, model errors, target query labels, training results.
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hydroda.regions.masks import (
    build_region_masks_from_bbox,
    check_no_overlap,
    compute_region_stats,
    generate_markdown_report,
    load_latlon_grid,
    save_region_masks_nc,
    save_stats_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build US scientific region masks from lat/lon grid + bbox definitions"
    )
    parser.add_argument(
        "--latlon-nc",
        required=True,
        help="Path to US_latlon.nc with latitude[H,W] and longitude[H,W]",
    )
    parser.add_argument(
        "--regions-spec",
        required=True,
        help="Path to regions_v2.yaml with region bbox definitions",
    )
    parser.add_argument(
        "--out-masks",
        required=True,
        help="Output path for US_region_masks.nc",
    )
    parser.add_argument(
        "--out-stats",
        required=True,
        help="Output path for US_region_stats.json",
    )
    parser.add_argument(
        "--out-md",
        required=True,
        help="Output path for US_region_mask_summary.md",
    )
    args = parser.parse_args()

    # Load geolocation grid
    print("Loading lat/lon grid...")
    lat, lon = load_latlon_grid(args.latlon_nc)
    print(f"  Grid shape: {lat.shape}")
    print(f"  Lat range: [{lat.min():.2f}, {lat.max():.2f}]")
    print(f"  Lon range: [{lon.min():.2f}, {lon.max():.2f}]")

    # Load regions spec
    print(f"\nLoading regions spec: {args.regions_spec}")
    with open(args.regions_spec) as f:
        regions_spec = yaml.safe_load(f)

    # Build masks
    print("\nBuilding region masks from bounding boxes...")
    mask_int, mask_onehot = build_region_masks_from_bbox(lat, lon, regions_spec)

    # Overlap check
    print("\nChecking for overlaps...")
    no_overlap = check_no_overlap(mask_onehot)
    if no_overlap:
        print("  PASS: No overlaps detected")
    else:
        print("  FAIL: Overlaps detected!")
        sys.exit(1)

    # Compute stats
    print("\nComputing region statistics...")
    stats = compute_region_stats(mask_int, lat, lon, regions_spec)

    # Create output directory
    Path(args.out_masks).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)

    # Save outputs
    print("\nSaving outputs...")
    save_region_masks_nc(
        mask_int,
        mask_onehot,
        lat,
        lon,
        args.out_masks,
        args.regions_spec,
        args.latlon_nc,
    )
    save_stats_json(stats, args.out_stats)
    generate_markdown_report(
        stats,
        args.out_md,
        args.regions_spec,
        args.latlon_nc,
        no_overlap,
    )

    print("\nDone! Phase 2A US region masks created.")
    print(f"  Masks: {args.out_masks}")
    print(f"  Stats: {args.out_stats}")
    print(f"  Report: {args.out_md}")


if __name__ == "__main__":
    main()