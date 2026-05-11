"""US Scientific Region Mask Building Utilities.

No-leakage declaration:
    Region masks are built ONLY from fixed lat/lon bounding boxes
    from regions_v2.yaml and lat/lon grids from US_latlon.nc.
    NOT from: DA analysis increments, model errors, target query labels,
    or any training results.
"""

from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import xarray as xr


def load_latlon_grid(latlon_nc_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load 2D latitude and longitude grids from NetCDF.

    Args:
        latlon_nc_path: Path to US_latlon.nc with latitude[H,W] and longitude[H,W]

    Returns:
        Tuple of (latitude_2d, longitude_2d) arrays, both shape (H, W)
    """
    ds = xr.open_dataset(latlon_nc_path)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    ds.close()
    return lat, lon


def build_region_masks_from_bbox(
    lat: np.ndarray,
    lon: np.ndarray,
    regions_spec: Dict,
    country: str = "US",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build integer and one-hot region masks from bounding box definitions.

    A grid cell belongs to a region if:
        lat_min <= lat <= lat_max AND lon_min <= lon <= lon_max

    Args:
        lat: 2D latitude array (H, W)
        lon: 2D longitude array (H, W)
        regions_spec: Parsed regions_v2.yaml dict
        country: Country key (default "US")

    Returns:
        Tuple of (mask_integer, mask_onehot) where:
        - mask_integer: shape (H, W), values 0-6 (0=outside all regions)
        - mask_onehot: shape (6, H, W), one-hot encoding per region
    """
    H, W = lat.shape
    mask_int = np.zeros((H, W), dtype=np.int8)

    country_regions = regions_spec["countries"][country]["regions"]

    # Build one-hot: shape (num_regions, H, W)
    num_regions = len(country_regions)
    mask_onehot = np.zeros((num_regions, H, W), dtype=np.int8)

    for idx, region in enumerate(country_regions):
        region_id = region["id"]
        bbox = region["bbox"]
        lat_min = bbox["lat_min"]
        lat_max = bbox["lat_max"]
        lon_min = bbox["lon_min"]
        lon_max = bbox["lon_max"]

        # Axis-aligned bounding box assignment (half-open interval per spec)
        in_region = (
            (lat >= lat_min)
            & (lat < lat_max)
            & (lon >= lon_min)
            & (lon < lon_max)
        )

        mask_int = np.where(in_region, idx + 1, mask_int)
        mask_onehot[idx] = in_region.astype(np.int8)

        print(f"  {region_id}: {in_region.sum():,} cells "
              f"(lat [{lat_min},{lat_max}], lon [{lon_min},{lon_max}])")

    return mask_int, mask_onehot


def compute_region_stats(
    mask_int: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    regions_spec: Dict,
    country: str = "US",
) -> Dict:
    """Compute per-region statistics.

    Args:
        mask_int: Integer mask (H, W), values 0-6
        lat: 2D latitude array (H, W)
        lon: 2D longitude array (H, W)
        regions_spec: Parsed regions_v2.yaml dict
        country: Country key (default "US")

    Returns:
        Dict with per-region stats including pixel_count, lat/lon ranges, etc.
    """
    H, W = lat.shape
    total_cells = H * W
    stats = {}

    country_regions = regions_spec["countries"][country]["regions"]

    for idx, region in enumerate(country_regions):
        region_id = region["id"]
        region_name = region["name"]
        regime = region["regime"]
        bbox = region["bbox"]

        # Get cells belonging to this region
        region_mask = mask_int == (idx + 1)
        pixel_count = int(region_mask.sum())
        fraction = pixel_count / total_cells

        # Get lat/lon values for cells in region
        lat_vals = lat[region_mask]
        lon_vals = lon[region_mask]

        if pixel_count > 0:
            lat_min_actual = float(np.nanmin(lat_vals))
            lat_max_actual = float(np.nanmax(lat_vals))
            lon_min_actual = float(np.nanmin(lon_vals))
            lon_max_actual = float(np.nanmax(lon_vals))
            quality_flag = "ok" if fraction > 0.001 else "low_coverage"
        else:
            lat_min_actual = lat_max_actual = lon_min_actual = lon_max_actual = None
            quality_flag = "empty"

        stats[region_id] = {
            "region_name": region_name,
            "regime": regime,
            "pixel_count": pixel_count,
            "fraction_of_grid": round(fraction, 6),
            "lat_min": lat_min_actual,
            "lat_max": lat_max_actual,
            "lon_min": lon_min_actual,
            "lon_max": lon_max_actual,
            "bbox_spec": bbox,
            "quality_flag": quality_flag,
        }

    return stats


def check_no_overlap(mask_onehot: np.ndarray) -> bool:
    """Verify that region masks do not overlap.

    Args:
        mask_onehot: One-hot mask array (num_regions, H, W)

    Returns:
        True if no overlaps exist (each cell belongs to at most one region)
    """
    overlap_count = mask_onehot.sum(axis=0)
    # Allow value 0 (outside all regions) or 1 (exactly one region)
    return bool(np.all(overlap_count <= 1))


def save_region_masks_nc(
    mask_int: np.ndarray,
    mask_onehot: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    output_path: str,
    regions_spec_path: str,
    latlon_nc_path: str,
) -> None:
    """Save region masks to NetCDF.

    Args:
        mask_int: Integer mask (H, W)
        mask_onehot: One-hot mask (num_regions, H, W)
        lat: 2D latitude array (H, W)
        lon: 2D longitude array (H, W)
        output_path: Output NetCDF path
        regions_spec_path: Source regions_v2.yaml path
        latlon_nc_path: Source lat/lon NetCDF path
    """
    H, W = lat.shape
    num_regions = mask_onehot.shape[0]

    # Create coordinate arrays for region dimension
    region_ids = [f"US-R{i+1}" for i in range(num_regions)]

    ds = xr.Dataset(
        {
            "latitude": (["height", "width"], lat),
            "longitude": (["height", "width"], lon),
            "region_mask_integer": (["height", "width"], mask_int),
            "region_mask_onehot": (
                ["region_id", "height", "width"],
                mask_onehot.astype(np.float32),
            ),
        },
        coords={
            "height": np.arange(H),
            "width": np.arange(W),
            "region_id": region_ids,
        },
    )

    ds["latitude"].attrs["_FillValue"] = np.nan
    ds["longitude"].attrs["_FillValue"] = np.nan
    ds["region_mask_integer"].attrs["_FillValue"] = -1
    ds["region_mask_onehot"].attrs["_FillValue"] = np.nan

    # Global attributes for no-leakage declaration
    ds.attrs["region_definitions_source"] = regions_spec_path
    ds.attrs["geolocation_source"] = latlon_nc_path
    ds.attrs["method"] = "bbox polygon grid assignment"
    ds.attrs["no_leakage"] = (
        "analysis_increments/model_errors/target_labels not used"
    )

    ds.to_netcdf(output_path)
    ds.close()
    print(f"Saved: {output_path}")


def save_stats_json(stats: Dict, output_path: str) -> None:
    """Save region statistics to JSON.

    Args:
        stats: Region statistics dict
        output_path: Output JSON path
    """
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {output_path}")


def generate_markdown_report(
    stats: Dict,
    output_path: str,
    regions_spec_path: str,
    latlon_nc_path: str,
    no_overlap: bool,
) -> None:
    """Generate human-readable markdown report.

    Args:
        stats: Region statistics dict
        output_path: Output markdown path
        regions_spec_path: Source regions_v2.yaml path
        latlon_nc_path: Source lat/lon NetCDF path
        no_overlap: Whether overlap check passed
    """
    lines = [
        "# US Scientific Region Mask Summary",
        "",
        "## Metadata",
        f"- Region definitions: `{regions_spec_path}`",
        f"- Geolocation source: `{latlon_nc_path}`",
        f"- Method: Bounding box polygon grid assignment",
        f"- No-leakage: analysis_increments/model_errors/target_labels NOT used",
        "",
        "## Overlap Check",
        f"- **Passed**: Regions do not overlap" if no_overlap
        else "- **FAILED**: Regions overlap detected!",
        "",
        "## Region Statistics",
        "",
        "| Region ID | Name | Regime | Pixels | Fraction | "
        "Lat Range | Lon Range | Quality |",
        "|-----------|------|--------|--------|----------|"
        "----------|----------|--------|",
    ]

    for region_id, s in stats.items():
        lat_range = f"[{s['lat_min']:.2f}, {s['lat_max']:.2f}]" if s['lat_min'] else "N/A"
        lon_range = f"[{s['lon_min']:.2f}, {s['lon_max']:.2f}]" if s['lon_min'] else "N/A"
        lines.append(
            f"| {region_id} | {s['region_name']} | {s['regime']} | "
            f"{s['pixel_count']:,} | {s['fraction_of_grid']:.6f} | "
            f"{lat_range} | {lon_range} | {s['quality_flag']} |"
        )

    lines.append("")
    lines.append("## No-Leakage Declaration")
    lines.append("")
    lines.append("Region masks were constructed **only** from:")
    lines.append("- Fixed lat/lon bounding boxes from `specs/regions_v2.yaml`")
    lines.append("- Lat/lon grids from `artifacts/geolocation/US_latlon.nc`")
    lines.append("")
    lines.append("**NOT** from:")
    lines.append("- DA.nc analysis increments")
    lines.append("- Model prediction errors")
    lines.append("- target_query labels")
    lines.append("- Any training or evaluation results")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")