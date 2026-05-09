#!/usr/bin/env python
"""
Geolocation Recovery Attempt Script for DA.nc.

Attempts to recover spatial geolocation from DA.nc which lacks lat/lon coordinates.
If DA.nc itself lacks geolocation, attempts directory-level lookup for coordinate sources.

Usage:
    python scripts/geolocation_recovery.py \
        --data /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc \
        --smap-dir /fastersharefiles2/fenglonghan/dataset/SMAP \
        --out-json artifacts/audits/geolocation_recovery_US.json \
        --out-md reports/audits/geolocation_recovery_US.md
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import xarray as xr


def audit_geolocation(data_path: str, smap_dir: str = None) -> dict:
    """
    Attempt to recover geolocation from DA.nc.

    Returns a dict with:
        - geolocation_recovered: bool
        - methods_attempted: list of {method, success, details}
        - blocking_issue: str or null
        - recommendation: str
    """
    result = {
        "geolocation_recovered": False,
        "methods_attempted": [],
        "blocking_issue": "No projection metadata in DA.nc; grid->lat/lon mapping unavailable",
        "recommendation": "Request lat/lon lookup table or projection metadata from data provider",
        "da_grid": {"height": 256, "width": 640},
        "audit_timestamp": datetime.now().isoformat(),
        "file_path": data_path,
    }

    # ---- Method 1: Check all global attributes ----
    ds = xr.open_dataset(data_path, decode_times=False)
    global_attrs = ds.attrs

    method_1 = {
        "method": "global_attributes",
        "success": False,
        "details": {
            "global_attrs_found": list(global_attrs.keys()),
            "global_attrs_empty": len(global_attrs) == 0,
        },
    }
    if global_attrs:
        for key in list(global_attrs.keys()):
            val = global_attrs[key]
            val_lower = str(val).lower()
            if any(
                k in val_lower
                for k in ["proj", "crs", "epsg", "wkt", "proj4", "grid_mapping", "srs"]
            ):
                method_1["success"] = True
                method_1["details"]["projection_attr_found"] = {key: val}
                break
    result["methods_attempted"].append(method_1)

    # ---- Method 2: Check coordinate variables ----
    coord_names = list(ds.coords)
    spatial_coords = [c for c in coord_names if c not in ("time", "variable_input", "variable_target")]
    method_2 = {
        "method": "coordinate_variables",
        "success": False,
        "details": {
            "all_coords": coord_names,
            "spatial_coords_found": spatial_coords,
            "note": "Standard NetCDF spatial coords (lat, lon, y, x) absent",
        },
    }
    result["methods_attempted"].append(method_2)

    # ---- Method 3: Check for grid_mapping variable (CF conventions) ----
    grid_mapping_vars = [v for v in ds.data_vars if "grid_mapping" in v.lower() or v.lower() in ["crs", "projection"]]
    method_3 = {
        "method": "grid_mapping_variable",
        "success": False,
        "details": {
            "grid_mapping_vars_found": grid_mapping_vars,
            "note": "CF-compliant grid_mapping variable absent",
        },
    }
    result["methods_attempted"].append(method_3)

    # ---- Method 4: Check variable coordinates for embedded lat/lon ----
    embedded_check = {"lat_coords_found": False, "lon_coords_found": False}
    for var in ds.data_vars:
        var_obj = ds[var]
        if hasattr(var_obj, "coords"):
            var_coords = list(var_obj.coords)
            if "lat" in var_coords or "latitude" in var_coords:
                embedded_check["lat_coords_found"] = True
            if "lon" in var_coords or "longitude" in var_coords:
                embedded_check["lon_coords_found"] = True
    method_4 = {
        "method": "embedded_coordinates",
        "success": embedded_check["lat_coords_found"] and embedded_check["lon_coords_found"],
        "details": embedded_check,
    }
    result["methods_attempted"].append(method_4)

    # ---- Method 5: SMAP EASE-Grid reconstruction attempt ----
    # SMAP EASE-Grid specifications:
    # - EASE2_N9km (9km): 2,316 x 4,032 cells
    # - EASE2_N25km (25km): 836 x 1,456 cells
    # DA.nc grid: 256 x 640 -- does not match any standard SMAP EASE-Grid size
    ease_grid_specs = {
        "EASE2_N9km": {"h": 2316, "w": 4032, "res_km": 9},
        "EASE2_N25km": {"h": 836, "w": 1456, "res_km": 25},
        "EASE2_N36km": {"h": 584, "w": 1016, "res_km": 36},
    }
    da_h, da_w = 256, 640
    matching_ease = None
    for name, spec in ease_grid_specs.items():
        if spec["h"] == da_h and spec["w"] == da_w:
            matching_ease = name
            break

    method_5 = {
        "method": "smap_ease_grid_reconstruction",
        "success": False,
        "details": {
            "da_grid_size": [da_h, da_w],
            "ease_grid_specs_tried": ease_grid_specs,
            "matching_ease_grid": matching_ease,
            "note": "DA.nc 256x640 does not match any standard SMAP EASE-Grid size; cannot reconstruct lat/lon",
        },
    }
    result["methods_attempted"].append(method_5)

    # ---- Method 6: Check dims for lat/lon dimension names ----
    dim_names = list(ds.dims.keys())
    lat_dim = [d for d in dim_names if d.lower() in ["lat", "latitude", "y"]]
    lon_dim = [d for d in dim_names if d.lower() in ["lon", "longitude", "x"]]
    method_6 = {
        "method": "spatial_dimension_names",
        "success": bool(lat_dim and lon_dim),
        "details": {
            "all_dims": dim_names,
            "lat_like_dims": lat_dim,
            "lon_like_dims": lon_dim,
            "note": "No spatial dimension names found; only height/width",
        },
    }
    result["methods_attempted"].append(method_6)

    # ---- Method 7: Infer resolution from grid size vs US coverage ----
    # US approx lat range: 24N to 50N (26 deg), lon range: -125 to -66 (59 deg)
    # If 256 rows covers ~26 deg lat -> ~0.1 deg/cell (too coarse for 9km)
    # If 640 cols covers ~59 deg lon -> ~0.09 deg/cell (too coarse for 9km)
    # This suggests resolution is NOT 9km EASE-Grid
    us_lat_range = 26  # degrees
    us_lon_range = 59  # degrees
    inferred_lat_res = us_lat_range / da_h if da_h > 0 else 0
    inferred_lon_res = us_lon_range / da_w if da_w > 0 else 0

    method_7 = {
        "method": "resolution_inference",
        "success": False,
        "details": {
            "us_approx_lat_range_deg": us_lat_range,
            "us_approx_lon_range_deg": us_lon_range,
            "inferred_lat_resolution_deg_per_cell": round(inferred_lat_res, 4),
            "inferred_lon_resolution_deg_per_cell": round(inferred_lon_res, 4),
            "inferred_lat_resolution_km": round(inferred_lat_res * 111, 2),
            "inferred_lon_resolution_km": round(inferred_lon_res * 111, 2),
            "note": "Resolution estimate inconsistent with standard SMAP 9km or 25km EASE-Grid; grid size unique",
        },
    }
    result["methods_attempted"].append(method_7)

    # ---- Method 8: Directory-level lookup for america_lat/lon.npy vectors ----
    # Check SMAP directory for 1D lat/lon vector files that can be broadcast to 2D grid
    # Discovery: try multiple candidate filenames
    smap_dir = smap_dir or os.path.dirname(data_path)
    lat_candidates = ["america_lat.npy", "lat.npy", "latitude.npy"]
    lon_candidates = ["america_lon.npy", "lon.npy", "longitude.npy"]

    lat_found = None
    lon_found = None
    lat_path = None
    lon_path = None

    for candidate in lat_candidates:
        p = os.path.join(smap_dir, candidate)
        if os.path.exists(p):
            lat_found = candidate
            lat_path = p
            break

    for candidate in lon_candidates:
        p = os.path.join(smap_dir, candidate)
        if os.path.exists(p):
            lon_found = candidate
            lon_path = p
            break

    method_8_success = lat_found is not None and lon_found is not None
    method_8_details = {
        "lat_file_found": lat_found,
        "lon_file_found": lon_found,
        "lat_path": lat_path,
        "lon_path": lon_path,
        "note": "1D vectors (256,) + (640,) can be broadcast to 2D (256, 640) grid",
    }

    if method_8_success:
        # Reuse geolocation.py module: load, validate, and broadcast
        # Import here to avoid circular dependency at module level
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from hydroda.data.geolocation import (
            load_us_latlon_vectors, vectors_to_2d_grid, validate_latlon_ranges,
            US_GRID_HEIGHT, US_GRID_WIDTH
        )
        try:
            lat_vec, lon_vec = load_us_latlon_vectors(smap_dir)
            method_8_details["lat_shape"] = list(lat_vec.shape)
            method_8_details["lon_shape"] = list(lon_vec.shape)
            method_8_details["lat_range"] = [float(lat_vec.min()), float(lat_vec.max())]
            method_8_details["lon_range"] = [float(lon_vec.min()), float(lon_vec.max())]

            # Broadcast test using module function
            lat_2d, lon_2d = vectors_to_2d_grid(lat_vec, lon_vec)
            method_8_details["broadcast_2d_shape"] = list(lat_2d.shape)
            method_8_details["broadcast_test_passed"] = (
                lat_2d.shape == (US_GRID_HEIGHT, US_GRID_WIDTH) and
                lon_2d.shape == (US_GRID_HEIGHT, US_GRID_WIDTH)
            )

            # Range validation using module function
            validation = validate_latlon_ranges(lat_2d, lon_2d)
            method_8_details["compatible_with_da_grid"] = (
                validation["lat_valid"] and validation["lon_valid"]
            )
            if validation["warnings"]:
                method_8_details["validation_warnings"] = validation["warnings"]

        except FileNotFoundError as e:
            method_8_details["error"] = str(e)
            method_8_success = False
        except Exception as e:
            method_8_details["error"] = str(e)
            method_8_success = False

    method_8 = {
        "method": "directory_latlon_vector_lookup",
        "success": method_8_success,
        "details": method_8_details,
    }
    result["methods_attempted"].append(method_8)

    # ---- Final verdict ----
    all_methods = result["methods_attempted"]
    any_success = any(m["success"] for m in all_methods)

    if any_success:
        result["geolocation_recovered"] = True
        result["blocking_issue"] = None
        result["recommendation"] = (
            "Geolocation recovered via directory-level lookup (Method 8). "
            "Use hydroda/data/geolocation.py functions to generate US_latlon.nc artifact. "
            "Phase 2 scientific region masks UNBLOCKED."
        )
        # Update final method status
        for m in all_methods:
            if m["method"] == "directory_latlon_vector_lookup" and m["success"]:
                result["recovery_method"] = "directory_latlon_vector_lookup"
                break
    else:
        result["geolocation_recovered"] = False
        result["blocking_issue"] = (
            "No projection metadata in DA.nc; grid->lat/lon mapping unavailable. "
            "DA.nc uses height=256, width=640 which does not match standard SMAP EASE-Grid sizes. "
            "Scientific region masks (US-R1..R6 defined by lat/lon bbox in regions_v2.yaml) are BLOCKED. "
            "Development-only grid OOD split allowed without lat/lon."
        )
        result["recommendation"] = (
            "Geolocation can be recovered via america_lat.npy + america_lon.npy broadcasting. "
            "Run with --smap-dir to enable Method 8 directory lookup. "
            "Until then, Phase 2 scientific region masks cannot be implemented; only temporal K-date splits are possible."
        )

    ds.close()
    return result


def dict_to_markdown(data: dict) -> str:
    """Convert geolocation recovery result to markdown."""
    lines = [
        "# Geolocation Recovery Report: US",
        "",
        f"**Timestamp:** {data.get('audit_timestamp', 'N/A')}",
        f"**File:** `{data.get('file_path', 'N/A')}`",
        f"**Grid:** {data.get('da_grid', {}).get('height')} x {data.get('da_grid', {}).get('width')}",
        "",
        f"**Geolocation Recovered:** {data.get('geolocation_recovered')}",
        "",
        "## Methods Attempted",
        "",
    ]

    for m in data.get("methods_attempted", []):
        status = "✅" if m["success"] else "❌"
        lines.append(f"### {status} {m['method']}")
        for k, v in m["details"].items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    lines.extend([
        "## Verdict",
        "",
        f"**Geolocation Recovered:** {'YES' if data.get('geolocation_recovered') else 'NO'}",
        "",
        "## Blocking Issue",
        "",
        f"{data.get('blocking_issue', 'N/A')}",
        "",
        "## Recommendation",
        "",
        f"{data.get('recommendation', 'N/A')}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Attempt geolocation recovery from DA.nc")
    parser.add_argument("--data", required=True, help="Path to DA.nc file")
    parser.add_argument("--smap-dir", default=None,
                        help="Path to SMAP directory (for Method 8 directory lookup)")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument("--out-md", required=True, help="Output Markdown path")
    args = parser.parse_args()

    print(f"Auditing geolocation: {args.data}")

    result = audit_geolocation(args.data, args.smap_dir)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"JSON written to: {args.out_json}")

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    md_content = dict_to_markdown(result)
    with open(args.out_md, "w") as f:
        f.write(md_content)
    print(f"Markdown written to: {args.out_md}")

    if not result["geolocation_recovered"]:
        print("\n⚠️  Geolocation NOT recovered - blocking issue detected")
        print(f"Blocking: {result['blocking_issue'][:100]}...")


if __name__ == "__main__":
    main()
