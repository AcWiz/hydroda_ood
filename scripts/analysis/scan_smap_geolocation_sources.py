#!/usr/bin/env python
"""
Recursive SMAP directory scan for geolocation source files.

Outputs:
  - artifacts/audits/smap_directory_manifest.json
  - reports/audits/smap_directory_manifest.md

Usage:
    python scripts/analysis/scan_smap_geolocation_sources.py \
        --smap-dir /fastersharefiles2/fenglonghan/dataset/SMAP \
        --out-manifest-json artifacts/audits/smap_directory_manifest.json \
        --out-manifest-md reports/audits/smap_directory_manifest.md
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PRIORITY_KEYWORDS = [
    "lat", "latitude", "lon", "longitude", "coord", "coords", "grid",
    "geolocation", "geo", "projection", "proj", "crs", "transform",
    "affine", "ease", "smap", "domain", "mask", "land", "conus",
    "us", "usa", "north_america", "crop", "resample", "row", "col",
    "ij", "xy",
]

CANDIDATE_EXTENSIONS = {
    ".nc", ".nc4", ".h5", ".hdf5", ".zarr",
    ".npy", ".npz", ".pkl", ".pickle", ".mat",
    ".csv", ".txt", ".json", ".yaml", ".yml",
    ".tif", ".tiff", ".vrt", ".shp", ".gpkg",
    ".parquet", ".feather",
    ".png", ".jpg", ".pdf", ".md", ".py", ".ipynb",
}

# Candidate classes per protocol section 4
CANDIDATE_CLASSES = {
    "latlon_npy": {
        "priority": 1,
        "description": "lat.npy/lon.npy or america_lat.npy/america_lon.npy 1D vectors",
        "keywords": ["lat", "lon", "latitude", "longitude"],
        "extensions": [".npy"],
    },
    "grid_coord_nc": {
        "priority": 2,
        "description": "NetCDF with embedded lat/lon coords or grid_mapping",
        "keywords": ["lat", "lon", "grid", "coord", "proj", "crs"],
        "extensions": [".nc", ".nc4"],
    },
    "proj_config": {
        "priority": 3,
        "description": "JSON/YAML with proj4/epsg/crs/transform config",
        "keywords": ["epsg", "proj4", "crs", "transform", "affine", "geotransform"],
        "extensions": [".json", ".yaml", ".yml", ".txt"],
    },
    "script_with_proj": {
        "priority": 4,
        "description": "Python/Jupyter with EASE/pyproj/cartopy usage",
        "keywords": ["ease", "pyproj", "cartopy", "meshgrid", "rasterio"],
        "extensions": [".py", ".ipynb"],
    },
    "shape_arrays": {
        "priority": 5,
        "description": "Arrays with shape [256], [640], [256,640] matching grid",
        "extensions": [".npy", ".npz", ".pkl", ".mat"],
    },
}


def filename_matches(filename: str) -> bool:
    """Check if filename contains any priority keyword (case-insensitive)."""
    name_lower = filename.lower()
    return any(kw in name_lower for kw in PRIORITY_KEYWORDS)


def scan_directory(root_dir: str, max_depth: int = 5) -> dict:
    """
    Recursively scan directory for geolocation candidate files.

    Returns dict with scan results.
    """
    root_path = Path(root_dir)
    scan_timestamp = datetime.now().isoformat()

    all_files = []
    candidates = {"latlon_npy": [], "grid_coord_nc": [], "proj_config": [],
                  "script_with_proj": [], "shape_arrays": []}
    non_candidates = []
    errors = []

    # Track directory structure
    dir_tree = {}

    def process_file(filepath: Path, rel_path: str):
        """Process a single file and categorize it."""
        fname = filepath.name
        ext = filepath.suffix

        file_info = {
            "path": str(filepath),
            "relative_path": rel_path,
            "filename": fname,
            "extension": ext,
            "size_bytes": filepath.stat().st_size if filepath.exists() else 0,
        }

        # Check for numpy arrays with shape info
        if ext == ".npy":
            try:
                arr = np.load(filepath, allow_pickle=False)
                file_info["npy_shape"] = list(arr.shape) if hasattr(arr, 'shape') else None
                file_info["npy_dtype"] = str(arr.dtype) if hasattr(arr, 'dtype') else None
                if hasattr(arr, 'min') and hasattr(arr, 'max'):
                    file_info["npy_range"] = [float(arr.min()), float(arr.max())]
            except Exception as e:
                file_info["npy_error"] = str(e)

        all_files.append(file_info)

        # Categorize as candidate
        categorized = False

        # Priority 1: lat/lon npy files
        if ext == ".npy" and filename_matches(fname):
            candidates["latlon_npy"].append(file_info)
            categorized = True

        # Priority 2: NetCDF with potential spatial coords
        elif ext in (".nc", ".nc4"):
            candidates["grid_coord_nc"].append(file_info)
            categorized = True

        # Priority 3: proj config files
        elif ext in (".json", ".yaml", ".yml", ".txt"):
            if filename_matches(fname):
                candidates["proj_config"].append(file_info)
                categorized = True

        # Priority 4: scripts with projection keywords
        elif ext in (".py", ".ipynb"):
            if filename_matches(fname):
                candidates["script_with_proj"].append(file_info)
                categorized = True

        # Priority 5: shape arrays matching grid
        elif ext in (".npy", ".npz", ".pkl", ".mat"):
            npy_shape = file_info.get("npy_shape", [])
            if npy_shape and (
                npy_shape == [256] or npy_shape == [640] or
                npy_shape == [256, 640] or npy_shape == [640, 256]
            ):
                file_info["candidate_class"] = "shape_arrays"
                candidates["shape_arrays"].append(file_info)
                categorized = True

        if not categorized:
            non_candidates.append(file_info)

    def scan_path(path: Path, rel_prefix: str = "", depth: int = 0):
        """Recursively scan a path."""
        if depth > max_depth:
            return

        try:
            for entry in sorted(path.iterdir()):
                rel_path = os.path.join(rel_prefix, entry.name) if rel_prefix else entry.name

                if entry.is_file():
                    process_file(entry, rel_path)
                elif entry.is_dir():
                    # Skip hidden dirs and large data dirs
                    if entry.name.startswith('.') or entry.name in ('lzx_2018', 'year_norm_tensor2'):
                        continue
                    dir_tree[rel_path] = {"type": "directory", "children": []}
                    scan_path(entry, rel_path, depth + 1)
        except PermissionError as e:
            errors.append({"path": str(path), "error": "Permission denied"})
        except Exception as e:
            errors.append({"path": str(path), "error": str(e)})

    scan_path(root_path)

    return {
        "scan_timestamp": scan_timestamp,
        "root_dir": str(root_path),
        "max_depth": max_depth,
        "all_files_count": len(all_files),
        "candidates": candidates,
        "non_candidates": non_candidates[:50],  # limit non-candidates
        "errors": errors,
        "directory_tree": dir_tree,
    }


def candidates_to_markdown(result: dict) -> str:
    """Generate human-readable markdown report."""
    lines = [
        "# SMAP Directory Geolocation Candidate Manifest",
        "",
        f"**Scan Timestamp:** {result['scan_timestamp']}",
        f"**Root Directory:** `{result['root_dir']}`",
        f"**Max Depth:** {result['max_depth']}",
        f"**Total Files Found:** {result['all_files_count']}",
        "",
    ]

    for class_name, class_info in {
        "latlon_npy": {"priority": 1, "description": "Lat/Lon Vector Files (.npy)"},
        "grid_coord_nc": {"priority": 2, "description": "NetCDF with Spatial Coords"},
        "proj_config": {"priority": 3, "description": "Projection Config (JSON/YAML/TXT)"},
        "script_with_proj": {"priority": 4, "description": "Scripts with Projection Code"},
        "shape_arrays": {"priority": 5, "description": "Arrays Matching Grid Shape [256/640]"},
    }.items():
        files = result["candidates"].get(class_name, [])
        lines.append(f"## Priority {class_info['priority']}: {class_info['description']}")
        lines.append(f"**Found:** {len(files)} candidate(s)")
        lines.append("")

        if not files:
            lines.append("*None found*")
        else:
            for f in files:
                lines.append(f"### `{f['relative_path']}`")
                lines.append(f"- Size: {f['size_bytes']:,} bytes")
                if 'npy_shape' in f:
                    lines.append(f"- Shape: {f['npy_shape']}")
                    lines.append(f"- Dtype: {f['npy_dtype']}")
                    if 'npy_range' in f:
                        lines.append(f"- Range: [{f['npy_range'][0]:.4f}, {f['npy_range'][1]:.4f}]")
                lines.append("")

    if result.get("errors"):
        lines.append("## Errors")
        for err in result["errors"]:
            lines.append(f"- `{err['path']}`: {err['error']}")
        lines.append("")

    return "\n".join(lines)


def candidates_to_json(result: dict) -> dict:
    """Convert scan result to machine-readable JSON."""
    # Simplify for JSON serialization
    output = {
        "scan_timestamp": result["scan_timestamp"],
        "root_dir": result["root_dir"],
        "max_depth": result["max_depth"],
        "all_files_count": result["all_files_count"],
        "candidates": result["candidates"],
        "non_candidates_count": len(result.get("non_candidates", [])),
        "errors": result.get("errors", []),
    }
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Scan SMAP directory for geolocation source files"
    )
    parser.add_argument(
        "--smap-dir",
        required=True,
        help="Path to SMAP data directory",
    )
    parser.add_argument(
        "--out-manifest-json",
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--out-manifest-md",
        required=True,
        help="Output Markdown path",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum directory depth to scan",
    )
    args = parser.parse_args()

    print(f"Scanning: {args.smap_dir}")
    print(f"Max depth: {args.max_depth}")

    result = scan_directory(args.smap_dir, max_depth=args.max_depth)

    # Write JSON
    os.makedirs(os.path.dirname(args.out_manifest_json), exist_ok=True)
    with open(args.out_manifest_json, "w") as f:
        json.dump(candidates_to_json(result), f, indent=2, default=str)
    print(f"JSON written to: {args.out_manifest_json}")

    # Write Markdown
    os.makedirs(os.path.dirname(args.out_manifest_md), exist_ok=True)
    md_content = candidates_to_markdown(result)
    with open(args.out_manifest_md, "w") as f:
        f.write(md_content)
    print(f"Markdown written to: {args.out_manifest_md}")

    # Summary
    total_candidates = sum(len(v) for v in result["candidates"].values())
    print(f"\nSummary:")
    print(f"  Total files scanned: {result['all_files_count']}")
    print(f"  Total candidates: {total_candidates}")
    for class_name, files in result["candidates"].items():
        if files:
            print(f"    {class_name}: {len(files)}")


if __name__ == "__main__":
    main()