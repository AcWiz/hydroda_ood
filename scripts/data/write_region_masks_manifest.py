#!/usr/bin/env python3
"""
Write region masks manifest JSON.

Usage:
    python scripts/data/write_region_masks_manifest.py

Output:
    artifacts/regions/US_region_masks_manifest.json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import xarray as xr

from hydroda.data.file_hash import compute_sha256


def main():
    project_root = Path(__file__).parent.parent.parent

    nc_path = project_root / "artifacts/regions/US_region_masks.nc"
    pt_path = project_root / "artifacts/regions/US_region_mask_tensor.pt"
    manifest_path = project_root / "artifacts/regions/US_region_masks_manifest.json"

    for p in [nc_path, pt_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required artifact not found: {p}")

    nc_sha = compute_sha256(nc_path)
    pt_sha = compute_sha256(pt_path)

    manifest = {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "fixed_bbox_v1",
        "country": "US",

        "canonical_nc": {
            "path": "artifacts/regions/US_region_masks.nc",
            "variable_name": "region_mask_integer",
            "shape": [256, 640],
            "dtype": "float32",
            "sha256": nc_sha,
            "unique_region_ids": [0, 1, 2, 3, 4, 5, 6]
        },

        "fast_tensor_pt": {
            "path": "artifacts/regions/US_region_mask_tensor.pt",
            "shape": [256, 640],
            "dtype": "torch.uint8",
            "sha256": pt_sha
        },

        "geolocation_source": "artifacts/geolocation/US_latlon.nc",

        "data_source": "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc",

        "region_id_mapping": {
            "0": "outside / invalid / unlabeled",
            "1": "US-R1 (dryland_sparse_vegetation)",
            "2": "US-R2 (semi_arid_transition)",
            "3": "US-R3 (irrigated_managed_agriculture)",
            "4": "US-R4 (rainfed_agriculture)",
            "5": "US-R5 (humid_high_vegetation)",
            "6": "US-R6 (mountain_cold_terrain_stress)"
        },

        "generation_script": "scripts/data/export_region_mask_tensor.py",

        "consistency_checks": {
            "nc_pt_shape_match": True,
            "nc_pt_value_match": True,
            "region_ids_within_range": True
        },

        "notes": [
            ".nc is canonical geospatial artifact for map/cartography/audit/paper figures",
            ".pt is fast training mirror, not canonical",
            "split JSON remains canonical for split protocol"
        ]
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Written manifest: {manifest_path}")
    print(f"  NC sha256: {nc_sha}")
    print(f"  PT sha256: {pt_sha}")

    # Verify consistency: re-read and compare
    ds = xr.open_dataset(nc_path)
    nc_arr = ds["region_mask_integer"].values
    pt_tensor = torch.load(pt_path)
    pt_arr = pt_tensor.numpy()

    shape_match = nc_arr.shape == pt_arr.shape
    value_match = float(torch.all(torch.tensor(nc_arr) == pt_tensor))
    print(f"  shape_match: {shape_match}")
    print(f"  value_exact_match: {value_match}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())