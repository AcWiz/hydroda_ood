#!/usr/bin/env python3
"""
Export US region mask from canonical NC to fast PyTorch tensor mirror.

Usage:
    python scripts/export_region_mask_tensor.py

Output:
    artifacts/regions/US_region_mask_tensor.pt
    - Single torch.Tensor, shape=(256, 640), dtype=torch.uint8
    - Values: 0 (outside/invalid), 1-6 (US-R1..US-R6)
    - NC canonical is float32; PT fast mirror is uint8 (value-level equality after cast)
"""

import sys
from pathlib import Path

import torch
import xarray as xr

from hydroda.data.file_hash import compute_sha256


def main():
    project_root = Path(__file__).parent.parent
    nc_path = project_root / "artifacts/regions/US_region_masks.nc"
    pt_path = project_root / "artifacts/regions/US_region_mask_tensor.pt"

    print(f"Reading canonical NC: {nc_path}")
    ds = xr.open_dataset(nc_path)
    var_name = "region_mask_integer"
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in {nc_path}. Available: {list(ds.data_vars)}")

    arr = ds[var_name].values  # shape (256, 640)
    print(f"  array shape: {arr.shape}, dtype: {arr.dtype}")

    # Convert to torch.Tensor and cast to uint8 for compact fast mirror
    # NC stores as float32 but values are 0..6 integers; PT uses uint8
    tensor = torch.from_numpy(arr.copy()).to(torch.uint8)
    print(f"  tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
    unique_vals = sorted(tensor.unique().tolist())
    print(f"  unique ids: {[int(x) for x in unique_vals]}")

    # Save tensor (no dict wrapper, just the tensor)
    torch.save(tensor, pt_path)
    print(f"Saved .pt mirror: {pt_path}")

    # Verify by reloading from disk
    loaded = torch.load(pt_path)
    print(f"Verified: shape={loaded.shape}, dtype={loaded.dtype}")
    assert loaded.shape == (256, 640), f"Expected (256, 640), got {loaded.shape}"
    # Match PT dtype (uint8); NC is float32 but values are integer 0..6
    assert loaded.dtype == torch.uint8, f"Expected torch.uint8, got {loaded.dtype}"
    assert torch.allclose(loaded.float(), tensor.float()), "Loaded tensor does not match saved tensor"

    # Compute checksums
    nc_sha = compute_sha256(nc_path)
    pt_sha = compute_sha256(pt_path)
    print(f"NC sha256: {nc_sha}")
    print(f"PT sha256: {pt_sha}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())