"""
Region artifacts loader utility.

Provides unified interface for loading US region masks from canonical NC
and fast PyTorch tensor mirror.

Usage:
    from hydroda.data.region_artifacts import (
        load_region_mask_nc,
        load_region_mask_tensor,
        load_region_mask_fast,
        load_region_manifest,
        validate_region_mask_tensor,
    )
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import xarray as xr


# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

REGION_MASKS_NC = Path("artifacts/regions/US_region_masks.nc")
REGION_MASK_TENSOR_PT = Path("artifacts/regions/US_region_mask_tensor.pt")
REGION_MASKS_MANIFEST = Path("artifacts/regions/US_region_masks_manifest.json")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_region_mask_nc(
    path: str | Path = None,
    variable: str = "region_mask_integer",
) -> np.ndarray:
    """
    Load region mask from canonical NC file.

    Args:
        path: Path to .nc file. Defaults to REGION_MASKS_NC.
        variable: Variable name in the NC file.

    Returns:
        numpy.ndarray shape (256, 640), dtype float32
    """
    if path is None:
        path = REGION_MASKS_NC
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Region mask NC not found: {path}")

    with xr.open_dataset(path) as ds:
        if variable not in ds:
            available = list(ds.data_vars)
            raise ValueError(
                f"Variable '{variable}' not found in {path}. Available: {available}"
            )
        arr = ds[variable].values
    return arr  # shape (256, 640), dtype float32


def load_region_mask_tensor(
    path: str | Path = None,
) -> torch.Tensor:
    """
    Load region mask tensor from .pt mirror.

    Args:
        path: Path to .pt file. Defaults to REGION_MASK_TENSOR_PT.

    Returns:
        torch.Tensor shape (256, 640), dtype torch.uint8
    """
    if path is None:
        path = REGION_MASK_TENSOR_PT
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Region mask tensor .pt not found: {path}")

    tensor = torch.load(path)
    return tensor


def load_region_manifest(
    path: str | Path = None,
) -> dict:
    """
    Load region mask manifest JSON.

    Args:
        path: Path to manifest JSON. Defaults to REGION_MASKS_MANIFEST.

    Returns:
        dict with manifest fields.
    """
    if path is None:
        path = REGION_MASKS_MANIFEST
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Region mask manifest not found: {path}")

    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_region_mask_tensor(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, int] = (256, 640),
    allowed_ids: range = None,
) -> bool:
    """
    Validate tensor shape and region id range.

    Args:
        tensor: torch.Tensor to validate.
        expected_shape: Expected (height, width).
        allowed_ids: range of allowed region ids. Defaults to range(7) (0..6).

    Returns:
        True if valid.

    Raises:
        ValueError: If shape or values are out of range.
    """
    if allowed_ids is None:
        allowed_ids = range(7)

    if tensor.shape != expected_shape:
        raise ValueError(
            f"Region mask tensor shape {tensor.shape} != expected {expected_shape}"
        )

    unique_ids = sorted(tensor.unique().tolist())
    invalid_ids = [x for x in unique_ids if int(x) not in allowed_ids]
    if invalid_ids:
        raise ValueError(
            f"Region mask tensor contains invalid ids: {invalid_ids}. "
            f"Expected ids in {list(allowed_ids)}"
        )

    return True


# ---------------------------------------------------------------------------
# Fast loader with fallback
# ---------------------------------------------------------------------------

def load_region_mask_fast(
    prefer_pt: bool = True,
    nc_path: str | Path = None,
    pt_path: str | Path = None,
    variable: str = "region_mask_integer",
) -> np.ndarray:
    """
    Load region mask as numpy array. Tries .pt first if prefer_pt=True.

    This is the main entry point for dataset code that needs numpy arrays.

    Args:
        prefer_pt: If True, try .pt mirror first, fallback to .nc.
                   If False, always load from .nc.
        nc_path: Path to canonical NC file.
        pt_path: Path to fast .pt tensor file.
        variable: Variable name in NC file.

    Returns:
        numpy.ndarray shape (256, 640)
    """
    if nc_path is None:
        nc_path = REGION_MASKS_NC
    if pt_path is None:
        pt_path = REGION_MASK_TENSOR_PT

    nc_path = Path(nc_path)
    pt_path = Path(pt_path)

    if prefer_pt and pt_path.exists():
        # Load from .pt mirror, return as numpy
        tensor = torch.load(pt_path)
        return tensor.numpy()

    # Fallback to NC
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Neither .pt nor .nc found. Checked: {pt_path}, {nc_path}"
        )
    return load_region_mask_nc(nc_path, variable)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def load_region_mask_onehot() -> np.ndarray:
    """
    Load one-hot encoded region mask from canonical NC.

    Returns:
        np.ndarray shape (6, 256, 640), dtype float32
    """
    nc_path = Path(REGION_MASKS_NC)
    with xr.open_dataset(nc_path) as ds:
        arr = ds["region_mask_onehot"].values
    return arr