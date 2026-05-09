"""
Geolocation recovery utilities for HydroDA-OOD.

This module provides functions for loading US lat/lon vectors from america_lat.npy
and america_lon.npy, broadcasting them to 2D grids, and saving as NetCDF.

No leakage: These functions only use static coordinate vectors, never analysis
increments, model errors, or target query labels.
"""

import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr


# Expected shapes for US grid
US_GRID_HEIGHT = 256
US_GRID_WIDTH = 640

# Valid ranges for US continental coverage (with margin)
US_LAT_RANGE = (15.0, 75.0)
US_LON_RANGE = (-180.0, -50.0)  # Also accepts [180, 310] for 0-360


DEFAULT_SMAP_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"


def load_us_latlon_vectors(smap_dir: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load america_lat.npy (256,) and america_lon.npy (640,) vectors.

    Args:
        smap_dir: Path to SMAP directory containing america_lat.npy and america_lon.npy.
                  Defaults to DEFAULT_SMAP_DIR.

    Returns:
        Tuple of (lat_vector, lon_vector) both as numpy arrays.

    Raises:
        FileNotFoundError: If latitude or longitude files are missing.
        ValueError: If shapes don't match expected [256] and [640].
    """
    if smap_dir is None:
        smap_dir = DEFAULT_SMAP_DIR

    lat_path = Path(smap_dir) / "america_lat.npy"
    lon_path = Path(smap_dir) / "america_lon.npy"

    if not lat_path.exists():
        raise FileNotFoundError(f"Latitude file not found: {lat_path}")
    if not lon_path.exists():
        raise FileNotFoundError(f"Longitude file not found: {lon_path}")

    lat_vec = np.load(lat_path)
    lon_vec = np.load(lon_path)

    if lat_vec.shape != (US_GRID_HEIGHT,):
        raise ValueError(
            f"Latitude vector shape {lat_vec.shape} != expected ({US_GRID_HEIGHT},). "
            f"File: {lat_path}"
        )
    if lon_vec.shape != (US_GRID_WIDTH,):
        raise ValueError(
            f"Longitude vector shape {lon_vec.shape} != expected ({US_GRID_WIDTH},). "
            f"File: {lon_path}"
        )

    return lat_vec, lon_vec


def vectors_to_2d_grid(
    lat_vec: np.ndarray, lon_vec: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Broadcast 1D lat/lon vectors to 2D grids via numpy broadcasting.

    Args:
        lat_vec: 1D latitude vector of shape (256,)
        lon_vec: 1D longitude vector of shape (640,)

    Returns:
        Tuple of (lat_2d, lon_2d) each of shape (256, 640).

    The broadcasting works as:
        lat_2d = lat_vec[:, None]   -> shape (256, 1)
        lon_2d = lon_vec[None, :]    -> shape (1, 640)
        result = broadcast_to(lat_2d, (256, 640)) -> shape (256, 640)
    """
    if lat_vec.ndim != 1 or lat_vec.shape[0] != US_GRID_HEIGHT:
        raise ValueError(f"lat_vec must be 1D with shape ({US_GRID_HEIGHT},), got {lat_vec.shape}")
    if lon_vec.ndim != 1 or lon_vec.shape[0] != US_GRID_WIDTH:
        raise ValueError(f"lon_vec must be 1D with shape ({US_GRID_WIDTH},), got {lon_vec.shape}")

    lat_2d = np.broadcast_to(lat_vec[:, None], (US_GRID_HEIGHT, US_GRID_WIDTH))
    lon_2d = np.broadcast_to(lon_vec[None, :], (US_GRID_HEIGHT, US_GRID_WIDTH))

    return lat_2d.copy(), lon_2d.copy()


def validate_latlon_ranges(
    lat: np.ndarray, lon: np.ndarray
) -> dict:
    """
    Validate that lat/lon grids fall within acceptable US ranges.

    Args:
        lat: 2D latitude array of shape (256, 640)
        lon: 2D longitude array of shape (256, 640)

    Returns:
        Dict with validation results:
            - lat_valid: bool
            - lon_valid: bool
            - lat_range: tuple(min, max)
            - lon_range: tuple(min, max)
            - warnings: list of warning strings
    """
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())

    result = {
        "lat_valid": US_LAT_RANGE[0] <= lat_min and lat_max <= US_LAT_RANGE[1],
        "lon_valid": (US_LON_RANGE[0] <= lon_min and lon_max <= US_LON_RANGE[1]) or
                     (180 <= lon_min and lon_max <= 310),
        "lat_range": (lat_min, lat_max),
        "lon_range": (lon_min, lon_max),
        "warnings": [],
    }

    if not result["lat_valid"]:
        result["warnings"].append(
            f"Latitude range [{lat_min:.4f}, {lat_max:.4f}] outside US range "
            f"{US_LAT_RANGE}"
        )

    if not result["lon_valid"]:
        result["warnings"].append(
            f"Longitude range [{lon_min:.4f}, {lon_max:.4f}] outside US range "
            f"{US_LON_RANGE}"
        )

    return result


def save_latlon_nc(
    lat_2d: np.ndarray,
    lon_2d: np.ndarray,
    output_path: str,
    metadata: dict = None
) -> None:
    """
    Save 2D lat/lon grids as NetCDF file.

    Args:
        lat_2d: 2D latitude array of shape (256, 640)
        lon_2d: 2D longitude array of shape (256, 640)
        output_path: Path to output NetCDF file
        metadata: Optional metadata dict to attach as global attributes
    """
    if lat_2d.shape != lon_2d.shape:
        raise ValueError(
            f"lat_2d shape {lat_2d.shape} != lon_2d shape {lon_2d.shape}"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ds = xr.Dataset(
        data_vars={
            "latitude": (["height", "width"], lat_2d.astype(np.float64)),
            "longitude": (["height", "width"], lon_2d.astype(np.float64)),
        },
        attrs={
            "source": "america_lat.npy + america_lon.npy via numpy broadcasting",
            "grid_shape": f"{US_GRID_HEIGHT}x{US_GRID_WIDTH}",
            "transformation": "lat_vec[:, None] + lon_vec[None, :] broadcasting",
        }
    )

    if metadata:
        for k, v in metadata.items():
            ds.attrs[k] = v

    ds.to_netcdf(output_path, format="NETCDF4")
    ds.close()


def load_latlon_from_nc(nc_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load 2D lat/lon grids from a previously saved NetCDF file.

    Args:
        nc_path: Path to NetCDF file with latitude/longitude variables.

    Returns:
        Tuple of (lat_2d, lon_2d) arrays.
    """
    ds = xr.open_dataset(nc_path)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    ds.close()
    return lat, lon


def generate_grid_mapping_json(
    output_path: str,
    source_info: dict = None
) -> None:
    """
    Generate US_grid_mapping.json artifact documenting the coordinate system.

    Since we're using broadcasting from 1D vectors (not a projected CRS),
    this documents the implicit equirectangular projection.

    Args:
        output_path: Path to output JSON file
        source_info: Optional dict with source file information
    """
    mapping = {
        "coordinate_system": "equirectangular",
        "grid_shape": {"height": US_GRID_HEIGHT, "width": US_GRID_WIDTH},
        "transformation": "1D_lat[256] + 1D_lon[640] -> 2D via broadcasting",
        "lat_vector_source": "america_lat.npy",
        "lon_vector_source": "america_lon.npy",
        "note": "No map projection - direct lat/lon from broadcasting. "
                "US continental coverage from ~26.8N to ~49.8N and ~-126.7E to ~-67.1E.",
    }

    if source_info:
        mapping["source_info"] = source_info

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)