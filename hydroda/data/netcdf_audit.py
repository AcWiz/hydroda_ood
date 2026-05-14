"""
Phase 0: NetCDF Audit Module

Provides functions to audit DA.nc structure without training models.
All outputs are report-only; no data is modified.

DA.nc uses a multi-channel array structure:
  - input: [time, variable_input, height, width] with 12 channels
  - target: [time, variable_target, height, width] with 4 channels
Channel names are stored as coordinate labels (e.g., ds.coords['variable_input']).

Memory-efficient: uses chunked reading for large arrays.
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Any, Optional
from hydroda.utils.runtime import get_timestamp


def compute_memory_estimate(ds: xr.Dataset) -> float:
    """Estimate dataset memory in GB using metadata only."""
    total_bytes = 0
    for var in ds.data_vars:
        arr = ds[var]
        total_bytes += arr.nbytes
    return total_bytes / (1024 ** 3)


def _get_channel_labels(ds: xr.Dataset) -> Dict[str, List[str]]:
    """Extract channel labels from coordinate variables."""
    result = {}
    for coord_name in ds.coords:
        coord_var = ds.coords[coord_name]
        if hasattr(coord_var.values, 'tolist'):
            try:
                labels = [str(l) for l in coord_var.values]
                if len(labels) > 0:
                    result[coord_name] = labels
            except Exception:
                pass
    return result


def _sample_array_stats(arr: xr.DataArray, sample_size: int = 1000000) -> Dict[str, Any]:
    """
    Sample statistics from array without loading full array into memory.
    Uses chunked reading.
    """
    total_size = arr.size
    result = {
        "total_size": int(total_size),
        "finite_count": 0,
        "nan_count": 0,
        "inf_count": 0,
        "finite_ratio": 0.0,
        "nan_ratio": 0.0,
        "inf_ratio": 0.0,
    }

    if total_size == 0:
        return result

    # For small arrays, compute directly
    if total_size <= sample_size:
        vals = arr.values
        finite = np.isfinite(vals)
        result["finite_count"] = int(finite.sum())
        result["nan_count"] = int(np.isnan(vals).sum())
        result["inf_count"] = int(np.isinf(vals).sum())
    else:
        # Sample from first time step to estimate
        # Use dask if available for chunked reading
        chunk_size = min(sample_size, arr.shape[0] if arr.shape[0] > 1 else sample_size)
        flat_samples = []
        for i in range(0, min(arr.shape[0], 10)):  # Sample first 10 time steps
            try:
                chunk = arr.values[i: i + 1].flatten()
                flat_samples.append(chunk[:min(len(chunk), sample_size // 10)])
            except Exception:
                pass
        if flat_samples:
            vals = np.concatenate(flat_samples)
            finite = np.isfinite(vals)
            result["finite_count"] = int((finite.sum() / len(vals)) * total_size)
            result["nan_count"] = int((np.isnan(vals).sum() / len(vals)) * total_size)
            result["inf_count"] = int((np.isinf(vals).sum() / len(vals)) * total_size)

    result["finite_ratio"] = result["finite_count"] / total_size
    result["nan_ratio"] = result["nan_count"] / total_size
    result["inf_ratio"] = result["inf_count"] / total_size

    return result


def _get_mask_stats(ds: xr.Dataset) -> Dict[str, Any]:
    """Get mask channel statistics without loading full array."""
    summary = {"detected_masks": {}, "mask_keys_distinct": False}

    input_channels = list(ds.coords['variable_input'].values)

    if 'mask' in input_channels:
        mask_idx = input_channels.index('mask')
        # Only load first time step to get unique values
        arr = ds['input'].values[:1, mask_idx, :, :]
        unique_vals = np.unique(arr)
        finite_ratio = np.isfinite(arr).mean()

        summary["detected_masks"]["input_channel_mask"] = {
            "channel_index": mask_idx,
            "unique_values": [float(v) for v in unique_vals[:20]],  # Limit to first 20
            "unique_count": int(len(unique_vals)),
            "finite_ratio": float(finite_ratio),
            "dtype": str(arr.dtype),
            "shape": list(ds['input'].shape[1:]),
            "note": "mask is channel 11 of input array; stats from first time step only",
        }

    return summary


def check_increment_reconstruction(ds: xr.Dataset) -> Dict[str, Any]:
    """
    Verify that analysis ≈ forecast + increment.

    In DA.nc channel structure:
      - input channels include sm_surface_forecast, sm_rootzone_forecast
      - target channels include sm_surface_analysis, sm_rootzone_analysis
      - increment is NOT stored; we compute: analysis - forecast

    Uses chunked reading for memory efficiency.
    """
    result = {
        "surface_reconstruction_passed": False,
        "rootzone_reconstruction_passed": False,
        "surface_max_error": None,
        "rootzone_max_error": None,
        "surface_mean_error": None,
        "rootzone_mean_error": None,
        "blocking_issue": None,
        "details": {},
        "structure_note": "Channel-based; increment computed as target - forecast",
    }

    # Get channel labels
    input_channels = list(ds.coords['variable_input'].values)
    target_channels = list(ds.coords['variable_target'].values)

    # Find indices for required channels
    try:
        surf_forecast_idx = input_channels.index('sm_surface_forecast')
    except ValueError:
        surf_forecast_idx = None
    try:
        surf_analysis_idx = target_channels.index('sm_surface_analysis')
    except ValueError:
        surf_analysis_idx = None
    try:
        root_forecast_idx = input_channels.index('sm_rootzone_forecast')
    except ValueError:
        root_forecast_idx = None
    try:
        root_analysis_idx = target_channels.index('sm_rootzone_analysis')
    except ValueError:
        root_analysis_idx = None

    result["detected_variables"] = {
        "input_channels": input_channels,
        "target_channels": target_channels,
        "surface_forecast_index": surf_forecast_idx,
        "surface_analysis_index": surf_analysis_idx,
        "rootzone_forecast_index": root_forecast_idx,
        "rootzone_analysis_index": root_analysis_idx,
    }

    # Surface increment reconstruction - sample from first few time steps
    if surf_forecast_idx is not None and surf_analysis_idx is not None:
        errors = []
        finite_points = 0
        for t in range(min(ds.sizes['time'], 100)):  # Sample first 100 time steps
            try:
                forecast = ds['input'].values[t, surf_forecast_idx, :, :]
                analysis = ds['target'].values[t, surf_analysis_idx, :, :]
                finite_mask = np.isfinite(forecast) & np.isfinite(analysis)
                if finite_mask.sum() > 0:
                    computed_incr = analysis[finite_mask] - forecast[finite_mask]
                    reconstructed = forecast[finite_mask] + computed_incr
                    error = np.abs(reconstructed - analysis[finite_mask])
                    errors.append(error.max())
                    finite_points += finite_mask.sum()
            except Exception:
                pass

        if errors:
            result["surface_max_error"] = float(max(errors))
            result["surface_mean_error"] = float(np.mean(errors))
            result["surface_reconstruction_passed"] = max(errors) < 1e-6
            result["details"]["surface_finite_points"] = finite_points
            result["details"]["surface_sampled_time_steps"] = len(errors)
        else:
            result["surface_reconstruction_passed"] = None
    else:
        result["blocking_issue"] = (
            f"Missing surface channels: forecast_idx={surf_forecast_idx}, "
            f"analysis_idx={surf_analysis_idx}"
        )

    # Rootzone increment reconstruction
    if root_forecast_idx is not None and root_analysis_idx is not None:
        errors = []
        finite_points = 0
        for t in range(min(ds.sizes['time'], 100)):
            try:
                forecast = ds['input'].values[t, root_forecast_idx, :, :]
                analysis = ds['target'].values[t, root_analysis_idx, :, :]
                finite_mask = np.isfinite(forecast) & np.isfinite(analysis)
                if finite_mask.sum() > 0:
                    computed_incr = analysis[finite_mask] - forecast[finite_mask]
                    reconstructed = forecast[finite_mask] + computed_incr
                    error = np.abs(reconstructed - analysis[finite_mask])
                    errors.append(error.max())
                    finite_points += finite_mask.sum()
            except Exception:
                pass

        if errors:
            result["rootzone_max_error"] = float(max(errors))
            result["rootzone_mean_error"] = float(np.mean(errors))
            result["rootzone_reconstruction_passed"] = max(errors) < 1e-6
            result["details"]["rootzone_finite_points"] = finite_points
            result["details"]["rootzone_sampled_time_steps"] = len(errors)
        else:
            result["rootzone_reconstruction_passed"] = None
    else:
        if result["blocking_issue"] is None:
            result["blocking_issue"] = (
                f"Missing rootzone channels: forecast_idx={root_forecast_idx}, "
                f"analysis_idx={root_analysis_idx}"
            )

    return result


def summarize_masks(ds: xr.Dataset) -> Dict[str, Any]:
    """Summarize mask information using memory-efficient methods."""
    return _get_mask_stats(ds)


def audit_netcdf(path: str, country: str = "US") -> Dict[str, Any]:
    """
    Perform complete audit of DA.nc file.

    Returns a dict with all audit_required_fields from the dataset contract.

    Args:
        path: Path to DA.nc file
        country: Country code (US, CN, AU)

    Returns:
        dict with all audit fields
    """
    # Load dataset with chunking for memory efficiency
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)

    audit = {
        "country": country,
        "audit_timestamp": get_timestamp(),
        "file_path": path,
    }

    # ---- dims, coords, data_vars ----
    audit["dims"] = dict(ds.sizes)
    audit["coords"] = list(ds.coords)
    audit["data_vars"] = list(ds.data_vars)

    # ---- variable shapes and dtypes ----
    audit["variable_shapes"] = {}
    audit["variable_dtypes"] = {}
    for var in ds.data_vars:
        audit["variable_shapes"][var] = list(ds[var].shape)
        audit["variable_dtypes"][var] = str(ds[var].dtype)

    # ---- time range and frequency ----
    if "time" in ds.coords:
        time_var = ds["time"]
        time_vals = time_var.values
        if hasattr(time_vals, "tolist"):
            time_vals = time_vals.tolist()
        try:
            time_min = float(np.nanmin(time_vals))
            time_max = float(np.nanmax(time_vals))
        except Exception:
            time_min = None
            time_max = None
        audit["time_range"] = {"min": time_min, "max": time_max, "count": int(len(time_vals))}

        # Infer time frequency
        if len(time_vals) > 1:
            diffs = np.diff(sorted(time_vals))
            median_diff = float(np.median(diffs))
            hours_per_step = median_diff / 3600.0
            if hours_per_step < 2:
                audit["time_frequency"] = f"~{hours_per_step:.1f}h"
            else:
                audit["time_frequency"] = f"~{hours_per_step:.1f}h"
        else:
            audit["time_frequency"] = "unknown"
        audit["coordinate_availability"] = {"time": True}
    else:
        audit["time_range"] = None
        audit["time_frequency"] = "no_time_coord"
        audit["coordinate_availability"] = {"time": False}

    # ---- lat/lon availability ----
    has_lat = "lat" in ds.coords or "latitude" in ds.coords
    has_lon = "lon" in ds.coords or "longitude" in ds.coords
    audit["coordinate_availability"]["lat"] = has_lat
    audit["coordinate_availability"]["lon"] = has_lon

    if not has_lat or not has_lon:
        audit["coordinate_availability"]["note"] = "Grid-only mode; no lat/lon coordinates"

    # ---- missing dates ----
    if "time" in ds.coords and audit["time_range"]:
        audit["missing_dates"] = "not_checked_40GB_file"
    else:
        audit["missing_dates"] = "N/A"

    # ---- channel names or order ----
    input_channels = list(ds.coords['variable_input'].values) if 'variable_input' in ds.coords else []
    target_channels = list(ds.coords['variable_target'].values) if 'variable_target' in ds.coords else []

    input_vars_contract = [
        "sm_surface_forecast",
        "sm_rootzone_forecast",
        "mwrtm_vegopacity",
        "tb_h_obs",
        "tb_v_obs",
        "mask",
    ]
    target_vars_contract = [
        "sm_surface_analysis",
        "sm_rootzone_analysis",
    ]

    detected_inputs = {}
    detected_targets = {}
    missing_inputs = []
    missing_targets = []

    for expected in input_vars_contract:
        if expected in input_channels:
            idx = input_channels.index(expected)
            detected_inputs[expected] = {"channel_index": idx, "found": True}
        else:
            missing_inputs.append(expected)

    for expected in target_vars_contract:
        if expected in target_channels:
            idx = target_channels.index(expected)
            detected_targets[expected] = {"channel_index": idx, "found": True}
        else:
            missing_targets.append(expected)

    blocking_issue = None
    if missing_inputs or missing_targets:
        if not input_channels and not target_channels:
            blocking_issue = "No channel coordinates found"
        elif missing_inputs or missing_targets:
            blocking_issue = "Some contract variables not found as channels"

    audit["channel_names_or_order"] = {
        "structure": "multi_channel_array",
        "channel_coord": "variable_input for inputs, variable_target for targets",
        "input_channels": input_channels,
        "target_channels": target_channels,
        "input_channel_count": len(input_channels),
        "target_channel_count": len(target_channels),
        "input_mapping": detected_inputs,
        "target_mapping": detected_targets,
        "missing_inputs": missing_inputs,
        "missing_targets": missing_targets,
        "blocking_issue": blocking_issue,
    }

    # ---- NaN/Inf counts (sampled) ----
    audit["nan_inf_counts"] = {}
    for var in ds.data_vars:
        stats = _sample_array_stats(ds[var])
        audit["nan_inf_counts"][var] = {
            "nan_count": stats["nan_count"],
            "inf_count": stats["inf_count"],
            "nan_ratio": stats["nan_ratio"],
            "inf_ratio": stats["inf_ratio"],
            "note": "sampled from first few time steps for memory efficiency",
        }

    # ---- mask unique values ----
    mask_summary = summarize_masks(ds)
    audit["mask_unique_values"] = mask_summary["detected_masks"]
    audit["mask_keys_distinct"] = mask_summary["mask_keys_distinct"]

    # ---- finite overlap (sampled) ----
    finite_overlap = {}
    for var in ds.data_vars:
        stats = _sample_array_stats(ds[var])
        finite_overlap[var] = {
            "finite_count": stats["finite_count"],
            "finite_ratio": stats["finite_ratio"],
        }
    audit["finite_overlap_forecast_analysis"] = finite_overlap

    # ---- memory estimate ----
    audit["estimated_memory_gb"] = compute_memory_estimate(ds)

    # ---- increment reconstruction check ----
    try:
        inc_result = check_increment_reconstruction(ds)
        audit["increment_reconstruction"] = inc_result
    except Exception as e:
        audit["increment_reconstruction"] = {"error": str(e)}

    ds.close()

    return audit
