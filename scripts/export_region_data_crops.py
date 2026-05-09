#!/usr/bin/env python3
"""
Export US region data crops from DA.nc into per-region-per-year PT files.
Memory-efficient: process one region-year at a time, close DA.nc between reads.

Usage:
    python scripts/export_region_data_crops.py

Output structure:
    artifacts/region_crops/US/
        nc/
            US-R1.nc, ..., US-R6.nc
        pt/
            US-R1/
                region_mask.pt
                metadata.json
                da_full/2015/input.pt, target_increment.pt, target_analysis.pt, loss_mask.pt, metadata.json
                             2016/...
                forecast_only/2015/...
                               2016/...
            US-R2/
                ...
"""

import json
import netCDF4
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import xarray as xr

# DA.nc channel indices
INPUT_IDX_FULL = list(range(12))   # all 12 channels
INPUT_IDX_FORECAST_ONLY = [0, 1, 11]  # surface_fc, rootzone_fc, mask

DA_NC_PATH = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"


def build_year_slices():
    """Build year -> (time_start, time_end) slices based on DA.nc time coordinate."""
    ds = xr.open_dataset(DA_NC_PATH)
    time_vals = ds["time"].values  # datetime64[ns]
    years = {}
    for year in range(2015, 2026):
        if year == 2015:
            t_start_np = np.datetime64(f"{year}-04-01")
            t_end_np = np.datetime64(f"{year}-12-31T23:59:59")
        elif year == 2025:
            t_start_np = np.datetime64(f"{year}-01-01")
            t_end_np = time_vals[-1]
        else:
            t_start_np = np.datetime64(f"{year}-01-01")
            t_end_np = np.datetime64(f"{year}-12-31T23:59:59")

        idx_start = int(np.searchsorted(time_vals, t_start_np))
        idx_end = int(np.searchsorted(time_vals, t_end_np, side="right"))
        idx_start = min(idx_start, len(time_vals))
        idx_end = min(idx_end, len(time_vals))
        years[year] = (idx_start, idx_end)
        print(f"  year {year}: [{idx_start}, {idx_end}] ({idx_end - idx_start} steps)")

    ds.close()
    return years, time_vals


def build_latlon_index_map():
    """Build lon -> x index lookup from geolocation.
    Latitude-to-y mapping is done via actual mask pixel bounds (see compute_crop_bounds).
    Returns lon index functions and the sorted lon array.
    """
    latlon_path = Path("artifacts/geolocation/US_latlon.nc")
    ds = xr.open_dataset(latlon_path)
    lon_2d = ds["longitude"].values
    ds.close()

    lon_unique = np.sort(np.unique(lon_2d[0, :]))  # 640 unique longitudes (ascending)

    def lon_to_x_lower(lon_val):
        """Closed lower bound: find smallest index where lon >= lon_val."""
        idx = int(np.searchsorted(lon_unique, lon_val))
        return min(max(idx, 0), 639)

    def lon_to_x_upper(lon_val):
        """Closed upper bound: find largest index where lon <= lon_val."""
        idx = int(np.searchsorted(lon_unique, lon_val, side="right")) - 1
        return min(max(idx, 0), 639)

    return lon_to_x_lower, lon_to_x_upper, lon_unique


def load_region_mask():
    """Load full region mask tensor."""
    mask_path = Path("artifacts/regions/US_region_mask_tensor.pt")
    return torch.load(mask_path)


def resolve_lon_bounds(lon_to_x_lower, lon_to_x_upper, bbox_spec, lon_unique):
    """Map lon bbox to x index range (closed interval). Latitude bounds come from mask pixels."""
    lon_min, lon_max = bbox_spec["lon_min"], bbox_spec["lon_max"]
    x0 = lon_to_x_lower(lon_min)
    x1 = lon_to_x_upper(lon_max)
    resolved_lon_min = float(lon_unique[x0])
    resolved_lon_max = float(lon_unique[min(x1, 639)])
    return x0, x1, resolved_lon_min, resolved_lon_max


def compute_and_save_year(
    rid, regime, numeric_id, bbox_spec,
    y0, y1, x0, x1, crop_h, crop_w,
    rlat_min, rlat_max, rlon_min, rlon_max,
    year, t_start, t_end,
    data_input, data_target,
    region_crop, coverage_ratio,
    pt_dir
):
    """Process one year of pre-loaded data, compute derived arrays, save PT files. Returns dict."""
    # data_input and data_target are already sliced to the year from the caller
    inp_yr = data_input
    tgt_yr = data_target

    T = t_end - t_start

    # Increment = analysis - forecast (surface + rootzone only)
    target_increment = np.stack([
        tgt_yr[:, 0, :, :] - inp_yr[:, 0, :, :],   # surface
        tgt_yr[:, 1, :, :] - inp_yr[:, 1, :, :],   # rootzone
    ], axis=1).astype(np.float32)

    # Target analysis (surface + rootzone)
    target_analysis = tgt_yr[:, :2, :, :].astype(np.float32)

    # Loss mask: valid ∩ region ∩ finite(forecast) ∩ finite(analysis)
    base_valid = inp_yr[:, 11, :, :] > 0.5
    forecast_finite = np.isfinite(inp_yr[:, 0, :, :]) & np.isfinite(inp_yr[:, 1, :, :])
    analysis_finite = np.isfinite(tgt_yr[:, 0, :, :]) & np.isfinite(tgt_yr[:, 1, :, :])
    region_mask_bool = (region_crop == numeric_id)[np.newaxis, :, :]  # explicit bool conversion
    loss_mask = (base_valid & region_mask_bool & forecast_finite & analysis_finite).astype(np.bool_)

    # forecast_only input: channels [0, 1, 11]
    input_forecast_only = inp_yr[:, INPUT_IDX_FORECAST_ONLY, :, :].astype(np.float32)

    year_results = {}
    for subset, inp_arr, inc_arr, an_arr in [
        ("da_full",      inp_yr,                  target_increment, target_analysis),
        ("forecast_only", input_forecast_only,    target_increment, target_analysis),
    ]:
        year_pt_dir = pt_dir / subset / str(year)
        year_pt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(torch.from_numpy(inp_arr),        year_pt_dir / "input.pt")
        torch.save(torch.from_numpy(inc_arr),        year_pt_dir / "target_increment.pt")
        torch.save(torch.from_numpy(an_arr),        year_pt_dir / "target_analysis.pt")
        torch.save(torch.from_numpy(loss_mask),      year_pt_dir / "loss_mask.pt")

        year_meta = {
            "region_id": rid,
            "regime": regime,
            "year": year,
            "time_steps": int(T),
            "time_slice": [int(t_start), int(t_end)],
            "requested_latlon_bbox": {
                "lat_min": float(bbox_spec["lat_min"]), "lat_max": float(bbox_spec["lat_max"]),
                "lon_min": float(bbox_spec["lon_min"]), "lon_max": float(bbox_spec["lon_max"]),
            },
            "resolved_index_bbox": {
                "time_start": int(t_start), "time_end": int(t_end),
                "y_start": int(y0), "y_end": int(y1),
                "x_start": int(x0), "x_end": int(x1),
            },
            "resolved_latlon_extent": {
                "lat_min": float(rlat_min), "lat_max": float(rlat_max),
                "lon_min": float(rlon_min), "lon_max": float(rlon_max),
            },
            "crop_shape": [int(crop_h), int(crop_w)],
            "region_coverage_ratio": round(float(coverage_ratio), 4),
            "warnings": [],
        }
        with open(year_pt_dir / "metadata.json", "w") as f:
            json.dump(year_meta, f, indent=2)

        year_results[subset] = {"time_steps": int(T)}

    return year_results


def save_region_nc(rid, numeric_id, region_crop, crop_h, crop_w,
                   y0, y1, x0, x1, year_slices, out_root):
    """Save region NC file by reading and writing in chunks to avoid OOM."""
    input_names = [
        "sm_surface_forecast", "sm_rootzone_forecast",
        "soil_temp_layer1_forecast", "surface_temp_forecast",
        "mwrtm_vegopacity", "tb_h_obs", "tb_v_obs",
        "tb_h_obs_errstd", "tb_v_obs_errstd",
        "tb_h_obs_assim", "tb_v_obs_assim", "mask",
    ]
    target_names = [
        "sm_surface_analysis", "sm_rootzone_analysis",
        "soil_temp_layer1_analysis", "surface_temp_analysis",
    ]

    # Determine time range
    t_all_start = min(v[0] for v in year_slices.values())
    t_all_end = max(v[1] for v in year_slices.values())
    total_T = t_all_end - t_all_start

    print(f"  Reading {total_T} timesteps for NC with direct hyperslab...")
    # Direct netCDF4 hyperslab — reads only needed data efficiently
    nc_read = netCDF4.Dataset(DA_NC_PATH, 'r')
    time_var = nc_read.variables['time']
    input_arr = nc_read.variables['input'][t_all_start:t_all_end, :, y0:y1+1, x0:x1+1].astype(np.float32)
    target_arr = nc_read.variables['target'][t_all_start:t_all_end, :, y0:y1+1, x0:x1+1].astype(np.float32)
    # Convert netCDF4 time values to datetime64
    time_values = time_var[t_all_start:t_all_end]
    time_units = time_var.units
    nc_read.close()
    del nc_read
    # Convert float seconds to datetime64 using netCDF4 num2date then astype
    dt_objects = netCDF4.num2date(time_values, time_units)
    nc_time = np.asarray(dt_objects).astype('datetime64[ns]')
    # Transpose to [T, H, W, C] for per-channel data_vars
    input_arr = np.moveaxis(input_arr, 1, -1)   # [T, C, H, W] -> [T, H, W, C]
    target_arr = np.moveaxis(target_arr, 1, -1)  # [T, C, H, W] -> [T, H, W, C]
    print(f"  Read input: {input_arr.shape}, target: {target_arr.shape}")

    # Build xarray Dataset with per-channel data_vars
    data_vars = {}
    for i, ch_name in enumerate(input_names):
        data_vars[f"input_{ch_name}"] = (["time", "y", "x"], input_arr[:, :, :, i])
    for i, ch_name in enumerate(target_names):
        data_vars[f"target_{ch_name}"] = (["time", "y", "x"], target_arr[:, :, :, i])
    data_vars["region_id_int"] = (["y", "x"], (region_crop == numeric_id).astype(np.float32))

    nc_ds = xr.Dataset(
        data_vars,
        coords={"time": nc_time, "y": np.arange(crop_h), "x": np.arange(crop_w)},
    )

    nc_path = out_root / "nc" / f"{rid}.nc"
    # Write to temp file then rename to avoid blocking on large files
    tmp_path = out_root / "nc" / f"{rid}.nc.tmp"
    nc_ds.to_netcdf(tmp_path)
    shutil.move(tmp_path, nc_path)
    nc_ds.close()
    del input_arr, target_arr, data_vars, nc_ds

    print(f"  NC saved: {nc_path}")


def export_region_crops():
    project_root = Path(__file__).parent.parent
    out_root = project_root / "artifacts/region_crops" / "US"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "nc").mkdir(parents=True, exist_ok=True)

    # Load actual bbox from US_region_stats.json (NOT from specs/regions_v2.yaml)
    # These bbox values are the ones actually used when generating the masks.
    stats_path = project_root / "artifacts/regions" / "US_region_stats.json"
    with open(stats_path) as f:
        region_stats = json.load(f)

    # Build lon index map and load lat grid
    print("Building lon index map...")
    lon_to_x_lower, lon_to_x_upper, lon_unique = build_latlon_index_map()

    # Load lat grid for computing lat bounds
    latlon_path = Path("artifacts/geolocation/US_latlon.nc")
    ds_latlon = xr.open_dataset(latlon_path)
    lat_2d = ds_latlon["latitude"].values
    ds_latlon.close()

    print("Building year time slices...")
    year_slices, time_vals = build_year_slices()

    # Load full region mask once
    full_mask = load_region_mask().numpy()

    all_region_results = []

    # Region definitions from US_region_stats.json keys
    us_region_order = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
    regime_map = {
        "US-R1": "dryland_sparse_vegetation",
        "US-R2": "semi_arid_transition",
        "US-R3": "irrigated_managed_agriculture",
        "US-R4": "rainfed_agriculture",
        "US-R5": "humid_high_vegetation",
        "US-R6": "mountain_cold_terrain_stress",
    }
    numeric_id_map = {
        "US-R1": 1, "US-R2": 2, "US-R3": 3,
        "US-R4": 4, "US-R5": 5, "US-R6": 6,
    }

    for rid in us_region_order:
        stats = region_stats[rid]
        bbox_spec = stats["bbox_spec"]  # closed-interval bbox from mask generation
        regime = regime_map[rid]
        numeric_id = numeric_id_map[rid]

        print(f"\n{'='*60}")
        print(f"Processing {rid}: {stats['region_name']}")
        print(f"  bbox (closed): lat[{bbox_spec['lat_min']}, {bbox_spec['lat_max']}], lon[{bbox_spec['lon_min']}, {bbox_spec['lon_max']}]")
        sys.stdout.flush()

        # Compute x bounds from lon bbox (this works correctly)
        x0, x1, rlon_min, rlon_max = resolve_lon_bounds(
            lon_to_x_lower, lon_to_x_upper, bbox_spec, lon_unique
        )

        # Compute y bounds directly from mask pixel positions (most reliable)
        region_mask_bool = full_mask == numeric_id
        y_indices, x_indices = np.where(region_mask_bool)
        if len(y_indices) == 0:
            print(f"  ERROR: No pixels found for {rid} in mask!")
            continue
        y0_mask, y1_mask = int(y_indices.min()), int(y_indices.max())
        x0_mask, x1_mask = int(x_indices.min()), int(x_indices.max())

        # Use the mask-derived y bounds; use lon-derived x bounds
        y0, y1 = y0_mask, y1_mask
        crop_h = y1 - y0 + 1  # inclusive range
        crop_w = x1 - x0 + 1

        # Get actual lat/lon at crop boundaries from grid
        rlat_min = float(lat_2d[y1, 0])  # y1 is south (lower lat)
        rlat_max = float(lat_2d[y0, 0])  # y0 is north (higher lat)
        print(f"  mask y=[{y0}:{y1}]({crop_h}), x=[{x0}:{x1}]({crop_w})")
        print(f"  mask lat: [{rlat_min:.4f}, {rlat_max:.4f}], lon: [{rlon_min:.4f}, {rlon_max:.4f}]")
        sys.stdout.flush()

        # Crop region mask and compute coverage
        region_crop = full_mask[y0:y1+1, x0:x1+1].copy()
        region_pixels = int(np.sum(region_crop == numeric_id))
        total_pixels = crop_h * crop_w
        coverage_ratio = region_pixels / total_pixels
        print(f"  region_pixels={region_pixels}, total={total_pixels}, coverage={coverage_ratio:.4f}")
        sys.stdout.flush()

        # PT output directory
        pt_dir = out_root / "pt" / rid
        pt_dir.mkdir(parents=True, exist_ok=True)

        # Save region_mask.pt
        torch.save(torch.from_numpy(region_crop.copy()).to(torch.uint8), pt_dir / "region_mask.pt")
        print(f"  region_mask.pt saved")
        sys.stdout.flush()

        # Idempotency: check which year directories already have complete data
        def year_complete(subset, year):
            ydir = pt_dir / subset / str(year)
            if not ydir.exists():
                return False
            for fname in ["input.pt", "target_increment.pt", "target_analysis.pt", "loss_mask.pt"]:
                if not (ydir / fname).exists():
                    return False
            return True

        # Process each year: open, read slice, compute, write, close
        year_results = {}
        for year, (t_start, t_end) in sorted(year_slices.items()):
            if t_end <= t_start:
                print(f"  Year {year}: no data, skipping")
                sys.stdout.flush()
                continue

            # Idempotency: skip if both subsets already complete
            full_done = year_complete("da_full", year)
            fc_done = year_complete("forecast_only", year)
            if full_done and fc_done:
                print(f"  Year {year}: already complete (idempotent skip)")
                T = t_end - t_start
                year_results[year] = {"da_full": {"time_steps": T}, "forecast_only": {"time_steps": T}}
                sys.stdout.flush()
                continue

            print(f"  Year {year}: time [{t_start}:{t_end}] ({t_end - t_start} steps)")
            sys.stdout.flush()
            # Direct netCDF4 hyperslab read — MUCH faster than xarray with chunked storage
            nc_ds = netCDF4.Dataset(DA_NC_PATH, 'r')
            inp_yr = nc_ds.variables['input'][t_start:t_end, :, y0:y1+1, x0:x1+1].astype(np.float32)
            tgt_yr = nc_ds.variables['target'][t_start:t_end, :, y0:y1+1, x0:x1+1].astype(np.float32)
            nc_ds.close()
            del nc_ds
            sys.stdout.flush()
            yr = compute_and_save_year(
                rid, regime, numeric_id, bbox_spec,
                y0, y1, x0, x1, crop_h, crop_w,
                rlat_min, rlat_max, rlon_min, rlon_max,
                year, t_start, t_end,
                inp_yr, tgt_yr,
                region_crop, coverage_ratio,
                pt_dir
            )
            del inp_yr, tgt_yr
            year_results[year] = yr
            print(f"  Year {year}: done")
            sys.stdout.flush()
        print(f"  Writing metadata.json...")
        sys.stdout.flush()
        region_meta = {
            "region_id": rid,
            "regime": regime,
            "numeric_id": numeric_id,
            "years": {str(y): {"time_steps": year_results[y]["da_full"]["time_steps"]} for y in sorted(year_results.keys())},
            "requested_latlon_bbox": {
                "lat_min": float(bbox_spec["lat_min"]), "lat_max": float(bbox_spec["lat_max"]),
                "lon_min": float(bbox_spec["lon_min"]), "lon_max": float(bbox_spec["lon_max"]),
            },
            "resolved_index_bbox": {
                "y_start": int(y0), "y_end": int(y1),
                "x_start": int(x0), "x_end": int(x1),
            },
            "resolved_latlon_extent": {
                "lat_min": float(rlat_min), "lat_max": float(rlat_max),
                "lon_min": float(rlon_min), "lon_max": float(rlon_max),
            },
            "crop_shape": [int(crop_h), int(crop_w)],
            "region_coverage_ratio": round(float(coverage_ratio), 4),
            "warnings": [],
        }
        with open(pt_dir / "metadata.json", "w") as f:
            json.dump(region_meta, f, indent=2)

        # Save region NC file (skip if already exists and is valid size)
        nc_path = out_root / "nc" / f"{rid}.nc"
        if nc_path.exists() and nc_path.stat().st_size > 1000:
            print(f"  NC file already exists, skipping: {nc_path}")
        else:
            print(f"  Saving NC file...")
            sys.stdout.flush()
            save_region_nc(rid, numeric_id, region_crop, crop_h, crop_w,
                          y0, y1, x0, x1, year_slices, out_root)
            print(f"  NC save complete")
        sys.stdout.flush()

        all_region_results.append({
            "region_id": rid,
            "nc_path": f"nc/{rid}.nc",
            "pt_dir": f"pt/{rid}",
            "coverage_ratio": round(float(coverage_ratio), 4),
            "crop_shape": [int(crop_h), int(crop_w)],
        })

        del region_crop

    # Write top-level manifest
    manifest = {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "country": "US",
        "regions": all_region_results,
        "years": {str(y): {"time_start": v[0], "time_end": v[1]} for y, v in sorted(year_slices.items())},
        "input_channel_subset": {
            "da_full": "all 12 channels (idx 0-11)",
            "forecast_only": "channels [0,1,11] — sm_surface_fc, sm_rootzone_fc, mask",
        },
        "target_channel_subset": {
            "target_increment": "2 channels — delta_SM_surface, delta_SM_rootzone",
            "target_analysis": "2 channels — sm_surface_analysis, sm_rootzone_analysis",
        },
    }
    manifest_path = out_root / "manifest_region_crops_US.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(export_region_crops())