#!/usr/bin/env python3
"""Export region/year tensor caches from DA.nc.

The output is intentionally compatible with the current HydroDADataset sample
contract: target increment is analysis minus forecast, and loss_mask is active
region pixels with finite forecast/analysis values. The base-valid diagnostic
channel is saved in input but is not used to gate loss_mask.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import netCDF4
import numpy as np
import torch
import xarray as xr


DEFAULT_DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
DEFAULT_REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
DEFAULT_OUT_DIR = "artifacts/region_crops/US"
DEFAULT_REGIONS = [f"US-R{i}" for i in range(1, 7)]
DEFAULT_YEARS = list(range(2015, 2026))
MASK_CONTRACT = "region_and_finite_forecast_analysis_no_base_valid_gate"
SCRIPT_VERSION = "1.0"


def _region_numeric_id(region_id: str) -> int:
    try:
        prefix, value = region_id.split("-R", maxsplit=1)
        if prefix != "US":
            raise ValueError
        numeric_id = int(value)
    except Exception as exc:
        raise ValueError(f"unsupported region id {region_id!r}; expected form US-R<N>") from exc
    if numeric_id <= 0:
        raise ValueError(f"unsupported region id {region_id!r}; numeric id must be positive")
    return numeric_id


def _load_region_mask(region_masks_nc: Path) -> np.ndarray:
    ds = xr.open_dataset(region_masks_nc)
    try:
        if "region_mask_integer" not in ds:
            raise KeyError(f"{region_masks_nc} does not contain variable 'region_mask_integer'")
        return ds["region_mask_integer"].values.astype(np.int16)
    finally:
        ds.close()


def _load_latitude(region_masks_nc: Path) -> np.ndarray | None:
    ds = xr.open_dataset(region_masks_nc)
    try:
        if "latitude" not in ds:
            return None
        return ds["latitude"].values.astype(np.float32)
    finally:
        ds.close()


def _crop_bounds(mask: np.ndarray, numeric_id: int) -> Dict[str, int]:
    y_indices, x_indices = np.where(mask == numeric_id)
    if len(y_indices) == 0:
        raise ValueError(f"region numeric id {numeric_id} has no pixels in region mask")
    return {
        "y_start": int(y_indices.min()),
        "y_end": int(y_indices.max()),
        "x_start": int(x_indices.min()),
        "x_end": int(x_indices.max()),
    }


def _date_years(ds: netCDF4.Dataset) -> np.ndarray:
    time_var = ds.variables["time"]
    values = np.asarray(time_var[:])
    units = getattr(time_var, "units", None)
    if units is None:
        raise ValueError("DA.nc time variable must define units for year slicing")
    calendar = getattr(time_var, "calendar", "standard")
    dates = netCDF4.num2date(
        values,
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=False,
    )
    return np.asarray([int(d.year) for d in dates], dtype=np.int16)


def _year_slices(years_by_time: np.ndarray, years: Sequence[int]) -> Dict[int, Dict[str, int]]:
    result: Dict[int, Dict[str, int]] = {}
    for year in years:
        indices = np.where(years_by_time == int(year))[0]
        if len(indices) == 0:
            result[int(year)] = {"time_start": 0, "time_end": 0, "time_steps": 0}
            continue
        start = int(indices.min())
        end = int(indices.max()) + 1
        if not np.array_equal(indices, np.arange(start, end)):
            raise ValueError(f"time indices for year {year} are not contiguous")
        result[int(year)] = {"time_start": start, "time_end": end, "time_steps": end - start}
    return result


def _var_summary(var: netCDF4.Variable) -> Dict[str, Any]:
    try:
        chunking = var.chunking()
    except Exception:
        chunking = None
    return {
        "dimensions": list(var.dimensions),
        "shape": [int(x) for x in var.shape],
        "dtype": str(var.dtype),
        "chunking": chunking,
    }


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_json_write(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _read_year_metadata(year_dir: Path) -> Dict[str, Any]:
    meta_path = year_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _tensor_matches(path: Path, expected_shape: Sequence[int], expected_dtype: torch.dtype) -> bool:
    if not path.exists():
        return False
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    return tuple(tensor.shape) == tuple(expected_shape) and tensor.dtype == expected_dtype


def _complete_year_dir(
    year_dir: Path,
    *,
    region_id: str,
    numeric_id: int,
    year: int,
    time_steps: int,
    crop_h: int,
    crop_w: int,
    bounds: Mapping[str, int],
) -> bool:
    meta = _read_year_metadata(year_dir)
    expected_meta = {
        "region_id": region_id,
        "numeric_id": int(numeric_id),
        "year": int(year),
        "time_steps": int(time_steps),
        "crop_shape": [int(crop_h), int(crop_w)],
        "resolved_index_bbox": dict(bounds),
        "mask_contract": MASK_CONTRACT,
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            return False
    return (
        _tensor_matches(year_dir / "input.pt", (time_steps, 12, crop_h, crop_w), torch.float32)
        and _tensor_matches(year_dir / "target_increment.pt", (time_steps, 2, crop_h, crop_w), torch.float32)
        and _tensor_matches(year_dir / "target_analysis.pt", (time_steps, 2, crop_h, crop_w), torch.float32)
        and _tensor_matches(year_dir / "loss_mask.pt", (time_steps, crop_h, crop_w), torch.bool)
    )


def _save_year_tensors(
    *,
    ds: netCDF4.Dataset,
    year_dir: Path,
    region_id: str,
    numeric_id: int,
    year: int,
    time_slice: Mapping[str, int],
    full_region_mask: np.ndarray,
    latitude: np.ndarray | None,
    bounds: Mapping[str, int],
    da_nc: Path,
    region_masks_nc: Path,
    force: bool,
) -> Dict[str, Any]:
    t_start = int(time_slice["time_start"])
    t_end = int(time_slice["time_end"])
    y0 = int(bounds["y_start"])
    y1 = int(bounds["y_end"])
    x0 = int(bounds["x_start"])
    x1 = int(bounds["x_end"])
    crop_h = y1 - y0 + 1
    crop_w = x1 - x0 + 1
    time_steps = int(t_end - t_start)

    if not force and _complete_year_dir(
        year_dir,
        region_id=region_id,
        numeric_id=numeric_id,
        year=year,
        time_steps=time_steps,
        crop_h=crop_h,
        crop_w=crop_w,
        bounds=bounds,
    ):
        meta = _read_year_metadata(year_dir)
        meta["status"] = "complete_existing"
        return meta

    year_dir.mkdir(parents=True, exist_ok=True)
    region_crop = full_region_mask[y0 : y1 + 1, x0 : x1 + 1]
    _atomic_torch_save(torch.from_numpy(region_crop.astype(np.uint8, copy=True)), year_dir.parent.parent / "region_mask.pt")
    if latitude is not None:
        latitude_crop = latitude[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32, copy=True)
        _atomic_torch_save(torch.from_numpy(latitude_crop), year_dir.parent.parent / "latitude.pt")
        latitude_weight = np.cos(np.deg2rad(latitude_crop.astype(np.float64))).clip(min=0.0).astype(np.float32)
        _atomic_torch_save(torch.from_numpy(latitude_weight), year_dir.parent.parent / "latitude_weight.pt")

    if t_end <= t_start:
        input_arr = np.empty((0, 12, crop_h, crop_w), dtype=np.float32)
        target_arr = np.empty((0, 4, crop_h, crop_w), dtype=np.float32)
    else:
        input_arr = ds.variables["input"][t_start:t_end, :, y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
        target_arr = ds.variables["target"][t_start:t_end, :, y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)

    target_increment = np.stack(
        [
            target_arr[:, 0, :, :] - input_arr[:, 0, :, :],
            target_arr[:, 1, :, :] - input_arr[:, 1, :, :],
        ],
        axis=1,
    ).astype(np.float32)
    target_analysis = target_arr[:, :2, :, :].astype(np.float32)
    region_bool = (region_crop == numeric_id)[np.newaxis, :, :]
    finite_mask = (
        np.isfinite(input_arr[:, 0, :, :])
        & np.isfinite(input_arr[:, 1, :, :])
        & np.isfinite(target_arr[:, 0, :, :])
        & np.isfinite(target_arr[:, 1, :, :])
        & np.isfinite(target_increment[:, 0, :, :])
        & np.isfinite(target_increment[:, 1, :, :])
    )
    loss_mask = (region_bool & finite_mask).astype(np.bool_)

    _atomic_torch_save(torch.from_numpy(input_arr), year_dir / "input.pt")
    _atomic_torch_save(torch.from_numpy(target_increment), year_dir / "target_increment.pt")
    _atomic_torch_save(torch.from_numpy(target_analysis), year_dir / "target_analysis.pt")
    _atomic_torch_save(torch.from_numpy(loss_mask), year_dir / "loss_mask.pt")

    metadata = {
        "script": "scripts/data/export_region_tensor_cache.py",
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "written",
        "region_id": region_id,
        "numeric_id": int(numeric_id),
        "year": int(year),
        "time_steps": int(time_steps),
        "time_slice": [t_start, t_end],
        "da_nc": str(da_nc),
        "region_masks_nc": str(region_masks_nc),
        "subset": "da_full",
        "input_channels": "all 12 input channels",
        "target_increment_channels": ["surface_analysis_minus_forecast", "rootzone_analysis_minus_forecast"],
        "target_analysis_channels": ["surface_analysis", "rootzone_analysis"],
        "mask_contract": MASK_CONTRACT,
        "resolved_index_bbox": dict(bounds),
        "crop_shape": [int(crop_h), int(crop_w)],
        "region_pixel_count": int((region_crop == numeric_id).sum()),
        "region_coverage_ratio": float((region_crop == numeric_id).sum() / max(region_crop.size, 1)),
        "tensor_files": {
            "input": "input.pt",
            "target_increment": "target_increment.pt",
            "target_analysis": "target_analysis.pt",
            "loss_mask": "loss_mask.pt",
        },
    }
    _atomic_json_write(metadata, year_dir / "metadata.json")
    return metadata


def export_region_tensor_cache(
    *,
    da_nc: str | Path = DEFAULT_DA_NC,
    region_masks_nc: str | Path = DEFAULT_REGION_MASKS_NC,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    regions: Sequence[str] = DEFAULT_REGIONS,
    years: Sequence[int] = DEFAULT_YEARS,
    force: bool = False,
) -> Dict[str, Any]:
    da_nc = Path(da_nc)
    region_masks_nc = Path(region_masks_nc)
    out_dir = Path(out_dir)
    if not da_nc.exists():
        raise FileNotFoundError(f"DA.nc not found: {da_nc}")
    if not region_masks_nc.exists():
        raise FileNotFoundError(f"region masks not found: {region_masks_nc}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pt").mkdir(parents=True, exist_ok=True)
    full_region_mask = _load_region_mask(region_masks_nc)
    latitude = _load_latitude(region_masks_nc)
    years = [int(y) for y in years]
    regions = list(regions)

    manifest_regions: List[Dict[str, Any]] = []
    with netCDF4.Dataset(da_nc, "r") as ds:
        for required_var in ("time", "input", "target"):
            if required_var not in ds.variables:
                raise KeyError(f"{da_nc} does not contain variable {required_var!r}")
        years_by_time = _date_years(ds)
        slices = _year_slices(years_by_time, years)
        variable_summary = {
            "input": _var_summary(ds.variables["input"]),
            "target": _var_summary(ds.variables["target"]),
            "time": _var_summary(ds.variables["time"]),
        }

        for region_id in regions:
            numeric_id = _region_numeric_id(region_id)
            bounds = _crop_bounds(full_region_mask, numeric_id)
            region_dir = out_dir / "pt" / region_id
            (region_dir / "da_full").mkdir(parents=True, exist_ok=True)
            region_years: Dict[str, Any] = {}
            for year in years:
                year_dir = region_dir / "da_full" / str(year)
                meta = _save_year_tensors(
                    ds=ds,
                    year_dir=year_dir,
                    region_id=region_id,
                    numeric_id=numeric_id,
                    year=year,
                    time_slice=slices[year],
                    full_region_mask=full_region_mask,
                    latitude=latitude,
                    bounds=bounds,
                    da_nc=da_nc,
                    region_masks_nc=region_masks_nc,
                    force=force,
                )
                region_years[str(year)] = {
                    "time_steps": int(meta.get("time_steps", slices[year]["time_steps"])),
                    "time_slice": meta.get("time_slice", [slices[year]["time_start"], slices[year]["time_end"]]),
                    "status": meta.get("status", "unknown"),
                }
            region_meta = {
                "region_id": region_id,
                "numeric_id": int(numeric_id),
                "pt_dir": f"pt/{region_id}",
                "resolved_index_bbox": dict(bounds),
                "crop_shape": [
                    int(bounds["y_end"] - bounds["y_start"] + 1),
                    int(bounds["x_end"] - bounds["x_start"] + 1),
                ],
                "years": region_years,
            }
            _atomic_json_write(region_meta, region_dir / "metadata.json")
            manifest_regions.append(region_meta)

    manifest = {
        "manifest_version": "1.0",
        "script": "scripts/data/export_region_tensor_cache.py",
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "country": "US",
        "da_nc": str(da_nc),
        "region_masks_nc": str(region_masks_nc),
        "mask_contract": MASK_CONTRACT,
        "variables": variable_summary,
        "years": {str(y): slices[y] for y in years},
        "regions": manifest_regions,
    }
    manifest_path = out_dir / "manifest_region_crops_US.json"
    _atomic_json_write(manifest, manifest_path)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-region per-year PT tensor cache from DA.nc")
    parser.add_argument("--da_nc", type=Path, default=Path(DEFAULT_DA_NC))
    parser.add_argument("--region_masks_nc", type=Path, default=Path(DEFAULT_REGION_MASKS_NC))
    parser.add_argument("--out_dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--force", action="store_true", help="Regenerate existing complete region-year tensor files.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = export_region_tensor_cache(
        da_nc=args.da_nc,
        region_masks_nc=args.region_masks_nc,
        out_dir=args.out_dir,
        regions=args.regions,
        years=args.years,
        force=args.force,
    )
    print(f"tensor cache manifest: {Path(args.out_dir) / 'manifest_region_crops_US.json'}")
    print(f"regions={len(manifest['regions'])} years={len(manifest['years'])} mask_contract={manifest['mask_contract']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
