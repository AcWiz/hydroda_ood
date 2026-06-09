#!/usr/bin/env python3
"""Audit region tensor cache against the original DA.nc file."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import netCDF4
import numpy as np
import torch
import xarray as xr

from scripts.data.export_region_tensor_cache import (
    DEFAULT_DA_NC,
    DEFAULT_OUT_DIR,
    DEFAULT_REGION_MASKS_NC,
    DEFAULT_REGIONS,
    DEFAULT_YEARS,
    MASK_CONTRACT,
    _crop_bounds,
    _region_numeric_id,
)


DEFAULT_JSON = "artifacts/region_crops/US/audit_tensor_cache_vs_nc.json"
DEFAULT_MD = "reports/audits/tensor_cache_vs_nc_US.md"


def _load_region_mask(region_masks_nc: Path) -> np.ndarray:
    ds = xr.open_dataset(region_masks_nc)
    try:
        return ds["region_mask_integer"].values.astype(np.int16)
    finally:
        ds.close()


def _sample_indices(n: int, samples_per_year: int) -> List[int]:
    if n <= 0 or samples_per_year <= 0:
        return []
    if n <= samples_per_year:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, samples_per_year)})


def _load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a - b)
    if diff.size == 0 or np.all(np.isnan(diff)):
        return 0.0
    return float(np.nanmax(diff))


def _arrays_close(a: np.ndarray, b: np.ndarray, atol: float) -> bool:
    return bool(np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True))


def _record_failure(
    failures: List[Dict[str, Any]],
    *,
    region_id: str,
    year: int,
    local_time_index: int | None,
    check: str,
    message: str,
    max_abs_diff: float | None = None,
) -> None:
    item: Dict[str, Any] = {
        "region_id": region_id,
        "year": int(year),
        "local_time_index": local_time_index,
        "check": check,
        "message": message,
    }
    if max_abs_diff is not None:
        item["max_abs_diff"] = float(max_abs_diff)
    failures.append(item)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_year(
    *,
    ds: netCDF4.Dataset,
    cache_dir: Path,
    region_id: str,
    numeric_id: int,
    year: int,
    bounds: Mapping[str, int],
    samples_per_year: int,
    exhaustive: bool,
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    y0 = int(bounds["y_start"])
    y1 = int(bounds["y_end"])
    x0 = int(bounds["x_start"])
    x1 = int(bounds["x_end"])
    crop_h = y1 - y0 + 1
    crop_w = x1 - x0 + 1
    year_dir = cache_dir / "pt" / region_id / "da_full" / str(year)
    meta = _read_json(year_dir / "metadata.json")
    if not meta:
        _record_failure(
            failures,
            region_id=region_id,
            year=year,
            local_time_index=None,
            check="metadata",
            message=f"missing metadata.json at {year_dir}",
        )
        return {"status": "missing_metadata", "sampled_indices": []}

    if meta.get("mask_contract") != MASK_CONTRACT:
        _record_failure(
            failures,
            region_id=region_id,
            year=year,
            local_time_index=None,
            check="metadata.mask_contract",
            message=f"unexpected mask_contract={meta.get('mask_contract')!r}",
        )
    if meta.get("resolved_index_bbox") != dict(bounds):
        _record_failure(
            failures,
            region_id=region_id,
            year=year,
            local_time_index=None,
            check="metadata.resolved_index_bbox",
            message="metadata bbox does not match region mask bbox",
        )
    if meta.get("crop_shape") != [crop_h, crop_w]:
        _record_failure(
            failures,
            region_id=region_id,
            year=year,
            local_time_index=None,
            check="metadata.crop_shape",
            message=f"metadata crop_shape={meta.get('crop_shape')!r}, expected {[crop_h, crop_w]}",
        )

    try:
        input_t = _load_tensor(year_dir / "input.pt")
        inc_t = _load_tensor(year_dir / "target_increment.pt")
        analysis_t = _load_tensor(year_dir / "target_analysis.pt")
        mask_t = _load_tensor(year_dir / "loss_mask.pt")
    except Exception as exc:
        _record_failure(
            failures,
            region_id=region_id,
            year=year,
            local_time_index=None,
            check="tensor_load",
            message=str(exc),
        )
        return {"status": "tensor_load_failed", "sampled_indices": []}

    time_slice = meta.get("time_slice", [0, 0])
    t_start, t_end = int(time_slice[0]), int(time_slice[1])
    expected_t = int(t_end - t_start)
    expected_shapes = {
        "input": (expected_t, 12, crop_h, crop_w),
        "target_increment": (expected_t, 2, crop_h, crop_w),
        "target_analysis": (expected_t, 2, crop_h, crop_w),
        "loss_mask": (expected_t, crop_h, crop_w),
    }
    actual_shapes = {
        "input": tuple(input_t.shape),
        "target_increment": tuple(inc_t.shape),
        "target_analysis": tuple(analysis_t.shape),
        "loss_mask": tuple(mask_t.shape),
    }
    for key, expected in expected_shapes.items():
        if actual_shapes[key] != expected:
            _record_failure(
                failures,
                region_id=region_id,
                year=year,
                local_time_index=None,
                check=f"{key}.shape",
                message=f"actual={actual_shapes[key]}, expected={expected}",
            )
    if input_t.dtype != torch.float32:
        _record_failure(failures, region_id=region_id, year=year, local_time_index=None, check="input.dtype", message=str(input_t.dtype))
    if inc_t.dtype != torch.float32:
        _record_failure(failures, region_id=region_id, year=year, local_time_index=None, check="target_increment.dtype", message=str(inc_t.dtype))
    if analysis_t.dtype != torch.float32:
        _record_failure(failures, region_id=region_id, year=year, local_time_index=None, check="target_analysis.dtype", message=str(analysis_t.dtype))
    if mask_t.dtype != torch.bool:
        _record_failure(failures, region_id=region_id, year=year, local_time_index=None, check="loss_mask.dtype", message=str(mask_t.dtype))

    sampled = _sample_indices(expected_t, samples_per_year)
    region_crop = _load_tensor(cache_dir / "pt" / region_id / "region_mask.pt").numpy()
    region_bool = region_crop == numeric_id
    if exhaustive and expected_t > 0:
        nc_input_all = ds.variables["input"][t_start:t_end, :, y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
        nc_target_all = ds.variables["target"][t_start:t_end, :, y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
        expected_analysis_all = nc_target_all[:, :2].astype(np.float32)
        expected_inc_all = np.stack(
            [
                nc_target_all[:, 0] - nc_input_all[:, 0],
                nc_target_all[:, 1] - nc_input_all[:, 1],
            ],
            axis=1,
        ).astype(np.float32)
        expected_mask_all = (
            region_bool[np.newaxis, :, :]
            & np.isfinite(nc_input_all[:, 0])
            & np.isfinite(nc_input_all[:, 1])
            & np.isfinite(nc_target_all[:, 0])
            & np.isfinite(nc_target_all[:, 1])
            & np.isfinite(expected_inc_all[:, 0])
            & np.isfinite(expected_inc_all[:, 1])
        )
        full_checks = [
            ("input", input_t.numpy(), nc_input_all, 0.0),
            ("target_analysis", analysis_t.numpy(), expected_analysis_all, 0.0),
            ("target_increment", inc_t.numpy(), expected_inc_all, 1e-6),
        ]
        for name, actual, expected, atol in full_checks:
            if not _arrays_close(actual, expected, atol=atol):
                _record_failure(
                    failures,
                    region_id=region_id,
                    year=year,
                    local_time_index=None,
                    check=name,
                    message=f"{name} exhaustive mismatch against NC for full year crop",
                    max_abs_diff=_max_abs_diff(actual, expected),
                )
        actual_mask_all = mask_t.numpy()
        if not np.array_equal(actual_mask_all, expected_mask_all):
            _record_failure(
                failures,
                region_id=region_id,
                year=year,
                local_time_index=None,
                check="loss_mask",
                message="loss_mask exhaustive mismatch against NC for full year crop",
                max_abs_diff=float(np.count_nonzero(actual_mask_all != expected_mask_all)),
            )
        return {
            "status": "audited",
            "mode": "exhaustive",
            "time_steps": expected_t,
            "sampled_indices": sampled,
            "n_checked_time_steps": expected_t,
            "actual_shapes": {k: list(v) for k, v in actual_shapes.items()},
        }

    for local_idx in sampled:
        global_idx = t_start + local_idx
        nc_input = ds.variables["input"][global_idx, :, y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
        nc_target = ds.variables["target"][global_idx, :, y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
        expected_analysis = nc_target[:2].astype(np.float32)
        expected_inc = np.stack(
            [
                nc_target[0] - nc_input[0],
                nc_target[1] - nc_input[1],
            ],
            axis=0,
        ).astype(np.float32)
        expected_mask = (
            region_bool
            & np.isfinite(nc_input[0])
            & np.isfinite(nc_input[1])
            & np.isfinite(nc_target[0])
            & np.isfinite(nc_target[1])
            & np.isfinite(expected_inc[0])
            & np.isfinite(expected_inc[1])
        )

        checks = [
            ("input", input_t[local_idx].numpy(), nc_input, 0.0),
            ("target_analysis", analysis_t[local_idx].numpy(), expected_analysis, 0.0),
            ("target_increment", inc_t[local_idx].numpy(), expected_inc, 1e-6),
        ]
        for name, actual, expected, atol in checks:
            if not _arrays_close(actual, expected, atol=atol):
                _record_failure(
                    failures,
                    region_id=region_id,
                    year=year,
                    local_time_index=local_idx,
                    check=name,
                    message=f"{name} mismatch against NC at global time_index={global_idx}",
                    max_abs_diff=_max_abs_diff(actual, expected),
                )
        actual_mask = mask_t[local_idx].numpy()
        if not np.array_equal(actual_mask, expected_mask):
            _record_failure(
                failures,
                region_id=region_id,
                year=year,
                local_time_index=local_idx,
                check="loss_mask",
                message=f"loss_mask mismatch against NC at global time_index={global_idx}",
                max_abs_diff=float(np.count_nonzero(actual_mask != expected_mask)),
            )

    return {
        "status": "audited",
        "mode": "sampled",
        "time_steps": expected_t,
        "sampled_indices": sampled,
        "n_checked_time_steps": len(sampled),
        "actual_shapes": {k: list(v) for k, v in actual_shapes.items()},
    }


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tensor Cache vs NC Audit",
        "",
        f"- all_passed: {report['all_passed']}",
        f"- n_failures: {report['n_failures']}",
        f"- regions: {', '.join(report['regions'])}",
        f"- years: {', '.join(str(y) for y in report['years'])}",
        f"- samples_per_year: {report['samples_per_year']}",
        f"- exhaustive: {report['exhaustive']}",
        f"- cache_dir: `{report['cache_dir']}`",
        f"- da_nc: `{report['da_nc']}`",
        "",
    ]
    if report["failures"]:
        lines.append("## Failures")
        for item in report["failures"][:200]:
            lines.append(
                f"- {item['region_id']} {item['year']} {item['check']} "
                f"local={item['local_time_index']}: {item['message']}"
            )
        if len(report["failures"]) > 200:
            lines.append(f"- ... truncated {len(report['failures']) - 200} additional failures")
    else:
        lines.append("No mismatches found in audited samples.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_region_tensor_cache(
    *,
    cache_dir: str | Path = DEFAULT_OUT_DIR,
    da_nc: str | Path = DEFAULT_DA_NC,
    region_masks_nc: str | Path = DEFAULT_REGION_MASKS_NC,
    regions: Sequence[str] = DEFAULT_REGIONS,
    years: Sequence[int] = DEFAULT_YEARS,
    samples_per_year: int = 3,
    exhaustive: bool = False,
    out_json: str | Path | None = None,
    out_md: str | Path | None = None,
) -> Dict[str, Any]:
    cache_dir = Path(cache_dir)
    da_nc = Path(da_nc)
    region_masks_nc = Path(region_masks_nc)
    regions = list(regions)
    years = [int(y) for y in years]
    full_region_mask = _load_region_mask(region_masks_nc)
    failures: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {}

    manifest_path = cache_dir / "manifest_region_crops_US.json"
    if not manifest_path.exists():
        _record_failure(
            failures,
            region_id="ALL",
            year=-1,
            local_time_index=None,
            check="manifest",
            message=f"missing manifest: {manifest_path}",
        )

    with netCDF4.Dataset(da_nc, "r") as ds:
        for region_id in regions:
            numeric_id = _region_numeric_id(region_id)
            bounds = _crop_bounds(full_region_mask, numeric_id)
            region_result: Dict[str, Any] = {}
            for year in years:
                region_result[str(year)] = _audit_year(
                    ds=ds,
                    cache_dir=cache_dir,
                    region_id=region_id,
                    numeric_id=numeric_id,
                    year=year,
                    bounds=bounds,
                    samples_per_year=samples_per_year,
                    exhaustive=exhaustive,
                    failures=failures,
                )
            results[region_id] = region_result

    report = {
        "audit_type": "tensor_cache_vs_nc",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "da_nc": str(da_nc),
        "region_masks_nc": str(region_masks_nc),
        "mask_contract": MASK_CONTRACT,
        "regions": regions,
        "years": years,
        "samples_per_year": int(samples_per_year),
        "exhaustive": bool(exhaustive),
        "all_passed": len(failures) == 0,
        "n_failures": len(failures),
        "failures": failures,
        "results": results,
    }
    if out_json is not None:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if out_md is not None:
        _write_markdown(report, Path(out_md))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit per-region PT tensor cache against DA.nc")
    parser.add_argument("--cache_dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--da_nc", type=Path, default=Path(DEFAULT_DA_NC))
    parser.add_argument("--region_masks_nc", type=Path, default=Path(DEFAULT_REGION_MASKS_NC))
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--samples_per_year", type=int, default=3)
    parser.add_argument("--exhaustive", action="store_true", help="Compare every time step in every region-year crop.")
    parser.add_argument("--out_json", type=Path, default=Path(DEFAULT_JSON))
    parser.add_argument("--out_md", type=Path, default=Path(DEFAULT_MD))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = audit_region_tensor_cache(
        cache_dir=args.cache_dir,
        da_nc=args.da_nc,
        region_masks_nc=args.region_masks_nc,
        regions=args.regions,
        years=args.years,
        samples_per_year=args.samples_per_year,
        exhaustive=args.exhaustive,
        out_json=args.out_json,
        out_md=args.out_md,
    )
    print(
        f"tensor cache audit all_passed={report['all_passed']} "
        f"n_failures={report['n_failures']} out_json={args.out_json}"
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
