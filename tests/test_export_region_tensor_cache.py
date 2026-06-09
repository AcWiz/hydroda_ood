import json
from pathlib import Path

import netCDF4
import numpy as np
import torch
import xarray as xr


def _write_tiny_da_nc(path: Path) -> None:
    times = np.array([0.0, 1.0, 366.0], dtype=np.float64)
    input_arr = np.zeros((3, 12, 3, 4), dtype=np.float32)
    target_arr = np.zeros((3, 4, 3, 4), dtype=np.float32)
    for t in range(3):
        for c in range(12):
            input_arr[t, c] = 1000 * t + 100 * c + np.arange(12, dtype=np.float32).reshape(3, 4)
        for c in range(4):
            target_arr[t, c] = 2000 * t + 50 * c + np.arange(12, dtype=np.float32).reshape(3, 4)
    target_arr[:, 0] = input_arr[:, 0] + 10.0
    target_arr[:, 1] = input_arr[:, 1] - 5.0

    # base_valid is intentionally false inside the region. The tensor-cache
    # exporter must follow HydroDADataset's current loss mask contract, which
    # does not gate training/eval labels on this diagnostic channel.
    input_arr[:, 11] = 0.0
    input_arr[0, 0, 0, 1] = np.nan
    target_arr[0, 1, 1, 2] = np.nan

    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 3)
        ds.createDimension("variable_input", 12)
        ds.createDimension("variable_target", 4)
        ds.createDimension("height", 3)
        ds.createDimension("width", 4)

        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2015-01-01 00:00:00"
        time.calendar = "standard"
        time[:] = times
        ds.createVariable("input", "f4", ("time", "variable_input", "height", "width"))[:] = input_arr
        ds.createVariable("target", "f4", ("time", "variable_target", "height", "width"))[:] = target_arr


def _write_tiny_region_masks(path: Path) -> None:
    mask = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int16,
    )
    latitude = np.array(
        [
            [40.0, 40.0, 40.0, 40.0],
            [39.0, 39.0, 39.0, 39.0],
            [38.0, 38.0, 38.0, 38.0],
        ],
        dtype=np.float32,
    )
    xr.Dataset(
        {
            "region_mask_integer": (["height", "width"], mask),
            "latitude": (["height", "width"], latitude),
        }
    ).to_netcdf(path)


def test_export_region_tensor_cache_writes_expected_tensors(tmp_path):
    from scripts.data.export_region_tensor_cache import export_region_tensor_cache

    da_nc = tmp_path / "DA.nc"
    masks_nc = tmp_path / "regions.nc"
    out_dir = tmp_path / "cache"
    _write_tiny_da_nc(da_nc)
    _write_tiny_region_masks(masks_nc)

    manifest = export_region_tensor_cache(
        da_nc=da_nc,
        region_masks_nc=masks_nc,
        out_dir=out_dir,
        regions=["US-R1"],
        years=[2015],
    )

    year_dir = out_dir / "pt" / "US-R1" / "da_full" / "2015"
    assert (year_dir / "input.pt").exists()
    assert (year_dir / "target_increment.pt").exists()
    assert (year_dir / "target_analysis.pt").exists()
    assert (year_dir / "loss_mask.pt").exists()
    assert (year_dir / "metadata.json").exists()
    assert (out_dir / "manifest_region_crops_US.json").exists()

    x = torch.load(year_dir / "input.pt", weights_only=True)
    inc = torch.load(year_dir / "target_increment.pt", weights_only=True)
    analysis = torch.load(year_dir / "target_analysis.pt", weights_only=True)
    loss_mask = torch.load(year_dir / "loss_mask.pt", weights_only=True)

    assert x.shape == (2, 12, 2, 2)
    assert inc.shape == (2, 2, 2, 2)
    assert analysis.shape == (2, 2, 2, 2)
    assert loss_mask.dtype == torch.bool
    torch.testing.assert_close(inc[1, 0], torch.full((2, 2), 10.0))
    torch.testing.assert_close(inc[1, 1], torch.full((2, 2), -5.0))

    # First pixel has NaN forecast surface, last pixel has NaN analysis rootzone.
    # Other region pixels remain valid even though base_valid channel is zero.
    expected_mask_t0 = torch.tensor([[False, True], [True, False]])
    assert torch.equal(loss_mask[0], expected_mask_t0)

    meta = json.loads((year_dir / "metadata.json").read_text())
    assert meta["mask_contract"] == "region_and_finite_forecast_analysis_no_base_valid_gate"
    assert meta["resolved_index_bbox"] == {"y_start": 0, "y_end": 1, "x_start": 1, "x_end": 2}
    assert manifest["regions"][0]["region_id"] == "US-R1"
    assert manifest["years"]["2015"]["time_steps"] == 2


def test_export_region_tensor_cache_is_idempotent(tmp_path):
    from scripts.data.export_region_tensor_cache import export_region_tensor_cache

    da_nc = tmp_path / "DA.nc"
    masks_nc = tmp_path / "regions.nc"
    out_dir = tmp_path / "cache"
    _write_tiny_da_nc(da_nc)
    _write_tiny_region_masks(masks_nc)

    export_region_tensor_cache(da_nc=da_nc, region_masks_nc=masks_nc, out_dir=out_dir, regions=["US-R1"], years=[2015])
    year_dir = out_dir / "pt" / "US-R1" / "da_full" / "2015"
    first_mtime = (year_dir / "input.pt").stat().st_mtime_ns

    manifest = export_region_tensor_cache(
        da_nc=da_nc,
        region_masks_nc=masks_nc,
        out_dir=out_dir,
        regions=["US-R1"],
        years=[2015],
    )

    assert (year_dir / "input.pt").stat().st_mtime_ns == first_mtime
    assert manifest["regions"][0]["years"]["2015"]["status"] == "complete_existing"


def test_export_region_tensor_cache_rewrites_wrong_shape_existing_tensor(tmp_path):
    from scripts.data.export_region_tensor_cache import export_region_tensor_cache

    da_nc = tmp_path / "DA.nc"
    masks_nc = tmp_path / "regions.nc"
    out_dir = tmp_path / "cache"
    _write_tiny_da_nc(da_nc)
    _write_tiny_region_masks(masks_nc)

    export_region_tensor_cache(da_nc=da_nc, region_masks_nc=masks_nc, out_dir=out_dir, regions=["US-R1"], years=[2015])
    year_dir = out_dir / "pt" / "US-R1" / "da_full" / "2015"
    torch.save(torch.empty((1, 12, 1, 1), dtype=torch.float32), year_dir / "input.pt")

    manifest = export_region_tensor_cache(
        da_nc=da_nc,
        region_masks_nc=masks_nc,
        out_dir=out_dir,
        regions=["US-R1"],
        years=[2015],
    )

    restored = torch.load(year_dir / "input.pt", weights_only=True)
    assert restored.shape == (2, 12, 2, 2)
    assert manifest["regions"][0]["years"]["2015"]["status"] == "written"


def test_audit_region_tensor_cache_detects_nc_pt_mismatch(tmp_path):
    from scripts.data.audit_region_tensor_cache import audit_region_tensor_cache
    from scripts.data.export_region_tensor_cache import export_region_tensor_cache

    da_nc = tmp_path / "DA.nc"
    masks_nc = tmp_path / "regions.nc"
    out_dir = tmp_path / "cache"
    _write_tiny_da_nc(da_nc)
    _write_tiny_region_masks(masks_nc)
    export_region_tensor_cache(da_nc=da_nc, region_masks_nc=masks_nc, out_dir=out_dir, regions=["US-R1"], years=[2015])

    clean = audit_region_tensor_cache(
        cache_dir=out_dir,
        da_nc=da_nc,
        region_masks_nc=masks_nc,
        regions=["US-R1"],
        years=[2015],
        samples_per_year=2,
        exhaustive=True,
    )
    assert clean["all_passed"] is True
    assert clean["n_failures"] == 0

    year_dir = out_dir / "pt" / "US-R1" / "da_full" / "2015"
    inc = torch.load(year_dir / "target_increment.pt", weights_only=True)
    inc[0, 0, 0, 1] += 1.0
    torch.save(inc, year_dir / "target_increment.pt")

    failed = audit_region_tensor_cache(
        cache_dir=out_dir,
        da_nc=da_nc,
        region_masks_nc=masks_nc,
        regions=["US-R1"],
        years=[2015],
        samples_per_year=2,
        exhaustive=True,
    )
    assert failed["all_passed"] is False
    assert failed["n_failures"] >= 1
    assert any("target_increment" in item["check"] for item in failed["failures"])
