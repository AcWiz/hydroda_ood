import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import torch
import xarray as xr


def _write_tiny_da_nc(path: Path) -> None:
    # 2015-01-01, 2022-01-01, 2023-01-01.
    times = np.array([0.0, 2557.0, 2922.0], dtype=np.float64)
    input_arr = np.zeros((3, 12, 3, 4), dtype=np.float32)
    target_arr = np.zeros((3, 4, 3, 4), dtype=np.float32)
    grid = np.arange(12, dtype=np.float32).reshape(3, 4)
    for t in range(3):
        for c in range(12):
            input_arr[t, c] = 1000 * t + 100 * c + grid
        for c in range(4):
            target_arr[t, c] = 2000 * t + 50 * c + grid
    target_arr[:, 0] = input_arr[:, 0] + 10.0
    target_arr[:, 1] = input_arr[:, 1] - 5.0
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
            [2, 1, 1, 2],
            [2, 0, 0, 2],
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


def _write_split_manifest(path: Path) -> None:
    split = {
        "target_region_id": "US-R1",
        "source_region_ids": ["US-R1"],
        "adaptation_setting": "target_full_train",
        "K": None,
        "seed": 0,
        "source_train_dates": [{"time_index": 0, "date_str": "2015-01-01", "datetime_str": "2015-01-01T00:00:00"}],
        "source_val_dates": [{"time_index": 1, "date_str": "2022-01-01", "datetime_str": "2022-01-01T00:00:00"}],
        "target_train_dates": [{"time_index": 0, "date_str": "2015-01-01", "datetime_str": "2015-01-01T00:00:00"}],
        "target_eval_dates": [{"time_index": 2, "date_str": "2023-01-01", "datetime_str": "2023-01-01T00:00:00"}],
        "target_train_dates_hash": "trainhash",
        "target_eval_dates_hash": "evalhash",
        "split_manifest_sha256": "splithash",
    }
    path.write_text(json.dumps({"splits": [split]}, indent=2), encoding="utf-8")


def _build_tiny_cache(tmp_path: Path, regions: list[str] | None = None):
    from scripts.data.export_region_tensor_cache import export_region_tensor_cache

    da_nc = tmp_path / "DA.nc"
    masks_nc = tmp_path / "regions.nc"
    splits_json = tmp_path / "splits.json"
    cache_dir = tmp_path / "cache"
    _write_tiny_da_nc(da_nc)
    _write_tiny_region_masks(masks_nc)
    _write_split_manifest(splits_json)
    export_region_tensor_cache(
        da_nc=da_nc,
        region_masks_nc=masks_nc,
        out_dir=cache_dir,
        regions=regions or ["US-R1"],
        years=[2015, 2022, 2023],
    )
    return da_nc, masks_nc, splits_json, cache_dir


def test_tensor_cache_dataset_sample_matches_netcdf_region_crop(tmp_path):
    from hydroda.data.dataset import HydroDADataset, TensorCacheHydroDADataset

    da_nc, masks_nc, splits_json, cache_dir = _build_tiny_cache(tmp_path)
    nc_dataset = HydroDADataset(
        da_nc_path=str(da_nc),
        region_masks_nc=str(masks_nc),
        splits_json=str(splits_json),
        target_region="US-R1",
        split_type="source_fit",
        K=None,
        seed=0,
        adaptation_setting="target_full_train",
        freeze_manifest=None,
    )
    nc_dataset.set_active_region("US-R1")
    tensor_dataset = TensorCacheHydroDADataset(
        da_nc_path=str(da_nc),
        region_masks_nc=str(masks_nc),
        splits_json=str(splits_json),
        target_region="US-R1",
        split_type="source_fit",
        K=None,
        seed=0,
        adaptation_setting="target_full_train",
        freeze_manifest=None,
        active_region_id="US-R1",
        tensor_cache_dir=str(cache_dir),
    )

    nc_sample = nc_dataset[0]
    tensor_sample = tensor_dataset[0]
    bbox = tensor_dataset.region_metadata["resolved_index_bbox"]
    ys = slice(bbox["y_start"], bbox["y_end"] + 1)
    xs = slice(bbox["x_start"], bbox["x_end"] + 1)

    np.testing.assert_allclose(tensor_sample["x"], nc_sample["x"][:, ys, xs], equal_nan=True)
    np.testing.assert_allclose(tensor_sample["increment_surface"], nc_sample["increment_surface"][ys, xs], equal_nan=True)
    np.testing.assert_allclose(tensor_sample["increment_rootzone"], nc_sample["increment_rootzone"][ys, xs], equal_nan=True)
    np.testing.assert_allclose(tensor_sample["analysis_surface"], nc_sample["analysis_surface"][ys, xs], equal_nan=True)
    np.testing.assert_allclose(tensor_sample["analysis_rootzone"], nc_sample["analysis_rootzone"][ys, xs], equal_nan=True)
    np.testing.assert_array_equal(tensor_sample["loss_mask"], nc_sample["loss_mask"][ys, xs])
    np.testing.assert_array_equal(tensor_sample["metric_mask"], tensor_sample["loss_mask"])
    np.testing.assert_allclose(tensor_sample["latitude_weight"], nc_sample["latitude_weight"][ys, xs])
    valid = tensor_sample["loss_mask"] > 0.5
    np.testing.assert_allclose(
        (tensor_sample["forecast_surface"] + tensor_sample["increment_surface"])[valid],
        tensor_sample["analysis_surface"][valid],
    )
    np.testing.assert_allclose(
        (tensor_sample["forecast_rootzone"] + tensor_sample["increment_rootzone"])[valid],
        tensor_sample["analysis_rootzone"][valid],
    )
    assert tensor_sample["sample_region_id"] == "US-R1"
    assert tensor_sample["active_region_ids"] == ["US-R1"]
    assert tensor_sample["split_manifest_sha256"] == "splithash"
    nc_dataset.close()
    tensor_dataset.close()


def test_tensor_cache_load_mode_mmap_passes_mmap_to_torch_load(tmp_path, monkeypatch):
    from hydroda.data.dataset import TensorCacheHydroDADataset

    da_nc, masks_nc, splits_json, cache_dir = _build_tiny_cache(tmp_path)
    calls = []
    original_torch_load = torch.load

    def tracking_torch_load(*args, **kwargs):
        calls.append({"path": Path(args[0]), "kwargs": dict(kwargs)})
        kwargs.pop("mmap", None)
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracking_torch_load)

    dataset = TensorCacheHydroDADataset(
        da_nc_path=str(da_nc),
        region_masks_nc=str(masks_nc),
        splits_json=str(splits_json),
        target_region="US-R1",
        split_type="source_fit",
        K=None,
        seed=0,
        adaptation_setting="target_full_train",
        freeze_manifest=None,
        active_region_id="US-R1",
        tensor_cache_dir=str(cache_dir),
        tensor_cache_load_mode="mmap",
    )

    dataset[0]

    assert calls
    assert all(call["kwargs"].get("weights_only") is True for call in calls)
    assert all(call["kwargs"].get("mmap") is True for call in calls)
    dataset.close()


def test_tensor_cache_eager_and_mmap_samples_match(tmp_path):
    from hydroda.data.dataset import TensorCacheHydroDADataset

    da_nc, masks_nc, splits_json, cache_dir = _build_tiny_cache(tmp_path)
    common_kwargs = dict(
        da_nc_path=str(da_nc),
        region_masks_nc=str(masks_nc),
        splits_json=str(splits_json),
        target_region="US-R1",
        split_type="source_fit",
        K=None,
        seed=0,
        adaptation_setting="target_full_train",
        freeze_manifest=None,
        active_region_id="US-R1",
        tensor_cache_dir=str(cache_dir),
    )
    eager = TensorCacheHydroDADataset(**common_kwargs, tensor_cache_load_mode="eager")
    mmap = TensorCacheHydroDADataset(**common_kwargs, tensor_cache_load_mode="mmap")

    eager_sample = eager[0]
    mmap_sample = mmap[0]

    for key in (
        "x",
        "forecast_surface",
        "forecast_rootzone",
        "analysis_surface",
        "analysis_rootzone",
        "increment_surface",
        "increment_rootzone",
        "base_valid_mask",
        "label_valid_mask",
        "region_mask_integer",
        "active_region_mask",
        "region_mask",
        "loss_mask",
        "metric_mask",
        "latitude_weight",
    ):
        assert eager_sample[key].shape == mmap_sample[key].shape
        np.testing.assert_allclose(eager_sample[key], mmap_sample[key], equal_nan=True)
    for key in (
        "date_str",
        "month",
        "season",
        "time_index",
        "target_region_id",
        "active_region_ids",
        "sample_region_id",
        "split_role",
    ):
        assert eager_sample[key] == mmap_sample[key]

    eager.close()
    mmap.close()


def test_tensor_cache_target_eval_reads_only_manifest_eval_dates(tmp_path):
    from hydroda.data.dataset import TensorCacheHydroDADataset

    da_nc, masks_nc, splits_json, cache_dir = _build_tiny_cache(tmp_path)
    train_dataset = TensorCacheHydroDADataset(
        da_nc_path=str(da_nc),
        region_masks_nc=str(masks_nc),
        splits_json=str(splits_json),
        target_region="US-R1",
        split_type="source_fit",
        K=None,
        seed=0,
        adaptation_setting="target_full_train",
        freeze_manifest=None,
        active_region_id="US-R1",
        tensor_cache_dir=str(cache_dir),
    )
    eval_dataset = TensorCacheHydroDADataset(
        da_nc_path=str(da_nc),
        region_masks_nc=str(masks_nc),
        splits_json=str(splits_json),
        target_region="US-R1",
        split_type="target_eval",
        K=None,
        seed=0,
        adaptation_setting="target_full_train",
        freeze_manifest=None,
        active_region_id="US-R1",
        tensor_cache_dir=str(cache_dir),
    )

    assert [d["date_str"] for d in train_dataset._date_records] == ["2015-01-01"]
    assert [d["date_str"] for d in eval_dataset._date_records] == ["2023-01-01"]
    assert eval_dataset[0]["time_index"] == 2
    train_dataset.close()
    eval_dataset.close()


def test_tensor_cache_dataset_bounds_year_cache_to_limit(tmp_path):
    from hydroda.data.dataset import TensorCacheHydroDADataset

    da_nc, masks_nc, splits_json, cache_dir = _build_tiny_cache(tmp_path)
    region_dir = cache_dir / "pt" / "US-R1"
    shutil.copytree(region_dir / "da_full" / "2022", region_dir / "da_full" / "2020")
    shutil.copytree(region_dir / "da_full" / "2023", region_dir / "da_full" / "2021")
    region_meta_path = region_dir / "metadata.json"
    region_meta = json.loads(region_meta_path.read_text(encoding="utf-8"))
    region_meta["years"]["2020"] = region_meta["years"]["2022"]
    region_meta["years"]["2021"] = region_meta["years"]["2023"]
    region_meta_path.write_text(json.dumps(region_meta), encoding="utf-8")

    split_data = json.loads(splits_json.read_text(encoding="utf-8"))
    split_data["splits"][0]["target_train_dates"] = [
        {"time_index": 0, "date_str": "2015-01-01", "datetime_str": "2015-01-01T00:00:00"},
        {"time_index": 1, "date_str": "2020-01-01", "datetime_str": "2020-01-01T00:00:00"},
        {"time_index": 2, "date_str": "2021-01-01", "datetime_str": "2021-01-01T00:00:00"},
    ]
    splits_json.write_text(json.dumps(split_data), encoding="utf-8")

    dataset = TensorCacheHydroDADataset(
        da_nc_path=str(da_nc),
        region_masks_nc=str(masks_nc),
        splits_json=str(splits_json),
        target_region="US-R1",
        split_type="target_train",
        K=None,
        seed=0,
        adaptation_setting="target_full_train",
        freeze_manifest=None,
        active_region_id="US-R1",
        tensor_cache_dir=str(cache_dir),
        max_year_cache_entries=1,
    )

    dataset[0]
    dataset[1]
    dataset[2]

    assert list(dataset._year_cache.keys()) == [2021]
    assert len(dataset._year_cache) == 1
    dataset.close()


def test_tensor_cache_input_side_sample_populates_year_cache(tmp_path, monkeypatch):
    from hydroda.data.dataset import TensorCacheHydroDADataset

    da_nc, masks_nc, splits_json, cache_dir = _build_tiny_cache(tmp_path)
    dataset = TensorCacheHydroDADataset(
        da_nc_path=str(da_nc),
        region_masks_nc=str(masks_nc),
        splits_json=str(splits_json),
        target_region="US-R1",
        split_type="source_fit",
        K=None,
        seed=0,
        adaptation_setting="target_full_train",
        freeze_manifest=None,
        active_region_id="US-R1",
        tensor_cache_dir=str(cache_dir),
        max_year_cache_entries=1,
    )
    load_count = {"input": 0}
    original_load_tensor = dataset._load_tensor

    def counting_load_tensor(path):
        if Path(path).name == "input.pt":
            load_count["input"] += 1
        return original_load_tensor(path)

    monkeypatch.setattr(dataset, "_load_tensor", counting_load_tensor)

    first = dataset.get_input_side_sample(0)
    second = dataset.get_input_side_sample(0)

    assert first["sample_region_id"] == "US-R1"
    assert second["sample_region_id"] == "US-R1"
    assert load_count["input"] == 1
    assert list(dataset._year_cache.keys()) == [2015]
    assert "input" in dataset._year_cache[2015]
    dataset.close()


def test_tensor_cache_full_sample_after_input_side_cache_loads_missing_tensors(tmp_path):
    from hydroda.data.dataset import TensorCacheHydroDADataset

    da_nc, masks_nc, splits_json, cache_dir = _build_tiny_cache(tmp_path)
    dataset = TensorCacheHydroDADataset(
        da_nc_path=str(da_nc),
        region_masks_nc=str(masks_nc),
        splits_json=str(splits_json),
        target_region="US-R1",
        split_type="source_fit",
        K=None,
        seed=0,
        adaptation_setting="target_full_train",
        freeze_manifest=None,
        active_region_id="US-R1",
        tensor_cache_dir=str(cache_dir),
        max_year_cache_entries=1,
    )

    dataset.get_input_side_sample(0)
    sample = dataset[0]

    assert "target_increment" in dataset._year_cache[2015]
    assert "target_analysis" in dataset._year_cache[2015]
    assert "loss_mask" in dataset._year_cache[2015]
    assert sample["increment_surface"].shape == sample["loss_mask"].shape
    dataset.close()


def test_tensor_cache_dataset_missing_cache_errors_clearly(tmp_path):
    from hydroda.data.dataset import TensorCacheHydroDADataset

    da_nc, masks_nc, splits_json, _cache_dir = _build_tiny_cache(tmp_path)
    with pytest.raises(FileNotFoundError, match="tensor cache manifest"):
        TensorCacheHydroDADataset(
            da_nc_path=str(da_nc),
            region_masks_nc=str(masks_nc),
            splits_json=str(splits_json),
            target_region="US-R1",
            split_type="source_fit",
            K=None,
            seed=0,
            adaptation_setting="target_full_train",
            freeze_manifest=None,
            active_region_id="US-R1",
            tensor_cache_dir=str(tmp_path / "missing-cache"),
        )


def test_tensor_cache_dataset_missing_year_errors_clearly(tmp_path):
    from hydroda.data.dataset import TensorCacheHydroDADataset

    da_nc, masks_nc, splits_json, cache_dir = _build_tiny_cache(tmp_path)
    split_data = json.loads(splits_json.read_text(encoding="utf-8"))
    split_data["splits"][0]["target_eval_dates"] = [
        {"time_index": 999, "date_str": "2024-01-01", "datetime_str": "2024-01-01T00:00:00"}
    ]
    splits_json.write_text(json.dumps(split_data), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="cache year 2024"):
        TensorCacheHydroDADataset(
            da_nc_path=str(da_nc),
            region_masks_nc=str(masks_nc),
            splits_json=str(splits_json),
            target_region="US-R1",
            split_type="target_eval",
            K=None,
            seed=0,
            adaptation_setting="target_full_train",
            freeze_manifest=None,
            active_region_id="US-R1",
            tensor_cache_dir=str(cache_dir),
        )


def test_tensor_cache_dataset_mask_contract_mismatch_errors_clearly(tmp_path):
    from hydroda.data.dataset import TensorCacheHydroDADataset

    da_nc, masks_nc, splits_json, cache_dir = _build_tiny_cache(tmp_path)
    manifest_path = cache_dir / "manifest_region_crops_US.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["mask_contract"] = "old_contract"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="mask contract"):
        TensorCacheHydroDADataset(
            da_nc_path=str(da_nc),
            region_masks_nc=str(masks_nc),
            splits_json=str(splits_json),
            target_region="US-R1",
            split_type="source_fit",
            K=None,
            seed=0,
            adaptation_setting="target_full_train",
            freeze_manifest=None,
            active_region_id="US-R1",
            tensor_cache_dir=str(cache_dir),
        )


def test_build_hydroda_dataset_tensor_cache_backend(tmp_path):
    from hydroda.data.dataset import TensorCacheHydroDADataset, build_hydroda_dataset

    da_nc, masks_nc, splits_json, cache_dir = _build_tiny_cache(tmp_path)
    dataset = build_hydroda_dataset(
        da_nc_path=str(da_nc),
        region_masks_nc=str(masks_nc),
        splits_json=str(splits_json),
        target_region="US-R1",
        split_type="source_fit",
        K=None,
        seed=0,
        adaptation_setting="target_full_train",
        freeze_manifest=None,
        dataset_backend="tensor_cache",
        active_region_id="US-R1",
        tensor_cache_dir=str(cache_dir),
    )
    assert isinstance(dataset, TensorCacheHydroDADataset)
    assert dataset[0]["x"].shape == (12, 2, 2)
    dataset.close()


def test_build_hydroda_dataset_tensor_cache_multi_region_backend(tmp_path):
    from hydroda.data.dataset import MultiRegionTensorCacheHydroDADataset, build_hydroda_dataset

    da_nc, masks_nc, splits_json, cache_dir = _build_tiny_cache(tmp_path, regions=["US-R1", "US-R2"])
    dataset = build_hydroda_dataset(
        da_nc_path=str(da_nc),
        region_masks_nc=str(masks_nc),
        splits_json=str(splits_json),
        target_region="US-R1",
        split_type="source_fit",
        K=None,
        seed=0,
        adaptation_setting="target_full_train",
        freeze_manifest=None,
        dataset_backend="tensor_cache",
        active_region_ids=["US-R1", "US-R2"],
        tensor_cache_dir=str(cache_dir),
    )

    assert isinstance(dataset, MultiRegionTensorCacheHydroDADataset)
    assert len(dataset) == 2
    assert dataset._active_region_ids == ["US-R1", "US-R2"]
    assert [dataset[i]["sample_region_id"] for i in range(len(dataset))] == ["US-R1", "US-R2"]
    assert len(dataset._date_records) == 2
    assert {record["sample_region_id"] for record in dataset._date_records} == {"US-R1", "US-R2"}
    dataset.close()


def test_build_hydroda_dataset_netcdf_source_region_episode_backend(tmp_path):
    from hydroda.data.dataset import SourceRegionEpisodeHydroDADataset, build_hydroda_dataset

    da_nc, masks_nc, splits_json, _cache_dir = _build_tiny_cache(tmp_path, regions=["US-R1", "US-R2"])
    dataset = build_hydroda_dataset(
        da_nc_path=str(da_nc),
        region_masks_nc=str(masks_nc),
        splits_json=str(splits_json),
        target_region="US-R1",
        split_type="source_fit",
        K=None,
        seed=0,
        adaptation_setting="target_full_train",
        freeze_manifest=None,
        dataset_backend="netcdf",
        active_region_ids=["US-R1", "US-R2"],
    )

    assert isinstance(dataset, SourceRegionEpisodeHydroDADataset)
    assert len(dataset) == 2
    samples = [dataset[i] for i in range(len(dataset))]
    assert [sample["sample_region_id"] for sample in samples] == ["US-R1", "US-R2"]
    assert [sample["active_region_ids"] for sample in samples] == [["US-R1"], ["US-R2"]]
    assert np.count_nonzero(samples[0]["loss_mask"]) > 0
    assert np.count_nonzero(samples[1]["loss_mask"]) > 0
    assert np.all(samples[0]["loss_mask"][samples[0]["region_mask_integer"] == 2] == 0.0)
    assert np.all(samples[1]["loss_mask"][samples[1]["region_mask_integer"] == 1] == 0.0)
    dataset.close()


def test_collate_hydroda_samples_pads_mixed_region_tensor_crops(tmp_path):
    from hydroda.data.dataset import build_hydroda_dataset, collate_hydroda_samples

    da_nc, masks_nc, splits_json, cache_dir = _build_tiny_cache(tmp_path, regions=["US-R1", "US-R2"])
    dataset = build_hydroda_dataset(
        da_nc_path=str(da_nc),
        region_masks_nc=str(masks_nc),
        splits_json=str(splits_json),
        target_region="US-R1",
        split_type="source_fit",
        K=None,
        seed=0,
        adaptation_setting="target_full_train",
        freeze_manifest=None,
        dataset_backend="tensor_cache",
        active_region_ids=["US-R1", "US-R2"],
        tensor_cache_dir=str(cache_dir),
    )

    batch = collate_hydroda_samples([dataset[0], dataset[1]])

    assert batch["x"].shape == (2, 12, 2, 4)
    assert batch["increment_surface"].shape == (2, 2, 4)
    assert batch["loss_mask"].shape == (2, 2, 4)
    assert batch["latitude_weight"].shape == (2, 2, 4)
    assert torch.count_nonzero(batch["loss_mask"][0, :, 2:]) == 0
    assert torch.count_nonzero(batch["latitude_weight"][0, :, 2:]) == 0
    assert torch.count_nonzero(batch["loss_mask"][1]) > 0
    dataset.close()
