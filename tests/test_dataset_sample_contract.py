"""Test HydroDADataset sample contract keys and shapes."""

import numpy as np
import pytest
from hydroda.data.dataset import HydroDADataset


class TestDatasetSampleContract:
    """Verify sample dict keys, dtypes, and shapes match contract."""

    DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
    REGION_MASKS = "artifacts/regions/US_region_masks.nc"
    SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
    MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"

    @pytest.fixture
    def source_train_ds(self):
        ds = HydroDADataset(
            da_nc_path=f"{self.DATA_DIR}/DA.nc",
            region_masks_nc=self.REGION_MASKS,
            splits_json=self.SPLITS_JSON,
            target_region="US-R1",
            split_type="source_train",
            K=0, seed=0,
            freeze_manifest=self.MANIFEST,
        )
        yield ds
        ds.close()

    @pytest.fixture
    def target_support_ds(self):
        ds = HydroDADataset(
            da_nc_path=f"{self.DATA_DIR}/DA.nc",
            region_masks_nc=self.REGION_MASKS,
            splits_json=self.SPLITS_JSON,
            target_region="US-R1",
            split_type="target_support",
            K=4, seed=0,
            freeze_manifest=self.MANIFEST,
        )
        yield ds
        ds.close()

    @pytest.fixture
    def target_query_ds(self):
        ds = HydroDADataset(
            da_nc_path=f"{self.DATA_DIR}/DA.nc",
            region_masks_nc=self.REGION_MASKS,
            splits_json=self.SPLITS_JSON,
            target_region="US-R1",
            split_type="target_query",
            K=0, seed=0,
            freeze_manifest=self.MANIFEST,
        )
        yield ds
        ds.close()

    def test_sample_keys(self, source_train_ds):
        sample = source_train_ds[0]
        required_keys = {
            "x", "forecast_surface", "forecast_rootzone",
            "analysis_surface", "analysis_rootzone",
            "increment_surface", "increment_rootzone",
            "base_valid_mask", "label_valid_mask",
            "region_mask_integer",
            "active_region_mask", "region_mask",
            "loss_mask", "metric_mask",
            "latitude", "latitude_weight",
            "date_str", "month", "season",
            "time_index", "country_id",
            "target_region_id", "active_region_ids",
            "split_role", "regime_id", "split_id",
            "K", "seed",
        }
        assert required_keys.issubset(set(sample.keys())), (
            f"Missing keys: {required_keys - set(sample.keys())}"
        )

    def test_x_shape(self, source_train_ds):
        sample = source_train_ds[0]
        assert sample["x"].shape == (12, 256, 640), (
            f"x shape {sample['x'].shape} != (12, 256, 640)"
        )

    def test_forecast_analysis_shapes(self, source_train_ds):
        sample = source_train_ds[0]
        for key in ["forecast_surface", "forecast_rootzone",
                    "analysis_surface", "analysis_rootzone",
                    "increment_surface", "increment_rootzone"]:
            assert sample[key].shape == (256, 640), (
                f"{key} shape {sample[key].shape} != (256, 640)"
            )

    def test_increment_surface_calculation(self, source_train_ds):
        sample = source_train_ds[0]
        expected = sample["analysis_surface"] - sample["forecast_surface"]
        np.testing.assert_allclose(
            sample["increment_surface"], expected, rtol=1e-5, atol=1e-8
        )

    def test_increment_rootzone_calculation(self, source_train_ds):
        sample = source_train_ds[0]
        expected = sample["analysis_rootzone"] - sample["forecast_rootzone"]
        np.testing.assert_allclose(
            sample["increment_rootzone"], expected, rtol=1e-5, atol=1e-8
        )

    def test_region_mask_integer_dtype(self, source_train_ds):
        sample = source_train_ds[0]
        assert sample["region_mask_integer"].dtype == np.int16, (
            f"region_mask_integer dtype {sample['region_mask_integer'].dtype} != int16"
        )

    def test_region_mask_integer_values(self, source_train_ds):
        sample = source_train_ds[0]
        vals = np.unique(sample["region_mask_integer"])
        assert set(vals).issubset(set(range(7))), (
            f"region_mask_integer contains values {vals} outside 0..6"
        )

    def test_base_valid_mask_binary(self, source_train_ds):
        sample = source_train_ds[0]
        unique = np.unique(sample["base_valid_mask"])
        assert set(unique).issubset({0.0, 1.0}), (
            f"base_valid_mask has values {unique} outside {{0.0, 1.0}}"
        )

    def test_source_train_has_length(self, source_train_ds):
        assert len(source_train_ds) > 0

    def test_target_support_empty_when_k0(self):
        ds = HydroDADataset(
            da_nc_path=f"{self.DATA_DIR}/DA.nc",
            region_masks_nc=self.REGION_MASKS,
            splits_json=self.SPLITS_JSON,
            target_region="US-R1",
            split_type="target_support",
            K=0, seed=0,
            freeze_manifest=self.MANIFEST,
        )
        assert len(ds) == 0
        ds.close()

    def test_target_query_has_length(self, target_query_ds):
        assert len(target_query_ds) > 0

    def test_dataset_returns_latitude_weight(self, source_train_ds):
        """Verify dataset returns latitude and latitude_weight fields."""
        sample = source_train_ds[0]
        assert "latitude" in sample, "Sample must contain 'latitude'"
        assert "latitude_weight" in sample, "Sample must contain 'latitude_weight'"

        lat = sample["latitude"]
        latw = sample["latitude_weight"]

        assert lat.shape == (256, 640), f"latitude shape {lat.shape} != (256, 640)"
        assert latw.shape == (256, 640), f"latitude_weight shape {latw.shape} != (256, 640)"
        assert lat.dtype == np.float32, f"latitude dtype {lat.dtype}"
        assert latw.dtype == np.float32, f"latitude_weight dtype {latw.dtype}"

        # Latitude weight should be cos(lat), between 0 and 1
        assert np.all(latw >= 0.0), "latitude_weight should be >= 0"
        assert np.all(latw <= 1.0), "latitude_weight should be <= 1"

        # Latitude should be in valid US range (~25-50 deg N)
        assert 20.0 <= lat.max() <= 55.0, f"latitude max {lat.max()} outside expected US range"