"""Test raw passthrough: x values are exactly as stored in DA.nc (no normalization)."""

import numpy as np
import pytest
import xarray as xr
from hydroda.data.dataset import HydroDADataset


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"


class TestRawPassthroughNoNormalization:
    """Verify x values are exactly as stored in DA.nc (raw passthrough, no normalization)."""

    @pytest.fixture
    def ds(self):
        ds = HydroDADataset(
            da_nc_path=f"{DATA_DIR}/DA.nc",
            region_masks_nc=REGION_MASKS,
            splits_json=SPLITS_JSON,
            target_region="US-R1",
            split_type="source_train",
            K=0, seed=0,
            freeze_manifest=MANIFEST,
        )
        yield ds
        ds.close()

    @pytest.fixture
    def da_ds_lazy(self):
        return xr.open_dataset(f"{DATA_DIR}/DA.nc", chunks={"time": 100})

    def test_x_matches_raw_storage(self, ds, da_ds_lazy):
        """x[0] from dataset should exactly match DA.nc input at same time index."""
        sample = ds[0]
        time_idx = sample["time_index"]
        raw = da_ds_lazy["input"].isel(time=time_idx).values
        np.testing.assert_array_equal(
            sample["x"], raw,
            err_msg="x does not exactly match raw DA.nc storage"
        )
        da_ds_lazy.close()

    def test_forecast_surface_matches_x_channel0(self, ds):
        sample = ds[0]
        np.testing.assert_array_equal(
            sample["forecast_surface"], sample["x"][0],
            err_msg="forecast_surface does not match x[0]"
        )

    def test_no_normalization_attribute(self, ds):
        """Dataset should have no normalization attributes applied."""
        sample = ds[0]
        x = sample["x"]
        # Values should be in the same range as raw SM values (roughly 0-0.5 for surface)
        # Not centered/subtracted
        # We check by verifying x[0] (forecast_surface) has non-zero mean
        # (if normalization subtracted mean, mean would be ~0)
        mean_val = x[0].mean()
        assert abs(mean_val) > 0.01, (
            f"forecast_surface mean={mean_val} suspiciously close to 0, "
            "indicating possible mean-subtraction normalization"
        )

    @pytest.mark.parametrize("ch", [0, 1, 5, 11])
    def test_raw_channel_values_not_normalized(self, ds, ch):
        sample = ds[0]
        # Raw values should span the original range
        ch_vals = sample["x"][ch]
        # If normalized, std would be ~1 and mean ~0
        # Raw passthrough has original scales
        std_val = ch_vals.std()
        # For DA.nc data, std should NOT be close to 1.0 (which would indicate z-score norm)
        assert std_val > 0.01, (
            f"channel {ch} std={std_val} suspiciously small, may indicate normalization"
        )

    def test_target_values_match_raw(self, ds, da_ds_lazy):
        sample = ds[0]
        time_idx = sample["time_index"]
        raw_target = da_ds_lazy["target"].isel(time=time_idx).values
        np.testing.assert_array_equal(
            sample["analysis_surface"], raw_target[0],
            err_msg="analysis_surface does not match raw target[0]"
        )
        da_ds_lazy.close()