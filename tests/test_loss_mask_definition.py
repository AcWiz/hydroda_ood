"""Test loss_mask definition: active_region_mask AND base_valid_mask AND finite conditions."""

import numpy as np
import pytest
from hydroda.data.dataset import HydroDADataset


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"


class TestLossMaskDefinition:
    """Verify loss_mask = active_region_mask AND base_valid_mask AND finite conditions."""

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

    def test_loss_mask_binary_float32(self, ds):
        sample = ds[0]
        lm = sample["loss_mask"]
        assert lm.dtype == np.float32
        assert set(np.unique(lm)).issubset({0.0, 1.0})

    def test_loss_mask_zeros_outside_active_region(self, ds):
        sample = ds[0]
        active = sample["active_region_mask"]
        lm = sample["loss_mask"]
        # Where active_region_mask is 0, loss_mask should also be 0
        assert (lm[active == 0] == 0).all()

    def test_loss_mask_zeros_where_base_valid_mask_zero(self, ds):
        sample = ds[0]
        bvm = sample["base_valid_mask"]
        lm = sample["loss_mask"]
        # Where base_valid_mask is 0, loss_mask should also be 0
        assert (lm[bvm == 0] == 0).all()

    def test_loss_mask_zeros_where_forecast_surface_nonfinite(self, ds):
        sample = ds[0]
        fs = sample["forecast_surface"]
        lm = sample["loss_mask"]
        # Where forecast_surface is non-finite, loss_mask should be 0
        nonfinite = ~np.isfinite(fs)
        if nonfinite.any():
            assert (lm[nonfinite] == 0).all()

    def test_loss_mask_zeros_where_forecast_rootzone_nonfinite(self, ds):
        sample = ds[0]
        fr = sample["forecast_rootzone"]
        lm = sample["loss_mask"]
        nonfinite = ~np.isfinite(fr)
        if nonfinite.any():
            assert (lm[nonfinite] == 0).all()

    def test_loss_mask_zeros_where_analysis_surface_nonfinite(self, ds):
        sample = ds[0]
        a_s = sample["analysis_surface"]
        lm = sample["loss_mask"]
        nonfinite = ~np.isfinite(a_s)
        if nonfinite.any():
            assert (lm[nonfinite] == 0).all()

    def test_loss_mask_zeros_where_analysis_rootzone_nonfinite(self, ds):
        sample = ds[0]
        a_r = sample["analysis_rootzone"]
        lm = sample["loss_mask"]
        nonfinite = ~np.isfinite(a_r)
        if nonfinite.any():
            assert (lm[nonfinite] == 0).all()

    def test_metric_mask_same_as_loss_mask(self, ds):
        sample = ds[0]
        np.testing.assert_array_equal(sample["metric_mask"], sample["loss_mask"])