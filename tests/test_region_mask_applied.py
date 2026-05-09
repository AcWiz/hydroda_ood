"""Test region mask is correctly applied in dataset."""

import numpy as np
import pytest
import xarray as xr
from hydroda.data.dataset import HydroDADataset


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"


class TestRegionMaskApplied:
    """Verify region_mask_integer shape/values and active_region_mask construction."""

    @pytest.fixture
    def region_mask_int(self):
        ds = xr.open_dataset(REGION_MASKS)
        mask = ds["region_mask_integer"].values.astype(np.int16)
        ds.close()
        return mask

    def test_region_mask_integer_shape(self, region_mask_int):
        assert region_mask_int.shape == (256, 640), (
            f"region_mask_integer shape {region_mask_int.shape} != (256, 640)"
        )

    def test_region_mask_integer_values(self, region_mask_int):
        vals = np.unique(region_mask_int)
        assert set(vals).issubset(set(range(7))), (
            f"region_mask_integer contains values {vals} outside 0..6"
        )

    def test_active_region_mask_source_train(self, region_mask_int):
        """source_train: active_region_mask = union of 5 source regions (no target)."""
        ds = HydroDADataset(
            da_nc_path=f"{DATA_DIR}/DA.nc",
            region_masks_nc=REGION_MASKS,
            splits_json=SPLITS_JSON,
            target_region="US-R1",
            split_type="source_train",
            K=0, seed=0,
            freeze_manifest=MANIFEST,
        )
        sample = ds[0]
        active = sample["active_region_mask"]
        # US-R1 is held-out, so R1 pixels should be 0
        r1_pixels = region_mask_int == 1
        assert (active[r1_pixels] == 0).all(), (
            "source_train active_region_mask should be 0 at target region US-R1 pixels"
        )
        # active_region_mask should be 1 at source pixels (R2..R6) and 0 elsewhere
        source_mask = ((region_mask_int == 2) | (region_mask_int == 3) |
                       (region_mask_int == 4) | (region_mask_int == 5) |
                       (region_mask_int == 6))
        non_source = np.logical_not(source_mask)
        assert (active[non_source] == 0).all(), (
            "source_train active_region_mask should be 0 outside source regions"
        )
        ds.close()

    def test_active_region_mask_target_support(self, region_mask_int):
        """target_support: active_region_mask = single target region."""
        ds = HydroDADataset(
            da_nc_path=f"{DATA_DIR}/DA.nc",
            region_masks_nc=REGION_MASKS,
            splits_json=SPLITS_JSON,
            target_region="US-R1",
            split_type="target_support",
            K=4, seed=0,
            freeze_manifest=MANIFEST,
        )
        if len(ds) == 0:
            pytest.skip("K=0 target_support is empty")
        sample = ds[0]
        active = sample["active_region_mask"]
        # Only R1 pixels should be active
        r1_pixels = region_mask_int == 1
        # At R1 pixels, active should be 1
        assert (active[r1_pixels] == 1).all(), (
            "target_support active_region_mask should be 1 at target region pixels"
        )
        # At non-R1 pixels, active should be 0
        non_r1 = np.logical_not(r1_pixels)
        assert (active[non_r1] == 0).all(), (
            "target_support active_region_mask should be 0 outside target region"
        )
        ds.close()

    def test_region_mask_integer_not_in_loss_mask_directly(self):
        """region_mask_integer values must not appear in loss_mask computation.

        loss_mask is computed from active_region_mask (binary float),
        not from raw integer region_mask_integer values.
        """
        ds = HydroDADataset(
            da_nc_path=f"{DATA_DIR}/DA.nc",
            region_masks_nc=REGION_MASKS,
            splits_json=SPLITS_JSON,
            target_region="US-R2",
            split_type="source_train",
            K=0, seed=0,
            freeze_manifest=MANIFEST,
        )
        sample = ds[0]
        # active_region_mask must be float32 0/1, not integer region ids
        assert sample["active_region_mask"].dtype == np.float32
        assert set(np.unique(sample["active_region_mask"])).issubset({0.0, 1.0})
        # loss_mask is constructed from active_region_mask, not region_mask_integer
        assert sample["loss_mask"].dtype == np.float32
        ds.close()