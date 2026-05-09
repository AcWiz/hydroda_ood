"""Test base_valid_mask is binary 0/1 float32 from input ch 11."""

import numpy as np
import pytest
from hydroda.data.dataset import HydroDADataset


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"


class TestBaseValidMaskBinary:
    """Verify base_valid_mask is binary 0/1 float32 derived from input ch 11."""

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

    def test_base_valid_mask_unique_values(self, ds):
        sample = ds[0]
        unique = np.unique(sample["base_valid_mask"])
        assert set(unique).issubset({0.0, 1.0}), (
            f"base_valid_mask unique values {unique} not subset of {{0.0, 1.0}}"
        )

    def test_base_valid_mask_float32(self, ds):
        sample = ds[0]
        assert sample["base_valid_mask"].dtype == np.float32

    def test_base_valid_mask_matches_threshold(self, ds):
        sample = ds[0]
        x = sample["x"]
        raw_ch11 = x[11]
        expected_mask = (raw_ch11 > 0.5).astype(np.float32)
        np.testing.assert_array_equal(
            sample["base_valid_mask"], expected_mask,
            err_msg="base_valid_mask does not match (x[11] > 0.5)"
        )

    @pytest.mark.parametrize("idx", [0, 50, 200])
    def test_base_valid_mask_binary_at_indices(self, idx, ds):
        sample = ds[idx]
        unique = np.unique(sample["base_valid_mask"])
        assert set(unique).issubset({0.0, 1.0}), (
            f"idx={idx}: base_valid_mask values {unique} not binary"
        )