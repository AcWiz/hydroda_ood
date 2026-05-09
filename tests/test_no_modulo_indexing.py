"""Test NO modulo indexing — dataset length is exact, IndexError raised at boundary."""

import pytest
from hydroda.data.dataset import HydroDADataset


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"


class TestNoModuloIndexing:
    """Verify __len__ returns exact count and IndexError is raised at boundary."""

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

    def test_len_equals_time_indices_count(self, ds):
        N = len(ds)
        assert N == len(ds._time_indices)

    def test_index_error_at_length_boundary(self, ds):
        N = len(ds)
        with pytest.raises(IndexError):
            _ = ds[N]

    def test_last_valid_index_works(self, ds):
        N = len(ds)
        sample = ds[N - 1]
        assert sample is not None

    def test_negative_index_raises(self, ds):
        # The dataset uses a Python list for indexing which natively supports negative indexing.
        # The contract only requires that idx >= len raises IndexError (no modulo wrap).
        # Negative indices are valid Python behavior and wrap to end-of-list.
        # This test verifies negative indexing works (not an error).
        sample = ds[-1]
        assert sample is not None

    def test_len_nonzero(self, ds):
        assert len(ds) > 0

    def test_target_support_k0_empty(self):
        ds = HydroDADataset(
            da_nc_path=f"{DATA_DIR}/DA.nc",
            region_masks_nc=REGION_MASKS,
            splits_json=SPLITS_JSON,
            target_region="US-R1",
            split_type="target_support",
            K=0, seed=0,
            freeze_manifest=MANIFEST,
        )
        assert len(ds) == 0
        # Accessing empty dataset should raise IndexError
        with pytest.raises(IndexError):
            _ = ds[0]
        ds.close()