"""Test that target_query statistics are NOT used in normalization.

No-leakage: target_query samples are for offline evaluation only;
no query statistics are used in dataset normalization.
"""

import pytest
from hydroda.data.dataset import HydroDADataset


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"


class TestNoQueryStatsInNormalization:
    """Verify dataset does not compute or store target_query statistics.

    No-leakage declaration:
        - Dataset performs NO normalization by default
        - Even if normalization were added, target_query statistics
          would not be available (query is held-out)
    """

    def test_dataset_has_no_normalization_stats(self):
        """Dataset should not store any normalization statistics."""
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
        # No mean/std attributes on the dataset
        assert not hasattr(ds, "_mean"), "Dataset should not have _mean attribute"
        assert not hasattr(ds, "_std"), "Dataset should not have _std attribute"
        assert not hasattr(ds, "_norm_stats"), "Dataset should not have _norm_stats"
        ds.close()

    def test_source_train_uses_source_only_dates(self):
        """source_train dataset only uses source_train_dates from manifest."""
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
        # time_index should come from source_train_dates (2015-04-01 onwards)
        assert sample["time_index"] >= 0
        # date_str should be in source_train period
        year = int(sample["date_str"].split("-")[0])
        assert 2015 <= year <= 2020, (
            f"source_train sample has date_str {sample['date_str']} outside 2015-2020"
        )
        ds.close()

    def test_target_support_k0_uses_no_support(self):
        """K=0 target_support dataset has empty time indices."""
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
        ds.close()

    def test_target_query_contains_analysis_for_evaluation(self):
        """target_query samples contain analysis_surface/increment for offline eval."""
        ds = HydroDADataset(
            da_nc_path=f"{DATA_DIR}/DA.nc",
            region_masks_nc=REGION_MASKS,
            splits_json=SPLITS_JSON,
            target_region="US-R1",
            split_type="target_query",
            K=0, seed=0,
            freeze_manifest=MANIFEST,
        )
        sample = ds[0]
        # Must have analysis fields for evaluation
        assert "analysis_surface" in sample
        assert "analysis_rootzone" in sample
        assert "increment_surface" in sample
        assert "increment_rootzone" in sample
        ds.close()