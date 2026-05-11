"""Test dataset split loading — all 240 splits from US_loro_kdate_splits.json."""

import json
import pytest
from hydroda.data.dataset import HydroDADataset


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"

REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
K_VALUES = [0, 4, 12]
SEEDS = [0, 1, 2]


class TestDatasetSplitLoading:
    """Load all 54 splits and verify structural properties."""

    @pytest.fixture
    def splits_data(self):
        with open(SPLITS_JSON) as f:
            return json.load(f)

    def test_total_split_count(self, splits_data):
        assert len(splits_data["splits"]) == 54

    def test_all_splits_have_exact_time_indices(self, splits_data):
        """time_indices are exact integers; source_train may have duplicates (multiple obs/day)."""
        for split in splits_data["splits"]:
            for date_list_key in ["source_train_dates", "target_support_dates", "target_query_dates"]:
                indices = [d["time_index"] for d in split[date_list_key]]
                # All integers
                assert all(isinstance(i, int) for i in indices), (
                    f"Split {split['target_region_id']}-K{split['K']}-S{split['seed']} "
                    f"{date_list_key} has non-integer time_index"
                )
                # source_train_dates may have duplicate time_indices (multiple obs/day at different times)
                # target_support_dates and target_query_dates should have no duplicates
                if date_list_key in ("target_support_dates", "target_query_dates"):
                    assert len(indices) == len(set(indices)), (
                        f"Split {split['target_region_id']}-K{split['K']}-S{split['seed']} "
                        f"{date_list_key} has duplicate time_indices"
                    )

    def test_source_train_cycle_count_positive(self, splits_data):
        for split in splits_data["splits"]:
            assert split["source_train_cycle_count"] > 0, (
                f"Split {split['target_region_id']}-K{split['K']}-S{split['seed']} "
                f"has source_train_cycle_count=0"
            )

    def test_target_query_cycle_count_positive(self, splits_data):
        for split in splits_data["splits"]:
            assert split["target_query_cycle_count"] > 0, (
                f"Split {split['target_region_id']}-K{split['K']}-S{split['seed']} "
                f"has target_query_cycle_count=0"
            )

    def test_k0_has_empty_target_support(self, splits_data):
        for split in splits_data["splits"]:
            if split["K"] == 0:
                assert len(split["target_support_dates"]) == 0, (
                    f"K=0 split {split['target_region_id']} has non-empty support_dates"
                )

    @pytest.mark.parametrize("K", K_VALUES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_smoke_load_all_combinations(self, K, seed, splits_data):
        """Smoke test: iterate all 54 combinations of region x K x seed."""
        for target_region in REGIONS:
            ds = HydroDADataset(
                da_nc_path=f"{DATA_DIR}/DA.nc",
                region_masks_nc=REGION_MASKS,
                splits_json=SPLITS_JSON,
                target_region=target_region,
                split_type="source_train",
                K=K, seed=seed,
                freeze_manifest=MANIFEST,
            )
            assert len(ds) > 0
            ds.close()