"""Test dataset split loading for target-full-train split manifests."""

import json
from pathlib import Path

import pytest
from hydroda.data.dataset import HydroDADataset


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_target_train_splits.json"
MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"

REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
SEEDS = [0]


class TestDatasetSplitLoading:
    """Load split manifests and verify structural properties."""

    @pytest.fixture
    def splits_data(self):
        pytest.importorskip("xarray")
        if not Path(SPLITS_JSON).exists():
            pytest.skip(f"Splits file not found: {SPLITS_JSON}")
        with open(SPLITS_JSON) as f:
            return json.load(f)

    def test_total_split_count(self, splits_data):
        assert len(splits_data["splits"]) >= len(REGIONS)

    def test_all_splits_have_exact_time_indices(self, splits_data):
        """time_indices are exact integers; source_train may have duplicates (multiple obs/day)."""
        for split in splits_data["splits"]:
            for date_list_key in ["source_train_dates", "target_train_dates", "target_eval_dates"]:
                indices = [d["time_index"] for d in split[date_list_key]]
                # All integers
                assert all(isinstance(i, int) for i in indices), (
                    f"Split {split['target_region_id']}-{split.get('adaptation_setting')}-S{split['seed']} "
                    f"{date_list_key} has non-integer time_index"
                )
                # source_train_dates may have duplicate time_indices (multiple obs/day at different times)
                # target_support_dates and target_query_dates should have no duplicates
                if date_list_key in ("target_train_dates", "target_eval_dates"):
                    assert len(indices) == len(set(indices)), (
                        f"Split {split['target_region_id']}-{split.get('adaptation_setting')}-S{split['seed']} "
                        f"{date_list_key} has duplicate time_indices"
                    )

    def test_source_train_cycle_count_positive(self, splits_data):
        for split in splits_data["splits"]:
            assert split["source_train_cycle_count"] > 0, (
                f"Split {split['target_region_id']}-{split.get('adaptation_setting')}-S{split['seed']} "
                f"has source_train_cycle_count=0"
            )

    def test_target_eval_cycle_count_positive(self, splits_data):
        for split in splits_data["splits"]:
            assert split["target_eval_cycle_count"] > 0, (
                f"Split {split['target_region_id']}-{split.get('adaptation_setting')}-S{split['seed']} "
                f"has target_eval_cycle_count=0"
            )

    def test_target_full_train_has_target_train_dates(self, splits_data):
        for split in splits_data["splits"]:
            if split.get("adaptation_setting") == "target_full_train":
                assert split.get("K") is None
                assert len(split["target_train_dates"]) > 0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_smoke_load_all_combinations(self, seed, splits_data):
        """Smoke test: iterate all target regions for the main adaptation setting."""
        for target_region in REGIONS:
            ds = HydroDADataset(
                da_nc_path=f"{DATA_DIR}/DA.nc",
                region_masks_nc=REGION_MASKS,
                splits_json=SPLITS_JSON,
                target_region=target_region,
                split_type="source_train",
                K=None,
                seed=seed,
                adaptation_setting="target_full_train",
                freeze_manifest=MANIFEST,
            )
            assert len(ds) > 0
            ds.close()
