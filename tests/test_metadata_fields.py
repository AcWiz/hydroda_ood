"""Test metadata fields in sample dict."""

import pytest
from hydroda.data.dataset import HydroDADataset


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"


class TestMetadataFields:
    """Verify metadata fields are correctly populated."""

    def test_source_train_active_region_ids_count(self):
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
        assert len(sample["active_region_ids"]) == 5, (
            f"source_train should have 5 active_region_ids, got {len(sample['active_region_ids'])}"
        )
        ds.close()

    def test_source_train_active_region_ids_excludes_target(self):
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
        assert "US-R1" not in sample["active_region_ids"]
        ds.close()

    def test_target_support_active_region_ids_single(self):
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
        assert len(sample["active_region_ids"]) == 1
        assert sample["active_region_ids"][0] == "US-R1"
        ds.close()

    def test_target_query_active_region_ids_single(self):
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
        assert len(sample["active_region_ids"]) == 1
        assert sample["active_region_ids"][0] == "US-R1"
        ds.close()

    def test_split_id_format(self):
        ds = HydroDADataset(
            da_nc_path=f"{DATA_DIR}/DA.nc",
            region_masks_nc=REGION_MASKS,
            splits_json=SPLITS_JSON,
            target_region="US-R1",
            split_type="source_train",
            K=4, seed=3,
            freeze_manifest=MANIFEST,
        )
        sample = ds[0]
        assert sample["split_id"] == "US-R1-K4-S3-source_train"
        ds.close()

    def test_regime_id_dryland_sparse(self):
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
        assert sample["regime_id"] == "dryland_sparse_vegetation"
        ds.close()

    def test_country_id_us(self):
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
        assert sample["country_id"] == "US"
        assert sample["target_region_id"] == "US-R1"
        ds.close()

    def test_split_role_values(self):
        for split_type, expected_role in [
            ("source_train", "source_train"),
            ("target_support", "target_support"),
            ("target_query", "target_query"),
        ]:
            K_val = 0
            ds = HydroDADataset(
                da_nc_path=f"{DATA_DIR}/DA.nc",
                region_masks_nc=REGION_MASKS,
                splits_json=SPLITS_JSON,
                target_region="US-R1",
                split_type=split_type,
                K=K_val,
                seed=0,
                freeze_manifest=MANIFEST,
            )
            if len(ds) == 0:
                ds.close()
                continue
            sample = ds[0]
            assert sample["split_role"] == expected_role, (
                f"split_role={sample['split_role']!r} != expected {expected_role!r}"
            )
            ds.close()

    def test_K_and_seed_in_sample(self):
        ds = HydroDADataset(
            da_nc_path=f"{DATA_DIR}/DA.nc",
            region_masks_nc=REGION_MASKS,
            splits_json=SPLITS_JSON,
            target_region="US-R2",
            split_type="source_train",
            K=12, seed=7,
            freeze_manifest=MANIFEST,
        )
        sample = ds[0]
        assert sample["K"] == 12
        assert sample["seed"] == 7
        ds.close()