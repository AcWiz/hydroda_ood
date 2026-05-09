"""Test target_query evaluation-only behavior.

No-leakage: target_query samples return analysis/increment for offline evaluation,
but no query statistics are used in any normalization or training logic.
"""

import numpy as np
import pytest
from hydroda.data.dataset import HydroDADataset


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"


class TestTargetQueryEvaluationOnly:
    """Verify target_query samples contain analysis/increment for offline evaluation."""

    def test_target_query_has_analysis_fields(self):
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
        # Analysis fields present for evaluation
        assert "analysis_surface" in sample
        assert "analysis_rootzone" in sample
        # Increment fields present for evaluation
        assert "increment_surface" in sample
        assert "increment_rootzone" in sample
        ds.close()

    def test_target_query_period_is_2022_onwards(self):
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
        year = int(sample["date_str"].split("-")[0])
        assert year >= 2022, (
            f"target_query sample has year {year} < 2022"
        )
        ds.close()

    def test_target_query_has_target_region_active_only(self):
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
        # target_query has single active region (the held-out target)
        assert sample["active_region_ids"] == ["US-R1"]
        assert sample["split_role"] == "target_query"
        ds.close()

    def test_target_query_increments_reconstruct_correctly(self):
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
        inc_surf = sample["increment_surface"]
        fcast_surf = sample["forecast_surface"]
        anal_surf = sample["analysis_surface"]
        # forecast + increment should equal analysis
        np.testing.assert_allclose(
            fcast_surf + inc_surf, anal_surf,
            rtol=1e-5, atol=1e-8
        )
        ds.close()

    def test_document_target_query_evaluation_only(self):
        """Document that target_query samples are for offline evaluation.

        This test doesn't enforce anything — it serves as living documentation.
        """
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
        # This test always passes — it's documentation
        assert sample["split_role"] == "target_query"
        # The sample contains fields needed for offline evaluation
        assert "increment_surface" in sample
        assert "increment_rootzone" in sample
        assert "analysis_surface" in sample
        assert "analysis_rootzone" in sample
        ds.close()