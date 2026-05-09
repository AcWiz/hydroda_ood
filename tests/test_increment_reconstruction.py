"""Test increment reconstruction: forecast + increment = analysis."""

import numpy as np
import pytest
from hydroda.data.dataset import HydroDADataset


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"


class TestIncrementReconstruction:
    """Verify forecast + increment = analysis for surface and rootzone."""

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

    @pytest.mark.parametrize("idx", [0, 100, 500])
    def test_surface_reconstruction(self, ds, idx):
        sample = ds[idx]
        rec = sample["forecast_surface"] + sample["increment_surface"]
        np.testing.assert_allclose(
            rec, sample["analysis_surface"],
            rtol=1e-5, atol=1e-8,
            err_msg=f"Surface reconstruction failed at idx={idx}"
        )

    @pytest.mark.parametrize("idx", [0, 100, 500])
    def test_rootzone_reconstruction(self, ds, idx):
        sample = ds[idx]
        rec = sample["forecast_rootzone"] + sample["increment_rootzone"]
        np.testing.assert_allclose(
            rec, sample["analysis_rootzone"],
            rtol=1e-5, atol=1e-8,
            err_msg=f"Rootzone reconstruction failed at idx={idx}"
        )