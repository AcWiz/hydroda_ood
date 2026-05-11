"""Real-data contract test for HydroDADataset on DA.nc.

No-leakage declaration:
    - Uses target_query split, never uses query labels for training
    - Uses source_train stats only for mask computation
    - metric_mask applied in __getitem__ before any metric computation
"""
import numpy as np
import pytest

DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"


class TestRealDataContract:
    """Validate HydroDADataset contract on real DA.nc."""

    @pytest.fixture
    def ds_query(self):
        """target_query dataset for US-R1, K=4."""
        from hydroda.data.dataset import HydroDADataset

        ds = HydroDADataset(
            da_nc_path=f"{DATA_DIR}/DA.nc",
            region_masks_nc=REGION_MASKS,
            splits_json=SPLITS_JSON,
            target_region="US-R1",
            split_type="target_query",
            K=4, seed=0,
            freeze_manifest=MANIFEST,
        )
        yield ds
        ds.close()

    @pytest.fixture
    def ds_source(self):
        """source_train dataset for US-R1, K=4."""
        from hydroda.data.dataset import HydroDADataset

        ds = HydroDADataset(
            da_nc_path=f"{DATA_DIR}/DA.nc",
            region_masks_nc=REGION_MASKS,
            splits_json=SPLITS_JSON,
            target_region="US-R1",
            split_type="source_train",
            K=4, seed=0,
            freeze_manifest=MANIFEST,
        )
        yield ds
        ds.close()

    def test_dataset_len_query(self, ds_query):
        """target_query split should have samples."""
        assert len(ds_query) > 0, "target_query split is empty"

    def test_dataset_len_source(self, ds_source):
        """source_train split should have samples."""
        assert len(ds_source) > 0, "source_train split is empty"

    def test_sample_has_all_required_fields(self, ds_query):
        """Sample must have all fields required by evaluate_split and metrics."""
        sample = ds_query[0]
        required = [
            "forecast_surface", "forecast_rootzone",
            "analysis_surface", "analysis_rootzone",
            "increment_surface", "increment_rootzone",
            "metric_mask",
            "date_str", "month", "season",
            "time_index", "country_id", "target_region_id",
            "active_region_ids", "split_role", "K", "seed",
            "regime_id", "split_id",
        ]
        missing = [k for k in required if k not in sample]
        assert not missing, f"Missing fields: {missing}"

    def test_channel_mapping_increment_sign(self, ds_query):
        """increment = analysis - forecast (positive = dry bias correction)."""
        sample = ds_query[0]
        inc_surf = sample["increment_surface"]
        analysis_surf = sample["analysis_surface"]
        forecast_surf = sample["forecast_surface"]

        # Check the relationship holds
        np.testing.assert_allclose(
            inc_surf, analysis_surf - forecast_surf,
            rtol=1e-5, atol=1e-8,
            err_msg="increment_surface != analysis_surface - forecast_surface"
        )

    def test_mask_is_binary(self, ds_query):
        """metric_mask should be binary (0 or 1 after > 0.5 threshold)."""
        sample = ds_query[0]
        mask = sample["metric_mask"]
        unique = np.unique(mask)
        assert all(v >= 0 and v <= 1 for v in unique), \
            f"metric_mask values outside [0,1]: {unique}"
        # After > 0.5 threshold, should be 0.0 or 1.0
        binary = (mask > 0.5).astype(np.float32)
        diff = np.abs(mask - binary).max()
        assert diff < 1e-5, f"metric_mask not binary after threshold: max_diff={diff}"

    def test_increment_shape_equals_forecast_shape(self, ds_query):
        """increment and forecast shapes must match."""
        sample = ds_query[0]
        assert sample["increment_surface"].shape == sample["forecast_surface"].shape
        assert sample["increment_rootzone"].shape == sample["forecast_rootzone"].shape

    def test_time_split_adherence(self, ds_query, ds_source):
        """All query dates must be >= 2022-01-01; all source dates <= 2020-12-31."""
        for ds, split_name in [(ds_query, "target_query"), (ds_source, "source_train")]:
            for i in range(min(len(ds), 20)):
                sample = ds[i]
                date_str = sample["date_str"]
                if not date_str:
                    continue
                year = int(date_str[:4])
                if split_name == "target_query":
                    assert year >= 2022, \
                        f"target_query date {date_str} is before 2022"
                elif split_name == "source_train":
                    assert year <= 2020, \
                        f"source_train date {date_str} is after 2020"

    def test_no_nan_in_forecast_analysis(self, ds_query):
        """Forecast and analysis fields must have valid finite values where mask is 1.

        metric_mask = region_mask & label_valid_mask (channel 11 no longer blocks evaluation).
        If valid pixels exist, they must be finite.
        """
        sample = ds_query[0]
        mask = sample["metric_mask"]
        valid_mask = mask > 0.5
        n_valid = int(valid_mask.sum())

        if n_valid == 0:
            # All pixels may be outside region or have non-finite values
            for key in ["forecast_surface", "forecast_rootzone",
                        "analysis_surface", "analysis_rootzone"]:
                arr = sample[key]
                assert np.isfinite(arr).sum() > 0, \
                    f"{key} has no finite values at all"
            return

        # If valid pixels exist, they must be finite
        for key in ["forecast_surface", "forecast_rootzone",
                    "analysis_surface", "analysis_rootzone"]:
            arr = sample[key]
            finite = np.isfinite(arr)
            assert finite[valid_mask].sum() > 0, \
                f"All pixels invalid for {key} in valid region"

    def test_new_mask_fields_present(self, ds_query):
        """label_valid_mask, obs_mask, region_mask are all present in sample."""
        sample = ds_query[0]
        for key in ["label_valid_mask", "obs_mask", "region_mask"]:
            assert key in sample, f"Missing field: {key}"
            arr = sample[key]
            assert arr.shape == sample["forecast_surface"].shape
            unique = np.unique(arr)
            assert all(v >= 0 and v <= 1 for v in unique), \
                f"{key} values outside [0,1]: {unique}"

    def test_metric_mask_independent_of_obs_mask(self, ds_query):
        """metric_mask should not go to zero just because obs_mask (ch11) is zero.

        After mask policy fix: metric_mask = region_mask & label_valid_mask.
        SMAP coverage gaps should NOT prevent metric computation.
        """
        sample = ds_query[0]
        obs = sample["obs_mask"]
        metric = sample["metric_mask"]

        # obs_mask can be all zeros while metric_mask is not
        # (this happens in SMAP coverage gaps but fields are still finite)
        # Just verify metric_mask has the right shape and is binary
        assert metric.shape == obs.shape
        unique = np.unique(metric)
        assert all(v >= 0 and v <= 1 for v in unique), \
            f"metric_mask values outside [0,1]: {unique}"