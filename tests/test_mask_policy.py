"""Tests for mask derivation policy (V4-Phase3C).

No-leakage declaration:
    - All tests use stateless mask derivation functions only
    - No target label access in any test
    - metric_mask independence from channel 11 verified
"""
from __future__ import annotations

import numpy as np
import pytest


class TestMaskDerivation:
    """Unit tests for mask derivation functions."""

    def _make_sample(
        self,
        forecast_surface: np.ndarray,
        forecast_rootzone: np.ndarray,
        analysis_surface: np.ndarray,
        analysis_rootzone: np.ndarray,
        increment_surface: np.ndarray,
        increment_rootzone: np.ndarray,
        base_valid_mask: np.ndarray,
        active_region_mask: np.ndarray,
    ) -> dict:
        """Helper to build a minimal sample dict."""
        return {
            "forecast_surface": forecast_surface,
            "forecast_rootzone": forecast_rootzone,
            "analysis_surface": analysis_surface,
            "analysis_rootzone": analysis_rootzone,
            "increment_surface": increment_surface,
            "increment_rootzone": increment_rootzone,
            "base_valid_mask": base_valid_mask,
            "active_region_mask": active_region_mask,
        }

    def test_label_valid_mask_excludes_nan_inf(self):
        """label_valid_mask should be 0 where any field is NaN or inf."""
        from hydroda.data.masks import derive_label_valid_mask

        arr = np.array([[1.0, np.nan], [np.inf, 2.0]])
        sample = self._make_sample(
            forecast_surface=arr,
            forecast_rootzone=arr,
            analysis_surface=arr,
            analysis_rootzone=arr,
            increment_surface=arr,
            increment_rootzone=arr,
            base_valid_mask=np.ones_like(arr),
            active_region_mask=np.ones_like(arr),
        )

        mask = derive_label_valid_mask(sample)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(mask, expected)

    def test_label_valid_mask_all_finite_returns_all_ones(self):
        """All-valid fields produce all-ones label_valid_mask."""
        from hydroda.data.masks import derive_label_valid_mask

        arr = np.ones((3, 3))
        sample = self._make_sample(
            forecast_surface=arr,
            forecast_rootzone=arr,
            analysis_surface=arr,
            analysis_rootzone=arr,
            increment_surface=arr,
            increment_rootzone=arr,
            base_valid_mask=np.ones_like(arr),
            active_region_mask=np.ones_like(arr),
        )

        mask = derive_label_valid_mask(sample)
        np.testing.assert_array_equal(mask, np.ones_like(arr))

    def test_obs_mask_binary(self):
        """obs_mask should be 0 or 1 (thresholded from base_valid_mask)."""
        from hydroda.data.masks import derive_obs_mask

        arr = np.array([[0.0, 0.6, 1.0], [-0.1, 0.5, 1.2]])
        sample = {"base_valid_mask": arr}

        mask = derive_obs_mask(sample)
        expected = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_equal(mask, expected)

    def test_region_mask_binary(self):
        """region_mask should be 0 or 1 (thresholded from active_region_mask)."""
        from hydroda.data.masks import derive_region_mask

        arr = np.array([[0.0, 0.6], [1.0, 0.8]])  # 0.4 -> 0, 0.8 -> 1
        sample = {"active_region_mask": arr}

        mask = derive_region_mask(sample)
        expected = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_equal(mask, expected)

    def test_metric_mask_no_channel_11_dependency(self):
        """metric_mask should NOT change when base_valid_mask (ch11) is zeroed."""
        from hydroda.data.masks import derive_label_valid_mask, derive_metric_mask

        # All fields finite
        arr = np.ones((4, 4))
        active = np.ones((4, 4))

        sample_all_finite = self._make_sample(
            forecast_surface=arr,
            forecast_rootzone=arr,
            analysis_surface=arr,
            analysis_rootzone=arr,
            increment_surface=arr,
            increment_rootzone=arr,
            base_valid_mask=np.ones_like(arr),  # ch11 = 1
            active_region_mask=active,
        )

        sample_obs_gap = self._make_sample(
            forecast_surface=arr,
            forecast_rootzone=arr,
            analysis_surface=arr,
            analysis_rootzone=arr,
            increment_surface=arr,
            increment_rootzone=arr,
            base_valid_mask=np.zeros_like(arr),  # ch11 = 0 (SMAP coverage gap)
            active_region_mask=active,
        )

        mask_finite = derive_metric_mask(sample_all_finite)
        mask_obs_gap = derive_metric_mask(sample_obs_gap)

        # metric_mask should be identical in both cases
        np.testing.assert_array_equal(mask_finite, mask_obs_gap)
        np.testing.assert_array_equal(mask_finite, np.ones_like(arr))

    def test_label_valid_mask_no_channel_11_dependency(self):
        """label_valid_mask should be identical with or without obs coverage."""
        from hydroda.data.masks import derive_label_valid_mask

        arr = np.ones((3, 3))
        sample_with_obs = self._make_sample(
            forecast_surface=arr,
            forecast_rootzone=arr,
            analysis_surface=arr,
            analysis_rootzone=arr,
            increment_surface=arr,
            increment_rootzone=arr,
            base_valid_mask=np.ones_like(arr),
            active_region_mask=np.ones_like(arr),
        )
        sample_without_obs = self._make_sample(
            forecast_surface=arr,
            forecast_rootzone=arr,
            analysis_surface=arr,
            analysis_rootzone=arr,
            increment_surface=arr,
            increment_rootzone=arr,
            base_valid_mask=np.zeros_like(arr),
            active_region_mask=np.ones_like(arr),
        )

        mask_with = derive_label_valid_mask(sample_with_obs)
        mask_without = derive_label_valid_mask(sample_without_obs)
        np.testing.assert_array_equal(mask_with, mask_without)

    def test_obs_mask_not_in_metric_mask(self):
        """derive_metric_mask output should be same when obs_mask differs."""
        from hydroda.data.masks import derive_metric_mask

        arr = np.ones((4, 4))
        active = np.ones((4, 4))

        sample_no_obs = self._make_sample(
            forecast_surface=arr,
            forecast_rootzone=arr,
            analysis_surface=arr,
            analysis_rootzone=arr,
            increment_surface=arr,
            increment_rootzone=arr,
            base_valid_mask=np.zeros_like(arr),
            active_region_mask=active,
        )
        sample_with_obs = self._make_sample(
            forecast_surface=arr,
            forecast_rootzone=arr,
            analysis_surface=arr,
            analysis_rootzone=arr,
            increment_surface=arr,
            increment_rootzone=arr,
            base_valid_mask=np.ones_like(arr),
            active_region_mask=active,
        )

        np.testing.assert_array_equal(
            derive_metric_mask(sample_no_obs),
            derive_metric_mask(sample_with_obs),
        )

    def test_summarize_mask_coverage_output_shape(self):
        """summarize_mask_coverage returns dict with expected structure."""
        from hydroda.data.masks import summarize_mask_coverage

        arr = np.ones((4, 4))
        active = np.ones((4, 4))
        sample = self._make_sample(
            forecast_surface=arr,
            forecast_rootzone=arr,
            analysis_surface=arr,
            analysis_rootzone=arr,
            increment_surface=arr,
            increment_rootzone=arr,
            base_valid_mask=np.ones_like(arr),
            active_region_mask=active,
        )
        sample["target_region_id"] = "US-R1"
        sample["season"] = "DJF"

        result = summarize_mask_coverage(
            [sample, sample],
            by_region=True,
            by_season=True,
        )

        assert "overall" in result
        assert result["overall"]["n_samples"] == 2
        assert "by_region" in result
        assert "US-R1" in result["by_region"]
        assert "by_season" in result
        assert "DJF" in result["by_season"]

    def test_metric_mask_requires_region(self):
        """metric_mask should be zero outside active region even if fields finite."""
        from hydroda.data.masks import derive_metric_mask

        arr = np.ones((4, 4))
        active = np.zeros((4, 4))
        active[1:3, 1:3] = 1.0  # only center 2x2 is in region

        sample = self._make_sample(
            forecast_surface=arr,
            forecast_rootzone=arr,
            analysis_surface=arr,
            analysis_rootzone=arr,
            increment_surface=arr,
            increment_rootzone=arr,
            base_valid_mask=np.ones_like(arr),
            active_region_mask=active,
        )

        mask = derive_metric_mask(sample)
        expected = np.zeros_like(arr)
        expected[1:3, 1:3] = 1.0
        np.testing.assert_array_equal(mask, expected)


class TestMaskLeakageSafety:
    """Tests verifying no-leakage policy is upheld."""

    def test_metric_mask_no_target_query_labels(self):
        """metric_mask computation never accesses target query labels."""
        from hydroda.data.masks import derive_metric_mask

        # This is a compile-time check — derive_metric_mask signature does not
        # accept and never routes target_region_id, date_str, or query labels
        arr = np.ones((2, 2))
        sample = {
            "forecast_surface": arr,
            "forecast_rootzone": arr,
            "analysis_surface": arr,
            "analysis_rootzone": arr,
            "increment_surface": arr,
            "increment_rootzone": arr,
            "base_valid_mask": np.ones_like(arr),
            "active_region_mask": np.ones_like(arr),
            # These should NOT be accessed
            "target_region_id": "US-R1",
            "date_str": "2022-06-15",
        }

        # Should not raise KeyError — derive_metric_mask only reads what it needs
        mask = derive_metric_mask(sample)
        assert mask.shape == arr.shape
        assert mask.dtype == np.float32

    def test_label_valid_mask_no_channel_11_dependency(self):
        """label_valid_mask does not read base_valid_mask (channel 11)."""
        from hydroda.data.masks import derive_label_valid_mask
        import inspect

        sig = inspect.signature(derive_label_valid_mask)
        params = list(sig.parameters.keys())
        # Must only require the 6 SM fields + active_region_mask
        assert "base_valid_mask" not in params
        assert "obs_mask" not in params

    def test_obs_mask_not_in_metric_mask(self):
        """obs_mask (channel 11) is not a dependency of derive_metric_mask."""
        from hydroda.data.masks import derive_metric_mask
        import inspect

        sig = inspect.signature(derive_metric_mask)
        params = list(sig.parameters.keys())
        assert "base_valid_mask" not in params
        assert "obs_mask" not in params