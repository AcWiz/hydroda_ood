"""Tests for target_monthly_support_increment baseline.

No-leakage declaration:
    - Uses calendar month (not target_query labels) for grouping
    - metric_mask applied before mean computation
    - Only fit on target_support dates
"""

import numpy as np
import pytest

from hydroda.baselines.monthly_mean import TargetMonthlySupportIncrementBaseline


class TestTargetMonthlySupportIncrementBaseline:
    """Tests for TargetMonthlySupportIncrementBaseline."""

    def test_fit_predict_basic(self):
        """Fit on monthly samples, predict using correct month."""
        baseline = TargetMonthlySupportIncrementBaseline()

        samples = [
            # January samples
            {
                "increment_surface": np.ones((4, 4)) * 0.1,
                "increment_rootzone": np.ones((4, 4)) * 0.05,
                "metric_mask": np.ones((4, 4)),
                "month": 1,
            },
            {
                "increment_surface": np.ones((4, 4)) * 0.15,
                "increment_rootzone": np.ones((4, 4)) * 0.08,
                "metric_mask": np.ones((4, 4)),
                "month": 1,
            },
            # July samples
            {
                "increment_surface": np.ones((4, 4)) * 0.3,
                "increment_rootzone": np.ones((4, 4)) * 0.2,
                "metric_mask": np.ones((4, 4)),
                "month": 7,
            },
            {
                "increment_surface": np.ones((4, 4)) * 0.35,
                "increment_rootzone": np.ones((4, 4)) * 0.25,
                "metric_mask": np.ones((4, 4)),
                "month": 7,
            },
        ]
        baseline.fit(samples)

        # January prediction
        pred_jan = baseline.predict({
            "forecast_surface": np.ones((4, 4)) * 0.5,
            "forecast_rootzone": np.ones((4, 4)) * 0.4,
            "month": 1,
        })
        # Mean of [0.1, 0.15] = 0.125, mean of [0.05, 0.08] = 0.065
        np.testing.assert_allclose(pred_jan["pred_increment_surface"], 0.125, rtol=1e-5)
        np.testing.assert_allclose(pred_jan["pred_increment_rootzone"], 0.065, rtol=1e-5)

        # July prediction
        pred_jul = baseline.predict({
            "forecast_surface": np.ones((4, 4)) * 0.5,
            "forecast_rootzone": np.ones((4, 4)) * 0.4,
            "month": 7,
        })
        # Mean of [0.3, 0.35] = 0.325, mean of [0.2, 0.25] = 0.225
        np.testing.assert_allclose(pred_jul["pred_increment_surface"], 0.325, rtol=1e-5)
        np.testing.assert_allclose(pred_jul["pred_increment_rootzone"], 0.225, rtol=1e-5)

    def test_month_12_wraparound(self):
        """December uses 12 as key, not 0."""
        baseline = TargetMonthlySupportIncrementBaseline()
        samples = [
            {
                "increment_surface": np.ones((4, 4)) * 0.5,
                "increment_rootzone": np.ones((4, 4)) * 0.3,
                "metric_mask": np.ones((4, 4)),
                "month": 12,
            },
        ]
        baseline.fit(samples)

        pred = baseline.predict({
            "forecast_surface": np.ones((4, 4)) * 0.5,
            "forecast_rootzone": np.ones((4, 4)) * 0.4,
            "month": 12,
        })
        np.testing.assert_allclose(pred["pred_increment_surface"], 0.5, rtol=1e-5)

    def test_missing_month_defaults_to_zero(self):
        """If a month has no samples, predict 0 for that month."""
        baseline = TargetMonthlySupportIncrementBaseline()
        samples = [
            {
                "increment_surface": np.ones((4, 4)) * 0.1,
                "increment_rootzone": np.ones((4, 4)) * 0.05,
                "metric_mask": np.ones((4, 4)),
                "month": 3,  # Only March data
            },
        ]
        baseline.fit(samples)

        # February has no data → should default to 0
        pred = baseline.predict({
            "forecast_surface": np.ones((4, 4)) * 0.5,
            "forecast_rootzone": np.ones((4, 4)) * 0.4,
            "month": 2,
        })
        np.testing.assert_allclose(pred["pred_increment_surface"], 0.0, atol=1e-6)
        np.testing.assert_allclose(pred["pred_increment_rootzone"], 0.0, atol=1e-6)

    def test_monthly_mean_respects_mask(self):
        """Mean computed only over valid pixels."""
        baseline = TargetMonthlySupportIncrementBaseline()

        mask = np.array([
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        samples = [
            {
                "increment_surface": np.ones((4, 4)) * 0.1,
                "increment_rootzone": np.ones((4, 4)) * 0.05,
                "metric_mask": mask,
                "month": 6,
            },
        ]
        baseline.fit(samples)
        assert baseline._monthly_mean_surface[6] == 0.1
        assert baseline._monthly_mean_rootzone[6] == 0.05

    def test_predict_without_fit_raises(self):
        """predict() before fit() raises RuntimeError."""
        baseline = TargetMonthlySupportIncrementBaseline()
        sample = {
            "forecast_surface": np.ones((4, 4)),
            "forecast_rootzone": np.ones((4, 4)),
            "month": 6,
        }
        with pytest.raises(RuntimeError, match="Must call fit"):
            baseline.predict(sample)

    def test_output_shapes_match_forecast(self):
        """pred_increment shapes match forecast shapes."""
        baseline = TargetMonthlySupportIncrementBaseline()
        samples = [
            {
                "increment_surface": np.ones((8, 12)) * 0.1,
                "increment_rootzone": np.ones((8, 12)) * 0.05,
                "metric_mask": np.ones((8, 12)),
                "month": 6,
            },
        ]
        baseline.fit(samples)
        sample = {
            "forecast_surface": np.ones((8, 12)) * 0.5,
            "forecast_rootzone": np.ones((8, 12)) * 0.4,
            "month": 6,
        }
        pred = baseline.predict(sample)
        assert pred["pred_increment_surface"].shape == (8, 12)
        assert pred["pred_increment_rootzone"].shape == (8, 12)
        assert pred["pred_analysis_surface"].shape == (8, 12)
        assert pred["pred_analysis_rootzone"].shape == (8, 12)

    def test_all_12_months_initialized(self):
        """All 12 months have an entry after fit."""
        baseline = TargetMonthlySupportIncrementBaseline()
        samples = [
            {
                "increment_surface": np.ones((4, 4)) * 0.1,
                "increment_rootzone": np.ones((4, 4)) * 0.05,
                "metric_mask": np.ones((4, 4)),
                "month": 1,
            },
        ]
        baseline.fit(samples)
        assert len(baseline._monthly_mean_surface) == 12
        assert len(baseline._monthly_mean_rootzone) == 12
        for m in range(1, 13):
            assert m in baseline._monthly_mean_surface
            assert m in baseline._monthly_mean_rootzone
