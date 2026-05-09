"""Tests for source_mean and target_support_mean baselines.

No-leakage declaration:
    - Tests use synthetic data with known structure
    - No real DA.nc data required for unit tests
    - metric_mask properly applied before mean computation
"""

import numpy as np
import pytest

from hydroda.baselines.mean_increment import (
    SourceMeanIncrementBaseline,
    TargetSupportMeanIncrementBaseline,
)
from hydroda.metrics.skill import analysis_skill_vs_forecast, increment_rmse


class TestSourceMeanIncrementBaseline:
    """Tests for SourceMeanIncrementBaseline."""

    def test_fit_predict_basic(self):
        """Fit on samples, predict on new sample."""
        baseline = SourceMeanIncrementBaseline()

        # Synthetic source_train samples
        samples = [
            {
                "increment_surface": np.ones((4, 4)) * 0.1,
                "increment_rootzone": np.ones((4, 4)) * 0.05,
                "metric_mask": np.ones((4, 4)),
            },
            {
                "increment_surface": np.ones((4, 4)) * 0.2,
                "increment_rootzone": np.ones((4, 4)) * 0.1,
                "metric_mask": np.ones((4, 4)),
            },
        ]
        baseline.fit(samples)

        # Predict on a new sample
        sample = {
            "forecast_surface": np.ones((4, 4)) * 0.5,
            "forecast_rootzone": np.ones((4, 4)) * 0.4,
        }
        pred = baseline.predict(sample)

        # Mean of [0.1, 0.2] = 0.15, mean of [0.05, 0.1] = 0.075
        np.testing.assert_allclose(
            pred["pred_increment_surface"], 0.15, rtol=1e-5
        )
        np.testing.assert_allclose(
            pred["pred_increment_rootzone"], 0.075, rtol=1e-5
        )
        # analysis = forecast + increment
        np.testing.assert_allclose(
            pred["pred_analysis_surface"], 0.5 + 0.15, rtol=1e-5
        )
        np.testing.assert_allclose(
            pred["pred_analysis_rootzone"], 0.4 + 0.075, rtol=1e-5
        )

    def test_fit_respects_metric_mask(self):
        """Mean computed only over metric_mask > 0.5 pixels."""
        baseline = SourceMeanIncrementBaseline()

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
            },
        ]
        baseline.fit(samples)
        # Only 4 valid pixels with value 0.1
        assert baseline._mean_inc_surface == 0.1

    def test_no_fit_data_raises(self):
        """No valid pixels raises ValueError."""
        baseline = SourceMeanIncrementBaseline()
        samples = [
            {
                "increment_surface": np.full((4, 4), np.nan),
                "increment_rootzone": np.full((4, 4), np.nan),
                "metric_mask": np.zeros((4, 4)),
            },
        ]
        with pytest.raises(ValueError, match="No valid pixels"):
            baseline.fit(samples)

    def test_predict_without_fit_raises(self):
        """predict() before fit() raises RuntimeError."""
        baseline = SourceMeanIncrementBaseline()
        sample = {
            "forecast_surface": np.ones((4, 4)),
            "forecast_rootzone": np.ones((4, 4)),
        }
        with pytest.raises(RuntimeError, match="Must call fit"):
            baseline.predict(sample)

    def test_output_shapes(self):
        """pred_increment shape matches forecast shape."""
        baseline = SourceMeanIncrementBaseline()
        samples = [
            {
                "increment_surface": np.ones((8, 12)) * 0.1,
                "increment_rootzone": np.ones((8, 12)) * 0.05,
                "metric_mask": np.ones((8, 12)),
            },
        ]
        baseline.fit(samples)
        sample = {
            "forecast_surface": np.ones((8, 12)) * 0.5,
            "forecast_rootzone": np.ones((8, 12)) * 0.4,
        }
        pred = baseline.predict(sample)
        assert pred["pred_increment_surface"].shape == (8, 12)
        assert pred["pred_increment_rootzone"].shape == (8, 12)
        assert pred["pred_analysis_surface"].shape == (8, 12)
        assert pred["pred_analysis_rootzone"].shape == (8, 12)

    def test_source_mean_no_target_labels_used(self):
        """SourceMeanIncrementBaseline doesn't access target_query labels."""
        baseline = SourceMeanIncrementBaseline()
        samples = [
            {
                "increment_surface": np.ones((4, 4)) * 0.1,
                "increment_rootzone": np.ones((4, 4)) * 0.05,
                "metric_mask": np.ones((4, 4)),
            },
        ]
        baseline.fit(samples)
        # No target_query labels in predict — only forecast
        sample = {
            "forecast_surface": np.ones((4, 4)) * 0.5,
            "forecast_rootzone": np.ones((4, 4)) * 0.4,
            # These should NOT be used
            "increment_surface": np.ones((4, 4)) * 999.0,
            "increment_rootzone": np.ones((4, 4)) * 999.0,
        }
        pred = baseline.predict(sample)
        # Should NOT use the 999 values
        np.testing.assert_allclose(pred["pred_increment_surface"], 0.1, rtol=1e-5)


class TestTargetSupportMeanIncrementBaseline:
    """Tests for TargetSupportMeanIncrementBaseline."""

    def test_fit_predict_basic(self):
        """Fit on target_support samples, predict on query sample."""
        baseline = TargetSupportMeanIncrementBaseline()

        samples = [
            {
                "increment_surface": np.ones((4, 4)) * 0.15,
                "increment_rootzone": np.ones((4, 4)) * 0.08,
                "metric_mask": np.ones((4, 4)),
            },
            {
                "increment_surface": np.ones((4, 4)) * 0.25,
                "increment_rootzone": np.ones((4, 4)) * 0.12,
                "metric_mask": np.ones((4, 4)),
            },
        ]
        baseline.fit(samples)

        sample = {
            "forecast_surface": np.ones((4, 4)) * 0.5,
            "forecast_rootzone": np.ones((4, 4)) * 0.4,
        }
        pred = baseline.predict(sample)

        # Mean of [0.15, 0.25] = 0.2, mean of [0.08, 0.12] = 0.1
        np.testing.assert_allclose(pred["pred_increment_surface"], 0.2, rtol=1e-5)
        np.testing.assert_allclose(pred["pred_increment_rootzone"], 0.1, rtol=1e-5)

    def test_fitted_attributes_set(self):
        """After fit, _mean_inc_surface and _mean_inc_rootzone are set."""
        baseline = TargetSupportMeanIncrementBaseline()
        samples = [
            {
                "increment_surface": np.ones((4, 4)) * 0.1,
                "increment_rootzone": np.ones((4, 4)) * 0.05,
                "metric_mask": np.ones((4, 4)),
            },
        ]
        baseline.fit(samples)
        assert baseline._fitted is True
        assert isinstance(baseline._mean_inc_surface, float)
        assert isinstance(baseline._mean_inc_rootzone, float)

    def test_same_api_as_source_mean(self):
        """Both baselines have the same predict() API."""
        for cls in [SourceMeanIncrementBaseline, TargetSupportMeanIncrementBaseline]:
            baseline = cls()
            samples = [
                {
                    "increment_surface": np.ones((4, 4)) * 0.1,
                    "increment_rootzone": np.ones((4, 4)) * 0.05,
                    "metric_mask": np.ones((4, 4)),
                },
            ]
            baseline.fit(samples)
            sample = {
                "forecast_surface": np.ones((4, 4)) * 0.5,
                "forecast_rootzone": np.ones((4, 4)) * 0.4,
            }
            pred = baseline.predict(sample)
            assert "pred_increment_surface" in pred
            assert "pred_increment_rootzone" in pred
            assert "pred_analysis_surface" in pred
            assert "pred_analysis_rootzone" in pred


class TestMeanBaselineIntegration:
    """Integration tests with metric computations on synthetic data."""

    def test_increment_rmse_reflects_mean_quality(self):
        """When true increment equals fitted mean, RMSE should be ~0."""
        baseline = SourceMeanIncrementBaseline()

        # All samples have true increment = 0.1
        true_inc = 0.1
        samples = [
            {
                "increment_surface": np.ones((10, 10)) * true_inc,
                "increment_rootzone": np.ones((10, 10)) * true_inc,
                "metric_mask": np.ones((10, 10)),
            },
        ]
        baseline.fit(samples)

        # Query sample with same increment
        query = {
            "forecast_surface": np.ones((10, 10)) * 0.5,
            "forecast_rootzone": np.ones((10, 10)) * 0.4,
        }
        pred = baseline.predict(query)

        inc_rmse = increment_rmse(
            pred["pred_increment_surface"],
            np.ones((10, 10)) * true_inc,
            np.ones((10, 10)),
        )
        # Use allclose since floating point precision means it's ~1.49e-09, not exactly 0
        np.testing.assert_allclose(inc_rmse, 0.0, atol=1e-6,
                                   err_msg="RMSE should be ~0 when prediction equals truth")

    def test_source_mean_forecast_skill_is_zero(self):
        """Forecast baseline (zero increment) has skill = 0 on non-trivial data."""
        from hydroda.baselines.forecast import ForecastBaseline
        baseline = ForecastBaseline()

        # Query with non-trivial data where increment is non-zero
        # forecast = 0.5, analysis = 0.55 (increment = 0.05)
        query = {
            "forecast_surface": np.ones((10, 10)) * 0.5,
            "forecast_rootzone": np.ones((10, 10)) * 0.4,
            "analysis_surface": np.ones((10, 10)) * 0.55,  # 0.05 increment
            "analysis_rootzone": np.ones((10, 10)) * 0.43,
            "increment_surface": np.ones((10, 10)) * 0.05,
            "increment_rootzone": np.ones((10, 10)) * 0.03,
            "metric_mask": np.ones((10, 10)),
        }
        pred = baseline.predict(query)

        skill = analysis_skill_vs_forecast(
            pred["pred_analysis_surface"],  # = forecast (no correction)
            query["analysis_surface"],  # true = forecast + 0.05
            query["forecast_surface"],
            np.ones((10, 10)),
        )
        # For forecast-only (zero increment), skill should be ~0
        # since pred == forecast, and skill = 1 - RMSE(pred)/RMSE(fcst) = 1 - 1 = 0
        assert not np.isnan(skill), "Skill should be defined (not nan)"
        assert abs(skill) < 0.01, f"Forecast baseline skill should be near 0, got {skill}"
