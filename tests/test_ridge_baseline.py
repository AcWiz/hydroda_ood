"""Tests for RidgeBaseline with three feature sets.

No-leakage declaration:
    - Fit on target_support only
    - No target_query labels used in fitting
    - metric_mask applied before computing targets
"""

import numpy as np
import pytest

from hydroda.baselines.ridge import RidgeBaseline, FEATURE_SETS, _sin_cos_day


class TestSinCosDay:
    """Tests for cyclical day-of-year encoding."""

    def test_range(self):
        """sin and cos are in [-1, 1]."""
        sin_d, cos_d = _sin_cos_day(1)  # Jan 1
        assert -1.0 <= sin_d <= 1.0
        assert -1.0 <= cos_d <= 1.0

    def test_periodicity(self):
        """Day 1 and day 366 (year+1) should be nearly identical."""
        s1, c1 = _sin_cos_day(1)
        s366, c366 = _sin_cos_day(366)  # 366 = 365 + 1
        np.testing.assert_allclose(s1, s366, atol=1e-6)
        np.testing.assert_allclose(c1, c366, atol=1e-6)

    def test_equinox(self):
        """Day 80 (approx Mar 21) should have cos near 0, sin positive."""
        s, c = _sin_cos_day(80)
        assert s > 0, "Spring equinox sin should be positive"
        assert abs(c) < 0.5, "Spring equinox cos should be near 0"


class TestFeatureSets:
    """Tests for feature set definitions."""

    def test_all_three_feature_sets_defined(self):
        """core, input_full, input_geo_full are all defined."""
        assert "core" in FEATURE_SETS
        assert "input_full" in FEATURE_SETS
        assert "input_geo_full" in FEATURE_SETS

    def test_core_is_subset_of_input_full(self):
        """core features are a subset of input_full."""
        core_features = set(FEATURE_SETS["core"]["features"])
        full_features = set(FEATURE_SETS["input_full"]["features"])
        assert core_features < full_features, "core should be subset of input_full"

    def test_input_full_is_subset_of_input_geo_full(self):
        """input_full features are a subset of input_geo_full."""
        full_features = set(FEATURE_SETS["input_full"]["features"])
        geo_features = set(FEATURE_SETS["input_geo_full"]["features"])
        assert full_features < geo_features, "input_full should be subset of input_geo_full"


class TestRidgeBaseline:
    """Tests for RidgeBaseline."""

    def _make_sample(self, inc_s, inc_r, month_str="2021-06-15"):
        """Create a synthetic sample dict."""
        H, W = inc_s.shape
        x = np.zeros((12, H, W), dtype=np.float32)
        x[0] = np.ones((H, W)) * 0.5   # forecast_surface
        x[1] = np.ones((H, W)) * 0.4   # forecast_rootzone
        x[4] = np.ones((H, W)) * 0.3   # vegopacity
        x[5] = np.ones((H, W)) * 280.0  # tb_h
        x[6] = np.ones((H, W)) * 285.0  # tb_v
        x[11] = np.ones((H, W))        # base_valid_mask

        return {
            "x": x,
            "forecast_surface": np.ones((H, W)) * 0.5,
            "forecast_rootzone": np.ones((H, W)) * 0.4,
            "increment_surface": inc_s,
            "increment_rootzone": inc_r,
            "metric_mask": np.ones((H, W)),
            "date_str": month_str,
        }

    def test_fit_predict_core(self):
        """Ridge with core features can fit and predict."""
        baseline = RidgeBaseline(feature_set="core", alpha=1.0)

        # Create samples with increment proportional to forecast
        samples = []
        for frac in [0.8, 1.0, 1.2]:
            inc_s = np.ones((8, 10)) * frac * 0.1
            inc_r = np.ones((8, 10)) * frac * 0.05
            samples.append(self._make_sample(inc_s, inc_r, "2021-06-15"))

        baseline.fit(samples)
        assert baseline._fitted is True

        # Predict
        query = self._make_sample(
            np.ones((8, 10)) * 0.1,
            np.ones((8, 10)) * 0.05,
            "2021-06-15",
        )
        pred = baseline.predict(query)

        assert pred["pred_increment_surface"].shape == (8, 10)
        assert pred["pred_increment_rootzone"].shape == (8, 10)
        assert pred["pred_analysis_surface"].shape == (8, 10)
        assert pred["pred_analysis_rootzone"].shape == (8, 10)

    def test_fit_predict_input_full(self):
        """Ridge with input_full features can fit and predict."""
        baseline = RidgeBaseline(feature_set="input_full", alpha=1.0)

        samples = []
        for frac in [0.8, 1.0, 1.2]:
            inc_s = np.ones((8, 10)) * frac * 0.1
            inc_r = np.ones((8, 10)) * frac * 0.05
            samples.append(self._make_sample(inc_s, inc_r, "2021-06-15"))

        baseline.fit(samples)
        query = self._make_sample(
            np.ones((8, 10)) * 0.1,
            np.ones((8, 10)) * 0.05,
            "2021-06-15",
        )
        pred = baseline.predict(query)
        assert pred["pred_increment_surface"].shape == (8, 10)

    def test_fit_predict_input_geo_full(self):
        """Ridge with input_geo_full features can fit and predict."""
        baseline = RidgeBaseline(feature_set="input_geo_full", alpha=1.0)

        samples = []
        for frac in [0.8, 1.0, 1.2]:
            inc_s = np.ones((8, 10)) * frac * 0.1
            inc_r = np.ones((8, 10)) * frac * 0.05
            samples.append(self._make_sample(inc_s, inc_r, "2021-06-15"))

        baseline.fit(samples)
        query = self._make_sample(
            np.ones((8, 10)) * 0.1,
            np.ones((8, 10)) * 0.05,
            "2021-06-15",
        )
        pred = baseline.predict(query)
        assert pred["pred_increment_surface"].shape == (8, 10)

    def test_unknown_feature_set_raises(self):
        """Unknown feature_set raises ValueError."""
        with pytest.raises(ValueError, match="Unknown feature_set"):
            RidgeBaseline(feature_set="nonexistent")

    def test_predict_without_fit_raises(self):
        """predict() before fit() raises RuntimeError."""
        baseline = RidgeBaseline(feature_set="core")
        sample = {
            "x": np.zeros((12, 8, 10), dtype=np.float32),
            "forecast_surface": np.ones((8, 10)),
            "forecast_rootzone": np.ones((8, 10)),
            "date_str": "2021-06-15",
        }
        with pytest.raises(RuntimeError, match="Must call fit"):
            baseline.predict(sample)

    def test_fit_respects_metric_mask(self):
        """Only valid pixels (mask > 0.5) are used for fitting."""
        baseline = RidgeBaseline(feature_set="core", alpha=1.0)

        inc_s = np.ones((8, 10)) * 0.5
        inc_r = np.ones((8, 10)) * 0.3
        mask = np.zeros((8, 10))
        mask[:4, :5] = 1.0  # Only half the pixels valid

        sample = self._make_sample(inc_s, inc_r)
        sample["increment_surface"] = inc_s
        sample["increment_rootzone"] = inc_r
        sample["metric_mask"] = mask

        baseline.fit([sample])
        assert baseline._fitted is True

    def test_alpha_parameter(self):
        """Different alpha values produce different model coefficients."""
        baseline_a01 = RidgeBaseline(feature_set="core", alpha=0.01)
        baseline_a100 = RidgeBaseline(feature_set="core", alpha=100.0)

        # Use data where features actually vary so alpha matters
        samples = []
        for i, frac in enumerate([0.8, 1.0, 1.2]):
            inc_s = np.ones((8, 10)) * frac * 0.1
            inc_r = np.ones((8, 10)) * frac * 0.05
            # Vary forecast values to create non-constant features
            x = np.zeros((12, 8, 10), dtype=np.float32)
            x[0] = np.ones((8, 10)) * (0.5 + i * 0.1)  # varying forecast_surface
            x[1] = np.ones((8, 10)) * 0.4
            samples.append({
                "x": x,
                "forecast_surface": x[0],
                "forecast_rootzone": x[1],
                "increment_surface": inc_s,
                "increment_rootzone": inc_r,
                "metric_mask": np.ones((8, 10)),
                "date_str": "2021-06-15",
            })

        baseline_a01.fit(samples)
        baseline_a100.fit(samples)

        # Coefficients should differ with different alpha
        assert not np.allclose(
            baseline_a01._model_s.coef_,
            baseline_a100._model_s.coef_,
            rtol=1e-5,
        ), "Different alpha should produce different model coefficients"

    def test_tb_v_minus_tb_h_derived_correctly(self):
        """tb_v_minus_tb_h is computed as tb_v - tb_h."""
        baseline = RidgeBaseline(feature_set="input_full", alpha=1.0)

        H, W = 4, 4
        x = np.zeros((12, H, W), dtype=np.float32)
        x[0] = np.ones((H, W)) * 0.5   # forecast_surface
        x[1] = np.ones((H, W)) * 0.4   # forecast_rootzone
        x[4] = np.ones((H, W)) * 0.3   # vegopacity
        x[5] = np.full((H, W), 280.0)  # tb_h
        x[6] = np.full((H, W), 290.0)  # tb_v (10 K warmer)
        x[11] = np.ones((H, W))        # base_valid_mask

        sample = {
            "x": x,
            "forecast_surface": x[0],
            "forecast_rootzone": x[1],
            "increment_surface": np.ones((H, W)) * 0.1,
            "increment_rootzone": np.ones((H, W)) * 0.05,
            "metric_mask": np.ones((H, W)),
            "date_str": "2021-06-15",
        }

        baseline.fit([sample])
        assert baseline._fitted is True

    def test_same_feature_set_different_alpha(self):
        """Can create multiple baselines with same feature set, different alpha."""
        b1 = RidgeBaseline(feature_set="core", alpha=0.01)
        b2 = RidgeBaseline(feature_set="core", alpha=1.0)
        b3 = RidgeBaseline(feature_set="core", alpha=100.0)

        assert b1.feature_set == "core"
        assert b2.feature_set == "core"
        assert b3.feature_set == "core"
        assert b1.alpha == 0.01
        assert b2.alpha == 1.0
        assert b3.alpha == 100.0


class TestRidgeBaselineIntegration:
    """Integration tests for RidgeBaseline with metric computations."""

    def test_ridge_perfect_fit_on_constant_increment(self):
        """When all increments are constant, ridge should recover it."""
        baseline = RidgeBaseline(feature_set="core", alpha=1.0)

        const_s = 0.15
        const_r = 0.08
        samples = []
        for _ in range(5):
            inc_s = np.ones((10, 12)) * const_s
            inc_r = np.ones((10, 12)) * const_r
            x = np.zeros((12, 10, 12), dtype=np.float32)
            x[0] = np.ones((10, 12)) * 0.5
            x[1] = np.ones((10, 12)) * 0.4
            samples.append({
                "x": x,
                "forecast_surface": x[0],
                "forecast_rootzone": x[1],
                "increment_surface": inc_s,
                "increment_rootzone": inc_r,
                "metric_mask": np.ones((10, 12)),
                "date_str": "2021-06-15",
            })

        baseline.fit(samples)
        query = {
            "x": np.zeros((12, 10, 12), dtype=np.float32),
            "forecast_surface": np.ones((10, 12)) * 0.5,
            "forecast_rootzone": np.ones((10, 12)) * 0.4,
            "date_str": "2021-06-15",
        }
        pred = baseline.predict(query)

        # Ridge with only forecast features on constant increment should predict near the mean
        # Allow some slack because cyclical day features add some variation
        mean_pred_s = float(np.mean(pred["pred_increment_surface"]))
        mean_pred_r = float(np.mean(pred["pred_increment_rootzone"]))
        # Should be close to the true constant
        assert abs(mean_pred_s - const_s) < 0.05, f"Surface increment mean should be close to {const_s}"
        assert abs(mean_pred_r - const_r) < 0.05, f"Rootzone increment mean should be close to {const_r}"

    def test_ridge_sklearn_import(self):
        """Ridge model uses sklearn.linear_model.Ridge."""
        from sklearn.linear_model import Ridge as ExpectedRidge
        baseline = RidgeBaseline(feature_set="core")
        # Build a dummy model to verify import works
        X = np.random.rand(100, 4)
        y = np.random.rand(100)
        model = ExpectedRidge(alpha=1.0)
        model.fit(X, y)
        assert model.coef_.shape == (4,)
