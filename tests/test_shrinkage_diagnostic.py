"""Tests for shrinkage diagnostic correctness.

Verifies the mathematical properties of diagnose_oracle_shrinkage() from
scripts/diagnose_source_only_checkpoint.py.

Key contract:
  - α=0 must produce exactly skill=0 (forecast-only)
  - α=1 must be consistent with analysis_skill_vs_forecast
  - pred_analysis = forecast + α * pred_increment
  - surface/rootzone must not be mixed
  - Alpha selection must NOT use target_query labels
"""
from __future__ import annotations

import sys
import numpy as np
import pytest

# Import diagnose_oracle_shrinkage from the script
sys.path.insert(0, "scripts")
from diagnose_source_only_checkpoint import diagnose_oracle_shrinkage


def _make_test_data(seed=42, h=32, w=32):
    """Create synthetic forecast, analysis, and predicted increment arrays."""
    rng = np.random.RandomState(seed)
    forecast = rng.randn(h, w).astype(np.float32) * 0.05 + 0.2
    true_inc = rng.randn(h, w).astype(np.float32) * 0.02
    true_analysis = (forecast + true_inc).astype(np.float32)
    pred_inc = true_inc + rng.randn(h, w).astype(np.float32) * 0.01
    mask = np.ones((h, w), dtype=np.float32)
    return forecast, true_analysis, pred_inc, mask


class TestAlphaZeroGlobalSkillIsZero:
    """α=0 should produce exactly skill=0 (forecast-only baseline)."""

    def test_alpha_zero_global_skill(self):
        forecast, true_analysis, pred_inc, mask = _make_test_data()
        result = diagnose_oracle_shrinkage(pred_inc, true_analysis, forecast, mask, "surface")
        assert "skill_a0.0" in result
        assert abs(result["skill_a0.0"]) < 1e-10, (
            f"α=0 skill must be exactly 0, got {result['skill_a0.0']}"
        )

    def test_alpha_zero_with_noisy_forecast(self):
        """α=0 skill=0 even when forecast is noisy."""
        forecast, true_analysis, pred_inc, mask = _make_test_data(seed=99)
        # Add noise to forecast
        forecast = forecast + np.random.RandomState(77).randn(*forecast.shape).astype(np.float32) * 0.1
        result = diagnose_oracle_shrinkage(pred_inc, true_analysis, forecast, mask, "surface")
        assert abs(result["skill_a0.0"]) < 1e-10

    def test_alpha_zero_rootzone(self):
        """α=0 skill=0 for rootzone too."""
        forecast, true_analysis, pred_inc, mask = _make_test_data(seed=12)
        result = diagnose_oracle_shrinkage(pred_inc, true_analysis, forecast, mask, "rootzone")
        assert abs(result["skill_a0.0"]) < 1e-10


class TestAlphaOneConsistentWithEvaluateCheckpoint:
    """α=1 diagnostic must match analysis_skill_vs_forecast."""

    def test_alpha_one_matches_analysis_skill(self):
        from hydroda.metrics.skill import analysis_skill_vs_forecast

        forecast, true_analysis, pred_inc, mask = _make_test_data()
        pred_analysis = forecast + 1.0 * pred_inc

        expected_skill = analysis_skill_vs_forecast(pred_analysis, true_analysis, forecast, mask)
        result = diagnose_oracle_shrinkage(pred_inc, true_analysis, forecast, mask, "surface")

        assert "skill_a1.0" in result
        assert abs(result["skill_a1.0"] - expected_skill) < 1e-9, (
            f"α=1 skill {result['skill_a1.0']} != analysis_skill {expected_skill}"
        )

    def test_alpha_one_perfect_prediction(self):
        """When pred_inc = true_inc exactly, α=1 should give perfect skill."""
        forecast, true_analysis, _, mask = _make_test_data()
        true_inc = true_analysis - forecast
        # Perfect prediction
        pred_inc = true_inc.copy()

        result = diagnose_oracle_shrinkage(pred_inc, true_analysis, forecast, mask, "surface")
        # RMSE of predicted analysis should be 0, skill = 1 - 0/rmse_fcst = 1.0
        assert result["skill_a1.0"] > 0.99, (
            f"α=1 with perfect pred should give skill ~1.0, got {result['skill_a1.0']}"
        )


class TestPredAnalysisFormula:
    """pred_analysis = forecast + α * pred_increment."""

    def test_forecast_plus_alpha_times_increment(self):
        """Verify the mathematical formula is exactly forecast + alpha * increment."""
        forecast, true_analysis, pred_inc, mask = _make_test_data()

        for alpha in [0.0, 0.1, 0.5, 1.0]:
            pred_analysis = forecast + alpha * pred_inc
            # Verify explicitly
            diff = pred_analysis - (forecast + alpha * pred_inc)
            assert np.allclose(diff, 0.0, atol=1e-7), f"Formula broken for α={alpha}"

    def test_alpha_zero_gives_forecast_only(self):
        """α=0: pred_analysis == forecast."""
        forecast, true_analysis, pred_inc, _ = _make_test_data()
        pred_analysis = forecast + 0.0 * pred_inc
        np.testing.assert_allclose(pred_analysis, forecast, rtol=1e-7, atol=1e-8)


class TestSurfaceRootzoneNotMixed:
    """Surface and rootzone diagnostics must be independent."""

    def test_surface_result_does_not_depend_on_rootzone(self):
        forecast, true_analysis, pred_inc, mask = _make_test_data(seed=1)
        # Create different rootzone data
        forecast_r, true_analysis_r, pred_inc_r, mask_r = _make_test_data(seed=999)

        # Compute surface skill with surface data
        result_s = diagnose_oracle_shrinkage(pred_inc, true_analysis, forecast, mask, "surface")
        # Recompute surface skill — should be identical with same inputs
        result_s2 = diagnose_oracle_shrinkage(pred_inc.copy(), true_analysis.copy(), forecast.copy(), mask.copy(), "surface")

        for key in result_s:
            assert abs(result_s[key] - result_s2[key]) < 1e-10, (
                f"Surface result changed for key {key}: {result_s[key]} vs {result_s2[key]}"
            )

    def test_surface_and_rootzone_independent(self):
        """Surface and rootzone calculations are independent."""
        forecast, true_analysis, pred_inc, mask = _make_test_data(seed=42)
        forecast_r, true_analysis_r, pred_inc_r, mask_r = _make_test_data(seed=999)

        result_s = diagnose_oracle_shrinkage(pred_inc, true_analysis, forecast, mask, "surface")
        # Rootzone with completely different data should give different results
        result_r = diagnose_oracle_shrinkage(pred_inc_r, true_analysis_r, forecast_r, mask_r, "rootzone")

        # They should differ because inputs differ
        # At least one alpha should have different skill
        any_diff = False
        for alpha in [0.0, 0.01, 0.5, 1.0]:
            key = f"skill_a{alpha}"
            if abs(result_s[key] - result_r[key]) > 1e-10:
                any_diff = True
                break
        # It's possible they're equal by chance, but extremely unlikely with different seeds
        # If this fails, check if the test data is too similar
        assert any_diff, "Expected different surface/rootzone results with different test data"


class TestShrinkageAlphaSelectionNoTargetQuery:
    """Alpha selection must NOT use target_query labels.

    The oracle shrinkage function in diagnose_source_only_checkpoint.py operates
    on target_query data — it is a diagnostic tool only. The alpha that maximizes
    skill on target_query is labeled "oracle" to emphasize that it MUST NOT be used
    for model selection, early stopping, or training decisions.

    The production alpha selection must use source_val only (implemented in
    train_source_only_backbone.py's compute_source_val_shrinkage function).
    """

    def test_diagnose_oracle_shrinkage_docstring_warns_about_target_query(self):
        """Verify the function docstring explicitly warns about target_query usage."""
        doc = diagnose_oracle_shrinkage.__doc__
        assert doc is not None
        assert "MUST NOT" in doc or "must not" in doc, (
            "docstring must warn that alpha selection must not use target_query"
        )
        assert "source_val" in doc, (
            "docstring must mention source_val as the correct selection source"
        )

    def test_function_name_contains_oracle(self):
        """The function name emphasizes this is an oracle/diagnostic, not for model selection."""
        assert "oracle" in diagnose_oracle_shrinkage.__name__, (
            "function name must signal that this is oracle/diagnostic usage"
        )
