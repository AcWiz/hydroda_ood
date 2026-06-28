from __future__ import annotations

import numpy as np
import pytest
import torch


def _sample_x() -> np.ndarray:
    x = np.zeros((12, 2, 2), dtype=np.float32)
    x[0] = 0.20
    x[1] = 0.35
    x[4] = 0.40
    x[5] = np.array([[10.0, 12.0], [14.0, 16.0]], dtype=np.float32)
    x[6] = np.array([[20.0, 22.0], [24.0, 26.0]], dtype=np.float32)
    x[7] = 2.0
    x[8] = 4.0
    x[9] = np.array([[8.0, 10.0], [12.0, 14.0]], dtype=np.float32)
    x[10] = np.array([[18.0, 20.0], [22.0, 24.0]], dtype=np.float32)
    x[11] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    return x


def test_phys_trust_d0_computes_input_side_da_diagnostics():
    from hydroda.models.phys_trust import phys_trust_d0_diagnostics_from_tensor

    diagnostics = phys_trust_d0_diagnostics_from_tensor(_sample_x())

    assert diagnostics["tb_h_normalized_innovation_abs_median"] == pytest.approx(2.0 / 3.0)
    assert diagnostics["tb_v_normalized_innovation_abs_median"] == pytest.approx(2.0 / 5.0)
    assert diagnostics["tb_h_obs_error_confidence"] == pytest.approx(1.0 / 3.0)
    assert diagnostics["tb_v_obs_error_confidence"] == pytest.approx(1.0 / 5.0)
    assert diagnostics["vegopacity_median"] == pytest.approx(0.40)
    assert diagnostics["surface_rootzone_forecast_contrast_abs_median"] == pytest.approx(0.15)
    assert diagnostics["finite_input_coverage"] == pytest.approx(1.0)
    assert diagnostics["base_valid_mask_fraction_diagnostic_only"] == pytest.approx(0.5)


def test_phys_trust_d0_base_valid_is_not_a_hard_mask():
    from hydroda.models.phys_trust import phys_trust_d0_diagnostics_from_tensor

    x_zero = _sample_x()
    x_one = _sample_x()
    x_zero[11] = 0.0
    x_one[11] = 1.0

    zero = phys_trust_d0_diagnostics_from_tensor(x_zero)
    one = phys_trust_d0_diagnostics_from_tensor(x_one)

    for key in zero:
        if key == "base_valid_mask_fraction_diagnostic_only":
            continue
        assert zero[key] == pytest.approx(one[key])
    assert zero["base_valid_mask_fraction_diagnostic_only"] == pytest.approx(0.0)
    assert one["base_valid_mask_fraction_diagnostic_only"] == pytest.approx(1.0)


def test_phys_trust_d0_region_mask_limits_diagnostic_region():
    from hydroda.models.phys_trust import phys_trust_d0_diagnostics_from_tensor

    x = _sample_x()
    x[5, 0, 0] = 100.0
    x[5, 0, 1] = 100.0
    x[9, 0, 0] = 0.0
    x[9, 0, 1] = 0.0
    x[:, 1, 0] = np.nan
    mask_without_outlier = np.ones((2, 2), dtype=np.float32)
    mask_without_outlier[0, 0] = 0.0
    mask_without_outlier[0, 1] = 0.0
    mask_without_outlier[1, 0] = 0.0

    unmasked = phys_trust_d0_diagnostics_from_tensor(x)
    masked = phys_trust_d0_diagnostics_from_tensor(x, region_mask=mask_without_outlier)

    assert unmasked["tb_h_normalized_innovation_abs_median"] > masked["tb_h_normalized_innovation_abs_median"]
    assert masked["tb_h_normalized_innovation_abs_median"] == pytest.approx(2.0 / 3.0)
    assert unmasked["finite_input_coverage"] < 1.0
    assert masked["finite_input_coverage"] == pytest.approx(1.0)


def test_phys_trust_d0_sample_summary_reads_only_input_side_fields():
    from hydroda.models.phys_trust import phys_trust_d0_summary_from_samples

    class LabelPoisonSample(dict):
        def __getitem__(self, key):
            if key.startswith("analysis") or key.startswith("increment") or key.startswith("target_"):
                raise AssertionError(f"forbidden target-side field read: {key}")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key.startswith("analysis") or key.startswith("increment") or key.startswith("target_"):
                raise AssertionError(f"forbidden target-side field read: {key}")
            return super().get(key, default)

    sample = LabelPoisonSample(
        {
            "x": _sample_x(),
            "month": 1,
            "region_mask": np.ones((2, 2), dtype=np.float32),
            "analysis_surface": object(),
            "increment_rootzone": object(),
            "target_eval_statistics": object(),
        }
    )

    summary = phys_trust_d0_summary_from_samples(
        [sample],
        hyperda_trust_summary_by_month={
            "1": {
                "enabled": True,
                "nearest_distance_bounded": 0.25,
                "trust_strength": 0.50,
            }
        },
    )

    jan = summary["monthly"]["1"]
    assert summary["label_usage"] == "none"
    assert summary["target_val_usage"] == "unused_in_main_protocol"
    assert summary["target_eval_usage"] == "final_eval_only_no_selection"
    assert jan["count"] == 1
    assert jan["trust_gate_diagnostic"] == pytest.approx(0.875)
    assert jan["per_variable_trust"]["surface"]["trust_gate_diagnostic"] == pytest.approx(0.875)
    assert jan["per_variable_trust"]["rootzone"]["trust_gate_diagnostic"] == pytest.approx(0.875)


def test_phys_trust_d0_monthly_row_summary_matches_sample_summary():
    from hydroda.models.phys_trust import (
        phys_trust_d0_diagnostics_from_tensor,
        phys_trust_d0_summary_from_monthly_rows,
        phys_trust_d0_summary_from_samples,
    )

    x = _sample_x()
    trust = {"1": {"enabled": True, "nearest_distance_bounded": 0.25, "trust_strength": 0.50}}
    from_samples = phys_trust_d0_summary_from_samples(
        [{"x": x, "month": 1, "region_mask": np.ones((2, 2), dtype=np.float32)}],
        hyperda_trust_summary_by_month=trust,
    )
    from_rows = phys_trust_d0_summary_from_monthly_rows(
        {"1": [phys_trust_d0_diagnostics_from_tensor(x, region_mask=np.ones((2, 2), dtype=np.float32))]},
        hyperda_trust_summary_by_month=trust,
    )

    assert from_rows["overall"] == pytest.approx(from_samples["overall"])
    assert from_rows["monthly"]["1"]["count"] == from_samples["monthly"]["1"]["count"]
    assert from_rows["monthly"]["1"]["trust_gate_diagnostic"] == pytest.approx(
        from_samples["monthly"]["1"]["trust_gate_diagnostic"]
    )


def _low_risk_guard_x() -> np.ndarray:
    x = np.zeros((12, 2, 2), dtype=np.float32)
    x[0] = 0.25
    x[1] = 0.25
    x[4] = 0.0
    x[5] = 100.0
    x[6] = 110.0
    x[7] = 0.0
    x[8] = 0.0
    x[9] = 100.0
    x[10] = 110.0
    x[11] = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    return x


def _guard_source_state(q50: float = 0.0, q90: float = 1.0) -> dict:
    row = {"count": 4, "q50": q50, "q75": q90, "q90": q90, "q95": q90, "max": q90}
    return {
        "monthly_vertical_decoupling_quantiles": {str(month): dict(row) for month in range(1, 13)},
        "global_vertical_decoupling_quantiles": dict(row),
    }


def test_phys_consistency_guard_low_risk_is_identity():
    from hydroda.models.phys_trust import phys_consistency_guard_from_raw_tensor

    gate, summary = phys_consistency_guard_from_raw_tensor(
        _low_risk_guard_x(),
        region_mask=np.ones((2, 2), dtype=np.float32),
        month=1,
        source_state=_guard_source_state(),
    )

    assert gate.shape == (1, 2)
    assert torch.allclose(gate, torch.ones_like(gate))
    assert summary["label_usage"] == "none"
    assert summary["target_eval_usage"] == "final_eval_only_no_selection"
    assert summary["base_valid_mask_usage"] == "diagnostic_only_not_loss_metric_obs_region_mask_or_gate_mask"


def test_phys_consistency_guard_high_tb_innovation_shrinks_both_variables():
    from hydroda.models.phys_trust import phys_consistency_guard_from_raw_tensor

    x = _low_risk_guard_x()
    x[5] = 200.0
    x[9] = 100.0
    x[7] = 1.0

    gate, summary = phys_consistency_guard_from_raw_tensor(
        x,
        region_mask=np.ones((2, 2), dtype=np.float32),
        month=1,
        source_state=_guard_source_state(),
    )

    assert gate[0, 0].item() == pytest.approx(0.95)
    assert gate[0, 1].item() == pytest.approx(0.90)
    assert summary["r_enkf"]["max"] == pytest.approx(1.0)


def test_phys_consistency_guard_vertical_decoupling_shrinks_rootzone_more_than_surface():
    from hydroda.models.phys_trust import phys_consistency_guard_from_raw_tensor

    x = _low_risk_guard_x()
    x[1] = 0.75

    gate, summary = phys_consistency_guard_from_raw_tensor(
        x,
        region_mask=np.ones((2, 2), dtype=np.float32),
        month=1,
        source_state=_guard_source_state(q50=0.0, q90=0.10),
    )

    assert gate[0, 0].item() == pytest.approx(1.0)
    assert gate[0, 1].item() == pytest.approx(0.90)
    assert gate[0, 1].item() < gate[0, 0].item()
    assert summary["r_vert"]["max"] == pytest.approx(1.0)


def test_phys_consistency_guard_respects_region_mask_and_bounds():
    from hydroda.models.phys_trust import phys_consistency_guard_from_raw_tensor

    x = _low_risk_guard_x()
    x[5] = np.array([[1000.0, 1000.0], [1000.0, 100.0]], dtype=np.float32)
    x[9] = np.array([[0.0, 0.0], [0.0, 100.0]], dtype=np.float32)
    x[7] = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    mask_without_outlier = np.zeros((2, 2), dtype=np.float32)
    mask_without_outlier[1, 1] = 1.0

    masked_gate, _ = phys_consistency_guard_from_raw_tensor(
        x,
        region_mask=mask_without_outlier,
        month=1,
        source_state=_guard_source_state(),
    )
    unmasked_gate, _ = phys_consistency_guard_from_raw_tensor(
        x,
        region_mask=np.ones((2, 2), dtype=np.float32),
        month=1,
        source_state=_guard_source_state(),
    )

    assert torch.allclose(masked_gate, torch.ones_like(masked_gate))
    assert unmasked_gate[0, 0].item() == pytest.approx(0.95)
    assert unmasked_gate[0, 1].item() == pytest.approx(0.90)


def test_phys_consistency_source_state_reads_only_input_side_fields():
    from hydroda.models.phys_trust import phys_consistency_source_state_from_samples

    class LabelPoisonSample(dict):
        def __getitem__(self, key):
            if key.startswith("analysis") or key.startswith("increment") or key.startswith("target_"):
                raise AssertionError(f"forbidden target-side field read: {key}")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key.startswith("analysis") or key.startswith("increment") or key.startswith("target_"):
                raise AssertionError(f"forbidden target-side field read: {key}")
            return super().get(key, default)

    state = phys_consistency_source_state_from_samples(
        [
            LabelPoisonSample(
                {
                    "x": _low_risk_guard_x(),
                    "month": 3,
                    "region_mask": np.ones((2, 2), dtype=np.float32),
                    "analysis_surface": object(),
                    "increment_rootzone": object(),
                    "target_eval_statistics": object(),
                }
            )
        ]
    )

    assert state["label_usage"] == "none"
    assert state["target_eval_usage"] == "final_eval_only_no_selection"
    assert state["monthly_vertical_decoupling_quantiles"]["3"]["count"] == 1


def test_phys_consistency_guard_channel_11_is_diagnostic_only():
    from hydroda.models.phys_trust import phys_consistency_guard_from_raw_tensor

    x_zero = _low_risk_guard_x()
    x_one = _low_risk_guard_x()
    x_zero[11] = 0.0
    x_one[11] = 1.0

    zero_gate, zero_summary = phys_consistency_guard_from_raw_tensor(
        x_zero,
        region_mask=np.ones((2, 2), dtype=np.float32),
        month=1,
        source_state=_guard_source_state(),
    )
    one_gate, one_summary = phys_consistency_guard_from_raw_tensor(
        x_one,
        region_mask=np.ones((2, 2), dtype=np.float32),
        month=1,
        source_state=_guard_source_state(),
    )

    assert torch.allclose(zero_gate, one_gate)
    assert zero_summary["base_valid_mask_fraction_diagnostic_only"]["mean"] == pytest.approx(0.0)
    assert one_summary["base_valid_mask_fraction_diagnostic_only"]["mean"] == pytest.approx(1.0)


def test_phys_formula_features_are_bounded_region_masked_and_channel_11_diagnostic_only():
    from hydroda.models.phys_trust import (
        PHYS_FORMULA_FEATURE_SCHEMA,
        PHYS_FORMULA_SOURCE,
        phys_formula_features_from_raw_tensor,
    )

    x_zero = _low_risk_guard_x()
    x_one = _low_risk_guard_x()
    x_zero[11] = 0.0
    x_one[11] = 1.0
    x_zero[5] = np.array([[1000.0, 1000.0], [1000.0, 100.0]], dtype=np.float32)
    x_one[5] = x_zero[5]
    x_zero[9] = np.array([[0.0, 0.0], [0.0, 100.0]], dtype=np.float32)
    x_one[9] = x_zero[9]
    x_zero[7] = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    x_one[7] = x_zero[7]
    mask_without_outlier = np.zeros((2, 2), dtype=np.float32)
    mask_without_outlier[1, 1] = 1.0

    zero_features, zero_summary = phys_formula_features_from_raw_tensor(
        x_zero,
        region_mask=mask_without_outlier,
        month=1,
        source_state=_guard_source_state(),
    )
    one_features, one_summary = phys_formula_features_from_raw_tensor(
        x_one,
        region_mask=mask_without_outlier,
        month=1,
        source_state=_guard_source_state(),
    )
    unmasked_features, _ = phys_formula_features_from_raw_tensor(
        x_one,
        region_mask=np.ones((2, 2), dtype=np.float32),
        month=1,
        source_state=_guard_source_state(),
    )

    assert zero_features.shape == (1, len(PHYS_FORMULA_FEATURE_SCHEMA))
    assert torch.isfinite(zero_features).all()
    assert torch.all((zero_features >= 0.0) & (zero_features <= 1.0))
    diagnostic_idx = PHYS_FORMULA_FEATURE_SCHEMA.index("base_valid_mask_fraction_diagnostic_only")
    non_diagnostic = [idx for idx in range(zero_features.shape[1]) if idx != diagnostic_idx]
    assert torch.allclose(zero_features[:, non_diagnostic], one_features[:, non_diagnostic])
    assert zero_features[0, diagnostic_idx].item() == pytest.approx(0.0)
    assert one_features[0, diagnostic_idx].item() == pytest.approx(1.0)
    assert zero_features[0, PHYS_FORMULA_FEATURE_SCHEMA.index("r_enkf")].item() == pytest.approx(0.0)
    assert unmasked_features[0, PHYS_FORMULA_FEATURE_SCHEMA.index("r_enkf")].item() == pytest.approx(1.0)
    assert zero_summary["phys_formula_source"] == PHYS_FORMULA_SOURCE
    assert zero_summary["base_valid_mask_usage"] == "diagnostic_only_not_loss_metric_obs_region_mask_or_gate_mask"
    assert one_summary["target_eval_usage"] == "final_eval_only_no_selection"


def test_enhanced_phys_formula_features_add_m3_9_input_side_risks_and_keep_channel_11_diagnostic_only():
    from hydroda.models.phys_trust import (
        PHYS_FORMULA_ENHANCED_SOURCE,
        PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA,
        phys_formula_features_from_raw_tensor,
    )

    x_zero = _low_risk_guard_x()
    x_one = _low_risk_guard_x()
    x_zero[11] = 0.0
    x_one[11] = 1.0
    x_zero[0] = 0.10
    x_zero[1] = 0.95
    x_one[0] = x_zero[0]
    x_one[1] = x_zero[1]
    x_zero[3] = 290.0
    x_zero[2] = 280.0
    x_one[3] = x_zero[3]
    x_one[2] = x_zero[2]
    x_zero[5] = np.array([[105.0, 1000.0], [105.0, 105.0]], dtype=np.float32)
    x_zero[6] = np.array([[90.0, 1000.0], [90.0, 90.0]], dtype=np.float32)
    x_zero[9] = np.array([[100.0, 0.0], [100.0, 100.0]], dtype=np.float32)
    x_zero[10] = np.array([[100.0, 0.0], [100.0, 100.0]], dtype=np.float32)
    x_zero[7] = np.array([[2.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    x_zero[8] = np.array([[4.0, 1.0], [4.0, 4.0]], dtype=np.float32)
    x_one[5:11] = x_zero[5:11]
    mask_without_outlier = np.ones((2, 2), dtype=np.float32)
    mask_without_outlier[0, 1] = 0.0

    zero_features, zero_summary = phys_formula_features_from_raw_tensor(
        x_zero,
        region_mask=mask_without_outlier,
        month=1,
        source_state=_guard_source_state(q50=0.0, q90=0.10),
        source=PHYS_FORMULA_ENHANCED_SOURCE,
    )
    one_features, one_summary = phys_formula_features_from_raw_tensor(
        x_one,
        region_mask=mask_without_outlier,
        month=1,
        source_state=_guard_source_state(q50=0.0, q90=0.10),
        source=PHYS_FORMULA_ENHANCED_SOURCE,
    )
    unmasked_features, _ = phys_formula_features_from_raw_tensor(
        x_one,
        region_mask=np.ones((2, 2), dtype=np.float32),
        month=1,
        source_state=_guard_source_state(q50=0.0, q90=0.10),
        source=PHYS_FORMULA_ENHANCED_SOURCE,
    )

    assert zero_features.shape == (1, len(PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA))
    assert torch.isfinite(zero_features).all()
    assert torch.all((zero_features >= 0.0) & (zero_features <= 1.0))
    diagnostic_idx = PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA.index("base_valid_mask_fraction_diagnostic_only")
    non_diagnostic = [idx for idx in range(zero_features.shape[1]) if idx != diagnostic_idx]
    assert torch.allclose(zero_features[:, non_diagnostic], one_features[:, non_diagnostic])
    assert zero_features[0, diagnostic_idx].item() == pytest.approx(0.0)
    assert one_features[0, diagnostic_idx].item() == pytest.approx(1.0)
    assert zero_features[0, PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA.index("tb_h_normalized_innovation_risk")].item() == pytest.approx(1.0)
    assert zero_features[0, PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA.index("tb_v_normalized_innovation_risk")].item() == pytest.approx(1.0)
    assert zero_features[0, PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA.index("tb_innovation_asymmetry_risk")].item() > 0.0
    assert zero_features[0, PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA.index("surface_rootzone_hydraulic_gradient_proxy")].item() == pytest.approx(0.0)
    assert unmasked_features[0, PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA.index("tb_h_normalized_innovation_risk")].item() == pytest.approx(1.0)
    assert zero_summary["phys_formula_source"] == PHYS_FORMULA_ENHANCED_SOURCE
    assert zero_summary["schema_version"] == "phys_formula_operator_v2_enhanced_input_side"
    assert zero_summary["target_eval_usage"] == "final_eval_only_no_selection"
    assert one_summary["base_valid_mask_usage"] == "diagnostic_only_not_loss_metric_obs_region_mask_or_gate_mask"
    assert "target_eval" not in " ".join(zero_summary["feature_schema"])


def test_m3_14_formula_gain_features_encode_dry_and_wet_support_and_keep_channel_11_diagnostic_only():
    from hydroda.models.phys_trust import (
        PHYS_FORMULA_GAIN_FEATURE_SCHEMA,
        PHYS_FORMULA_GAIN_SOURCE,
        phys_formula_features_from_raw_tensor,
    )

    x_zero = _low_risk_guard_x()
    x_one = _low_risk_guard_x()
    x_zero[0] = 0.30
    x_zero[1] = 0.20
    x_zero[2] = 285.0
    x_zero[3] = 280.0
    x_zero[4] = 0.10
    x_zero[5] = 102.0
    x_zero[6] = 112.0
    x_zero[7] = 1.0
    x_zero[8] = 1.0
    x_zero[9] = 100.0
    x_zero[10] = 110.0
    x_one[:] = x_zero
    x_zero[11] = 0.0
    x_one[11] = 1.0

    zero_features, zero_summary = phys_formula_features_from_raw_tensor(
        x_zero,
        region_mask=np.ones((2, 2), dtype=np.float32),
        month=1,
        source=PHYS_FORMULA_GAIN_SOURCE,
        source_state={"m3_14_gain_prior_summary": {"surface": 0.75, "rootzone": 0.25}},
    )
    one_features, one_summary = phys_formula_features_from_raw_tensor(
        x_one,
        region_mask=np.ones((2, 2), dtype=np.float32),
        month=1,
        source=PHYS_FORMULA_GAIN_SOURCE,
        source_state={"m3_14_gain_prior_summary": {"surface": 0.75, "rootzone": 0.25}},
    )

    assert zero_features.shape == (1, len(PHYS_FORMULA_GAIN_FEATURE_SCHEMA))
    assert torch.isfinite(zero_features).all()
    assert torch.all((zero_features >= 0.0) & (zero_features <= 1.0))
    d_h_idx = PHYS_FORMULA_GAIN_FEATURE_SCHEMA.index("d_H_dry_direction")
    m_h_idx = PHYS_FORMULA_GAIN_FEATURE_SCHEMA.index("m_H_wet_support")
    diagnostic_idx = PHYS_FORMULA_GAIN_FEATURE_SCHEMA.index("base_valid_mask_fraction_diagnostic_only")
    assert zero_features[0, d_h_idx].item() > 0.5
    assert zero_features[0, m_h_idx].item() < 0.5
    non_diagnostic = [idx for idx in range(zero_features.shape[1]) if idx != diagnostic_idx]
    assert torch.allclose(zero_features[:, non_diagnostic], one_features[:, non_diagnostic])
    assert zero_features[0, diagnostic_idx].item() == pytest.approx(0.0)
    assert one_features[0, diagnostic_idx].item() == pytest.approx(1.0)
    assert zero_features[0, PHYS_FORMULA_GAIN_FEATURE_SCHEMA.index("source_gain_prior_surface_summary")].item() == pytest.approx(0.75)
    assert zero_features[0, PHYS_FORMULA_GAIN_FEATURE_SCHEMA.index("source_gain_prior_rootzone_summary")].item() == pytest.approx(0.25)
    assert zero_summary["phys_formula_source"] == PHYS_FORMULA_GAIN_SOURCE
    assert zero_summary["schema_version"] == "m3_14_raw_input_side_formula_gain_v1"
    assert zero_summary["final_output_residual_allowed"] is False
    assert zero_summary["coefficient_injection_role"] == "bounded_operator_coefficient_logit_delta_only"
    assert zero_summary["source_fit_regularization_lambda_default"] == pytest.approx(0.01)
    assert one_summary["channel_11_usage"] == "diagnostic_only_not_hard_mask"


@pytest.mark.parametrize("forbidden_role", ["target_context", "target_val", "target_eval"])
def test_m3_16_source_fit_gain_bank_rejects_target_side_roles(forbidden_role: str):
    from hydroda.models.phys_trust import build_phys_gain_source_bank

    sample = {
        "split_role": forbidden_role,
        "sample_region_id": "US-R2",
        "month": 1,
        "x": _sample_x(),
        "increment_surface": torch.ones(2, 2),
        "increment_rootzone": torch.ones(2, 2),
        "loss_mask": torch.ones(2, 2, dtype=torch.bool),
    }

    with pytest.raises(ValueError, match="refuses sample split_role|source_fit"):
        build_phys_gain_source_bank([sample], source_split_roles=("source_fit",))

    source_fit_sample = dict(sample)
    source_fit_sample["split_role"] = "source_fit"
    with pytest.raises(ValueError, match="forbids target-side"):
        build_phys_gain_source_bank([source_fit_sample], source_split_roles=(forbidden_role,))


def test_phys_formula_product_guard_keeps_rootzone_gate_in_m3_8_band():
    from hydroda.models.phys_trust import (
        PHYS_CONSISTENCY_GUARD_PRODUCT_MODE,
        PHYS_FORMULA_SOURCE,
        phys_consistency_guard_from_raw_tensor,
    )

    x = _low_risk_guard_x()
    x[0] = 0.20
    x[1] = 0.80
    x[5] = 200.0
    x[9] = 100.0
    x[7] = 1.0

    gate, summary = phys_consistency_guard_from_raw_tensor(
        x,
        region_mask=np.ones((2, 2), dtype=np.float32),
        month=1,
        source_state=_guard_source_state(q50=0.0, q90=0.10),
        mode=PHYS_CONSISTENCY_GUARD_PRODUCT_MODE,
        min_surface=0.97,
        min_rootzone=0.98,
        strength_surface=0.05,
        strength_rootzone=0.02,
    )

    assert gate.shape == (1, 2)
    assert gate[0, 0].item() == pytest.approx(0.97)
    assert gate[0, 1].item() == pytest.approx(0.98)
    assert gate[0, 1].item() >= 0.98
    assert gate[0, 1].item() <= 1.0
    assert summary["mode"] == PHYS_CONSISTENCY_GUARD_PRODUCT_MODE
    assert summary["phys_consistency_source"] == PHYS_FORMULA_SOURCE
    assert summary["guard_action"] == "surface_primary_product_shrink_or_identity_variable_trust_gate"
    assert "* r_vert" in summary["formula"]["g_rootzone"]


def test_phys_consistency_guard_records_enhanced_formula_source_when_requested():
    from hydroda.models.phys_trust import (
        PHYS_CONSISTENCY_GUARD_PRODUCT_MODE,
        PHYS_FORMULA_ENHANCED_SOURCE,
        phys_consistency_guard_from_raw_tensor,
    )

    gate, summary = phys_consistency_guard_from_raw_tensor(
        _low_risk_guard_x(),
        region_mask=np.ones((2, 2), dtype=np.float32),
        month=1,
        source_state=_guard_source_state(),
        mode=PHYS_CONSISTENCY_GUARD_PRODUCT_MODE,
        source=PHYS_FORMULA_ENHANCED_SOURCE,
        min_surface=0.98,
        min_rootzone=0.99,
        strength_surface=0.02,
        strength_rootzone=0.01,
    )

    assert gate.shape == (1, 2)
    assert summary["phys_consistency_source"] == PHYS_FORMULA_ENHANCED_SOURCE
    assert summary["phys_formula_source"] == PHYS_FORMULA_ENHANCED_SOURCE
    assert summary["target_eval_usage"] == "final_eval_only_no_selection"
