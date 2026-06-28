from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_k_artifacts(base: Path, k: int, target_region: str = "US-R1") -> None:
    eval_dir = base / f"K{k}" / "eval" / target_region
    adapt_dir = base / f"K{k}" / "adapt"
    eval_dir.mkdir(parents=True, exist_ok=True)
    adapt_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": f"method_k{k}",
                "target_region": target_region,
                "K": k,
                "adaptation_setting": "zero_shot_context" if k == 0 else f"few_shot_k{k}",
                "seed": 0,
                "split_type": "target_eval",
                "adapt_mix_rho": 1.0,
                "surface": {
                    "skill_primary": 0.1 + k,
                    "skill_latw_primary": 0.2 + k,
                    "skill_median": 0.0,
                    "rmse_latw_mean": 1.0,
                    "corr_latw_mean": 0.0,
                },
                "rootzone": {
                    "skill_primary": 0.3 + k,
                    "skill_latw_primary": 0.4 + k,
                    "skill_median": 0.0,
                    "rmse_latw_mean": 2.0,
                    "corr_latw_mean": 0.0,
                },
                "stage3_protocol": {
                    "stage3_posterior_decision": "no_update" if k == 0 else "accepted",
                    "paper_facing_run": k == 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (adapt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "K": k,
                "seed": 0,
                "adapt_recipe": "source_anchor",
                "adapt_scope": "none" if k == 0 else "safe_operator",
                "stage3_kshot_mode": "paper_safe" if k == 0 else "diagnostic_direct_kshot",
                "paper_facing_run": k == 0,
                "policy_source": "zero_shot_no_policy" if k == 0 else "diagnostic_direct_target_support",
                "stage3_posterior_decision": "no_update" if k == 0 else "accepted",
                "support_gate_reject_reason": [] if k == 0 else ["support_gradient_conflict"],
                "target_parameter_l2_drift_pre_anchor": {
                    "total": 0.0 if k == 0 else 1.5,
                    "target_prompt": 0.0 if k == 0 else 1.25,
                },
                "target_parameter_l2_drift_post_anchor": {
                    "total": 0.0 if k == 0 else 0.5,
                    "target_prompt": 0.0 if k == 0 else 0.0,
                    "monthly_gain": 0.0,
                },
                "target_labels_used_for_adaptation": k > 0,
                "actual_optimizer_steps": 0 if k == 0 else 2,
                "support_gain_calibration": {
                    "selection_rule": "stable_high_alpha_with_mean_skill_guard",
                    "best_alpha_raw": 0.5,
                    "stable_candidate_alphas": [0.5, 0.75],
                    "selection_margin": 0.015,
                    "target_eval_usage": "final_eval_only_no_selection",
                }
                if k == 12
                else {},
                "context_tta_effective": k == 0,
                "context_tta_source_stat_status": "ok" if k == 0 else "not_requested",
                "prompt_l2_delta_mean": 0.125 if k == 0 else 0.0,
                "prediction_delta_vs_no_tta": 0.25 if k == 0 else 0.0,
            }
        ),
        encoding="utf-8",
    )


def test_stage3_overview_scans_existing_k_directories_on_partial_rerun(tmp_path):
    from scripts.eval.stage3_kshot_overview import discover_k_values, rebuild_overview

    _write_k_artifacts(tmp_path, 0)
    _write_k_artifacts(tmp_path, 4)
    _write_k_artifacts(tmp_path, 12)
    (tmp_path / "K4" / "eval" / "US-R1" / "metrics_long.csv.gz").write_bytes(b"compressed placeholder")

    assert discover_k_values(tmp_path, requested_k_values=["12"]) == ["0", "4", "12"]

    rows = rebuild_overview(
        output_base=tmp_path,
        target_region="US-R1",
        requested_k_values=["12"],
        seed=0,
    )

    assert [row["K"] for row in rows] == ["0", "4", "12"]
    by_k = {row["K"]: row for row in rows}
    assert by_k["4"]["stage3_kshot_mode"] == "diagnostic_direct_kshot"
    assert by_k["4"]["policy_source"] == "diagnostic_direct_target_support"
    assert by_k["4"]["target_labels_used_for_adaptation"] is True
    assert by_k["4"]["support_gate_reject_reason"] == '["support_gradient_conflict"]'
    assert by_k["4"]["target_parameter_l2_drift_pre_anchor_target_prompt"] == 1.25
    assert by_k["4"]["target_parameter_l2_drift_post_anchor_target_prompt"] == 0.0
    assert by_k["4"]["metrics_long"].endswith("metrics_long.csv.gz")
    assert by_k["12"]["actual_optimizer_steps"] == 2
    assert by_k["12"]["support_gain_selection_rule"] == "stable_high_alpha_with_mean_skill_guard"
    assert by_k["12"]["support_gain_best_alpha_raw"] == 0.5
    assert by_k["12"]["support_gain_stable_candidate_alphas"] == "[0.5, 0.75]"
    assert by_k["12"]["support_gain_selection_margin"] == 0.015
    assert by_k["12"]["support_gain_target_eval_usage"] == "final_eval_only_no_selection"
    assert by_k["0"]["context_tta_effective"] is True
    assert by_k["0"]["context_tta_source_stat_status"] == "ok"
    assert by_k["0"]["prompt_l2_delta_mean"] == 0.125
    assert by_k["0"]["prediction_delta_vs_no_tta"] == 0.25
    assert (tmp_path / "overview.csv").exists()
    assert (tmp_path / "overview.json").exists()
    assert (tmp_path / "overview.md").exists()
    assert "prompt_l2_delta_mean" in (tmp_path / "overview.md").read_text(encoding="utf-8")
    assert "support_gate_reject_reason" in (tmp_path / "overview.md").read_text(encoding="utf-8")


def test_stage3_overview_exposes_v4_nested_support_gain_metadata(tmp_path):
    from scripts.eval.stage3_kshot_overview import rebuild_overview

    _write_k_artifacts(tmp_path, 12)
    metadata_path = tmp_path / "K12" / "adapt" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "stage3_kshot_mode": "diagnostic_support_gain_v4_nested_stable",
            "support_nesting_policy": "run_local_k12_nested_k4_plus_8_original_k12_nonduplicate",
            "nested_support_dates_hash": "nestedhash12",
            "support_gain_calibration": {
                "selection_rule": "support_uncertainty_stable_high_alpha_with_dual_guard",
                "best_alpha_raw": 0.5,
                "stable_candidate_alphas": [0.5, 0.75],
                "selection_margin": 0.015,
                "stability_tolerance": 0.025,
                "paired_support_se_capped": 0.0125,
                "support_nesting_policy": "run_local_k12_nested_k4_plus_8_original_k12_nonduplicate",
                "nested_support_dates_hash": "nestedhash12",
                "target_eval_usage": "final_eval_only_no_selection",
            },
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    rows = rebuild_overview(output_base=tmp_path, target_region="US-R1", requested_k_values=["12"], seed=0)

    row = rows[0]
    assert row["stage3_kshot_mode"] == "diagnostic_support_gain_v4_nested_stable"
    assert row["support_nesting_policy"] == "run_local_k12_nested_k4_plus_8_original_k12_nonduplicate"
    assert row["nested_support_dates_hash"] == "nestedhash12"
    assert row["support_gain_selection_rule"] == "support_uncertainty_stable_high_alpha_with_dual_guard"
    assert row["support_gain_stability_tolerance"] == 0.025
    assert row["support_gain_paired_support_se_capped"] == 0.0125
    csv_text = (tmp_path / "overview.csv").read_text(encoding="utf-8")
    assert "support_nesting_policy" in csv_text
    assert "support_gain_paired_support_se_capped" in csv_text


def test_stage3_overview_exposes_support_affine_and_k_specific_hash_metadata(tmp_path):
    from scripts.eval.stage3_kshot_overview import rebuild_overview

    _write_k_artifacts(tmp_path, 4)
    _write_k_artifacts(tmp_path, 12)
    k4_summary = tmp_path / "K4" / "eval" / "US-R1" / "summary.json"
    k12_summary = tmp_path / "K12" / "eval" / "US-R1" / "summary.json"
    summary4 = json.loads(k4_summary.read_text(encoding="utf-8"))
    summary12 = json.loads(k12_summary.read_text(encoding="utf-8"))
    summary4["prediction_content_hash"] = "hash-k4"
    summary12["prediction_content_hash"] = "hash-k12"
    k4_summary.write_text(json.dumps(summary4), encoding="utf-8")
    k12_summary.write_text(json.dumps(summary12), encoding="utf-8")

    metadata_path = tmp_path / "K12" / "adapt" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "stage3_kshot_mode": "diagnostic_support_affine_v1_nested",
            "support_nesting_policy": "run_local_k12_nested_k4_plus_8_original_k12_nonduplicate",
            "nested_support_dates_hash": "nestedhash12",
            "support_affine_calibration": {
                "calibration_mode": "target_support_residual_affine_v1_nested",
                "label_source": "target_support_only",
                "target_eval_usage": "final_eval_only_no_selection",
                "support_affine_coefficients": {
                    "surface": {"a": 1.25, "b": -0.1},
                    "rootzone": {"a": 0.75, "b": 0.2},
                },
                "effective_calibration_dof": 4.0,
                "support_nesting_policy": "run_local_k12_nested_k4_plus_8_original_k12_nonduplicate",
                "nested_support_dates_hash": "nestedhash12",
            },
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    rows = rebuild_overview(output_base=tmp_path, target_region="US-R1", requested_k_values=["12"], seed=0)

    by_k = {row["K"]: row for row in rows}
    assert by_k["12"]["stage3_kshot_mode"] == "diagnostic_support_affine_v1_nested"
    assert by_k["12"]["support_affine_target_eval_usage"] == "final_eval_only_no_selection"
    assert by_k["12"]["support_affine_surface_a"] == 1.25
    assert by_k["12"]["support_affine_rootzone_b"] == 0.2
    assert by_k["12"]["effective_calibration_dof"] == 4.0
    assert by_k["12"]["k_specific_prediction_changed"] is True
    assert by_k["4"]["k_specific_prediction_changed"] is False
    csv_text = (tmp_path / "overview.csv").read_text(encoding="utf-8")
    assert "support_affine_surface_a" in csv_text
    assert "k_specific_prediction_changed" in csv_text


def test_stage3_overview_exposes_v13_aggressive_calibration_pool_metadata(tmp_path):
    from scripts.eval.stage3_kshot_overview import rebuild_overview

    _write_k_artifacts(tmp_path, 12)
    metadata_path = tmp_path / "K12" / "adapt" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "stage3_kshot_mode": "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
            "selected_support_candidate_id": "k12_alpha2d_fine",
            "support_selection_objective": "k12_aggressive_nested_cv_calibration_pool_support_only",
            "support_candidate_pool": [{"candidate_id": "k12_alpha2d_fine"}],
            "support_cv_objective_delta": -1.0e-6,
            "support_cv_objective_delta_se": 2.0e-7,
            "support_cv_objective_delta_t": -5.0,
            "support_cv_nested_k4_objective_delta": -1.2e-6,
            "support_cv_added_objective_delta": -0.8e-6,
            "k12_vs_k4_cv_objective_delta": -1.0e-6,
            "k12_vs_k4_cv_rootzone_delta": 4.0e-6,
            "support_cycle_improvement_fraction": 0.75,
            "support_calibration_dof": 2.0,
            "target_eval_usage": "final_eval_only_no_selection",
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    rows = rebuild_overview(output_base=tmp_path, target_region="US-R1", requested_k_values=["12"], seed=0)

    row = rows[0]
    assert row["stage3_kshot_mode"] == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool"
    assert row["selected_support_candidate_id"] == "k12_alpha2d_fine"
    assert row["support_selection_objective"] == "k12_aggressive_nested_cv_calibration_pool_support_only"
    assert row["support_cv_objective_delta"] == pytest.approx(-1.0e-6)
    assert row["support_cv_objective_delta_se"] == pytest.approx(2.0e-7)
    assert row["support_cv_objective_delta_t"] == pytest.approx(-5.0)
    assert row["support_cv_nested_k4_objective_delta"] == pytest.approx(-1.2e-6)
    assert row["support_cv_added_objective_delta"] == pytest.approx(-0.8e-6)
    assert row["k12_vs_k4_cv_objective_delta"] == pytest.approx(-1.0e-6)
    assert row["k12_vs_k4_cv_rootzone_delta"] == pytest.approx(4.0e-6)
    assert row["support_cycle_improvement_fraction"] == pytest.approx(0.75)
    assert row["support_calibration_dof"] == pytest.approx(2.0)
    assert json.loads(row["support_candidate_pool"]) == [{"candidate_id": "k12_alpha2d_fine"}]
    csv_text = (tmp_path / "overview.csv").read_text(encoding="utf-8")
    assert "k12_vs_k4_cv_objective_delta" in csv_text
    assert "support_calibration_dof" in csv_text


def test_stage3_overview_exposes_linearized_ridge_v6_metadata(tmp_path):
    from scripts.eval.stage3_kshot_overview import rebuild_overview

    _write_k_artifacts(tmp_path, 12)
    metadata_path = tmp_path / "K12" / "adapt" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "stage3_kshot_mode": "diagnostic_linearized_coeff_ridge_v6_nested",
            "adapt_solver": "ridge_coeff",
            "adaptation_step_policy_source": (
                "diagnostic_linearized_coeff_ridge_v6_closed_form_no_adam_steps"
            ),
            "ridge_lambda": 1.0,
            "ridge_weighting": "cycle_variable_balanced_huber",
            "ridge_effective_calibration_dof": 17.0,
            "ridge_diagnostics": {
                "ridge_weighting": "cycle_variable_balanced_huber",
                "target_eval_usage": "final_eval_only_no_selection",
                "parameter_scope": "adapter_coefficient_residuals_only",
                "effective_calibration_dof": 17.0,
            },
            "linearized_ridge_calibration": {
                "calibration_mode": "target_support_linearized_coeff_ridge_v7_balanced_nested",
                "target_eval_usage": "final_eval_only_no_selection",
                "parameter_scope": "adapter_coefficient_residuals_only",
                "effective_calibration_dof": 17.0,
            },
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    rows = rebuild_overview(output_base=tmp_path, target_region="US-R1", requested_k_values=["12"], seed=0)

    row = rows[0]
    assert row["stage3_kshot_mode"] == "diagnostic_linearized_coeff_ridge_v6_nested"
    assert row["adapt_solver"] == "ridge_coeff"
    assert row["adaptation_step_policy_source"] == (
        "diagnostic_linearized_coeff_ridge_v6_closed_form_no_adam_steps"
    )
    assert row["ridge_lambda"] == pytest.approx(1.0)
    assert row["ridge_weighting"] == "cycle_variable_balanced_huber"
    assert row["ridge_effective_calibration_dof"] == pytest.approx(17.0)
    assert row["effective_calibration_dof"] == pytest.approx(17.0)
    assert row["linearized_ridge_calibration_mode"] == "target_support_linearized_coeff_ridge_v7_balanced_nested"
    assert row["linearized_ridge_target_eval_usage"] == "final_eval_only_no_selection"
    assert row["linearized_ridge_parameter_scope"] == "adapter_coefficient_residuals_only"
    csv_text = (tmp_path / "overview.csv").read_text(encoding="utf-8")
    assert "adaptation_step_policy_source" in csv_text
    assert "linearized_ridge_target_eval_usage" in csv_text


def test_stage3_overview_reconstructs_legacy_linearized_ridge_v6_metadata(tmp_path):
    from scripts.eval.stage3_kshot_overview import rebuild_overview

    _write_k_artifacts(tmp_path, 4)
    metadata_path = tmp_path / "K4" / "adapt" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "stage3_kshot_mode": "diagnostic_linearized_coeff_ridge_v6_nested",
            "adapt_solver": "ridge_coeff",
            "ridge_diagnostics": {
                "ridge_lambda": 2.0,
                "ridge_weighting": "global_pixel_l2",
                "target_eval_usage": "final_eval_only_no_selection",
                "parameter_scope": "adapter_coefficient_residuals_only",
                "effective_calibration_dof": 24.0,
            },
            "linearized_ridge_calibration": {
                "ridge_lambda": 2.0,
                "target_eval_usage": "final_eval_only_no_selection",
                "parameter_scope": "adapter_coefficient_residuals_only",
                "effective_calibration_dof": 24.0,
            },
            "resolved_mode_defaults": {
                "adaptation_step_policy_source": (
                    "diagnostic_linearized_coeff_ridge_v6_closed_form_no_adam_steps"
                )
            },
        }
    )
    metadata.pop("ridge_lambda", None)
    metadata.pop("ridge_effective_calibration_dof", None)
    metadata.pop("adaptation_step_policy_source", None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    rows = rebuild_overview(output_base=tmp_path, target_region="US-R1", requested_k_values=["4"], seed=0)

    row = rows[0]
    assert row["ridge_lambda"] == pytest.approx(2.0)
    assert row["ridge_weighting"] == "global_pixel_l2"
    assert row["ridge_effective_calibration_dof"] == pytest.approx(24.0)
    assert row["linearized_ridge_target_eval_usage"] == "final_eval_only_no_selection"
    assert row["linearized_ridge_parameter_scope"] == "adapter_coefficient_residuals_only"
    assert row["adaptation_step_policy_source"] == (
        "diagnostic_linearized_coeff_ridge_v6_closed_form_no_adam_steps"
    )
