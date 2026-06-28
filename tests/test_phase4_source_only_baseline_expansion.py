from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from hydroda.baselines.registry import assert_allowed_for_table


@pytest.mark.parametrize(
    "method_id",
    [
        "swad_source_pooled_global_backbone",
        "mixstyle_source_pooled_global_backbone",
        "disam_source_domain_sharpness_alignment",
        "udim_unknown_domain_inconsistency_minimization",
        "moment_alignment_source_domain_invariance",
        "identify_unlearn_source_domain_gradient_ascent",
    ],
)
def test_new_k0_domain_generalization_baselines_are_paper_main_candidates(method_id):
    assert_allowed_for_table(method_id, "paper_main")


@pytest.mark.parametrize(
    "method_id",
    [
        "deep_coral_target_context_alignment",
        "ssa_reg_target_context_subspace_alignment",
        "tca_target_context_correlation_alignment",
        "self_bootstrap_target_context_consistency_tta",
        "weatherpeft_weather_fm_future_baseline",
    ],
)
def test_old_or_future_methods_are_not_paper_main_candidates(method_id):
    with pytest.raises(ValueError, match="paper_main"):
        assert_allowed_for_table(method_id, "paper_main")


def test_hyperda_rise_remains_diagnostic_not_paper_main():
    with pytest.raises(ValueError, match="paper_main"):
        assert_allowed_for_table("hyperda_rise_k0_context_router", "paper_main")


def test_baseline_spec_lists_dg_methods_with_v4_4_split_contract():
    spec = yaml.safe_load(Path("specs/baselines.yaml").read_text(encoding="utf-8"))
    main = spec["paper_main_baselines"]["zero_few_shot_generalization"]

    for method_id in [
        "swad_source_pooled_global_backbone",
        "mixstyle_source_pooled_global_backbone",
        "disam_source_domain_sharpness_alignment",
        "udim_unknown_domain_inconsistency_minimization",
        "moment_alignment_source_domain_invariance",
        "identify_unlearn_source_domain_gradient_ascent",
    ]:
        assert method_id in main
        definition = spec["baseline_definitions"][method_id]
        assert definition["source_fit"] == "2015-2021"
        assert definition["source_val"] == 2022
        assert definition["target_eval"] == "2023-2025"
        assert definition["paper_table_eligible"] is True
        assert definition["predicts"] == "DA_increment"

    for method_id in [
        "disam_source_domain_sharpness_alignment",
        "udim_unknown_domain_inconsistency_minimization",
        "moment_alignment_source_domain_invariance",
        "identify_unlearn_source_domain_gradient_ascent",
    ]:
        definition = spec["baseline_definitions"][method_id]
        assert definition["status"] == "paper_main_candidate_source_only_dg"
        assert definition["target_context_allowed"] == "none"
        assert definition["target_context_usage"] == "not_used"
        assert definition["target_eval_usage"] == "final_offline_evaluation_only"
        assert definition["model_selection_source"] == "source_val_2022"
        assert definition["checkpoint_selection_metric"] == "source_val_loss"
        assert definition["source_region_episode_batching"] is False
        assert definition["source_domain_grouping"] == "pooled_sample_region_masks"

    iu_params = spec["baseline_definitions"]["identify_unlearn_source_domain_gradient_ascent"][
        "default_parameters"
    ]
    assert iu_params["iu_lambda"] == 0.001
    assert iu_params["iu_score_cap"] == 10.0
    assert iu_params["iu_objective"] == "bounded_domain_specific_feature_penalty"

    coral = spec["baseline_definitions"]["deep_coral_target_context_alignment"]
    assert "deep_coral_target_context_alignment" not in main
    assert coral["paper_table_eligible"] is False
    assert coral["status"] == "internal_diagnostic_old_baseline_not_paper_main"

    for method_id in [
        "ssa_reg_target_context_subspace_alignment",
        "tca_target_context_correlation_alignment",
        "self_bootstrap_target_context_consistency_tta",
    ]:
        definition = spec["baseline_definitions"][method_id]
        assert method_id not in main
        assert "diagnostic" in definition["status"]
        assert definition["paper_table_eligible"] is False
        assert definition["target_context_allowed"] == "2015-2021 input-side target_context only"
        assert definition["target_context_labels_allowed"] is False
        assert definition["target_eval_usage"] == "final_offline_evaluation_only"

    weatherpeft = spec["baseline_definitions"]["weatherpeft_weather_fm_future_baseline"]
    assert weatherpeft["paper_table_eligible"] is False
    assert weatherpeft["status"] == "related_work_future_weather_fm_baseline"


def test_phase4_dg_wrapper_dry_run_lists_requested_methods():
    script = Path("run/phase4_dg_baselines_us_r1_seed0.sh")
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "--method-list",
            "swad",
            "mixstyle",
            "disam",
            "udim",
            "moment_align",
            "iu",
            "ssa_reg",
            "tca",
            "self_bootstrap",
            "--target-region",
            "US-R2",
            "--seed",
            "3",
            "--cuda-device",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "DRY_RUN=1" in result.stdout
    assert "target_region=US-R2" in result.stdout
    assert "seed=3" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=0" in result.stdout
    assert "--dg_method swad" in result.stdout
    assert "--dg_method mixstyle" in result.stdout
    assert "--dg_method disam" in result.stdout
    assert "--dg_method udim" in result.stdout
    assert "--dg_method moment_align" in result.stdout
    assert "--dg_method iu" in result.stdout
    assert "--dg_method ssa_reg" in result.stdout
    assert "--dg_method tca" in result.stdout
    assert "--dg_method self_bootstrap" in result.stdout
    assert "swad_source_pooled_global_backbone" in result.stdout
    assert "mixstyle_source_pooled_global_backbone" in result.stdout
    assert "disam_source_domain_sharpness_alignment" in result.stdout
    assert "udim_unknown_domain_inconsistency_minimization" in result.stdout
    assert "moment_alignment_source_domain_invariance" in result.stdout
    assert "identify_unlearn_source_domain_gradient_ascent" in result.stdout
    assert "ssa_reg_target_context_subspace_alignment" in result.stdout
    assert "tca_target_context_correlation_alignment" in result.stdout
    assert "self_bootstrap_target_context_consistency_tta" in result.stdout
    assert "deep_coral_target_context_alignment" not in result.stdout


def test_phase4_dg_wrapper_default_dry_run_lists_source_only_dg_methods():
    script = Path("run/phase4_dg_baselines_us_r1_seed0.sh")
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "--target-region",
            "US-R1",
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "methods=swad mixstyle disam udim moment_align iu" in result.stdout
    assert "disam_source_domain_sharpness_alignment" in result.stdout
    assert "udim_unknown_domain_inconsistency_minimization" in result.stdout
    assert "moment_alignment_source_domain_invariance" in result.stdout
    assert "identify_unlearn_source_domain_gradient_ascent" in result.stdout
    assert "ssa_reg_target_context_subspace_alignment" not in result.stdout
    assert "tca_target_context_correlation_alignment" not in result.stdout
    assert "self_bootstrap_target_context_consistency_tta" not in result.stdout
    assert "deep_coral_target_context_alignment" not in result.stdout


def test_phase4_dg_wrapper_full_mode_defaults_to_batch_size_8():
    script = Path("run/phase4_dg_baselines_us_r1_seed0.sh")
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "--full",
            "--method-list",
            "tca",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "run_mode=full" in result.stdout
    assert "--batch_size 8" in result.stdout


def test_phase4_dg_wrapper_dry_run_for_new_source_only_methods_prints_parameters():
    script = Path("run/phase4_dg_baselines_us_r1_seed0.sh")
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "--method-list",
            "disam",
            "udim",
            "moment_align",
            "iu",
            "--target-region",
            "US-R1",
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "methods=disam udim moment_align iu" in result.stdout
    assert "method_id=disam_source_domain_sharpness_alignment" in result.stdout
    assert "method_id=udim_unknown_domain_inconsistency_minimization" in result.stdout
    assert "method_id=moment_alignment_source_domain_invariance" in result.stdout
    assert "method_id=identify_unlearn_source_domain_gradient_ascent" in result.stdout
    assert "--dg_method disam" in result.stdout
    assert "--dg_method udim" in result.stdout
    assert "--dg_method moment_align" in result.stdout
    assert "--dg_method iu" in result.stdout
    assert "--disam_rho 0.05" in result.stdout
    assert "--disam_lambda 0.1" in result.stdout
    assert "--udim_rho 0.05" in result.stdout
    assert "--udim_lambda 0.1" in result.stdout
    assert "--moment_align_lambda 0.01" in result.stdout
    assert "--moment_align_feature_layer bottleneck" in result.stdout
    assert "--moment_align_order 2" in result.stdout
    assert "--iu_lambda 0.001" in result.stdout
    assert "--iu_feature_layer bottleneck" in result.stdout
    assert "--iu_top_fraction 0.25" in result.stdout
    assert "--iu_sample_top_fraction 0.5" in result.stdout
    assert "--iu_score_cap 10.0" in result.stdout
    assert "target_context=unused for default source-only DG methods" in result.stdout
    assert "source_domain_grouping=pooled sample masks; no 5x source-region episode expansion" in result.stdout
    assert "target_context_alignment" not in result.stdout
