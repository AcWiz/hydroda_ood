from pathlib import Path

import pytest


def test_hyperda_safe_diagnostic_entrypoint_declares_source_anchor_protocol():
    text = Path("run/hyperda_safe_us_r1_seed0.sh").read_text()

    assert "SAFE diagnostic" in text
    assert "Source-Anchored Few-Shot Operator Refinement" in text
    assert 'TARGET_REGION="${TARGET_REGION:-US-R1}"' in text
    assert 'SEED="${SEED:-0}"' in text
    assert 'K_LIST="${K_LIST:-0 4 12}"' in text
    assert "target_context=2015-2021 input-side only" in text
    assert "target_support=K labeled cycles" in text
    assert "target_val=unused_in_main_protocol" in text
    assert "target_eval=2023-2025 final offline evaluation" in text
    assert "model_selection_source=source_val_preregistered" in text
    assert "ADAPT_RECIPE=source_anchor" in text
    assert "ANCHOR_ALPHA_K4=0.75" in text
    assert "ANCHOR_ALPHA_K12=0.25" in text
    assert 'cd "$(dirname "$0")/.."' in text
    assert "run/phase5_hyperda_zero_few_shot_eval.sh" in text
    assert "phase7_hyperda_apo" not in text
    assert "bora_residual_adapter" not in text
    assert "surface_residual_ridge" not in text


@pytest.mark.skip(
    reason="run/stage3_calibrate_safe_policy_and_eval_us_r1_seed0.sh was intentionally "
    "retired from the active tree; the wrapper-contract assertion no longer applies."
)
def test_stage3_calibration_wrapper_defaults_to_m2_1_stable_mainline_checkpoint():
    text = Path("run/stage3_calibrate_safe_policy_and_eval_us_r1_seed0.sh").read_text()

    assert "M2_1_rank_gated_dora_stable" in text
    assert "resolve_auto_m2_1_source_checkpoint" in text
    assert "artifacts/runs/phase4_hyperda_staged_ablation/M2_1_rank_gated_dora_stable" in text
    assert "M2_1_source_checkpoint" in text
    assert "M2_rank_gated_dora/" not in text
    assert "M2_2_source_saliency_prior" not in text


def test_active_docs_define_hyperda_trust_and_demote_safe_while_retiring_phase6_phase7():
    combined = "\n".join(
        [
            Path("context/01_RESEARCH_CONTRACT.md").read_text(),
            Path("docs/COAUTHOR_CONTEXT.md").read_text(),
            Path("specs/hyperda_v4.yaml").read_text(),
            Path("run/README.md").read_text(),
        ]
    )

    assert "HyperDA-TRUST" in combined
    assert "Source-Manifold Trust-Routed Operator Generation" in combined
    assert "SAFE" in combined
    assert "diagnostic" in combined
    assert "rejected_to_k0_anchor" in combined
    assert "retired_failed_exploration_not_paper_main" in combined
    assert "phase7_hyperda_apo" in combined
    assert "phase6_surface_residual_ridge" in combined
    assert "phase6_bora_residual_adapter" in combined
    assert "not a paper-main method" in combined


def test_active_docs_fix_m2_1_as_stable_mainline_and_demote_other_source_priors():
    combined = "\n".join(
        [
            Path("context/01_RESEARCH_CONTRACT.md").read_text(),
            Path("docs/COAUTHOR_CONTEXT.md").read_text(),
            Path("specs/hyperda_v4.yaml").read_text(),
            Path("run/README.md").read_text(),
            Path("tasks/phase5_hyperda_safe_zero_few_shot.md").read_text(),
        ]
    )

    assert "stable rank-gated bounded-DoRA HyperDA prior" in combined
    assert "HyperDA Operator Generator" in combined
    assert "M2_1_rank_gated_dora_stable" in combined
    assert "shared_layer_aware_rank_gated_stable" in combined
    assert "dora_like_gain_bounded" in combined
    assert "temperature `2.0`" in combined
    assert "`USE_AMP=0`" in combined
    assert "`LR=2e-4`" in combined
    assert "M2_rank_gated_dora" in combined
    assert "retired_failed_exploration_not_paper_main" in combined
    assert "AMP skip/numerical failure" in combined
    assert "M2_2_source_saliency_prior" in combined
    assert "secondary diagnostic" in combined
    assert "M2_3_source_safe_residual_hyperda" in combined
    assert "source-safe residual" in combined.lower()
    assert "M2_5a_da_aware_prompt_only" in combined
    assert "negative_diagnostic_non_strict_prompt_only" in combined
    assert "source_val_improved_target_k0_degraded" in combined
    assert "M2_5b_da_aware_conservative_router" in combined
    assert "robust_input_side_da_diagnostics_raw" in combined
    assert "raw input-side `x`" in combined or "raw input-side" in combined
    assert "DA-aware robust diagnostics" in combined
    assert "soft metadata" in combined
    assert "Hessian/Fisher/top-parameter selection" in combined
    assert "future source-side ablation" in combined
    assert "M3_1_hyperda_trust_medium" in combined
    assert "Source-Manifold Trust" in combined


def test_phase6_phase7_untracked_code_entrypoints_are_removed_from_active_tree():
    retired_paths = [
        "configs/experiments/phase7_hyperda_apo_us_r1_seed0.yaml",
        "configs/phase6/bora_residual_adapter.yaml",
        "configs/phase6/surface_residual_ridge.yaml",
        "configs/phase6/surface_residual_ridge_dx.yaml",
        "hydroda/adaptation/bora_residual_adapter.py",
        "hydroda/adaptation/hyperda_apo.py",
        "hydroda/adaptation/surface_residual_diagnostics.py",
        "hydroda/adaptation/surface_residual_ridge.py",
        "hydroda/models/hyperda_apo.py",
        "run/phase6a_dx_surface_residual_ridge.sh",
        "run/phase6b_bora_residual_adapter.sh",
        "run/phase7_hyperda_apo_us_r1_seed0.sh",
        "scripts/eval/eval_bora_residual_adapter.py",
        "scripts/eval/eval_hyperda_apo.py",
        "scripts/eval/eval_hyperda_surface_residual_ridge.py",
        "scripts/eval/eval_phase6a_dx_surface_residual_ridge.py",
        "scripts/train/train_hyperda_apo.py",
        "tasks/BORA_HyperDA_Codex_Plan..md",
        "tasks/BORA_HyperDA_Codex_Plan_dx_phase6b.md",
        "tests/test_bora_residual_adapter.py",
        "tests/test_phase7_hyperda_apo.py",
        "tests/test_phase7_hyperda_apo_protocol_text.py",
        "tests/test_phase7_hyperda_apo_scripts.py",
        "tests/test_surface_residual_ridge.py",
        "tests/test_surface_residual_ridge_diagnostics.py",
    ]

    assert not [path for path in retired_paths if Path(path).exists()]
