import os
import subprocess
from pathlib import Path


def test_research_plan_protocol_block_uses_zero_few_shot_dates():
    text = Path("完整研究计划方案.md").read_text()

    assert "Source validation:          2021" not in text
    assert "Target train / adaptation:    2022" not in text
    assert "target 2022 input-side prompt" not in text
    assert "Source fit/train:           2015-2021" in text
    assert "Source validation:          2022" in text
    contract = Path("context/01_RESEARCH_CONTRACT.md").read_text()
    assert "target_full_train" in contract
    assert "legacy/internal" in contract
    assert "Target eval/test:           2023-2025" in text


def test_project_experiment_plan_documents_use_zero_few_shot_main_protocol():
    docs = [
        Path("context/01_RESEARCH_CONTRACT.md").read_text(),
        Path("docs/COAUTHOR_CONTEXT.md").read_text(),
        Path("specs/protocol_v4.yaml").read_text(),
        Path("specs/hyperda_v4.yaml").read_text(),
    ]
    combined = "\n".join(docs)

    assert "target_context" in combined
    assert "target_support" in combined
    assert "few_shot_k4" in combined
    assert "few_shot_k12" in combined
    assert "unused_in_main_protocol" in combined
    assert "adapter coefficient residual" in combined or "adapter_coefficient_residual" in combined
    assert "safe_policy.json" in combined
    assert "policy_source=source_side_episode_calibration" in combined
    assert "adapt_mix_rho" in combined


def test_logging_and_metrics_docs_name_main_hyperda_methods_and_metadata():
    combined = "\n".join(
        [
            Path("docs/EXPERIMENT_LOGGING.md").read_text(),
            Path("specs/metrics.yaml").read_text(),
            Path("run/README.md").read_text(),
        ]
    )

    for method in [
        "forecast_only",
        "source_only_backbone",
        "prompt_conditioned_shared_backbone",
        "hyperda_zero_shot_context",
        "hyperda_safe_few_shot_k4",
        "hyperda_safe_few_shot_k12",
    ]:
        assert method in combined
    assert "target_context_prompt_state" in combined
    assert "target_context_dates_hash" in combined
    assert "target_support_dates_hash" in combined
    assert "model_selection_source" in combined
    assert "source_val_preregistered" in combined
    assert "SAFE_POLICY_JSON" in combined
    assert "safe_policy.json" in combined
    assert "REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT=0" in combined
    assert "policy_source" in combined
    assert "source_side_episode_calibration" in combined
    assert "adapt_mix_rho" in combined


def test_zero_few_shot_wrapper_declares_main_protocol():
    text = Path("run/phase5_hyperda_zero_few_shot.sh").read_text()

    assert "US_loro_zero_few_shot_splits.json" in text
    assert "--K" in text
    assert "--adaptation_setting" in text
    assert "target_val=unused_in_main_protocol" in text
    assert "model_selection_source=source_val_preregistered" in text
    assert "artifacts/runs/phase4_hyperda_staged/${TARGET_REGION}" in text
    assert "source_stage_checkpoint_provenance=phase4_hyperda_staged" in text
    assert "scripts/train/train_hyperda_few_shot_adapt.py" in text
    assert "safe_operator" in text
    assert 'ADAPT_SCOPE="${ADAPT_SCOPE:-safe_operator}"' in text
    assert 'AUDIT_IDENTITY="${AUDIT_IDENTITY:-0}"' in text
    assert 'TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES:-0}"' in text
    assert '--target_context_max_samples "${TARGET_CONTEXT_MAX_SAMPLES}"' in text
    assert 'ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-${BATCH_SIZE:-8}}"' in text
    assert "--adapt_scope" in text
    assert "--adapt_solver adamw" in text
    assert "--audit_identity" in text
    assert 'FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN:-0}"' in text
    assert '$(if [[ "${FREEZE_MONTHLY_GAIN}" == "1"' in text
    active_log = text.split('echo "============================================"', 1)[1]
    for legacy_token in [
        "ridge_lambda=",
        "trust_region_mode=",
        "--enable_target_spatial_refine",
    ]:
        assert legacy_token not in active_log


def test_zero_few_shot_eval_wrapper_runs_all_main_k_settings_on_target_eval():
    text = Path("run/phase5_hyperda_zero_few_shot_eval.sh").read_text()

    assert 'K_LIST="${K_LIST:-0 4 12}"' in text
    assert 'OUTPUT_BASE="${5:-' in text
    assert 'artifacts/runs/phase4_hyperda_staged/${TARGET_REGION}' in text
    assert '-path "*s${SEED}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt"' in text
    assert "run/phase5_hyperda_zero_few_shot.sh" in text
    assert "checkpoint_final_preregistered.pt" in text
    assert "scripts/eval/evaluate_checkpoint.py" in text
    assert "--predictor_type hyperda_target_adapt" in text
    assert "--split_type target_eval" in text
    assert 'EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"' in text
    assert 'ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-${BATCH_SIZE:-8}}"' in text
    assert 'EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE:-8}}"' in text
    assert 'ADAPT_RECIPE="${ADAPT_RECIPE:-source_anchor}"' in text
    assert 'ADAPT_SCOPE="${ADAPT_SCOPE:-safe_operator}"' in text
    assert 'ANCHOR_ALPHA_K4="${ANCHOR_ALPHA_K4:-0.75}"' in text
    assert 'ANCHOR_ALPHA_K12="${ANCHOR_ALPHA_K12:-0.25}"' in text
    assert 'LR_K12="${LR_K12:-3e-4}"' in text
    assert 'MAX_STEPS_K12="${MAX_STEPS_K12:-80}"' in text
    assert 'SAFE_POLICY_JSON="${SAFE_POLICY_JSON:-}"' in text
    assert 'SAFE_POLICY_JSON="${SAFE_POLICY_JSON}"' in text
    assert 'REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-1}"' in text
    assert 'REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_FOR_K}"' in text
    assert 'ADAPT_MIX_RHO_WAS_SET="${ADAPT_MIX_RHO+x}"' in text
    assert 'FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN:-0}"' in text
    assert 'TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES:-0}"' in text
    assert 'TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES}"' in text
    assert 'STAGE3_K0_CONTEXT_SHRINKAGE="${STAGE3_K0_CONTEXT_SHRINKAGE:-0}"' in text
    assert '--stage3_k0_context_shrinkage' in text
    assert '--stage3_k0_context_shrinkage_rho_cap "${STAGE3_K0_CONTEXT_SHRINKAGE_RHO_CAP}"' in text
    assert '--stage3_k0_context_shrinkage_policy "${STAGE3_K0_CONTEXT_SHRINKAGE_POLICY}"' in text
    assert '--stage3_k0_context_shrinkage_surface_rho_cap "${STAGE3_K0_CONTEXT_SHRINKAGE_SURFACE_RHO_CAP}"' in text
    assert '--stage3_k0_context_shrinkage_rootzone_rho_cap "${STAGE3_K0_CONTEXT_SHRINKAGE_ROOTZONE_RHO_CAP}"' in text
    assert '--stage3_k0_context_shrinkage_policy_json "${STAGE3_K0_CONTEXT_SHRINKAGE_POLICY_JSON}"' in text
    assert 'FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN}"' in text
    assert "resolve_policy_adapt_mix_rho" in text
    assert "policy-derived adapt_mix_rho" in text
    assert 'ADAPT_MIX_RHO_FOR_K="$(resolve_policy_adapt_mix_rho "${SAFE_POLICY_JSON}" "${ADAPTATION_SETTING}" "${K}")"' in text
    assert '--adapt_mix_rho "${ADAPT_MIX_RHO_FOR_K}"' in text
    assert "policy_source=source_side_episode_calibration" in text
    assert "--max_samples" in text
    assert 'ADAPTATION_MAX_STEPS="0"' in text
    assert 'ADAPTATION_MAX_STEPS="${ADAPT_MAX_STEPS_OVERRIDE:-${MAX_STEPS_K4:-${MAX_STEPS:-100}}}"' in text
    assert 'ADAPTATION_MAX_STEPS="${ADAPT_MAX_STEPS_OVERRIDE:-${MAX_STEPS_K12:-${MAX_STEPS:-80}}}"' in text
    assert 'ADAPT_MAX_STEPS="${ADAPTATION_MAX_STEPS}"' in text
    assert 'BATCH_SIZE="${ADAPT_BATCH_SIZE}" SPLITS_JSON="${SPLITS_JSON}" OUTPUT_DIR="${ADAPT_DIR}"' in text
    assert 'ANCHOR_ALPHA="${ANCHOR_ALPHA}"' in text
    assert '--batch_size "${EVAL_BATCH_SIZE}"' in text
    assert "print_markdown_table" in text
    assert "Quick target_eval WRMSE table" in text
    assert "overview.csv" in text
    assert "overview.md" in text
    assert "overview.json" in text
    assert "metrics_by_region.csv" in text
    assert "metrics_by_season.csv" in text
    assert "surface_corr_latw" in text
    assert "rootzone_corr_latw" in text
    assert "skill_global" in text
    assert "rmse_mean" in text
    assert "target_eval_dates_hash" in text
    assert "target_support_dates" in text
    assert "k4_support_subset_of_k12" in text
    assert "support_batch_count" in text
    assert "effective_support_passes" in text
    assert "adapt_batch_size" in text
    assert "eval_batch_size" in text
    assert "adaptation_steps" in text
    assert "lr" in text
    assert "weight_decay" in text
    assert "trainable_parameter_count" in text
    assert "support_final_loss" in text
    assert "support_loss_delta" in text
    assert "anchor_alpha" in text
    assert "target_parameter_l2_drift" in text
    assert 'AUDIT_IDENTITY="${AUDIT_IDENTITY:-0}"' in text
    assert 'AUDIT_IDENTITY_TOLERANCE="${AUDIT_IDENTITY_TOLERANCE:-1e-8}"' in text
    assert "identity_audit.json" in text
    assert "compare_identity_audit" in text
    assert 'ADAPT_SCOPE_FOR_K="none"' in text
    assert 'ADAPTATION_MAX_STEPS="0"' in text
    assert 'ANCHOR_ALPHA="0.0"' in text
    assert 'AUDIT_IDENTITY_FOR_K="0"' in text
    assert 'AUDIT_IDENTITY_FOR_K="1"' in text
    assert 'AUDIT_IDENTITY="${AUDIT_IDENTITY_FOR_K}"' in text
    assert 'ADAPT_SCOPE="${ADAPT_SCOPE_FOR_K}"' in text
    assert "source_checkpoint_sha256" in text
    assert "target_labels_loaded_for_adaptation" in text
    assert "target_labels_used_for_adaptation" in text
    assert "target_support_count" in text
    assert "def stage3_decision_from_artifacts" in text
    assert "def stage3_overview_status" in text
    assert "if isinstance(value, bool):" in text
    assert "rejected_to_k0_anchor" in text
    assert "stage3_posterior_decision" in text
    assert "stage3_acceptance_basis" in text
    assert "support_gate_status" in text
    assert "support_only_gate_status" in text
    assert "paper_facing_run" in text
    assert "diagnostic_run_reason" in text
    assert "safe_policy_json_sha256" in text
    assert "k0_anchor_state_hash" in text
    assert "K-shot rows marked rejected_to_k0_anchor are K0-equivalent fallback" in text
    assert "results/checkpoint_*/target_eval/{target_region}/summary.json" in text
    assert "return json.load(f), str(path)" in text
    assert "return load_json(path), str(path)" not in text
    terminal_block = text.split("terminal_columns = [", 1)[1].split("]", 1)[0]
    assert '("decision", "stage3_posterior_decision")' in terminal_block
    assert '("paper", "paper_facing_run")' in terminal_block
    assert '("rho", "adapt_mix_rho")' in terminal_block
    assert '("surface_WRMSE", "surface_rmse_latw")' in terminal_block
    assert '("rootzone_WRMSE", "rootzone_rmse_latw")' in terminal_block
    assert "surface_skill_primary" not in terminal_block
    assert "surface_corr_latw" not in terminal_block
    assert "rootzone_skill_primary" not in terminal_block
    assert "rootzone_corr_latw" not in terminal_block
    assert "n_samples_evaluated" not in terminal_block

    for k, setting in [
        ("0", "zero_shot_context"),
        ("4", "few_shot_k4"),
        ("12", "few_shot_k12"),
    ]:
        assert f'if [[ "${{K}}" == "{k}" ]]' in text or f'elif [[ "${{K}}" == "{k}" ]]' in text
        assert f'ADAPTATION_SETTING="{setting}"' in text

    forbidden = [
        "--target_train_residual_gain_calibration",
        "--allow_legacy_target_label_calibration",
        "--adaptation_setting target_full_train",
        "target_full_train",
        "target_val",
        "RIDGE_LAMBDA",
            "RIDGE_MAX_FEATURE_PIXELS",
            "TRUST_REGION_MODE",
            "ADAPT_SOLVER",
        "ridge_coefficient_norm",
        "target_spatial_refine",
    ]
    for token in forbidden:
        assert token not in text


def test_hyperda_safe_paper_entrypoint_requires_source_calibrated_policy_for_kshot():
    text = Path("run/hyperda_safe_us_r1_seed0.sh").read_text()

    assert "HyperDA-SAFE paper-facing US-R1 seed0 entrypoint" in text
    assert 'SAFE_POLICY_JSON="${SAFE_POLICY_JSON:-}"' in text
    assert 'REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-1}"' in text
    assert 'export SAFE_POLICY_JSON="${SAFE_POLICY_JSON}"' in text
    assert 'export REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT}"' in text
    assert "source_side_episode_calibration" in text
    assert "safe_policy.json" in text
    assert "run/phase5_hyperda_zero_few_shot_eval.sh" in text


def test_stage3_calibrate_safe_policy_wrapper_runs_source_calibration_then_us_r1_eval():
    text = Path("run/stage3_calibrate_safe_policy_and_eval_us_r1_seed0.sh").read_text()

    assert "Stage 3 source-side SAFE policy calibration + US-R1 seed0 eval" in text
    assert 'TARGET_REGION="${TARGET_REGION:-US-R1}"' in text
    assert 'SEED="${SEED:-0}"' in text
    assert 'SOURCE_QUERY_MAX_SAMPLES="${SOURCE_QUERY_MAX_SAMPLES:-256}"' in text
    assert 'TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES:-0}"' in text
    assert 'EVIDENCE_LEVEL="${EVIDENCE_LEVEL:-weaker}"' in text
    assert "scripts/eval/run_stage3_source_safe_policy_calibration.py" in text
    assert "scripts/eval/calibrate_source_safe_guard.py" in text
    assert 'CANDIDATE_SET="${CANDIDATE_SET:-stage3_k0_m2_4a_variable_v1}"' in text
    assert '--candidate_set "${CANDIDATE_SET}"' in text
    assert "--allow_in_checkpoint_source_episodes" in text
    assert "EVIDENCE_LEVEL=strict" in text
    assert "SOURCE_HELDOUT_CHECKPOINT_MAP" in text
    assert 'STAGE3_K0_CONTEXT_SHRINKAGE_POLICY="${STAGE3_K0_CONTEXT_SHRINKAGE_POLICY:-source_episode_calibrated_v1}"' in text
    assert 'STAGE3_K0_CONTEXT_SHRINKAGE_POLICY_JSON="${STAGE3_K0_CONTEXT_SHRINKAGE_POLICY_JSON:-${CALIBRATION_DIR}/safe_policy.json}"' in text
    assert "bash run/hyperda_safe_us_r1_seed0.sh" in text


def test_stage3_source_safe_row_builder_has_source_only_contract():
    text = Path("scripts/eval/run_stage3_source_safe_policy_calibration.py").read_text()

    assert "--active_region_override" in text
    assert "--split_type source_val" in text
    assert "--prediction_record_path" in text
    assert "--target_context_max_samples" in text
    assert "source_safe_candidate_rows.csv" in text
    assert "calibration_rows.csv" in text
    assert "source_val_pseudo_query" in text
    assert "target_eval_usage" in text
    assert "unused_in_main_protocol" in text
    assert "allow_in_checkpoint_source_episodes" in text
    assert "strict" in text


def test_stage3_hyperda_posterior_eval_wrapper_is_safe_entrypoint():
    text = Path("run/stage3_hyperda_posterior_eval.sh").read_text()

    assert 'MODE="${MODE:-full}"' in text
    assert 'STAGE3_STRICT_PAPER_POLICY="${STAGE3_STRICT_PAPER_POLICY:-1}"' in text
    assert 'STAGE3_POSTERIOR_POLICY="${STAGE3_POSTERIOR_POLICY:-conservative_coeff_posterior}"' in text
    assert 'ADAPT_SCOPE="${ADAPT_SCOPE:-coeff_only}"' in text
    assert 'FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN:-1}"' in text
    assert "adapter_coefficient_residuals_only" in text
    assert 'SUPPORT_GATE="${SUPPORT_GATE:-auto}"' in text
    assert "safe_operator_ablation" in text
    assert 'K_LIST="${K_LIST:-0}"' in text
    assert 'K_LIST="${K_LIST:-0 4 12}"' in text
    assert 'REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-1}"' in text
    assert 'REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-0}"' in text
    assert "diagnostic_no_source_safe_policy_json" in text
    assert "STAGE3_STRICT_PAPER_POLICY=1" in text
    assert "stage3_posterior_state_dict" in text
    assert "stage3_source_prior_unchanged" in text
    assert 'STAGE3_K0_CONTEXT_SHRINKAGE="${STAGE3_K0_CONTEXT_SHRINKAGE:-0}"' in text
    assert 'STAGE3_K0_CONTEXT_SHRINKAGE_RHO_CAP="${STAGE3_K0_CONTEXT_SHRINKAGE_RHO_CAP:-1.0}"' in text
    assert 'STAGE3_K0_CONTEXT_SHRINKAGE_POLICY="${STAGE3_K0_CONTEXT_SHRINKAGE_POLICY:-variable_reliability_v1}"' in text
    assert 'STAGE3_K0_CONTEXT_SHRINKAGE_SURFACE_RHO_CAP="${STAGE3_K0_CONTEXT_SHRINKAGE_SURFACE_RHO_CAP:-${STAGE3_K0_CONTEXT_SHRINKAGE_RHO_CAP}}"' in text
    assert 'STAGE3_K0_CONTEXT_SHRINKAGE_ROOTZONE_RHO_CAP="${STAGE3_K0_CONTEXT_SHRINKAGE_ROOTZONE_RHO_CAP:-${STAGE3_K0_CONTEXT_SHRINKAGE_RHO_CAP}}"' in text
    assert "run/phase5_hyperda_zero_few_shot_eval.sh" in text
    assert "target_context=2015-2021 input-side only" in text
    assert "target_val=unused_in_main_protocol" in text
    assert "target_eval=2023-2025 final offline evaluation" in text


def test_stage3_k0_m2_4_is_documented_as_target_context_diagnostic_not_source_stage():
    combined = "\n".join(
        [
            Path("run/README.md").read_text(),
            Path("docs/COAUTHOR_CONTEXT.md").read_text(),
            Path("context/01_RESEARCH_CONTRACT.md").read_text(),
            Path("specs/hyperda_v4.yaml").read_text(),
            Path("tasks/phase5_hyperda_safe_zero_few_shot.md").read_text(),
        ]
    )

    assert "Stage 3 K=0 target-context conservative shrinkage diagnostic" in combined
    assert "not a source-stage ablation" in combined
    assert "freeze the M2.1 prior" in combined
    assert "extra_source_finetune: false" in combined
    assert "target_labels_used_for_adaptation: false" in combined
    assert "target_eval_input_stats_used_for_update: false" in combined
    assert "M2.4a" in combined
    assert "source_episode_calibrated_v1" in combined
    assert "STAGE3_K0_CONTEXT_SHRINKAGE=1" in combined


def test_stage3_hyperda_posterior_smoke_wrapper_bakes_diagnostic_defaults():
    text = Path("run/stage3_hyperda_posterior_smoke.sh").read_text()

    assert 'MODE="${MODE:-smoke}"' in text
    assert 'K_LIST="${K_LIST:-0}"' in text
    assert 'EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-2}"' in text
    assert 'BATCH_SIZE="${BATCH_SIZE:-1}"' in text
    assert 'ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-1}"' in text
    assert 'EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"' in text
    assert 'STAGE3_STRICT_PAPER_POLICY="${STAGE3_STRICT_PAPER_POLICY:-0}"' in text
    assert 'REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-0}"' in text
    assert "snapshot" in text
    assert "run/stage3_hyperda_posterior_eval.sh" in text


def test_stage3_hyperda_posterior_full_inference_wrapper_bakes_diagnostic_defaults():
    text = Path("run/stage3_hyperda_posterior_full_inference.sh").read_text()

    assert 'MODE="${MODE:-full}"' in text
    assert 'K_LIST="${K_LIST:-0 4 12}"' in text
    assert 'EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"' in text
    assert 'BATCH_SIZE="${BATCH_SIZE:-1}"' in text
    assert 'ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-1}"' in text
    assert 'EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"' in text
    assert 'STAGE3_STRICT_PAPER_POLICY="${STAGE3_STRICT_PAPER_POLICY:-0}"' in text
    assert 'REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-0}"' in text
    assert 'STAGE3_K0_CONTEXT_SHRINKAGE="${STAGE3_K0_CONTEXT_SHRINKAGE:-0}"' in text
    assert 'STAGE3_K0_CONTEXT_SHRINKAGE_RHO_CAP="${STAGE3_K0_CONTEXT_SHRINKAGE_RHO_CAP:-1.0}"' in text
    assert 'STAGE3_K0_CONTEXT_SHRINKAGE_POLICY="${STAGE3_K0_CONTEXT_SHRINKAGE_POLICY:-variable_reliability_v1}"' in text
    assert "diagnostic_full_inference" in text
    assert "snapshot" in text
    assert "run/stage3_hyperda_posterior_eval.sh" in text


def test_prompt_conditioned_inference_wrapper_uses_target_eval_protocol():
    text = Path("run/phase4_prompt_conditioned_inference.sh").read_text()

    assert "context2022_query2023_2025_k0_4_12" not in text
    assert "--K 0" in text
    assert "--split_type target_eval" in text
    assert "target_query" not in text
    assert "checkpoint_best_source_val_transfer_safe_score.pt" in text
    assert "--split_type target_eval" in text
    assert "--target_context_prompt" in text


def test_hyperda_train_wrapper_is_source_stage_prior():
    text = Path("run/phase4_hyperda.sh").read_text()
    staged = Path("run/phase4_hyperda_staged.sh").read_text()
    readme = Path("run/README.md").read_text()

    assert "compatibility wrapper" in text
    assert "phase4_hyperda_staged.sh" in text
    assert "exec bash" in text
    assert '[[ "${1:-}" =~ ^[A-Z]{2}-R[0-9]+$ ]]' in text
    assert 'phase4_hyperda_staged.sh" auto "$@"' in text
    assert "US_loro_kdate_splits.json" not in text
    assert "phase4_hyperda_staged.sh" in readme
    assert "Train staged HyperDA source prior" in readme
    assert "phase4_hyperda_staged.sh auto US-R1 0 1" in readme
    assert "Stage 1 source-only checkpoint" in readme
    assert "source_base_frozen_adapter_film" in staged


def test_staged_hyperda_wrapper_declares_frozen_source_base_stage2():
    text = Path("run/phase4_hyperda_staged.sh").read_text()
    current = Path("run/phase4_hyperda.sh").read_text()

    assert "source_base_frozen_adapter_film" not in current
    assert "artifacts/splits/US_loro_zero_few_shot_splits.json" in text
    assert "artifacts/runs/phase4_hyperda_staged" in text
    assert "--init_from_source_base_checkpoint" in text
    assert "--trainable_scope source_base_frozen_adapter_film" in text
    assert "--model_type hyperda_basis_adapter" in text
    assert "--selection_metric source_val_transfer_safe_score" in text
    assert "checkpoint_best_source_val_safe_score.pt" in text
    assert "artifacts/runs/phase4_source_only" in text
    assert "--dry-run" in text
    assert "target_val=unused_in_main_protocol" in text
    assert "target_labels=none" in text
    assert "target_eval_input_stats_used_for_update=false" in text
    assert "stage1_method=source_pooled_global_backbone" in text
    assert "stage2_trainable=prompt_encoder,film,basis_adapter_generation" in text
    assert "stage2_frozen=source_base_backbone_and_head" in text
    assert 'SOURCE_EPISODE_PROMPT_POLICY="${SOURCE_EPISODE_PROMPT_POLICY:-context_monthly_prototype}"' in text
    assert 'SOURCE_ANCHOR_BLEND_CALIBRATION="${SOURCE_ANCHOR_BLEND_CALIBRATION:-1}"' in text
    assert 'HYPER_OUTPUT_HEAD_RESIDUAL="${HYPER_OUTPUT_HEAD_RESIDUAL:-1}"' in text
    assert 'ZERO_SHOT_PRIOR_FORM="${ZERO_SHOT_PRIOR_FORM:-source_base_residual_reliability_gated}"' in text
    assert 'SOURCE_RESIDUAL_RHO="${SOURCE_RESIDUAL_RHO:-1.0}"' in text
    assert 'SOURCE_RESIDUAL_GATE="${SOURCE_RESIDUAL_GATE:-prompt_reliability_scalar}"' in text
    assert 'DATASET_BACKEND="${DATASET_BACKEND:-auto}"' in text
    assert 'TENSOR_CACHE_DIR="${TENSOR_CACHE_DIR:-artifacts/region_crops/US}"' in text
    assert 'NUM_WORKERS="${NUM_WORKERS:-0}"' in text
    assert 'EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-1}"' in text
    assert 'LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-100}"' in text
    assert 'TENSOR_CACHE_LOAD_MODE="${TENSOR_CACHE_LOAD_MODE:-eager}"' in text
    assert 'TRAIN_BATCH_SAMPLER="${TRAIN_BATCH_SAMPLER:-random}"' in text
    assert 'RESOLVED_DATASET_BACKEND="tensor_cache"' in text
    assert "source_episode_prompt_policy=${SOURCE_EPISODE_PROMPT_POLICY}" in text
    assert "source_anchor_blend_calibration=${SOURCE_ANCHOR_BLEND_CALIBRATION}" in text
    assert "hyper_output_head_residual=${HYPER_OUTPUT_HEAD_RESIDUAL}" in text
    assert "zero_shot_prior_form=${ZERO_SHOT_PRIOR_FORM}" in text
    assert "source_residual_rho=${SOURCE_RESIDUAL_RHO}" in text
    assert "dataset_backend=${RESOLVED_DATASET_BACKEND}" in text
    assert "tensor_cache_load_mode=${TENSOR_CACHE_LOAD_MODE}" in text
    assert "train_batch_sampler=${TRAIN_BATCH_SAMPLER}" in text
    assert "--zero_shot_prior_form" in text
    assert "--source_residual_rho" in text
    assert "--source_residual_gate" in text
    assert "--dataset_backend" in text
    assert "--tensor_cache_dir" in text
    assert "--max_year_cache_entries" in text
    assert "--tensor_cache_load_mode" in text
    assert "--train_batch_sampler" in text
    assert '--num_workers "${NUM_WORKERS}"' in text
    assert '--log_every_steps "${LOG_EVERY_STEPS}"' in text
    assert '--eval_every_epochs "${EVAL_EVERY_EPOCHS}"' in text
    assert 'RUN_NAME="${RUN_NAME:-phase4_hyperda_staged_${TARGET_REGION}_s${SEED}_${TIMESTAMP}}"' in text


def test_staged_hyperda_fast_trial_wrapper_is_separate_tensor_cache_entrypoint():
    text = Path("run/phase4_hyperda_staged_fast_trial.sh").read_text()
    staged = Path("run/phase4_hyperda_staged.sh").read_text()

    assert "phase4_hyperda_staged_fast_trial" in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/runs/phase4_hyperda_staged_fast_trial}"' in text
    assert 'DATASET_BACKEND="${DATASET_BACKEND:-auto}"' in text
    assert 'NUM_WORKERS="${NUM_WORKERS:-2}"' in text
    assert 'MAX_YEAR_CACHE_ENTRIES="${MAX_YEAR_CACHE_ENTRIES:-2}"' in text
    assert 'TENSOR_CACHE_LOAD_MODE="${TENSOR_CACHE_LOAD_MODE:-mmap}"' in text
    assert 'TRAIN_BATCH_SAMPLER="${TRAIN_BATCH_SAMPLER:-source_region_year_grouped}"' in text
    assert 'BATCH_SIZE="${BATCH_SIZE:-64}"' in text
    assert 'ACCUM_STEPS="${ACCUM_STEPS:-2}"' in text
    assert 'EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-5}"' in text
    assert 'LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-200}"' in text
    assert 'SOURCE_PROTOTYPE_CACHE_MODE="${SOURCE_PROTOTYPE_CACHE_MODE:-read_write}"' in text
    assert 'SOURCE_PROTOTYPE_CACHE_DIR="${SOURCE_PROTOTYPE_CACHE_DIR:-artifacts/cache/source_context_monthly_prototypes}"' in text
    assert "run/phase4_hyperda_staged.sh" in text
    assert "exec bash" in text
    assert "phase4_hyperda_staged_fast_trial" not in staged


def test_hyperda_legacy_capacity_wrapper_is_removed_from_active_run_entries():
    assert not Path("run/phase4_hyperda_legacy_capacity.sh").exists()

    readme = Path("run/README.md").read_text()
    assert "phase4_hyperda_legacy_capacity.sh" not in readme
    assert "legacy-capacity" not in readme


def test_prompt_conditioned_train_cli_accepts_explicit_split_protocol_overrides():
    text = Path("scripts/train/train_prompt_conditioned_shared.py").read_text()

    assert 'parser.add_argument("--splits_json"' in text
    assert 'parser.add_argument("--protocol_freeze_id"' in text
    assert 'parser.add_argument("--split_manifest_path"' in text
    assert 'parser.add_argument("--init_from_source_base_checkpoint"' in text
    assert 'parser.add_argument("--trainable_scope"' in text
    assert "robust_input_side_da_diagnostics_raw" in text
    assert '"splits_json": args.splits_json' in text
    assert '"split_manifest_path": args.split_manifest_path' in text
    assert '"protocol_freeze_id": args.protocol_freeze_id' in text
    assert '"init_from_source_base_checkpoint":' in text
    assert '"trainable_scope": args.trainable_scope' in text
    assert 'parser.add_argument("--source_episode_prompt_policy"' in text
    assert 'parser.add_argument("--source_anchor_blend_calibration"' in text
    assert 'parser.add_argument("--hyper_output_head_residual"' in text
    assert 'parser.add_argument("--hyper_coeff_generator"' in text
    assert 'parser.add_argument("--hyper_rank_gate_top_k"' in text
    assert 'parser.add_argument("--hyper_rank_gate_temperature_init"' in text
    assert 'parser.add_argument("--hyper_adapter_param_style"' in text
    assert 'parser.add_argument("--hyper_reliability_gate"' in text
    assert 'parser.add_argument("--hyper_reliability_init"' in text
    assert 'parser.add_argument("--hyper_source_saliency_prior_path"' in text
    assert 'parser.add_argument("--hyper_source_saliency_prior_beta"' in text
    assert 'parser.add_argument("--hyper_prompt_manifold_reliability"' in text
    assert 'parser.add_argument("--hyper_prompt_manifold_reliability_strength"' in text
    assert 'parser.add_argument("--hyper_enable_film"' in text
    assert 'parser.add_argument("--hyper_enable_adapters"' in text
    assert 'parser.add_argument("--amp_init_scale"' in text
    assert 'parser.add_argument("--amp_min_scale"' in text
    assert 'parser.add_argument("--amp_skip_abort_threshold"' in text
    assert 'parser.add_argument("--dataset_backend"' in text
    assert 'parser.add_argument("--tensor_cache_dir"' in text
    assert 'parser.add_argument("--max_year_cache_entries"' in text
    assert 'parser.add_argument("--tensor_cache_load_mode"' in text
    assert 'parser.add_argument("--train_batch_sampler"' in text
    assert 'parser.add_argument("--source_prototype_cache_dir"' in text
    assert 'parser.add_argument("--source_prototype_cache_mode"' in text
    assert '"source_episode_prompt_policy": args.source_episode_prompt_policy' in text
    assert '"source_anchor_blend_calibration": bool(args.source_anchor_blend_calibration)' in text
    assert '"hyper_output_head_residual": bool(args.hyper_output_head_residual)' in text
    assert '"hyper_coeff_generator": args.hyper_coeff_generator' in text
    assert '"hyper_rank_gate_top_k": args.hyper_rank_gate_top_k' in text
    assert '"hyper_rank_gate_temperature_init": args.hyper_rank_gate_temperature_init' in text
    assert '"hyper_adapter_param_style": args.hyper_adapter_param_style' in text
    assert '"hyper_reliability_gate": args.hyper_reliability_gate' in text
    assert '"hyper_reliability_init": args.hyper_reliability_init' in text
    assert '"hyper_source_saliency_prior_path": args.hyper_source_saliency_prior_path' in text
    assert '"hyper_source_saliency_prior_beta": args.hyper_source_saliency_prior_beta' in text
    assert '"hyper_prompt_manifold_reliability": bool(args.hyper_prompt_manifold_reliability)' in text
    assert '"hyper_prompt_manifold_reliability_strength": args.hyper_prompt_manifold_reliability_strength' in text
    assert '"prompt_diagnostic_input_domain": prompt_diagnostic_input_domain(args.context_encoder)' in text
    assert '"normalized_input_used_for_prompt_diagnostics": prompt_normalized_input_used(args.context_encoder)' in text
    assert '"hyper_enable_film": bool(args.hyper_enable_film)' in text
    assert '"hyper_enable_adapters": bool(args.hyper_enable_adapters)' in text
    assert '"amp_init_scale": args.amp_init_scale' in text
    assert '"amp_min_scale": args.amp_min_scale' in text
    assert '"amp_skip_abort_threshold": args.amp_skip_abort_threshold' in text
    assert '"dataset_backend": args.dataset_backend' in text
    assert '"tensor_cache_dir": args.tensor_cache_dir' in text
    assert '"max_year_cache_entries": args.max_year_cache_entries' in text
    assert '"tensor_cache_load_mode": args.tensor_cache_load_mode' in text
    assert '"train_batch_sampler": args.train_batch_sampler' in text
    assert '"source_prototype_cache_mode": args.source_prototype_cache_mode' in text
    assert '"source_prototype_cache_dir": args.source_prototype_cache_dir' in text
    assert "splits_json=args.splits_json" in text
    assert "protocol_freeze_id=args.protocol_freeze_id" in text
    assert "split_manifest_path=args.split_manifest_path" in text


def test_staged_hyperda_ablation_wrapper_maps_source_stage_presets_without_m2_4():
    text = Path("run/phase4_hyperda_staged_ablation.sh").read_text()
    staged = Path("run/phase4_hyperda_staged.sh").read_text()

    assert "US_loro_zero_few_shot_splits.json" in text
    assert "--init_from_source_base_checkpoint" in text
    assert "--trainable_scope source_base_frozen_adapter_film" in text
    assert "--selection_metric source_val_transfer_safe_score" in text
    assert 'ABLATION_ID="${ABLATION_ID:-M0_current}"' in text
    assert "M0_current" in text
    assert "M1_shared_coeff" in text
    assert "M2_shared_coeff_gate" in text
    assert "M2_rank_gated_dora" in text
    assert "M2_1_rank_gated_dora_stable" in text
    assert "M2_2_source_saliency_prior" in text
    assert "M2_3_source_safe_residual_hyperda" in text
    assert "M2_5a_da_aware_prompt_only" in text
    assert "M2_5b_da_aware_conservative_router" in text
    assert "M2_4_target_context_conservative_hyperda" not in text
    assert "M3_film_only" in text
    assert "M4_adapter_only" in text
    assert "HYPER_COEFF_GENERATOR=per_adapter" in text
    assert "HYPER_COEFF_GENERATOR=shared_layer_aware" in text
    assert "HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated" in text
    assert "HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated_stable" in text
    assert "HYPER_ADAPTER_PARAM_STYLE=dora_like_gain" in text
    assert "HYPER_ADAPTER_PARAM_STYLE=dora_like_gain_bounded" in text
    assert "HYPER_RELIABILITY_GATE=none" in text
    assert "HYPER_RELIABILITY_GATE=prompt_scalar" in text
    assert "HYPER_ENABLE_FILM=0" in text
    assert "HYPER_ENABLE_FILM=1" in text
    assert "HYPER_ENABLE_ADAPTERS=0" in text
    assert "HYPER_ENABLE_ADAPTERS=1" in text
    assert "--hyper_coeff_generator" in text
    assert "--hyper_rank_gate_top_k" in text
    assert "--hyper_rank_gate_temperature_init" in text
    assert "--hyper_adapter_param_style" in text
    assert "--hyper_reliability_gate" in text
    assert "--hyper_reliability_init" in text
    assert "--hyper_source_saliency_prior_path" in text
    assert "--hyper_source_saliency_prior_beta" in text
    assert "--hyper_source_saliency_prior_application" in text
    assert "--hyper_prompt_manifold_reliability" in text
    assert "--hyper_prompt_manifold_reliability_strength" in text
    assert "--hyper_enable_film" in text
    assert "--hyper_enable_adapters" in text
    assert "--hyper_residual_magnitude_penalty" in text
    assert "--hyper_coeff_entropy_floor" in text
    assert "--hyper_coeff_entropy_penalty" in text
    assert "target_val=unused_in_main_protocol" in text
    assert "target_labels=none" in text
    assert 'BUILD_ABLATION_TABLE="${BUILD_ABLATION_TABLE:-1}"' in text
    assert 'ABLATION_TABLE_OUTPUT_DIR="${ABLATION_TABLE_OUTPUT_DIR:-reports/ablations/hyperda_staged_v1}"' in text
    assert 'DATASET_BACKEND="${DATASET_BACKEND:-auto}"' in text
    assert 'TENSOR_CACHE_LOAD_MODE="${TENSOR_CACHE_LOAD_MODE:-mmap}"' in text
    assert 'TRAIN_BATCH_SAMPLER="${TRAIN_BATCH_SAMPLER:-source_region_year_grouped}"' in text
    assert 'BATCH_SIZE="${BATCH_SIZE:-64}"' in text
    assert 'USE_AMP="${USE_AMP:-1}"' in text
    assert "USE_AMP=0" in text
    assert "LR=2e-4" in text
    assert "HYPER_RANK_GATE_TEMPERATURE_INIT=2.0" in text
    assert "HYPER_SOURCE_SALIENCY_PRIOR_PATH" in text
    assert 'HYPER_SOURCE_SALIENCY_PRIOR_DIR="${HYPER_SOURCE_SALIENCY_PRIOR_DIR:-artifacts/priors/source_basis_saliency}"' in text
    assert 'HYPER_SOURCE_SALIENCY_AUTO_BUILD="${HYPER_SOURCE_SALIENCY_AUTO_BUILD:-1}"' in text
    assert 'HYPER_SOURCE_SALIENCY_MAX_BATCHES="${HYPER_SOURCE_SALIENCY_MAX_BATCHES:-16}"' in text
    assert 'HYPER_SOURCE_SALIENCY_BATCH_SIZE="${HYPER_SOURCE_SALIENCY_BATCH_SIZE:-4}"' in text
    assert 'HYPER_SOURCE_SALIENCY_PRIOR_PATH="${HYPER_SOURCE_SALIENCY_PRIOR_DIR}/${TARGET_REGION}_s${SEED}.pt"' in text
    assert "resolve_auto_m2_1_saliency_source_checkpoint" in text
    assert "M2_1_rank_gated_dora_stable" in text
    assert "M2.2 source saliency prior not found; building it from M2.1 stable checkpoint" in text
    assert "scripts/train/build_source_basis_saliency_prior.py" in text
    assert '--dataset_backend "${RESOLVED_DATASET_BACKEND}"' in text
    assert '--max_batches "${HYPER_SOURCE_SALIENCY_MAX_BATCHES}"' in text
    assert '--batch_size "${HYPER_SOURCE_SALIENCY_BATCH_SIZE}"' in text
    assert "Auto-build is disabled by HYPER_SOURCE_SALIENCY_AUTO_BUILD" in text
    assert "HYPER_SOURCE_SALIENCY_PRIOR_BETA=0.5" in text
    assert "HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION=legacy_gate_logit_bias_before_topk" in text
    assert "HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION=soft_regularization_metadata" in text
    assert "HYPER_RESIDUAL_MAGNITUDE_PENALTY=0.001" in text
    assert "HYPER_COEFF_ENTROPY_FLOOR=0.5" in text
    assert "HYPER_COEFF_ENTROPY_PENALTY=0.0001" in text
    assert 'M2_3_INIT_FROM_M2_1_CHECKPOINT="${M2_3_INIT_FROM_M2_1_CHECKPOINT-auto}"' in text
    assert "m2_3_init_from_m2_1_checkpoint" in text
    assert '--init_from_prompt_checkpoint "${M2_3_INIT_FROM_M2_1_CHECKPOINT}"' in text
    assert "target_eval_input_stats_used_for_update=false" in text
    assert 'CONTEXT_ENCODER="${CONTEXT_ENCODER:-current_mean_std}"' in text
    assert "CONTEXT_ENCODER=robust_input_side_da_diagnostics" in text
    assert "CONTEXT_ENCODER=robust_input_side_da_diagnostics_raw" in text
    assert "DA-aware prompt/router diagnostics" in text
    assert "prompt_diagnostic_input_domain" in text
    assert "--context_encoder" in text
    assert 'SOURCE_PROTOTYPE_CACHE_MODE="${SOURCE_PROTOTYPE_CACHE_MODE:-read_write}"' in text
    assert '--dataset_backend "${RESOLVED_DATASET_BACKEND}"' in text
    assert '--tensor_cache_load_mode "${TENSOR_CACHE_LOAD_MODE}"' in text
    assert '--train_batch_sampler "${TRAIN_BATCH_SAMPLER}"' in text
    assert '--source_prototype_cache_mode "${SOURCE_PROTOTYPE_CACHE_MODE}"' in text
    assert 'cmd+=(--amp)' in text
    assert "scripts/analysis/build_hyperda_staged_ablation_table.py" in text
    assert "--runs_root" in text
    assert "--output_dir" in text
    assert "--target_region" in text
    assert "--seed" in text

    assert "ABLATION_ID" not in staged
    assert "--hyper_coeff_generator" not in staged
    assert "--hyper_reliability_gate" not in staged


def test_staged_ablation_m2_1_does_not_inherit_residual_or_context_shrinkage_knobs(tmp_path):
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source checkpoint")

    result = subprocess.run(
        [
            "bash",
            "run/phase4_hyperda_staged_ablation.sh",
            str(source_checkpoint),
            "US-R1",
            "0",
            "0",
            "--dry-run",
        ],
        env={**os.environ, "ABLATION_ID": "M2_1_rank_gated_dora_stable"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M2_1_rank_gated_dora_stable" in result.stdout
    assert "zero_shot_prior_form=direct_hyper" in result.stdout
    assert "source_residual_rho=0.0" in result.stdout
    assert "hyper_residual_magnitude_penalty=0.0" in result.stdout
    assert "hyper_coeff_entropy_floor=0.0" in result.stdout
    assert "hyper_coeff_entropy_penalty=0.0" in result.stdout
    assert "target_context_conservative_shrinkage" not in result.stdout
    assert "context_shrinkage_policy" not in result.stdout
    assert "target_eval_input_stats_used_for_update=false" in result.stdout


def test_staged_ablation_m2_5a_prompt_only_inherits_m2_1_and_uses_da_aware_context(tmp_path):
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source checkpoint")

    result = subprocess.run(
        [
            "bash",
            "run/phase4_hyperda_staged_ablation.sh",
            str(source_checkpoint),
            "US-R1",
            "0",
            "0",
            "--dry-run",
        ],
        env={**os.environ, "ABLATION_ID": "M2_5a_da_aware_prompt_only"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M2_5a_da_aware_prompt_only" in result.stdout
    assert "context_encoder=robust_input_side_da_diagnostics" in result.stdout
    assert "hyper_coeff_generator=shared_layer_aware_rank_gated_stable" in result.stdout
    assert "hyper_adapter_param_style=dora_like_gain_bounded" in result.stdout
    assert "hyper_rank_gate_top_k=4" in result.stdout
    assert "hyper_rank_gate_temperature_init=2.0" in result.stdout
    assert "use_amp=0" in result.stdout
    assert "batch_size=64 accum_steps=2 lr=2e-4" in result.stdout
    assert "zero_shot_prior_form=direct_hyper" in result.stdout
    assert "source_residual_rho=0.0" in result.stdout
    assert "--context_encoder robust_input_side_da_diagnostics" in result.stdout
    assert "hyper_residual_magnitude_penalty=0.0" in result.stdout
    assert "target_eval_input_stats_used_for_update=false" in result.stdout


def test_staged_ablation_m2_5b_uses_raw_da_conservative_router(tmp_path):
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source checkpoint")

    result = subprocess.run(
        [
            "bash",
            "run/phase4_hyperda_staged_ablation.sh",
            str(source_checkpoint),
            "US-R1",
            "0",
            "0",
            "--dry-run",
        ],
        env={**os.environ, "ABLATION_ID": "M2_5b_da_aware_conservative_router"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M2_5b_da_aware_conservative_router" in result.stdout
    assert "context_encoder=robust_input_side_da_diagnostics_raw" in result.stdout
    assert "prompt_diagnostic_input_domain=raw_input_side" in result.stdout
    assert "zero_shot_prior_form=source_base_residual_reliability_gated" in result.stdout
    assert "source_residual_rho=1.0" in result.stdout
    assert "source_residual_gate_init=0.90" in result.stdout
    assert "hyper_prompt_manifold_reliability=1" in result.stdout
    assert "hyper_prompt_manifold_reliability_strength=0.25" in result.stdout
    assert "hyper_coeff_generator=shared_layer_aware_rank_gated_stable" in result.stdout
    assert "hyper_adapter_param_style=dora_like_gain_bounded" in result.stdout
    assert "hyper_rank_gate_top_k=4" in result.stdout
    assert "hyper_rank_gate_temperature_init=2.0" in result.stdout
    assert "use_amp=0" in result.stdout
    assert "batch_size=64 accum_steps=2 lr=2e-4" in result.stdout
    assert "--context_encoder robust_input_side_da_diagnostics_raw" in result.stdout
    assert "--zero_shot_prior_form source_base_residual_reliability_gated" in result.stdout
    assert "--source_residual_rho 1.0" in result.stdout


def test_staged_ablation_rejects_m2_4_because_it_is_stage3_k0_diagnostic(tmp_path):
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source checkpoint")

    result = subprocess.run(
        [
            "bash",
            "run/phase4_hyperda_staged_ablation.sh",
            str(source_checkpoint),
            "US-R1",
            "0",
            "0",
            "--dry-run",
        ],
        env={
            **os.environ,
            "ABLATION_ID": "M2_4_target_context_conservative_hyperda",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "Unsupported ABLATION_ID=M2_4_target_context_conservative_hyperda" in result.stderr
    assert "M2.4 is a Stage 3 K=0 target-context diagnostic" in result.stderr


def test_hyperda_plus_source_prior_matrix_design_is_removed_from_active_tree():
    retired_paths = [
        "configs/experiments/hyperda_plus_source_prior_matrix.yaml",
        "scripts/report/hyperda_plus_source_prior_matrix.py",
        "tests/test_hyperda_plus_source_prior_matrix_report.py",
    ]

    assert not [path for path in retired_paths if Path(path).exists()]


def test_hyperda_plus_matrix_wrapper_is_retired_from_active_run_entries():
    assert not Path("run/phase4_hyperda_plus_matrix.sh").exists()

    readme = Path("run/README.md").read_text()
    assert "hyperda_safe_us_r1_seed0.sh" in readme
    assert "HyperDA-SAFE" in readme
    assert "phase4_hyperda_plus_matrix.sh" not in readme
    assert "TODO_H4_episode_prior.md" not in readme


def test_hyperda_inference_wrapper_uses_target_eval_protocol():
    text = Path("run/phase4_hyperda_inference.sh").read_text()

    assert "context2022_query2023_2025_k0_4_12" not in text
    assert "--K 0" in text
    assert "--split_type target_eval" in text
    assert "target_query" not in text
    assert "checkpoint_best_source_val_transfer_safe_score.pt" in text
    assert "artifacts/runs/phase4_hyperda_staged" in text
    assert "phase4_hyperda_staged" in text
    assert 'TARGET_CONTEXT_PROMPT="${TARGET_CONTEXT_PROMPT:-1}"' in text
    assert 'TARGET_TRAIN_RESIDUAL_GAIN_CALIBRATION="${TARGET_TRAIN_RESIDUAL_GAIN_CALIBRATION:-0}"' in text


def test_hyperda_target_adaptation_wrapper_declares_legacy_status():
    text = Path("run/phase5_hyperda_target_adapt.sh").read_text()

    assert "legacy_internal_not_paper_main" in text
    assert "target_train=2015-2021" in text
    assert "target_val=2022" in text
    assert "target_eval=2023-2025" in text
    assert "freeze_hypernetwork=true" in text
    assert "trainable=target_latent,adapter_coefficient_residuals,residual_gain" in text
    assert "target_eval labels are never used for adaptation" in text
    assert "scripts/train/train_hyperda_target_adapt.py" in text
    assert "HyperDA target adaptation modules are implemented, but the full dataset" not in text


def test_episode_prior_run_wrapper_is_retired_from_paper_path():
    bank = Path("scripts/train/build_hyperda_source_episode_bank.py").read_text()
    prior = Path("scripts/train/train_hyperda_episode_prior.py").read_text()

    assert "source operator episode bank" in bank
    assert "must not read target_val or target_eval" in bank
    assert "source_fit" in bank
    assert "target modules only" in bank
    assert "build_source_episode_adapter_bank.py" in bank
    assert "lightweight adapter coefficients" in bank
    assert "prompt_to_zeta_prior" in prior
    assert "source_side_episodic_validation" in prior
    assert "does not use target_val" in prior
    assert not Path("run/phase5_hyperda_episode_prior_eval.sh").exists()
