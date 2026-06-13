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
        "hyperda_few_shot_k4",
        "hyperda_few_shot_k12",
    ]:
        assert method in combined
    assert "target_context_prompt_state" in combined
    assert "target_context_dates_hash" in combined
    assert "target_support_dates_hash" in combined
    assert "model_selection_source" in combined
    assert "source_val_preregistered" in combined


def test_zero_few_shot_wrapper_declares_main_protocol():
    text = Path("run/phase5_hyperda_zero_few_shot.sh").read_text()

    assert "US_loro_zero_few_shot_splits.json" in text
    assert "--K" in text
    assert "--adaptation_setting" in text
    assert "target_val=unused_in_main_protocol" in text
    assert "model_selection_source=source_val_preregistered" in text
    assert "scripts/train/train_hyperda_few_shot_adapt.py" in text
    assert 'ADAPT_SCOPE="${ADAPT_SCOPE:-all}"' in text
    assert 'AUDIT_IDENTITY="${AUDIT_IDENTITY:-0}"' in text
    assert 'ADAPT_SOLVER="${ADAPT_SOLVER:-adamw}"' in text
    assert 'FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN:-0}"' in text
    assert 'RIDGE_LAMBDA="${RIDGE_LAMBDA:-1.0}"' in text
    assert 'RIDGE_MAX_FEATURE_PIXELS="${RIDGE_MAX_FEATURE_PIXELS:-20000}"' in text
    assert 'RIDGE_STANDARDIZE_FEATURES="${RIDGE_STANDARDIZE_FEATURES:-0}"' in text
    assert 'ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-${BATCH_SIZE:-8}}"' in text
    assert "--adapt_scope" in text
    assert "--adapt_solver" in text
    assert "--freeze_monthly_gain" in text
    assert "--ridge_lambda" in text
    assert "--ridge_max_feature_pixels" in text
    assert "--ridge_standardize_features" in text
    assert "--audit_identity" in text


def test_zero_few_shot_eval_wrapper_runs_all_main_k_settings_on_target_eval():
    text = Path("run/phase5_hyperda_zero_few_shot_eval.sh").read_text()

    assert 'K_LIST="${K_LIST:-0 4 12}"' in text
    assert 'OUTPUT_BASE="${5:-' in text
    assert "run/phase5_hyperda_zero_few_shot.sh" in text
    assert "checkpoint_final_preregistered.pt" in text
    assert "scripts/eval/evaluate_checkpoint.py" in text
    assert "--predictor_type hyperda_target_adapt" in text
    assert "--split_type target_eval" in text
    assert 'EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"' in text
    assert 'ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-${BATCH_SIZE:-8}}"' in text
    assert 'EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE:-8}}"' in text
    assert 'ADAPT_RECIPE="${ADAPT_RECIPE:-source_anchor}"' in text
    assert 'ANCHOR_ALPHA_K4="${ANCHOR_ALPHA_K4:-0.75}"' in text
    assert 'ANCHOR_ALPHA_K12="${ANCHOR_ALPHA_K12:-0.25}"' in text
    assert 'LR_K12="${LR_K12:-3e-4}"' in text
    assert 'MAX_STEPS_K12="${MAX_STEPS_K12:-80}"' in text
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
    assert 'ADAPT_SCOPE="${ADAPT_SCOPE:-all}"' in text
    assert 'ADAPT_SOLVER="${ADAPT_SOLVER:-adamw}"' in text
    assert 'FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN:-0}"' in text
    assert 'RIDGE_LAMBDA="${RIDGE_LAMBDA:-1.0}"' in text
    assert 'RIDGE_MAX_FEATURE_PIXELS="${RIDGE_MAX_FEATURE_PIXELS:-20000}"' in text
    assert 'RIDGE_STANDARDIZE_FEATURES="${RIDGE_STANDARDIZE_FEATURES:-0}"' in text
    assert 'AUDIT_IDENTITY="${AUDIT_IDENTITY:-0}"' in text
    assert 'AUDIT_IDENTITY_TOLERANCE="${AUDIT_IDENTITY_TOLERANCE:-1e-8}"' in text
    assert "identity_audit.json" in text
    assert "compare_identity_audit" in text
    assert 'ADAPT_SCOPE_FOR_K="none"' in text
    assert 'ADAPT_SOLVER_FOR_K="adamw"' in text
    assert 'ADAPTATION_MAX_STEPS="0"' in text
    assert 'ANCHOR_ALPHA="0.0"' in text
    assert 'AUDIT_IDENTITY_FOR_K="0"' in text
    assert 'AUDIT_IDENTITY_FOR_K="1"' in text
    assert 'AUDIT_IDENTITY="${AUDIT_IDENTITY_FOR_K}"' in text
    assert 'ADAPT_SCOPE="${ADAPT_SCOPE_FOR_K}"' in text
    assert 'ADAPT_SOLVER="${ADAPT_SOLVER_FOR_K}"' in text
    assert 'FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN}"' in text
    assert 'RIDGE_LAMBDA="${RIDGE_LAMBDA}"' in text
    assert 'RIDGE_MAX_FEATURE_PIXELS="${RIDGE_MAX_FEATURE_PIXELS}"' in text
    assert 'RIDGE_STANDARDIZE_FEATURES="${RIDGE_STANDARDIZE_FEATURES}"' in text
    assert "freeze_monthly_gain" in text
    assert "ridge_coefficient_norm" in text
    assert "ridge_feature_observation_count" in text
    assert "ridge_masked_observation_count" in text
    assert "source_checkpoint_sha256" in text
    assert "target_labels_loaded_for_adaptation" in text
    assert "target_labels_used_for_adaptation" in text
    assert "target_support_count" in text
    assert "results/checkpoint_*/target_eval/{target_region}/summary.json" in text
    assert "return json.load(f), str(path)" in text
    assert "return load_json(path), str(path)" not in text
    terminal_block = text.split("terminal_columns = [", 1)[1].split("]", 1)[0]
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
    ]
    for token in forbidden:
        assert token not in text


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

    assert "US_loro_kdate_splits.json" not in text
    assert "--model_type hyperda_basis_adapter" in text
    assert "--selection_metric source_val_transfer_safe_score" in text


def test_hyperda_inference_wrapper_uses_target_eval_protocol():
    text = Path("run/phase4_hyperda_inference.sh").read_text()

    assert "context2022_query2023_2025_k0_4_12" not in text
    assert "--K 0" in text
    assert "--split_type target_eval" in text
    assert "target_query" not in text
    assert "checkpoint_best_source_val_transfer_safe_score.pt" in text
    assert "phase4_prompt_conditioned_hyperda_basis_adapter_*" in text
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


def test_episode_prior_entrypoints_exist_as_protocol_safe_scaffolds():
    bank = Path("scripts/train/build_hyperda_source_episode_bank.py").read_text()
    prior = Path("scripts/train/train_hyperda_episode_prior.py").read_text()
    wrapper = Path("run/phase5_hyperda_episode_prior_eval.sh").read_text()

    assert "source operator episode bank" in bank
    assert "must not read target_val or target_eval" in bank
    assert "source_fit" in bank
    assert "target modules only" in bank
    assert "build_source_episode_adapter_bank.py" in bank
    assert "lightweight adapter coefficients" in bank
    assert "prompt_to_zeta_prior" in prior
    assert "source_side_episodic_validation" in prior
    assert "does not use target_val" in prior
    assert "ADAPT_RECIPE=episode_prior" in wrapper
    assert "phase5_hyperda_zero_few_shot_eval.sh" in wrapper
