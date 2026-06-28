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
        "hyperda_trust_zero_shot_context",
    ]:
        assert method in combined
    baselines = Path("specs/baselines.yaml").read_text()
    paper_main_block = baselines.split("baseline_definitions:", 1)[0]
    assert "hyperda_safe_few_shot_k4" not in paper_main_block
    assert "hyperda_safe_few_shot_k12" not in paper_main_block
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
    assert '--adapt_solver "${ADAPT_SOLVER}"' in text
    assert "--audit_identity" in text
    assert 'FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN:-0}"' in text
    assert '$(if [[ "${FREEZE_MONTHLY_GAIN}" == "1"' in text
    assert 'STAGE3_CONTEXT_TTA="${STAGE3_CONTEXT_TTA:-none}"' in text
    assert '--context_tta_mode "${STAGE3_CONTEXT_TTA}"' in text
    active_log = text.split('echo "============================================"', 1)[1]
    for legacy_token in [
        "--enable_target_spatial_refine",
    ]:
        assert legacy_token not in active_log
    assert "ridge_weighting=" in active_log


def test_zero_few_shot_eval_wrapper_runs_all_main_k_settings_on_target_eval():
    text = Path("run/phase5_hyperda_zero_few_shot_eval.sh").read_text()

    assert 'K_LIST="${K_LIST:-0 4 12}"' in text
    assert 'STAGE3_KSHOT_MODE="${STAGE3_KSHOT_MODE:-paper_safe}"' in text
    assert 'EVAL_OUTPUT_LEVEL="${EVAL_OUTPUT_LEVEL:-compact}"' in text
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
    assert 'DIAGNOSTIC_KSHOT_STRENGTH="${DIAGNOSTIC_KSHOT_STRENGTH:-strong}"' in text
    assert 'SAFE_POLICY_JSON="${SAFE_POLICY_JSON:-}"' in text
    assert 'SAFE_POLICY_JSON="${SAFE_POLICY_JSON}"' in text
    assert 'SAFE_POLICY_CACHE_ROOT="${SAFE_POLICY_CACHE_ROOT:-artifacts/runs/stage3_source_safe_policy_cache}"' in text
    assert 'AUTO_GENERATE_SAFE_POLICY="${AUTO_GENERATE_SAFE_POLICY:-0}"' in text
    assert "resolve_safe_policy_cache_path" in text
    assert "SAFE_POLICY_CACHE_KEY" in text
    assert "SAFE_POLICY_CACHE_STATUS" in text
    assert "cached safe_policy.json" in text
    assert "AUTO_GENERATE_SAFE_POLICY=1" in text
    assert "scripts/eval/run_stage3_source_safe_policy_calibration.py" in text
    assert "scripts/eval/calibrate_source_safe_guard.py" in text
    assert "safe_policy_cache_manifest.json" in text
    assert 'REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-1}"' in text
    assert 'REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_FOR_K}"' in text
    assert 'ADAPT_MIX_RHO_WAS_SET="${ADAPT_MIX_RHO+x}"' in text
    assert 'FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN:-0}"' in text
    assert 'TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES:-0}"' in text
    assert 'TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES}"' in text
    assert 'STAGE3_CONTEXT_TTA="${STAGE3_CONTEXT_TTA:-none}"' in text
    assert 'STAGE3_CONTEXT_TTA="${STAGE3_CONTEXT_TTA}"' in text
    assert 'EVAL_RAW_ADAPTED_BEFORE_MIX="${EVAL_RAW_ADAPTED_BEFORE_MIX:-1}"' in text
    assert 'RAW_ADAPTED_EVAL_ARGS+=(--eval_raw_adapted_before_mix)' in text
    assert '"${RAW_ADAPTED_EVAL_ARGS[@]}"' in text
    assert 'STAGE3_KSHOT_MODE="${STAGE3_KSHOT_MODE}"' in text
    assert '--stage3_kshot_mode "${STAGE3_KSHOT_MODE}"' in Path("run/phase5_hyperda_zero_few_shot.sh").read_text()
    assert "diagnostic_direct_kshot" in text
    assert "diagnostic_direct_kshot_v2" in text
    assert '--output_level "${EVAL_OUTPUT_LEVEL}"' in text
    assert "STAGE3_K0_CONTEXT_SHRINKAGE" not in text
    assert "--stage3_k0_context_shrinkage" not in text
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
    assert 'ADAPTATION_MAX_STEPS="200"' in text
    assert 'ADAPTATION_MAX_STEPS="100"' in text
    assert 'ADAPT_MAX_STEPS="${ADAPTATION_MAX_STEPS}"' in text
    assert 'BATCH_SIZE="${ADAPT_BATCH_SIZE}" SPLITS_JSON="${SPLITS_JSON}" OUTPUT_DIR="${ADAPT_DIR}"' in text
    assert 'ANCHOR_ALPHA="${ANCHOR_ALPHA}"' in text
    assert '--batch_size "${EVAL_BATCH_SIZE}"' in text
    assert "print_markdown_table" in text
    assert "Quick target_eval WRMSE table" in text
    assert "overview.csv" in text
    assert "overview.md" in text
    assert "overview.json" in text
    assert 'for child in output_base.glob("K*")' in text
    assert "stage3_kshot_mode" in text
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
    assert '("raw_delta", "raw_to_k0_mean_abs_delta")' in terminal_block
    assert '("final_delta", "final_mix_to_k0_mean_abs_delta")' in terminal_block
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


def test_zero_few_shot_eval_wrapper_support_gain_v2_is_checkpoint_side_not_eval_legacy():
    text = Path("run/phase5_hyperda_zero_few_shot_eval.sh").read_text()

    assert "diagnostic_support_gain_v2" in text
    assert "diagnostic_support_gain_v3_stable" in text
    assert "diagnostic_support_gain_v4_nested_stable" in text
    assert 'STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v1"' in text
    assert 'STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v2"' not in (
        text.split("SUPPORT_GAIN_EVAL_ARGS=()", 1)[1].split('echo ""', 1)[0]
    )
    assert 'STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v3_stable"' not in (
        text.split("SUPPORT_GAIN_EVAL_ARGS=()", 1)[1].split('echo ""', 1)[0]
    )
    assert 'STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v4_nested_stable"' not in (
        text.split("SUPPORT_GAIN_EVAL_ARGS=()", 1)[1].split('echo ""', 1)[0]
    )


def test_zero_few_shot_eval_wrapper_v4_nested_k12_uses_run_local_manifest():
    text = Path("run/phase5_hyperda_zero_few_shot_eval.sh").read_text()

    assert "prepare_nested_k12_support_manifest" in text
    assert "run_local_k12_nested_k4_plus_8_original_k12_nonduplicate" in text
    assert 'mapfile -t NESTED_K12_INFO < <(prepare_nested_k12_support_manifest' in text
    assert 'NESTED_K12_SPLITS_JSON="${NESTED_K12_INFO[0]}"' in text
    assert 'if [[ "${K}" == "12" && -n "${NESTED_K12_SPLITS_JSON}" ]]' in text
    assert 'SPLITS_JSON_FOR_K="${NESTED_K12_SPLITS_JSON}"' in text
    assert 'SPLITS_JSON="${SPLITS_JSON_FOR_K}" OUTPUT_DIR="${ADAPT_DIR}"' in text
    assert '--splits_json "${SPLITS_JSON_FOR_K}"' in text


def test_zero_few_shot_wrappers_expose_v7_balanced_ridge_mode():
    eval_text = Path("run/phase5_hyperda_zero_few_shot_eval.sh").read_text()
    adapt_text = Path("run/phase5_hyperda_zero_few_shot.sh").read_text()

    assert "diagnostic_linearized_coeff_ridge_v7_balanced_nested" in eval_text
    assert "diagnostic_linearized_coeff_ridge_v7_balanced_nested" in adapt_text
    assert 'RIDGE_WEIGHTING="${RIDGE_WEIGHTING:-global_pixel_l2}"' in eval_text
    assert 'RIDGE_WEIGHTING="${RIDGE_WEIGHTING:-global_pixel_l2}"' in adapt_text
    assert "cycle_variable_balanced_huber" in eval_text
    assert "cycle_variable_balanced_huber" in adapt_text
    assert '--ridge_weighting "${RIDGE_WEIGHTING}"' in adapt_text


def test_hyperda_safe_paper_entrypoint_requires_source_calibrated_policy_for_kshot():
    text = Path("run/hyperda_safe_us_r1_seed0.sh").read_text()

    assert "SAFE diagnostic US-R1 seed0 entrypoint" in text
    assert 'SAFE_POLICY_JSON="${SAFE_POLICY_JSON:-}"' in text
    assert 'SAFE_POLICY_CACHE_ROOT="${SAFE_POLICY_CACHE_ROOT:-artifacts/runs/stage3_source_safe_policy_cache}"' in text
    assert 'AUTO_GENERATE_SAFE_POLICY="${AUTO_GENERATE_SAFE_POLICY:-0}"' in text
    assert 'REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-1}"' in text
    assert 'export SAFE_POLICY_JSON="${SAFE_POLICY_JSON}"' in text
    assert 'export SAFE_POLICY_CACHE_ROOT="${SAFE_POLICY_CACHE_ROOT}"' in text
    assert 'export AUTO_GENERATE_SAFE_POLICY="${AUTO_GENERATE_SAFE_POLICY}"' in text
    assert 'export REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT}"' in text
    assert "source_side_episode_calibration" in text
    assert "safe_policy.json" in text
    assert "auto-cache" in text
    assert "rejected_to_k0_anchor" in text
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
    assert 'CANDIDATE_SET="${CANDIDATE_SET:-stage3_conservative_v1}"' in text
    assert '--candidate_set "${CANDIDATE_SET}"' in text
    assert "--allow_in_checkpoint_source_episodes" in text
    assert "EVIDENCE_LEVEL=strict" in text
    assert "SOURCE_HELDOUT_CHECKPOINT_MAP" in text
    assert "STAGE3_K0_CONTEXT_SHRINKAGE" not in text
    assert "stage3_k0_m2_4a_variable_v1" not in text
    assert "bash run/hyperda_safe_us_r1_seed0.sh" in text


def test_stage3_source_safe_row_builder_has_source_only_contract():
    text = Path("scripts/eval/run_stage3_source_safe_policy_calibration.py").read_text()

    assert "--active_region_override" in text
    assert "--split_type source_val" in text
    assert "--prediction_record_path" in text
    assert "--output_level" in text
    assert '"full"' in text
    assert "--target_context_max_samples" in text
    assert "source_safe_candidate_rows.csv" in text
    assert "calibration_rows.csv" in text
    assert "source_val_pseudo_query" in text
    assert "target_eval_usage" in text
    assert "unused_in_main_protocol" in text
    assert "allow_in_checkpoint_source_episodes" in text
    assert "strict" in text


def test_zero_few_shot_eval_wrapper_reports_cache_miss_before_paper_safe_kshot(tmp_path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"placeholder")
    splits = tmp_path / "splits.json"
    splits.write_text('{"splits": []}\n', encoding="utf-8")
    cache_root = tmp_path / "policy_cache"

    result = subprocess.run(
        [
            "bash",
            "run/phase5_hyperda_zero_few_shot_eval.sh",
            str(checkpoint),
            "US-R1",
            "0",
            "0",
            str(tmp_path / "out"),
        ],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "PYTHONPATH": ".",
            "K_LIST": "4",
            "SPLITS_JSON": str(splits),
            "SAFE_POLICY_CACHE_ROOT": str(cache_root),
            "AUTO_GENERATE_SAFE_POLICY": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "No cached safe_policy.json found" in result.stderr
    assert "AUTO_GENERATE_SAFE_POLICY=1" in result.stderr
    assert "STAGE3_KSHOT_MODE=diagnostic_direct_kshot" in result.stderr


def test_zero_few_shot_eval_wrapper_reuses_cached_safe_policy_before_adaptation(tmp_path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"placeholder")
    splits = tmp_path / "splits.json"
    splits.write_text('{"splits": []}\n', encoding="utf-8")
    cache_root = tmp_path / "policy_cache"

    miss = subprocess.run(
        [
            "bash",
            "run/phase5_hyperda_zero_few_shot_eval.sh",
            str(checkpoint),
            "US-R1",
            "0",
            "0",
            str(tmp_path / "out_miss"),
        ],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "PYTHONPATH": ".",
            "K_LIST": "4",
            "SPLITS_JSON": str(splits),
            "SAFE_POLICY_CACHE_ROOT": str(cache_root),
            "AUTO_GENERATE_SAFE_POLICY": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    cache_line = next(line for line in miss.stderr.splitlines() if "No cached safe_policy.json found at:" in line)
    cached_policy = Path(cache_line.split(" at:", 1)[1].strip())
    cached_policy.parent.mkdir(parents=True, exist_ok=True)
    cached_policy.write_text(
        '{"policy_source":"source_side_episode_calibration","target_val_usage":"unused_in_main_protocol",'
        '"target_eval_usage":"final_eval_only_no_selection","policies":{}}\n',
        encoding="utf-8",
    )

    reused = subprocess.run(
        [
            "bash",
            "run/phase5_hyperda_zero_few_shot_eval.sh",
            str(checkpoint),
            "US-R1",
            "0",
            "0",
            str(tmp_path / "out_reused"),
        ],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "PYTHONPATH": ".",
            "K_LIST": "4",
            "SPLITS_JSON": str(splits),
            "SAFE_POLICY_CACHE_ROOT": str(cache_root),
            "AUTO_GENERATE_SAFE_POLICY": "0",
            "EVAL_MAX_SAMPLES": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    combined = reused.stdout + reused.stderr
    assert "SAFE_POLICY_CACHE_STATUS=reused cached safe_policy.json" in combined
    assert f"SAFE_POLICY_JSON={cached_policy}" in combined
    assert (cached_policy.parent / "safe_policy_cache_manifest.json").exists()


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
    assert "STAGE3_K0_CONTEXT_SHRINKAGE" not in text
    assert "run/phase5_hyperda_zero_few_shot_eval.sh" in text
    assert "target_context=2015-2021 input-side only" in text
    assert "target_val=unused_in_main_protocol" in text
    assert "target_eval=2023-2025 final offline evaluation" in text


def test_k0_target_context_residual_shrinkage_is_removed_from_active_protocol_surface():
    combined = "\n".join(
        [
            Path("run/README.md").read_text(),
            Path("docs/COAUTHOR_CONTEXT.md").read_text(),
            Path("context/01_RESEARCH_CONTRACT.md").read_text(),
            Path("specs/hyperda_v4.yaml").read_text(),
            Path("tasks/phase5_hyperda_safe_zero_few_shot.md").read_text(),
        ]
    )

    assert "M2.4a" not in combined
    assert "M2_4_target_context_conservative_hyperda" not in combined
    assert "STAGE3_K0_CONTEXT_SHRINKAGE" not in combined
    assert "source_episode_calibrated_v1" not in combined


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
    assert "STAGE3_K0_CONTEXT_SHRINKAGE" not in text
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
    assert 'parser.add_argument("--hyper_source_manifold_guard"' in text
    assert 'parser.add_argument("--hyper_source_manifold_guard_strength"' in text
    assert 'parser.add_argument("--hyper_source_manifold_guard_distance_key"' in text
    assert 'parser.add_argument("--source_manifold_guard_calibration"' in text
    assert 'parser.add_argument("--hyper_phys_context_modulation"' in text
    assert 'parser.add_argument("--phys_context_source"' in text
    assert 'parser.add_argument("--hyper_phys_formula_operator"' in text
    assert 'parser.add_argument("--phys_formula_mode"' in text
    assert 'parser.add_argument("--phys_formula_source"' in text
    assert 'parser.add_argument("--hyper_phys_delta_scale"' in text
    assert 'parser.add_argument("--hyper_phys_gate_init"' in text
    assert 'parser.add_argument("--hyper_operator_droppath_p"' in text
    assert 'parser.add_argument("--hyper_phys_consistency_guard"' in text
    assert 'parser.add_argument("--phys_consistency_guard_mode"' in text
    assert 'parser.add_argument("--phys_consistency_source"' in text
    assert 'parser.add_argument("--phys_consistency_min_surface"' in text
    assert 'parser.add_argument("--phys_consistency_min_rootzone"' in text
    assert 'parser.add_argument("--phys_consistency_strength_surface"' in text
    assert 'parser.add_argument("--phys_consistency_strength_rootzone"' in text
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
    assert '"hyper_source_manifold_guard": bool(args.hyper_source_manifold_guard)' in text
    assert '"hyper_source_manifold_guard_strength": args.hyper_source_manifold_guard_strength' in text
    assert '"source_manifold_guard_calibration": args.source_manifold_guard_calibration' in text
    assert '"hyper_phys_context_modulation": bool(args.hyper_phys_context_modulation)' in text
    assert '"phys_context_source": args.phys_context_source' in text
    assert '"hyper_phys_formula_operator": bool(args.hyper_phys_formula_operator)' in text
    assert '"phys_formula_mode": args.phys_formula_mode' in text
    assert '"phys_formula_source": args.phys_formula_source' in text
    assert '"phys_formula_feature_schema": list(phys_formula_feature_schema_for_source(args.phys_formula_source))' in text
    assert '"hyper_phys_delta_scale": args.hyper_phys_delta_scale' in text
    assert '"hyper_phys_gate_init": args.hyper_phys_gate_init' in text
    assert '"hyper_operator_droppath_p": args.hyper_operator_droppath_p' in text
    assert '"hyper_phys_consistency_guard": bool(args.hyper_phys_consistency_guard)' in text
    assert '"phys_consistency_guard_mode": args.phys_consistency_guard_mode' in text
    assert '"phys_consistency_source": args.phys_consistency_source' in text
    assert '"phys_consistency_min_surface": args.phys_consistency_min_surface' in text
    assert '"phys_consistency_min_rootzone": args.phys_consistency_min_rootzone' in text
    assert '"phys_consistency_strength_surface": args.phys_consistency_strength_surface' in text
    assert '"phys_consistency_strength_rootzone": args.phys_consistency_strength_rootzone' in text
    assert '"trust_routing_geometry": "prompt_embedding"' in text
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
    assert 'TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-source_base_frozen_adapter_film}"' in text
    assert '--trainable_scope "${TRAINABLE_SCOPE}"' in text
    assert 'SELECTION_METRIC="${SELECTION_METRIC:-source_val_transfer_safe_score}"' in text
    assert '--selection_metric "${SELECTION_METRIC}"' in text
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
    assert "M2_6_source_manifold_guarded_prior" in text
    assert "M2_4_target_context_conservative_hyperda" not in text
    assert "M3_film_only" in text
    assert "M4_adapter_only" in text
    assert "M3_7_phys_formula_consistency_guarded_trust" in text
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
    assert "--hyper_source_manifold_guard" in text
    assert "--hyper_source_manifold_guard_strength" in text
    assert "--hyper_source_manifold_guard_distance_key" in text
    assert "--source_manifold_guard_calibration" in text
    assert "--hyper_enable_film" in text
    assert "--hyper_enable_adapters" in text
    assert "--hyper_phys_consistency_guard" in text
    assert "--phys_consistency_guard_mode" in text
    assert "--phys_consistency_source" in text
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
    assert "HYPER_SOURCE_MANIFOLD_GUARD=1" in text
    assert "SOURCE_MANIFOLD_GUARD_CALIBRATION=source_fit_source_val_only" in text
    assert "HYPER_SOURCE_MANIFOLD_GUARD_DISTANCE_KEY=source_manifold_distance_bounded" in text
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


def test_staged_ablation_m2_6_source_manifold_guarded_prior_dry_run(tmp_path):
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
        env={**os.environ, "ABLATION_ID": "M2_6_source_manifold_guarded_prior"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M2_6_source_manifold_guarded_prior" in result.stdout
    assert "diagnostic_status=diagnostic_source_manifold_guarded_prior" in result.stdout
    assert "m2_1_anchor=context_encoder=current_mean_std" in result.stdout
    assert "context_encoder=current_mean_std" in result.stdout
    assert "hyper_coeff_generator=shared_layer_aware_rank_gated_stable" in result.stdout
    assert "hyper_adapter_param_style=dora_like_gain_bounded" in result.stdout
    assert "hyper_rank_gate_top_k=4" in result.stdout
    assert "hyper_rank_gate_temperature_init=2.0" in result.stdout
    assert "zero_shot_prior_form=source_base_residual_reliability_gated" in result.stdout
    assert "source_residual_rho=1.0" in result.stdout
    assert "source_residual_gate_init=0.95" in result.stdout
    assert "hyper_source_manifold_guard=1" in result.stdout
    assert "hyper_source_manifold_guard_distance_key=source_manifold_distance_bounded" in result.stdout
    assert "source_manifold_guard_calibration=source_fit_source_val_only" in result.stdout
    assert "target_eval_usage=final_eval_only_no_selection" in result.stdout
    assert "use_amp=0" in result.stdout
    assert "batch_size=64 accum_steps=2 lr=2e-4" in result.stdout
    assert "--hyper_source_manifold_guard 1" in result.stdout
    assert "--source_manifold_guard_calibration source_fit_source_val_only" in result.stdout


def test_staged_ablation_m3_hyperda_trust_medium_dry_run(tmp_path):
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
        env={**os.environ, "ABLATION_ID": "M3_1_hyperda_trust_medium"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_1_hyperda_trust_medium" in result.stdout
    assert "diagnostic_status=hyperda_trust_source_manifold_trust_routed_operator_generation" in result.stdout
    assert "trust_bank_source=source_fit_source_val_only" in result.stdout
    assert "label_usage=none" in result.stdout
    assert "target_eval_usage=final_eval_only_no_selection" in result.stdout
    assert "hyper_source_trust_routing=1" in result.stdout
    assert "hyper_source_trust_strength=0.50" in result.stdout
    assert "hyper_source_trust_top_m=4" in result.stdout
    assert "hyper_source_trust_variable_gate=1" in result.stdout
    assert "source_trust_bank_calibration=source_fit_source_val_only" in result.stdout
    assert "selection_metric=source_val_dual_variable_cvar_safe_score" in result.stdout
    assert "--init_from_source_base_checkpoint" in result.stdout
    assert "--init_from_prompt_checkpoint" not in result.stdout.split("DRY_RUN:")[1]
    assert "--hyper_source_trust_routing 1" in result.stdout
    assert "--hyper_source_trust_strength 0.50" in result.stdout
    assert "--source_trust_bank_calibration source_fit_source_val_only" in result.stdout
    assert "--selection_metric source_val_dual_variable_cvar_safe_score" in result.stdout


def test_staged_ablation_m3_5_phys_agreement_guarded_trust_dry_run(tmp_path):
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
        env={**os.environ, "ABLATION_ID": "M3_5_phys_agreement_guarded_trust"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_5_phys_agreement_guarded_trust" in result.stdout
    assert "diagnostic_status=phys_agreement_guarded_trust_source_gated_candidate" in result.stdout
    assert "m3_1_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std" in result.stdout
    assert "prompt_trust_geometry=prompt_embedding" in result.stdout
    assert "phys_trust_usage=guard_only_shrink_no_enhance" in result.stdout
    assert "phys_guard_reads=x,x_raw,month,region_mask_only" in result.stdout
    assert "target_eval_usage=final_eval_only_no_selection" in result.stdout
    assert "source_val_gate=reject_if_dual_cvar_more_than_0.005_below_M3_1" in result.stdout
    assert "hyper_source_trust_routing=1" in result.stdout
    assert "hyper_source_trust_strength=0.50" in result.stdout
    assert "hyper_source_trust_top_m=4" in result.stdout
    assert "source_trust_query_mode=raw_input_side_da_diagnostics" in result.stdout
    assert "hyper_phys_agreement_guard=1" in result.stdout
    assert "hyper_phys_agreement_guard_strength=1.0" in result.stdout
    assert "source_trust_query_used_as_neighbor_geometry=false" in result.stdout
    assert "--hyper_phys_agreement_guard 1" in result.stdout
    assert "--hyper_phys_agreement_guard_strength 1.0" in result.stdout
    assert "--source_trust_query_mode raw_input_side_da_diagnostics" in result.stdout
    assert "--selection_metric source_val_dual_variable_cvar_safe_score" in result.stdout


def test_staged_ablation_m3_5b_conservative_floor_guard_dry_run(tmp_path):
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
        env={**os.environ, "ABLATION_ID": "M3_5b_phys_agreement_floor_guard"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_5b_phys_agreement_floor_guard" in result.stdout
    assert "diagnostic_status=phys_agreement_floor_guard_source_gated_candidate" in result.stdout
    assert "m3_1_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std" in result.stdout
    assert "phys_guard_risk_rule=and" in result.stdout
    assert "phys_guard_min_multiplier=0.8" in result.stdout
    assert "phys_trust_usage=guard_only_bounded_shrink_no_enhance" in result.stdout
    assert "source_trust_query_used_as_neighbor_geometry=false" in result.stdout
    assert "hyper_phys_agreement_guard=1" in result.stdout
    assert "hyper_phys_agreement_guard_strength=1.0" in result.stdout
    assert "hyper_phys_agreement_guard_min_multiplier=0.8" in result.stdout
    assert "hyper_phys_agreement_guard_risk_rule=and" in result.stdout
    assert "--hyper_phys_agreement_guard 1" in result.stdout
    assert "--hyper_phys_agreement_guard_strength 1.0" in result.stdout
    assert "--hyper_phys_agreement_guard_min_multiplier 0.8" in result.stdout
    assert "--hyper_phys_agreement_guard_risk_rule and" in result.stdout
    assert "--source_trust_query_mode raw_input_side_da_diagnostics" in result.stdout
    assert "--selection_metric source_val_dual_variable_cvar_safe_score" in result.stdout


def test_staged_ablation_m3_6_phys_token_operator_droppath_trust_dry_run(tmp_path):
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
            "ABLATION_ID": "M3_6_phys_token_operator_droppath_trust",
            "RESUME_FROM_M3_1_BEST": "0",
            "TRAINABLE_SCOPE": "phys_context_only",
            "MAX_EPOCHS": "5",
            "EVAL_EVERY_EPOCHS": "5",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_6_phys_token_operator_droppath_trust" in result.stdout
    assert "diagnostic_status=phys_token_operator_droppath_trust_source_gated_candidate" in result.stdout
    assert "stage2_candidate=true" in result.stdout
    assert "m3_1_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std" in result.stdout
    assert "prompt_trust_geometry=prompt_embedding" in result.stdout
    assert "trust_routing_geometry=prompt_embedding" in result.stdout
    assert "phys_token_usage=operator_coefficient_delta_only" in result.stdout
    assert "phys_context_source=raw_input_side_da_diagnostics" in result.stdout
    assert "phys_delta_scale=0.25" in result.stdout
    assert "phys_gate_init=0.90" in result.stdout
    assert "operator_droppath_p=0.10" in result.stdout
    assert "operator_droppath_train_only=true" in result.stdout
    assert "phys_delta_head_zero_init=true" in result.stdout
    assert "source_val_gate=reject_if_dual_cvar_more_than_0.005_below_M3_1" in result.stdout
    assert "source_trust_query_mode=prompt_embedding" in result.stdout
    assert "source_trust_query_used_as_neighbor_geometry=false" in result.stdout
    assert "trainable_scope=phys_context_only" in result.stdout
    assert "max_epochs 5" in result.stdout
    assert "--hyper_phys_context_modulation 1" in result.stdout
    assert "--phys_context_source raw_input_side_da_diagnostics" in result.stdout
    assert "--hyper_phys_delta_scale 0.25" in result.stdout
    assert "--hyper_phys_gate_init 0.90" in result.stdout
    assert "--hyper_operator_droppath_p 0.10" in result.stdout
    assert "--trainable_scope phys_context_only" in result.stdout
    assert "--selection_metric source_val_dual_variable_cvar_safe_score" in result.stdout


def test_staged_ablation_m3_7_phys_formula_consistency_guarded_trust_dry_run(tmp_path):
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
            "ABLATION_ID": "M3_7_phys_formula_consistency_guarded_trust",
            "RESUME_FROM_M3_1_BEST": "0",
            "TRAINABLE_SCOPE": "none",
            "MAX_EPOCHS": "0",
            "EVAL_EVERY_EPOCHS": "1",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_7_phys_formula_consistency_guarded_trust" in result.stdout
    assert "diagnostic_status=phys_formula_consistency_guarded_trust_source_gated_candidate" in result.stdout
    assert "stage2_candidate=false" in result.stdout
    assert "eval_only_guard_supported=true" in result.stdout
    assert "m3_1_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std" in result.stdout
    assert "prompt_trust_geometry=prompt_embedding" in result.stdout
    assert "trust_routing_geometry=prompt_embedding" in result.stdout
    assert "phys_consistency_usage=raw_input_side_formula_variable_trust_gate_shrink_or_identity" in result.stdout
    assert "phys_consistency_source=raw_input_side_formula" in result.stdout
    assert "phys_guard_reads=x_raw,month,region_mask_only" in result.stdout
    assert "source_val_gate=reject_if_dual_cvar_more_than_0.005_below_M3_1" in result.stdout
    assert "target_eval_usage=final_eval_only_no_selection" in result.stdout
    assert "source_trust_query_mode=prompt_embedding" in result.stdout
    assert "source_trust_query_used_as_neighbor_geometry=false" in result.stdout
    assert "warm_start_policy=M3_1_best_checkpoint_eval_only_source_gate" in result.stdout
    assert "hyper_phys_consistency_guard=1" in result.stdout
    assert "phys_consistency_guard_mode=enkf_rt_vertical" in result.stdout
    assert "phys_consistency_min_surface=0.95" in result.stdout
    assert "phys_consistency_min_rootzone=0.90" in result.stdout
    assert "phys_consistency_strength_surface=0.10" in result.stdout
    assert "phys_consistency_strength_rootzone=0.15" in result.stdout
    assert "trainable_scope=none" in result.stdout
    assert "max_epochs 0" in result.stdout
    assert "--hyper_phys_consistency_guard 1" in result.stdout
    assert "--phys_consistency_guard_mode enkf_rt_vertical" in result.stdout
    assert "--phys_consistency_source raw_input_side_formula" in result.stdout
    assert "--phys_consistency_min_surface 0.95" in result.stdout
    assert "--phys_consistency_min_rootzone 0.90" in result.stdout
    assert "--phys_consistency_strength_surface 0.10" in result.stdout
    assert "--phys_consistency_strength_rootzone 0.15" in result.stdout
    assert "--trainable_scope none" in result.stdout
    assert "--selection_metric source_val_dual_variable_cvar_safe_score" in result.stdout


def test_staged_ablation_m3_8_phys_formula_operator_trust_dry_run(tmp_path):
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
            "ABLATION_ID": "M3_8_phys_formula_operator_trust",
            "RESUME_FROM_M3_1_BEST": "0",
            "TRAINABLE_SCOPE": "phys_formula_context_only",
            "MAX_EPOCHS": "5",
            "EVAL_EVERY_EPOCHS": "1",
            "LR": "1e-4",
            "USE_AMP": "0",
            "SOURCE_FIT_MAX_BATCHES_PER_EPOCH": "384",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_8_phys_formula_operator_trust" in result.stdout
    assert "diagnostic_status=phys_formula_operator_trust_source_gated_candidate" in result.stdout
    assert "stage2_candidate=source_side_only" in result.stdout
    assert "m3_1_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std" in result.stdout
    assert "prompt_trust_geometry=prompt_embedding" in result.stdout
    assert "trust_routing_geometry=prompt_embedding" in result.stdout
    assert "phys_formula_usage=raw_input_side_formula_operator_coefficient_delta" in result.stdout
    assert "phys_context_source=raw_input_side_formula_v2" in result.stdout
    assert "phys_formula_mode=enkf_rt_vertical_temp" in result.stdout
    assert "phys_formula_source=raw_input_side_formula_v2" in result.stdout
    assert "phys_formula_delta_scale=0.10" in result.stdout
    assert "phys_formula_gate_init=0.50" in result.stdout
    assert "phys_formula_operator_droppath_p=0.10" in result.stdout
    assert "phys_consistency_usage=surface_primary_product_variable_trust_gate_shrink_or_identity" in result.stdout
    assert "phys_consistency_guard_mode=surface_primary_enkf_rt_vertical_product" in result.stdout
    assert "phys_consistency_source=raw_input_side_formula_v2" in result.stdout
    assert "channel_11_usage=diagnostic_only_not_hard_mask" in result.stdout
    assert "source_val_anchor_dual_cvar=0.446573390549" in result.stdout
    assert "source_val_anchor_rmse_surface=0.004712299814" in result.stdout
    assert "source_val_anchor_rmse_rootzone=0.000889948021" in result.stdout
    assert "target_eval_usage=final_eval_only_no_selection" in result.stdout
    assert "source_trust_query_used_as_neighbor_geometry=false" in result.stdout
    assert "warm_start_policy=M3_1_best_checkpoint_phys_formula_branch_only_first_screen" in result.stdout
    assert "hyper_phys_formula_operator=1" in result.stdout
    assert "hyper_phys_consistency_guard=1" in result.stdout
    assert "phys_consistency_min_surface=0.97" in result.stdout
    assert "phys_consistency_min_rootzone=0.98" in result.stdout
    assert "phys_consistency_strength_surface=0.05" in result.stdout
    assert "phys_consistency_strength_rootzone=0.02" in result.stdout
    assert "trainable_scope=phys_formula_context_only" in result.stdout
    assert "max_epochs 5" in result.stdout
    assert "source_fit_max_batches_per_epoch=384" in result.stdout
    assert "source_fit_fast_screen=enabled_source_fit_training_cap_full_source_val" in result.stdout
    assert "--hyper_phys_formula_operator 1" in result.stdout
    assert "--phys_formula_mode enkf_rt_vertical_temp" in result.stdout
    assert "--phys_formula_source raw_input_side_formula_v2" in result.stdout
    assert "--phys_context_source raw_input_side_formula_v2" in result.stdout
    assert "--hyper_phys_delta_scale 0.10" in result.stdout
    assert "--hyper_phys_gate_init 0.50" in result.stdout
    assert "--hyper_operator_droppath_p 0.10" in result.stdout
    assert "--phys_consistency_guard_mode surface_primary_enkf_rt_vertical_product" in result.stdout
    assert "--phys_consistency_source raw_input_side_formula_v2" in result.stdout
    assert "--phys_consistency_min_surface 0.97" in result.stdout
    assert "--phys_consistency_min_rootzone 0.98" in result.stdout
    assert "--phys_consistency_strength_surface 0.05" in result.stdout
    assert "--phys_consistency_strength_rootzone 0.02" in result.stdout
    assert "--trainable_scope phys_formula_context_only" in result.stdout
    assert "--source_fit_max_batches_per_epoch 384" in result.stdout
    assert "--selection_metric source_val_dual_variable_cvar_safe_score" in result.stdout


def test_staged_ablation_m3_8_defaults_to_three_epoch_source_screen(tmp_path):
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source checkpoint")

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"MAX_EPOCHS", "SOURCE_FIT_MAX_BATCHES_PER_EPOCH"}
    }
    env.update(
        {
            "PYTHONPATH": ".",
            "ABLATION_ID": "M3_8_phys_formula_operator_trust",
            "RESUME_FROM_M3_1_BEST": "0",
            "TRAINABLE_SCOPE": "phys_formula_context_only",
            "EVAL_EVERY_EPOCHS": "1",
            "LR": "1e-4",
            "USE_AMP": "0",
        }
    )
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
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--max_epochs 3" in result.stdout
    assert "source_fit_max_batches_per_epoch=0" in result.stdout
    assert "source_fit_fast_screen=disabled_full_source_fit_training" in result.stdout


def test_staged_ablation_m3_8_respects_explicit_phys_override_dry_run(tmp_path):
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
            "ABLATION_ID": "M3_8_phys_formula_operator_trust",
            "RESUME_FROM_M3_1_BEST": "0",
            "TRAINABLE_SCOPE": "phys_formula_context_only",
            "MAX_EPOCHS": "3",
            "EVAL_EVERY_EPOCHS": "2",
            "LR": "1e-4",
            "USE_AMP": "0",
            "SOURCE_FIT_MAX_BATCHES_PER_EPOCH": "384",
            "HYPER_PHYS_CONSISTENCY_GUARD": "0",
            "HYPER_PHYS_DELTA_SCALE": "0.05",
            "HYPER_PHYS_GATE_INIT": "0.35",
            "HYPER_OPERATOR_DROPPATH_P": "0.0",
            "PHYS_CONSISTENCY_GUARD_MODE": "enkf_rt_vertical",
            "PHYS_CONSISTENCY_SOURCE": "raw_input_side_formula",
            "PHYS_CONSISTENCY_MIN_SURFACE": "0.91",
            "PHYS_CONSISTENCY_MIN_ROOTZONE": "0.92",
            "PHYS_CONSISTENCY_STRENGTH_SURFACE": "0.03",
            "PHYS_CONSISTENCY_STRENGTH_ROOTZONE": "0.04",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_8_phys_formula_operator_trust" in result.stdout
    assert "hyper_phys_consistency_guard=0" in result.stdout
    assert "phys_formula_delta_scale=0.05" in result.stdout
    assert "phys_formula_gate_init=0.35" in result.stdout
    assert "phys_formula_operator_droppath_p=0.0" in result.stdout
    assert "phys_consistency_guard_mode=enkf_rt_vertical" in result.stdout
    assert "phys_consistency_source=raw_input_side_formula" in result.stdout
    assert "phys_consistency_min_surface=0.91" in result.stdout
    assert "phys_consistency_min_rootzone=0.92" in result.stdout
    assert "phys_consistency_strength_surface=0.03" in result.stdout
    assert "phys_consistency_strength_rootzone=0.04" in result.stdout
    assert "source_fit_max_batches_per_epoch=384" in result.stdout
    assert "source_fit_fast_screen=enabled_source_fit_training_cap_full_source_val" in result.stdout
    assert "--hyper_phys_consistency_guard 0" in result.stdout
    assert "--hyper_phys_delta_scale 0.05" in result.stdout
    assert "--hyper_phys_gate_init 0.35" in result.stdout
    assert "--hyper_operator_droppath_p 0.0" in result.stdout
    assert "--phys_consistency_guard_mode enkf_rt_vertical" in result.stdout
    assert "--phys_consistency_source raw_input_side_formula" in result.stdout
    assert "--phys_consistency_min_surface 0.91" in result.stdout
    assert "--phys_consistency_min_rootzone 0.92" in result.stdout
    assert "--phys_consistency_strength_surface 0.03" in result.stdout
    assert "--phys_consistency_strength_rootzone 0.04" in result.stdout
    assert "--source_fit_max_batches_per_epoch 384" in result.stdout


def test_staged_ablation_m3_8b_phys_formula_light_guarded_trust_dry_run(tmp_path):
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source checkpoint")

    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "MAX_EPOCHS",
            "SOURCE_FIT_MAX_BATCHES_PER_EPOCH",
            "TRAINABLE_SCOPE",
            "LR",
            "USE_AMP",
            "RESUME_FROM_M3_1_BEST",
            "M3_8_INIT_FROM_M3_1_CHECKPOINT",
            "HYPER_PHYS_DELTA_SCALE",
            "HYPER_PHYS_GATE_INIT",
            "HYPER_PHYS_CONSISTENCY_GUARD",
            "PHYS_CONSISTENCY_MIN_SURFACE",
            "PHYS_CONSISTENCY_MIN_ROOTZONE",
            "PHYS_CONSISTENCY_STRENGTH_SURFACE",
            "PHYS_CONSISTENCY_STRENGTH_ROOTZONE",
        }
    }
    env.update(
        {
            "PYTHONPATH": ".",
            "ABLATION_ID": "M3_8b_phys_formula_light_guarded_trust",
            "M3_8_INIT_FROM_M3_1_CHECKPOINT": str(tmp_path / "must_not_be_used.pt"),
        }
    )
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
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_8b_phys_formula_light_guarded_trust" in result.stdout
    assert "diagnostic_status=phys_formula_light_guarded_trust_main_method_candidate" in result.stdout
    assert "main_method_candidate=Physics-informed HyperDA-TRUST" in result.stdout
    assert "m3_1_design_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std" in result.stdout
    assert "source_stage_initialization=source_only_checkpoint_clean_hypernetwork_training" in result.stdout
    assert "prompt_checkpoint_warm_start=none_for_clean_physics_candidate" in result.stdout
    assert "warm_start_policy=none_clean_source_only_checkpoint_full_hypernetwork_training" in result.stdout
    assert "phys_formula_usage=raw_input_side_formula_operator_coefficient_delta" in result.stdout
    assert "phys_formula_delta_scale=0.05" in result.stdout
    assert "phys_formula_gate_init=0.35" in result.stdout
    assert "hyper_phys_consistency_guard=1" in result.stdout
    assert "phys_consistency_usage=shrink_only_high_floor_variable_trust_gate" in result.stdout
    assert "phys_consistency_min_surface=0.985" in result.stdout
    assert "phys_consistency_min_rootzone=0.99" in result.stdout
    assert "phys_consistency_strength_surface=0.02" in result.stdout
    assert "phys_consistency_strength_rootzone=0.01" in result.stdout
    assert "source_val_selection_rule=dual_cvar_gte_M3_1_minus_0.005_choose_best_source_safe_score_tie_rmse" in result.stdout
    assert "target_eval_usage=final_eval_only_no_selection" in result.stdout
    assert "trainable_scope=source_base_frozen_adapter_film" in result.stdout
    assert "use_amp=0" in result.stdout
    assert "batch_size=64 accum_steps=2 lr=2e-4" in result.stdout
    assert "--max_epochs 50" in result.stdout
    assert "--eval_every_epochs 5" in result.stdout
    assert "--init_from_source_base_checkpoint" in result.stdout
    assert "--init_from_prompt_checkpoint" not in result.stdout.split("DRY_RUN:")[1]
    assert "--hyper_phys_formula_operator 1" in result.stdout
    assert "--phys_formula_source raw_input_side_formula_v2" in result.stdout
    assert "--phys_context_source raw_input_side_formula_v2" in result.stdout
    assert "--hyper_phys_delta_scale 0.05" in result.stdout
    assert "--hyper_phys_gate_init 0.35" in result.stdout
    assert "--hyper_phys_consistency_guard 1" in result.stdout
    assert "--phys_consistency_min_surface 0.985" in result.stdout
    assert "--phys_consistency_min_rootzone 0.99" in result.stdout
    assert "--phys_consistency_strength_surface 0.02" in result.stdout
    assert "--phys_consistency_strength_rootzone 0.01" in result.stdout
    assert "--selection_metric source_val_dual_variable_cvar_safe_score" in result.stdout
    assert "target_eval" not in result.stdout.split("DRY_RUN:")[1]


def test_staged_ablation_m3_8c_phys_formula_operator_only_trust_dry_run(tmp_path):
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source checkpoint")

    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "MAX_EPOCHS",
            "SOURCE_FIT_MAX_BATCHES_PER_EPOCH",
            "TRAINABLE_SCOPE",
            "LR",
            "USE_AMP",
            "RESUME_FROM_M3_1_BEST",
            "M3_8_INIT_FROM_M3_1_CHECKPOINT",
            "HYPER_PHYS_DELTA_SCALE",
            "HYPER_PHYS_GATE_INIT",
            "HYPER_PHYS_CONSISTENCY_GUARD",
        }
    }
    env.update(
        {
            "PYTHONPATH": ".",
            "ABLATION_ID": "M3_8c_phys_formula_light_operator_only_trust",
            "M3_8_INIT_FROM_M3_1_CHECKPOINT": str(tmp_path / "must_not_be_used.pt"),
        }
    )
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
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_8c_phys_formula_light_operator_only_trust" in result.stdout
    assert "diagnostic_status=phys_formula_light_operator_only_trust_guard_ablation" in result.stdout
    assert "main_method_candidate=Physics-informed HyperDA-TRUST" in result.stdout
    assert "m3_1_design_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std" in result.stdout
    assert "source_stage_initialization=source_only_checkpoint_clean_hypernetwork_training" in result.stdout
    assert "prompt_checkpoint_warm_start=none_for_clean_physics_candidate" in result.stdout
    assert "warm_start_policy=none_clean_source_only_checkpoint_full_hypernetwork_training" in result.stdout
    assert "phys_formula_usage=raw_input_side_formula_operator_coefficient_delta" in result.stdout
    assert "phys_formula_delta_scale=0.05" in result.stdout
    assert "phys_formula_gate_init=0.35" in result.stdout
    assert "hyper_phys_consistency_guard=0" in result.stdout
    assert "phys_consistency_usage=disabled_operator_only_guard_ablation" in result.stdout
    assert "source_val_selection_rule=dual_cvar_gte_M3_1_minus_0.005_choose_best_source_safe_score_tie_rmse" in result.stdout
    assert "target_eval_usage=final_eval_only_no_selection" in result.stdout
    assert "trainable_scope=source_base_frozen_adapter_film" in result.stdout
    assert "use_amp=0" in result.stdout
    assert "batch_size=64 accum_steps=2 lr=2e-4" in result.stdout
    assert "--max_epochs 50" in result.stdout
    assert "--eval_every_epochs 5" in result.stdout
    assert "--init_from_source_base_checkpoint" in result.stdout
    assert "--init_from_prompt_checkpoint" not in result.stdout.split("DRY_RUN:")[1]
    assert "--hyper_phys_formula_operator 1" in result.stdout
    assert "--phys_formula_source raw_input_side_formula_v2" in result.stdout
    assert "--phys_context_source raw_input_side_formula_v2" in result.stdout
    assert "--hyper_phys_delta_scale 0.05" in result.stdout
    assert "--hyper_phys_gate_init 0.35" in result.stdout
    assert "--hyper_phys_consistency_guard 0" in result.stdout
    assert "--selection_metric source_val_dual_variable_cvar_safe_score" in result.stdout
    assert "target_eval" not in result.stdout.split("DRY_RUN:")[1]


def test_staged_ablation_m3_8b_8c_reject_m3_1_checkpoint_resume(tmp_path):
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source checkpoint")

    for ablation_id in [
        "M3_8b_phys_formula_light_guarded_trust",
        "M3_8c_phys_formula_light_operator_only_trust",
    ]:
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
                "PYTHONPATH": ".",
                "ABLATION_ID": ablation_id,
                "RESUME_FROM_M3_1_BEST": "1",
            },
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 2
        assert "clean source-stage physics preset cannot warm-start from M3_1" in result.stderr


def test_staged_ablation_m3_9_phys_formula_enhanced_trust_dry_run_defaults_to_cheap_screen(tmp_path):
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source checkpoint")

    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "MAX_EPOCHS",
            "SOURCE_FIT_MAX_BATCHES_PER_EPOCH",
            "TRAINABLE_SCOPE",
            "LR",
            "USE_AMP",
            "BATCH_SIZE",
            "ACCUM_STEPS",
            "HYPER_PHYS_DELTA_SCALE",
            "HYPER_PHYS_GATE_INIT",
            "HYPER_OPERATOR_DROPPATH_P",
            "HYPER_PHYS_CONSISTENCY_GUARD",
            "PHYS_CONSISTENCY_MIN_SURFACE",
            "PHYS_CONSISTENCY_MIN_ROOTZONE",
            "PHYS_CONSISTENCY_STRENGTH_SURFACE",
            "PHYS_CONSISTENCY_STRENGTH_ROOTZONE",
        }
    }
    env.update(
        {
            "PYTHONPATH": ".",
            "ABLATION_ID": "M3_9_phys_formula_enhanced_trust",
            "RESUME_FROM_M3_1_BEST": "0",
        }
    )
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
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_9_phys_formula_enhanced_trust" in result.stdout
    assert "diagnostic_status=phys_formula_enhanced_trust_source_gated_candidate" in result.stdout
    assert "m3_1_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std" in result.stdout
    assert "prompt_trust_geometry=prompt_embedding" in result.stdout
    assert "trust_routing_geometry=prompt_embedding" in result.stdout
    assert "phys_formula_usage=enhanced_raw_input_side_formula_operator_coefficient_delta" in result.stdout
    assert "phys_context_source=raw_input_side_formula_v3_enhanced" in result.stdout
    assert "phys_formula_source=raw_input_side_formula_v3_enhanced" in result.stdout
    assert "phys_formula_schema=phys_formula_operator_v2_enhanced_input_side" in result.stdout
    assert "phys_formula_enhanced_features=tb_hv_normalized_innovation,innovation_asymmetry,polarization_mismatch,vegopacity,weak_obs,finite,temp_contrast,vertical_decoupling,hydraulic_gradient,channel11_diagnostic" in result.stdout
    assert "phys_formula_delta_scale=0.05" in result.stdout
    assert "phys_formula_gate_init=0.35" in result.stdout
    assert "phys_formula_operator_droppath_p=0.10" in result.stdout
    assert "phys_delta_head_zero_init=true" in result.stdout
    assert "phys_consistency_usage=shrink_only_high_floor_variable_trust_gate" in result.stdout
    assert "phys_consistency_min_surface=0.98" in result.stdout
    assert "phys_consistency_min_rootzone=0.99" in result.stdout
    assert "source_side_cheap_screen=US-R1_source_val_only_cap384_no_target_eval" in result.stdout
    assert "source_fit_max_batches_per_epoch=384" in result.stdout
    assert "source_fit_fast_screen=enabled_source_fit_training_cap_full_source_val" in result.stdout
    assert "source_val_gate=cap384_select_best_rootzone_if_dual_cvar_within_0.0005_of_M3_8_V1_and_surface_degrade_lte_0.2pct_else_stop" in result.stdout
    assert "full_confirmation_gate=dual_cvar_gte_0.441573390549_surface_rootzone_rmse_degrade_lte_0.5pct_vs_M3_1_leakage_clean" in result.stdout
    assert "target_eval_usage=final_eval_only_no_selection" in result.stdout
    assert "source_trust_query_used_as_neighbor_geometry=false" in result.stdout
    assert "warm_start_policy=M3_1_best_checkpoint_phys_formula_branch_only_first_screen" in result.stdout
    assert "hyper_phys_formula_operator=1" in result.stdout
    assert "hyper_phys_consistency_guard=1" in result.stdout
    assert "trainable_scope=phys_formula_context_only" in result.stdout
    assert "use_amp=0" in result.stdout
    assert "batch_size=64 accum_steps=2 lr=1e-4" in result.stdout
    assert "--max_epochs 3" in result.stdout
    assert "--hyper_phys_formula_operator 1" in result.stdout
    assert "--phys_formula_source raw_input_side_formula_v3_enhanced" in result.stdout
    assert "--phys_context_source raw_input_side_formula_v3_enhanced" in result.stdout
    assert "--hyper_phys_delta_scale 0.05" in result.stdout
    assert "--hyper_phys_gate_init 0.35" in result.stdout
    assert "--hyper_operator_droppath_p 0.10" in result.stdout
    assert "--phys_consistency_min_surface 0.98" in result.stdout
    assert "--phys_consistency_min_rootzone 0.99" in result.stdout
    assert "--source_fit_max_batches_per_epoch 384" in result.stdout
    assert "--selection_metric source_val_dual_variable_cvar_safe_score" in result.stdout
    assert "--split_manifest_path artifacts/splits/US_loro_zero_few_shot_splits.json" in result.stdout
    assert "target_eval" not in result.stdout.split("DRY_RUN:")[1]


def test_staged_ablation_m3_1plus_preregistered_candidate_dry_runs(tmp_path):
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source checkpoint")

    expected = {
        "M3_1a_trust_medium_dualalpha": ("0.50", "4"),
        "M3_1b_trust_mid_high": ("0.375", "4"),
        "M3_1c_trust_medium_local": ("0.50", "2"),
        "M3_1d_trust_medium_broad": ("0.50", "6"),
    }

    for ablation_id, (trust_strength, top_m) in expected.items():
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
            env={**os.environ, "ABLATION_ID": ablation_id},
            text=True,
            capture_output=True,
            check=True,
        )

        assert f"ablation_id={ablation_id}" in result.stdout
        assert "diagnostic_status=hyperda_trust_source_manifold_trust_routed_operator_generation" in result.stdout
        assert "trust_bank_source=source_fit_source_val_only" in result.stdout
        assert "label_usage=none" in result.stdout
        assert "target_eval_usage=final_eval_only_no_selection" in result.stdout
        assert "alpha_selection_objective=dual_variable_cvar_safe_score" in result.stdout
        assert f"hyper_source_trust_strength={trust_strength}" in result.stdout
        assert f"hyper_source_trust_top_m={top_m}" in result.stdout
        assert "selection_metric=source_val_dual_variable_cvar_safe_score" in result.stdout
        assert f"--hyper_source_trust_strength {trust_strength}" in result.stdout
        assert f"--hyper_source_trust_top_m {top_m}" in result.stdout
        assert "--selection_metric source_val_dual_variable_cvar_safe_score" in result.stdout


def test_hyperda_spec_registers_m3_1plus_preregistered_matrix():
    text = Path("specs/hyperda_v4.yaml").read_text()

    for ablation_id in [
        "M3_1a_trust_medium_dualalpha",
        "M3_1b_trust_mid_high",
        "M3_1c_trust_medium_local",
        "M3_1d_trust_medium_broad",
    ]:
        assert ablation_id in text
    assert "alpha_selection_objective: dual_variable_cvar_safe_score" in text
    assert "source_neighbor_top_m: 2" in text
    assert "source_neighbor_top_m: 6" in text
    assert "optional_budget: true" in text


def test_hyperda_spec_and_phase5_task_register_m3_8b_8c_physics_candidates():
    combined = "\n".join(
        [
            Path("specs/hyperda_v4.yaml").read_text(),
            Path("tasks/phase5_hyperda_safe_zero_few_shot.md").read_text(),
        ]
    )

    for ablation_id in [
        "M3_8b_phys_formula_light_guarded_trust",
        "M3_8c_phys_formula_light_operator_only_trust",
    ]:
        assert ablation_id in combined
    assert "Physics-informed HyperDA-TRUST" in combined
    assert "checkpoint_start: source_pooled_global_backbone" in combined
    assert "not_init_from: M3_1_checkpoint_or_M2_1_checkpoint" in combined
    assert "trainable_scope: source_base_frozen_adapter_film" in combined
    assert "max_epochs: 50" in combined
    assert "lr: 2e-4" in combined
    assert "not M3_1 checkpoint fine-tunes" in combined
    assert "--init_from_prompt_checkpoint" in combined
    assert "RESUME_FROM_M3_1_BEST=1" in combined
    assert "source_val_dual_variable_cvar_safe_score >= M3_1 - 0.005" in combined
    assert "HYPER_PHYS_CONSISTENCY_GUARD=0" in combined
    assert "PHYS_CONSISTENCY_MIN_SURFACE=0.985" in combined
    assert "target_eval final-only" in combined


def test_staged_ablation_m3_raw_reliability_uses_raw_trust_query_not_main_prompt(tmp_path):
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
        env={**os.environ, "ABLATION_ID": "M3_2_hyperda_trust_raw_reliability"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_2_hyperda_trust_raw_reliability" in result.stdout
    assert "context_encoder=current_mean_std" in result.stdout
    assert "source_trust_query_mode=raw_input_side_da_diagnostics" in result.stdout
    assert "--context_encoder current_mean_std" in result.stdout
    assert "--source_trust_query_mode raw_input_side_da_diagnostics" in result.stdout


def test_staged_ablation_m3_phys_trust_alias_uses_raw_da_query(tmp_path):
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
        env={**os.environ, "ABLATION_ID": "M3_2_phys_trust_raw_da_query"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_2_phys_trust_raw_da_query" in result.stdout
    assert "context_encoder=current_mean_std" in result.stdout
    assert "source_trust_query_mode=raw_input_side_da_diagnostics" in result.stdout
    assert "--source_trust_query_mode raw_input_side_da_diagnostics" in result.stdout


def test_staged_ablation_m3_2a_fixed_raw_query_uses_separate_raw_bank(tmp_path):
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
        env={**os.environ, "ABLATION_ID": "M3_2a_phys_trust_raw_query_fixed"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_2a_phys_trust_raw_query_fixed" in result.stdout
    assert "context_encoder=current_mean_std" in result.stdout
    assert "source_trust_query_mode=raw_input_side_da_diagnostics" in result.stdout
    assert "has_separate_source_trust_query_required=true" in result.stdout
    assert "--source_trust_query_mode raw_input_side_da_diagnostics" in result.stdout
    assert "--selection_metric source_val_dual_variable_cvar_safe_score" in result.stdout


def test_staged_ablation_m3_4_blended_query_uses_conservative_query_only(tmp_path):
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
        env={**os.environ, "ABLATION_ID": "M3_4_phys_trust_blended_query"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_4_phys_trust_blended_query" in result.stdout
    assert "context_encoder=current_mean_std" in result.stdout
    assert "source_trust_query_mode=blended_prompt_raw_da_0p25" in result.stdout
    assert "source_trust_query_blend_lambda=0.25" in result.stdout
    assert "main_prompt_unchanged_by_blended_query=true" in result.stdout
    assert "--source_trust_query_mode blended_prompt_raw_da_0p25" in result.stdout
    assert "--selection_metric source_val_dual_variable_cvar_safe_score" in result.stdout


def test_staged_ablation_m3_selection_only_dry_run_does_not_change_forward(tmp_path):
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
        env={**os.environ, "ABLATION_ID": "M3_3_hyperda_trust_selection_only"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ablation_id=M3_3_hyperda_trust_selection_only" in result.stdout
    assert "hyper_source_trust_routing=0" in result.stdout
    assert "hyper_source_trust_strength=0.0" in result.stdout
    assert "hyper_source_trust_variable_gate=0" in result.stdout
    assert "selection_metric=source_val_dual_variable_cvar_safe_score" in result.stdout


def test_staged_ablation_rejects_unknown_ablation_without_m2_4_hint(tmp_path):
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
    assert "M2.4" not in result.stderr


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
    assert "hyperda_trust_zero_shot_context" in readme
    assert "HyperDA-TRUST" in readme
    assert "HyperDA-SAFE" not in readme
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
