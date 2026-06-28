"""Baseline registry for HyperDA V4.

Heuristic baselines may exist for debugging, but only the listed paper-main
baselines should be used for manuscript tables.
"""

PAPER_MAIN_BASELINES = {
    "zero_few_shot_generalization": [
        "forecast_only",
        "source_pooled_global_backbone",
        "swad_source_pooled_global_backbone",
        "mixstyle_source_pooled_global_backbone",
        "disam_source_domain_sharpness_alignment",
        "udim_unknown_domain_inconsistency_minimization",
        "moment_alignment_source_domain_invariance",
        "identify_unlearn_source_domain_gradient_ascent",
        # Backward-compatible display alias for old docs/artifacts.
        "source_only_backbone",
        "prompt_conditioned_shared_backbone",
        "source_regime_specialist_bank",
        "hyperda_zero_shot_context",
        "hyperda_trust_zero_shot_context",
    ],
}

SECONDARY_ABLATION_BASELINES = {
    "adapter_lora_kshot": [
        "adapter_tuning",
        "lora_tuning",
    ],
    "legacy_full_target_train": [
        "adapter_tuning_full_target_train",
        "lora_tuning_full_target_train",
        "hyperda_generated_operator_full_target_train",
        "hyperda_refine_full_target_train",
        "target_full_history_region_oracle",
    ],
}

INTERNAL_SANITY_ONLY = {
    "legacy_all_regions_sanity",
    "source_mean_increment",
    "target_train_mean_increment",
    "source_monthly_mean_increment",
    "target_monthly_train_increment",
    "ridge_calibration",
    "nearest_source_specialist",
    "prompt_weighted_specialist",
    "knn_parameter_interpolation",
    "linear_prompt_to_parameter",
    "hyperda_basis_adapter_shared",
    "hyperda_few_shot_k4",
    "hyperda_few_shot_k12",
    "hyperda_safe_few_shot_k4",
    "hyperda_safe_few_shot_k12",
    "hyperda_rise_source_side_router_prior",
    "hyperda_rise_k0_context_router",
    "hyperda_rise_k4_support_posterior",
    "hyperda_rise_k12_support_posterior",
    "deep_coral_target_context_alignment",
    "ssa_reg_target_context_subspace_alignment",
    "tca_target_context_correlation_alignment",
    "self_bootstrap_target_context_consistency_tta",
    "weatherpeft_weather_fm_future_baseline",
    # Deprecated names retained in old artifacts/scripts.
    "hyperda_basis_adapter",
    "target_support_mean_increment",
    "target_monthly_support_increment",
    "source_only_all_regions",
    "source_only_region_specific",
}


def assert_allowed_for_table(method: str, table: str) -> None:
    if table == "paper_main":
        allowed = set(PAPER_MAIN_BASELINES["zero_few_shot_generalization"])
        if method not in allowed:
            raise ValueError(
                f"Method {method!r} is not allowed in paper_main table under HyperDA V4.4. "
                "Use it only as internal_sanity if needed."
            )
