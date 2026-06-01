"""Baseline registry for HyperDA V4.

Heuristic baselines may exist for debugging, but only the listed paper-main
baselines should be used for manuscript tables.
"""

PAPER_MAIN_BASELINES = {
    "target_full_train": [
        "forecast_only",
        "source_only_backbone",
        "prompt_conditioned_shared_backbone",
        "adapter_tuning_full_target_train",
        "lora_tuning_full_target_train",
        "hyperda_generated_operator_full_target_train",
        "hyperda_refine_full_target_train",
    ],
}

LEGACY_FEW_SHOT_BASELINES = {
    "K0": [
        "forecast_only",
        "source_only_backbone",
        "prompt_conditioned_shared_backbone",
        "hyperda_zero",
    ],
    "K4_K12": [
        "forecast_only",
        "source_only_backbone",
        "adapter_tuning",
        "lora_tuning",
        "prompt_conditioned_shared_backbone_with_calibration_prompt",
        "hyperda_calib",
        "hyperda_refine",
    ],
}

INTERNAL_SANITY_ONLY = {
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
    # Deprecated names retained in old artifacts/scripts.
    "hyperda_basis_adapter",
    "target_support_mean_increment",
    "target_monthly_support_increment",
}


def assert_allowed_for_table(method: str, table: str) -> None:
    if table == "paper_main":
        allowed = set(PAPER_MAIN_BASELINES["target_full_train"])
        if method not in allowed:
            raise ValueError(
                f"Method {method!r} is not allowed in paper_main table under HyperDA V4.2. "
                "Use it only as internal_sanity if needed."
            )
