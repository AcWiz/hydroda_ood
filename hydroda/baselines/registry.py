"""Baseline registry for HyperDA V4.

Heuristic baselines may exist for debugging, but only the listed paper-main
baselines should be used for manuscript tables.
"""

PAPER_MAIN_BASELINES = {
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
    "target_support_mean_increment",
    "source_monthly_mean_increment",
    "target_monthly_support_increment",
    "ridge_calibration",
    "nearest_source_specialist",
    "prompt_weighted_specialist",
    "knn_parameter_interpolation",
    "linear_prompt_to_parameter",
}


def assert_allowed_for_table(method: str, table: str) -> None:
    if table == "paper_main":
        allowed = set(PAPER_MAIN_BASELINES["K0"]) | set(PAPER_MAIN_BASELINES["K4_K12"])
        if method not in allowed:
            raise ValueError(
                f"Method {method!r} is not allowed in paper_main table under HyperDA V4. "
                "Use it only as internal_sanity if needed."
            )
