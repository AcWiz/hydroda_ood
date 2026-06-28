#!/usr/bin/env python3
"""Rebuild Stage 3 K-shot overview tables from existing K directories.

The overview is intentionally artifact-driven. It scans run-local ``K0``,
``K4``, and ``K12`` directories so a partial rerun cannot erase rows from a
previously completed K setting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SETTINGS = {
    "0": "zero_shot_context",
    "4": "few_shot_k4",
    "12": "few_shot_k12",
}

FIELDNAMES = [
    "target_region",
    "K",
    "adaptation_setting",
    "stage3_kshot_mode",
    "adapt_recipe",
    "adapt_scope",
    "adapt_solver",
    "support_loss_reduction",
    "adapt_mix_rho",
    "status",
    "method",
    "paper_facing_run",
    "diagnostic_run_reason",
    "policy_source",
    "adaptation_step_policy_source",
    "stage3_posterior_decision",
    "stage3_acceptance_basis",
    "support_gate_status",
    "support_gate_reject_reason",
    "actual_optimizer_steps",
    "optimizer_steps_run",
    "anchor_alpha",
    "target_parameter_l2_drift_pre_anchor_total",
    "target_parameter_l2_drift_post_anchor_total",
    "target_parameter_l2_drift_pre_anchor_target_prompt",
    "target_parameter_l2_drift_post_anchor_target_prompt",
    "target_parameter_l2_drift_monthly_gain",
    "target_labels_used_for_adaptation",
    "target_support_labels_used_for_parameter_update",
    "target_support_labels_used_for_calibration",
    "few_shot_update_type",
    "n_samples_evaluated",
    "n_metric_rows",
    "prediction_content_hash",
    "raw_adapted_prediction_content_hash",
    "post_gate_prediction_content_hash",
    "final_mixed_prediction_content_hash",
    "raw_to_k0_mean_abs_delta",
    "final_mix_to_k0_mean_abs_delta",
    "context_tta_effective",
    "context_tta_source_stat_status",
    "prompt_l2_delta_mean",
    "prediction_delta_vs_no_tta",
    "support_gain_selection_rule",
    "support_gain_best_alpha_raw",
    "support_gain_stable_candidate_alphas",
    "support_gain_selection_margin",
    "support_gain_stability_tolerance",
    "support_gain_paired_support_se_capped",
    "support_nesting_policy",
    "nested_support_dates_hash",
    "support_gain_target_eval_usage",
    "selected_support_candidate_id",
    "support_selection_objective",
    "support_cv_objective_delta",
    "support_cv_objective_delta_se",
    "support_cv_objective_delta_t",
    "support_cv_nested_k4_objective_delta",
    "support_cv_added_objective_delta",
    "k12_vs_k4_cv_objective_delta",
    "k12_vs_k4_cv_rootzone_delta",
    "support_cycle_improvement_fraction",
    "support_gate_cycle_improvement_min_fraction",
    "k12_reference_policy",
    "support_candidate_pool",
    "ridge_lambda",
    "ridge_weighting",
    "ridge_effective_calibration_dof",
    "linearized_ridge_calibration_mode",
    "linearized_ridge_target_eval_usage",
    "linearized_ridge_parameter_scope",
    "support_affine_surface_a",
    "support_affine_surface_b",
    "support_affine_rootzone_a",
    "support_affine_rootzone_b",
    "effective_calibration_dof",
    "support_calibration_dof",
    "support_affine_target_eval_usage",
    "k_specific_prediction_changed",
    "surface_skill_primary",
    "rootzone_skill_primary",
    "eval_time_s",
    "summary",
    "adapt_metadata",
    "metrics_long",
    "metrics_by_region",
    "metrics_by_season",
    "checkpoint",
    "adapted_checkpoint_sha256",
]


def _json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _nested_present(*mappings_and_paths: tuple[dict[str, Any], tuple[str, ...]]) -> Any:
    for mapping, path in mappings_and_paths:
        value: Any = mapping
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value is not None:
            return value
    return None


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _metrics_long_artifact(region_eval_dir: Path) -> Path:
    csv_path = region_eval_dir / "metrics_long.csv"
    gz_path = region_eval_dir / "metrics_long.csv.gz"
    if csv_path.exists():
        return csv_path
    if gz_path.exists():
        return gz_path
    return csv_path


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:.10f}"
    if value is None:
        return "NA"
    return str(value)


def _as_k_value(path: Path) -> str | None:
    name = path.name
    if not name.startswith("K"):
        return None
    suffix = name[1:]
    if suffix in SETTINGS:
        return suffix
    return None


def discover_k_values(output_base: str | Path, requested_k_values: Iterable[str] = ()) -> list[str]:
    """Return requested plus already-present K directories, sorted by protocol order."""
    base = Path(output_base)
    found = {str(k) for k in requested_k_values if str(k) in SETTINGS}
    if base.exists():
        for child in base.iterdir():
            if child.is_dir():
                k_value = _as_k_value(child)
                if k_value is not None:
                    found.add(k_value)
    return [k for k in ("0", "4", "12") if k in found]


def _stage3_decision(summary: dict[str, Any], metadata: dict[str, Any]) -> Any:
    return _nested_present(
        (metadata, ("stage3_posterior_decision",)),
        (metadata, ("stage3_posterior_state", "stage3_posterior_decision")),
        (summary, ("stage3_protocol", "stage3_posterior_decision")),
    )


def _paper_facing(k: str, summary: dict[str, Any], metadata: dict[str, Any]) -> Any:
    decision = _stage3_decision(summary, metadata)
    if int(k) > 0 and decision in {"rejected_to_k0_anchor", "no_update"}:
        return False
    return _nested_present(
        (metadata, ("paper_facing_run",)),
        (summary, ("stage3_protocol", "paper_facing_run")),
    )


def _status(k: str, summary_status: str, summary: dict[str, Any], metadata: dict[str, Any]) -> str:
    if summary_status != "ok":
        return summary_status
    decision = _stage3_decision(summary, metadata)
    paper_facing = _paper_facing(k, summary, metadata)
    if int(k) == 0:
        return "ok"
    if decision == "rejected_to_k0_anchor":
        return "rejected_to_k0_anchor"
    if decision == "no_update":
        return "source_policy_no_update_k0_equivalent"
    if decision == "accepted" and paper_facing is False:
        return "diagnostic_accepted"
    return "ok"


def _metric_block(summary: dict[str, Any], variable: str) -> dict[str, Any]:
    block = summary.get(variable, {}) or {}
    return {
        f"{variable}_skill_primary": block.get("skill_primary", block.get("skill_global")),
    }


def _drift_value(metadata: dict[str, Any], stage: str, key: str) -> Any:
    drift = metadata.get(f"target_parameter_l2_drift_{stage}", {})
    if not isinstance(drift, dict):
        return None
    if key == "monthly_gain":
        return drift.get("monthly_gain", drift.get("monthly_residual_gain"))
    return drift.get(key)


def _support_affine_coeff(metadata: dict[str, Any], variable: str, key: str) -> Any:
    affine = metadata.get("support_affine_calibration", {})
    if not isinstance(affine, dict):
        return None
    coeffs = affine.get("support_affine_coefficients", {})
    if not isinstance(coeffs, dict):
        return None
    block = coeffs.get(variable, {})
    if not isinstance(block, dict):
        return None
    return block.get(key)


def rebuild_overview(
    *,
    output_base: str | Path,
    target_region: str,
    requested_k_values: Iterable[str] = (),
    seed: int | str = 0,
) -> list[dict[str, Any]]:
    """Scan run artifacts, write overview CSV/JSON/MD, and return row dicts."""
    base = Path(output_base)
    k_values = discover_k_values(base, requested_k_values)
    rows: list[dict[str, Any]] = []
    for k in k_values:
        region_eval_dir = base / f"K{k}" / "eval" / target_region
        summary_path = region_eval_dir / "summary.json"
        metadata_path = base / f"K{k}" / "adapt" / "metadata.json"
        metrics_long_path = _metrics_long_artifact(region_eval_dir)
        metrics_by_region_path = region_eval_dir / "metrics_by_region.csv"
        metrics_by_season_path = region_eval_dir / "metrics_by_season.csv"
        checkpoint_path = base / f"K{k}" / "adapt" / "checkpoints" / "checkpoint_final_preregistered.pt"

        summary = _json_load(summary_path)
        metadata = _json_load(metadata_path)
        summary_status = "ok" if summary else "missing_summary"
        row_seed = summary.get("seed", metadata.get("seed", seed))
        row = {
            "target_region": target_region,
            "K": k,
            "adaptation_setting": summary.get("adaptation_setting", SETTINGS[k]),
            "stage3_kshot_mode": metadata.get(
                "stage3_kshot_mode",
                (summary.get("stage3_protocol", {}) or {}).get("stage3_kshot_mode", "paper_safe"),
            ),
            "adapt_recipe": metadata.get("adapt_recipe"),
            "adapt_scope": metadata.get("adapt_scope"),
            "adapt_solver": metadata.get("adapt_solver"),
            "support_loss_reduction": metadata.get("support_loss_reduction"),
            "adapt_mix_rho": summary.get("adapt_mix_rho", metadata.get("adapt_mix_rho")),
            "status": _status(k, summary_status, summary, metadata),
            "method": summary.get("method", metadata.get("method")),
            "paper_facing_run": _paper_facing(k, summary, metadata),
            "diagnostic_run_reason": metadata.get(
                "diagnostic_run_reason",
                (summary.get("stage3_protocol", {}) or {}).get("diagnostic_run_reason"),
            ),
            "policy_source": metadata.get(
                "policy_source",
                (summary.get("stage3_protocol", {}) or {}).get("policy_source"),
            ),
            "adaptation_step_policy_source": _nested_present(
                (metadata, ("adaptation_step_policy_source",)),
                (metadata, ("resolved_mode_defaults", "adaptation_step_policy_source")),
            ),
            "stage3_posterior_decision": _stage3_decision(summary, metadata),
            "stage3_acceptance_basis": _nested_present(
                (metadata, ("stage3_acceptance_basis",)),
                (metadata, ("stage3_posterior_state", "stage3_acceptance_basis")),
                (summary, ("stage3_protocol", "stage3_acceptance_basis")),
            ),
            "support_gate_status": _nested_present(
                (metadata, ("support_gate_status",)),
                (metadata, ("stage3_posterior_state", "support_gate_status")),
                (summary, ("stage3_protocol", "support_gate_status")),
            ),
            "support_gate_reject_reason": json.dumps(
                _nested_present(
                    (metadata, ("support_gate_reject_reason",)),
                    (metadata, ("stage3_posterior_state", "support_gate_reject_reason")),
                    (summary, ("stage3_protocol", "support_gate_reject_reason")),
                )
                or [],
                sort_keys=True,
            ),
            "actual_optimizer_steps": metadata.get("actual_optimizer_steps", metadata.get("optimizer_steps_run")),
            "optimizer_steps_run": metadata.get("optimizer_steps_run"),
            "anchor_alpha": metadata.get("anchor_alpha"),
            "target_parameter_l2_drift_pre_anchor_total": _drift_value(metadata, "pre_anchor", "total"),
            "target_parameter_l2_drift_post_anchor_total": _drift_value(metadata, "post_anchor", "total"),
            "target_parameter_l2_drift_pre_anchor_target_prompt": _drift_value(
                metadata, "pre_anchor", "target_prompt"
            ),
            "target_parameter_l2_drift_post_anchor_target_prompt": _drift_value(
                metadata, "post_anchor", "target_prompt"
            ),
            "target_parameter_l2_drift_monthly_gain": _drift_value(metadata, "post_anchor", "monthly_gain"),
            "target_labels_used_for_adaptation": metadata.get("target_labels_used_for_adaptation"),
            "target_support_labels_used_for_parameter_update": metadata.get(
                "target_support_labels_used_for_parameter_update"
            ),
            "target_support_labels_used_for_calibration": metadata.get(
                "target_support_labels_used_for_calibration"
            ),
            "few_shot_update_type": metadata.get("few_shot_update_type"),
            "n_samples_evaluated": summary.get("n_samples_evaluated"),
            "n_metric_rows": summary.get("n_metric_rows"),
            "prediction_content_hash": summary.get("prediction_content_hash"),
            "raw_adapted_prediction_content_hash": summary.get("raw_adapted_prediction_content_hash"),
            "post_gate_prediction_content_hash": summary.get("post_gate_prediction_content_hash"),
            "final_mixed_prediction_content_hash": summary.get("final_mixed_prediction_content_hash"),
            "raw_to_k0_mean_abs_delta": summary.get("raw_to_k0_mean_abs_delta"),
            "final_mix_to_k0_mean_abs_delta": summary.get("final_mix_to_k0_mean_abs_delta"),
            "context_tta_effective": _nested_present(
                (summary, ("context_tta_effective",)),
                (metadata, ("context_tta_effective",)),
                (summary, ("target_prompt", "context_tta_effective")),
                (metadata, ("target_context_prompt_state", "context_tta_effective")),
            ),
            "context_tta_source_stat_status": _nested_present(
                (summary, ("context_tta_source_stat_status",)),
                (metadata, ("context_tta_source_stat_status",)),
                (summary, ("target_prompt", "context_tta_source_stat_status")),
                (metadata, ("target_context_prompt_state", "context_tta_source_stat_status")),
            ),
            "prompt_l2_delta_mean": _nested_present(
                (summary, ("prompt_l2_delta_mean",)),
                (metadata, ("prompt_l2_delta_mean",)),
                (summary, ("target_prompt", "prompt_l2_delta_mean")),
                (metadata, ("target_context_prompt_state", "prompt_l2_delta_mean")),
            ),
            "prediction_delta_vs_no_tta": _nested_present(
                (summary, ("prediction_delta_vs_no_tta",)),
                (metadata, ("prediction_delta_vs_no_tta",)),
            ),
            "support_gain_selection_rule": _nested_present(
                (metadata, ("support_gain_calibration", "selection_rule")),
                (summary, ("target_train_residual_gain_calibration", "selection_rule")),
            ),
            "support_gain_best_alpha_raw": _nested_present(
                (metadata, ("support_gain_calibration", "best_alpha_raw")),
                (summary, ("target_train_residual_gain_calibration", "best_alpha_raw")),
            ),
            "support_gain_stable_candidate_alphas": json.dumps(
                _nested_present(
                    (metadata, ("support_gain_calibration", "stable_candidate_alphas")),
                    (summary, ("target_train_residual_gain_calibration", "stable_candidate_alphas")),
                )
                or [],
                sort_keys=True,
            ),
            "support_gain_selection_margin": _nested_present(
                (metadata, ("support_gain_calibration", "selection_margin")),
                (summary, ("target_train_residual_gain_calibration", "selection_margin")),
            ),
            "support_gain_stability_tolerance": _nested_present(
                (metadata, ("support_gain_calibration", "stability_tolerance")),
                (summary, ("target_train_residual_gain_calibration", "stability_tolerance")),
            ),
            "support_gain_paired_support_se_capped": _nested_present(
                (metadata, ("support_gain_calibration", "paired_support_se_capped")),
                (summary, ("target_train_residual_gain_calibration", "paired_support_se_capped")),
            ),
            "support_nesting_policy": _nested_present(
                (metadata, ("support_nesting_policy",)),
                (metadata, ("support_gain_calibration", "support_nesting_policy")),
                (summary, ("stage3_protocol", "support_nesting_policy")),
            ),
            "nested_support_dates_hash": _nested_present(
                (metadata, ("nested_support_dates_hash",)),
                (metadata, ("support_gain_calibration", "nested_support_dates_hash")),
                (summary, ("stage3_protocol", "nested_support_dates_hash")),
            ),
            "support_gain_target_eval_usage": _nested_present(
                (metadata, ("support_gain_calibration", "target_eval_usage")),
                (summary, ("target_train_residual_gain_calibration", "target_eval_usage")),
            ),
            "selected_support_candidate_id": _nested_present(
                (metadata, ("selected_support_candidate_id",)),
                (metadata, ("linearized_ridge_calibration", "selected_support_candidate_id")),
                (metadata, ("support_pool_selection", "selected_support_candidate_id")),
                (summary, ("stage3_protocol", "selected_support_candidate_id")),
            ),
            "support_selection_objective": _nested_present(
                (metadata, ("support_selection_objective",)),
                (metadata, ("linearized_ridge_calibration", "support_selection_objective")),
                (metadata, ("support_pool_selection", "support_selection_objective")),
                (summary, ("stage3_protocol", "support_selection_objective")),
            ),
            "support_cv_objective_delta": _nested_present(
                (metadata, ("support_cv_objective_delta",)),
                (metadata, ("support_gate", "support_cv_objective_delta")),
                (metadata, ("support_gain_calibration", "support_gain_cv_summary", "cv_objective_delta")),
                (metadata, ("linearized_ridge_calibration", "support_cv_summary", "cv_objective_delta")),
                (metadata, ("support_pool_selection", "selected_cv_summary", "cv_objective_delta")),
                (summary, ("stage3_protocol", "support_cv_objective_delta")),
            ),
            "support_cv_objective_delta_se": _nested_present(
                (metadata, ("support_cv_objective_delta_se",)),
                (metadata, ("support_gate", "support_cv_objective_delta_se")),
                (metadata, ("support_gain_calibration", "support_gain_cv_summary", "cv_objective_delta_se")),
                (summary, ("stage3_protocol", "support_cv_objective_delta_se")),
            ),
            "support_cv_objective_delta_t": _nested_present(
                (metadata, ("support_cv_objective_delta_t",)),
                (metadata, ("support_gate", "support_cv_objective_delta_t")),
                (metadata, ("support_gain_calibration", "support_gain_cv_summary", "cv_objective_delta_t")),
                (summary, ("stage3_protocol", "support_cv_objective_delta_t")),
            ),
            "support_cv_nested_k4_objective_delta": _nested_present(
                (metadata, ("support_cv_nested_k4_objective_delta",)),
                (metadata, ("support_gate", "support_cv_nested_k4_objective_delta")),
                (
                    metadata,
                    (
                        "support_gain_calibration",
                        "support_gain_cv_summary",
                        "cv_nested_k4_objective_delta",
                    ),
                ),
                (metadata, ("linearized_ridge_calibration", "support_cv_summary", "cv_nested_k4_objective_delta")),
                (metadata, ("support_pool_selection", "selected_cv_summary", "cv_nested_k4_objective_delta")),
                (summary, ("stage3_protocol", "support_cv_nested_k4_objective_delta")),
            ),
            "support_cv_added_objective_delta": _nested_present(
                (metadata, ("support_cv_added_objective_delta",)),
                (metadata, ("support_gate", "support_cv_added_objective_delta")),
                (metadata, ("support_gain_calibration", "support_gain_cv_summary", "cv_added_objective_delta")),
                (metadata, ("linearized_ridge_calibration", "support_cv_summary", "cv_added_objective_delta")),
                (metadata, ("support_pool_selection", "selected_cv_summary", "cv_added_objective_delta")),
                (summary, ("stage3_protocol", "support_cv_added_objective_delta")),
            ),
            "k12_vs_k4_cv_objective_delta": _nested_present(
                (metadata, ("k12_vs_k4_cv_objective_delta",)),
                (metadata, ("support_gain_calibration", "support_gain_gate", "k12_vs_k4_cv_objective_delta")),
                (metadata, ("support_pool_selection", "support_gain_gate", "k12_vs_k4_cv_objective_delta")),
                (summary, ("stage3_protocol", "k12_vs_k4_cv_objective_delta")),
            ),
            "k12_vs_k4_cv_rootzone_delta": _nested_present(
                (metadata, ("k12_vs_k4_cv_rootzone_delta",)),
                (metadata, ("support_gain_calibration", "support_gain_gate", "k12_vs_k4_cv_rootzone_delta")),
                (metadata, ("support_pool_selection", "support_gain_gate", "k12_vs_k4_cv_rootzone_delta")),
                (summary, ("stage3_protocol", "k12_vs_k4_cv_rootzone_delta")),
            ),
            "support_cycle_improvement_fraction": _nested_present(
                (metadata, ("support_cycle_improvement_fraction",)),
                (metadata, ("linearized_ridge_calibration", "support_cycle_improvement_fraction")),
                (summary, ("stage3_protocol", "support_cycle_improvement_fraction")),
            ),
            "support_gate_cycle_improvement_min_fraction": _nested_present(
                (metadata, ("support_gate_cycle_improvement_min_fraction",)),
                (metadata, ("resolved_mode_defaults", "support_gate_cycle_improvement_min_fraction")),
                (summary, ("stage3_protocol", "support_gate_cycle_improvement_min_fraction")),
            ),
            "k12_reference_policy": _nested_present(
                (metadata, ("k12_reference_policy",)),
                (metadata, ("linearized_ridge_calibration", "k12_reference_policy")),
                (summary, ("stage3_protocol", "k12_reference_policy")),
            ),
            "support_candidate_pool": json.dumps(
                _nested_present(
                    (metadata, ("support_candidate_pool",)),
                    (metadata, ("linearized_ridge_calibration", "support_candidate_pool")),
                    (metadata, ("support_pool_selection", "support_candidate_pool")),
                    (summary, ("stage3_protocol", "support_candidate_pool")),
                )
                or [],
                sort_keys=True,
            ),
            "ridge_lambda": _nested_present(
                (metadata, ("ridge_lambda",)),
                (metadata, ("linearized_ridge_calibration", "ridge_lambda")),
                (metadata, ("ridge_diagnostics", "ridge_lambda")),
                (metadata, ("resolved_mode_defaults", "ridge_lambda")),
            ),
            "ridge_weighting": _nested_present(
                (metadata, ("ridge_weighting",)),
                (metadata, ("ridge_diagnostics", "ridge_weighting")),
                (metadata, ("linearized_ridge_calibration", "ridge_weighting")),
                (metadata, ("resolved_mode_defaults", "ridge_weighting")),
            ),
            "ridge_effective_calibration_dof": _nested_present(
                (metadata, ("ridge_effective_calibration_dof",)),
                (metadata, ("ridge_diagnostics", "effective_calibration_dof")),
                (metadata, ("linearized_ridge_calibration", "effective_calibration_dof")),
            ),
            "linearized_ridge_calibration_mode": _nested_present(
                (metadata, ("linearized_ridge_calibration", "calibration_mode")),
                (metadata, ("ridge_diagnostics", "solver")),
            ),
            "linearized_ridge_target_eval_usage": _nested_present(
                (metadata, ("linearized_ridge_calibration", "target_eval_usage")),
                (metadata, ("ridge_diagnostics", "target_eval_usage")),
            ),
            "linearized_ridge_parameter_scope": _nested_present(
                (metadata, ("linearized_ridge_calibration", "parameter_scope")),
                (metadata, ("ridge_diagnostics", "parameter_scope")),
            ),
            "support_affine_surface_a": _support_affine_coeff(metadata, "surface", "a"),
            "support_affine_surface_b": _support_affine_coeff(metadata, "surface", "b"),
            "support_affine_rootzone_a": _support_affine_coeff(metadata, "rootzone", "a"),
            "support_affine_rootzone_b": _support_affine_coeff(metadata, "rootzone", "b"),
            "effective_calibration_dof": _nested_present(
                (metadata, ("support_affine_calibration", "effective_calibration_dof")),
                (metadata, ("linearized_ridge_calibration", "effective_calibration_dof")),
                (metadata, ("ridge_diagnostics", "effective_calibration_dof")),
                (summary, ("stage3_protocol", "support_affine_calibration", "effective_calibration_dof")),
            ),
            "support_calibration_dof": _nested_present(
                (metadata, ("support_calibration_dof",)),
                (metadata, ("support_gain_calibration", "support_calibration_dof")),
                (metadata, ("support_pool_selection", "support_calibration_dof")),
                (summary, ("stage3_protocol", "support_calibration_dof")),
            ),
            "support_affine_target_eval_usage": _nested_present(
                (metadata, ("support_affine_calibration", "target_eval_usage")),
                (summary, ("stage3_protocol", "support_affine_calibration", "target_eval_usage")),
            ),
            "k_specific_prediction_changed": False,
            "eval_time_s": summary.get("eval_time_s"),
            "summary": str(summary_path),
            "adapt_metadata": str(metadata_path),
            "metrics_long": str(metrics_long_path),
            "metrics_by_region": str(metrics_by_region_path),
            "metrics_by_season": str(metrics_by_season_path),
            "checkpoint": str(checkpoint_path),
            "adapted_checkpoint_sha256": _file_sha256(checkpoint_path),
            "seed": row_seed,
        }
        row.update(_metric_block(summary, "surface"))
        row.update(_metric_block(summary, "rootzone"))
        rows.append(row)

    previous_hash = ""
    for row in rows:
        current_hash = str(row.get("prediction_content_hash") or "")
        row["k_specific_prediction_changed"] = bool(previous_hash and current_hash and current_hash != previous_hash)
        if current_hash:
            previous_hash = current_hash

    base.mkdir(parents=True, exist_ok=True)
    with (base / "overview.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (base / "overview.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    md_headers = [
        "K",
        "adaptation_setting",
        "stage3_kshot_mode",
        "status",
        "paper_facing_run",
        "policy_source",
        "adaptation_step_policy_source",
        "stage3_posterior_decision",
        "support_gate_status",
        "support_gate_reject_reason",
        "actual_optimizer_steps",
        "target_parameter_l2_drift_pre_anchor_target_prompt",
        "target_parameter_l2_drift_post_anchor_target_prompt",
        "target_labels_used_for_adaptation",
        "raw_to_k0_mean_abs_delta",
        "final_mix_to_k0_mean_abs_delta",
        "context_tta_effective",
        "context_tta_source_stat_status",
        "prompt_l2_delta_mean",
        "prediction_delta_vs_no_tta",
        "support_gain_selection_rule",
        "support_gain_best_alpha_raw",
        "support_gain_stable_candidate_alphas",
        "support_gain_selection_margin",
        "support_gain_stability_tolerance",
        "support_gain_paired_support_se_capped",
        "support_nesting_policy",
        "nested_support_dates_hash",
        "selected_support_candidate_id",
        "support_selection_objective",
        "support_cv_objective_delta",
        "support_cv_nested_k4_objective_delta",
        "support_cv_added_objective_delta",
        "k12_vs_k4_cv_objective_delta",
        "k12_vs_k4_cv_rootzone_delta",
        "support_cycle_improvement_fraction",
        "k12_reference_policy",
        "ridge_lambda",
        "ridge_weighting",
        "ridge_effective_calibration_dof",
        "linearized_ridge_calibration_mode",
        "linearized_ridge_target_eval_usage",
        "linearized_ridge_parameter_scope",
        "support_affine_surface_a",
        "support_affine_surface_b",
        "support_affine_rootzone_a",
        "support_affine_rootzone_b",
        "effective_calibration_dof",
        "support_calibration_dof",
        "support_affine_target_eval_usage",
        "k_specific_prediction_changed",
        "surface_skill_primary",
        "rootzone_skill_primary",
    ]
    lines = [
        "# HyperDA Zero/Few-Shot Target Eval Overview",
        "",
        f"Target region: `{target_region}`",
        "",
        "|" + "|".join(md_headers) + "|",
        "|" + "|".join([":--"] * len(md_headers)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(_fmt(row.get(key)) for key in md_headers) + "|")
    (base / "overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Stage 3 K-shot overview from existing artifacts")
    parser.add_argument("--output_base", required=True)
    parser.add_argument("--target_region", required=True)
    parser.add_argument("--k_list", default="")
    parser.add_argument("--seed", default="0")
    args = parser.parse_args()
    rows = rebuild_overview(
        output_base=args.output_base,
        target_region=args.target_region,
        requested_k_values=args.k_list.split(),
        seed=args.seed,
    )
    print(f"Wrote {len(rows)} overview rows under {args.output_base}")


if __name__ == "__main__":
    main()
