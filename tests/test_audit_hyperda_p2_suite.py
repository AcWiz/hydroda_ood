from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_yaml(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _identity_audit(
    run_dir: Path,
    *,
    status: str = "passed",
    source_sha_match: bool = True,
    split_sha_match: bool = True,
    prediction_hash_match: bool = True,
    optimizer_steps: int = 0,
    k12_labels_loaded: bool = False,
    max_drift: float = 0.0,
) -> Path:
    return _write_json(
        run_dir / "identity_audit.json",
        {
            "status": status,
            "max_abs_metric_diff": 0.0,
            "tolerance": 1e-8,
            "hash_matches": {
                "source_checkpoint_sha256": {
                    "K0": "source-sha",
                    "K12": "source-sha" if source_sha_match else "different",
                    "match": source_sha_match,
                },
                "split_manifest_sha256": {
                    "K0": "split-sha",
                    "K12": "split-sha" if split_sha_match else "different",
                    "match": split_sha_match,
                },
                "target_context_dates_hash": {"K0": "context-hash", "K12": "context-hash", "match": True},
                "target_eval_dates_hash": {"K0": "eval-hash", "K12": "eval-hash", "match": True},
                "prediction_content_hash": {
                    "K0": "pred-hash",
                    "K12": "pred-hash" if prediction_hash_match else "different",
                    "match": prediction_hash_match,
                },
                "metric_values_content_hash": {"K0": "metric-hash", "K12": "metric-hash", "match": True},
            },
            "k12_identity_checks": {
                "target_labels_loaded_for_adaptation": k12_labels_loaded,
                "target_labels_used_for_adaptation": False,
                "actual_optimizer_steps": optimizer_steps,
                "max_target_parameter_l2_drift": max_drift,
            },
            "K0_metadata": str(run_dir / "K0" / "adapt" / "metadata.json"),
            "K12_metadata": str(run_dir / "K12" / "adapt" / "metadata.json"),
            "K0_summary": str(run_dir / "K0" / "eval" / "summary.json"),
            "K12_summary": str(run_dir / "K12" / "eval" / "summary.json"),
            "failures": [] if status == "passed" else ["synthetic failure"],
        },
    )


def _basic_summary_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "target_region": "US-R1",
        "seed": 0,
        "K": 12,
        "adapt_scope": "all",
        "schedule_label": "original_K12",
        "lr": 0.0003,
        "adaptation_steps": 80,
        "anchor_alpha": 0.25,
        "support_loss_reduction": "global_pixel",
        "trust_region_mode": "none",
        "trust_total_radius": 3.8,
        "trust_prompt_radius": 3.6,
        "trust_gain_radius": 0.33,
        "trust_coeff_radius": 0.68,
        "adapt_mix_rho": 1.0,
        "rho_policy": "fixed_1.0",
        "surface_skill_primary": 0.18,
        "rootzone_skill_primary": 0.22,
        "surface_rmse_latw": 0.003,
        "rootzone_rmse_latw": 0.0003,
        "surface_corr_latw": 0.4,
        "rootzone_corr_latw": 0.5,
        "target_parameter_l2_drift_post_anchor_target_prompt": 0.3,
        "support_gradient_negative_fraction": 0.1,
        "support_gradient_cosine_min": 0.0,
        "source_checkpoint_sha256": "source-sha",
        "split_manifest_sha256": "split-sha",
        "target_context_dates_hash": "context-hash",
        "target_eval_dates_hash": "eval-hash",
        "prediction_content_hash": "pred-hash",
        "metric_values_content_hash": "metric-hash",
        "checkpoint": "checkpoint.pt",
    }
    row.update(overrides)
    return row


def _p2_8_complete(
    root: Path,
    *,
    run_name: str = "US-R1_s0_20260613T010000Z",
    split_type: str = "source_val",
    query_role: str = "source_val_pseudo_query",
    include_top5: bool = True,
    include_loso: bool = True,
) -> Path:
    run_dir = root / "phase5_hyperda_p2_8_source_safe_guard_calibration" / run_name
    row = _basic_summary_row(
        candidate_id="C0_fixed_1_0",
        query_role=query_role,
        split_type=split_type,
        adaptation_setting="few_shot_k12",
    )
    summary = {
        "schema_version": "p2_8_source_safe_guard_calibration_v1",
        "selection_source": "source_val_pseudo_query_only",
        "final_target_region": "US-R1",
        "seed": 0,
        "status": "ok",
        "top_candidate": row,
        "ranked_candidates": [row],
        "calibration_audit": {
            "row_count": 3,
            "source_pseudo_target_episode_count": 2,
            "source_regions": ["US-R2", "US-R3"],
            "row_file_hashes": {"rows.csv": "row-sha"},
        },
    }
    _write_json(run_dir / "source_safe_calibration_summary.json", summary)
    _write_yaml(
        run_dir / "selected_guard_config.yaml",
        {
            "schema_version": "p2_8_source_safe_guard_config_v1",
            "candidate_id": "C0_fixed_1_0",
            "K": 12,
            "adapt_scope": "all",
            "schedule_label": "original_K12",
            "lr": 0.0003,
            "adaptation_steps": 80,
            "anchor_alpha": 0.25,
            "support_loss_reduction": "global_pixel",
            "trust_policy": "none",
            "rho_policy": "fixed_1.0",
            "adapt_mix_rho": 1.0,
            "selection_source": "source_val_pseudo_query_only",
        },
    )
    if include_top5:
        _write_csv(run_dir / "top5_stability.csv", [row])
    if include_loso:
        _write_csv(run_dir / "leave_one_source_region_out_stability.csv", [row])
    return run_dir


def _run_audit(tmp_path: Path, runs_root: Path):
    from scripts.analysis.audit_hyperda_p2_suite import audit_hyperda_p2_suite

    output_dir = tmp_path / "reports"
    result = audit_hyperda_p2_suite(
        target_region="US-R1",
        seed=0,
        runs_root=runs_root,
        output_dir=output_dir,
    )
    return result, output_dir


def test_p0_fail_blocks_locked_eval(tmp_path):
    runs_root = tmp_path / "runs"
    _identity_audit(
        runs_root / "phase5_hyperda_identity_gate" / "US-R1_s0_identity",
        status="failed",
        optimizer_steps=1,
        k12_labels_loaded=True,
    )
    _p2_8_complete(runs_root)

    _, output_dir = _run_audit(tmp_path, runs_root)

    recommendation = yaml.safe_load((output_dir / "selected_recipe_recommendation.yaml").read_text(encoding="utf-8"))
    summary_md = (output_dir / "p2_suite_summary.md").read_text(encoding="utf-8")
    assert recommendation["locked_eval_ready"] is False
    assert recommendation["final_conclusion"] == "pipeline failed"
    assert "fix pipeline before target_eval" in recommendation["blocked_reasons"]
    assert "Final conclusion A: `pipeline failed`" in summary_md


def test_target_eval_calibration_row_triggers_leakage_fail(tmp_path):
    runs_root = tmp_path / "runs"
    _identity_audit(runs_root / "phase5_hyperda_identity_gate" / "US-R1_s0_identity")
    _p2_8_complete(runs_root, split_type="target_eval", query_role="target_eval")

    _, output_dir = _run_audit(tmp_path, runs_root)

    leakage = json.loads((output_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    recommendation = yaml.safe_load((output_dir / "selected_recipe_recommendation.yaml").read_text(encoding="utf-8"))
    assert leakage["verdict"] == "leakage_fail"
    assert leakage["checks"]["target_eval_metrics_not_used"]["status"] == "fail"
    assert recommendation["locked_eval_ready"] is False
    assert any("leakage_fail" in reason for reason in recommendation["blocked_reasons"])


def test_missing_p2_8_stability_triggers_evidence_incomplete(tmp_path):
    runs_root = tmp_path / "runs"
    _identity_audit(runs_root / "phase5_hyperda_identity_gate" / "US-R1_s0_identity")
    _p2_8_complete(runs_root, include_top5=False, include_loso=False)

    _, output_dir = _run_audit(tmp_path, runs_root)

    leakage = json.loads((output_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    missing_md = (output_dir / "missing_or_invalid_artifacts.md").read_text(encoding="utf-8")
    recommendation = yaml.safe_load((output_dir / "selected_recipe_recommendation.yaml").read_text(encoding="utf-8"))
    assert leakage["evidence_status"] == "evidence_incomplete"
    assert recommendation["locked_eval_ready"] is False
    assert "top5_stability" in missing_md
    assert "leave-one-source-region-out" in missing_md


def test_selected_yaml_contains_required_keys_and_forbidden_input_checklist(tmp_path):
    runs_root = tmp_path / "runs"
    _identity_audit(runs_root / "phase5_hyperda_identity_gate" / "US-R1_s0_identity")
    selected_dir = _p2_8_complete(runs_root)

    _, output_dir = _run_audit(tmp_path, runs_root)

    recommendation = yaml.safe_load((output_dir / "selected_recipe_recommendation.yaml").read_text(encoding="utf-8"))
    required_keys = {
        "target_region",
        "seed",
        "locked_eval_ready",
        "selection_source",
        "selected_guard_config_path",
        "candidate_id",
        "K",
        "adapt_scope",
        "schedule_label",
        "lr",
        "adaptation_steps",
        "anchor_alpha",
        "support_loss_reduction",
        "trust_policy",
        "rho_policy",
        "adapt_mix_rho",
        "source_checkpoint_sha256",
        "split_manifest_sha256",
        "forbidden_input_checklist",
    }
    assert required_keys <= set(recommendation)
    assert recommendation["locked_eval_ready"] is True
    assert recommendation["selection_source"] == "source_val_pseudo_query_only"
    assert recommendation["selected_guard_config_path"] == str(selected_dir / "selected_guard_config.yaml")
    checklist = recommendation["forbidden_input_checklist"]
    assert set(checklist) == {
        "target_eval_metrics_not_used",
        "target_eval_labels_not_used",
        "target_eval_prediction_records_not_used",
        "target_val_not_used",
        "target_full_train_not_used",
    }
    assert all(item["status"] == "pass" for item in checklist.values())


def test_p2_8_nested_source_metadata_hashes_are_reported(tmp_path):
    runs_root = tmp_path / "runs"
    _identity_audit(runs_root / "phase5_hyperda_identity_gate" / "US-R1_s0_identity")
    _p2_8_complete(runs_root)
    run_dir = runs_root / "phase5_hyperda_p2_8_source_safe_guard_calibration" / "US-R1_s0_20260613T010000Z"

    summary = json.loads((run_dir / "source_safe_calibration_summary.json").read_text(encoding="utf-8"))
    top = summary.pop("top_candidate")
    top.pop("source_checkpoint_sha256", None)
    top.pop("split_manifest_sha256", None)
    top.pop("target_context_dates_hash", None)
    top.pop("target_eval_dates_hash", None)
    selected_config = yaml.safe_load((run_dir / "selected_guard_config.yaml").read_text(encoding="utf-8"))
    selected_config["source_metadata"] = {
        "checkpoint_hashes": ["nested-source-sha"],
        "split_manifest_hashes": ["nested-split-sha"],
    }
    selected_config["target_eval_dates_hash"] = "not_used_source_val_query"
    (run_dir / "source_safe_calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "selected_guard_config.yaml").write_text(yaml.safe_dump(selected_config, sort_keys=False), encoding="utf-8")

    _, output_dir = _run_audit(tmp_path, runs_root)

    recommendation = yaml.safe_load((output_dir / "selected_recipe_recommendation.yaml").read_text(encoding="utf-8"))
    missing_md = (output_dir / "missing_or_invalid_artifacts.md").read_text(encoding="utf-8")
    assert recommendation["source_checkpoint_sha256"] == "nested-source-sha"
    assert recommendation["split_manifest_sha256"] == "nested-split-sha"
    assert "P2.8" not in missing_md


def test_p2_8_selected_guard_config_from_summary_is_normalized(tmp_path):
    runs_root = tmp_path / "runs"
    _identity_audit(runs_root / "phase5_hyperda_identity_gate" / "US-R1_s0_identity")
    _p2_8_complete(runs_root)
    run_dir = runs_root / "phase5_hyperda_p2_8_source_safe_guard_calibration" / "US-R1_s0_20260613T010000Z"

    summary = json.loads((run_dir / "source_safe_calibration_summary.json").read_text(encoding="utf-8"))
    selected = summary.pop("top_candidate")
    selected["rho_policy"] = "rule_a"
    selected.pop("source_checkpoint_sha256", None)
    selected.pop("split_manifest_sha256", None)
    selected.pop("target_context_dates_hash", None)
    selected.pop("target_eval_dates_hash", None)
    selected["source_metadata"] = {
        "checkpoint_hashes": ["nested-source-sha"],
        "split_manifest_hashes": ["nested-split-sha"],
    }
    summary["selected_guard_config"] = selected
    (run_dir / "source_safe_calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _, output_dir = _run_audit(tmp_path, runs_root)

    rows = list(csv.DictReader((output_dir / "p2_suite_summary.csv").open(encoding="utf-8", newline="")))
    p2_8 = next(row for row in rows if row["phase"] == "P2.8")
    assert p2_8["rho_policy"] == "rule_a"
    assert p2_8["source_checkpoint_sha256"] == "nested-source-sha"
    assert p2_8["split_manifest_sha256"] == "nested-split-sha"
    assert p2_8["missing_fields"] == ""


def test_discovery_chooses_latest_matching_run_for_region_and_seed(tmp_path):
    runs_root = tmp_path / "runs"
    _identity_audit(
        runs_root / "phase5_hyperda_identity_gate" / "US-R1_s0_20260101T000000Z",
        source_sha_match=False,
    )
    _identity_audit(runs_root / "phase5_hyperda_identity_gate" / "US-R1_s0_20260613T010000Z")
    _identity_audit(
        runs_root / "phase5_hyperda_identity_gate" / "US-R2_s0_20260613T020000Z",
        status="failed",
    )
    _p2_8_complete(runs_root)

    _, output_dir = _run_audit(tmp_path, runs_root)

    rows = list(csv.DictReader((output_dir / "p2_suite_summary.csv").open(encoding="utf-8", newline="")))
    p0_rows = [row for row in rows if row["phase"] == "P0"]
    assert len(p0_rows) == 1
    assert p0_rows[0]["run_id"] == "US-R1_s0_20260613T010000Z"
    assert p0_rows[0]["source_checkpoint_sha256"] == "source-sha"
