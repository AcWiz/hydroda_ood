from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml


FROZEN_RECIPE = {
    "K": 12,
    "adapt_scope": "all",
    "schedule_label": "K4_schedule_on_K12",
    "lr": 0.001,
    "adaptation_steps": 100,
    "anchor_alpha": 0.75,
    "support_loss_reduction": "global_pixel",
    "trust_policy": "none",
    "rho_policy": "rule_a",
}


def _write_locked_overview(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "overview.csv").write_text(
        "K,adaptation_setting,surface_rmse_latw,rootzone_rmse_latw,support_loss_delta,adaptation_steps,lr,anchor_alpha\n"
        "12,few_shot_k12,0.12,0.34,-0.05,100,0.001,0.75\n",
        encoding="utf-8",
    )


def _write_locked_artifacts(tmp_path: Path, *, ready: bool = True, verdict: str = "pass") -> dict[str, Path]:
    source_checkpoint = tmp_path / "checkpoint.pt"
    source_checkpoint.write_bytes(b"source checkpoint")
    split_manifest = tmp_path / "splits.json"
    split_manifest.write_text('{"splits": []}\n', encoding="utf-8")
    source_sha = hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()
    split_sha = hashlib.sha256(split_manifest.read_bytes()).hexdigest()
    selected_guard = tmp_path / "selected_guard_config.yaml"
    selected_guard.write_text(
        json.dumps(
            {
                **FROZEN_RECIPE,
                "candidate_id": "schedule_K4_schedule_on_K12__loss_global_pixel__rho_rule_a__trust_none",
                "guard_config_hash": "ce5ae92f20ac9e33aab30e258a837802686e0786a8731178eeaaaeb554a461b2",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    recommendation = tmp_path / "selected_recipe_recommendation.yaml"
    recommendation.write_text(
        yaml.safe_dump(
            {
                "target_region": "US-R1",
                "seed": 0,
                "locked_eval_ready": ready,
                "selected_guard_config_path": str(selected_guard),
                "selected_guard_config_hash": "ce5ae92f20ac9e33aab30e258a837802686e0786a8731178eeaaaeb554a461b2",
                "source_checkpoint_path": str(source_checkpoint),
                "source_checkpoint_sha256": source_sha,
                "split_manifest_path": str(split_manifest),
                "split_manifest_sha256": split_sha,
                **FROZEN_RECIPE,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    leakage_audit = tmp_path / "leakage_audit.json"
    leakage_audit.write_text(json.dumps({"verdict": verdict}, indent=2), encoding="utf-8")
    calibration_summary = tmp_path / "final_source_safe_calibration_summary.json"
    calibration_summary.write_text(
        json.dumps({"selected_guard_config_path": str(selected_guard)}, indent=2),
        encoding="utf-8",
    )
    return {
        "source_checkpoint": source_checkpoint,
        "split_manifest": split_manifest,
        "selected_guard": selected_guard,
        "recommendation": recommendation,
        "leakage": leakage_audit,
        "calibration_summary": calibration_summary,
    }

def test_hyperda_run_review_flags_overfit_and_legacy_baseline(tmp_path):
    from scripts.analysis.review_hyperda_zero_few_shot_run import build_run_review

    overview = tmp_path / "overview.csv"
    with overview.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "K",
                "adaptation_setting",
                "surface_rmse_latw",
                "support_loss_delta",
                "adaptation_steps",
                "lr",
                "anchor_alpha",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "K": "0",
                "adaptation_setting": "zero_shot_context",
                "surface_rmse_latw": "1.0",
                "support_loss_delta": "",
                "adaptation_steps": "0",
                "lr": "0.001",
                "anchor_alpha": "0.0",
            }
        )
        writer.writerow(
            {
                "K": "12",
                "adaptation_setting": "few_shot_k12",
                "surface_rmse_latw": "1.2",
                "support_loss_delta": "-0.05",
                "adaptation_steps": "80",
                "lr": "0.0003",
                "anchor_alpha": "0.25",
            }
        )

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            [
                {
                    "method": "source_only_backbone",
                    "protocol_version": "v4.3",
                    "adaptation_setting": "target_full_train",
                }
            ]
        ),
        encoding="utf-8",
    )

    review = build_run_review(overview_path=overview, baseline_path=baseline)

    assert review["summary"]["status"] == "needs_audit"
    assert any("K=12" in warning and "support_loss_delta" in warning for warning in review["warnings"])
    assert any("target_full_train" in warning for warning in review["warnings"])
    assert review["k_rows"]["12"]["anchor_alpha"] == "0.25"
    assert review["baseline_protocol"]["same_protocol"] is False


def test_hyperda_run_review_cli_writes_json_and_markdown(tmp_path):
    from scripts.analysis.review_hyperda_zero_few_shot_run import main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "overview.csv").write_text(
        "K,adaptation_setting,surface_rmse_latw,support_loss_delta,adaptation_steps,lr,anchor_alpha\n"
        "0,zero_shot_context,1.0,,0,0.001,0.0\n",
        encoding="utf-8",
    )

    out_json = tmp_path / "review.json"
    out_md = tmp_path / "review.md"
    main(
        [
            "--run_dir",
            str(run_dir),
            "--output_json",
            str(out_json),
            "--output_md",
            str(out_md),
        ]
    )

    assert json.loads(out_json.read_text(encoding="utf-8"))["summary"]["overview_path"].endswith("overview.csv")
    md = out_md.read_text(encoding="utf-8")
    assert "# HyperDA Zero/Few-Shot Run Review" in md
    assert "source-only" in md


def test_locked_eval_preflight_blocks_if_not_ready_or_leakage_not_pass(tmp_path):
    from scripts.analysis.review_hyperda_zero_few_shot_run import locked_eval_preflight

    paths = _write_locked_artifacts(tmp_path, ready=False, verdict="pass")
    with pytest.raises(ValueError, match="locked_eval_ready"):
        locked_eval_preflight(
            recommendation_path=paths["recommendation"],
            leakage_audit_path=paths["leakage"],
            selected_guard_config_path=paths["selected_guard"],
            calibration_summary_path=paths["calibration_summary"],
            source_checkpoint_path=paths["source_checkpoint"],
            split_manifest_path=paths["split_manifest"],
        )

    paths = _write_locked_artifacts(tmp_path, ready=True, verdict="leakage_fail")
    with pytest.raises(ValueError, match="leakage"):
        locked_eval_preflight(
            recommendation_path=paths["recommendation"],
            leakage_audit_path=paths["leakage"],
            selected_guard_config_path=paths["selected_guard"],
            calibration_summary_path=paths["calibration_summary"],
            source_checkpoint_path=paths["source_checkpoint"],
            split_manifest_path=paths["split_manifest"],
        )


def test_locked_eval_output_dir_writes_summary_and_metadata_audit(tmp_path):
    from scripts.analysis.review_hyperda_zero_few_shot_run import main

    run_dir = tmp_path / "locked_run"
    _write_locked_overview(run_dir)
    paths = _write_locked_artifacts(tmp_path)
    output_dir = tmp_path / "review"

    main(
        [
            "--run_dir",
            str(run_dir),
            "--output_dir",
            str(output_dir),
            "--recommendation",
            str(paths["recommendation"]),
            "--leakage_audit",
            str(paths["leakage"]),
            "--selected_guard_config",
            str(paths["selected_guard"]),
            "--calibration_summary",
            str(paths["calibration_summary"]),
            "--source_checkpoint",
            str(paths["source_checkpoint"]),
            "--split_manifest",
            str(paths["split_manifest"]),
            "--target_region",
            "US-R1",
            "--seed",
            "0",
        ]
    )

    assert (output_dir / "locked_eval_summary.md").exists()
    summary = json.loads((output_dir / "locked_eval_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((output_dir / "locked_eval_metadata_audit.json").read_text(encoding="utf-8"))
    assert summary["summary"]["overview_path"].endswith("overview.csv")
    assert audit["target_eval_not_used_for_selection"] is True
    assert audit["target_region"] == "US-R1"
    assert audit["seed"] == 0
    assert audit["source_checkpoint_sha256"] == audit["source_checkpoint"]["sha256"]
    assert audit["source_checkpoint_path"] == str(paths["source_checkpoint"])
    assert audit["selected_guard_config_path"] == str(paths["selected_guard"])
    assert audit["selected_guard_config_hash"] == "ce5ae92f20ac9e33aab30e258a837802686e0786a8731178eeaaaeb554a461b2"
    assert audit["split_manifest_sha256"] == audit["split_manifest"]["sha256"]
    assert "target_support_dates" in audit["support_dates"]
    assert audit["source_checkpoint"]["sha256"]
    assert audit["selected_guard_config"]["guard_config_hash"] == "ce5ae92f20ac9e33aab30e258a837802686e0786a8731178eeaaaeb554a461b2"
    assert audit["selected_guard_config"]["recipe"]["schedule_label"] == "K4_schedule_on_K12"
    assert audit["split_manifest"]["sha256"]


def test_locked_eval_review_preserves_duplicate_k12_rows_and_fills_summary_metrics(tmp_path):
    from scripts.analysis.review_hyperda_zero_few_shot_run import build_run_review

    run_dir = tmp_path / "locked_run"
    selected_summary = run_dir / "K12_selected_guarded" / "K12" / "eval" / "US-R1" / "summary.json"
    selected_summary.parent.mkdir(parents=True, exist_ok=True)
    selected_summary.write_text(
        json.dumps(
            {
                "surface": {"rmse_latw_mean": 0.24},
                "rootzone": {"rmse_latw_mean": 0.06},
            }
        ),
        encoding="utf-8",
    )
    overview = run_dir / "p2_8_locked_guard_target_eval_summary.csv"
    overview.parent.mkdir(parents=True, exist_ok=True)
    with overview.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "K",
                "p2_8_role",
                "adaptation_setting",
                "surface_rmse_latw",
                "rootzone_rmse_latw",
                "summary",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "K": "12",
                "p2_8_role": "baseline",
                "adaptation_setting": "few_shot_k12",
                "surface_rmse_latw": "0.35",
                "rootzone_rmse_latw": "0.15",
                "summary": "",
            }
        )
        writer.writerow(
            {
                "K": "12",
                "p2_8_role": "selected_guarded",
                "adaptation_setting": "few_shot_k12",
                "surface_rmse_latw": "",
                "rootzone_rmse_latw": "",
                "summary": str(selected_summary),
            }
        )

    review = build_run_review(overview)

    assert review["k_rows"]["K12_baseline"]["surface_rmse_latw"] == "0.35"
    selected = review["k_rows"]["K12_selected_guarded"]
    assert selected["surface_rmse_latw"] == "0.24"
    assert selected["rootzone_rmse_latw"] == "0.06"
    assert selected["status"] == "ok"
