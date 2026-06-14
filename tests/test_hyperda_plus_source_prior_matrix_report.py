from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.report import hyperda_plus_source_prior_matrix as report


def _write_candidate_run(
    root: Path,
    candidate_id: str,
    metrics: dict,
    *,
    timestamp: str = "20260613_010101",
) -> Path:
    run_dir = root / candidate_id / timestamp
    run_dir.mkdir(parents=True)
    (run_dir / "source_val_metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "checkpoint": "checkpoint_best_source_val_transfer_safe_score.pt",
            },
            indent=2,
        )
    )
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump({"candidate_id": candidate_id, "selection_metric": "source_val_transfer_safe_score"})
    )
    return run_dir


def test_report_selects_by_source_val_and_ignores_target_eval_files(tmp_path):
    runs_root = tmp_path / "runs"
    report_dir = tmp_path / "reports"
    _write_candidate_run(
        runs_root,
        "H0_current",
        {
            "source_val_transfer_safe_score": 0.4,
            "source_val_skill_rootzone": 0.1,
        },
    )
    h1 = _write_candidate_run(
        runs_root,
        "H1_capacity_safe",
        {
            "source_val_transfer_safe_score": 0.3,
            "source_val_skill_rootzone": 0.2,
        },
    )
    (h1 / "target_eval_metrics.json").write_text(
        json.dumps({"source_val_transfer_safe_score": 99.0, "target_eval_skill": 99.0})
    )

    result = report.build_report(
        runs_root=runs_root,
        report_dir=report_dir,
        candidate_order=["H0_current", "H1_capacity_safe"],
    )

    assert result["best_candidate_id"] == "H0_current"
    assert (report_dir / "summary.csv").exists()
    assert (report_dir / "summary.md").exists()
    best = yaml.safe_load((report_dir / "best_source_prior.yaml").read_text())
    assert best["candidate_id"] == "H0_current"
    assert "target_eval" not in (report_dir / "summary.md").read_text()


def test_missing_optional_tiebreakers_do_not_promote_capacity_candidate(tmp_path):
    runs_root = tmp_path / "runs"
    report_dir = tmp_path / "reports"
    _write_candidate_run(
        runs_root,
        "H0_current",
        {
            "source_val_transfer_safe_score": 0.5,
            "source_val_skill_rootzone": 0.25,
        },
    )
    _write_candidate_run(
        runs_root,
        "H1_capacity_safe",
        {
            "source_val_transfer_safe_score": 0.5,
            "source_val_skill_rootzone": 0.25,
            "source_pseudo_target_k12_non_degradation_rate": 1.0,
        },
    )

    result = report.build_report(
        runs_root=runs_root,
        report_dir=report_dir,
        candidate_order=["H0_current", "H1_capacity_safe"],
    )

    assert result["best_candidate_id"] == "H0_current"
    rows = {row["candidate_id"]: row for row in result["rows"]}
    assert rows["H0_current"]["source_pseudo_target_k12_non_degradation_rate"] == "not_available"
    assert rows["H0_current"]["source_pseudo_target_adaptation_drift"] == "not_available"
