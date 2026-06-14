from __future__ import annotations

import csv
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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_summary(path: Path, *, surface: float, rootzone: float, method: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "method": method,
                "target_region": "US-R1",
                "seed": 0,
                "surface": {"rmse_latw_mean": surface},
                "rootzone": {"rmse_latw_mean": rootzone},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _write_source_only_loro_summary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "method": "source_only_backbone",
                "target_region": "US-R1",
                "seed": 0,
                "surface": {"rmse_mean": 9.0},
                "rootzone": {"rmse_mean": 8.0},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(
        path.parent / "metrics_by_region.csv",
        [
            {
                "target_region_id": "US-R1",
                "variable": "surface",
                "metric": "analysis_rmse_latw",
                "mean": 0.123,
                "count": 10,
            },
            {
                "target_region_id": "US-R1",
                "variable": "surface",
                "metric": "increment_rmse_latw",
                "mean": 9.123,
                "count": 10,
            },
            {
                "target_region_id": "US-R1",
                "variable": "rootzone",
                "metric": "analysis_rmse_latw",
                "mean": 0.045,
                "count": 10,
            },
            {
                "target_region_id": "US-R1",
                "variable": "rootzone",
                "metric": "increment_rmse_latw",
                "mean": 9.045,
                "count": 10,
            },
            {
                "target_region_id": "US-R2",
                "variable": "surface",
                "metric": "analysis_rmse_latw",
                "mean": 99.0,
                "count": 10,
            },
        ],
    )
    return path


def _write_fixture_artifacts(tmp_path: Path) -> dict[str, Path]:
    runs_root = tmp_path / "runs"
    p2_suite = tmp_path / "p2_suite"
    locked_eval = tmp_path / "locked_eval"
    guard = tmp_path / "selected_guard_config.yaml"
    guard.write_text(
        json.dumps(
            {
                **FROZEN_RECIPE,
                "guard_config_hash": "ce5ae92f20ac9e33aab30e258a837802686e0786a8731178eeaaaeb554a461b2",
                "candidate_id": "schedule_K4_schedule_on_K12__loss_global_pixel__rho_rule_a__trust_none",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    recommendation = p2_suite / "selected_recipe_recommendation.yaml"
    recommendation.parent.mkdir(parents=True, exist_ok=True)
    recommendation.write_text(
        yaml.safe_dump(
            {
                "target_region": "US-R1",
                "seed": 0,
                "locked_eval_ready": True,
                "selected_guard_config_path": str(guard),
                "selected_guard_config_hash": "ce5ae92f20ac9e33aab30e258a837802686e0786a8731178eeaaaeb554a461b2",
                **FROZEN_RECIPE,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (p2_suite / "leakage_audit.json").write_text(json.dumps({"verdict": "pass"}), encoding="utf-8")
    _write_csv(
        p2_suite / "p2_suite_summary.csv",
        [
            {
                "phase": "P2.6",
                "run_id": "A0_k0_identity_base",
                "K": 0,
                "surface_rmse_latw": 0.40,
                "rootzone_rmse_latw": 0.20,
            },
            {
                "phase": "P2.6",
                "run_id": "A1_k4_all_original_schedule",
                "K": 4,
                "surface_rmse_latw": 0.30,
                "rootzone_rmse_latw": 0.10,
            },
            {
                "phase": "P2.6",
                "run_id": "A2_k12_all_original_schedule",
                "K": 12,
                "surface_rmse_latw": 0.35,
                "rootzone_rmse_latw": 0.15,
            },
        ],
    )

    locked_eval.mkdir(parents=True, exist_ok=True)
    _write_csv(
        locked_eval / "p2_8_locked_guard_target_eval_summary.csv",
        [
            {
                "p2_8_role": "selected_guarded",
                "selected_guard_config_hash": "ce5ae92f20ac9e33aab30e258a837802686e0786a8731178eeaaaeb554a461b2",
                "K": 12,
                "adapt_scope": "all",
                "schedule_label": "K4_schedule_on_K12",
                "lr": 0.001,
                "adaptation_steps": 100,
                "anchor_alpha": 0.75,
                "support_loss_reduction": "global_pixel",
                "trust_policy": "none",
                "rho_policy": "rule_a",
                "surface_rmse_latw": 0.25,
                "rootzone_rmse_latw": 0.05,
            }
        ],
    )
    (locked_eval / "selected_guard_env.json").write_text(
        json.dumps({"guard_config_hash": "ce5ae92f20ac9e33aab30e258a837802686e0786a8731178eeaaaeb554a461b2"}),
        encoding="utf-8",
    )

    _write_summary(
        runs_root / "phase3_forecast_only_all_regions" / "target_eval" / "US-R1" / "summary.json",
        surface=1.0,
        rootzone=2.0,
        method="forecast_only",
    )
    _write_summary(
        runs_root
        / "phase4_source_only"
        / "phase4_source_only_source_only_US-R1_w32_e50_lr0.0003_norm_zero_s0_20260529_095207"
        / "results"
        / "checkpoint_epoch_029"
        / "target_eval"
        / "US-R1"
        / "summary.json",
        surface=0.123,
        rootzone=0.045,
        method="source_only_backbone",
    )
    _write_source_only_loro_summary(
        runs_root
        / "phase4_source_only"
        / "phase4_source_only_source_only_US-R1_w32_e50_lr0.0003_norm_zero_s0_20260529_095207"
        / "results"
        / "checkpoint_epoch_029"
        / "target_eval"
        / "US-R1"
        / "summary.json",
    )
    _write_summary(
        runs_root
        / "phase4_source_only_region_specific"
        / "phase4_source_only_region_specific_source_only_US-R1_w32_e50_lr0.0003_norm_nozero_s0_20260606_191508"
        / "results"
        / "checkpoint_best_source_val_safe_score"
        / "target_eval"
        / "US-R1"
        / "summary.json",
        surface=0.9,
        rootzone=1.9,
        method="source_only_backbone",
    )
    return {
        "runs_root": runs_root,
        "p2_suite": p2_suite,
        "locked_eval": locked_eval,
        "guard": guard,
    }


def test_wrmse_table_fixed_order_and_na_for_missing_artifacts(tmp_path):
    from scripts.analysis.build_us_r1_wrmse_ablation_table import build_us_r1_wrmse_ablation_table

    paths = _write_fixture_artifacts(tmp_path)
    output_dir = tmp_path / "table"

    result = build_us_r1_wrmse_ablation_table(
        runs_root=paths["runs_root"],
        locked_eval_dir=paths["locked_eval"],
        p2_suite_report_dir=paths["p2_suite"],
        output_dir=output_dir,
    )

    rows = list(csv.DictReader((output_dir / "us_r1_wrmse_ablation.csv").open(encoding="utf-8", newline="")))
    assert [row["method_id"] for row in rows] == [
        "forecast_only",
        "source_only_backbone",
        "prompt_conditioned_shared_backbone",
        "hyperda_k0_zero_shot",
        "hyperda_k4_original",
        "hyperda_k12_original",
        "hyperda_sg_k12_selected_guard",
    ]
    prompt_row = rows[2]
    assert prompt_row["surface_rmse_latw"] == "NA"
    assert prompt_row["mean_surface_rootzone_rmse_latw"] == "NA"
    source_only = rows[1]
    assert source_only["surface_rmse_latw"] == "0.123"
    assert source_only["rootzone_rmse_latw"] == "0.045"
    assert source_only["mean_surface_rootzone_rmse_latw"] == "0.084"
    assert "phase4_source_only_region_specific" not in source_only["artifact"]
    selected = rows[-1]
    assert selected["surface_rmse_latw"] == "0.25"
    assert selected["rootzone_rmse_latw"] == "0.05"
    assert selected["mean_surface_rootzone_rmse_latw"] == "0.15"

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "seed=0 pilot" in summary
    assert "HyperDA++" not in summary
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    excluded = metadata["excluded_baseline_artifacts"]
    assert len(excluded) == 1
    assert "phase4_source_only_region_specific" in excluded[0]["path"]
    assert "region-oracle" in excluded[0]["reason"]
    assert result["row_count"] == 7


def test_wrmse_table_validates_selected_recipe_and_guard_config(tmp_path):
    from scripts.analysis.build_us_r1_wrmse_ablation_table import build_us_r1_wrmse_ablation_table

    paths = _write_fixture_artifacts(tmp_path)
    recommendation = paths["p2_suite"] / "selected_recipe_recommendation.yaml"
    payload = yaml.safe_load(recommendation.read_text(encoding="utf-8"))
    payload["anchor_alpha"] = 0.25
    recommendation.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="anchor_alpha"):
        build_us_r1_wrmse_ablation_table(
            runs_root=paths["runs_root"],
            locked_eval_dir=paths["locked_eval"],
            p2_suite_report_dir=paths["p2_suite"],
            output_dir=tmp_path / "table",
        )


def test_wrmse_table_uses_locked_selected_summary_json_when_csv_metrics_blank(tmp_path):
    from scripts.analysis.build_us_r1_wrmse_ablation_table import build_us_r1_wrmse_ablation_table

    paths = _write_fixture_artifacts(tmp_path)
    selected_summary = _write_summary(
        paths["locked_eval"] / "K12_selected_guarded_hash" / "K12" / "eval" / "US-R1" / "summary.json",
        surface=0.24,
        rootzone=0.06,
        method="hyperda_few_shot_k12",
    )
    _write_csv(
        paths["locked_eval"] / "p2_8_locked_guard_target_eval_summary.csv",
        [
            {
                "p2_8_role": "selected_guarded",
                "selected_guard_config_hash": "ce5ae92f20ac9e33aab30e258a837802686e0786a8731178eeaaaeb554a461b2",
                "K": 12,
                "adapt_scope": "all",
                "schedule_label": "K4_schedule_on_K12",
                "lr": 0.001,
                "adaptation_steps": 100,
                "anchor_alpha": 0.75,
                "support_loss_reduction": "global_pixel",
                "trust_policy": "none",
                "rho_policy": "rule_a",
                "summary": str(selected_summary),
                "surface_rmse_latw": "",
                "rootzone_rmse_latw": "",
            }
        ],
    )

    build_us_r1_wrmse_ablation_table(
        runs_root=paths["runs_root"],
        locked_eval_dir=paths["locked_eval"],
        p2_suite_report_dir=paths["p2_suite"],
        output_dir=tmp_path / "table",
    )

    rows = list(csv.DictReader((tmp_path / "table" / "us_r1_wrmse_ablation.csv").open(encoding="utf-8", newline="")))
    selected = rows[-1]
    assert selected["surface_rmse_latw"] == "0.24"
    assert selected["rootzone_rmse_latw"] == "0.06"
    metadata = json.loads((tmp_path / "table" / "metadata.json").read_text(encoding="utf-8"))
    assert "HyperDA++" not in json.dumps(metadata)
