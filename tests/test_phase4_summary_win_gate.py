from __future__ import annotations

import json

import pandas as pd

from scripts.eval import rebuild_phase4_summary_table as phase4_summary
from scripts.eval.rebuild_phase4_summary_table import all_region_wrmse_win_gate


def _summary(surface: float, rootzone: float) -> dict:
    return {
        "target_eval": {
            "surface": {"analysis_rmse_latw_mean": surface},
            "rootzone": {"analysis_rmse_latw_mean": rootzone},
        }
    }


def test_all_region_wrmse_win_gate_requires_every_region_and_variable_to_win():
    regions = ["US-R1", "US-R2"]
    candidate = {
        "US-R1": _summary(0.10, 0.20),
        "US-R2": _summary(0.11, 0.21),
    }
    all_regions = {
        "US-R1": _summary(0.12, 0.22),
        "US-R2": _summary(0.13, 0.23),
    }
    region_specific = {
        "US-R1": _summary(0.15, 0.25),
        "US-R2": _summary(0.16, 0.26),
    }

    result = all_region_wrmse_win_gate(candidate, all_regions, region_specific, regions=regions)

    assert result["pass"] is True
    assert result["n_checks"] == 4
    assert result["failures"] == []


def test_all_region_wrmse_win_gate_fails_on_single_missing_or_losing_cell():
    regions = ["US-R1", "US-R2"]
    candidate = {
        "US-R1": _summary(0.10, 0.20),
        "US-R2": _summary(0.11, 0.24),
    }
    all_regions = {
        "US-R1": _summary(0.12, 0.22),
        "US-R2": _summary(0.13, 0.23),
    }
    region_specific = {
        "US-R1": _summary(0.15, 0.25),
        "US-R2": _summary(0.16, 0.26),
    }

    result = all_region_wrmse_win_gate(candidate, all_regions, region_specific, regions=regions)

    assert result["pass"] is False
    assert result["n_checks"] == 4
    assert result["failures"] == [
        {
            "region": "US-R2",
            "variable": "rootzone",
            "candidate_wrmse": 0.24,
            "all_regions_wrmse": 0.23,
            "region_specific_wrmse": 0.26,
        }
    ]


def _metric_row(region: str, variable: str, metric: str, value: float, query_date: str = "2023-01-01") -> dict:
    return {
        "query_date": query_date,
        "sample_region_id": region,
        "target_region_id": region if query_date != "global" else "",
        "active_region_ids": region if query_date != "global" else "",
        "variable": variable,
        "metric": metric,
        "value": value,
        "n_valid_pixels": 10,
    }


def test_all_regions_collection_prefers_metrics_long_global_skill(tmp_path, monkeypatch):
    run_dir = tmp_path / "phase4_source_only_all_regions_source_only_US-ALL_demo"
    target_eval_dir = run_dir / "results" / "checkpoint_best_source_val_safe_score" / "target_eval"
    target_eval_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "checkpoint_best_source_val_safe_score.pt").write_text("stub")

    rows = []
    for region in ["US-R1", "US-R2"]:
        for variable, rmse, skill in [
            ("surface", 0.10 if region == "US-R1" else 0.20, 0.70 if region == "US-R1" else 0.60),
            ("rootzone", 0.30 if region == "US-R1" else 0.40, 0.50 if region == "US-R1" else 0.40),
        ]:
            rows.append(_metric_row(region, variable, "analysis_rmse_latw", rmse))
            rows.append(_metric_row(region, variable, "increment_rmse_latw", rmse + 1.0))
            rows.append(_metric_row(region, variable, "increment_corr_latw", 0.25))
            rows.append(_metric_row(region, variable, "analysis_skill_vs_forecast_latw", -999.0))
            rows.append(
                _metric_row(
                    region,
                    variable,
                    "analysis_skill_vs_forecast_latw_global",
                    skill,
                    query_date="global",
                )
            )
    pd.DataFrame(rows).to_csv(target_eval_dir / "metrics_long.csv", index=False)

    # This stale file mirrors the historical bug: per-sample skill means are
    # available but must not be used for paper-facing skill summaries.
    stale_per_region = {
        "US-R1": {
            "surface": {"analysis_skill_vs_forecast_latw": {"mean": -999.0, "std": 0.0, "n": 1}},
            "rootzone": {"analysis_skill_vs_forecast_latw": {"mean": -999.0, "std": 0.0, "n": 1}},
        },
        "US-R2": {
            "surface": {"analysis_skill_vs_forecast_latw": {"mean": -999.0, "std": 0.0, "n": 1}},
            "rootzone": {"analysis_skill_vs_forecast_latw": {"mean": -999.0, "std": 0.0, "n": 1}},
        },
    }
    (target_eval_dir / "per_region_summary.json").write_text(json.dumps(stale_per_region))

    monkeypatch.setattr(phase4_summary, "REGIONS", ["US-R1", "US-R2"])
    monkeypatch.setattr(phase4_summary, "ALL_REGIONS_BASE", tmp_path)

    results = phase4_summary.collect_all_regions_results()

    assert results["US-R1"]["target_eval"]["surface"]["analysis_rmse_latw_mean"] == 0.10
    assert results["US-R1"]["target_eval"]["surface"]["analysis_skill_vs_forecast_latw_global"] == 0.70
    assert results["US-R2"]["target_eval"]["rootzone"]["analysis_rmse_latw_mean"] == 0.40
    assert results["US-R2"]["target_eval"]["rootzone"]["analysis_skill_vs_forecast_latw_global"] == 0.40
    assert results["US-R1"]["source_test"]["surface"]["analysis_skill_vs_forecast_latw_global"] == 0.60


def test_combined_summary_payload_names_paper_facing_baselines(monkeypatch):
    monkeypatch.setattr(phase4_summary, "REGIONS", ["US-R1"])
    monkeypatch.setattr(phase4_summary, "SPLIT_TYPES", ["target_eval", "source_test"])

    rs_results = {"US-R1": {"target_eval": {"surface": {"analysis_rmse_latw_mean": 0.1}, "rootzone": {}}}}
    ar_results = {"US-R1": {"target_eval": {"surface": {"analysis_rmse_latw_mean": 0.2}, "rootzone": {}}}}
    fo_results = {"US-R1": {"target_eval": {"surface": {"analysis_rmse_latw_mean": 0.3}, "rootzone": {}}}}

    payload = phase4_summary.build_combined_summary_payload(
        rs_results=rs_results,
        ar_results=ar_results,
        fo_results=fo_results,
    )

    assert payload["baselines"]["region_specific"]["paper_name"] == "RS-Scratch"
    assert payload["baselines"]["all_regions"]["paper_name"] == "Pooled Global"
    assert payload["baselines"]["forecast_only"]["paper_name"] == "Forecast-Only"
    assert payload["forecast_only"]["US-R1"]["target_eval"]["surface"]["analysis_rmse_latw_mean"] == 0.3
