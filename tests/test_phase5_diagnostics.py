from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch


def _metric_rows(method: str, values: dict[tuple[str, str, str], float]) -> pd.DataFrame:
    rows = []
    for (query_date, variable, metric), value in values.items():
        month = int(query_date[5:7])
        season = "DJF" if month in {12, 1, 2} else "JJA"
        rows.append(
            {
                "method": method,
                "query_date": query_date,
                "month": month,
                "season": season,
                "target_region_id": "US-R1",
                "variable": variable,
                "metric": metric,
                "value": value,
            }
        )
    return pd.DataFrame(rows)


def test_phase5_diagnostics_compare_runs_reports_groups_and_win_rates():
    from scripts.eval.diagnose_phase5_runs import compare_metric_frames

    baseline = _metric_rows(
        "hydro_msr",
        {
            ("2023-01-01", "surface", "increment_rmse_latw"): 0.20,
            ("2023-01-02", "surface", "increment_rmse_latw"): 0.30,
            ("2023-06-01", "surface", "increment_rmse_latw"): 0.40,
            ("2023-01-01", "rootzone", "increment_rmse_latw"): 0.10,
            ("2023-01-02", "rootzone", "increment_rmse_latw"): 0.12,
            ("2023-06-01", "rootzone", "increment_rmse_latw"): 0.14,
            ("2023-01-01", "surface", "increment_corr_latw"): 0.50,
            ("2023-01-02", "surface", "increment_corr_latw"): 0.60,
        },
    )
    candidate = _metric_rows(
        "hydro_msr_gain",
        {
            ("2023-01-01", "surface", "increment_rmse_latw"): 0.18,
            ("2023-01-02", "surface", "increment_rmse_latw"): 0.33,
            ("2023-06-01", "surface", "increment_rmse_latw"): 0.36,
            ("2023-01-01", "rootzone", "increment_rmse_latw"): 0.09,
            ("2023-01-02", "rootzone", "increment_rmse_latw"): 0.11,
            ("2023-06-01", "rootzone", "increment_rmse_latw"): 0.13,
            ("2023-01-01", "surface", "increment_corr_latw"): 0.55,
            ("2023-01-02", "surface", "increment_corr_latw"): 0.58,
        },
    )

    report = compare_metric_frames(baseline, candidate)

    assert report["overall"]["surface"]["increment_rmse_latw"]["candidate_mean"] == pytest.approx(0.29)
    assert report["overall"]["surface"]["increment_rmse_latw"]["relative_improvement"] == pytest.approx(0.0333333333)
    assert report["win_rate"]["surface"]["increment_rmse_latw"]["candidate_win_rate"] == pytest.approx(2 / 3)
    assert report["by_season"]["DJF"]["surface"]["increment_rmse_latw"]["candidate_mean"] == pytest.approx(0.255)
    assert report["by_month"]["1"]["surface"]["increment_rmse_latw"]["candidate_mean"] == pytest.approx(0.255)
    assert report["djf_surface_regression"]["baseline_mean"] == pytest.approx(0.25)
    assert report["djf_surface_regression"]["candidate_mean"] == pytest.approx(0.255)
    assert report["djf_surface_regression"]["regressed"] is True


def test_phase5_diagnostics_parameter_norms_group_target_modules(tmp_path: Path):
    from scripts.eval.diagnose_phase5_runs import checkpoint_parameter_norms

    ckpt = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": {
                "target_prompt.latent": torch.ones(3),
                "residual_gain.gain_delta": torch.ones(2) * 2,
                "target_spatial_refine.gain_mixer.net.2.weight": torch.ones(4),
                "head.weight": torch.ones(5),
            }
        },
        ckpt,
    )

    norms = checkpoint_parameter_norms(ckpt)

    assert norms["target_prompt"]["parameter_count"] == 3
    assert norms["residual_gain"]["l2_norm"] == pytest.approx((8.0) ** 0.5)
    assert norms["target_spatial_refine"]["parameter_count"] == 4
    assert norms["source_prior_or_other"]["parameter_count"] == 5
