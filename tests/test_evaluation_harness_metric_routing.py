import numpy as np
import pandas as pd

from hydroda.baselines.forecast import ForecastBaseline
from hydroda.evaluation.harness import evaluate_split, summarize_metric_rows


class TinyDataset:
    def __init__(self):
        self.sample = {
            "forecast_surface": np.zeros((2, 2), dtype=np.float32),
            "forecast_rootzone": np.zeros((2, 2), dtype=np.float32),
            "analysis_surface": np.ones((2, 2), dtype=np.float32),
            "analysis_rootzone": np.ones((2, 2), dtype=np.float32) * 2,
            "increment_surface": np.ones((2, 2), dtype=np.float32),
            "increment_rootzone": np.ones((2, 2), dtype=np.float32) * 2,
            "metric_mask": np.ones((2, 2), dtype=np.float32),
            "latitude_weight": np.ones((2, 2), dtype=np.float32),
            "date_str": "2022-01-01",
            "month": 1,
            "season": "DJF",
            "time_index": 0,
            "country_id": "US",
            "target_region_id": "US-R1",
            "active_region_ids": ["US-R1"],
            "K": 0,
            "seed": 0,
        }

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sample

    def preload(self):
        return {0: self.sample}


def test_forecast_only_increment_metrics_use_increment_not_analysis():
    rows = evaluate_split(
        TinyDataset(),
        ForecastBaseline(),
        split_role="target_query",
        experiment_id="tiny",
        protocol_freeze_id="test",
        method="forecast_only",
    )
    lookup = {(r["variable"], r["metric"]): r["value"] for r in rows}
    assert np.isclose(lookup[("surface", "analysis_skill_vs_forecast")], 0.0)
    assert np.isclose(lookup[("rootzone", "analysis_skill_vs_forecast")], 0.0)
    assert np.isclose(lookup[("surface", "increment_rmse")], 1.0)
    assert np.isclose(lookup[("rootzone", "increment_rmse")], 2.0)


def test_summary_prefers_global_skill_and_keeps_per_sample_diagnostics():
    rows = [
        {"variable": "surface", "metric": "analysis_skill_vs_forecast", "value": -100.0, "query_date": "2023-01-01"},
        {"variable": "surface", "metric": "analysis_skill_vs_forecast", "value": 0.5, "query_date": "2023-01-02"},
        {"variable": "surface", "metric": "analysis_skill_vs_forecast_global", "value": 0.25, "query_date": "global"},
        {"variable": "surface", "metric": "analysis_skill_vs_forecast_latw_global", "value": 0.2, "query_date": "global"},
        {"variable": "surface", "metric": "increment_rmse", "value": 1.0, "query_date": "2023-01-01"},
        {"variable": "surface", "metric": "increment_corr", "value": 0.8, "query_date": "2023-01-01"},
    ]
    summary = summarize_metric_rows(pd.DataFrame(rows))

    assert summary["surface"]["skill_primary"] == 0.25
    assert summary["surface"]["skill_latw_primary"] == 0.2
    assert summary["surface"]["skill_median"] == -49.75
    assert summary["surface"]["skill_p05"] < -90.0
    assert summary["surface"]["skill_negative_outlier_count"] == 1
