import numpy as np
import pandas as pd

from hydroda.baselines.forecast import ForecastBaseline
from hydroda.evaluation.harness import (
    evaluate_split,
    metric_values_content_hash,
    mix_prediction_with_zero_shot,
    summarize_metric_rows,
)


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


def test_global_metric_rows_carry_zero_few_shot_metadata():
    rows = evaluate_split(
        TinyDataset(),
        ForecastBaseline(),
        split_role="target_eval",
        experiment_id="tiny",
        protocol_freeze_id="test",
        method="forecast_only",
        split_file="splits.json",
        mask_file="masks.nc",
        target_context_dates_hash="context-hash",
        target_support_dates_hash="support-hash",
        target_eval_dates_hash="eval-hash",
        split_manifest_sha256="split-sha",
    )

    global_rows = [row for row in rows if row["query_date"] == "global"]
    assert global_rows
    for row in global_rows:
        assert row["target_context_dates_hash"] == "context-hash"
        assert row["target_support_dates_hash"] == "support-hash"
        assert row["target_eval_dates_hash"] == "eval-hash"
        assert row["split_manifest_sha256"] == "split-sha"
        assert row["split_file"] == "splits.json"
        assert row["mask_file"] == "masks.nc"
        assert row["country_id"] == "US"
        assert row["target_region_id"] == "US-R1"
        assert row["split_role"] == "target_eval"


def test_evaluate_split_can_return_prediction_hashes():
    rows, hashes = evaluate_split(
        TinyDataset(),
        ForecastBaseline(),
        split_role="target_eval",
        experiment_id="tiny",
        protocol_freeze_id="test",
        method="forecast_only",
        return_hashes=True,
    )

    assert rows
    assert hashes["prediction_content_hash"]
    assert hashes["prediction_record_count"] == 1
    assert hashes["metric_content_hash"]
    assert hashes["metric_row_count"] == len(rows)

    repeat_rows, repeat_hashes = evaluate_split(
        TinyDataset(),
        ForecastBaseline(),
        split_role="target_eval",
        experiment_id="tiny",
        protocol_freeze_id="test",
        method="forecast_only",
        return_hashes=True,
    )
    assert len(repeat_rows) == len(rows)
    assert repeat_hashes == hashes


def test_evaluate_split_can_persist_source_safe_prediction_records(tmp_path):
    record_path = tmp_path / "prediction_records.jsonl"

    rows, hashes = evaluate_split(
        TinyDataset(),
        ForecastBaseline(),
        split_role="source_val",
        experiment_id="tiny",
        protocol_freeze_id="test",
        method="forecast_only",
        return_hashes=True,
        prediction_record_path=record_path,
    )

    assert rows
    assert record_path.exists()
    records = [__import__("json").loads(line) for line in record_path.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == "hydroda_prediction_record_v1"
    assert record["split_role"] == "source_val"
    assert record["prediction_content_hash"] == hashes["prediction_content_hash"]
    for key in [
        "forecast_surface",
        "analysis_surface",
        "increment_surface",
        "pred_increment_surface",
        "metric_mask",
        "latitude_weight",
    ]:
        assert key in record["arrays"]
        assert record["arrays"][key]["shape"] == [2, 2]


def test_metric_value_hash_ignores_run_metadata_but_keeps_values():
    rows_a = evaluate_split(
        TinyDataset(),
        ForecastBaseline(),
        split_role="target_eval",
        experiment_id="tiny_a",
        protocol_freeze_id="test",
        method="method_a",
    )
    rows_b = evaluate_split(
        TinyDataset(),
        ForecastBaseline(),
        split_role="target_eval",
        experiment_id="tiny_b",
        protocol_freeze_id="test",
        method="method_b",
    )
    assert metric_values_content_hash(rows_a) == metric_values_content_hash(rows_b)

    rows_b[0] = dict(rows_b[0])
    rows_b[0]["value"] = rows_b[0]["value"] + 1.0

    assert metric_values_content_hash(rows_a) != metric_rows_content_hash_for_test(rows_a)
    assert metric_values_content_hash(rows_a) != metric_values_content_hash(rows_b)


def metric_rows_content_hash_for_test(rows):
    from hydroda.evaluation.harness import metric_rows_content_hash

    return metric_rows_content_hash(rows)


def test_mix_prediction_rho_zero_and_one_reproduce_endpoints():
    sample = {
        "forecast_surface": np.ones((2, 2), dtype=np.float32) * 10,
        "forecast_rootzone": np.ones((2, 2), dtype=np.float32) * 20,
    }
    zero = {
        "pred_increment_surface": np.ones((2, 2), dtype=np.float32),
        "pred_increment_rootzone": np.ones((2, 2), dtype=np.float32) * 2,
    }
    adapted = {
        "pred_increment_surface": np.ones((2, 2), dtype=np.float32) * 5,
        "pred_increment_rootzone": np.ones((2, 2), dtype=np.float32) * 8,
    }

    mixed0 = mix_prediction_with_zero_shot(sample, adapted, zero, rho=0.0)
    mixed1 = mix_prediction_with_zero_shot(sample, adapted, zero, rho=1.0)
    mixed_half = mix_prediction_with_zero_shot(sample, adapted, zero, rho=0.5)

    assert np.allclose(mixed0["pred_increment_surface"], zero["pred_increment_surface"])
    assert np.allclose(mixed0["pred_analysis_surface"], sample["forecast_surface"] + zero["pred_increment_surface"])
    assert np.allclose(mixed1["pred_increment_rootzone"], adapted["pred_increment_rootzone"])
    assert np.allclose(mixed_half["pred_increment_surface"], np.ones((2, 2), dtype=np.float32) * 3)
