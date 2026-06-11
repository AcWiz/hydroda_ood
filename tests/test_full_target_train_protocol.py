"""Regression tests for legacy full-target historical adaptation.

No-leakage declaration:
    - Target training/adaptation dates are limited to the 2015-2021 target train period.
    - Source validation dates are limited to the 2022 source validation period.
    - Target evaluation labels remain 2023-2025 offline-evaluation-only.
    - Final test labels never drive training, adaptation, normalization, or selection.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydroda.data.leakage_guard import LeakageGuard
from hydroda.data.protocol import ProtocolConfig
from hydroda.evaluation.harness import evaluate_split
from hydroda.splits.manifest import create_split_manifest, validate_no_leakage


def test_protocol_main_adaptation_setting_replaces_main_k_shot():
    p = ProtocolConfig()

    assert "zero_few_shot_generalization" in p.protocol_freeze_id
    assert p.role_for_date("2021-06-01") == "source_fit"
    assert p.role_for_date("2022-06-01") == "source_val"
    assert p.role_for_date("2023-06-01") == "target_eval"
    assert tuple(p.main_adaptation_settings) == ("zero_shot_context", "few_shot_k4", "few_shot_k12")
    assert tuple(p.main_K_values) == (0, 4, 12)
    p.assert_dates_within(["2019-06-01", "2021-12-31"], ["target_train"], "target_adaptation")

    p.assert_supported_adaptation_setting("few_shot_k4")
    p.assert_supported_adaptation_setting("target_full_train", allow_legacy_full_target_train=True)
    with pytest.raises(ValueError):
        p.assert_supported_adaptation_setting("target_full_train")


def test_leakage_guard_allows_target_train_adaptation_but_not_selection():
    guard = LeakageGuard(ProtocolConfig())

    guard.check_target_adaptation_scope(
        ["2016-03-01", "2021-09-01"],
        purpose="adapter_tuning",
        labels_allowed=True,
    )
    guard.check_model_selection_scope(
        ["2022-06-01"],
        purpose="checkpoint_selection",
        model_selection_source="source_val_preregistered",
    )

    with pytest.raises(ValueError):
        guard.check_target_adaptation_scope(["2023-01-01"], purpose="adapter_tuning", labels_allowed=True)
    with pytest.raises(ValueError):
        guard.check_model_selection_scope(["2021-06-01"], purpose="checkpoint_selection")
    with pytest.raises(ValueError):
        guard.check_normalization_scope(["2022-06-01"], scope_name="source_fit_only")


def test_full_target_train_manifest_schema_and_validation():
    manifest = create_split_manifest(
        target_region="US-R1",
        source_regions=["US-R2"],
        adaptation_setting="target_full_train",
        seed=0,
        source_train_dates=[
            {"time_index": 0, "date_str": "2021-01-15", "datetime_str": "2021-01-15T00:00:00Z"},
        ],
        source_val_dates=[
            {"time_index": 10, "date_str": "2022-06-15", "datetime_str": "2022-06-15T00:00:00Z"},
        ],
        target_train_dates=[
            {"time_index": 20, "date_str": "2015-03-15", "datetime_str": "2015-03-15T00:00:00Z"},
            {"time_index": 21, "date_str": "2021-09-15", "datetime_str": "2021-09-15T00:00:00Z"},
        ],
        query_dates=[
            {"time_index": 30, "date_str": "2023-01-15", "datetime_str": "2023-01-15T00:00:00Z"},
        ],
        allow_legacy_full_target_train=True,
    )

    assert manifest["adaptation_setting"] == "target_full_train"
    assert manifest["adaptation_protocol"] == "legacy_full_target_train"
    assert manifest["target_train_cycle_count"] == 2
    assert manifest["target_adaptation_cycle_count"] == 2
    assert manifest["target_support_dates"] == manifest["target_train_dates"]
    assert manifest["K"] is None
    assert manifest["K_legacy"] is None
    assert manifest["target_train_dates_hash"]
    assert manifest["target_eval_dates_hash"]
    assert manifest["target_full_train_usage"] == "legacy_internal_only"

    results = validate_no_leakage(manifest)
    assert results["target_train_in_train_year"] is True
    assert results["query_in_query_years"] is True
    assert results["no_target_train_eval_overlap"] is True
    assert results["selection_uses_query_labels_false"] is True


def test_evaluation_rows_include_adaptation_metadata():
    class MockPredictor:
        def predict(self, sample):
            return {
                "pred_increment_surface": np.zeros((2, 2), dtype=np.float32),
                "pred_increment_rootzone": np.zeros((2, 2), dtype=np.float32),
                "pred_analysis_surface": sample["forecast_surface"],
                "pred_analysis_rootzone": sample["forecast_rootzone"],
            }

    class MockDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            arr = np.ones((2, 2), dtype=np.float32)
            return {
                "forecast_surface": arr,
                "forecast_rootzone": arr,
                "analysis_surface": arr,
                "analysis_rootzone": arr,
                "increment_surface": np.zeros((2, 2), dtype=np.float32),
                "increment_rootzone": np.zeros((2, 2), dtype=np.float32),
                "metric_mask": np.ones((2, 2), dtype=np.float32),
                "latitude_weight": np.ones((2, 2), dtype=np.float32),
                "date_str": "2023-01-01",
                "month": 1,
                "season": "DJF",
                "time_index": 30,
                "country_id": "US",
                "target_region_id": "US-R1",
                "active_region_ids": ["US-R1"],
                "adaptation_setting": "target_full_train",
                "target_context_dates_hash": "contexthash",
                "target_support_dates_hash": "supporthash",
                "support_dates_hash": "supporthash",
                "target_train_dates_hash": "trainhash",
                "target_eval_dates_hash": "evalhash",
                "split_manifest_sha256": "manifesthash",
                "K": None,
                "seed": 0,
            }

    rows = evaluate_split(
        dataset=MockDataset(),
        predictor=MockPredictor(),
        split_role="target_eval",
        experiment_id="full_target_train_eval",
        protocol_freeze_id="test_full_target_train",
        method="forecast_only",
        split_manifest_sha256="manifesthash",
    )

    first = rows[0]
    assert first["adaptation_setting"] == "target_full_train"
    assert first["target_context_dates_hash"] == "contexthash"
    assert first["target_support_dates_hash"] == "supporthash"
    assert first["support_dates_hash"] == "supporthash"
    assert first["target_train_dates_hash"] == "trainhash"
    assert first["target_eval_dates_hash"] == "evalhash"
    assert first["split_manifest_sha256"] == "manifesthash"
    assert first["K"] == "legacy_none"

    global_row = next(row for row in rows if row["query_date"] == "global")
    assert global_row["target_context_dates_hash"] == "contexthash"
    assert global_row["target_support_dates_hash"] == "supporthash"
    assert global_row["support_dates_hash"] == "supporthash"
    assert global_row["target_train_dates_hash"] == "trainhash"
    assert global_row["target_eval_dates_hash"] == "evalhash"
