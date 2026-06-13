from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _array_payload(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": "float32",
        "values": np.asarray(value, dtype=np.float32).reshape(-1).tolist(),
    }


def _record(path: Path, *, split_role: str, pred_value: float, kind: str) -> Path:
    record = {
        "schema_version": "hydroda_prediction_record_v1",
        "sample_idx": 0,
        "query_time_index": 10,
        "query_date": "2022-01-01",
        "split_role": split_role,
        "target_region_id": "US-R2",
        "sample_region_id": "US-R2",
        "active_region_ids": ["US-R2"],
        "adaptation_setting": "few_shot_k12",
        "K": 12,
        "seed": 0,
        "kind": kind,
        "arrays": {
            "forecast_surface": _array_payload(np.zeros((2, 2), dtype=np.float32)),
            "forecast_rootzone": _array_payload(np.zeros((2, 2), dtype=np.float32)),
            "analysis_surface": _array_payload(np.ones((2, 2), dtype=np.float32) * 2),
            "analysis_rootzone": _array_payload(np.ones((2, 2), dtype=np.float32) * 4),
            "increment_surface": _array_payload(np.ones((2, 2), dtype=np.float32) * 2),
            "increment_rootzone": _array_payload(np.ones((2, 2), dtype=np.float32) * 4),
            "pred_increment_surface": _array_payload(np.ones((2, 2), dtype=np.float32) * pred_value),
            "pred_increment_rootzone": _array_payload(np.ones((2, 2), dtype=np.float32) * pred_value * 2),
            "metric_mask": _array_payload(np.ones((2, 2), dtype=np.float32)),
            "latitude_weight": _array_payload(np.ones((2, 2), dtype=np.float32)),
        },
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def test_mix_prediction_records_endpoints_and_fixed_interpolation(tmp_path):
    from scripts.eval import mix_prediction_records as mixer

    k0_path = _record(tmp_path / "k0.jsonl", split_role="source_val", pred_value=1.0, kind="zero_shot")
    adapted_path = _record(
        tmp_path / "adapted.jsonl",
        split_role="source_val",
        pred_value=3.0,
        kind="adapted",
    )

    mixed0 = mixer.mix_prediction_record_files(k0_path, adapted_path, rho=0.0, candidate_id="rho0")
    mixed1 = mixer.mix_prediction_record_files(k0_path, adapted_path, rho=1.0, candidate_id="rho1")
    mixed_half = mixer.mix_prediction_record_files(k0_path, adapted_path, rho=0.5, candidate_id="rhohalf")

    assert mixed0["records"][0]["arrays"]["pred_increment_surface"]["values"] == [1.0] * 4
    assert mixed1["records"][0]["arrays"]["pred_increment_surface"]["values"] == [3.0] * 4
    assert mixed_half["records"][0]["arrays"]["pred_increment_surface"]["values"] == [2.0] * 4
    assert mixed0["verification"]["rho0_equals_k0"] is True
    assert mixed1["verification"]["rho1_equals_adapted"] is True
    assert mixed_half["mixed_prediction_content_hash"]
    assert mixed_half["zero_shot_prediction_content_hash"]
    assert mixed_half["adapted_prediction_content_hash"]
    assert mixed_half["summary"]["surface"]["skill_primary"] == pytest.approx(1.0)


@pytest.mark.parametrize("split_role", ["target_eval", "target_query", "target_val", "target_full_train"])
def test_mix_prediction_records_reject_target_side_splits_in_calibration_mode(tmp_path, split_role):
    from scripts.eval import mix_prediction_records as mixer

    k0_path = _record(tmp_path / "k0.jsonl", split_role=split_role, pred_value=1.0, kind="zero_shot")
    adapted_path = _record(tmp_path / "adapted.jsonl", split_role="source_val", pred_value=3.0, kind="adapted")

    with pytest.raises(ValueError, match=split_role):
        mixer.mix_prediction_record_files(k0_path, adapted_path, rho=0.5, candidate_id="bad")


def test_conflict_rule_logical_rows_are_generated_without_new_adaptation(tmp_path):
    from scripts.eval import mix_prediction_records as mixer

    k0_path = _record(tmp_path / "k0.jsonl", split_role="source_val", pred_value=1.0, kind="zero_shot")
    adapted_path = _record(tmp_path / "adapted.jsonl", split_role="source_val", pred_value=3.0, kind="adapted")

    rows = mixer.generate_conflict_rule_logical_rows(
        zero_shot_records=k0_path,
        adapted_records=adapted_path,
        base_candidate={
            "candidate_id": "C0",
            "schedule_label": "original_K12",
            "support_loss_reduction": "global_pixel",
            "trust_policy": "none",
            "support_gradient_negative_fraction": 0.6,
            "support_gradient_cosine_min": -0.1,
        },
        rho_policies=["rule_a", "rule_b", "rule_c"],
    )

    by_policy = {row["rho_policy"]: row for row in rows}
    assert by_policy["rule_a"]["adapt_mix_rho"] == pytest.approx(0.5)
    assert by_policy["rule_b"]["adapt_mix_rho"] == pytest.approx(0.5)
    assert by_policy["rule_c"]["adapt_mix_rho"] == pytest.approx(0.25)
    assert all(row["logical_offline_mix"] is True for row in rows)
    assert all(row["source_prediction_record_hash"] for row in rows)
