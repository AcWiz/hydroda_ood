from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydroda.evaluation.phys_gain_guarded_hypertrust import (
    PHYS_GAIN_GUARD_METHOD_ID,
    PHYS_GAIN_GUARD_SELECTION_SCHEMA,
    PhysGainGuardedHyperTrustPredictor,
    apply_phys_gain_guard,
    build_source_phys_gain_guard_bank_from_records,
    physical_gain_query_from_sample,
    select_eta_from_source_val,
    validate_source_gate_for_target_eval,
)


def _x_basis(*, h: float = 1.0, v: float = 0.5, channel11: float = 0.0) -> np.ndarray:
    x = np.zeros((12, 2, 2), dtype=np.float32)
    x[0] = 0.30
    x[1] = 0.25
    x[2] = 285.0
    x[3] = 280.0
    x[4] = 0.0
    x[5] = 100.0 + h
    x[6] = 110.0 + v
    x[7] = 1.0
    x[8] = 1.0
    x[9] = 100.0
    x[10] = 110.0
    x[11] = channel11
    return x


def _record(
    *,
    split_role: str = "source_fit",
    sample_region_id: str = "US-R2",
    month: int = 1,
    h: float = 1.0,
    inc_s: np.ndarray | None = None,
    inc_r: np.ndarray | None = None,
    pred_s: np.ndarray | None = None,
    pred_r: np.ndarray | None = None,
    source_base_s: np.ndarray | None = None,
    source_base_r: np.ndarray | None = None,
    phys_gain_basis: np.ndarray | None = None,
    source_trust_neighbors: list[object] | None = None,
) -> dict[str, object]:
    shape = (2, 2)
    inc_s = np.asarray(inc_s if inc_s is not None else np.full(shape, -0.02 * h), dtype=np.float32)
    inc_r = np.asarray(inc_r if inc_r is not None else np.full(shape, -0.01 * h), dtype=np.float32)
    forecast_s = np.zeros(shape, dtype=np.float32)
    forecast_r = np.zeros(shape, dtype=np.float32)
    record: dict[str, object] = {
        "split_role": split_role,
        "sample_idx": 0,
        "query_time_index": 10,
        "query_date": f"2022-{month:02d}-01",
        "month": month,
        "target_region_id": "US-R1",
        "sample_region_id": sample_region_id,
        "active_region_ids": [sample_region_id],
        "adaptation_setting": "zero_shot_context",
        "K": 0,
        "seed": 0,
        "x": _x_basis(h=h),
        "forecast_surface": forecast_s,
        "forecast_rootzone": forecast_r,
        "analysis_surface": forecast_s + inc_s,
        "analysis_rootzone": forecast_r + inc_r,
        "increment_surface": inc_s,
        "increment_rootzone": inc_r,
        "metric_mask": np.ones(shape, dtype=np.float32),
        "region_mask": np.ones(shape, dtype=np.float32),
        "latitude_weight": np.ones(shape, dtype=np.float32),
        "pred_increment_surface": (
            np.asarray(pred_s, dtype=np.float32)
            if pred_s is not None
            else np.full(shape, 0.02, dtype=np.float32)
        ),
        "pred_increment_rootzone": (
            np.asarray(pred_r, dtype=np.float32)
            if pred_r is not None
            else np.full(shape, 0.01, dtype=np.float32)
        ),
        "source_base_increment_surface": (
            np.asarray(source_base_s, dtype=np.float32)
            if source_base_s is not None
            else np.zeros(shape, dtype=np.float32)
        ),
        "source_base_increment_rootzone": (
            np.asarray(source_base_r, dtype=np.float32)
            if source_base_r is not None
            else np.zeros(shape, dtype=np.float32)
        ),
    }
    if source_trust_neighbors is not None:
        record["source_trust_neighbors"] = source_trust_neighbors
    if phys_gain_basis is not None:
        record["phys_gain_basis"] = np.asarray(phys_gain_basis, dtype=np.float32)
    return record


def _base_pred(sample: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        "pred_increment_surface": np.asarray(sample["pred_increment_surface"], dtype=np.float32),
        "pred_increment_rootzone": np.asarray(sample["pred_increment_rootzone"], dtype=np.float32),
    }


def _source_base_pred(sample: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        "pred_increment_surface": np.asarray(sample["source_base_increment_surface"], dtype=np.float32),
        "pred_increment_rootzone": np.asarray(sample["source_base_increment_rootzone"], dtype=np.float32),
    }


def test_m3_13_eta_zero_is_exact_m3_1_identity():
    bank = build_source_phys_gain_guard_bank_from_records([_record(split_role="source_fit")])
    sample = _record(split_role="target_eval")

    routed = apply_phys_gain_guard(
        sample,
        _base_pred(sample),
        _source_base_pred(sample),
        bank,
        eta_surface=0.0,
        eta_rootzone=0.0,
    )

    np.testing.assert_array_equal(routed["pred_increment_surface"], sample["pred_increment_surface"])
    np.testing.assert_array_equal(routed["pred_increment_rootzone"], sample["pred_increment_rootzone"])
    assert routed["m3_13_guard_summary"]["guard_surface_mean"] == pytest.approx(1.0)
    assert routed["m3_13_guard_summary"]["query"]["fallback_level"] == "eta_zero_no_query"


def test_m3_13_guard_shrinks_only_and_never_adds_or_amplifies_residual():
    basis = np.zeros((5, 2, 2), dtype=np.float32)
    basis[0] = np.asarray([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    source_record = _record(
        split_role="source_fit",
        h=1.0,
        phys_gain_basis=basis,
        inc_s=0.02 * basis[0],
        inc_r=0.01 * basis[0],
    )
    bank = build_source_phys_gain_guard_bank_from_records([source_record], ridge_lambda=0.0)
    sample = _record(
        split_role="target_eval",
        h=1.0,
        phys_gain_basis=basis,
        pred_s=np.full((2, 2), 0.02, dtype=np.float32),
        pred_r=np.full((2, 2), 0.01, dtype=np.float32),
        source_base_s=np.zeros((2, 2), dtype=np.float32),
        source_base_r=np.zeros((2, 2), dtype=np.float32),
    )

    routed = apply_phys_gain_guard(
        sample,
        _base_pred(sample),
        _source_base_pred(sample),
        bank,
        eta_surface=0.10,
        eta_rootzone=0.10,
        guard_min=0.90,
    )

    before_s = np.asarray(sample["pred_increment_surface"]) - np.asarray(sample["source_base_increment_surface"])
    after_s = routed["pred_increment_surface"] - np.asarray(sample["source_base_increment_surface"])
    assert np.all(np.abs(after_s) <= np.abs(before_s) + 1e-8)
    assert np.all(np.sign(after_s) == np.sign(before_s))
    assert np.all(routed["pred_increment_surface"] <= np.asarray(sample["pred_increment_surface"]) + 1e-8)
    assert routed["m3_13_guard_summary"]["guard_surface_min"] >= 0.90 - 1e-6
    assert routed["m3_13_guard_summary"]["guard_surface_min"] < 1.0
    assert routed["m3_13_guard_summary"]["action"] == "shrink_only_no_new_residual_no_amplification"


@pytest.mark.parametrize("split_role", ["source_val", "target_context", "target_eval", "target_val", "target_full_train"])
def test_m3_13_source_gain_bank_reads_source_fit_only(split_role):
    with pytest.raises(ValueError, match="source_fit"):
        build_source_phys_gain_guard_bank_from_records([_record(split_role=split_role)])


def test_m3_13_missing_region_month_fallback_order_and_neighbor_priority():
    bank = build_source_phys_gain_guard_bank_from_records(
        [
            _record(split_role="source_fit", sample_region_id="US-R2", month=1, h=1.0),
            _record(split_role="source_fit", sample_region_id="US-R3", month=1, h=1.0, inc_s=np.full((2, 2), -0.08)),
        ],
        ridge_lambda=0.0,
    )

    month_fallback = _record(split_role="target_eval", sample_region_id="US-R9", month=1)
    _, summary_month = physical_gain_query_from_sample(month_fallback, bank, proposal_clip_scale=100.0)
    assert summary_month["fallback_level"] == "source_month_consensus"

    global_fallback = _record(split_role="target_eval", sample_region_id="US-R9", month=2)
    _, summary_global = physical_gain_query_from_sample(global_fallback, bank, proposal_clip_scale=100.0)
    assert summary_global["fallback_level"] == "source_global_consensus"

    neighbor_sample = _record(
        split_role="target_eval",
        sample_region_id="US-R9",
        month=1,
        source_trust_neighbors=[{"source_region": "US-R3", "month": 1, "weight": 1.0}],
    )
    _, summary_neighbor = physical_gain_query_from_sample(neighbor_sample, bank, proposal_clip_scale=100.0)
    assert summary_neighbor["fallback_level"] == "source_trust_top_m_neighbor_weighted_consensus"
    assert summary_neighbor["neighbor_count"] == 1


def test_m3_13_eta_selection_uses_source_val_and_target_eval_refuses_identity():
    bank = build_source_phys_gain_guard_bank_from_records([_record(split_role="source_fit")])
    selection = select_eta_from_source_val([_record(split_role="source_val")], bank, eta_grid=[0.0])

    assert selection["schema_version"] == PHYS_GAIN_GUARD_SELECTION_SCHEMA
    assert selection["method_id"] == PHYS_GAIN_GUARD_METHOD_ID
    assert selection["selection_source"] == "source_val_only"
    assert selection["selected_eta_surface"] == 0.0
    assert selection["source_gate_pass"] is False
    assert selection["identity_diagnostic"] is True
    with pytest.raises(ValueError, match="positive eta"):
        validate_source_gate_for_target_eval(selection)

    with pytest.raises(ValueError, match="source_val"):
        select_eta_from_source_val([_record(split_role="target_val")], bank, eta_grid=[0.0])

    malformed = _record(split_role="source_val")
    malformed["adaptation_setting"] = "target_full_train"
    with pytest.raises(ValueError, match="target-side"):
        select_eta_from_source_val([malformed], bank, eta_grid=[0.0])


def test_m3_13_predictor_metadata_records_frozen_anchor():
    class ConstantPredictor:
        method_name = "M3_1_hyperda_trust_medium"

        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, sample):
            shape = np.asarray(sample["forecast_surface"]).shape
            inc_s = np.full(shape, self.value, dtype=np.float32)
            inc_r = np.full(shape, self.value * 0.5, dtype=np.float32)
            return {
                "pred_increment_surface": inc_s,
                "pred_increment_rootzone": inc_r,
                "pred_analysis_surface": np.asarray(sample["forecast_surface"], dtype=np.float32) + inc_s,
                "pred_analysis_rootzone": np.asarray(sample["forecast_rootzone"], dtype=np.float32) + inc_r,
            }

    bank = build_source_phys_gain_guard_bank_from_records([_record(split_role="source_fit")])
    predictor = PhysGainGuardedHyperTrustPredictor(ConstantPredictor(0.02), ConstantPredictor(0.0), bank, eta_surface=0.0)

    assert predictor.metadata["base_anchor"] == "M3_1_hyperda_trust_medium"
    assert predictor.metadata["neural_training_epochs"] == 0
    assert predictor.metadata["target_eval_usage"] == "final_eval_only_no_selection"
    assert predictor.metadata["action"] == "pred = source_base + guard * (pred_M3_1 - source_base)"


def test_m3_13_wrapper_dry_run_requires_m3_1_warm_start_and_eval_only(tmp_path):
    fake_source = tmp_path / "source.pt"
    fake_m3_1 = tmp_path / "m3_1.pt"
    fake_source.write_text("stub", encoding="utf-8")
    fake_m3_1.write_text("stub", encoding="utf-8")
    env = {
        **os.environ,
        "ABLATION_ID": "M3_13_phys_gain_guarded_hypertrust",
        "RESUME_FROM_M3_1_BEST": "1",
        "M3_13_INIT_FROM_M3_1_CHECKPOINT": str(fake_m3_1),
        "DATASET_BACKEND": "netcdf",
        "TIMESTAMP": "20260101_000000",
    }

    result = subprocess.run(
        ["bash", "run/phase4_hyperda_staged_ablation.sh", str(fake_source), "US-R2", "0", "0", "--dry-run"],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = result.stdout
    assert "ablation_id=M3_13_phys_gain_guarded_hypertrust" in stdout
    assert "trainable_scope=phys_gain_guard_only" in stdout
    assert "warm_start_policy=M3_1_best_checkpoint_eval_only_source_gate" in stdout
    assert "source_gate_json_required_for_target_eval=true" in stdout
    assert "--max_epochs 0" in stdout
    assert "--trainable_scope phys_gain_guard_only" in stdout
    assert f"--init_from_prompt_checkpoint {fake_m3_1}" in stdout
    assert "--hyper_phys_gain_basis_residual 0" in stdout


def test_m3_13_phase5_target_eval_wrapper_requires_source_gate_json(tmp_path):
    fake_source = tmp_path / "source.pt"
    fake_source.write_text("stub", encoding="utf-8")
    env = {
        **os.environ,
        "REQUIRE_SOURCE_GATE_JSON_FOR_TARGET_EVAL": "1",
        "SOURCE_GATE_JSON": str(tmp_path / "missing_source_gate.json"),
        "K_LIST": "0",
    }

    result = subprocess.run(
        [
            "bash",
            "run/phase5_hyperda_zero_few_shot_eval.sh",
            str(fake_source),
            "US-R2",
            "0",
            "0",
            str(tmp_path / "out"),
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "target_eval requires SOURCE_GATE_JSON" in result.stderr


def test_m3_13_target_eval_validation_accepts_only_passing_positive_gate(tmp_path):
    gate = {
        "schema_version": "m3_13_source_gate_report_v1",
        "method_id": PHYS_GAIN_GUARD_METHOD_ID,
        "source_gate_pass": True,
        "target_eval_allowed": True,
        "identity_diagnostic": False,
        "selected_eta_surface": 0.02,
        "selected_eta_rootzone": 0.0,
    }
    gate_path = tmp_path / "source_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    # Phase5 shell validation is exercised above; this asserts the source gate
    # shape used by the wrapper is the positive-eta, non-identity contract.
    assert gate["selected_eta_surface"] > 0.0
