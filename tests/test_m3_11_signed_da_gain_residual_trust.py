from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydroda.evaluation.signed_da_gain_residual_trust import (
    SIGNED_DA_GAIN_BANK_SCHEMA,
    SIGNED_DA_GAIN_METHOD_ID,
    SignedDAGainBankAccumulator,
    SignedDAGainResidualTrustPredictor,
    blend_prediction_with_signed_da_gain,
    build_source_signed_da_gain_bank_from_records,
    da_innovations_from_x,
    physical_proposal_from_sample,
    select_eta_from_source_val,
    validate_source_gate_for_target_eval,
)


def _x_from_innovations(d_h: np.ndarray, d_v: np.ndarray, *, channel11: float = 0.0) -> np.ndarray:
    x = np.zeros((12, *d_h.shape), dtype=np.float32)
    x[5] = d_h
    x[6] = d_v
    x[7] = 1.0
    x[8] = 1.0
    x[9] = 0.0
    x[10] = 0.0
    x[11] = channel11
    return x


def _record(
    *,
    split_role: str = "source_fit",
    sample_region_id: str = "US-R2",
    month: int = 1,
    d_h: np.ndarray | None = None,
    d_v: np.ndarray | None = None,
    inc_s: np.ndarray | None = None,
    inc_r: np.ndarray | None = None,
    pred_s: np.ndarray | None = None,
    pred_r: np.ndarray | None = None,
    channel11: float = 0.0,
    source_trust_neighbors: list[object] | None = None,
) -> dict[str, object]:
    d_h = np.asarray(d_h if d_h is not None else [[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    d_v = np.asarray(d_v if d_v is not None else [[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    inc_s = np.asarray(inc_s if inc_s is not None else 2.0 * d_h, dtype=np.float32)
    inc_r = np.asarray(inc_r if inc_r is not None else 3.0 * d_h + 4.0 * inc_s, dtype=np.float32)
    forecast_s = np.zeros_like(d_h, dtype=np.float32)
    forecast_r = np.zeros_like(d_h, dtype=np.float32)
    record = {
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
        "x": _x_from_innovations(d_h, d_v, channel11=channel11),
        "forecast_surface": forecast_s,
        "forecast_rootzone": forecast_r,
        "analysis_surface": forecast_s + inc_s,
        "analysis_rootzone": forecast_r + inc_r,
        "increment_surface": inc_s,
        "increment_rootzone": inc_r,
        "metric_mask": np.ones_like(d_h, dtype=np.float32),
        "region_mask": np.ones_like(d_h, dtype=np.float32),
        "latitude_weight": np.ones_like(d_h, dtype=np.float32),
        "pred_increment_surface": (
            np.asarray(pred_s, dtype=np.float32)
            if pred_s is not None
            else np.full_like(d_h, 0.5, dtype=np.float32)
        ),
        "pred_increment_rootzone": (
            np.asarray(pred_r, dtype=np.float32)
            if pred_r is not None
            else np.full_like(d_h, 0.25, dtype=np.float32)
        ),
    }
    if source_trust_neighbors is not None:
        record["source_trust_neighbors"] = source_trust_neighbors
    return record


def test_m3_11_da_innovation_sign_and_formula():
    d_h = np.asarray([[2.0, -3.0]], dtype=np.float32)
    d_v = np.asarray([[-4.0, 5.0]], dtype=np.float32)
    x = _x_from_innovations(d_h, d_v)

    got_h, got_v = da_innovations_from_x(x, eps=0.0)

    np.testing.assert_allclose(got_h, d_h)
    np.testing.assert_allclose(got_v, d_v)


@pytest.mark.parametrize("split_role", ["source_val", "source_train", "target_eval", "target_val", "target_full_train"])
def test_m3_11_gain_bank_reads_source_fit_only(split_role):
    with pytest.raises(ValueError, match="source_fit"):
        build_source_signed_da_gain_bank_from_records([_record(split_role=split_role)])


def test_m3_11_gain_bank_toy_covariance_schema_and_formula():
    d = np.asarray([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    zeros = np.zeros_like(d)
    bank = build_source_signed_da_gain_bank_from_records(
        [
            _record(
                split_role="source_fit",
                d_h=d,
                d_v=zeros,
                inc_s=2.0 * d,
                inc_r=11.0 * d,
            )
        ],
        ridge_lambda=0.0,
    )

    entry = bank["entries"]["US-R2|01"]
    assert bank["schema_version"] == SIGNED_DA_GAIN_BANK_SCHEMA
    assert bank["method_id"] == SIGNED_DA_GAIN_METHOD_ID
    assert entry["gains"]["surface"]["H"] == pytest.approx(2.0)
    assert entry["gains"]["surface"]["V"] == pytest.approx(0.0)
    assert entry["gains"]["rootzone"]["H"] == pytest.approx(11.0)
    assert entry["C_rz"] == pytest.approx(5.5)
    assert bank["target_eval_usage"] == "not_used_for_bank_or_eta_selection"


def test_m3_11_streaming_accumulator_matches_records_builder():
    records = [
        _record(
            split_role="source_fit",
            sample_region_id="US-R2",
            month=1,
            d_h=np.asarray([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32),
        ),
        _record(
            split_role="source_fit",
            sample_region_id="US-R2",
            month=1,
            d_h=np.asarray([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
        ),
    ]
    expected = build_source_signed_da_gain_bank_from_records(records, ridge_lambda=0.0)
    accumulator = SignedDAGainBankAccumulator(ridge_lambda=0.0)
    for record in records:
        accumulator.update(record)
    actual = accumulator.finalize()

    assert actual["accumulator"] == "streaming_covariance_moments_v1"
    assert actual["n_source_records_seen"] == 2
    assert actual["n_source_records_used"] == 2
    assert actual["entries"]["US-R2|01"]["C_rz"] == pytest.approx(expected["entries"]["US-R2|01"]["C_rz"])
    assert actual["bank_content_hash"]


def test_m3_11_eta_zero_output_strictly_matches_base_prediction():
    d = np.asarray([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    bank = build_source_signed_da_gain_bank_from_records(
        [_record(split_role="source_fit", d_h=d, d_v=np.zeros_like(d), inc_s=2.0 * d, inc_r=3.0 * d)],
        ridge_lambda=0.0,
    )
    sample = _record(split_role="target_eval", d_h=d, d_v=np.zeros_like(d))
    base_pred = {
        "pred_increment_surface": np.asarray(sample["pred_increment_surface"], dtype=np.float32),
        "pred_increment_rootzone": np.asarray(sample["pred_increment_rootzone"], dtype=np.float32),
        "pred_analysis_surface": np.asarray(sample["forecast_surface"], dtype=np.float32)
        + np.asarray(sample["pred_increment_surface"], dtype=np.float32),
        "pred_analysis_rootzone": np.asarray(sample["forecast_rootzone"], dtype=np.float32)
        + np.asarray(sample["pred_increment_rootzone"], dtype=np.float32),
    }

    final = blend_prediction_with_signed_da_gain(sample, base_pred, bank, eta_surface=0.0, eta_rootzone=0.0)

    np.testing.assert_array_equal(final["pred_increment_surface"], base_pred["pred_increment_surface"])
    np.testing.assert_array_equal(final["pred_increment_rootzone"], base_pred["pred_increment_rootzone"])
    np.testing.assert_array_equal(final["pred_analysis_surface"], base_pred["pred_analysis_surface"])
    np.testing.assert_array_equal(final["pred_analysis_rootzone"], base_pred["pred_analysis_rootzone"])


def test_m3_11_missing_region_month_fallback_order_skips_region_consensus():
    d = np.asarray([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    bank = build_source_signed_da_gain_bank_from_records(
        [_record(split_role="source_fit", sample_region_id="US-R2", month=1, d_h=d, d_v=np.zeros_like(d))],
        ridge_lambda=0.0,
    )

    same_month_missing_region = _record(split_role="target_eval", sample_region_id="US-R9", month=1, d_h=d)
    _, summary_month = physical_proposal_from_sample(same_month_missing_region, bank)
    assert summary_month["fallback_level"] == "source_month_consensus"

    missing_month = _record(split_role="target_eval", sample_region_id="US-R9", month=2, d_h=d)
    _, summary_global = physical_proposal_from_sample(missing_month, bank)
    assert summary_global["fallback_level"] == "source_global_consensus"


def test_m3_11_explicit_source_trust_neighbors_take_priority():
    d = np.asarray([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    bank = build_source_signed_da_gain_bank_from_records(
        [
            _record(split_role="source_fit", sample_region_id="US-R2", month=1, d_h=d, d_v=np.zeros_like(d), inc_s=2.0 * d),
            _record(split_role="source_fit", sample_region_id="US-R3", month=1, d_h=d, d_v=np.zeros_like(d), inc_s=4.0 * d),
        ],
        ridge_lambda=0.0,
    )
    sample = _record(
        split_role="target_eval",
        sample_region_id="US-R9",
        month=1,
        d_h=d,
        source_trust_neighbors=[
            {"source_region": "US-R3", "month": 1, "weight": 1.0},
        ],
    )

    proposal, summary = physical_proposal_from_sample(sample, bank, proposal_clip_scale=100.0)

    assert summary["fallback_level"] == "source_trust_top_m_neighbor_weighted_consensus"
    np.testing.assert_allclose(proposal["surface"], 4.0 * d, rtol=1e-5, atol=1e-5)


def test_m3_11_channel_11_does_not_enter_hard_mask():
    d = np.asarray([[1.0, 1.0]], dtype=np.float32)
    bank = build_source_signed_da_gain_bank_from_records(
        [_record(split_role="source_fit", d_h=d, d_v=np.zeros_like(d), inc_s=2.0 * d, inc_r=3.0 * d)],
        ridge_lambda=0.0,
    )
    sample_channel_zero = _record(split_role="target_eval", d_h=d, d_v=np.zeros_like(d), channel11=0.0)
    sample_channel_one = _record(split_role="target_eval", d_h=d, d_v=np.zeros_like(d), channel11=1.0)
    base_pred = {
        "pred_increment_surface": np.zeros_like(d, dtype=np.float32),
        "pred_increment_rootzone": np.zeros_like(d, dtype=np.float32),
        "pred_analysis_surface": np.zeros_like(d, dtype=np.float32),
        "pred_analysis_rootzone": np.zeros_like(d, dtype=np.float32),
    }

    pred_zero = blend_prediction_with_signed_da_gain(sample_channel_zero, base_pred, bank, eta_surface=1.0, eta_rootzone=1.0)
    pred_one = blend_prediction_with_signed_da_gain(sample_channel_one, base_pred, bank, eta_surface=1.0, eta_rootzone=1.0)

    np.testing.assert_allclose(pred_zero["pred_increment_surface"], pred_one["pred_increment_surface"])
    assert bank["channel_11_usage"] == "diagnostic_coverage_only_not_obs_loss_metric_region_mask"


def test_m3_11_eta_selection_uses_source_val_and_target_eval_gate_refuses_eta_zero():
    d = np.asarray([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    bank = build_source_signed_da_gain_bank_from_records([_record(split_role="source_fit", d_h=d)], ridge_lambda=0.0)
    selection = select_eta_from_source_val(
        [_record(split_role="source_val", d_h=d)],
        bank,
        eta_grid=[0.0],
        require_corr_or_sign_nondecline=False,
    )

    assert selection["selection_source"] == "source_val_only"
    assert selection["selected_eta_surface"] == 0.0
    assert selection["source_gate_pass"] is False
    with pytest.raises(ValueError, match="positive eta"):
        validate_source_gate_for_target_eval(selection)

    with pytest.raises(ValueError, match="source_val"):
        select_eta_from_source_val([_record(split_role="target_val", d_h=d)], bank, eta_grid=[0.0])

    malformed = _record(split_role="source_val", d_h=d)
    malformed["adaptation_setting"] = "target_full_train"
    with pytest.raises(ValueError, match="target-side"):
        select_eta_from_source_val([malformed], bank, eta_grid=[0.0])


def test_m3_11_predictor_metadata_records_frozen_anchor():
    class ConstantPredictor:
        method_name = "M3_1_hyperda_trust_medium"

        def predict(self, sample):
            shape = np.asarray(sample["forecast_surface"]).shape
            inc_s = np.ones(shape, dtype=np.float32)
            inc_r = np.ones(shape, dtype=np.float32) * 2
            return {
                "pred_increment_surface": inc_s,
                "pred_increment_rootzone": inc_r,
                "pred_analysis_surface": np.asarray(sample["forecast_surface"], dtype=np.float32) + inc_s,
                "pred_analysis_rootzone": np.asarray(sample["forecast_rootzone"], dtype=np.float32) + inc_r,
            }

    bank = build_source_signed_da_gain_bank_from_records([_record(split_role="source_fit")])
    predictor = SignedDAGainResidualTrustPredictor(ConstantPredictor(), bank, eta_surface=0.025)

    assert predictor.metadata["base_anchor"] == "M3_1_hyperda_trust_medium"
    assert predictor.metadata["neural_training_epochs"] == 0
    assert predictor.metadata["target_eval_usage"] == "final_eval_only_no_selection"
