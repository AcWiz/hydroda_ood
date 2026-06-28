from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydroda.evaluation.da_gain_router import (
    DAGainBankAccumulator,
    DAGainRouterPredictor,
    blend_prediction_with_da_gain,
    build_source_da_gain_bank_from_records,
    select_eta_from_source_val,
)


def _x_from_innovations(d_h: np.ndarray, d_v: np.ndarray) -> np.ndarray:
    x = np.zeros((12, *d_h.shape), dtype=np.float32)
    x[5] = d_h
    x[6] = d_v
    x[7] = 1.0
    x[8] = 1.0
    x[9] = 0.0
    x[10] = 0.0
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
) -> dict[str, object]:
    d_h = np.asarray(d_h if d_h is not None else [[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    d_v = np.asarray(d_v if d_v is not None else [[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    inc_s = np.asarray(inc_s if inc_s is not None else 2.0 * d_h, dtype=np.float32)
    inc_r = np.asarray(inc_r if inc_r is not None else 3.0 * d_h + 4.0 * inc_s, dtype=np.float32)
    forecast_s = np.zeros_like(d_h, dtype=np.float32)
    forecast_r = np.zeros_like(d_h, dtype=np.float32)
    return {
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
        "x": _x_from_innovations(d_h, d_v),
        "forecast_surface": forecast_s,
        "forecast_rootzone": forecast_r,
        "analysis_surface": forecast_s + inc_s,
        "analysis_rootzone": forecast_r + inc_r,
        "increment_surface": inc_s,
        "increment_rootzone": inc_r,
        "metric_mask": np.ones_like(d_h, dtype=np.float32),
        "latitude_weight": np.ones_like(d_h, dtype=np.float32),
        "pred_increment_surface": np.full_like(d_h, 0.5, dtype=np.float32),
        "pred_increment_rootzone": np.full_like(d_h, 0.25, dtype=np.float32),
    }


@pytest.mark.parametrize("split_role", ["source_val", "source_train", "target_eval", "target_val"])
def test_da_gain_bank_reads_source_fit_only(split_role):
    with pytest.raises(ValueError, match="source_fit"):
        build_source_da_gain_bank_from_records([_record(split_role=split_role)])


def test_da_gain_bank_toy_covariance_is_numeric():
    d = np.asarray([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    zeros = np.zeros_like(d)
    bank = build_source_da_gain_bank_from_records(
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
    assert entry["gains"]["surface"]["H"] == pytest.approx(2.0)
    assert entry["gains"]["surface"]["V"] == pytest.approx(0.0)
    assert entry["gains"]["rootzone"]["H"] == pytest.approx(11.0)
    assert entry["C_rz"] == pytest.approx(5.5)
    assert bank["source_label_usage"] == "source_fit_labels_only"
    assert bank["exploratory_after_us_r1_target_eval_seen"] is True


def test_streaming_da_gain_bank_matches_records_builder():
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
    expected = build_source_da_gain_bank_from_records(records, ridge_lambda=0.0)
    accumulator = DAGainBankAccumulator(ridge_lambda=0.0)
    for record in records:
        accumulator.update(record)
    actual = accumulator.finalize()

    assert actual["accumulator"] == "streaming_covariance_moments_v1"
    assert actual["n_source_records_seen"] == 2
    assert actual["n_source_records_used"] == 2
    for variable in ("surface", "rootzone"):
        for pol in ("H", "V"):
            assert actual["entries"]["US-R2|01"]["gains"][variable][pol] == pytest.approx(
                expected["entries"]["US-R2|01"]["gains"][variable][pol]
            )
    assert actual["entries"]["US-R2|01"]["C_rz"] == pytest.approx(expected["entries"]["US-R2|01"]["C_rz"])


def test_eta_zero_output_strictly_matches_base_prediction():
    d = np.asarray([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    bank = build_source_da_gain_bank_from_records(
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

    final = blend_prediction_with_da_gain(sample, base_pred, bank, eta_surface=0.0, eta_rootzone=0.0)

    np.testing.assert_array_equal(final["pred_increment_surface"], base_pred["pred_increment_surface"])
    np.testing.assert_array_equal(final["pred_increment_rootzone"], base_pred["pred_increment_rootzone"])
    np.testing.assert_array_equal(final["pred_analysis_surface"], base_pred["pred_analysis_surface"])
    np.testing.assert_array_equal(final["pred_analysis_rootzone"], base_pred["pred_analysis_rootzone"])


def test_predictor_metadata_records_exploratory_status():
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

    bank = build_source_da_gain_bank_from_records([_record(split_role="source_fit")])
    predictor = DAGainRouterPredictor(ConstantPredictor(), bank, eta_surface=0.025)

    assert predictor.metadata["neural_training_epochs"] == 0
    assert predictor.metadata["target_eval_usage"] == "final_eval_only_no_selection"
    assert predictor.metadata["exploratory_after_us_r1_target_eval_seen"] is True


def test_eta_selection_uses_source_val_and_rejects_target_val():
    bank = build_source_da_gain_bank_from_records([_record(split_role="source_fit")])
    selection = select_eta_from_source_val(
        [_record(split_role="source_val")],
        bank,
        eta_grid=[0.0],
        require_corr_or_sign_gain=False,
    )

    assert selection["selection_source"] == "source_val_only"
    assert selection["target_eval_usage"] == "not_used_for_eta_selection"

    with pytest.raises(ValueError, match="source_val"):
        select_eta_from_source_val([_record(split_role="target_val")], bank, eta_grid=[0.0])
