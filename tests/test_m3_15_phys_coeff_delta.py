from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hydroda.evaluation.m3_15_phys_coeff_delta import (
    M3_15_METHOD_ID,
    M3_15_SELECTION_SCHEMA,
    apply_m3_15_interpolation,
    select_eta_from_source_val,
    source_gate_report_from_selection,
    validate_source_gate_for_target_eval,
)


def _record(
    *,
    split_role: str = "source_val",
    sample_region_id: str = "US-R2",
    season: str = "DJF",
    truth_s: float = 1.0,
    truth_r: float = 1.0,
    m3_s: float = 0.6,
    m3_r: float = 0.6,
    phys_s: float = 0.99,
    phys_r: float = 0.6,
) -> dict[str, object]:
    shape = (2, 2)
    inc_s = np.full(shape, truth_s, dtype=np.float32)
    inc_r = np.full(shape, truth_r, dtype=np.float32)
    forecast_s = np.zeros(shape, dtype=np.float32)
    forecast_r = np.zeros(shape, dtype=np.float32)
    return {
        "split_role": split_role,
        "sample_idx": 0,
        "query_time_index": 10,
        "query_date": "2022-01-01",
        "month": 1,
        "season": season,
        "target_region_id": "US-R1",
        "sample_region_id": sample_region_id,
        "active_region_ids": [sample_region_id],
        "adaptation_setting": "zero_shot_context",
        "K": 0,
        "seed": 0,
        "forecast_surface": forecast_s,
        "forecast_rootzone": forecast_r,
        "analysis_surface": forecast_s + inc_s,
        "analysis_rootzone": forecast_r + inc_r,
        "increment_surface": inc_s,
        "increment_rootzone": inc_r,
        "metric_mask": np.ones(shape, dtype=np.float32),
        "region_mask": np.ones(shape, dtype=np.float32),
        "latitude_weight": np.ones(shape, dtype=np.float32),
        "pred_m3_1_increment_surface": np.full(shape, m3_s, dtype=np.float32),
        "pred_m3_1_increment_rootzone": np.full(shape, m3_r, dtype=np.float32),
        "pred_phys_coeff_increment_surface": np.full(shape, phys_s, dtype=np.float32),
        "pred_phys_coeff_increment_rootzone": np.full(shape, phys_r, dtype=np.float32),
    }


def test_m3_15_eta_zero_is_exact_m3_1_identity():
    sample = _record()
    m3_1 = {
        "pred_increment_surface": np.full((2, 2), 0.6, dtype=np.float32),
        "pred_increment_rootzone": np.full((2, 2), 0.7, dtype=np.float32),
    }
    phys = {
        "pred_increment_surface": np.full((2, 2), 0.9, dtype=np.float32),
        "pred_increment_rootzone": np.full((2, 2), 0.2, dtype=np.float32),
    }

    routed = apply_m3_15_interpolation(sample, m3_1, phys, eta_surface=0.0, eta_rootzone=0.0)

    np.testing.assert_array_equal(routed["pred_increment_surface"], m3_1["pred_increment_surface"])
    np.testing.assert_array_equal(routed["pred_increment_rootzone"], m3_1["pred_increment_rootzone"])
    assert routed["m3_15_summary"]["eta_zero_identity"] is True


def test_m3_15_eta_selection_uses_source_val_and_passes_strict_gate():
    records = [
        _record(sample_region_id="US-R2", season="DJF"),
        _record(sample_region_id="US-R3", season="MAM"),
    ]

    selection = select_eta_from_source_val(
        records,
        eta_grid=[0.0, 1.0],
        anchor_dual_cvar=0.0,
        min_dual_cvar_delta=0.001,
    )

    assert selection["schema_version"] == M3_15_SELECTION_SCHEMA
    assert selection["method_id"] == M3_15_METHOD_ID
    assert selection["selection_source"] == "source_val_only"
    assert selection["selected_eta_surface"] == pytest.approx(1.0)
    assert selection["selected_eta_rootzone"] == pytest.approx(0.0)
    assert selection["source_gate_pass"] is True
    assert selection["identity_diagnostic"] is False
    validate_source_gate_for_target_eval(selection)
    gate = source_gate_report_from_selection(selection, target_region="US-R1", K=0, seed=0)
    assert gate["target_eval_allowed"] is True


def test_m3_15_source_gate_refuses_identity_and_target_records():
    identity = select_eta_from_source_val([_record()], eta_grid=[0.0], anchor_dual_cvar=0.0)

    assert identity["selected_eta_surface"] == 0.0
    assert identity["selected_eta_rootzone"] == 0.0
    assert identity["identity_diagnostic"] is True
    with pytest.raises(ValueError, match="positive eta"):
        validate_source_gate_for_target_eval(identity)

    with pytest.raises(ValueError, match="source_val"):
        select_eta_from_source_val([_record(split_role="target_eval")], eta_grid=[0.0, 1.0])

    malformed = _record()
    malformed["adaptation_setting"] = "target_full_train"
    with pytest.raises(ValueError, match="target-side"):
        select_eta_from_source_val([malformed], eta_grid=[0.0, 1.0])


def test_m3_15_wrapper_refuses_non_m3_1_warm_start(tmp_path: Path):
    env = {
        **os.environ,
        "ABLATION_ID": "M3_15_m31_anchored_source_safe_phys_coeff_delta",
        "RESUME_FROM_M3_1_BEST": "0",
        "DATASET_BACKEND": "netcdf",
    }

    result = subprocess.run(
        ["bash", "run/phase4_hyperda_staged_ablation.sh", "auto", "US-R1", "0", "0", "--dry-run"],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "requires RESUME_FROM_M3_1_BEST=1" in result.stderr


def test_m3_15_wrapper_dry_run_uses_m31_anchor_and_coeff_delta_only(tmp_path: Path):
    fake_m3_1 = tmp_path / "m3_1.pt"
    fake_m3_1.write_text("stub", encoding="utf-8")
    env = {
        **os.environ,
        "ABLATION_ID": "M3_15_m31_anchored_source_safe_phys_coeff_delta",
        "M3_15_INIT_FROM_M3_1_CHECKPOINT": str(fake_m3_1),
        "DATASET_BACKEND": "netcdf",
        "TIMESTAMP": "20260101_000000",
    }

    result = subprocess.run(
        ["bash", "run/phase4_hyperda_staged_ablation.sh", "auto", "US-R1", "0", "0", "--dry-run"],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = result.stdout
    assert "ablation_id=M3_15_m31_anchored_source_safe_phys_coeff_delta" in stdout
    assert "checkpoint_start=M3_1_best_checkpoint" in stdout
    assert "source_base_checkpoint_usage=not_loaded_for_m3_15" in stdout
    assert "trainable_scope=phys_coeff_delta_only" in stdout
    assert "trainable_modules=phys_operator_residual_only" in stdout
    assert "phys_consistency_usage=diagnostic_only_no_source_fit_sign_loss" in stdout
    assert "hyper_phys_consistency_regularization_weight=0.0" in stdout
    assert "eta_grid=0,0.1,0.25,0.5,1.0" in stdout
    assert "eta_zero_contract=exact_M3_1_prediction_identity" in stdout
    assert "final_output_residual_allowed=false" in stdout
    assert "--phys_context_source raw_input_side_formula_gain" in stdout
    assert "--trainable_scope phys_coeff_delta_only" in stdout
    assert f"--init_from_prompt_checkpoint {fake_m3_1}" in stdout
    assert "--init_from_source_base_checkpoint" not in stdout
