from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydroda.training.calibration import calibrate_residual_gain_region_aware


def _sample(pred_inc: float, true_inc: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    return [
        (
            np.full((2, 2), pred_inc, dtype=np.float32),
            np.full((2, 2), true_inc, dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
        )
    ]


def test_region_aware_alpha_selection_can_use_dual_variable_cvar_objective():
    samples_s_by_region = {
        "US-R2": _sample(1.00, 1.0),
        "US-R3": _sample(1.00, 1.0),
        "US-R4": _sample(1.00, 1.0),
        "US-R5": _sample(2.20, 1.0),
    }
    samples_r_by_region = {
        "US-R2": _sample(1.00, 1.0),
        "US-R3": _sample(1.00, 1.0),
        "US-R4": _sample(1.00, 1.0),
        "US-R5": _sample(1.00, 1.0),
    }

    transfer = calibrate_residual_gain_region_aware(
        samples_s_by_region=samples_s_by_region,
        samples_r_by_region=samples_r_by_region,
        alpha_grid=[0.0, 1.0],
        alpha_selection_objective="transfer_safe_score",
    )
    dual = calibrate_residual_gain_region_aware(
        samples_s_by_region=samples_s_by_region,
        samples_r_by_region=samples_r_by_region,
        alpha_grid=[0.0, 1.0],
        alpha_selection_objective="dual_variable_cvar_safe_score",
    )

    assert (transfer["best_alpha_surface"], transfer["best_alpha_rootzone"]) == (1.0, 1.0)
    assert transfer["alpha_selection_objective"] == "transfer_safe_score"
    assert transfer["selection_score"] == pytest.approx(
        transfer["selected_trace"]["transfer_safe_score"],
    )

    assert (dual["best_alpha_surface"], dual["best_alpha_rootzone"]) == (0.0, 1.0)
    assert dual["alpha_selection_objective"] == "dual_variable_cvar_safe_score"
    assert dual["selection_score"] == pytest.approx(
        dual["selected_trace"]["dual_variable_cvar_safe_score"],
    )
    assert dual["dual_variable_cvar_score"] == pytest.approx(dual["selection_score"])
    assert dual["selected_surface_region_skills"] == {
        "US-R2": pytest.approx(0.0),
        "US-R3": pytest.approx(0.0),
        "US-R4": pytest.approx(0.0),
        "US-R5": pytest.approx(0.0),
    }
    assert dual["selected_rootzone_region_skills"] == {
        "US-R2": pytest.approx(1.0),
        "US-R3": pytest.approx(1.0),
        "US-R4": pytest.approx(1.0),
        "US-R5": pytest.approx(1.0),
    }


def test_region_aware_alpha_selection_rejects_unknown_objective():
    with pytest.raises(ValueError, match="alpha_selection_objective"):
        calibrate_residual_gain_region_aware(
            samples_s_by_region={"US-R2": _sample(1.0, 1.0)},
            samples_r_by_region={"US-R2": _sample(1.0, 1.0)},
            alpha_grid=[0.0, 1.0],
            alpha_selection_objective="target_eval_rmse",
        )
