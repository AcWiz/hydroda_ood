import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydroda.evaluation.der_router import (
    DualExpertRouterPredictor,
    build_router_config,
    load_router_config,
    select_variable_experts,
    validate_eval_uses_router_config,
)

from scripts.eval import evaluate_der_router


class ConstantPredictor:
    def __init__(self, surface: float, rootzone: float, method_name: str = "constant"):
        self.surface = np.float32(surface)
        self.rootzone = np.float32(rootzone)
        self.method_name = method_name

    def predict(self, sample):
        shape = sample["forecast_surface"].shape
        inc_s = np.full(shape, self.surface, dtype=np.float32)
        inc_r = np.full(shape, self.rootzone, dtype=np.float32)
        return {
            "pred_increment_surface": inc_s,
            "pred_increment_rootzone": inc_r,
            "pred_analysis_surface": sample["forecast_surface"] + inc_s,
            "pred_analysis_rootzone": sample["forecast_rootzone"] + inc_r,
        }


def _sample():
    return {
        "forecast_surface": np.full((2, 2), 10.0, dtype=np.float32),
        "forecast_rootzone": np.full((2, 2), 20.0, dtype=np.float32),
    }


def test_dual_expert_predictor_routes_variables_and_recomputes_analysis():
    surface_expert = ConstantPredictor(surface=1.5, rootzone=99.0, method_name="surface")
    rootzone_expert = ConstantPredictor(surface=88.0, rootzone=-2.0, method_name="rootzone")
    predictor = DualExpertRouterPredictor(
        surface_expert=surface_expert,
        rootzone_expert=rootzone_expert,
        surface_metadata={"checkpoint": "phase4.pt", "predictor_type": "source_only"},
        rootzone_metadata={"checkpoint": "runA.pt", "predictor_type": "hyperda_target_adapt"},
    )

    pred = predictor.predict(_sample())

    assert np.allclose(pred["pred_increment_surface"], 1.5)
    assert np.allclose(pred["pred_increment_rootzone"], -2.0)
    assert np.allclose(pred["pred_analysis_surface"], 11.5)
    assert np.allclose(pred["pred_analysis_rootzone"], 18.0)
    assert pred["der_surface_expert_method"] == "surface"
    assert pred["der_rootzone_expert_method"] == "rootzone"


def test_select_variable_experts_uses_target_val_variable_wrmse_only():
    rows = [
        {
            "candidate_id": "phase4_global",
            "split_role": "target_val",
            "variable": "surface",
            "metric": "increment_rmse_latw",
            "value": 0.10,
        },
        {
            "candidate_id": "run_a",
            "split_role": "target_val",
            "variable": "surface",
            "metric": "increment_rmse_latw",
            "value": 0.30,
        },
        {
            "candidate_id": "phase4_global",
            "split_role": "target_val",
            "variable": "rootzone",
            "metric": "increment_rmse_latw",
            "value": 0.40,
        },
        {
            "candidate_id": "run_a",
            "split_role": "target_val",
            "variable": "rootzone",
            "metric": "increment_rmse_latw",
            "value": 0.05,
        },
        {
            "candidate_id": "run_a",
            "split_role": "target_eval",
            "variable": "surface",
            "metric": "increment_rmse_latw",
            "value": 0.01,
        },
    ]

    selection = select_variable_experts(pd.DataFrame(rows), metric="increment_rmse_latw")

    assert selection["surface"]["candidate_id"] == "phase4_global"
    assert selection["surface"]["target_val_metric_value"] == 0.10
    assert selection["rootzone"]["candidate_id"] == "run_a"
    assert selection["rootzone"]["target_val_metric_value"] == 0.05


def test_build_router_config_records_no_leakage_metadata(tmp_path):
    candidates = [
        {"candidate_id": "phase4_global", "checkpoint": "/ckpts/global.pt", "predictor_type": "source_only"},
        {"candidate_id": "run_a", "checkpoint": "/ckpts/runA.pt", "predictor_type": "hyperda_target_adapt"},
    ]
    selection = {
        "surface": {
            "candidate_id": "phase4_global",
            "target_val_metric": "increment_rmse_latw",
            "target_val_metric_value": 0.10,
        },
        "rootzone": {
            "candidate_id": "run_a",
            "target_val_metric": "increment_rmse_latw",
            "target_val_metric_value": 0.05,
        },
    }

    config = build_router_config(
        candidates=candidates,
        selection=selection,
        target_region="US-R1",
        adaptation_setting="target_full_train",
        seed=0,
        split_manifest_path="artifacts/splits/US_loro_target_train_splits.json",
        split_manifest_sha256="splithash",
        target_val_dates_hash="valhash",
        target_train_dates_hash="trainhash",
        target_eval_dates_hash="evalhash",
    )
    path = tmp_path / "router_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_router_config(path)

    assert loaded["method"] == "HydroDA-DER"
    assert loaded["router_selection_source"] == "target_val_2022"
    assert loaded["no_leakage_declaration"]["target_eval_used_for_selection"] is False
    assert loaded["selected_experts"]["surface"]["checkpoint"] == "/ckpts/global.pt"
    assert loaded["selected_experts"]["rootzone"]["checkpoint"] == "/ckpts/runA.pt"
    assert loaded["split_manifest_sha256"] == "splithash"
    assert loaded["target_val_dates_hash"] == "valhash"


def test_target_eval_requires_existing_router_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="router_config.json"):
        validate_eval_uses_router_config(
            split_type="target_eval",
            router_config_path=tmp_path / "missing_router_config.json",
        )

    config_path = tmp_path / "router_config.json"
    config_path.write_text(
        json.dumps(
            {
                "method": "HydroDA-DER",
                "router_selection_source": "target_val_2022",
                "selected_experts": {
                    "surface": {"checkpoint": "surface.pt", "predictor_type": "source_only"},
                    "rootzone": {"checkpoint": "root.pt", "predictor_type": "hyperda_target_adapt"},
                },
            }
        ),
        encoding="utf-8",
    )

    assert validate_eval_uses_router_config(
        split_type="target_eval",
        router_config_path=config_path,
    )["router_selection_source"] == "target_val_2022"


class TinyEvalDataset:
    def __init__(self, split_type: str):
        self.split_type = split_type
        self._split_entry = {
            "target_train_dates_hash": "trainhash",
            "target_val_dates_hash": "valhash",
            "target_eval_dates_hash": "evalhash",
        }
        if split_type == "target_val":
            increment_surface = np.full((2, 2), 1.2, dtype=np.float32)
            increment_rootzone = np.full((2, 2), 4.2, dtype=np.float32)
            date = "2022-01-01"
        else:
            increment_surface = np.full((2, 2), 1.2, dtype=np.float32)
            increment_rootzone = np.full((2, 2), 4.2, dtype=np.float32)
            date = "2023-01-01"
        forecast_surface = np.full((2, 2), 10.0, dtype=np.float32)
        forecast_rootzone = np.full((2, 2), 20.0, dtype=np.float32)
        self.sample = {
            "x": np.zeros((12, 2, 2), dtype=np.float32),
            "forecast_surface": forecast_surface,
            "forecast_rootzone": forecast_rootzone,
            "analysis_surface": forecast_surface + increment_surface,
            "analysis_rootzone": forecast_rootzone + increment_rootzone,
            "increment_surface": increment_surface,
            "increment_rootzone": increment_rootzone,
            "metric_mask": np.ones((2, 2), dtype=np.float32),
            "latitude_weight": np.ones((2, 2), dtype=np.float32),
            "date_str": date,
            "time_index": 0,
            "month": 1,
            "season": "DJF",
            "country_id": "US",
            "target_region_id": "US-R1",
            "sample_region_id": "US-R1",
            "active_region_ids": ["US-R1"],
            "adaptation_setting": "target_full_train",
            "K": None,
            "seed": 0,
            "target_train_dates_hash": "trainhash",
            "target_val_dates_hash": "valhash",
            "target_eval_dates_hash": "evalhash",
            "split_manifest_sha256": "splithash",
        }

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sample

    def close(self):
        return None


def _patch_tiny_cli(monkeypatch):
    def dataset_factory(**kwargs):
        return TinyEvalDataset(split_type=kwargs["split_type"])

    def predictor_factory(*, checkpoint, predictor_type, device="cpu", target_region=None):
        if "surface" in str(checkpoint):
            return ConstantPredictor(surface=1.0, rootzone=9.0, method_name="surface_expert")
        if "rootzone" in str(checkpoint):
            return ConstantPredictor(surface=8.0, rootzone=4.0, method_name="rootzone_expert")
        raise AssertionError(f"unexpected checkpoint {checkpoint}")

    monkeypatch.setattr(evaluate_der_router, "HydroDADataset", dataset_factory)
    monkeypatch.setattr(evaluate_der_router, "create_predictor", predictor_factory)
    monkeypatch.setattr(evaluate_der_router, "compute_sha256", lambda path: "splithash")


def test_select_then_eval_mock_flow_writes_router_config_and_metadata(tmp_path, monkeypatch):
    _patch_tiny_cli(monkeypatch)
    router_config_path = tmp_path / "router_config.json"

    select_summary = evaluate_der_router.run_select(
        candidates=[
            {"candidate_id": "phase4_global", "checkpoint": "surface.pt", "predictor_type": "source_only"},
            {"candidate_id": "run_a", "checkpoint": "rootzone.pt", "predictor_type": "hyperda_target_adapt"},
        ],
        target_region="US-R1",
        adaptation_setting="target_full_train",
        seed=0,
        output_dir=tmp_path / "select",
        router_config_path=router_config_path,
        device="cpu",
        max_samples=None,
        da_nc_path="DA.nc",
        region_masks_nc="masks.nc",
        splits_json="splits.json",
        freeze_manifest="freeze.json",
    )

    assert router_config_path.exists()
    config = json.loads(router_config_path.read_text(encoding="utf-8"))
    assert config["selected_experts"]["surface"]["candidate_id"] == "phase4_global"
    assert config["selected_experts"]["rootzone"]["candidate_id"] == "run_a"
    assert select_summary["router_selection_source"] == "target_val_2022"

    eval_summary = evaluate_der_router.run_eval(
        router_config_path=router_config_path,
        target_region="US-R1",
        adaptation_setting="target_full_train",
        seed=0,
        split_type="target_eval",
        output_dir=tmp_path / "eval",
        device="cpu",
        max_samples=None,
        da_nc_path="DA.nc",
        region_masks_nc="masks.nc",
        splits_json="splits.json",
        freeze_manifest="freeze.json",
    )

    assert eval_summary["method"] == "hydroda_der_variable_wise_dual_expert_router"
    assert eval_summary["router_selection_source"] == "target_val_2022"
    assert eval_summary["no_leakage_declaration"]["target_eval_used_for_selection"] is False
    assert eval_summary["selected_experts"]["surface"]["checkpoint"] == "surface.pt"
    assert eval_summary["selected_experts"]["rootzone"]["checkpoint"] == "rootzone.pt"
    assert (tmp_path / "eval" / "metrics_long.csv").exists()


def test_run_eval_refuses_target_eval_without_router_config(tmp_path, monkeypatch):
    _patch_tiny_cli(monkeypatch)

    with pytest.raises(FileNotFoundError, match="router_config.json"):
        evaluate_der_router.run_eval(
            router_config_path=tmp_path / "missing.json",
            target_region="US-R1",
            adaptation_setting="target_full_train",
            seed=0,
            split_type="target_eval",
            output_dir=tmp_path / "eval",
            device="cpu",
            max_samples=None,
            da_nc_path="DA.nc",
            region_masks_nc="masks.nc",
            splits_json="splits.json",
            freeze_manifest="freeze.json",
        )


def test_select_cli_accepts_named_dual_expert_checkpoint_flags():
    parser = evaluate_der_router.build_parser()
    args = parser.parse_args(
        [
            "select",
            "--target_region",
            "US-R1",
            "--output_dir",
            "out",
            "--router_config",
            "router_config.json",
            "--surface_checkpoint",
            "phase4_global.pt",
            "--surface_predictor_type",
            "source_only",
            "--rootzone_checkpoint",
            "run_a.pt",
            "--rootzone_predictor_type",
            "hyperda_target_adapt",
        ]
    )

    candidates = evaluate_der_router.candidates_from_args(args)

    assert candidates == [
        {
            "candidate_id": "surface_expert",
            "predictor_type": "source_only",
            "checkpoint": "phase4_global.pt",
        },
        {
            "candidate_id": "rootzone_expert",
            "predictor_type": "hyperda_target_adapt",
            "checkpoint": "run_a.pt",
        },
    ]


def test_hydroda_der_run_script_declares_two_step_no_leakage_protocol():
    text = Path("run/phase5_hydroda_der.sh").read_text()

    assert "target_val=2022 router selection only" in text
    assert "target_eval=2023-2025 final evaluation only" in text
    assert "scripts/eval/evaluate_der_router.py select" in text
    assert "scripts/eval/evaluate_der_router.py eval" in text
    assert "--surface_checkpoint" in text
    assert "--rootzone_checkpoint" in text
    assert "--router_config" in text
    assert "target_eval_used_for_selection=false" in text


def test_hydroda_der_run_script_autodetects_hydro_msr_rootzone_checkpoint():
    text = Path("run/phase5_hydroda_der.sh").read_text()

    assert "phase5_hydro_msr_${TARGET_REGION}_s${SEED}" in text
    assert "checkpoint_best_target_val_rootzone_wrmse.pt" in text
