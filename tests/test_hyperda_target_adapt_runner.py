from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hydroda.data import dataset as dataset_module
from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.prompt_encoder import RegionPromptEncoder


def _write_source_checkpoint(path: Path) -> None:
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        zero_raw_increment_init=True,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "config": {
                "model_type": "hyperda_basis_adapter",
                "width": 4,
                "prompt_dim": 8,
                "hyper_n_basis": 3,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "zero_raw_increment_init": True,
                "num_regions": 5,
                "ch_mean": [0.0] * 12,
                "ch_std": [1.0] * 12,
                "inc_mean": [0.0, 0.0],
                "inc_std": [1.0, 1.0],
                "source_regions": ["US-R2", "US-R3", "US-R4", "US-R5", "US-R6"],
                "source_region_global_indices": [1, 2, 3, 4, 5],
                "target_region": "US-R1",
                "adaptation_setting": "target_full_train",
            },
        },
        path,
    )


def test_dataset_contract_supports_target_val_split_key():
    assert dataset_module._SPLIT_TYPE_TO_DATES_KEY["target_val"] == "target_val_dates"
    assert dataset_module._DATES_KEY_FALLBACKS["target_val_dates"] == "source_val_dates"


def test_target_adaptation_parse_args_rejects_tensor_cache_backend(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_target_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hyperda_target_adapt.py",
            "--source_checkpoint",
            str(source_checkpoint),
            "--target_region",
            "US-R1",
            "--dataset_backend",
            "tensor_cache",
            "--tensor_cache_dir",
            str(tmp_path / "cache"),
            "--tensor_cache_max_years",
            "2",
        ],
    )

    with pytest.raises(SystemExit):
        runner.parse_args()


def test_target_adaptation_parse_args_rejects_removed_pigo_flag(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_target_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hyperda_target_adapt.py",
            "--source_checkpoint",
            str(source_checkpoint),
            "--target_region",
            "US-R1",
            "--enable_pigo",
        ],
    )

    with pytest.raises(SystemExit):
        runner.parse_args()


def test_load_source_checkpoint_for_target_adaptation_freezes_source_prior(tmp_path):
    from scripts.train.train_hyperda_target_adapt import load_source_checkpoint_for_target_adaptation

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)

    state = load_source_checkpoint_for_target_adaptation(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
    )

    trainable = state.model.target_trainable_parameter_names()
    assert trainable
    assert any(name.startswith("target_spatial_refine") for name in trainable)
    assert all(
        name.startswith("target_")
        or name.startswith("residual_gain")
        or "coefficient_residual" in name
        for name in trainable
    )
    assert state.prompt_encoder.training is False
    assert state.normalization["ch_mean"] == [0.0] * 12
    assert state.source_config["model_type"] == "hyperda_basis_adapter"


def test_load_source_checkpoint_for_target_adaptation_can_build_hydro_msr(tmp_path):
    from hydroda.models.target_adaptation import HydroMSROutputAdapter
    from scripts.train.train_hyperda_target_adapt import load_source_checkpoint_for_target_adaptation

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)

    state = load_source_checkpoint_for_target_adaptation(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_type="hydro_msr",
        enable_hydro_msr_da_film=True,
    )

    assert isinstance(state.model.target_spatial_refine, HydroMSROutputAdapter)
    assert state.model.target_spatial_refine_type == "hydro_msr"
    assert state.model.enable_hydro_msr_da_film is True
    assert any("target_spatial_refine" in name for name in state.model.target_trainable_parameter_names())


def test_load_source_checkpoint_for_target_adaptation_can_build_hydro_msr_gain(tmp_path):
    from hydroda.models.target_adaptation import HydroMSRGainOutputAdapter
    from scripts.train.train_hyperda_target_adapt import load_source_checkpoint_for_target_adaptation

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)

    state = load_source_checkpoint_for_target_adaptation(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_type="hydro_msr_gain",
    )

    assert isinstance(state.model.target_spatial_refine, HydroMSRGainOutputAdapter)
    assert state.model.target_spatial_refine_type == "hydro_msr_gain"
    assert any("target_spatial_refine.gain_mixer" in name for name in state.model.target_trainable_parameter_names())


def test_load_source_checkpoint_for_target_adaptation_can_build_hydro_msr_gain_lite(tmp_path):
    from hydroda.models.target_adaptation import HydroMSRGainLiteOutputAdapter
    from scripts.train.train_hyperda_target_adapt import load_source_checkpoint_for_target_adaptation

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)

    state = load_source_checkpoint_for_target_adaptation(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_type="hydro_msr_gain_lite",
        target_spatial_refine_gain_span=0.25,
    )

    assert isinstance(state.model.target_spatial_refine, HydroMSRGainLiteOutputAdapter)
    assert state.model.target_spatial_refine_type == "hydro_msr_gain_lite"
    assert state.model.target_spatial_refine_gain_span == 0.25
    assert state.model.target_spatial_refine.gain_mixer.gain_span == 0.25
    assert any("target_spatial_refine.gain_mixer" in name for name in state.model.target_trainable_parameter_names())


def test_load_source_checkpoint_for_target_adaptation_can_build_hydro_msr_rose(tmp_path):
    from hydroda.models.target_adaptation import HydroMSRROSEOutputAdapter
    from scripts.train.train_hyperda_target_adapt import load_source_checkpoint_for_target_adaptation

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)

    state = load_source_checkpoint_for_target_adaptation(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_type="hydro_msr_rose",
    )

    assert isinstance(state.model.target_spatial_refine, HydroMSRROSEOutputAdapter)
    assert state.model.target_spatial_refine_type == "hydro_msr_rose"
    assert any("target_spatial_refine.rose_encoder" in name for name in state.model.target_trainable_parameter_names())


def test_load_source_checkpoint_for_target_adaptation_has_no_pigo_path(tmp_path):
    from scripts.train.train_hyperda_target_adapt import load_source_checkpoint_for_target_adaptation

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)

    state = load_source_checkpoint_for_target_adaptation(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    trainable = state.model.target_trainable_parameter_names()
    assert not any("pigo" in name.lower() for name in trainable)
    assert not hasattr(state.model, "target_pigo")


def test_target_adaptation_checkpoint_metadata_records_protocol(tmp_path):
    from scripts.train.train_hyperda_target_adapt import (
        TargetAdaptationState,
        save_target_adaptation_checkpoint,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    source = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_type="hydro_msr",
        enable_hydro_msr_da_film=True,
    )
    model.freeze_source_prior_for_target_adaptation()
    prompt_encoder = RegionPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8).eval()
    state = TargetAdaptationState(
        model=model,
        prompt_encoder=prompt_encoder,
        source_checkpoint=source,
        source_config=source["config"],
        normalization={"ch_mean": [0.0] * 12, "ch_std": [1.0] * 12, "inc_mean": [0.0, 0.0], "inc_std": [1.0, 1.0]},
    )
    out = tmp_path / "adapted.pt"

    save_target_adaptation_checkpoint(
        path=out,
        state=state,
        optimizer_state_dict={"ok": True},
        epoch=2,
        tag="best_target_val",
        train_history=[{"epoch": 1, "target_train_loss": 0.3}],
        val_history=[{"epoch": 1, "target_val_loss": 0.4}],
        best_target_val_loss=0.4,
        best_epoch=1,
        config={
            "target_region": "US-R1",
            "adaptation_setting": "target_full_train",
            "source_checkpoint": str(ckpt_path),
            "split_manifest_path": "artifacts/splits/US_loro_target_train_splits.json",
            "split_manifest_sha256": "splitsha",
            "target_train_dates_hash": "trainhash",
            "target_val_dates_hash": "valhash",
            "target_eval_dates_hash": "evalhash",
            "adaptation_steps": 12,
            "enable_target_spatial_refine": True,
            "target_spatial_refine_hidden": 4,
            "target_spatial_refine_rootzone": False,
            "target_spatial_refine_input": "raw",
            "target_spatial_refine_type": "hydro_msr",
            "target_spatial_refine_gain_span": 0.25,
            "hydro_msr_hidden": 4,
            "enable_hydro_msr_da_film": True,
            "enable_da_regime_gain_mixer": False,
            "stage1_epochs": 10,
            "stage_schedule": "staged_global_then_spatial",
            "selection_rootzone_weight": 0.5,
            "selected_metric_name": "target_val_surface_wrmse_latw",
            "selected_metric_value": 0.0019,
        },
    )

    saved = torch.load(out, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    assert saved["tag"] == "best_target_val"
    assert saved["best_target_val_loss"] == 0.4
    assert cfg["target_train_period"] == "2015-2021"
    assert cfg["target_val_period"] == "2022"
    assert cfg["target_eval_period"] == "2023-2025"
    assert cfg["frozen_modules"] == ["theta0", "H_psi", "adapter_basis_bank", "prompt_encoder"]
    assert "target_latent" in cfg["trainable_modules"]
    assert "target_spatial_refine" in cfg["trainable_modules"]
    assert cfg["model_selection_source"] == "target_val_2022_preregistered_adaptation_selection"
    assert cfg["target_eval_usage"] == "final_eval_only_no_training_no_selection"
    assert cfg["target_spatial_refine_input"] == "raw"
    assert cfg["target_spatial_refine_type"] == "hydro_msr"
    assert cfg["target_spatial_refine_gain_span"] == 0.25
    assert cfg["hydro_msr_hidden"] == 4
    assert cfg["enable_hydro_msr_da_film"] is True
    assert cfg["enable_da_regime_gain_mixer"] is False
    assert cfg["stage1_epochs"] == 10
    assert cfg["stage_schedule"] == "staged_global_then_spatial"
    assert cfg["selection_rootzone_weight"] == 0.5
    assert cfg["selected_metric_name"] == "target_val_surface_wrmse_latw"
    assert cfg["selected_metric_value"] == 0.0019


def test_target_adaptation_train_and_eval_loop_smoke():
    from scripts.train.train_hyperda_target_adapt import (
        TargetAdaptationState,
        build_dataloader,
        evaluate_loss,
        train_one_epoch,
    )
    from hydroda.training.losses import WeightedMaskedHuberLoss

    class TinyTargetDataset:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            rng = np.random.default_rng(idx)
            h, w = 16, 16
            x = rng.normal(size=(12, h, w)).astype(np.float32)
            return {
                "x": x,
                "increment_surface": (0.01 * x[0]).astype(np.float32),
                "increment_rootzone": (0.01 * x[1]).astype(np.float32),
                "forecast_surface": x[0].astype(np.float32),
                "forecast_rootzone": x[1].astype(np.float32),
                "loss_mask": np.ones((h, w), dtype=np.float32),
                "latitude_weight": np.ones((h, w), dtype=np.float32),
                "month": 1 + idx,
            }

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
    )
    model.freeze_source_prior_for_target_adaptation()
    prompt_encoder = RegionPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8).eval()
    for param in prompt_encoder.parameters():
        param.requires_grad_(False)
    state = TargetAdaptationState(
        model=model,
        prompt_encoder=prompt_encoder,
        source_checkpoint={},
        source_config={
            "model_type": "hyperda_basis_adapter",
            "source_region_global_indices": [1, 2, 3, 4, 5],
        },
        normalization={"ch_mean": [0.0] * 12, "ch_std": [1.0] * 12, "inc_mean": [0.0, 0.0], "inc_std": [1.0, 1.0]},
    )
    loader = build_dataloader(TinyTargetDataset(), batch_size=2, num_workers=0, shuffle=False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss_fn = WeightedMaskedHuberLoss(delta=0.01, use_lat_weight=True)

    train_metrics = train_one_epoch(
        state=state,
        loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        target_region="US-R1",
        loss_fn=loss_fn,
        normalize_increment=True,
        grad_clip=1.0,
        lambda_prior=1e-4,
        lambda_latent=1e-4,
        lambda_gain=1e-3,
        lambda_gain_smooth=1e-3,
    )
    val_metrics = evaluate_loss(
        state=state,
        loader=loader,
        device=torch.device("cpu"),
        target_region="US-R1",
        loss_fn=loss_fn,
        normalize_increment=True,
    )

    assert train_metrics["total_loss"] > 0
    assert torch.isfinite(torch.tensor(val_metrics["target_val_loss"]))
    assert torch.isfinite(torch.tensor(val_metrics["target_val_surface_wrmse_latw"]))
    assert torch.isfinite(torch.tensor(val_metrics["target_val_rootzone_wrmse_latw"]))


def test_prompt_predictor_loads_phase5_target_adapt_checkpoint(tmp_path):
    from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        zero_raw_increment_init=True,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8)
    ckpt_path = tmp_path / "phase5_target_adapt.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "config": {
                "model_type": "hyperda_basis_adapter_target_adapt",
                "method": "hyperda_target_adapt",
                "target_latent_dim": 4,
                "enable_target_spatial_refine": True,
                "target_spatial_refine_hidden": 4,
                "target_spatial_refine_rootzone": False,
                "target_spatial_refine_input": "raw",
                "target_region": "US-R1",
                "adaptation_setting": "target_full_train",
            },
            "source_checkpoint_config": {
                "model_type": "hyperda_basis_adapter",
                "width": 4,
                "prompt_dim": 8,
                "hyper_n_basis": 3,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "zero_raw_increment_init": True,
                "num_regions": 5,
                "ch_mean": [0.0] * 12,
                "ch_std": [1.0] * 12,
                "inc_mean": [0.0, 0.0],
                "inc_std": [1.0, 1.0],
                "source_region_global_indices": [1, 2, 3, 4, 5],
                "target_region": "US-R1",
                "adaptation_setting": "target_full_train",
            },
        },
        ckpt_path,
    )

    predictor = PromptConditionedBackbonePredictor(
        checkpoint_path=str(ckpt_path),
        device="cpu",
        target_region="US-R1",
    )

    sample = {
        "x": np.zeros((12, 16, 16), dtype=np.float32),
        "forecast_surface": np.ones((16, 16), dtype=np.float32),
        "forecast_rootzone": np.ones((16, 16), dtype=np.float32),
        "target_region_id": "US-R1",
        "split_role": "target_eval",
        "month": 7,
    }
    pred = predictor.predict(sample)

    assert predictor.model_type == "hyperda_basis_adapter_target_adapt"
    assert predictor.method_name == "hyperda_target_adapt"
    assert predictor.model.target_spatial_refine is not None
    assert predictor.model.target_spatial_refine_input == "raw"
    assert pred["pred_increment_surface"].shape == (16, 16)
    np.testing.assert_allclose(pred["pred_analysis_surface"], sample["forecast_surface"])


def test_prompt_predictor_loads_hydro_msr_target_adapt_checkpoint(tmp_path):
    from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor
    from hydroda.models.target_adaptation import HydroMSROutputAdapter

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_input="raw",
        target_spatial_refine_type="hydro_msr",
        enable_hydro_msr_da_film=True,
        zero_raw_increment_init=True,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8)
    ckpt_path = tmp_path / "phase5_hydro_msr.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "config": {
                "model_type": "hyperda_basis_adapter_target_adapt",
                "method": "hyperda_target_adapt",
                "target_latent_dim": 4,
                "enable_target_spatial_refine": True,
                "target_spatial_refine_hidden": 4,
                "target_spatial_refine_rootzone": True,
                "target_spatial_refine_input": "raw",
                "target_spatial_refine_type": "hydro_msr",
                "hydro_msr_hidden": 4,
                "enable_hydro_msr_da_film": True,
                "target_region": "US-R1",
                "adaptation_setting": "target_full_train",
            },
            "source_checkpoint_config": {
                "model_type": "hyperda_basis_adapter",
                "width": 4,
                "prompt_dim": 8,
                "hyper_n_basis": 3,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "zero_raw_increment_init": True,
                "num_regions": 5,
                "ch_mean": [0.0] * 12,
                "ch_std": [1.0] * 12,
                "inc_mean": [0.0, 0.0],
                "inc_std": [1.0, 1.0],
                "source_region_global_indices": [1, 2, 3, 4, 5],
                "target_region": "US-R1",
                "adaptation_setting": "target_full_train",
            },
        },
        ckpt_path,
    )

    predictor = PromptConditionedBackbonePredictor(
        checkpoint_path=str(ckpt_path),
        device="cpu",
        target_region="US-R1",
    )

    sample = {
        "x": np.zeros((12, 16, 16), dtype=np.float32),
        "forecast_surface": np.ones((16, 16), dtype=np.float32),
        "forecast_rootzone": np.ones((16, 16), dtype=np.float32),
        "target_region_id": "US-R1",
        "split_role": "target_eval",
        "month": 7,
    }
    pred = predictor.predict(sample)

    assert isinstance(predictor.model.target_spatial_refine, HydroMSROutputAdapter)
    assert predictor.model.target_spatial_refine_type == "hydro_msr"
    assert predictor.model.enable_hydro_msr_da_film is True
    np.testing.assert_allclose(pred["pred_analysis_surface"], sample["forecast_surface"])


def test_prompt_predictor_loads_hydro_msr_gain_target_adapt_checkpoint(tmp_path):
    from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor
    from hydroda.models.target_adaptation import HydroMSRGainOutputAdapter

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_input="raw",
        target_spatial_refine_type="hydro_msr_gain",
        zero_raw_increment_init=True,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8)
    ckpt_path = tmp_path / "phase5_hydro_msr_gain.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "config": {
                "model_type": "hyperda_basis_adapter_target_adapt",
                "method": "hyperda_target_adapt",
                "target_latent_dim": 4,
                "enable_target_spatial_refine": True,
                "target_spatial_refine_hidden": 4,
                "target_spatial_refine_rootzone": True,
                "target_spatial_refine_input": "raw",
                "target_spatial_refine_type": "hydro_msr_gain",
                "hydro_msr_hidden": 4,
                "enable_da_regime_gain_mixer": True,
                "target_region": "US-R1",
                "adaptation_setting": "target_full_train",
            },
            "source_checkpoint_config": {
                "model_type": "hyperda_basis_adapter",
                "width": 4,
                "prompt_dim": 8,
                "hyper_n_basis": 3,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "zero_raw_increment_init": True,
                "num_regions": 5,
                "ch_mean": [0.0] * 12,
                "ch_std": [1.0] * 12,
                "inc_mean": [0.0, 0.0],
                "inc_std": [1.0, 1.0],
                "source_region_global_indices": [1, 2, 3, 4, 5],
                "target_region": "US-R1",
                "adaptation_setting": "target_full_train",
            },
        },
        ckpt_path,
    )

    predictor = PromptConditionedBackbonePredictor(
        checkpoint_path=str(ckpt_path),
        device="cpu",
        target_region="US-R1",
    )

    sample = {
        "x": np.zeros((12, 16, 16), dtype=np.float32),
        "forecast_surface": np.ones((16, 16), dtype=np.float32),
        "forecast_rootzone": np.ones((16, 16), dtype=np.float32),
        "target_region_id": "US-R1",
        "split_role": "target_eval",
        "month": 7,
    }
    pred = predictor.predict(sample)

    assert isinstance(predictor.model.target_spatial_refine, HydroMSRGainOutputAdapter)
    assert predictor.model.target_spatial_refine_type == "hydro_msr_gain"
    np.testing.assert_allclose(pred["pred_analysis_surface"], sample["forecast_surface"])


def test_prompt_predictor_loads_hydro_msr_gain_lite_target_adapt_checkpoint(tmp_path):
    from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor
    from hydroda.models.target_adaptation import HydroMSRGainLiteOutputAdapter

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_input="raw",
        target_spatial_refine_type="hydro_msr_gain_lite",
        target_spatial_refine_gain_span=0.25,
        zero_raw_increment_init=True,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8)
    ckpt_path = tmp_path / "phase5_hydro_msr_gain_lite.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "config": {
                "model_type": "hyperda_basis_adapter_target_adapt",
                "method": "hyperda_target_adapt",
                "target_latent_dim": 4,
                "enable_target_spatial_refine": True,
                "target_spatial_refine_hidden": 4,
                "target_spatial_refine_rootzone": True,
                "target_spatial_refine_input": "raw",
                "target_spatial_refine_type": "hydro_msr_gain_lite",
                "target_spatial_refine_gain_span": 0.25,
                "hydro_msr_hidden": 4,
                "enable_da_regime_gain_mixer": True,
                "target_region": "US-R1",
                "adaptation_setting": "target_full_train",
            },
            "source_checkpoint_config": {
                "model_type": "hyperda_basis_adapter",
                "width": 4,
                "prompt_dim": 8,
                "hyper_n_basis": 3,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "zero_raw_increment_init": True,
                "num_regions": 5,
                "ch_mean": [0.0] * 12,
                "ch_std": [1.0] * 12,
                "inc_mean": [0.0, 0.0],
                "inc_std": [1.0, 1.0],
                "source_region_global_indices": [1, 2, 3, 4, 5],
                "target_region": "US-R1",
                "adaptation_setting": "target_full_train",
            },
        },
        ckpt_path,
    )

    predictor = PromptConditionedBackbonePredictor(
        checkpoint_path=str(ckpt_path),
        device="cpu",
        target_region="US-R1",
    )

    sample = {
        "x": np.zeros((12, 16, 16), dtype=np.float32),
        "forecast_surface": np.ones((16, 16), dtype=np.float32),
        "forecast_rootzone": np.ones((16, 16), dtype=np.float32),
        "target_region_id": "US-R1",
        "split_role": "target_eval",
        "month": 7,
    }
    pred = predictor.predict(sample)

    assert isinstance(predictor.model.target_spatial_refine, HydroMSRGainLiteOutputAdapter)
    assert predictor.model.target_spatial_refine_type == "hydro_msr_gain_lite"
    assert predictor.model.target_spatial_refine_gain_span == 0.25
    np.testing.assert_allclose(pred["pred_analysis_surface"], sample["forecast_surface"])


def test_prompt_predictor_loads_anchor_soup_target_adapt_checkpoint(tmp_path):
    from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_input="raw",
        target_spatial_refine_type="hydro_msr_gain_lite",
        target_spatial_refine_gain_span=0.20,
        zero_raw_increment_init=True,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8)
    ckpt_path = tmp_path / "checkpoint_best_target_val_anchor_soup.pt"
    torch.save(
        {
            "tag": "best_target_val_anchor_soup",
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "config": {
                "model_type": "hyperda_basis_adapter_target_adapt",
                "method": "hyperda_target_adapt",
                "target_latent_dim": 4,
                "enable_target_spatial_refine": True,
                "target_spatial_refine_hidden": 4,
                "target_spatial_refine_rootzone": True,
                "target_spatial_refine_input": "raw",
                "target_spatial_refine_type": "hydro_msr_gain_lite",
                "target_spatial_refine_gain_span": 0.20,
                "hydro_msr_hidden": 4,
                "enable_adapter_anchor_soup": True,
                "adapter_anchor_soup_selection_metric": "surface_val_wrmse_plus_0.25_rootzone_val_wrmse",
                "adapter_anchor_soup_no_leakage": "target_val_2022_only_no_target_eval_labels",
                "target_region": "US-R1",
                "adaptation_setting": "target_full_train",
            },
            "source_checkpoint_config": {
                "model_type": "hyperda_basis_adapter",
                "width": 4,
                "prompt_dim": 8,
                "hyper_n_basis": 3,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "zero_raw_increment_init": True,
                "num_regions": 5,
                "ch_mean": [0.0] * 12,
                "ch_std": [1.0] * 12,
                "inc_mean": [0.0, 0.0],
                "inc_std": [1.0, 1.0],
                "source_region_global_indices": [1, 2, 3, 4, 5],
                "target_region": "US-R1",
                "adaptation_setting": "target_full_train",
            },
        },
        ckpt_path,
    )

    predictor = PromptConditionedBackbonePredictor(
        checkpoint_path=str(ckpt_path),
        device="cpu",
        target_region="US-R1",
    )

    sample = {
        "x": np.zeros((12, 16, 16), dtype=np.float32),
        "forecast_surface": np.ones((16, 16), dtype=np.float32),
        "forecast_rootzone": np.ones((16, 16), dtype=np.float32),
        "target_region_id": "US-R1",
        "split_role": "target_eval",
        "month": 7,
    }
    pred = predictor.predict(sample)

    assert predictor.model_type == "hyperda_basis_adapter_target_adapt"
    assert predictor.method_name == "hyperda_target_adapt"
    assert predictor.model.target_spatial_refine_type == "hydro_msr_gain_lite"
    assert pred["pred_increment_surface"].shape == (16, 16)


def test_prompt_predictor_loads_hydro_msr_rose_target_adapt_checkpoint(tmp_path):
    from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor
    from hydroda.models.target_adaptation import HydroMSRROSEOutputAdapter

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_input="raw",
        target_spatial_refine_type="hydro_msr_rose",
        zero_raw_increment_init=True,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8)
    ckpt_path = tmp_path / "phase5_hydro_msr_rose.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "config": {
                "model_type": "hyperda_basis_adapter_target_adapt",
                "method": "hyperda_target_adapt",
                "target_latent_dim": 4,
                "enable_target_spatial_refine": True,
                "target_spatial_refine_hidden": 4,
                "target_spatial_refine_rootzone": True,
                "target_spatial_refine_input": "raw",
                "target_spatial_refine_type": "hydro_msr_rose",
                "hydro_msr_hidden": 4,
                "target_region": "US-R1",
                "adaptation_setting": "target_full_train",
            },
            "source_checkpoint_config": {
                "model_type": "hyperda_basis_adapter",
                "width": 4,
                "prompt_dim": 8,
                "hyper_n_basis": 3,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "zero_raw_increment_init": True,
                "num_regions": 5,
                "ch_mean": [0.0] * 12,
                "ch_std": [1.0] * 12,
                "inc_mean": [0.0, 0.0],
                "inc_std": [1.0, 1.0],
                "source_region_global_indices": [1, 2, 3, 4, 5],
                "target_region": "US-R1",
                "adaptation_setting": "target_full_train",
            },
        },
        ckpt_path,
    )

    predictor = PromptConditionedBackbonePredictor(
        checkpoint_path=str(ckpt_path),
        device="cpu",
        target_region="US-R1",
    )

    sample = {
        "x": np.zeros((12, 16, 16), dtype=np.float32),
        "forecast_surface": np.ones((16, 16), dtype=np.float32),
        "forecast_rootzone": np.ones((16, 16), dtype=np.float32),
        "target_region_id": "US-R1",
        "split_role": "target_eval",
        "month": 7,
    }
    sample["x"][7:9] = 0.5
    pred = predictor.predict(sample)

    assert isinstance(predictor.model.target_spatial_refine, HydroMSRROSEOutputAdapter)
    assert predictor.model.target_spatial_refine_type == "hydro_msr_rose"
    np.testing.assert_allclose(pred["pred_analysis_surface"], sample["forecast_surface"])


def test_prompt_predictor_rejects_removed_pigo_checkpoint(tmp_path):
    from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        zero_raw_increment_init=True,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8)
    ckpt_path = tmp_path / "phase5_removed_pigo.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "config": {
                "model_type": "hyperda_basis_adapter_target_adapt",
                "method": "hyperda_target_adapt",
                "target_latent_dim": 4,
                "enable_pigo": True,
                "target_region": "US-R1",
                "adaptation_setting": "target_full_train",
            },
            "source_checkpoint_config": {
                "model_type": "hyperda_basis_adapter",
                "width": 4,
                "prompt_dim": 8,
                "hyper_n_basis": 3,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "zero_raw_increment_init": True,
                "num_regions": 5,
                "ch_mean": [0.0] * 12,
                "ch_std": [1.0] * 12,
                "inc_mean": [0.0, 0.0],
                "inc_std": [1.0, 1.0],
                "source_region_global_indices": [1, 2, 3, 4, 5],
            },
        },
        ckpt_path,
    )

    with pytest.raises(ValueError, match="PIGO target-adaptation checkpoints are no longer supported"):
        PromptConditionedBackbonePredictor(
            checkpoint_path=str(ckpt_path),
            device="cpu",
            target_region="US-R1",
        )


def test_phase5_inference_script_uses_target_adapt_predictor():
    script = Path("run/phase5_hyperda_target_adapt_inference.sh")

    text = script.read_text()

    assert "checkpoint_best_target_val_surface_wrmse.pt" in text
    assert "checkpoint_best_target_val_loss.pt" in text
    assert "SURFACE_CHECKPOINT" in text
    assert "--predictor_type hyperda_target_adapt" in text
    assert "--split_type target_eval" in text


def test_resume_checkpoint_restores_training_state(tmp_path):
    from scripts.train.train_hyperda_target_adapt import (
        TargetAdaptationState,
        restore_target_adaptation_resume,
        save_target_adaptation_checkpoint,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    source = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
    )
    model.freeze_source_prior_for_target_adaptation()
    prompt_encoder = RegionPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8).eval()
    state = TargetAdaptationState(
        model=model,
        prompt_encoder=prompt_encoder,
        source_checkpoint=source,
        source_config=source["config"],
        normalization={"ch_mean": [0.0] * 12, "ch_std": [1.0] * 12, "inc_mean": [0.0, 0.0], "inc_std": [1.0, 1.0]},
    )
    optimizer = torch.optim.AdamW([p for p in state.model.parameters() if p.requires_grad], lr=1e-3)
    for param in state.model.parameters():
        if param.requires_grad:
            param.data.fill_(0.25)
            break
    optimizer.zero_grad(set_to_none=True)
    loss = sum(p.sum() for p in state.model.parameters() if p.requires_grad)
    loss.backward()
    optimizer.step()
    resume_path = tmp_path / "run" / "checkpoints" / "last.pt"
    save_target_adaptation_checkpoint(
        path=resume_path,
        state=state,
        optimizer_state_dict=optimizer.state_dict(),
        epoch=19,
        tag="last",
        train_history=[{"epoch": 18, "target_train_loss": 0.5}],
        val_history=[{"epoch": 18, "target_val_loss": 0.4}],
        best_target_val_loss=0.4,
        best_epoch=18,
        config={
            "target_region": "US-R1",
            "adaptation_setting": "target_full_train",
            "source_checkpoint": str(ckpt_path),
            "target_latent_dim": 4,
        },
    )

    fresh = TargetAdaptationState(
        model=HyperAdapterConditionalResUNet(
            in_channels=12,
            out_channels=2,
            width=4,
            prompt_dim=8,
            hyper_n_basis=3,
            hyper_adapter_bottleneck=2,
            enable_target_adaptation=True,
            target_latent_dim=4,
        ),
        prompt_encoder=RegionPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8).eval(),
        source_checkpoint=source,
        source_config=source["config"],
        normalization=state.normalization,
    )
    fresh.model.freeze_source_prior_for_target_adaptation()
    fresh_optimizer = torch.optim.AdamW([p for p in fresh.model.parameters() if p.requires_grad], lr=1e-3)

    resume_state = restore_target_adaptation_resume(
        resume_from=str(resume_path),
        state=fresh,
        optimizer=fresh_optimizer,
        device=torch.device("cpu"),
        max_epochs=50,
    )

    assert resume_state.start_epoch == 20
    assert resume_state.best_target_val_loss == 0.4
    assert resume_state.best_epoch == 18
    assert resume_state.train_history == [{"epoch": 18, "target_train_loss": 0.5}]
    assert resume_state.val_history == [{"epoch": 18, "target_val_loss": 0.4}]
    assert fresh_optimizer.state_dict()["state"]


def test_phase5_adapt_script_forwards_resume_from():
    script = Path("run/phase5_hyperda_target_adapt.sh")

    text = script.read_text()

    assert "RESUME_FROM" in text
    assert "--resume_from" in text
    assert "ENABLE_TARGET_SPATIAL_REFINE" in text
    assert "--enable_target_spatial_refine" in text
    assert 'ENABLE_TARGET_SPATIAL_REFINE="${ENABLE_TARGET_SPATIAL_REFINE:-1}"' in text
    assert 'TARGET_SPATIAL_REFINE_ROOTZONE="${TARGET_SPATIAL_REFINE_ROOTZONE:-1}"' in text
    assert 'TARGET_SPATIAL_REFINE_INPUT="${TARGET_SPATIAL_REFINE_INPUT:-normalized}"' in text
    assert "--target_spatial_refine_input" in text
    assert "TARGET_SELECTION_METRIC" in text
    assert "--target_selection_metric" in text
    assert "ENABLE_PIGO" not in text
    assert "--enable_pigo" not in text
    assert "--pigo_hidden" not in text
    assert "--pigo_surface_only" not in text
    assert "--lambda_high_update" not in text
    assert "--lambda_vertical" not in text
    assert "--lambda_range" not in text
    assert "--surface_weight" in text
    assert "# --resume_from" not in text
    assert "checkpoint_epoch_019.pt" not in text


def test_phase5_hydro_msr_script_forwards_hydro_msr_flags():
    script = Path("run/phase5_hydro_msr.sh")

    text = script.read_text()

    assert "TARGET_SPATIAL_REFINE_TYPE" in text
    assert 'TARGET_SPATIAL_REFINE_TYPE="${TARGET_SPATIAL_REFINE_TYPE:-hydro_msr}"' in text
    assert "--target_spatial_refine_type" in text
    assert "HYDRO_MSR_HIDDEN" in text
    assert "--hydro_msr_hidden" in text
    assert "ENABLE_HYDRO_MSR_DA_FILM" in text
    assert "--enable_hydro_msr_da_film" in text
    assert "DEVICE" in text
    assert '--device "${DEVICE}"' in text
    assert 'ENABLE_TARGET_SPATIAL_REFINE="${ENABLE_TARGET_SPATIAL_REFINE:-1}"' in text
    assert 'TARGET_SPATIAL_REFINE_ROOTZONE="${TARGET_SPATIAL_REFINE_ROOTZONE:-1}"' in text
    assert "ENABLE_PIGO" not in text
    assert "--enable_pigo" not in text


def test_phase5_da_gain_adapter_script_forwards_gain_and_stage_flags():
    script = Path("run/phase5_da_gain_adapter.sh")

    text = script.read_text()

    assert 'TARGET_SPATIAL_REFINE_TYPE="${TARGET_SPATIAL_REFINE_TYPE:-hydro_msr_gain}"' in text
    assert "ENABLE_DA_REGIME_GAIN_MIXER" in text
    assert "--enable_da_regime_gain_mixer" in text
    assert "STAGE1_EPOCHS" in text
    assert "--stage1_epochs" in text
    assert "TARGET_SPATIAL_REFINE_GAIN_SPAN" in text
    assert "--target_spatial_refine_gain_span" in text
    assert "SELECTION_ROOTZONE_WEIGHT" in text
    assert "--selection_rootzone_weight" in text
    assert 'TARGET_SELECTION_METRIC="${TARGET_SELECTION_METRIC:-combined_val_wrmse}"' in text
    assert 'RUN_NAME="${RUN_NAME:-phase5_hydro_msr_gain_${TARGET_REGION}_s${SEED}}"' in text


def test_phase5_anchor_soup_gain_adapter_script_forwards_anchor_soup_defaults():
    script = Path("run/phase5_anchor_soup_gain_adapter.sh")

    text = script.read_text()

    assert 'export CUDA_VISIBLE_DEVICES="${4:-${CUDA_VISIBLE_DEVICES:-0}}"' in text
    assert 'TARGET_SPATIAL_REFINE_TYPE="${TARGET_SPATIAL_REFINE_TYPE:-hydro_msr_gain_lite}"' in text
    assert 'TARGET_SPATIAL_REFINE_GAIN_SPAN="${TARGET_SPATIAL_REFINE_GAIN_SPAN:-0.20}"' in text
    assert 'STAGE1_EPOCHS="${STAGE1_EPOCHS:-0}"' in text
    assert 'MAX_EPOCHS="${MAX_EPOCHS:-60}"' in text
    assert 'LR="${LR:-7.5e-4}"' in text
    assert 'SURFACE_WEIGHT="${SURFACE_WEIGHT:-4.0}"' in text
    assert 'ROOTZONE_WEIGHT="${ROOTZONE_WEIGHT:-0.75}"' in text
    assert 'LAMBDA_ANALYSIS="${LAMBDA_ANALYSIS:-0.35}"' in text
    assert 'TARGET_SELECTION_METRIC="${TARGET_SELECTION_METRIC:-combined_val_wrmse}"' in text
    assert 'SELECTION_ROOTZONE_WEIGHT="${SELECTION_ROOTZONE_WEIGHT:-0.25}"' in text
    assert 'ENABLE_ADAPTER_ANCHOR_SOUP="${ENABLE_ADAPTER_ANCHOR_SOUP:-1}"' in text
    assert "--enable_adapter_anchor_soup" in text
    assert 'RUN_NAME="${RUN_NAME:-phase5_anchor_soup_gain_${TARGET_REGION}_s${SEED}}"' in text


def test_phase5_hydro_msr_rose_script_forwards_rose_flags():
    script = Path("run/phase5_hydro_msr_rose.sh")

    text = script.read_text()

    assert 'TARGET_SPATIAL_REFINE_TYPE="${TARGET_SPATIAL_REFINE_TYPE:-hydro_msr_rose}"' in text
    assert "TARGET_SPATIAL_REFINE_INPUT" in text
    assert "--target_spatial_refine_input" in text
    assert "HYDRO_MSR_HIDDEN" in text
    assert "--hydro_msr_hidden" in text
    assert "ENABLE_HYDRO_MSR_DA_FILM" in text
    assert "--enable_hydro_msr_da_film" in text
    assert "TARGET_SELECTION_METRIC" in text
    assert "--target_selection_metric" in text
    assert "RESUME_FROM" in text
    assert "--resume_from" in text
    assert 'RUN_NAME="${RUN_NAME:-phase5_hydro_msr_rose_${TARGET_REGION}_s${SEED}}"' in text
    assert "ENABLE_PIGO" not in text
    assert "--enable_pigo" not in text


def test_target_adaptation_run_config_uses_target_val_dataset_hash():
    from scripts.train.train_hyperda_target_adapt import _target_val_dates_hash

    class DummyDataset:
        def __init__(self, split_entry):
            self._split_entry = split_entry

    train_dataset = DummyDataset(
        {
            "target_val_dates_hash": "wrong-train-target-val-hash",
            "source_val_dates_hash": "source-val-hash",
        }
    )
    val_dataset = DummyDataset({"target_val_dates_hash": "target-val-hash"})

    assert _target_val_dates_hash(train_dataset, val_dataset) == "target-val-hash"


def test_batch_loss_includes_analysis_loss_when_forecasts_are_available():
    from scripts.train.train_hyperda_target_adapt import TargetAdaptationState, _batch_loss
    from hydroda.training.losses import MaskedHuberLoss

    class ZeroModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.target_adapter_coefficient_residual_b = None
            self.target_adapter_coefficient_residual_d2 = None
            self.target_adapter_coefficient_residual_d1 = None
            self.target_prompt = None
            self.residual_gain = None

        def forward(self, x, z, month):
            return self.anchor * torch.ones(x.shape[0], 2, x.shape[2], x.shape[3], device=x.device)

    model = ZeroModel()
    prompt_encoder = RegionPromptEncoder(num_regions=1, input_channels=12, hidden_dim=8).eval()
    state = TargetAdaptationState(
        model=model,
        prompt_encoder=prompt_encoder,
        source_checkpoint={},
        source_config={},
        normalization={"ch_mean": [0.0] * 12, "ch_std": [1.0] * 12, "inc_mean": None, "inc_std": None},
    )
    batch = {
        "x": torch.zeros(1, 12, 2, 2),
        "months": torch.tensor([1]),
        "increment_surface": torch.ones(1, 2, 2),
        "increment_rootzone": torch.ones(1, 2, 2) * 2.0,
        "forecast_surface": torch.ones(1, 2, 2) * 10.0,
        "forecast_rootzone": torch.ones(1, 2, 2) * 20.0,
        "loss_mask": torch.ones(1, 2, 2),
    }

    losses = _batch_loss(
        state=state,
        batch=batch,
        device=torch.device("cpu"),
        target_region="US-R1",
        loss_fn=MaskedHuberLoss(delta=1.0),
        normalize_increment=False,
        lambda_prior=0.0,
        lambda_latent=0.0,
        lambda_gain=0.0,
        lambda_gain_smooth=0.0,
        lambda_analysis=0.25,
    )

    assert torch.isclose(losses["analysis_loss"], losses["total_loss"])
    assert torch.isclose(losses["objective"], losses["total_loss"] * 1.25)


def test_target_selection_metric_can_select_surface_and_rootzone_wrmse():
    from scripts.train.train_hyperda_target_adapt import _target_selection_value

    metrics = {
        "target_val_loss": 5.0,
        "total_loss": 3.0,
        "analysis_loss": 2.0,
        "target_val_surface_wrmse_latw": 0.1,
        "target_val_rootzone_wrmse_latw": 0.2,
    }

    assert _target_selection_value(metrics, "objective") == 5.0
    assert _target_selection_value(metrics, "total_loss") == 3.0
    assert _target_selection_value(metrics, "analysis_loss") == 2.0
    assert _target_selection_value(metrics, "surface_val_wrmse") == 0.1
    assert _target_selection_value(metrics, "rootzone_val_wrmse") == 0.2
    assert _target_selection_value(metrics, "combined_val_wrmse", rootzone_weight=0.5) == pytest.approx(0.2)


def test_combined_target_selection_uses_independent_selection_rootzone_weight():
    from scripts.train.train_hyperda_target_adapt import _target_selection_value

    metrics = {
        "target_val_loss": 5.0,
        "total_loss": 3.0,
        "analysis_loss": 2.0,
        "target_val_surface_wrmse_latw": 0.1,
        "target_val_rootzone_wrmse_latw": 0.2,
    }

    assert _target_selection_value(metrics, "combined_val_wrmse", selection_rootzone_weight=2.0) == pytest.approx(0.5)


def test_selected_metric_name_records_combined_val_wrmse():
    from scripts.train.train_hyperda_target_adapt import _selected_metric_name

    assert _selected_metric_name("combined_val_wrmse") == "combined_val_wrmse"


def test_best_checkpoint_names_include_all_preregistered_validation_metrics():
    from scripts.train.train_hyperda_target_adapt import _best_checkpoint_names

    names = _best_checkpoint_names()

    assert names["objective"] == "checkpoint_best_target_val_loss.pt"
    assert names["surface_val_wrmse"] == "checkpoint_best_target_val_surface_wrmse.pt"
    assert names["rootzone_val_wrmse"] == "checkpoint_best_target_val_rootzone_wrmse.pt"
    assert names["combined_val_wrmse"] == "checkpoint_best_target_val_combined_wrmse.pt"


def test_target_adapter_state_extraction_is_target_only():
    from scripts.train.train_hyperda_target_adapt import extract_target_adapter_state

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_type="hydro_msr_gain_lite",
    )

    state = extract_target_adapter_state(model)

    assert state
    assert any(name.startswith("target_prompt.") for name in state)
    assert any(name.startswith("target_spatial_refine.") for name in state)
    assert any(name.startswith("residual_gain.") for name in state)
    assert all(
        name.startswith("target_")
        or name.startswith("residual_gain.")
        or "target_adapter_coefficient_residual" in name
        for name in state
    )
    assert not any(name.startswith("enc") for name in state)
    assert not any(name.startswith("hyper_adapter") for name in state)


def test_apply_target_adapter_state_preserves_source_prior_and_interpolates_from_anchor():
    from scripts.train.train_hyperda_target_adapt import (
        apply_target_adapter_state,
        extract_target_adapter_state,
        interpolate_target_adapter_state,
    )

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
    )
    anchor = extract_target_adapter_state(model)
    epoch_state = {name: tensor + 2.0 for name, tensor in anchor.items()}
    source_before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if name not in anchor
    }

    interpolated = interpolate_target_adapter_state(anchor, epoch_state, alpha=0.25)
    apply_target_adapter_state(model, interpolated)

    after = model.state_dict()
    for name, tensor in interpolated.items():
        assert torch.allclose(after[name].cpu(), tensor)
    for name, tensor in source_before.items():
        assert torch.allclose(after[name].cpu(), tensor)


def test_interpolate_target_adapter_state_uses_source_anchor_alpha():
    from scripts.train.train_hyperda_target_adapt import interpolate_target_adapter_state

    anchor = {"target_prompt.latent": torch.tensor([0.0, 2.0])}
    adapted = {"target_prompt.latent": torch.tensor([4.0, 6.0])}

    interpolated = interpolate_target_adapter_state(anchor, adapted, alpha=0.25)

    assert torch.allclose(interpolated["target_prompt.latent"], torch.tensor([1.0, 3.0]))
    assert torch.allclose(anchor["target_prompt.latent"], torch.tensor([0.0, 2.0]))
    assert torch.allclose(adapted["target_prompt.latent"], torch.tensor([4.0, 6.0]))


def test_greedy_anchor_soup_accepts_only_target_val_improvements():
    from scripts.train.train_hyperda_target_adapt import (
        AdapterAnchorSoupCandidate,
        greedy_select_adapter_anchor_soup,
    )

    def evaluate_state(state):
        marker = float(state["target_prompt.latent"][0])
        metric_by_marker = {
            1.0: (0.8, 1.0),
            2.0: (0.7, 0.9),
            3.0: (1.0, 1.0),
            5.0: (1.1, 0.8),
        }
        surface, rootzone = metric_by_marker[marker]
        return {
            "target_val_surface_wrmse_latw": surface,
            "target_val_rootzone_wrmse_latw": rootzone,
        }

    candidates = [
        AdapterAnchorSoupCandidate(
            candidate_id="epoch010_alpha0.40",
            epoch=10,
            alpha=0.40,
            state={"target_prompt.latent": torch.tensor([1.0])},
        ),
        AdapterAnchorSoupCandidate(
            candidate_id="epoch015_alpha0.40",
            epoch=15,
            alpha=0.40,
            state={"target_prompt.latent": torch.tensor([3.0])},
        ),
        AdapterAnchorSoupCandidate(
            candidate_id="epoch020_alpha0.40",
            epoch=20,
            alpha=0.40,
            state={"target_prompt.latent": torch.tensor([5.0])},
        ),
    ]

    result = greedy_select_adapter_anchor_soup(
        candidates,
        evaluate_state=evaluate_state,
        selection_rootzone_weight=0.25,
        rootzone_guard=True,
    )

    assert result.selected_metric == pytest.approx(0.925)
    assert [candidate.candidate_id for candidate in result.accepted_candidates] == [
        "epoch010_alpha0.40",
        "epoch015_alpha0.40",
    ]
    assert torch.allclose(result.selected_state["target_prompt.latent"], torch.tensor([2.0]))
    assert any(row["decision"] == "reject" for row in result.trace_rows)


def test_stage_schedule_freezes_global_then_spatial_modules():
    from scripts.train.train_hyperda_target_adapt import apply_target_adaptation_stage

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_type="hydro_msr_gain",
    )
    model.freeze_source_prior_for_target_adaptation()

    stage1_names = apply_target_adaptation_stage(model, epoch=0, stage1_epochs=10)
    assert any(name.startswith("target_prompt") for name in stage1_names)
    assert any("target_adapter_coefficient_residual" in name for name in stage1_names)
    assert any(name.startswith("residual_gain") for name in stage1_names)
    assert not any(name.startswith("target_spatial_refine") for name in stage1_names)

    stage2_names = apply_target_adaptation_stage(model, epoch=10, stage1_epochs=10)
    assert any(name.startswith("target_spatial_refine") for name in stage2_names)
    assert not any(name.startswith("target_prompt") for name in stage2_names)
    assert not any("target_adapter_coefficient_residual" in name for name in stage2_names)
    assert not any(name.startswith("residual_gain") for name in stage2_names)


def test_target_val_wrmse_aggregates_latitude_weighted_sse_in_physical_space():
    from scripts.train.train_hyperda_target_adapt import _update_target_val_wrmse_accumulators

    accum = {
        "surface_sse": 0.0,
        "surface_weight": 0.0,
        "rootzone_sse": 0.0,
        "rootzone_weight": 0.0,
    }
    pred = torch.zeros(1, 2, 2, 2)
    target = torch.zeros(1, 2, 2, 2)
    pred[:, 0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pred[:, 1] = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    loss_mask = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
    lat_w = torch.tensor([[[1.0, 10.0], [2.0, 3.0]]])

    _update_target_val_wrmse_accumulators(
        accum,
        pred_analysis=pred,
        true_analysis=target,
        loss_mask=loss_mask,
        latitude_weight=lat_w,
    )

    expected_surface_sse = 1.0 * 1.0**2 + 2.0 * 3.0**2 + 3.0 * 4.0**2
    expected_rootzone_sse = 1.0 * 2.0**2 + 2.0 * 6.0**2 + 3.0 * 8.0**2
    expected_weight = 1.0 + 2.0 + 3.0
    assert accum["surface_sse"] == expected_surface_sse
    assert accum["rootzone_sse"] == expected_rootzone_sse
    assert accum["surface_weight"] == expected_weight
    assert accum["rootzone_weight"] == expected_weight
