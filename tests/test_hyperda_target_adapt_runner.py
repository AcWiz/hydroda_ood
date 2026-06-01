from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

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


def test_load_source_checkpoint_for_target_adaptation_freezes_source_prior(tmp_path):
    from scripts.train.train_hyperda_target_adapt import load_source_checkpoint_for_target_adaptation

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)

    state = load_source_checkpoint_for_target_adaptation(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    trainable = state.model.target_trainable_parameter_names()
    assert trainable
    assert all(
        name.startswith("target_")
        or name.startswith("residual_gain")
        or "coefficient_residual" in name
        for name in trainable
    )
    assert state.prompt_encoder.training is False
    assert state.normalization["ch_mean"] == [0.0] * 12
    assert state.source_config["model_type"] == "hyperda_basis_adapter"


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
    assert cfg["model_selection_source"] == "target_val_2022_preregistered_adaptation_selection"
    assert cfg["target_eval_usage"] == "final_eval_only_no_training_no_selection"


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
