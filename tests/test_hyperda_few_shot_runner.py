from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np
import torch

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
            },
        },
        path,
    )


def test_few_shot_parse_args_uses_preregistered_steps_and_no_target_val(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hyperda_few_shot_adapt.py",
            "--source_checkpoint",
            str(source_checkpoint),
            "--target_region",
            "US-R1",
            "--K",
            "4",
            "--seed",
            "0",
        ],
    )

    args = runner.parse_args()
    assert args.K == 4
    assert args.adaptation_setting == "few_shot_k4"
    assert args.adaptation_steps == 100
    assert args.target_val_usage == "unused_in_main_protocol"
    assert args.model_selection_source == "source_val_preregistered"
    assert args.enable_target_spatial_refine is False


def test_k12_parse_args_uses_conservative_source_anchor_defaults(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hyperda_few_shot_adapt.py",
            "--source_checkpoint",
            str(source_checkpoint),
            "--target_region",
            "US-R1",
            "--K",
            "12",
            "--seed",
            "0",
        ],
    )

    args = runner.parse_args()
    assert args.K == 12
    assert args.adaptation_setting == "few_shot_k12"
    assert args.adaptation_steps == 80
    assert args.lr == pytest.approx(3e-4)
    assert args.adapt_recipe == "source_anchor"
    assert args.anchor_alpha == pytest.approx(0.25)
    assert args.source_anchor_hyperparameter_source == "source_side_episodic_validation_preregistered"


def test_k0_parse_args_disables_target_label_training(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hyperda_few_shot_adapt.py",
            "--source_checkpoint",
            str(source_checkpoint),
            "--target_region",
            "US-R1",
            "--K",
            "0",
        ],
    )

    args = runner.parse_args()
    assert args.K == 0
    assert args.adaptation_steps == 0
    assert args.adaptation_setting == "zero_shot_context"
    assert args.anchor_alpha == pytest.approx(0.0)


def test_few_shot_dataset_plan_never_constructs_target_val():
    from scripts.train.train_hyperda_few_shot_adapt import build_dataset_plan

    plan = build_dataset_plan(K=12)
    assert "target_context" in plan
    assert "target_support" in plan
    assert "target_val" not in plan

    zero_plan = build_dataset_plan(K=0)
    assert zero_plan == ["target_context"]


def test_load_source_checkpoint_for_few_shot_freezes_source_prior(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import load_source_checkpoint_for_few_shot

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    trainable = state.model.target_trainable_parameter_names()
    assert trainable
    assert all(
        not name.startswith(prefix)
        for name in trainable
        for prefix in ("stem", "encoder", "decoder", "bottleneck", "hyper_basis")
    )
    assert all(
        name.startswith("target_")
        or name.startswith("residual_gain")
        or "coefficient_residual" in name
        for name in trainable
    )
    assert state.prompt_encoder.training is False
    assert all(not p.requires_grad for p in state.prompt_encoder.parameters())


def test_save_few_shot_checkpoint_metadata(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        FewShotAdaptationState,
        load_source_checkpoint_for_few_shot,
        save_few_shot_checkpoint,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    loaded = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    state = FewShotAdaptationState(
        model=loaded.model,
        prompt_encoder=loaded.prompt_encoder,
        source_checkpoint=loaded.source_checkpoint,
        source_config=loaded.source_config,
        normalization=loaded.normalization,
    )
    out = tmp_path / "checkpoint_final_preregistered.pt"
    prompt_state = {
        "schema_version": "target_context_prompt_state_v1",
        "prompt_source": "target_context_monthly_prompt_prototypes",
        "label_usage": "none",
        "context_hash": "contexthash",
        "date_start": "2015-01-01",
        "date_end": "2021-12-31",
        "monthly_counts": {str(i): 0 for i in range(1, 13)},
        "global_prototype": torch.zeros(8),
        "monthly_prototypes": {str(i): None for i in range(1, 13)},
        "metadata": {
            "eval_input_usage": "none_for_prompt_update",
            "eval_month_usage": "known_seasonal_phase_selector_only",
        },
    }

    save_few_shot_checkpoint(
        path=out,
        state=state,
        optimizer_state_dict={},
        config={
            "K": 4,
            "adaptation_setting": "few_shot_k4",
            "adapt_recipe": "source_anchor",
            "anchor_alpha": 0.75,
            "source_anchor_hyperparameter_source": "source_side_episodic_validation_preregistered",
            "target_region": "US-R1",
            "source_checkpoint": str(ckpt_path),
            "split_manifest_path": "artifacts/splits/US_loro_zero_few_shot_splits.json",
            "split_manifest_sha256": "splitsha",
            "target_support_dates_hash": "supporthash",
            "target_support_dates": ["2019-04-15", "2019-07-15"],
            "target_context_dates_hash": "contexthash",
            "target_eval_dates_hash": "evalhash",
            "adaptation_steps": 100,
            "lr": 1e-3,
            "support_final_loss": 0.08,
            "support_loss_delta": -0.02,
            "target_parameter_l2_drift": {"total": 0.1, "target_prompt": 0.1},
        },
        target_context_prompt_state=prompt_state,
        train_history=[{"step": 1, "loss": 0.1}],
    )

    saved = torch.load(out, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    assert saved["tag"] == "final_preregistered"
    assert cfg["target_support_dates_hash"] == "supporthash"
    assert cfg["target_support_dates"] == ["2019-04-15", "2019-07-15"]
    assert cfg["target_context_dates_hash"] == "contexthash"
    assert cfg["protocol_freeze_id"] == saved["protocol_freeze_id"]
    assert cfg["model_selection_source"] == "source_val_preregistered"
    assert cfg["target_val_usage"] == "unused_in_main_protocol"
    assert cfg["checkpoint_selection"] == "fixed_preregistered_final_step"
    assert cfg["adapt_recipe"] == "source_anchor"
    assert cfg["anchor_alpha"] == pytest.approx(0.75)
    assert cfg["source_anchor_hyperparameter_source"] == "source_side_episodic_validation_preregistered"
    assert cfg["support_final_loss"] == pytest.approx(0.08)
    assert cfg["support_loss_delta"] == pytest.approx(-0.02)
    assert cfg["target_parameter_l2_drift"]["total"] == pytest.approx(0.1)
    assert cfg["prompt_policy"] == "target_context_monthly_prompt_prototypes"
    assert cfg["eval_input_usage"] == "none_for_prompt_update"
    assert cfg["target_context_prompt_state_summary"]["context_date_hash"] == "contexthash"
    assert saved["target_context_prompt_state"]["context_hash"] == "contexthash"
    assert cfg["trainable_parameter_names"]
    assert not any(name.startswith("prompt_encoder") for name in cfg["trainable_parameter_names"])


def test_save_few_shot_checkpoint_records_setting_specific_method_and_context_date_hash(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        FewShotAdaptationState,
        load_source_checkpoint_for_few_shot,
        save_few_shot_checkpoint,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    loaded = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    state = FewShotAdaptationState(
        model=loaded.model,
        prompt_encoder=loaded.prompt_encoder,
        source_checkpoint=loaded.source_checkpoint,
        source_config=loaded.source_config,
        normalization=loaded.normalization,
    )
    out = tmp_path / "checkpoint_final_preregistered.pt"
    prompt_state = {
        "schema_version": "target_context_prompt_state_v1",
        "prompt_source": "target_context_monthly_prompt_prototypes",
        "label_usage": "none",
        "context_hash": "contexthash",
        "monthly_counts": {str(i): 0 for i in range(1, 13)},
        "global_prototype": torch.zeros(8),
        "monthly_prototypes": {str(i): None for i in range(1, 13)},
        "metadata": {},
    }

    save_few_shot_checkpoint(
        path=out,
        state=state,
        optimizer_state_dict={},
        config={
            "K": 12,
            "adaptation_setting": "few_shot_k12",
            "target_region": "US-R1",
            "source_checkpoint": str(ckpt_path),
            "split_manifest_path": "artifacts/splits/US_loro_zero_few_shot_splits.json",
            "split_manifest_sha256": "splitsha",
            "target_support_dates_hash": "supporthash",
            "target_context_dates_hash": "contexthash",
            "target_eval_dates_hash": "evalhash",
            "adaptation_steps": 200,
        },
        target_context_prompt_state=prompt_state,
        train_history=[],
    )

    saved = torch.load(out, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    assert cfg["method"] == "hyperda_few_shot_k12"
    assert saved["target_context_prompt_state"]["context_date_hash"] == "contexthash"
    assert saved["target_context_prompt_state"]["context_hash"] == "contexthash"
    assert cfg["target_context_prompt_state"]["context_date_hash"] == "contexthash"


def test_build_context_prompt_state_reads_only_input_side_fields(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import load_source_checkpoint_for_few_shot
    from hydroda.baselines.prompt_conditioned import build_target_context_prompt_state

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    class LabelPoisonSample(dict):
        def __getitem__(self, key):
            if key.startswith("analysis_") or key.startswith("increment_"):
                raise AssertionError(f"target labels must not be read for prompt state: {key}")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key.startswith("analysis_") or key.startswith("increment_"):
                raise AssertionError(f"target labels must not be read for prompt state: {key}")
            return super().get(key, default)

    prompt_state = build_target_context_prompt_state(
        samples=[
            LabelPoisonSample(
                {
                    "x": np.zeros((12, 8, 8), dtype=np.float32),
                    "month": 1,
                    "date_str": "2015-01-01",
                    "analysis_surface": object(),
                    "increment_surface": object(),
                }
            )
        ],
        prompt_encoder=state.prompt_encoder,
        normalize_x=lambda x: x,
        target_region_embedding=torch.zeros(1, 16),
        device=torch.device("cpu"),
    )

    assert prompt_state["label_usage"] == "none"
    assert prompt_state["monthly_counts"]["1"] == 1
    assert prompt_state["metadata"]["eval_input_usage"] == "none_for_prompt_update"


def test_source_anchor_interpolation_only_touches_target_adapter_state(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        apply_source_anchor_interpolation,
        extract_target_adapter_state,
        load_source_checkpoint_for_few_shot,
        target_parameter_l2_drift,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    anchor_state = extract_target_adapter_state(state.model)
    source_prior_state = {
        name: tensor.detach().clone()
        for name, tensor in state.model.state_dict().items()
        if name not in anchor_state
    }

    with torch.no_grad():
        for name, param in state.model.named_parameters():
            if name in anchor_state:
                param.add_(2.0)

    raw_adapted_state = extract_target_adapter_state(state.model)
    apply_source_anchor_interpolation(state.model, anchor_state, alpha=0.25)
    interpolated_state = extract_target_adapter_state(state.model)

    for name in anchor_state:
        expected = anchor_state[name] + 0.25 * (raw_adapted_state[name] - anchor_state[name])
        assert torch.allclose(interpolated_state[name], expected)
    for name, tensor in state.model.state_dict().items():
        if name not in anchor_state:
            assert torch.allclose(tensor, source_prior_state[name])

    drift = target_parameter_l2_drift(anchor_state, interpolated_state)
    assert drift["total"] > 0.0
    assert "target_prompt" in drift


def test_target_context_input_side_accessor_does_not_read_target_labels():
    from hydroda.data.dataset import HydroDADataset

    dataset = HydroDADataset.__new__(HydroDADataset)
    dataset._time_indices = [0]
    dataset.input_var = "input"
    dataset.target_var = "target"
    dataset.forecast_surface_channel = 0
    dataset.forecast_rootzone_channel = 1
    dataset.base_valid_mask_channel = 11
    dataset._region_mask_int = np.ones((2, 2), dtype=np.int16)
    dataset._active_region_mask = np.ones((2, 2), dtype=np.float32)
    dataset._latitude = np.ones((2, 2), dtype=np.float32)
    dataset._latitude_weight = np.ones((2, 2), dtype=np.float32)
    dataset._date_str_map = {0: "2019-05-01"}
    dataset.K = 0
    dataset.target_region = "US-R1"
    dataset.seed = 0
    dataset.split_type = "target_context"
    dataset.adaptation_setting = "zero_shot_context"
    dataset._active_region_ids = ["US-R1"]
    dataset.regime_id = "R1"
    dataset._split_entry = {}
    dataset.split_manifest_sha256 = ""

    class FakeVar:
        def __init__(self, values):
            self.values = values

        def isel(self, **_kwargs):
            return self

    class PoisonDataset:
        def __getitem__(self, key):
            if key == "target":
                raise AssertionError("target_context input-side accessor must not read target labels")
            if key != "input":
                raise KeyError(key)
            return FakeVar(np.zeros((12, 2, 2), dtype=np.float32))

    dataset._da_ds = PoisonDataset()

    sample = dataset.get_input_side_sample(0)

    assert "x" in sample
    assert "analysis_surface" not in sample
    assert "increment_surface" not in sample
    assert sample["month"] == 5


def test_few_shot_batch_loss_uses_context_prompt_state_not_support_input_stats(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        load_source_checkpoint_for_few_shot,
        few_shot_batch_loss,
    )
    from hydroda.training.losses import MaskedHuberLoss

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    def fail_compute_input_stats(_x):
        raise AssertionError("support input stats must not construct target-context prompt state during few-shot training")

    state.prompt_encoder._compute_input_stats = fail_compute_input_stats
    prompt_state = {
        "schema_version": "target_context_prompt_state_v1",
        "prompt_source": "target_context_monthly_prompt_prototypes",
        "label_usage": "none",
        "monthly_counts": {str(i): 0 for i in range(1, 13)},
        "global_prototype": torch.zeros(8),
        "monthly_prototypes": {str(i): None for i in range(1, 13)},
        "metadata": {},
    }
    batch = {
        "x": torch.zeros(1, 12, 16, 16),
        "months": torch.tensor([5], dtype=torch.long),
        "increment_surface": torch.zeros(1, 16, 16),
        "increment_rootzone": torch.zeros(1, 16, 16),
        "loss_mask": torch.ones(1, 16, 16),
        "forecast_surface": torch.zeros(1, 16, 16),
        "forecast_rootzone": torch.zeros(1, 16, 16),
    }

    losses = few_shot_batch_loss(
        state=state,
        batch=batch,
        device=torch.device("cpu"),
        target_context_prompt_state=prompt_state,
        loss_fn=MaskedHuberLoss(),
        normalize_increment=True,
        lambda_prior=0.0,
        lambda_latent=0.0,
        lambda_gain=0.0,
        lambda_gain_smooth=0.0,
    )

    assert torch.isfinite(losses["objective"])


def test_parse_args_rejects_full_target_train_without_legacy_flag(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hyperda_few_shot_adapt.py",
            "--source_checkpoint",
            str(source_checkpoint),
            "--target_region",
            "US-R1",
            "--K",
            "4",
            "--adaptation_setting",
            "target_full_train",
        ],
    )

    with pytest.raises(SystemExit):
        runner.parse_args()


def test_prompt_predictor_uses_main_hyperda_method_ids_for_zero_few_shot_checkpoints(tmp_path):
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
    prompt_state = {
        "schema_version": "target_context_prompt_state_v1",
        "prompt_source": "target_context_monthly_prompt_prototypes",
        "label_usage": "none",
        "context_hash": "contexthash",
        "monthly_counts": {str(i): 0 for i in range(1, 13)},
        "global_prototype": torch.zeros(8),
        "monthly_prototypes": {str(i): None for i in range(1, 13)},
        "metadata": {},
    }
    ckpt_path = tmp_path / "fewshot.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "target_context_prompt_state": prompt_state,
            "config": {
                "model_type": "hyperda_basis_adapter_target_adapt",
                "method": "hyperda_few_shot_k4",
                "adaptation_setting": "few_shot_k4",
                "K": 4,
                "width": 4,
                "prompt_dim": 8,
                "hyper_n_basis": 3,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "zero_raw_increment_init": True,
                "num_regions": 5,
                "target_latent_dim": 4,
                "source_region_global_indices": [1, 2, 3, 4, 5],
                "ch_mean": [0.0] * 12,
                "ch_std": [1.0] * 12,
                "inc_mean": [0.0, 0.0],
                "inc_std": [1.0, 1.0],
            },
        },
        ckpt_path,
    )

    predictor = PromptConditionedBackbonePredictor(
        checkpoint_path=str(ckpt_path),
        device="cpu",
        target_region="US-R1",
    )

    assert predictor.method_name == "hyperda_few_shot_k4"


def test_prompt_predictor_requires_saved_context_prompt_state_for_main_hyperda_checkpoint(tmp_path):
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
    ckpt_path = tmp_path / "fewshot_without_context_state.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "config": {
                "model_type": "hyperda_basis_adapter_target_adapt",
                "method": "hyperda_few_shot_k4",
                "adaptation_setting": "few_shot_k4",
                "K": 4,
                "width": 4,
                "prompt_dim": 8,
                "hyper_n_basis": 3,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "zero_raw_increment_init": True,
                "num_regions": 5,
                "target_latent_dim": 4,
                "source_region_global_indices": [1, 2, 3, 4, 5],
                "ch_mean": [0.0] * 12,
                "ch_std": [1.0] * 12,
                "inc_mean": [0.0, 0.0],
                "inc_std": [1.0, 1.0],
            },
        },
        ckpt_path,
    )

    with pytest.raises(ValueError, match="target_context_prompt_state"):
        PromptConditionedBackbonePredictor(
            checkpoint_path=str(ckpt_path),
            device="cpu",
            target_region="US-R1",
        )


def test_few_shot_run_metadata_sidecar_contains_protocol_fields(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import write_run_metadata_sidecar

    checkpoint_path = tmp_path / "checkpoints" / "checkpoint_final_preregistered.pt"
    config = {
        "method": "hyperda_zero_shot_context",
        "adaptation_setting": "zero_shot_context",
        "adapt_recipe": "source_anchor",
        "anchor_alpha": 0.0,
        "source_anchor_hyperparameter_source": "source_side_episodic_validation_preregistered",
        "K": 0,
        "seed": 2,
        "target_region": "US-R1",
        "split_manifest_path": "artifacts/splits/US_loro_zero_few_shot_splits.json",
        "split_manifest_sha256": "splitsha",
        "target_context_dates_hash": "contexthash",
        "target_support_dates_hash": "supporthash",
        "target_support_dates": [],
        "target_eval_dates_hash": "evalhash",
        "target_context_prompt_state": {"schema_version": "target_context_prompt_state_v1"},
        "trainable_parameter_count": 123,
        "adaptation_steps": 0,
        "lr": 3e-4,
        "support_final_loss": None,
        "support_loss_delta": None,
        "target_parameter_l2_drift": {"total": 0.0},
        "normalization_source": "source_fit_only_from_source_checkpoint",
        "model_selection_source": "source_val_preregistered",
        "target_val_usage": "unused_in_main_protocol",
    }

    write_run_metadata_sidecar(tmp_path, checkpoint_path, config)

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["checkpoint"] == str(checkpoint_path)
    assert metadata["method"] == "hyperda_zero_shot_context"
    assert metadata["adaptation_setting"] == "zero_shot_context"
    assert metadata["target_context_dates_hash"] == "contexthash"
    assert metadata["target_support_dates_hash"] == "supporthash"
    assert metadata["target_support_dates"] == []
    assert metadata["target_eval_dates_hash"] == "evalhash"
    assert metadata["model_selection_source"] == "source_val_preregistered"
    assert metadata["target_val_usage"] == "unused_in_main_protocol"
    assert metadata["trainable_parameter_count"] == 123
    assert metadata["adapt_recipe"] == "source_anchor"
    assert metadata["anchor_alpha"] == 0.0
    assert metadata["lr"] == pytest.approx(3e-4)
    assert metadata["source_anchor_hyperparameter_source"] == "source_side_episodic_validation_preregistered"
    assert metadata["support_final_loss"] is None
    assert metadata["support_loss_delta"] is None
    assert metadata["target_parameter_l2_drift"]["total"] == 0.0
    assert metadata["target_context_prompt_state"]["schema_version"] == "target_context_prompt_state_v1"
