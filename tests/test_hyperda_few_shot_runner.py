from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np
import torch

from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.prompt_encoder import RegionPromptEncoder, RobustInputSideDAPromptEncoder


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


def _write_rank_gated_source_checkpoint(path: Path) -> None:
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=5,
        hyper_adapter_bottleneck=2,
        hyper_coeff_generator="shared_layer_aware_rank_gated",
        hyper_rank_gate_top_k=2,
        hyper_adapter_param_style="dora_like_gain",
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
                "hyper_n_basis": 5,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "hyper_coeff_generator": "shared_layer_aware_rank_gated",
                "hyper_rank_gate_top_k": 2,
                "hyper_rank_gate_temperature_init": 1.0,
                "hyper_adapter_param_style": "dora_like_gain",
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


def _write_source_saliency_prior_checkpoint(path: Path) -> None:
    prior = torch.tensor(
        [
            [0.0, 1.0, -1.0, 0.5, -0.5],
            [1.0, 0.0, -1.0, 0.5, -0.5],
            [-1.0, 0.0, 1.0, -0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=5,
        hyper_adapter_bottleneck=2,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_source_saliency_prior=prior,
        hyper_source_saliency_prior_beta=0.5,
        hyper_prompt_manifold_reliability=True,
        hyper_prompt_manifold_reliability_strength=0.25,
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
                "hyper_n_basis": 5,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "hyper_coeff_generator": "shared_layer_aware_rank_gated_stable",
                "hyper_rank_gate_top_k": 2,
                "hyper_rank_gate_temperature_init": 2.0,
                "hyper_adapter_param_style": "dora_like_gain_bounded",
                "hyper_source_saliency_prior": prior.tolist(),
                "hyper_source_saliency_prior_beta": 0.5,
                "hyper_source_saliency_prior_path": "artifacts/prior/source_fit.pt",
                "hyper_prompt_manifold_reliability": True,
                "hyper_prompt_manifold_reliability_strength": 0.25,
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


def _write_robust_source_checkpoint(path: Path) -> None:
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        zero_raw_increment_init=True,
    )
    prompt_encoder = RobustInputSideDAPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "config": {
                "model_type": "hyperda_basis_adapter",
                "context_encoder": "robust_input_side_da_diagnostics",
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


def _write_raw_robust_source_checkpoint(path: Path) -> None:
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        zero_raw_increment_init=True,
    )
    prompt_encoder = RobustInputSideDAPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "config": {
                "model_type": "hyperda_basis_adapter",
                "context_encoder": "robust_input_side_da_diagnostics_raw",
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
    assert args.adapt_scope == "safe_operator"
    assert args.adapt_solver == "adamw"


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
    assert args.adapt_scope == "safe_operator"
    assert args.safe_policy_json is None
    assert args.policy_source == "preregistered_default"


def test_parse_args_applies_source_side_safe_policy_json(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    policy_path = tmp_path / "safe_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "hyperda_safe_policy_v1",
                "policy_source": "source_side_episode_calibration",
                "target_val_usage": "unused_in_main_protocol",
                "target_eval_usage": "final_eval_only_no_selection",
                "source_episode_regions": ["US-R2", "US-R3"],
                "policies": {
                    "few_shot_k4": {
                        "adapt_scope": "coeff_only",
                        "lr": 7e-4,
                        "adaptation_steps": 9,
                        "anchor_alpha": 0.5,
                        "rho_policy": "fixed_0.75",
                        "adapt_mix_rho": 0.75,
                        "support_loss_reduction": "cycle_balanced",
                        "lambda_analysis": 0.4,
                        "freeze_monthly_gain": True,
                        "source_calibrated_candidate_id": "K4_conservative",
                        "source_calibrated_guard_config_hash": "guardhash4",
                    },
                    "few_shot_k12": {
                        "adapt_scope": "coeff_only",
                        "lr": 2e-4,
                        "adaptation_steps": 11,
                        "anchor_alpha": 0.2,
                        "support_loss_reduction": "cycle_balanced",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
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
            "--safe_policy_json",
            str(policy_path),
        ],
    )

    args = runner.parse_args()

    assert args.policy_source == "source_side_episode_calibration"
    assert args.safe_policy_json == str(policy_path)
    assert args.safe_policy_json_sha256
    assert args.safe_policy["schema_version"] == "hyperda_safe_policy_v1"
    assert args.source_episode_regions == ["US-R2", "US-R3"]
    assert args.adapt_scope == "coeff_only"
    assert args.lr == pytest.approx(7e-4)
    assert args.adaptation_steps == 9
    assert args.anchor_alpha == pytest.approx(0.5)
    assert args.rho_policy == "fixed_0.75"
    assert args.adapt_mix_rho == pytest.approx(0.75)
    assert args.support_loss_reduction == "cycle_balanced"
    assert args.lambda_analysis == pytest.approx(0.4)
    assert args.freeze_monthly_gain is True
    assert args.source_policy_candidate_id == "K4_conservative"
    assert args.source_policy_guard_config_hash == "guardhash4"
    assert args.target_val_usage == "unused_in_main_protocol"
    assert args.target_eval_usage == "final_eval_only_no_selection"


def test_source_calibrated_mix_policy_accepts_source_side_safe_policy_json(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    policy_path = tmp_path / "safe_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy_source": "source_side_episode_calibration",
                "target_val_usage": "unused_in_main_protocol",
                "target_eval_usage": "final_eval_only_no_selection",
                "policies": {
                    "few_shot_k4": {
                        "adapt_scope": "coeff_only",
                        "lr": 5e-4,
                        "steps": 40,
                        "anchor_alpha": 0.5,
                        "adapt_mix_rho": 0.75,
                        "trust_region_mode": "groupwise",
                        "trust_coeff_radius": 0.2,
                        "trust_gain_radius": 0.1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
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
            "--stage3_posterior_policy",
            "source_calibrated_mix",
            "--require_safe_policy_json_for_kshot",
            "--safe_policy_json",
            str(policy_path),
        ],
    )

    args = runner.parse_args()

    assert args.stage3_posterior_policy == "source_calibrated_mix"
    assert args.adapt_scope == "coeff_only"
    assert args.adaptation_steps == 40
    assert args.lr == pytest.approx(5e-4)
    assert args.anchor_alpha == pytest.approx(0.5)
    assert args.adapt_mix_rho == pytest.approx(0.75)
    assert args.trust_region_mode == "groupwise"
    assert args.trust_coeff_radius == pytest.approx(0.2)
    assert args.trust_gain_radius == pytest.approx(0.1)
    assert args.safe_policy_json_sha256


def test_source_policy_can_explicitly_enable_monthly_gain_for_conservative_stage3(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    policy_path = tmp_path / "safe_policy_gain.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy_source": "source_side_episode_calibration",
                "target_val_usage": "unused_in_main_protocol",
                "target_eval_usage": "final_eval_only_no_selection",
                "policies": {
                    "few_shot_k12": {
                        "adapt_scope": "coeff_gain",
                        "lr": 2e-4,
                        "adaptation_steps": 12,
                        "anchor_alpha": 0.25,
                        "adapt_mix_rho": 0.4,
                        "freeze_monthly_gain": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
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
            "--stage3_posterior_policy",
            "conservative_coeff_posterior",
            "--require_safe_policy_json_for_kshot",
            "--safe_policy_json",
            str(policy_path),
        ],
    )

    args = runner.parse_args()

    assert args.policy_source == "source_side_episode_calibration"
    assert args.adapt_scope == "coeff_gain"
    assert args.freeze_monthly_gain is False
    assert args.adapt_mix_rho == pytest.approx(0.4)


def test_parse_args_rejects_target_side_safe_policy_json(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    policy_path = tmp_path / "unsafe_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy_source": "target_val_grid_search",
                "target_val_usage": "used_for_selection",
                "policies": {"few_shot_k4": {"lr": 1e-3}},
            }
        ),
        encoding="utf-8",
    )
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
            "--safe_policy_json",
            str(policy_path),
        ],
    )

    with pytest.raises(SystemExit):
        runner.parse_args()


def test_parse_args_rejects_kshot_without_policy_when_paper_facing(monkeypatch, tmp_path):
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
            "--require_safe_policy_json_for_kshot",
        ],
    )

    with pytest.raises(SystemExit):
        runner.parse_args()


def test_parse_args_accepts_adapt_scope_and_identity_audit(monkeypatch, tmp_path):
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
            "--adaptation_steps",
            "0",
            "--anchor_alpha",
            "0.0",
            "--adapt_scope",
            "none",
            "--stage3_posterior_policy",
            "conservative_coeff_posterior",
            "--audit_identity",
            "--audit_identity_tolerance",
            "1e-7",
        ],
    )

    args = runner.parse_args()
    assert args.adapt_scope == "none"
    assert args.stage3_posterior_policy == "conservative_coeff_posterior"
    assert args.audit_identity is True
    assert args.audit_identity_tolerance == pytest.approx(1e-7)


def test_parse_args_accepts_ridge_coeff_solver(monkeypatch, tmp_path):
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
            "--adapt_solver",
            "ridge_coeff",
            "--adapt_scope",
            "coeff_only",
            "--ridge_lambda",
            "2.5",
            "--ridge_clip_coeff_norm",
            "0.75",
            "--ridge_trust_region_radius",
            "0.5",
            "--ridge_max_feature_pixels",
            "1234",
            "--ridge_standardize_features",
        ],
    )

    args = runner.parse_args()
    assert args.adapt_solver == "ridge_coeff"
    assert args.adapt_scope == "coeff_only"
    assert args.ridge_lambda == pytest.approx(2.5)
    assert args.ridge_clip_coeff_norm == pytest.approx(0.75)
    assert args.ridge_trust_region_radius == pytest.approx(0.5)
    assert args.ridge_max_feature_pixels == 1234
    assert args.ridge_standardize_features is True


def test_parse_args_accepts_freeze_monthly_gain(monkeypatch, tmp_path):
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
            "--freeze_monthly_gain",
        ],
    )

    args = runner.parse_args()
    assert args.freeze_monthly_gain is True


def test_source_policy_must_define_adapt_mix_rho_for_kshot(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    policy_path = tmp_path / "safe_policy_missing_rho.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy_source": "source_side_episode_calibration",
                "target_val_usage": "unused_in_main_protocol",
                "target_eval_usage": "final_eval_only_no_selection",
                "policies": {
                    "few_shot_k4": {
                        "adapt_scope": "coeff_only",
                        "lr": 5e-4,
                        "steps": 40,
                        "anchor_alpha": 0.5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
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
            "--safe_policy_json",
            str(policy_path),
        ],
    )

    with pytest.raises(SystemExit):
        runner.parse_args()


def test_source_policy_can_select_k4_no_update_for_conservative_stage3(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    policy_path = tmp_path / "safe_policy_no_update.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy_source": "source_side_episode_calibration",
                "target_val_usage": "unused_in_main_protocol",
                "target_eval_usage": "final_eval_only_no_selection",
                "policies": {
                    "few_shot_k4": {
                        "adapt_scope": "none",
                        "lr": 0.0,
                        "adaptation_steps": 0,
                        "anchor_alpha": 0.0,
                        "adapt_mix_rho": 0.0,
                        "source_calibrated_candidate_id": "K4_source_calibrated_no_update",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
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
            "--stage3_posterior_policy",
            "conservative_coeff_posterior",
            "--require_safe_policy_json_for_kshot",
            "--safe_policy_json",
            str(policy_path),
        ],
    )

    args = runner.parse_args()

    assert args.adapt_scope == "none"
    assert args.adaptation_steps == 0
    assert args.anchor_alpha == pytest.approx(0.0)
    assert args.adapt_mix_rho == pytest.approx(0.0)
    assert args.source_policy_candidate_id == "K4_source_calibrated_no_update"


def test_source_calibrated_kshot_no_update_summary_is_not_k0_rejection():
    from scripts.train.train_hyperda_few_shot_adapt import (
        is_source_calibrated_kshot_no_update,
        support_gate_summary_for_source_calibrated_no_update,
    )

    class Args:
        K = 4
        policy_source = "source_side_episode_calibration"
        adapt_scope = "none"
        adaptation_steps = 0
        anchor_alpha = 0.0
        adapt_mix_rho = 0.0
        source_policy_candidate_id = "K4_source_calibrated_no_update"

    assert is_source_calibrated_kshot_no_update(Args()) is True

    summary = support_gate_summary_for_source_calibrated_no_update(Args())

    assert summary["stage3_posterior_decision"] == "no_update"
    assert summary["support_gate_status"] == "source_calibrated_no_update"
    assert summary["support_gate_policy_role"] == "source_side_policy_selected_no_update"
    assert summary["support_gate_reject_reason"] == []


def test_parse_args_accepts_trust_region_and_cycle_balanced_loss(monkeypatch, tmp_path):
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
            "--trust_region_mode",
            "groupwise",
            "--trust_total_radius",
            "3.8",
            "--trust_prompt_radius",
            "3.6",
            "--trust_gain_radius",
            "0.33",
            "--trust_coeff_radius",
            "0.68",
            "--trust_spatial_radius",
            "0.0",
            "--support_loss_reduction",
            "cycle_balanced",
        ],
    )

    args = runner.parse_args()
    assert args.trust_region_mode == "groupwise"
    assert args.trust_total_radius == pytest.approx(3.8)
    assert args.trust_prompt_radius == pytest.approx(3.6)
    assert args.trust_gain_radius == pytest.approx(0.33)
    assert args.trust_coeff_radius == pytest.approx(0.68)
    assert args.trust_spatial_radius == pytest.approx(0.0)
    assert args.support_loss_reduction == "cycle_balanced"


def test_parse_args_conservative_stage3_rejects_prompt_or_full_operator_scope(monkeypatch, tmp_path):
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
            "--stage3_posterior_policy",
            "conservative_coeff_posterior",
            "--adapt_scope",
            "safe_operator",
        ],
    )

    with pytest.raises(SystemExit):
        runner.parse_args()


def test_parse_args_conservative_stage3_defaults_to_coeff_only_with_support_gate(monkeypatch, tmp_path):
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
            "--stage3_posterior_policy",
            "conservative_coeff_posterior",
            "--support_gate",
            "auto",
            "--support_gate_min_delta",
            "0.001",
            "--support_gate_rootzone_tolerance",
            "0.02",
        ],
    )

    args = runner.parse_args()

    assert args.stage3_posterior_policy == "conservative_coeff_posterior"
    assert args.adapt_scope == "coeff_only"
    assert args.freeze_monthly_gain is True
    assert args.support_gate == "auto"
    assert args.support_gate_min_delta == pytest.approx(0.001)
    assert args.support_gate_rootzone_tolerance == pytest.approx(0.02)


def test_k0_conservative_stage3_requires_no_update(monkeypatch, tmp_path):
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
            "--stage3_posterior_policy",
            "conservative_coeff_posterior",
            "--adapt_scope",
            "none",
        ],
    )

    args = runner.parse_args()

    assert args.K == 0
    assert args.stage3_posterior_policy == "conservative_coeff_posterior"
    assert args.adapt_scope == "none"
    assert args.adaptation_steps == 0
    assert args.anchor_alpha == pytest.approx(0.0)


def test_parse_args_rejects_ridge_coeff_without_coeff_only_scope(monkeypatch, tmp_path):
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
            "--adapt_solver",
            "ridge_coeff",
            "--adapt_scope",
            "all",
        ],
    )

    with pytest.raises(SystemExit):
        runner.parse_args()


def test_support_gate_accepts_only_support_objective_improvement_without_rootzone_regression():
    from scripts.train.train_hyperda_few_shot_adapt import decide_support_gate

    before = {
        "standard_support_objective_full_support": 1.0,
        "standard_support_loss_full_support": 0.8,
        "standard_support_surface_loss_full_support": 0.4,
        "standard_support_rootzone_loss_full_support": 0.4,
    }

    accepted = decide_support_gate(
        before=before,
        after={
            "standard_support_objective_full_support": 0.9,
            "standard_support_loss_full_support": 0.7,
            "standard_support_surface_loss_full_support": 0.3,
            "standard_support_rootzone_loss_full_support": 0.39,
        },
        enabled=True,
        min_delta=0.0,
        rootzone_tolerance=0.0,
    )
    assert accepted["stage3_posterior_decision"] == "accepted"
    assert accepted["support_gate_status"] == "accepted"
    assert accepted["support_objective_delta"] == pytest.approx(-0.1)

    no_improvement = decide_support_gate(
        before=before,
        after={
            "standard_support_objective_full_support": 1.0,
            "standard_support_loss_full_support": 0.8,
            "standard_support_surface_loss_full_support": 0.4,
            "standard_support_rootzone_loss_full_support": 0.4,
        },
        enabled=True,
        min_delta=0.0,
        rootzone_tolerance=0.0,
    )
    assert no_improvement["stage3_posterior_decision"] == "rejected_to_k0_anchor"
    assert no_improvement["support_gate_status"] == "support_only_rejected_to_k0_anchor"
    assert no_improvement["support_gate_reject_reason"] == ["objective_not_improved"]

    rootzone_regression = decide_support_gate(
        before=before,
        after={
            "standard_support_objective_full_support": 0.9,
            "standard_support_loss_full_support": 0.7,
            "standard_support_surface_loss_full_support": 0.2,
            "standard_support_rootzone_loss_full_support": 0.5,
        },
        enabled=True,
        min_delta=0.0,
        rootzone_tolerance=0.0,
    )
    assert rootzone_regression["stage3_posterior_decision"] == "rejected_to_k0_anchor"
    assert rootzone_regression["support_gate_status"] == "support_only_rejected_to_k0_anchor"
    assert rootzone_regression["support_gate_reject_reason"] == ["rootzone_regression"]


def test_stage3_k0_posterior_metadata_declares_no_update_contract(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import build_stage3_target_posterior_state

    anchor = {
        "target_prompt.latent": torch.zeros(2),
        "residual_gain.log_gain": torch.zeros(2, 12),
    }
    posterior = build_stage3_target_posterior_state(
        anchor_state=anchor,
        final_state={name: tensor.clone() for name, tensor in anchor.items()},
        K=0,
        adapt_scope="none",
        anchor_alpha=0.0,
        adaptation_steps=0,
        target_labels_loaded=False,
        target_labels_used=False,
        source_prior_hash_before="sourcehash",
        source_prior_hash_after="sourcehash",
        stage3_posterior_policy="conservative_coeff_posterior",
        stage3_posterior_decision="no_update",
        support_gate_status="skipped_k0_no_support",
    )
    metadata = posterior["metadata"]

    assert metadata["stage3_no_update_contract"] == "K0_fixed_no_update_source_prior_identity"
    assert metadata["source_prior_unchanged"] is True
    assert metadata["k0_target_drift_zero"] is True
    assert metadata["target_labels_used_for_adaptation"] is False
    assert metadata["support_gate_policy_role"] == "not_applicable_k0_no_support"


def test_stage3_kshot_metadata_marks_support_gate_as_diagnostic():
    from scripts.train.train_hyperda_few_shot_adapt import build_stage3_target_posterior_state

    anchor = {"target_prompt.latent": torch.zeros(2)}
    final = {"target_prompt.latent": torch.ones(2)}
    posterior = build_stage3_target_posterior_state(
        anchor_state=anchor,
        final_state=final,
        K=4,
        adapt_scope="coeff_gain",
        anchor_alpha=0.5,
        adaptation_steps=10,
        target_labels_loaded=True,
        target_labels_used=True,
        source_prior_hash_before="sourcehash",
        source_prior_hash_after="sourcehash",
        stage3_posterior_policy="conservative_coeff_posterior",
        stage3_posterior_decision="accepted",
        support_gate_status="accepted",
    )

    assert posterior["metadata"]["paper_selection_basis"] == "source_side_safe_policy_only"
    assert posterior["metadata"]["support_gate_policy_role"] == "target_support_only_diagnostic_not_paper_selection"


def test_stage3_rejected_metadata_records_k0_anchor_hash():
    from scripts.train.train_hyperda_few_shot_adapt import build_stage3_target_posterior_state

    anchor = {"target_adapter_coefficient_residual_b.logit_delta": torch.zeros(3)}
    posterior = build_stage3_target_posterior_state(
        anchor_state=anchor,
        final_state={name: tensor.clone() for name, tensor in anchor.items()},
        K=12,
        adapt_scope="coeff_only",
        anchor_alpha=0.25,
        adaptation_steps=80,
        target_labels_loaded=True,
        target_labels_used=True,
        source_prior_hash_before="sourcehash",
        source_prior_hash_after="sourcehash",
        stage3_posterior_policy="conservative_coeff_posterior",
        stage3_posterior_decision="rejected_to_k0_anchor",
        support_gate_status="support_only_rejected_to_k0_anchor",
        paper_selection_basis="source_policy_or_gate_rejected_to_k0_anchor",
    )
    metadata = posterior["metadata"]

    assert metadata["stage3_posterior_decision"] == "rejected_to_k0_anchor"
    assert metadata["support_only_gate_status"] == "support_only_rejected_to_k0_anchor"
    assert metadata["k0_anchor_state_hash"] == posterior["target_adapter_anchor_hash"]
    assert metadata["stage3_acceptance_basis"] == "source_policy_or_gate_rejected_to_k0_anchor"


def test_rejected_kshot_metadata_is_diagnostic_not_paper_facing():
    from scripts.train.train_hyperda_few_shot_adapt import paper_facing_status_for_stage3

    status = paper_facing_status_for_stage3(
        K=4,
        policy_source="source_side_episode_calibration",
        stage3_posterior_decision="rejected_to_k0_anchor",
    )

    assert status["paper_facing_run"] is False
    assert status["diagnostic_run_reason"] == "source_policy_or_gate_rejected_to_k0_anchor"


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


def test_load_source_checkpoint_for_few_shot_preserves_robust_context_encoder(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import load_source_checkpoint_for_few_shot

    ckpt_path = tmp_path / "source_robust.pt"
    _write_robust_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    assert isinstance(state.prompt_encoder, RobustInputSideDAPromptEncoder)
    assert state.source_config["context_encoder"] == "robust_input_side_da_diagnostics"


def test_few_shot_target_context_prompt_state_uses_raw_domain_for_raw_da_encoder(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        build_few_shot_target_context_prompt_state,
        load_source_checkpoint_for_few_shot,
    )

    ckpt_path = tmp_path / "source_raw_robust.pt"
    _write_raw_robust_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    state.normalization["ch_mean"] = [100.0] * 12
    state.normalization["ch_std"] = [10.0] * 12
    x = np.zeros((12, 2, 2), dtype=np.float32)
    x[5] = 10.0
    x[9] = 4.0

    prompt_state = build_few_shot_target_context_prompt_state(
        state=state,
        samples=[
            {
                "x": x,
                "month": 1,
                "date_str": "2015-01-15",
                "region_mask": np.ones((2, 2), dtype=np.float32),
            }
        ],
        target_region="US-R1",
        device=torch.device("cpu"),
        context_hash="ctxhash",
    )

    assert isinstance(state.prompt_encoder, RobustInputSideDAPromptEncoder)
    assert state.source_config["context_encoder"] == "robust_input_side_da_diagnostics_raw"
    assert prompt_state["metadata"]["prompt_diagnostic_input_domain"] == "raw_input_side"
    assert prompt_state["metadata"]["normalized_input_used_for_prompt_diagnostics"] is False


def test_load_source_checkpoint_for_few_shot_preserves_rank_gated_hyperda_config(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import load_source_checkpoint_for_few_shot

    ckpt_path = tmp_path / "source_rank_gated.pt"
    _write_rank_gated_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    assert state.model.hyper_coeff_generator == "shared_layer_aware_rank_gated"
    assert state.model.hyper_rank_gate_top_k == 2
    assert state.model.hyper_adapter_param_style == "dora_like_gain"
    assert state.model.shared_coeff_generator is not None
    assert state.model.hyper_adapter_b.basis_gain_delta is not None


def test_load_source_checkpoint_for_few_shot_preserves_source_saliency_prior_config(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import load_source_checkpoint_for_few_shot

    ckpt_path = tmp_path / "source_saliency.pt"
    _write_source_saliency_prior_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    assert state.model.hyper_coeff_generator == "shared_layer_aware_rank_gated_stable"
    assert state.model.hyper_source_saliency_prior_beta == pytest.approx(0.5)
    assert state.model.hyper_source_saliency_prior_path == "artifacts/prior/source_fit.pt"
    assert state.model.hyper_prompt_manifold_reliability is True
    assert state.model.hyper_prompt_manifold_reliability_strength == pytest.approx(0.25)
    assert state.model.shared_coeff_generator is not None
    assert torch.isfinite(state.model.shared_coeff_generator.saliency_prior).all()
    assert tuple(state.model.shared_coeff_generator.saliency_prior.shape) == (3, 5)


def test_apply_adapt_scope_controls_trainable_target_variables(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        apply_adapt_scope,
        group_target_parameter_counts,
        load_source_checkpoint_for_few_shot,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    expected = {
        "none": [],
        "safe_operator": [
            "target_prompt.latent",
            "target_prompt.proj.weight",
            "target_prompt.proj.bias",
            "target_adapter_coefficient_residual_b.logit_delta",
            "target_adapter_coefficient_residual_d2.logit_delta",
            "target_adapter_coefficient_residual_d1.logit_delta",
            "residual_gain.gain_delta",
            "residual_gain.bias",
        ],
        "prompt_only": [
            "target_prompt.latent",
            "target_prompt.proj.weight",
            "target_prompt.proj.bias",
        ],
        "coeff_only": [
            "target_adapter_coefficient_residual_b.logit_delta",
            "target_adapter_coefficient_residual_d2.logit_delta",
            "target_adapter_coefficient_residual_d1.logit_delta",
        ],
        "gain_only": [
            "residual_gain.gain_delta",
            "residual_gain.bias",
        ],
        "coeff_gain": [
            "target_adapter_coefficient_residual_b.logit_delta",
            "target_adapter_coefficient_residual_d2.logit_delta",
            "target_adapter_coefficient_residual_d1.logit_delta",
            "residual_gain.gain_delta",
            "residual_gain.bias",
        ],
    }

    for scope, names in expected.items():
        state.model.freeze_source_prior_for_target_adaptation()
        apply_adapt_scope(state.model, scope)
        assert state.model.target_trainable_parameter_names() == names

    counts = group_target_parameter_counts(state.model)
    assert counts["target_prompt"] == 0
    assert counts["adapter_coeff_bottleneck"] == 3
    assert counts["adapter_coeff_dec2"] == 3
    assert counts["adapter_coeff_dec1"] == 3
    assert counts["monthly_gain"] == 48
    assert counts["total"] == 57


def test_apply_adapt_scope_safe_operator_excludes_spatial_refine_and_source_prior(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import apply_adapt_scope, load_source_checkpoint_for_few_shot

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
        enable_target_spatial_refine=True,
    )

    trainable = apply_adapt_scope(state.model, "safe_operator")

    assert trainable == [
        "target_prompt.latent",
        "target_prompt.proj.weight",
        "target_prompt.proj.bias",
        "target_adapter_coefficient_residual_b.logit_delta",
        "target_adapter_coefficient_residual_d2.logit_delta",
        "target_adapter_coefficient_residual_d1.logit_delta",
        "residual_gain.gain_delta",
        "residual_gain.bias",
    ]
    assert not any(name.startswith("target_spatial_refine.") for name in trainable)
    assert not any(
        name.startswith(prefix)
        for name in trainable
        for prefix in ("enc", "dec", "bottleneck", "film", "hyper_adapter", "head")
    )


def test_adapt_scope_all_keeps_current_target_variables_trainable(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import apply_adapt_scope, load_source_checkpoint_for_few_shot

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    before = state.model.target_trainable_parameter_names()
    apply_adapt_scope(state.model, "all")

    assert state.model.target_trainable_parameter_names() == before
    assert "residual_gain.gain_delta" in before
    assert "residual_gain.bias" in before


def test_freeze_monthly_gain_overrides_adapt_scope(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import apply_adapt_scope, load_source_checkpoint_for_few_shot

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    apply_adapt_scope(state.model, "all", freeze_monthly_gain=True)
    trainable = state.model.target_trainable_parameter_names()

    assert "residual_gain.gain_delta" not in trainable
    assert "residual_gain.bias" not in trainable
    assert "target_prompt.latent" in trainable
    assert "target_adapter_coefficient_residual_b.logit_delta" in trainable


def test_gain_only_with_freeze_monthly_gain_has_no_trainable_parameters(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        apply_adapt_scope,
        group_target_parameter_counts,
        load_source_checkpoint_for_few_shot,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    apply_adapt_scope(state.model, "gain_only", freeze_monthly_gain=True)

    assert state.model.target_trainable_parameter_names() == []
    counts = group_target_parameter_counts(state.model)
    assert counts["monthly_gain"] == 0
    assert counts["total"] == 0


def test_freeze_monthly_gain_drift_stays_zero_when_other_groups_change(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        apply_adapt_scope,
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
    apply_adapt_scope(state.model, "all", freeze_monthly_gain=True)
    anchor = extract_target_adapter_state(state.model)

    with torch.no_grad():
        state.model.target_prompt.latent.add_(1.0)
        state.model.target_adapter_coefficient_residual_b.logit_delta.add_(1.0)

    drift = target_parameter_l2_drift(anchor, extract_target_adapter_state(state.model))

    assert drift["monthly_gain"] == pytest.approx(0.0)
    assert drift["target_prompt"] > 0.0
    assert drift["adapter_coeff_bottleneck"] > 0.0


def test_conservative_coeff_gain_trainable_groups_exclude_prompt(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        apply_adapt_scope,
        load_source_checkpoint_for_few_shot,
        trainable_target_groups_for_names,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    trainable = apply_adapt_scope(state.model, "coeff_gain")

    assert trainable_target_groups_for_names(trainable) == [
        "adapter_coefficient_residuals",
        "monthly_residual_gain",
    ]
    assert not any(name.startswith("target_prompt.") for name in trainable)
    assert any(name.startswith("residual_gain.") for name in trainable)


def test_conservative_coeff_posterior_trainable_groups_are_coeff_only_by_default(tmp_path, monkeypatch):
    from scripts.train import train_hyperda_few_shot_adapt as runner
    from scripts.train.train_hyperda_few_shot_adapt import (
        apply_adapt_scope,
        load_source_checkpoint_for_few_shot,
        trainable_target_groups_for_names,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_hyperda_few_shot_adapt.py",
            "--source_checkpoint",
            str(ckpt_path),
            "--target_region",
            "US-R1",
            "--K",
            "4",
            "--stage3_posterior_policy",
            "conservative_coeff_posterior",
        ],
    )

    args = runner.parse_args()
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    trainable = apply_adapt_scope(state.model, args.adapt_scope, freeze_monthly_gain=args.freeze_monthly_gain)

    assert args.adapt_scope == "coeff_only"
    assert args.freeze_monthly_gain is True
    assert trainable == [
        "target_adapter_coefficient_residual_b.logit_delta",
        "target_adapter_coefficient_residual_d2.logit_delta",
        "target_adapter_coefficient_residual_d1.logit_delta",
    ]
    assert trainable_target_groups_for_names(trainable) == ["adapter_coefficient_residuals"]


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
            "schedule_label": "unit_schedule",
            "requested_lr": 1e-3,
            "requested_max_steps": 100,
            "requested_anchor_alpha": 0.75,
            "requested_weight_decay": 1e-4,
            "requested_grad_clip": 1.0,
            "standard_support_loss_before_full_support": 0.10,
            "standard_support_loss_after_full_support": 0.08,
            "standard_support_loss_delta_full_support": -0.02,
            "ridge_design_loss_before_sampled_pixels": None,
            "ridge_design_loss_after_sampled_pixels": None,
            "ridge_design_loss_delta_sampled_pixels": None,
            "target_parameter_l2_drift_pre_anchor": {"total": 0.4, "target_prompt": 0.4},
            "target_parameter_l2_drift_post_anchor": {"total": 0.1, "target_prompt": 0.1},
            "target_parameter_l2_drift": {"total": 0.1, "target_prompt": 0.1},
            "trust_region_mode": "groupwise",
            "trust_total_radius": 3.8,
            "trust_prompt_radius": 3.6,
            "trust_gain_radius": 0.33,
            "trust_coeff_radius": 0.68,
            "trust_spatial_radius": 0.0,
            "trust_projection_diagnostics": {
                "trust_projection_step_count": 25,
                "trust_projection_applied_count": 3,
                "trust_projection_pre_step_drift_max_total": 4.0,
                "trust_projection_post_step_drift_max_total": 3.8,
            },
            "support_loss_reduction": "cycle_balanced",
            "support_gradient_diagnostics": {
                "support_gradient_diagnostics_label_source": "target_support_only",
                "support_cycle_count": 2,
                "support_gradient_cosine_mean": -0.1,
                "support_gradient_cosine_min": -0.5,
                "support_gradient_negative_fraction": 0.5,
                "support_cycle_loss_improvement_mean": 0.2,
                "support_cycle_loss_improvement_std": 0.05,
            },
            "target_parameter_count_by_group": {"target_prompt": 44, "total": 101},
            "requires_grad_parameter_count": 101,
            "optimizer_parameter_count": 44,
            "adapt_scope": "safe_operator",
            "adapt_solver": "adamw",
            "freeze_monthly_gain": True,
            "ridge_lambda": 2.0,
            "ridge_clip_coeff_norm": 0.75,
            "ridge_trust_region_radius": 0.5,
            "ridge_max_feature_pixels": 2048,
            "ridge_standardize_features": True,
            "ridge_diagnostics": {
                "status": "solved",
                "coefficient_norm": 0.2,
                "delta_norm": 0.2,
                "raw_delta_norm": 0.3,
                "support_count": 2,
                "masked_pixel_count": 10,
                "masked_observation_count": 20,
                "feature_pixel_count": 8,
                "feature_observation_count": 16,
                "feature_dim": 9,
                "condition_number": 4.0,
                "rank": 9,
            },
            "audit_identity": False,
            "stage3_posterior_policy": "conservative_coeff_posterior",
            "stage3_posterior_decision": "accepted",
            "support_gate_enabled": True,
            "support_gate_status": "accepted",
            "support_gate_reject_reason": [],
            "support_objective_before": 0.14,
            "support_objective_after": 0.11,
            "support_objective_delta": -0.03,
            "support_surface_loss_before": 0.05,
            "support_surface_loss_after": 0.04,
            "support_rootzone_loss_before": 0.05,
            "support_rootzone_loss_after": 0.045,
            "paper_facing_run": True,
            "policy_source": "source_side_episode_calibration",
            "safe_policy_json": "/tmp/safe_policy.json",
            "safe_policy_json_sha256": "policysha",
            "safe_policy": {
                "schema_version": "hyperda_safe_policy_v1",
                "policy_hash": "embedded_policy_hash",
                "policy_source": "source_side_episode_calibration",
            },
            "source_episode_regions": ["US-R2", "US-R3"],
            "source_checkpoint_sha256": "abc123",
            "staged_source_checkpoint_sha256": "abc123",
            "source_stage_checkpoint_provenance": "phase4_hyperda_staged",
            "support_manifest_hash": "support_manifest_sha",
            "target_support_count": 2,
            "target_labels_loaded_for_adaptation": True,
            "target_labels_used_for_adaptation": True,
            "adapt_batch_size": 4,
            "max_steps_requested": 100,
            "actual_optimizer_steps": 25,
            "optimizer_steps_run": 25,
            "support_batch_count": 1,
            "effective_support_passes": 25.0,
            "weight_decay": 1e-4,
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
    assert cfg["schedule_label"] == "unit_schedule"
    assert cfg["requested_lr"] == pytest.approx(1e-3)
    assert cfg["requested_max_steps"] == 100
    assert cfg["requested_anchor_alpha"] == pytest.approx(0.75)
    assert cfg["requested_weight_decay"] == pytest.approx(1e-4)
    assert cfg["requested_grad_clip"] == pytest.approx(1.0)
    assert cfg["source_anchor_hyperparameter_source"] == "source_side_episodic_validation_preregistered"
    assert cfg["support_loss_before"] == pytest.approx(0.10)
    assert cfg["support_loss_after"] == pytest.approx(0.08)
    assert cfg["support_final_loss"] == pytest.approx(0.08)
    assert cfg["support_loss_delta"] == pytest.approx(-0.02)
    assert cfg["standard_support_loss_before_full_support"] == pytest.approx(0.10)
    assert cfg["standard_support_loss_after_full_support"] == pytest.approx(0.08)
    assert cfg["standard_support_loss_delta_full_support"] == pytest.approx(-0.02)
    assert cfg["ridge_design_loss_before_sampled_pixels"] is None
    assert cfg["ridge_design_loss_after_sampled_pixels"] is None
    assert cfg["ridge_design_loss_delta_sampled_pixels"] is None
    assert cfg["target_parameter_l2_drift_pre_anchor"]["total"] == pytest.approx(0.4)
    assert cfg["target_parameter_l2_drift_post_anchor"]["total"] == pytest.approx(0.1)
    assert cfg["target_parameter_l2_drift"]["total"] == pytest.approx(0.1)
    assert cfg["trust_region_mode"] == "groupwise"
    assert cfg["trust_prompt_radius"] == pytest.approx(3.6)
    assert cfg["trust_coeff_radius"] == pytest.approx(0.68)
    assert cfg["trust_projection_diagnostics"]["trust_projection_applied_count"] == 3
    assert cfg["support_loss_reduction"] == "cycle_balanced"
    assert cfg["support_gradient_diagnostics"]["support_cycle_count"] == 2
    assert cfg["support_gradient_cosine_mean"] == pytest.approx(-0.1)
    assert cfg["support_gradient_negative_fraction"] == pytest.approx(0.5)
    assert cfg["support_cycle_loss_improvement_mean"] == pytest.approx(0.2)
    assert cfg["target_parameter_count_by_group"]["target_prompt"] == 44
    assert cfg["requires_grad_parameter_count"] == 101
    assert cfg["optimizer_parameter_count"] == 44
    assert cfg["adapt_scope"] == "safe_operator"
    assert cfg["adapt_solver"] == "adamw"
    assert cfg["freeze_monthly_gain"] is True
    assert cfg["ridge_lambda"] == pytest.approx(2.0)
    assert cfg["ridge_max_feature_pixels"] == 2048
    assert cfg["ridge_standardize_features"] is True
    assert cfg["ridge_diagnostics"]["status"] == "solved"
    assert cfg["ridge_coefficient_norm"] == pytest.approx(0.2)
    assert cfg["ridge_masked_observation_count"] == 20
    assert cfg["ridge_feature_pixel_count"] == 8
    assert cfg["ridge_feature_observation_count"] == 16
    assert cfg["stage3_posterior_policy"] == "conservative_coeff_posterior"
    assert cfg["stage3_posterior_decision"] == "accepted"
    assert cfg["support_gate_enabled"] is True
    assert cfg["support_gate_status"] == "accepted"
    assert cfg["support_gate_reject_reason"] == []
    assert cfg["support_objective_before"] == pytest.approx(0.14)
    assert cfg["support_objective_after"] == pytest.approx(0.11)
    assert cfg["support_objective_delta"] == pytest.approx(-0.03)
    assert cfg["support_surface_loss_before"] == pytest.approx(0.05)
    assert cfg["support_surface_loss_after"] == pytest.approx(0.04)
    assert cfg["support_rootzone_loss_before"] == pytest.approx(0.05)
    assert cfg["support_rootzone_loss_after"] == pytest.approx(0.045)
    assert cfg["paper_facing_run"] is True
    assert cfg["policy_source"] == "source_side_episode_calibration"
    assert cfg["safe_policy_json_sha256"] == "policysha"
    assert cfg["safe_policy_hash"] == "embedded_policy_hash"
    assert cfg["stage3_acceptance_basis"] == "source_side_safe_policy_only"
    assert cfg["support_only_gate_status"] == "accepted"
    assert cfg["k0_anchor_state_hash"] == saved["stage3_posterior_state_dict"]["target_adapter_anchor_hash"]
    assert cfg["source_policy_candidate_id"] == ""
    assert cfg["source_episode_regions"] == ["US-R2", "US-R3"]
    assert cfg["support_manifest_hash"] == "support_manifest_sha"
    assert cfg["stage3_posterior_state"]["stage3_posterior_policy"] == "conservative_coeff_posterior"
    assert cfg["stage3_posterior_state"]["stage3_posterior_decision"] == "accepted"
    assert cfg["stage3_posterior_state"]["support_gate_status"] == "accepted"
    assert saved["stage3_posterior_state_dict"]["metadata"]["stage3_posterior_policy"] == "conservative_coeff_posterior"
    assert saved["stage3_posterior_state_dict"]["metadata"]["stage3_posterior_decision"] == "accepted"
    assert saved["stage3_posterior_state_dict"]["metadata"]["support_gate_status"] == "accepted"
    assert cfg["source_checkpoint_sha256"] == "abc123"
    assert cfg["staged_source_checkpoint_sha256"] == "abc123"
    assert cfg["source_stage_checkpoint_provenance"] == "phase4_hyperda_staged"
    assert cfg["target_support_count"] == 2
    assert cfg["target_labels_loaded_for_adaptation"] is True
    assert cfg["target_labels_used_for_adaptation"] is True
    assert cfg["adapt_batch_size"] == 4
    assert cfg["max_steps_requested"] == 100
    assert cfg["actual_optimizer_steps"] == 25
    assert cfg["optimizer_steps_run"] == 25
    assert cfg["support_batch_count"] == 1
    assert cfg["effective_support_passes"] == pytest.approx(25.0)
    assert cfg["weight_decay"] == pytest.approx(1e-4)
    assert cfg["prompt_policy"] == "target_context_monthly_prompt_prototypes"
    assert cfg["eval_input_usage"] == "none_for_prompt_update"
    assert cfg["target_context_prompt_state_summary"]["context_date_hash"] == "contexthash"
    assert saved["target_context_prompt_state"]["context_hash"] == "contexthash"
    assert cfg["trainable_parameter_names"]
    assert not any(name.startswith("prompt_encoder") for name in cfg["trainable_parameter_names"])
    assert cfg["frozen_source_groups"] == [
        "source_backbone",
        "source_head",
        "prompt_encoder",
        "film",
        "basis_adapter_hypernetwork",
        "adapter_basis_bank",
    ]
    assert cfg["trainable_target_groups"] == [
        "target_prompt",
        "adapter_coefficient_residuals",
        "monthly_residual_gain",
    ]


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
    assert cfg["method"] == "hyperda_diagnostic_few_shot_k12"
    assert cfg["paper_facing_run"] is False
    assert cfg["diagnostic_run_reason"] == "missing_source_side_safe_policy_json"
    assert saved["target_context_prompt_state"]["context_date_hash"] == "contexthash"
    assert saved["target_context_prompt_state"]["context_hash"] == "contexthash"
    assert cfg["target_context_prompt_state"]["context_date_hash"] == "contexthash"


def test_save_few_shot_checkpoint_rejects_target_context_hash_mismatch(tmp_path):
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
    prompt_state = {
        "schema_version": "target_context_prompt_state_v1",
        "prompt_source": "target_context_monthly_prompt_prototypes",
        "label_usage": "none",
        "context_hash": "prompt-context-hash",
        "monthly_counts": {str(i): 0 for i in range(1, 13)},
        "global_prototype": torch.zeros(8),
        "monthly_prototypes": {str(i): None for i in range(1, 13)},
        "metadata": {},
    }

    with pytest.raises(ValueError, match="target_context_dates_hash"):
        save_few_shot_checkpoint(
            path=tmp_path / "checkpoint_final_preregistered.pt",
            state=state,
            optimizer_state_dict={},
            config={
                "K": 0,
                "adaptation_setting": "zero_shot_context",
                "target_region": "US-R1",
                "target_context_dates_hash": "split-context-hash",
            },
            target_context_prompt_state=prompt_state,
            train_history=[],
        )


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


def test_pre_anchor_drift_is_larger_than_post_anchor_drift(tmp_path):
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

    with torch.no_grad():
        state.model.target_prompt.latent.add_(2.0)
        state.model.target_adapter_coefficient_residual_b.logit_delta.add_(1.0)
        state.model.residual_gain.bias.add_(0.5)

    pre_anchor = target_parameter_l2_drift(anchor_state, extract_target_adapter_state(state.model))
    apply_source_anchor_interpolation(state.model, anchor_state, alpha=0.25)
    post_anchor = target_parameter_l2_drift(anchor_state, extract_target_adapter_state(state.model))

    assert post_anchor["total"] == pytest.approx(pre_anchor["total"] * 0.25)
    assert post_anchor["target_prompt"] == pytest.approx(pre_anchor["target_prompt"] * 0.25)
    assert post_anchor["adapter_coeff_bottleneck"] == pytest.approx(pre_anchor["adapter_coeff_bottleneck"] * 0.25)
    assert post_anchor["monthly_gain"] == pytest.approx(pre_anchor["monthly_gain"] * 0.25)


def test_project_target_state_groupwise_caps_drift_and_preserves_source(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        extract_target_adapter_state,
        load_source_checkpoint_for_few_shot,
        project_target_state_to_trust_region,
        target_parameter_l2_drift,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    anchor = extract_target_adapter_state(state.model)
    source_prior = {
        name: tensor.detach().clone()
        for name, tensor in state.model.state_dict().items()
        if name not in anchor
    }

    with torch.no_grad():
        state.model.target_prompt.latent.add_(10.0)
        state.model.target_adapter_coefficient_residual_b.logit_delta.add_(5.0)
        state.model.target_adapter_coefficient_residual_d2.logit_delta.add_(5.0)
        state.model.target_adapter_coefficient_residual_d1.logit_delta.add_(5.0)
        state.model.residual_gain.bias.add_(2.0)

    result = project_target_state_to_trust_region(
        state.model,
        anchor,
        mode="groupwise",
        total_radius=10.0,
        prompt_radius=0.5,
        gain_radius=0.25,
        coeff_radius=0.75,
        spatial_radius=0.0,
    )
    drift = target_parameter_l2_drift(anchor, extract_target_adapter_state(state.model))

    assert result["projection_applied"] is True
    assert drift["target_prompt"] <= 0.5 + 1e-6
    assert drift["monthly_gain"] <= 0.25 + 1e-6
    coeff_drift = (
        drift["adapter_coeff_bottleneck"] ** 2
        + drift["adapter_coeff_dec2"] ** 2
        + drift["adapter_coeff_dec1"] ** 2
    ) ** 0.5
    assert coeff_drift <= 0.75 + 1e-6
    for name, tensor in state.model.state_dict().items():
        if name not in anchor:
            assert torch.allclose(tensor, source_prior[name])


def test_target_parameter_l2_drift_uses_fine_grained_groups(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
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

    with torch.no_grad():
        state.model.target_adapter_coefficient_residual_b.logit_delta.add_(1.0)
        state.model.target_adapter_coefficient_residual_d2.logit_delta.add_(2.0)
        state.model.target_adapter_coefficient_residual_d1.logit_delta.add_(3.0)
        state.model.residual_gain.bias.add_(4.0)

    drift = target_parameter_l2_drift(anchor_state, extract_target_adapter_state(state.model))

    assert drift["adapter_coeff_bottleneck"] == pytest.approx(3.0 ** 0.5)
    assert drift["adapter_coeff_dec2"] == pytest.approx(12.0 ** 0.5)
    assert drift["adapter_coeff_dec1"] == pytest.approx(27.0 ** 0.5)
    assert drift["monthly_gain"] == pytest.approx(384.0 ** 0.5)
    assert "adapter_coefficient_residuals" not in drift


def test_ridge_coeff_vector_helpers_touch_only_coefficients(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        coefficient_residual_vector,
        load_source_checkpoint_for_few_shot,
        set_coefficient_residual_vector,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    before = {name: tensor.detach().clone() for name, tensor in state.model.state_dict().items()}

    vector = torch.arange(9, dtype=torch.float32)
    set_coefficient_residual_vector(state.model, vector)

    assert torch.allclose(coefficient_residual_vector(state.model), vector)
    changed = [
        name
        for name, tensor in state.model.state_dict().items()
        if not torch.allclose(tensor, before[name])
    ]
    assert changed == [
        "target_adapter_coefficient_residual_b.logit_delta",
        "target_adapter_coefficient_residual_d2.logit_delta",
        "target_adapter_coefficient_residual_d1.logit_delta",
    ]


def test_solve_ridge_coefficients_returns_finite_and_stronger_lambda_reduces_norm():
    from scripts.train.train_hyperda_few_shot_adapt import solve_ridge_coefficients

    design_t_design = torch.tensor([[4.0, 1.0], [1.0, 2.0]])
    design_t_residual = torch.tensor([2.0, 1.0])

    weak = solve_ridge_coefficients(design_t_design, design_t_residual, ridge_lambda=0.01)
    strong = solve_ridge_coefficients(design_t_design, design_t_residual, ridge_lambda=10.0)

    assert torch.isfinite(weak.delta).all()
    assert torch.isfinite(strong.delta).all()
    assert torch.linalg.vector_norm(strong.delta) < torch.linalg.vector_norm(weak.delta)
    assert strong.diagnostics["ridge_lambda"] == pytest.approx(10.0)
    assert strong.diagnostics["condition_number"] >= 1.0


def test_run_ridge_coeff_adaptation_updates_only_coefficients(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        apply_adapt_scope,
        coefficient_residual_vector,
        extract_target_adapter_state,
        load_source_checkpoint_for_few_shot,
        run_ridge_coeff_adaptation,
        target_parameter_l2_drift,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    apply_adapt_scope(state.model, "coeff_only")
    anchor = extract_target_adapter_state(state.model)
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
        "increment_surface": torch.ones(1, 16, 16) * 0.1,
        "increment_rootzone": torch.zeros(1, 16, 16),
        "loss_mask": torch.ones(1, 16, 16),
        "forecast_surface": torch.zeros(1, 16, 16),
        "forecast_rootzone": torch.zeros(1, 16, 16),
    }
    loader = torch.utils.data.DataLoader([batch], batch_size=None)

    diagnostics = run_ridge_coeff_adaptation(
        state=state,
        loader=loader,
        device=torch.device("cpu"),
        target_context_prompt_state=prompt_state,
        normalize_increment=True,
        ridge_lambda=0.1,
        ridge_clip_coeff_norm=0.5,
        ridge_trust_region_radius=0.5,
        ridge_max_feature_pixels=0,
        ridge_standardize_features=False,
        surface_weight=3.0,
        rootzone_weight=1.0,
        use_lat_weighted_loss=False,
    )

    assert diagnostics["status"] in {"solved", "pinv_solved"}
    assert diagnostics["support_count"] == 1
    assert diagnostics["masked_pixel_count"] == 256
    assert diagnostics["masked_observation_count"] == 512
    assert diagnostics["feature_pixel_count"] == 256
    assert diagnostics["feature_observation_count"] == 512
    assert diagnostics["support_loss_before"] is not None
    assert diagnostics["support_loss_after"] is not None
    assert diagnostics["feature_dim"] == coefficient_residual_vector(state.model).numel()
    assert diagnostics["coefficient_norm"] <= 0.5 + 1e-6
    drift = target_parameter_l2_drift(anchor, extract_target_adapter_state(state.model))
    assert drift["target_prompt"] == pytest.approx(0.0)
    assert drift["monthly_gain"] == pytest.approx(0.0)
    assert drift["total"] == pytest.approx(
        (
            drift["adapter_coeff_bottleneck"] ** 2
            + drift["adapter_coeff_dec2"] ** 2
            + drift["adapter_coeff_dec1"] ** 2
        )
        ** 0.5
    )


def test_ridge_coeff_subsampling_limits_feature_observations(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        apply_adapt_scope,
        load_source_checkpoint_for_few_shot,
        run_ridge_coeff_adaptation,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    apply_adapt_scope(state.model, "coeff_only")
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
        "increment_surface": torch.ones(1, 16, 16) * 0.1,
        "increment_rootzone": torch.zeros(1, 16, 16),
        "loss_mask": torch.ones(1, 16, 16),
        "forecast_surface": torch.zeros(1, 16, 16),
        "forecast_rootzone": torch.zeros(1, 16, 16),
    }
    loader = torch.utils.data.DataLoader([batch], batch_size=None)

    diagnostics = run_ridge_coeff_adaptation(
        state=state,
        loader=loader,
        device=torch.device("cpu"),
        target_context_prompt_state=prompt_state,
        normalize_increment=True,
        ridge_lambda=0.1,
        ridge_clip_coeff_norm=0.5,
        ridge_trust_region_radius=0.5,
        ridge_max_feature_pixels=10,
        ridge_standardize_features=True,
        surface_weight=3.0,
        rootzone_weight=1.0,
        use_lat_weighted_loss=False,
    )

    assert diagnostics["masked_pixel_count"] == 256
    assert diagnostics["masked_observation_count"] == 512
    assert diagnostics["feature_pixel_count"] == 10
    assert diagnostics["feature_observation_count"] == 20
    assert diagnostics["ridge_standardize_features"] is True


def test_ridge_coeff_subsampling_counts_only_masked_feature_observations(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        apply_adapt_scope,
        load_source_checkpoint_for_few_shot,
        run_ridge_coeff_adaptation,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    apply_adapt_scope(state.model, "coeff_only")
    prompt_state = {
        "schema_version": "target_context_prompt_state_v1",
        "prompt_source": "target_context_monthly_prompt_prototypes",
        "label_usage": "none",
        "monthly_counts": {str(i): 0 for i in range(1, 13)},
        "global_prototype": torch.zeros(8),
        "monthly_prototypes": {str(i): None for i in range(1, 13)},
        "metadata": {},
    }
    sparse_mask = torch.zeros(1, 16, 16)
    sparse_mask[:, :4, :4] = 1.0
    batch = {
        "x": torch.zeros(1, 12, 16, 16),
        "months": torch.tensor([5], dtype=torch.long),
        "increment_surface": torch.ones(1, 16, 16) * 0.1,
        "increment_rootzone": torch.zeros(1, 16, 16),
        "loss_mask": sparse_mask,
        "forecast_surface": torch.zeros(1, 16, 16),
        "forecast_rootzone": torch.zeros(1, 16, 16),
    }
    loader = torch.utils.data.DataLoader([batch], batch_size=None)

    diagnostics = run_ridge_coeff_adaptation(
        state=state,
        loader=loader,
        device=torch.device("cpu"),
        target_context_prompt_state=prompt_state,
        normalize_increment=True,
        ridge_lambda=0.1,
        ridge_clip_coeff_norm=0.5,
        ridge_trust_region_radius=0.5,
        ridge_max_feature_pixels=100,
        ridge_standardize_features=False,
        surface_weight=3.0,
        rootzone_weight=1.0,
        use_lat_weighted_loss=False,
    )

    assert diagnostics["masked_pixel_count"] == 16
    assert diagnostics["masked_observation_count"] == 32
    assert diagnostics["feature_pixel_count"] == 16
    assert diagnostics["feature_observation_count"] == 32


def test_ridge_coeff_feature_pixel_cap_applies_across_support_batches(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        apply_adapt_scope,
        load_source_checkpoint_for_few_shot,
        run_ridge_coeff_adaptation,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    apply_adapt_scope(state.model, "coeff_only")
    prompt_state = {
        "schema_version": "target_context_prompt_state_v1",
        "prompt_source": "target_context_monthly_prompt_prototypes",
        "label_usage": "none",
        "monthly_counts": {str(i): 0 for i in range(1, 13)},
        "global_prototype": torch.zeros(8),
        "monthly_prototypes": {str(i): None for i in range(1, 13)},
        "metadata": {},
    }

    def batch(month: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.zeros(1, 12, 16, 16),
            "months": torch.tensor([month], dtype=torch.long),
            "increment_surface": torch.ones(1, 16, 16) * 0.1,
            "increment_rootzone": torch.zeros(1, 16, 16),
            "loss_mask": torch.ones(1, 16, 16),
            "forecast_surface": torch.zeros(1, 16, 16),
            "forecast_rootzone": torch.zeros(1, 16, 16),
        }

    loader = torch.utils.data.DataLoader([batch(5), batch(6)], batch_size=None)

    diagnostics = run_ridge_coeff_adaptation(
        state=state,
        loader=loader,
        device=torch.device("cpu"),
        target_context_prompt_state=prompt_state,
        normalize_increment=True,
        ridge_lambda=0.1,
        ridge_clip_coeff_norm=0.5,
        ridge_trust_region_radius=0.5,
        ridge_max_feature_pixels=10,
        ridge_standardize_features=False,
        surface_weight=3.0,
        rootzone_weight=1.0,
        use_lat_weighted_loss=False,
    )

    assert diagnostics["masked_pixel_count"] == 512
    assert diagnostics["masked_observation_count"] == 1024
    assert diagnostics["feature_pixel_count"] == 10
    assert diagnostics["feature_observation_count"] == 20


def test_cycle_balanced_loss_changes_reduction_but_preserves_mask_semantics():
    from scripts.train.train_hyperda_few_shot_adapt import masked_huber_loss_components

    pred = torch.tensor(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            [
                [[2.0, 0.0], [0.0, 0.0]],
                [[2.0, 0.0], [0.0, 0.0]],
            ],
        ],
        dtype=torch.float32,
    )
    target = torch.zeros_like(pred)
    mask = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    global_loss = masked_huber_loss_components(
        pred,
        target,
        mask,
        reduction="global_pixel",
        delta=1.0,
        surface_weight=1.0,
        rootzone_weight=1.0,
    )
    balanced_loss = masked_huber_loss_components(
        pred,
        target,
        mask,
        reduction="cycle_balanced",
        delta=1.0,
        surface_weight=1.0,
        rootzone_weight=1.0,
    )

    assert global_loss["valid_pixel_count"].item() == pytest.approx(10.0)
    assert balanced_loss["valid_pixel_count"].item() == pytest.approx(10.0)
    assert global_loss["total_loss"].item() != pytest.approx(balanced_loss["total_loss"].item())
    assert global_loss["surface_loss"].item() == pytest.approx(0.3)
    assert balanced_loss["surface_loss"].item() == pytest.approx(0.75)


def test_support_gradient_conflict_diagnostics_use_support_batches_only(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        apply_adapt_scope,
        compute_support_gradient_conflict_diagnostics,
        load_source_checkpoint_for_few_shot,
    )
    from hydroda.training.losses import MaskedHuberLoss

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    apply_adapt_scope(state.model, "prompt_only")
    prompt_state = {
        "schema_version": "target_context_prompt_state_v1",
        "prompt_source": "target_context_monthly_prompt_prototypes",
        "label_usage": "none",
        "monthly_counts": {str(i): 0 for i in range(1, 13)},
        "global_prototype": torch.zeros(8),
        "monthly_prototypes": {str(i): None for i in range(1, 13)},
        "metadata": {},
    }

    def batch(month: int, value: float) -> dict[str, torch.Tensor]:
        return {
            "x": torch.zeros(1, 12, 8, 8),
            "months": torch.tensor([month], dtype=torch.long),
            "increment_surface": torch.ones(1, 8, 8) * value,
            "increment_rootzone": torch.ones(1, 8, 8) * value,
            "loss_mask": torch.ones(1, 8, 8),
            "forecast_surface": torch.zeros(1, 8, 8),
            "forecast_rootzone": torch.zeros(1, 8, 8),
        }

    loader = torch.utils.data.DataLoader([batch(1, 0.1), batch(2, -0.1)], batch_size=None)
    diagnostics = compute_support_gradient_conflict_diagnostics(
        state=state,
        loader=loader,
        device=torch.device("cpu"),
        target_context_prompt_state=prompt_state,
        loss_fn=MaskedHuberLoss(delta=1.0, surface_weight=1.0, rootzone_weight=1.0),
        normalize_increment=True,
        lambda_prior=0.0,
        lambda_latent=0.0,
        lambda_gain=0.0,
        lambda_gain_smooth=0.0,
        lambda_analysis=0.0,
        support_loss_reduction="global_pixel",
    )

    assert diagnostics["support_gradient_diagnostics_label_source"] == "target_support_only"
    assert diagnostics["support_cycle_count"] == 2
    assert len(diagnostics["support_cycle_loss_before"]) == 2
    assert len(diagnostics["support_cycle_gradient_norm"]) == 2
    assert diagnostics["support_gradient_pair_count"] == 1
    assert -1.0 <= diagnostics["support_gradient_cosine_min"] <= 1.0


def test_date_str_records_can_read_preregistered_support_dates_without_support_dataset():
    from scripts.train.train_hyperda_few_shot_adapt import _date_str_records

    class ContextOnlyDataset:
        _split_entry = {
            "target_support_dates": [
                {"date_str": "2016-01-15"},
                {"date_str": "2016-07-15"},
            ]
        }

    assert _date_str_records(None, "target_support_dates") == []
    assert _date_str_records(ContextOnlyDataset(), "target_support_dates") == [
        "2016-01-15",
        "2016-07-15",
    ]


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
                "method": "hyperda_safe_few_shot_k4",
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

    assert predictor.method_name == "hyperda_safe_few_shot_k4"


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
        "support_manifest_hash": "support_manifest_sha",
        "support_nesting_hash": "nesting_sha",
        "support_nesting_status": "K0_no_support",
        "target_eval_dates_hash": "evalhash",
        "target_context_prompt_state": {"schema_version": "target_context_prompt_state_v1"},
        "frozen_source_groups": ["source_backbone", "prompt_encoder", "adapter_basis_bank"],
        "trainable_target_groups": ["target_prompt", "adapter_coefficient_residuals", "monthly_residual_gain"],
        "trainable_parameter_count": 123,
        "trainable_parameter_names": ["target_prompt.latent"],
        "target_parameter_count_by_group": {"target_prompt": 123, "total": 123},
        "adaptation_steps": 0,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "schedule_label": "k0_identity",
        "requested_lr": 3e-4,
        "requested_max_steps": 0,
        "requested_anchor_alpha": 0.0,
        "requested_weight_decay": 1e-4,
        "requested_grad_clip": 1.0,
        "adapt_batch_size": 8,
        "max_steps_requested": 0,
        "requires_grad_parameter_count": 123,
        "requires_grad_param_count": 123,
        "optimizer_parameter_count": 0,
        "optimizer_param_count": 0,
        "optimizer_steps_run": 0,
        "actual_optimizer_steps": 0,
        "support_batch_count": 0,
        "effective_support_passes": 0.0,
        "policy_source": "source_side_episode_calibration",
        "safe_policy_json": "/tmp/safe_policy.json",
        "safe_policy_json_sha256": "policysha",
        "safe_policy": {
            "schema_version": "hyperda_safe_policy_v1",
            "policy_hash": "embedded_policy_hash",
            "policy_source": "source_side_episode_calibration",
        },
        "source_episode_regions": ["US-R2", "US-R3"],
        "rho_policy": "fixed_1.0",
        "adapt_mix_rho": 1.0,
        "standard_support_loss_before_full_support": None,
        "standard_support_loss_after_full_support": None,
        "standard_support_loss_delta_full_support": None,
        "ridge_design_loss_before_sampled_pixels": 0.4,
        "ridge_design_loss_after_sampled_pixels": 0.3,
        "ridge_design_loss_delta_sampled_pixels": -0.1,
        "support_loss_before": 0.4,
        "support_loss_after": 0.3,
        "support_final_loss": None,
        "support_loss_delta": None,
        "target_parameter_l2_drift_pre_anchor": {"total": 0.0},
        "target_parameter_l2_drift_post_anchor": {"total": 0.0},
        "target_parameter_l2_drift": {"total": 0.0},
        "adapt_scope": "none",
        "adapt_solver": "ridge_coeff",
        "freeze_monthly_gain": True,
        "ridge_lambda": 1.25,
        "ridge_clip_coeff_norm": 0.8,
        "ridge_trust_region_radius": 0.7,
        "ridge_max_feature_pixels": 2000,
        "ridge_standardize_features": True,
        "ridge_diagnostics": {
            "status": "solved",
            "coefficient_norm": 0.1,
            "delta_norm": 0.1,
            "raw_delta_norm": 0.2,
            "support_count": 12,
            "masked_pixel_count": 100,
            "masked_observation_count": 200,
            "feature_pixel_count": 80,
            "feature_observation_count": 160,
            "feature_dim": 9,
            "condition_number": 2.0,
            "rank": 9,
        },
        "audit_identity": True,
        "audit_identity_tolerance": 1e-8,
        "source_checkpoint": "/tmp/source.pt",
        "source_checkpoint_sha256": "sourcesha",
        "staged_source_checkpoint_sha256": "sourcesha",
        "source_stage_checkpoint_provenance": "phase4_hyperda_staged",
        "target_support_count": 0,
        "target_labels_loaded_for_adaptation": False,
        "target_labels_used_for_adaptation": False,
        "normalization_source": "source_fit_only_from_source_checkpoint",
        "model_selection_source": "source_val_preregistered",
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "final_eval_only_no_selection",
    }

    write_run_metadata_sidecar(tmp_path, checkpoint_path, config)

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["checkpoint"] == str(checkpoint_path)
    assert metadata["method"] == "hyperda_zero_shot_context"
    assert metadata["adaptation_setting"] == "zero_shot_context"
    assert metadata["target_context_dates_hash"] == "contexthash"
    assert metadata["target_support_dates_hash"] == "supporthash"
    assert metadata["target_support_dates"] == []
    assert metadata["support_manifest_hash"] == "support_manifest_sha"
    assert metadata["support_nesting_hash"] == "nesting_sha"
    assert metadata["support_nesting_status"] == "K0_no_support"
    assert metadata["target_eval_dates_hash"] == "evalhash"
    assert metadata["model_selection_source"] == "source_val_preregistered"
    assert metadata["target_val_usage"] == "unused_in_main_protocol"
    assert metadata["trainable_parameter_count"] == 123
    assert metadata["trainable_parameter_names"] == ["target_prompt.latent"]
    assert metadata["target_parameter_count_by_group"]["target_prompt"] == 123
    assert metadata["adapt_recipe"] == "source_anchor"
    assert metadata["anchor_alpha"] == 0.0
    assert metadata["schedule_label"] == "k0_identity"
    assert metadata["lr"] == pytest.approx(3e-4)
    assert metadata["weight_decay"] == pytest.approx(1e-4)
    assert metadata["grad_clip"] == pytest.approx(1.0)
    assert metadata["requested_lr"] == pytest.approx(3e-4)
    assert metadata["requested_max_steps"] == 0
    assert metadata["requested_anchor_alpha"] == pytest.approx(0.0)
    assert metadata["requested_weight_decay"] == pytest.approx(1e-4)
    assert metadata["requested_grad_clip"] == pytest.approx(1.0)
    assert metadata["adapt_batch_size"] == 8
    assert metadata["max_steps_requested"] == 0
    assert metadata["actual_optimizer_steps"] == 0
    assert metadata["optimizer_steps_run"] == 0
    assert metadata["requires_grad_param_count"] == 123
    assert metadata["optimizer_param_count"] == 0
    assert metadata["support_batch_count"] == 0
    assert metadata["effective_support_passes"] == pytest.approx(0.0)
    assert metadata["source_anchor_hyperparameter_source"] == "source_side_episodic_validation_preregistered"
    assert metadata["policy_source"] == "source_side_episode_calibration"
    assert metadata["safe_policy_json"] == "/tmp/safe_policy.json"
    assert metadata["safe_policy_json_sha256"] == "policysha"
    assert metadata["safe_policy_hash"] == "embedded_policy_hash"
    assert metadata["source_episode_regions"] == ["US-R2", "US-R3"]
    assert metadata["rho_policy"] == "fixed_1.0"
    assert metadata["adapt_mix_rho"] == pytest.approx(1.0)
    assert metadata["support_loss_before"] == pytest.approx(0.4)
    assert metadata["support_loss_after"] == pytest.approx(0.3)
    assert metadata["support_final_loss"] is None
    assert metadata["support_loss_delta"] is None
    assert metadata["standard_support_loss_before_full_support"] is None
    assert metadata["standard_support_loss_after_full_support"] is None
    assert metadata["standard_support_loss_delta_full_support"] is None
    assert metadata["ridge_design_loss_before_sampled_pixels"] == pytest.approx(0.4)
    assert metadata["ridge_design_loss_after_sampled_pixels"] == pytest.approx(0.3)
    assert metadata["ridge_design_loss_delta_sampled_pixels"] == pytest.approx(-0.1)
    assert metadata["target_parameter_l2_drift_pre_anchor"]["total"] == pytest.approx(0.0)
    assert metadata["target_parameter_l2_drift_post_anchor"]["total"] == pytest.approx(0.0)
    assert metadata["target_parameter_l2_drift"]["total"] == 0.0
    assert metadata["target_context_prompt_state"]["schema_version"] == "target_context_prompt_state_v1"
    assert metadata["frozen_source_groups"] == ["source_backbone", "prompt_encoder", "adapter_basis_bank"]
    assert metadata["trainable_target_groups"] == [
        "target_prompt",
        "adapter_coefficient_residuals",
        "monthly_residual_gain",
    ]
    assert metadata["adapt_scope"] == "none"
    assert metadata["adapt_solver"] == "ridge_coeff"
    assert metadata["freeze_monthly_gain"] is True
    assert metadata["ridge_lambda"] == pytest.approx(1.25)
    assert metadata["ridge_max_feature_pixels"] == 2000
    assert metadata["ridge_standardize_features"] is True
    assert metadata["ridge_coefficient_norm"] == pytest.approx(0.1)
    assert metadata["ridge_masked_pixel_count"] == 100
    assert metadata["ridge_feature_pixel_count"] == 80
    assert metadata["ridge_feature_observation_count"] == 160
    assert metadata["audit_identity"] is True
    assert metadata["audit_identity_tolerance"] == pytest.approx(1e-8)
    assert metadata["source_checkpoint"] == "/tmp/source.pt"
    assert metadata["source_checkpoint_sha256"] == "sourcesha"
    assert metadata["staged_source_checkpoint_sha256"] == "sourcesha"
    assert metadata["source_stage_checkpoint_provenance"] == "phase4_hyperda_staged"
    assert metadata["target_support_count"] == 0
    assert metadata["target_labels_loaded_for_adaptation"] is False
    assert metadata["target_labels_used_for_adaptation"] is False
    assert metadata["target_eval_usage"] == "final_eval_only_no_selection"


def test_support_schedule_diagnostics_compute_batches_passes_and_subset():
    from scripts.train.train_hyperda_few_shot_adapt import (
        effective_support_passes,
        support_batch_count,
        support_dates_subset,
    )

    assert support_batch_count(0, 8) == 0
    assert support_batch_count(4, 8) == 1
    assert support_batch_count(12, 8) == 2
    assert effective_support_passes(80, 2) == pytest.approx(40.0)
    assert effective_support_passes(0, 0) == pytest.approx(0.0)

    k4 = ["2018-01-05", "2019-05-13"]
    k12_nested = ["2018-01-05", "2020-02-01", "2019-05-13"]
    k12_non_nested = ["2020-02-01", "2019-05-13"]
    assert support_dates_subset(k4, k12_nested) is True
    assert support_dates_subset(k4, k12_non_nested) is False
