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


def _write_phys_context_source_checkpoint(path: Path) -> None:
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
        hyper_reliability_gate="prompt_scalar",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=2,
        hyper_source_trust_variable_gate=True,
        hyper_phys_context_modulation=True,
        hyper_phys_delta_scale=0.25,
        hyper_phys_gate_init=0.90,
        hyper_operator_droppath_p=0.10,
        phys_context_source="raw_input_side_da_diagnostics",
        zero_shot_prior_form="source_base_residual_reliability_gated",
        source_residual_rho=1.0,
        zero_raw_increment_init=True,
    )
    with torch.no_grad():
        assert model.phys_operator_residual is not None
        model.phys_operator_residual.delta_head.bias.add_(0.125)
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
                "hyper_reliability_gate": "prompt_scalar",
                "hyper_reliability_init": 0.95,
                "hyper_source_trust_routing": True,
                "hyper_source_trust_strength": 0.5,
                "hyper_source_trust_top_m": 2,
                "hyper_source_trust_variable_gate": True,
                "hyper_phys_context_modulation": True,
                "hyper_phys_delta_scale": 0.25,
                "hyper_phys_gate_init": 0.90,
                "hyper_operator_droppath_p": 0.10,
                "phys_context_source": "raw_input_side_da_diagnostics",
                "zero_shot_prior_form": "source_base_residual_reliability_gated",
                "source_residual_rho": 1.0,
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
            "--stage3_kshot_mode",
            "diagnostic_direct_kshot",
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


def test_parse_args_accepts_wrapper_adapt_mix_rho_for_k0_conservative_posterior(monkeypatch, tmp_path):
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
            "--adaptation_steps",
            "0",
            "--anchor_alpha",
            "0.0",
            "--adapt_mix_rho",
            "1.0",
        ],
    )

    args = runner.parse_args()

    assert args.K == 0
    assert args.adaptation_setting == "zero_shot_context"
    assert args.adapt_mix_rho == pytest.approx(1.0)
    assert args.rho_policy == "not_applicable_k0"


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
            "--stage3_kshot_mode",
            "diagnostic_direct_kshot",
            "--seed",
            "0",
        ],
    )

    args = runner.parse_args()
    assert args.K == 12
    assert args.adaptation_setting == "few_shot_k12"
    assert args.adaptation_steps == 100
    assert args.lr == pytest.approx(3e-4)
    assert args.adapt_recipe == "source_anchor"
    assert args.anchor_alpha == pytest.approx(1.0)
    assert args.adaptation_step_policy_source == "legacy_default_steps_for_K"
    assert args.source_anchor_hyperparameter_source == "diagnostic_direct_kshot_fixed_defaults"
    assert args.adapt_scope == "safe_operator"
    assert args.safe_policy_json is None
    assert args.policy_source == "diagnostic_direct_target_support"


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


def test_parse_args_diagnostic_direct_kshot_allows_missing_policy_and_uses_direct_defaults(monkeypatch, tmp_path):
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
            "--stage3_kshot_mode",
            "diagnostic_direct_kshot",
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_direct_kshot"
    assert args.safe_policy_json is None
    assert args.require_safe_policy_json_for_kshot is False
    assert args.adapt_scope == "safe_operator"
    assert args.adaptation_steps > 0
    assert args.anchor_alpha == pytest.approx(1.0)
    assert args.adapt_mix_rho == pytest.approx(1.0)
    assert args.support_gate == "off"
    assert args.support_loss_reduction == "cycle_balanced"
    assert args.policy_source == "diagnostic_direct_target_support"
    assert args.source_anchor_hyperparameter_source == "diagnostic_direct_kshot_fixed_defaults"
    assert args.rho_policy == "diagnostic_direct_fixed_1.0"


def test_parse_args_diagnostic_direct_env_allows_missing_policy(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    monkeypatch.setenv("STAGE3_KSHOT_MODE", "diagnostic_direct_kshot")
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
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_direct_kshot"
    assert args.require_safe_policy_json_for_kshot is False
    assert args.anchor_alpha == pytest.approx(1.0)
    assert args.adapt_mix_rho == pytest.approx(1.0)
    assert args.policy_source == "diagnostic_direct_target_support"


def test_parse_args_diagnostic_direct_kshot_v2_uses_strong_full_safe_operator_defaults(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    monkeypatch.setenv("DIAGNOSTIC_KSHOT_STRENGTH", "strong")
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
            "--stage3_kshot_mode",
            "diagnostic_direct_kshot_v2",
            "--stage3_posterior_policy",
            "conservative_coeff_posterior",
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_direct_kshot_v2"
    assert args.safe_policy_json is None
    assert args.require_safe_policy_json_for_kshot is False
    assert args.adapt_scope == "safe_operator"
    assert args.freeze_monthly_gain is False
    assert args.adaptation_steps == 200
    assert args.lr == pytest.approx(1e-3)
    assert args.anchor_alpha == pytest.approx(1.0)
    assert args.adapt_mix_rho == pytest.approx(1.0)
    assert args.support_gate == "off"
    assert args.support_loss_reduction == "cycle_balanced"
    assert args.policy_source == "diagnostic_direct_target_support"
    assert args.source_anchor_hyperparameter_source == "diagnostic_direct_kshot_v2_strong_fixed_defaults"


def test_parse_args_diagnostic_conservative_kshot_v3_uses_coeff_trust_region_defaults(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    monkeypatch.setenv("DIAGNOSTIC_KSHOT_STRENGTH", "strong")
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
            "--stage3_kshot_mode",
            "diagnostic_conservative_kshot_v3",
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_conservative_kshot_v3"
    assert args.safe_policy_json is None
    assert args.require_safe_policy_json_for_kshot is False
    assert args.stage3_posterior_policy == "conservative_coeff_posterior"
    assert args.adapt_scope == "coeff_only"
    assert args.freeze_monthly_gain is True
    assert args.adaptation_steps == 80
    assert args.lr == pytest.approx(3e-4)
    assert args.anchor_alpha == pytest.approx(0.50)
    assert args.adapt_mix_rho == pytest.approx(0.60)
    assert args.rho_policy == "diagnostic_conservative_fixed_0.6"
    assert args.support_gate == "auto"
    assert args.support_gate_min_delta == pytest.approx(1e-8)
    assert args.support_gate_rootzone_tolerance == pytest.approx(1e-8)
    assert args.support_loss_reduction == "cycle_balanced"
    assert args.trust_region_mode == "groupwise"
    assert args.trust_prompt_radius == pytest.approx(0.0)
    assert args.trust_gain_radius == pytest.approx(0.0)
    assert args.trust_coeff_radius == pytest.approx(0.30)
    assert args.policy_source == "diagnostic_direct_target_support"
    assert (
        args.source_anchor_hyperparameter_source
        == "diagnostic_conservative_kshot_v3_strong_fixed_defaults"
    )


def test_parse_args_diagnostic_support_gain_v1_is_no_update_profile(monkeypatch, tmp_path):
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
            "--stage3_kshot_mode",
            "diagnostic_support_gain_v1",
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_support_gain_v1"
    assert args.adapt_scope == "none"
    assert args.adaptation_steps == 0
    assert args.anchor_alpha == pytest.approx(0.0)
    assert args.stage3_posterior_policy == "source_calibrated_mix"
    assert args.freeze_monthly_gain is True
    assert args.policy_source == "diagnostic_direct_target_support"


def test_parse_args_diagnostic_support_gain_v2_is_checkpoint_calibration_profile(monkeypatch, tmp_path):
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
            "--stage3_kshot_mode",
            "diagnostic_support_gain_v2",
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_support_gain_v2"
    assert args.adapt_scope == "none"
    assert args.adaptation_steps == 0
    assert args.anchor_alpha == pytest.approx(0.0)
    assert args.stage3_posterior_policy == "source_calibrated_mix"
    assert args.freeze_monthly_gain is True
    assert args.support_gate == "off"
    assert args.policy_source == "diagnostic_direct_target_support"
    assert (
        args.source_anchor_hyperparameter_source
        == "diagnostic_support_gain_v2_checkpoint_fixed_alpha_grid"
    )


def test_parse_args_diagnostic_support_gain_v3_stable_is_checkpoint_calibration_profile(monkeypatch, tmp_path):
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
            "--stage3_kshot_mode",
            "diagnostic_support_gain_v3_stable",
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_support_gain_v3_stable"
    assert args.adapt_scope == "none"
    assert args.adaptation_steps == 0
    assert args.anchor_alpha == pytest.approx(0.0)
    assert args.stage3_posterior_policy == "source_calibrated_mix"
    assert args.freeze_monthly_gain is True
    assert args.support_gate == "off"
    assert args.policy_source == "diagnostic_direct_target_support"
    assert (
        args.source_anchor_hyperparameter_source
        == "diagnostic_support_gain_v3_stable_checkpoint_stability_alpha_grid"
    )


def test_parse_args_diagnostic_support_gain_v4_nested_stable_is_no_update_checkpoint_profile(monkeypatch, tmp_path):
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
            "--stage3_kshot_mode",
            "diagnostic_support_gain_v4_nested_stable",
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_support_gain_v4_nested_stable"
    assert args.adapt_scope == "none"
    assert args.adaptation_steps == 0
    assert args.anchor_alpha == pytest.approx(0.0)
    assert args.adapt_mix_rho == pytest.approx(1.0)
    assert args.stage3_posterior_policy == "source_calibrated_mix"
    assert args.freeze_monthly_gain is True
    assert args.support_gate == "off"
    assert args.policy_source == "diagnostic_direct_target_support"
    assert (
        args.source_anchor_hyperparameter_source
        == "diagnostic_support_gain_v4_nested_stable_checkpoint_stability_alpha_grid"
    )


def test_parse_args_diagnostic_support_gain_v12_nested_cv_is_no_update_checkpoint_profile(monkeypatch, tmp_path):
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
            "--stage3_kshot_mode",
            "diagnostic_support_gain_v12_nested_cv",
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_support_gain_v12_nested_cv"
    assert args.adapt_scope == "none"
    assert args.adaptation_steps == 0
    assert args.anchor_alpha == pytest.approx(0.0)
    assert args.adapt_mix_rho == pytest.approx(1.0)
    assert args.stage3_posterior_policy == "source_calibrated_mix"
    assert args.freeze_monthly_gain is True
    assert args.support_gate == "off"
    assert args.policy_source == "diagnostic_direct_target_support"
    assert args.target_eval_usage == "final_eval_only_no_selection"
    assert (
        args.source_anchor_hyperparameter_source
        == "diagnostic_support_gain_v12_nested_cv_checkpoint_no_harm_alpha_grid"
    )


def test_parse_args_diagnostic_support_gain_v13_k4_uses_v12_reference_profile(monkeypatch, tmp_path):
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
            "--stage3_kshot_mode",
            "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool"
    assert args.adapt_scope == "none"
    assert args.adaptation_steps == 0
    assert args.anchor_alpha == pytest.approx(0.0)
    assert args.adapt_mix_rho == pytest.approx(1.0)
    assert args.stage3_posterior_policy == "source_calibrated_mix"
    assert args.freeze_monthly_gain is True
    assert args.support_gate == "off"
    assert args.policy_source == "diagnostic_direct_target_support"
    assert args.resolved_mode_defaults["support_selection_objective"] == "v12_global_residual_gain_reference_for_v13"
    assert (
        args.source_anchor_hyperparameter_source
        == "diagnostic_support_gain_v12_nested_cv_checkpoint_no_harm_alpha_grid"
    )


def test_parse_args_diagnostic_support_gain_v13_k12_uses_aggressive_pool_profile(monkeypatch, tmp_path):
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
            "--stage3_kshot_mode",
            "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
            "--k4_reference_checkpoint",
            str(source_checkpoint),
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool"
    assert args.adapt_scope == "none"
    assert args.adaptation_steps == 0
    assert args.anchor_alpha == pytest.approx(0.0)
    assert args.adapt_mix_rho == pytest.approx(1.0)
    assert args.stage3_posterior_policy == "source_calibrated_mix"
    assert args.freeze_monthly_gain is True
    assert args.support_gate == "off"
    assert args.policy_source == "diagnostic_direct_target_support"
    assert args.k4_reference_checkpoint == str(source_checkpoint)
    assert (
        args.source_anchor_hyperparameter_source
        == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool_fixed_exploratory_defaults"
    )
    assert (
        args.resolved_mode_defaults["support_selection_objective"]
        == "k12_aggressive_nested_cv_calibration_pool_support_only"
    )
    assert args.resolved_mode_defaults["support_gate_rootzone_tolerance"] == pytest.approx(5e-6)
    assert args.resolved_mode_defaults["support_gate_cycle_improvement_min_fraction"] == pytest.approx(0.5)
    assert len(args.resolved_mode_defaults["support_candidate_pool"]) == 7


def test_parse_args_diagnostic_finetune_support_gain_v14_uses_real_finetune_profile(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    _write_source_checkpoint(source_checkpoint)
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
            "--stage3_kshot_mode",
            "diagnostic_finetune_support_gain_v14_nested",
            "--k4_reference_checkpoint",
            str(source_checkpoint),
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_finetune_support_gain_v14_nested"
    assert args.adapt_scope == "coeff_only"
    assert args.adapt_solver == "adamw"
    assert args.adaptation_steps == 80
    assert args.lr == pytest.approx(3e-4)
    assert args.anchor_alpha == pytest.approx(0.50)
    assert args.adapt_mix_rho == pytest.approx(1.0)
    assert args.stage3_posterior_policy == "conservative_coeff_posterior"
    assert args.freeze_monthly_gain is True
    assert args.support_gate == "auto"
    assert args.support_loss_reduction == "cycle_balanced"
    assert args.trust_region_mode == "groupwise"
    assert args.trust_coeff_radius == pytest.approx(0.30)
    assert args.policy_source == "diagnostic_direct_target_support"
    assert args.target_eval_usage == "final_eval_only_no_selection"
    assert (
        args.resolved_mode_defaults["support_selection_objective"]
        == "target_support_parameter_finetune_then_nested_cv_support_gain"
    )
    assert (
        args.source_anchor_hyperparameter_source
        == "diagnostic_finetune_support_gain_v14_nested_support_only_coeff_finetune_plus_gain"
    )


def test_parse_args_diagnostic_support_affine_v1_nested_is_support_only_affine_profile(monkeypatch, tmp_path):
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
            "--stage3_kshot_mode",
            "diagnostic_support_affine_v1_nested",
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_support_affine_v1_nested"
    assert args.adapt_scope == "none"
    assert args.adaptation_steps == 0
    assert args.anchor_alpha == pytest.approx(0.0)
    assert args.stage3_posterior_policy == "source_calibrated_mix"
    assert args.freeze_monthly_gain is True
    assert args.support_gate == "off"
    assert args.policy_source == "diagnostic_direct_target_support"
    assert (
        args.source_anchor_hyperparameter_source
        == "diagnostic_support_affine_v1_nested_support_only_ridge_affine"
    )


def test_parse_args_diagnostic_safe_operator_v5_nested_defaults_to_coeff_only_profile(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    _write_source_checkpoint(source_checkpoint)
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
            "--stage3_kshot_mode",
            "diagnostic_safe_operator_v5_nested",
            "--k4_reference_checkpoint",
            str(source_checkpoint),
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_safe_operator_v5_nested"
    assert args.adapt_scope == "coeff_only"
    assert args.adaptation_steps == 80
    assert args.anchor_alpha == pytest.approx(0.50)
    assert args.lr == pytest.approx(3e-4)
    assert args.stage3_posterior_policy == "conservative_coeff_posterior"
    assert args.support_gate == "auto"
    assert args.freeze_monthly_gain is True
    assert args.trust_region_mode == "groupwise"
    assert args.trust_prompt_radius == pytest.approx(0.0)
    assert args.trust_gain_radius == pytest.approx(0.0)
    assert args.trust_coeff_radius == pytest.approx(0.30)
    assert args.k4_reference_checkpoint == str(source_checkpoint)
    assert (
        args.source_anchor_hyperparameter_source
        == "diagnostic_safe_operator_v5_nested_support_only_coeff_residual_policy"
    )


def test_diagnostic_v5_v6_k12_defaults_not_below_k4():
    from scripts.train.train_hyperda_few_shot_adapt import (
        diagnostic_linearized_coeff_ridge_v6_defaults,
        diagnostic_linearized_coeff_ridge_v7_defaults,
        diagnostic_linearized_coeff_ridge_v8_hybrid_defaults,
        diagnostic_linearized_coeff_ridge_v9_guarded_defaults,
        diagnostic_safe_operator_v5_defaults,
    )

    v5_k4 = diagnostic_safe_operator_v5_defaults(4, "strong")
    v5_k12 = diagnostic_safe_operator_v5_defaults(12, "strong")
    assert v5_k4["adaptation_steps"] == 40
    assert v5_k12["adaptation_steps"] >= v5_k4["adaptation_steps"]

    v6_k4 = diagnostic_linearized_coeff_ridge_v6_defaults(4, "strong")
    v6_k12 = diagnostic_linearized_coeff_ridge_v6_defaults(12, "strong")
    assert v6_k4["adaptation_steps"] == 0
    assert v6_k12["adaptation_steps"] >= v6_k4["adaptation_steps"]
    assert v6_k12["ridge_trust_region_radius"] > v6_k4["ridge_trust_region_radius"]
    assert v6_k12["ridge_lambda"] < v6_k4["ridge_lambda"]

    v7_k4 = diagnostic_linearized_coeff_ridge_v7_defaults(4, "strong")
    v7_k12 = diagnostic_linearized_coeff_ridge_v7_defaults(12, "strong")
    assert v7_k4["ridge_weighting"] == "cycle_variable_balanced_huber"
    assert v7_k12["adaptation_steps"] == 0
    assert v7_k12["ridge_trust_region_radius"] > v7_k4["ridge_trust_region_radius"]
    assert v7_k12["ridge_lambda"] < v7_k4["ridge_lambda"]

    v8_k4 = diagnostic_linearized_coeff_ridge_v8_hybrid_defaults(4, "strong")
    v8_k12 = diagnostic_linearized_coeff_ridge_v8_hybrid_defaults(12, "strong")
    assert v8_k4["ridge_weighting"] == "cycle_variable_balanced_huber"
    assert v8_k12["ridge_weighting"] == "global_pixel_l2"
    assert v8_k12["adaptation_steps"] == 0
    assert v8_k12["ridge_trust_region_radius"] > v8_k4["ridge_trust_region_radius"]
    assert v8_k12["ridge_lambda"] < v8_k4["ridge_lambda"]

    v9_k4 = diagnostic_linearized_coeff_ridge_v9_guarded_defaults(4, "strong")
    v9_k12 = diagnostic_linearized_coeff_ridge_v9_guarded_defaults(12, "strong")
    assert v9_k4["ridge_weighting"] == "cycle_variable_balanced_huber"
    assert v9_k4["support_gate_min_delta"] == pytest.approx(1e-8)
    assert v9_k12["ridge_weighting"] == "global_pixel_l2"
    assert v9_k12["support_gate_min_delta"] == pytest.approx(3e-3)
    assert v9_k12["adaptation_steps"] == 0


def test_parse_args_linearized_coeff_ridge_v6_nested_defaults_to_closed_form_profile(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    _write_source_checkpoint(source_checkpoint)
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
            "--stage3_kshot_mode",
            "diagnostic_linearized_coeff_ridge_v6_nested",
            "--k4_reference_checkpoint",
            str(source_checkpoint),
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v6_nested"
    assert args.adapt_solver == "ridge_coeff"
    assert args.adapt_scope == "coeff_only"
    assert args.adaptation_steps == 0
    assert args.anchor_alpha == pytest.approx(0.60)
    assert args.lr == pytest.approx(0.0)
    assert args.stage3_posterior_policy == "conservative_coeff_posterior"
    assert args.support_gate == "auto"
    assert args.freeze_monthly_gain is True
    assert args.support_loss_reduction == "cycle_balanced"
    assert args.ridge_lambda == pytest.approx(1.0)
    assert args.ridge_clip_coeff_norm == pytest.approx(0.40)
    assert args.ridge_trust_region_radius == pytest.approx(0.28)
    assert args.ridge_standardize_features is True
    assert args.adapt_mix_rho == pytest.approx(0.65)
    assert args.adaptation_step_policy_source == (
        "diagnostic_linearized_coeff_ridge_v6_closed_form_no_adam_steps"
    )
    assert args.resolved_mode_defaults["adapt_solver"] == "ridge_coeff"
    assert (
        args.source_anchor_hyperparameter_source
        == "diagnostic_linearized_coeff_ridge_v6_nested_source_side_policy_defaults"
    )


def test_parse_args_linearized_coeff_ridge_v7_defaults_to_balanced_support_profile(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    _write_source_checkpoint(source_checkpoint)
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
            "--stage3_kshot_mode",
            "diagnostic_linearized_coeff_ridge_v7_balanced_nested",
            "--k4_reference_checkpoint",
            str(source_checkpoint),
        ],
    )

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v7_balanced_nested"
    assert args.adapt_solver == "ridge_coeff"
    assert args.adapt_scope == "coeff_only"
    assert args.adaptation_steps == 0
    assert args.ridge_lambda == pytest.approx(1.0)
    assert args.ridge_trust_region_radius == pytest.approx(0.28)
    assert args.ridge_weighting == "cycle_variable_balanced_huber"
    assert args.adapt_mix_rho == pytest.approx(0.65)
    assert args.adaptation_step_policy_source == (
        "diagnostic_linearized_coeff_ridge_v7_balanced_closed_form_no_adam_steps"
    )
    assert (
        args.source_anchor_hyperparameter_source
        == "diagnostic_linearized_coeff_ridge_v7_balanced_nested_source_side_policy_defaults"
    )


@pytest.mark.parametrize(
    ("K", "expected_weighting", "expected_ridge_lambda", "expected_trust_radius", "expected_mix_rho"),
    [
        (4, "cycle_variable_balanced_huber", 2.0, 0.18, 0.50),
        (12, "global_pixel_l2", 1.0, 0.28, 0.65),
    ],
)
def test_parse_args_linearized_coeff_ridge_v8_hybrid_uses_k_specific_weighting(
    monkeypatch,
    tmp_path,
    K,
    expected_weighting,
    expected_ridge_lambda,
    expected_trust_radius,
    expected_mix_rho,
):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    _write_source_checkpoint(source_checkpoint)
    argv = [
        "train_hyperda_few_shot_adapt.py",
        "--source_checkpoint",
        str(source_checkpoint),
        "--target_region",
        "US-R1",
        "--K",
        str(K),
        "--stage3_kshot_mode",
        "diagnostic_linearized_coeff_ridge_v8_hybrid_nested",
    ]
    if K == 12:
        argv.extend(["--k4_reference_checkpoint", str(source_checkpoint)])
    monkeypatch.setattr("sys.argv", argv)

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested"
    assert args.adapt_solver == "ridge_coeff"
    assert args.adapt_scope == "coeff_only"
    assert args.adaptation_steps == 0
    assert args.ridge_lambda == pytest.approx(expected_ridge_lambda)
    assert args.ridge_trust_region_radius == pytest.approx(expected_trust_radius)
    assert args.ridge_weighting == expected_weighting
    assert args.adapt_mix_rho == pytest.approx(expected_mix_rho)
    assert args.adaptation_step_policy_source == (
        "diagnostic_linearized_coeff_ridge_v8_hybrid_closed_form_no_adam_steps"
    )
    assert (
        args.source_anchor_hyperparameter_source
        == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested_source_side_policy_defaults"
    )


@pytest.mark.parametrize(
    "K,expected_weighting,expected_support_gate_min_delta",
    [
        (4, "cycle_variable_balanced_huber", 1e-8),
        (12, "global_pixel_l2", 3e-3),
    ],
)
def test_parse_args_linearized_coeff_ridge_v9_guarded_uses_k12_margin(
    monkeypatch,
    tmp_path,
    K,
    expected_weighting,
    expected_support_gate_min_delta,
):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    _write_source_checkpoint(source_checkpoint)
    argv = [
        "train_hyperda_few_shot_adapt.py",
        "--source_checkpoint",
        str(source_checkpoint),
        "--target_region",
        "US-R1",
        "--K",
        str(K),
        "--stage3_kshot_mode",
        "diagnostic_linearized_coeff_ridge_v9_guarded_nested",
    ]
    if K == 12:
        argv.extend(["--k4_reference_checkpoint", str(source_checkpoint)])
    monkeypatch.setattr("sys.argv", argv)

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v9_guarded_nested"
    assert args.adapt_solver == "ridge_coeff"
    assert args.adapt_scope == "coeff_only"
    assert args.adaptation_steps == 0
    assert args.ridge_weighting == expected_weighting
    assert args.support_gate_min_delta == pytest.approx(expected_support_gate_min_delta)
    assert args.adaptation_step_policy_source == (
        "diagnostic_linearized_coeff_ridge_v9_guarded_closed_form_no_adam_steps"
    )
    assert (
        args.source_anchor_hyperparameter_source
        == "diagnostic_linearized_coeff_ridge_v9_guarded_nested_source_side_policy_defaults"
    )


def test_v10_support_pool_ids_and_parameters_are_fixed():
    from scripts.train.train_hyperda_few_shot_adapt import (
        diagnostic_linearized_coeff_ridge_v10_defaults,
        diagnostic_linearized_coeff_ridge_v10_support_pool,
    )

    k4 = diagnostic_linearized_coeff_ridge_v10_support_pool(4)
    assert k4 == [
        {
            "candidate_id": "k4_conservative_balanced",
            "ridge_weighting": "cycle_variable_balanced_huber",
            "lambda": 4.0,
            "radius": 0.10,
            "alpha": 0.30,
            "rho": 0.35,
        },
        {
            "candidate_id": "k4_current_balanced",
            "ridge_weighting": "cycle_variable_balanced_huber",
            "lambda": 2.0,
            "radius": 0.18,
            "alpha": 0.40,
            "rho": 0.50,
        },
        {
            "candidate_id": "k4_stronger_balanced",
            "ridge_weighting": "cycle_variable_balanced_huber",
            "lambda": 1.0,
            "radius": 0.22,
            "alpha": 0.45,
            "rho": 0.55,
        },
    ]
    k12 = diagnostic_linearized_coeff_ridge_v10_support_pool(12)
    assert [candidate["candidate_id"] for candidate in k12] == [
        "k12_conservative_global",
        "k12_current_global",
        "k12_balanced_huber",
        "k12_medium_global",
    ]
    assert k12[0]["lambda"] == pytest.approx(2.0)
    assert k12[1]["rho"] == pytest.approx(0.65)
    assert k12[2]["ridge_weighting"] == "cycle_variable_balanced_huber"
    assert k12[3]["radius"] == pytest.approx(0.22)

    defaults = diagnostic_linearized_coeff_ridge_v10_defaults(12)
    assert defaults["adapt_solver"] == "ridge_coeff"
    assert defaults["adapt_scope"] == "coeff_only"
    assert defaults["support_gate_min_delta"] == pytest.approx(1e-3)
    assert defaults["support_gate_cycle_improvement_min_fraction"] == pytest.approx(8.0 / 12.0)
    assert defaults["support_selection_objective"] == "exact_mixed_target_support_only"
    assert defaults["k12_reference_policy"] == "k4_safe_nested_reference"
    assert defaults["target_eval_usage"] == "final_eval_only_no_selection"


@pytest.mark.parametrize(
    "K,expected_min_delta,expected_fraction",
    [(4, 1e-4, 0.75), (12, 1e-3, 8.0 / 12.0)],
)
def test_parse_args_linearized_coeff_ridge_v10_support_pool(monkeypatch, tmp_path, K, expected_min_delta, expected_fraction):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    _write_source_checkpoint(source_checkpoint)
    argv = [
        "train_hyperda_few_shot_adapt.py",
        "--source_checkpoint",
        str(source_checkpoint),
        "--target_region",
        "US-R1",
        "--K",
        str(K),
        "--stage3_kshot_mode",
        "diagnostic_linearized_coeff_ridge_v10_support_pool_nested",
    ]
    if K == 12:
        argv.extend(["--k4_reference_checkpoint", str(source_checkpoint)])
    monkeypatch.setattr("sys.argv", argv)

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v10_support_pool_nested"
    assert args.adapt_solver == "ridge_coeff"
    assert args.adapt_scope == "coeff_only"
    assert args.adaptation_steps == 0
    assert args.lr == pytest.approx(0.0)
    assert args.support_gate == "auto"
    assert args.support_gate_min_delta == pytest.approx(expected_min_delta)
    assert args.freeze_monthly_gain is True
    assert args.stage3_posterior_policy == "conservative_coeff_posterior"
    assert args.resolved_mode_defaults["support_gate_cycle_improvement_min_fraction"] == pytest.approx(expected_fraction)
    assert args.resolved_mode_defaults["support_selection_objective"] == "exact_mixed_target_support_only"
    assert args.target_eval_usage == "final_eval_only_no_selection"


def test_v10_mixed_support_scoring_rho_endpoints(monkeypatch, tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        apply_adapt_scope,
        exact_mixed_support_objective_from_loader,
        load_source_checkpoint_for_few_shot,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    apply_adapt_scope(state.model, "coeff_only")

    def fake_forward_prediction(*, state, batch, device, target_context_prompt_state):
        del target_context_prompt_state
        value = state.model.target_adapter_coefficient_residual_b.logit_delta.reshape(-1)[0].detach()
        height, width = batch["increment_surface"].shape[-2:]
        return torch.ones((1, 2, height, width), dtype=torch.float32, device=device) * value.to(device)

    monkeypatch.setattr(
        "scripts.train.train_hyperda_few_shot_adapt._ridge_forward_prediction",
        fake_forward_prediction,
    )
    anchor = {name: tensor.detach().clone() for name, tensor in state.model.state_dict().items() if name.startswith("target_") or name.startswith("residual_gain.")}
    candidate = {name: tensor.clone() for name, tensor in anchor.items()}
    candidate["target_adapter_coefficient_residual_b.logit_delta"] = candidate[
        "target_adapter_coefficient_residual_b.logit_delta"
    ] + 0.2
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
        "x": torch.zeros(1, 12, 8, 8),
        "months": torch.tensor([5], dtype=torch.long),
        "increment_surface": torch.ones(1, 8, 8) * 0.1,
        "increment_rootzone": torch.zeros(1, 8, 8),
        "loss_mask": torch.ones(1, 8, 8),
        "forecast_surface": torch.zeros(1, 8, 8),
        "forecast_rootzone": torch.zeros(1, 8, 8),
    }
    loader = torch.utils.data.DataLoader([batch], batch_size=None)

    anchor_score = exact_mixed_support_objective_from_loader(
        state=state,
        loader=loader,
        device=torch.device("cpu"),
        target_context_prompt_state=prompt_state,
        anchor_state=anchor,
        candidate_state=anchor,
        rho=1.0,
        normalize_increment=True,
        surface_weight=3.0,
        rootzone_weight=1.0,
        use_lat_weighted_loss=False,
    )
    rho0 = exact_mixed_support_objective_from_loader(
        state=state,
        loader=loader,
        device=torch.device("cpu"),
        target_context_prompt_state=prompt_state,
        anchor_state=anchor,
        candidate_state=candidate,
        rho=0.0,
        normalize_increment=True,
        surface_weight=3.0,
        rootzone_weight=1.0,
        use_lat_weighted_loss=False,
    )
    candidate_score = exact_mixed_support_objective_from_loader(
        state=state,
        loader=loader,
        device=torch.device("cpu"),
        target_context_prompt_state=prompt_state,
        anchor_state=anchor,
        candidate_state=candidate,
        rho=1.0,
        normalize_increment=True,
        surface_weight=3.0,
        rootzone_weight=1.0,
        use_lat_weighted_loss=False,
    )

    assert rho0["standard_support_objective_full_support"] == pytest.approx(
        anchor_score["standard_support_objective_full_support"]
    )
    assert candidate_score["standard_support_objective_full_support"] != pytest.approx(
        anchor_score["standard_support_objective_full_support"]
    )
    assert candidate_score["support_cycle_count"] == 1


def test_v10_k4_gate_rejects_weak_candidate_to_k0():
    from scripts.train.train_hyperda_few_shot_adapt import decide_v10_k4_support_pool_gate

    k0 = {
        "standard_support_objective_full_support": 1.0,
        "standard_support_loss_full_support": 1.0,
        "standard_support_surface_loss_full_support": 0.5,
        "standard_support_rootzone_loss_full_support": 0.5,
    }
    weak = {
        "standard_support_objective_full_support": 0.99995,
        "standard_support_loss_full_support": 0.99995,
        "standard_support_surface_loss_full_support": 0.4999,
        "standard_support_rootzone_loss_full_support": 0.5,
    }

    rejected = decide_v10_k4_support_pool_gate(
        candidate=weak,
        k0_anchor=k0,
        min_delta=1e-4,
        min_cycle_improvement_fraction=0.75,
        cycle_improvement_fraction=1.0,
        selected_candidate_id="k4_current_balanced",
        support_candidate_pool=[{"candidate_id": "k4_current_balanced"}],
    )

    assert rejected["stage3_posterior_decision"] == "rejected_to_k0_anchor"
    assert rejected["support_gate_status"] == "support_only_rejected_to_k0_anchor"
    assert "objective_not_improved" in rejected["support_gate_reject_reason"]
    assert rejected["selected_support_candidate_id"] == "k4_current_balanced"
    assert rejected["support_selection_objective"] == "exact_mixed_target_support_only"
    assert rejected["target_eval_usage"] == "final_eval_only_no_selection"


def test_v10_k12_gate_requires_margin_cycles_rootzone_and_nested_subset():
    from scripts.train.train_hyperda_few_shot_adapt import decide_v10_k12_support_pool_gate

    ref = {
        "standard_support_objective_full_support": 1.0,
        "standard_support_loss_full_support": 1.0,
        "standard_support_surface_loss_full_support": 0.5,
        "standard_support_rootzone_loss_full_support": 0.5,
    }
    strong = {
        "standard_support_objective_full_support": 0.9985,
        "standard_support_loss_full_support": 0.9985,
        "standard_support_surface_loss_full_support": 0.499,
        "standard_support_rootzone_loss_full_support": 0.4995,
    }
    nested_ref = {**ref, "standard_support_objective_full_support": 0.8}
    nested_candidate = {**strong, "standard_support_objective_full_support": 0.8}

    accepted = decide_v10_k12_support_pool_gate(
        candidate=strong,
        k4_reference=ref,
        candidate_nested_k4=nested_candidate,
        k4_reference_nested=nested_ref,
        min_delta=1e-3,
        min_cycle_improvement_fraction=8.0 / 12.0,
        cycle_improvement_fraction=8.0 / 12.0,
        selected_candidate_id="k12_current_global",
        support_candidate_pool=[{"candidate_id": "k12_current_global"}],
        k4_reference_adapt_mix_rho=0.5,
        support_nesting_policy="run_local_k12_nested_k4_plus_8_original_k12_nonduplicate",
        nested_support_dates_hash="nestedhash",
    )
    assert accepted["stage3_posterior_decision"] == "accepted"
    assert accepted["support_gate_status"] == "support_only_v10_k12_beats_k4_reference"
    assert accepted["k12_reference_policy"] == "k4_safe_nested_reference"

    weak_margin = decide_v10_k12_support_pool_gate(
        candidate={**strong, "standard_support_objective_full_support": 0.9995},
        k4_reference=ref,
        candidate_nested_k4=nested_candidate,
        k4_reference_nested=nested_ref,
        min_delta=1e-3,
        min_cycle_improvement_fraction=8.0 / 12.0,
        cycle_improvement_fraction=1.0,
        selected_candidate_id="k12_current_global",
        support_candidate_pool=[{"candidate_id": "k12_current_global"}],
    )
    assert weak_margin["stage3_posterior_decision"] == "fallback_to_k4_reference"
    assert "k12_not_better_than_k4_reference_on_nested_support" in weak_margin["support_gate_reject_reason"]

    weak_cycles = decide_v10_k12_support_pool_gate(
        candidate=strong,
        k4_reference=ref,
        candidate_nested_k4=nested_candidate,
        k4_reference_nested=nested_ref,
        min_delta=1e-3,
        min_cycle_improvement_fraction=8.0 / 12.0,
        cycle_improvement_fraction=7.0 / 12.0,
        selected_candidate_id="k12_current_global",
        support_candidate_pool=[{"candidate_id": "k12_current_global"}],
    )
    assert "insufficient_support_cycle_improvement_fraction" in weak_cycles["support_gate_reject_reason"]

    weak_nested = decide_v10_k12_support_pool_gate(
        candidate=strong,
        k4_reference=ref,
        candidate_nested_k4={**nested_candidate, "standard_support_objective_full_support": 0.801},
        k4_reference_nested=nested_ref,
        min_delta=1e-3,
        min_cycle_improvement_fraction=8.0 / 12.0,
        cycle_improvement_fraction=1.0,
        selected_candidate_id="k12_current_global",
        support_candidate_pool=[{"candidate_id": "k12_current_global"}],
    )
    assert "nested_k4_subset_worse_than_k4_reference" in weak_nested["support_gate_reject_reason"]


def test_v11_support_pool_ids_and_parameters_are_fixed():
    from scripts.train.train_hyperda_few_shot_adapt import (
        diagnostic_linearized_coeff_ridge_v11_defaults,
        diagnostic_linearized_coeff_ridge_v11_support_pool,
    )

    k4 = diagnostic_linearized_coeff_ridge_v11_support_pool(4)
    assert k4 == [
        {
            "candidate_id": "k4_eb_shrink_balanced",
            "ridge_weighting": "cycle_variable_balanced_huber",
            "lambda": 8.0,
            "radius": 0.06,
            "alpha": 0.20,
            "rho": 0.20,
        },
        {
            "candidate_id": "k4_l2sp_balanced",
            "ridge_weighting": "cycle_variable_balanced_huber",
            "lambda": 4.0,
            "radius": 0.08,
            "alpha": 0.25,
            "rho": 0.25,
        },
        {
            "candidate_id": "k4_wiseft_balanced",
            "ridge_weighting": "cycle_variable_balanced_huber",
            "lambda": 2.0,
            "radius": 0.10,
            "alpha": 0.30,
            "rho": 0.30,
        },
    ]
    k12 = diagnostic_linearized_coeff_ridge_v11_support_pool(12)
    assert [candidate["candidate_id"] for candidate in k12] == [
        "k12_eb_shrink_global",
        "k12_l2sp_balanced",
        "k12_wiseft_global",
    ]
    assert k12[0]["lambda"] == pytest.approx(6.0)
    assert k12[1]["ridge_weighting"] == "cycle_variable_balanced_huber"
    assert k12[2]["rho"] == pytest.approx(0.40)

    defaults = diagnostic_linearized_coeff_ridge_v11_defaults(12)
    assert defaults["adapt_solver"] == "ridge_coeff"
    assert defaults["adapt_scope"] == "coeff_only"
    assert defaults["support_gate_min_delta"] == pytest.approx(2e-6)
    assert defaults["support_gate_cycle_improvement_min_fraction"] == pytest.approx(2.0 / 3.0)
    assert defaults["support_selection_objective"] == "loocv_mixed_raw_increment_wrmse_target_support_only"
    assert defaults["k12_reference_policy"] == "k4_safe_nested_reference"
    assert defaults["target_eval_usage"] == "final_eval_only_no_selection"


@pytest.mark.parametrize(
    "K,expected_min_delta,expected_fraction",
    [(4, 5e-6, 1.0), (12, 2e-6, 2.0 / 3.0)],
)
def test_parse_args_linearized_coeff_ridge_v11_loocv_support_pool(
    monkeypatch,
    tmp_path,
    K,
    expected_min_delta,
    expected_fraction,
):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    _write_source_checkpoint(source_checkpoint)
    argv = [
        "train_hyperda_few_shot_adapt.py",
        "--source_checkpoint",
        str(source_checkpoint),
        "--target_region",
        "US-R1",
        "--K",
        str(K),
        "--stage3_kshot_mode",
        "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested",
    ]
    if K == 12:
        argv.extend(["--k4_reference_checkpoint", str(source_checkpoint)])
    monkeypatch.setattr("sys.argv", argv)

    args = runner.parse_args()

    assert args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested"
    assert args.adapt_solver == "ridge_coeff"
    assert args.adapt_scope == "coeff_only"
    assert args.adaptation_steps == 0
    assert args.lr == pytest.approx(0.0)
    assert args.support_gate == "auto"
    assert args.support_gate_min_delta == pytest.approx(expected_min_delta)
    assert args.freeze_monthly_gain is True
    assert args.stage3_posterior_policy == "conservative_coeff_posterior"
    assert args.resolved_mode_defaults["support_gate_cycle_improvement_min_fraction"] == pytest.approx(expected_fraction)
    assert (
        args.resolved_mode_defaults["support_selection_objective"]
        == "loocv_mixed_raw_increment_wrmse_target_support_only"
    )
    assert args.target_eval_usage == "final_eval_only_no_selection"


def test_v11_mixed_raw_wrmse_scoring_rho_endpoints(monkeypatch, tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        apply_adapt_scope,
        load_source_checkpoint_for_few_shot,
        mixed_raw_increment_wrmse_objective_from_loader,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    apply_adapt_scope(state.model, "coeff_only")

    def fake_forward_prediction(*, state, batch, device, target_context_prompt_state):
        del target_context_prompt_state
        value = state.model.target_adapter_coefficient_residual_b.logit_delta.reshape(-1)[0].detach()
        height, width = batch["increment_surface"].shape[-2:]
        return torch.ones((1, 2, height, width), dtype=torch.float32, device=device) * value.to(device)

    monkeypatch.setattr(
        "scripts.train.train_hyperda_few_shot_adapt._ridge_forward_prediction",
        fake_forward_prediction,
    )
    anchor = {
        name: tensor.detach().clone()
        for name, tensor in state.model.state_dict().items()
        if name.startswith("target_") or name.startswith("residual_gain.")
    }
    candidate = {name: tensor.clone() for name, tensor in anchor.items()}
    candidate["target_adapter_coefficient_residual_b.logit_delta"] = candidate[
        "target_adapter_coefficient_residual_b.logit_delta"
    ] + 0.2
    batch = {
        "x": torch.zeros(1, 12, 4, 4),
        "months": torch.tensor([5], dtype=torch.long),
        "increment_surface": torch.ones(1, 4, 4) * 0.1,
        "increment_rootzone": torch.zeros(1, 4, 4),
        "loss_mask": torch.ones(1, 4, 4),
        "latitude_weight": torch.ones(1, 4, 4) * 2.0,
    }
    loader = torch.utils.data.DataLoader([batch], batch_size=None)
    prompt_state = {"global_prototype": torch.zeros(8), "monthly_prototypes": {}, "metadata": {}}

    anchor_score = mixed_raw_increment_wrmse_objective_from_loader(
        state=state,
        loader=loader,
        device=torch.device("cpu"),
        target_context_prompt_state=prompt_state,
        anchor_state=anchor,
        candidate_state=anchor,
        rho=1.0,
        normalize_increment=True,
        surface_weight=3.0,
        rootzone_weight=1.0,
        use_lat_weighted_loss=True,
    )
    rho0 = mixed_raw_increment_wrmse_objective_from_loader(
        state=state,
        loader=loader,
        device=torch.device("cpu"),
        target_context_prompt_state=prompt_state,
        anchor_state=anchor,
        candidate_state=candidate,
        rho=0.0,
        normalize_increment=True,
        surface_weight=3.0,
        rootzone_weight=1.0,
        use_lat_weighted_loss=True,
    )
    candidate_score = mixed_raw_increment_wrmse_objective_from_loader(
        state=state,
        loader=loader,
        device=torch.device("cpu"),
        target_context_prompt_state=prompt_state,
        anchor_state=anchor,
        candidate_state=candidate,
        rho=1.0,
        normalize_increment=True,
        surface_weight=3.0,
        rootzone_weight=1.0,
        use_lat_weighted_loss=True,
    )

    assert rho0["standard_support_objective_full_support"] == pytest.approx(
        anchor_score["standard_support_objective_full_support"]
    )
    assert candidate_score["standard_support_objective_full_support"] != pytest.approx(
        anchor_score["standard_support_objective_full_support"]
    )
    assert candidate_score["support_cycle_count"] == 1
    assert candidate_score["standard_support_increment_wrmse_full_support"] is not None


def test_v11_cv_summary_splits_nested_and_added_folds():
    from scripts.train.train_hyperda_few_shot_adapt import summarize_v11_cv_folds

    folds = [
        {"holdout_index": 0, "objective_delta": -0.2, "surface_delta": -0.1, "rootzone_delta": -0.1},
        {"holdout_index": 1, "objective_delta": -0.1, "surface_delta": -0.1, "rootzone_delta": 0.0},
        {"holdout_index": 4, "objective_delta": 0.1, "surface_delta": 0.0, "rootzone_delta": -0.1},
        {"holdout_index": 5, "objective_delta": -0.3, "surface_delta": -0.2, "rootzone_delta": -0.2},
    ]

    summary = summarize_v11_cv_folds(folds, nested_count=4)

    assert summary["cv_fold_count"] == 4
    assert summary["cv_cycle_improvement_fraction"] == pytest.approx(0.75)
    assert summary["cv_nested_k4_objective_delta"] == pytest.approx(-0.15)
    assert summary["cv_nested_k4_improvement_fraction"] == pytest.approx(1.0)
    assert summary["cv_added_objective_delta"] == pytest.approx(-0.1)
    assert summary["cv_added_improvement_fraction"] == pytest.approx(0.5)
    assert summary["cv_rootzone_nonregression_fraction"] == pytest.approx(1.0)


def test_v11_k4_gate_requires_loocv_all_cycles_and_rootzone():
    from scripts.train.train_hyperda_few_shot_adapt import decide_v11_k4_support_pool_gate

    ref = {
        "standard_support_objective_full_support": 1.0,
        "standard_support_rootzone_loss_full_support": 0.5,
    }
    candidate = {
        "standard_support_objective_full_support": 0.999,
        "standard_support_rootzone_loss_full_support": 0.499,
    }
    accepted = decide_v11_k4_support_pool_gate(
        full_candidate=candidate,
        full_reference=ref,
        cv_summary={
            "cv_reference_objective": 1.0,
            "cv_candidate_objective": 0.99999,
            "cv_objective_delta": -1.1e-5,
            "cv_rootzone_delta": 0.0,
            "cv_cycle_improvement_fraction": 1.0,
        },
        min_delta=5e-6,
        min_cycle_improvement_fraction=1.0,
        selected_candidate_id="k4_eb_shrink_balanced",
        support_candidate_pool=[{"candidate_id": "k4_eb_shrink_balanced"}],
    )
    assert accepted["stage3_posterior_decision"] == "accepted"
    assert accepted["support_selection_objective"] == "loocv_mixed_raw_increment_wrmse_target_support_only"

    weak = decide_v11_k4_support_pool_gate(
        full_candidate=candidate,
        full_reference=ref,
        cv_summary={
            "cv_reference_objective": 1.0,
            "cv_candidate_objective": 0.99999,
            "cv_objective_delta": -1.1e-5,
            "cv_rootzone_delta": 0.0,
            "cv_cycle_improvement_fraction": 0.75,
        },
        min_delta=5e-6,
        min_cycle_improvement_fraction=1.0,
        selected_candidate_id="k4_eb_shrink_balanced",
        support_candidate_pool=[{"candidate_id": "k4_eb_shrink_balanced"}],
    )
    assert weak["stage3_posterior_decision"] == "rejected_to_k0_anchor"
    assert "insufficient_loocv_cycle_improvement_fraction" in weak["support_gate_reject_reason"]


def test_v11_k12_gate_requires_nested_and_added_cv_stability():
    from scripts.train.train_hyperda_few_shot_adapt import decide_v11_k12_support_pool_gate

    ref = {
        "standard_support_objective_full_support": 1.0,
        "standard_support_rootzone_loss_full_support": 0.5,
    }
    candidate = {
        "standard_support_objective_full_support": 0.999,
        "standard_support_rootzone_loss_full_support": 0.499,
    }
    good_cv = {
        "cv_reference_objective": 1.0,
        "cv_candidate_objective": 0.99999,
        "cv_objective_delta": -3e-6,
        "cv_rootzone_delta": 0.0,
        "cv_cycle_improvement_fraction": 0.75,
        "cv_nested_k4_objective_delta": -2e-6,
        "cv_nested_k4_improvement_fraction": 0.75,
        "cv_added_objective_delta": -2e-6,
        "cv_added_improvement_fraction": 0.625,
    }

    accepted = decide_v11_k12_support_pool_gate(
        full_candidate=candidate,
        full_reference=ref,
        cv_summary=good_cv,
        min_delta=2e-6,
        min_cycle_improvement_fraction=2.0 / 3.0,
        selected_candidate_id="k12_eb_shrink_global",
        support_candidate_pool=[{"candidate_id": "k12_eb_shrink_global"}],
        support_nesting_policy="run_local_k12_nested_k4_plus_8_original_k12_nonduplicate",
        nested_support_dates_hash="nestedhash",
    )
    assert accepted["stage3_posterior_decision"] == "accepted"
    assert accepted["support_gate_status"] == "support_only_v11_loocv_k12_beats_k4_reference"
    assert accepted["k12_reference_policy"] == "k4_safe_nested_reference"

    weak_nested = decide_v11_k12_support_pool_gate(
        full_candidate=candidate,
        full_reference=ref,
        cv_summary={**good_cv, "cv_nested_k4_objective_delta": 1e-6},
        min_delta=2e-6,
        min_cycle_improvement_fraction=2.0 / 3.0,
        selected_candidate_id="k12_eb_shrink_global",
        support_candidate_pool=[{"candidate_id": "k12_eb_shrink_global"}],
    )
    assert weak_nested["stage3_posterior_decision"] == "fallback_to_k4_reference"
    assert "nested_k4_loocv_worse_than_k4_reference" in weak_nested["support_gate_reject_reason"]

    weak_added = decide_v11_k12_support_pool_gate(
        full_candidate=candidate,
        full_reference=ref,
        cv_summary={**good_cv, "cv_added_improvement_fraction": 0.5},
        min_delta=2e-6,
        min_cycle_improvement_fraction=2.0 / 3.0,
        selected_candidate_id="k12_eb_shrink_global",
        support_candidate_pool=[{"candidate_id": "k12_eb_shrink_global"}],
    )
    assert "added_support_loocv_not_stable" in weak_added["support_gate_reject_reason"]


def test_v12_support_gain_cv_summary_records_uncertainty_and_nested_added_splits():
    from scripts.train.train_hyperda_few_shot_adapt import summarize_support_gain_cv_folds

    folds = [
        {"holdout_index": 0, "candidate_objective": 0.8, "reference_objective": 1.0, "objective_delta": -0.2},
        {"holdout_index": 1, "candidate_objective": 0.9, "reference_objective": 1.0, "objective_delta": -0.1},
        {"holdout_index": 4, "candidate_objective": 1.1, "reference_objective": 1.0, "objective_delta": 0.1},
        {"holdout_index": 5, "candidate_objective": 0.7, "reference_objective": 1.0, "objective_delta": -0.3},
    ]

    summary = summarize_support_gain_cv_folds(folds, nested_count=4)
    deltas = np.asarray([-0.2, -0.1, 0.1, -0.3], dtype=np.float64)
    expected_se = float(np.std(deltas, ddof=1) / np.sqrt(float(len(deltas))))

    assert summary["cv_fold_count"] == 4
    assert summary["cv_candidate_objective"] == pytest.approx(0.875)
    assert summary["cv_reference_objective"] == pytest.approx(1.0)
    assert summary["cv_objective_delta"] == pytest.approx(-0.125)
    assert summary["cv_objective_delta_se"] == pytest.approx(expected_se)
    assert summary["cv_objective_delta_t"] == pytest.approx(-0.125 / expected_se)
    assert summary["cv_cycle_improvement_fraction"] == pytest.approx(0.75)
    assert summary["cv_nested_k4_objective_delta"] == pytest.approx(-0.15)
    assert summary["cv_nested_k4_improvement_fraction"] == pytest.approx(1.0)
    assert summary["cv_added_objective_delta"] == pytest.approx(-0.1)
    assert summary["cv_added_improvement_fraction"] == pytest.approx(0.5)


def test_v12_k4_gate_rejects_weak_support_gain_to_k0():
    from scripts.train.train_hyperda_few_shot_adapt import decide_support_gain_v12_gate

    gate = decide_support_gain_v12_gate(
        K=4,
        cv_summary={
            "cv_reference_objective": 1.0,
            "cv_candidate_objective": 0.999999,
            "cv_objective_delta": -1e-6,
            "cv_objective_delta_se": 1e-7,
            "cv_objective_delta_t": -10.0,
            "cv_cycle_improvement_fraction": 1.0,
        },
        selected_alpha=0.75,
        min_delta=5e-6,
        min_cycle_improvement_fraction=1.0,
    )

    assert gate["stage3_posterior_decision"] == "rejected_to_k0_anchor"
    assert gate["support_gate_status"] == "support_only_v12_nested_cv_support_gain_rejected_to_k0_anchor"
    assert "support_gain_cv_objective_not_materially_improved" in gate["support_gate_reject_reason"]
    assert gate["support_cv_objective_delta"] == pytest.approx(0.0)
    assert gate["support_selection_objective"] == "nested_cv_raw_increment_wrmse_support_gain_target_support_only"
    assert gate["target_eval_usage"] == "final_eval_only_no_selection"


def test_v12_k12_gate_rejects_noisy_or_added_weak_signal_to_k4():
    from scripts.train.train_hyperda_few_shot_adapt import decide_support_gain_v12_gate

    gate = decide_support_gain_v12_gate(
        K=12,
        cv_summary={
            "cv_reference_objective": 1.0,
            "cv_candidate_objective": 0.9999972,
            "cv_objective_delta": -2.8e-6,
            "cv_objective_delta_se": 2.2e-6,
            "cv_objective_delta_t": -1.27,
            "cv_cycle_improvement_fraction": 0.75,
            "cv_nested_k4_objective_delta": -7.0e-6,
            "cv_nested_k4_improvement_fraction": 0.75,
            "cv_added_objective_delta": -6.0e-7,
            "cv_added_improvement_fraction": 0.75,
        },
        selected_alpha=0.5,
        min_delta=2e-6,
        min_cycle_improvement_fraction=0.75,
    )

    assert gate["stage3_posterior_decision"] == "fallback_to_k4_reference"
    assert gate["support_gate_status"] == "support_only_v12_nested_cv_support_gain_fallback_to_k4_reference"
    assert "added_support_gain_cv_not_independently_material" in gate["support_gate_reject_reason"]
    assert gate["k12_reference_policy"] == "k4_safe_nested_reference"
    assert gate["support_cv_objective_delta"] == pytest.approx(0.0)


def test_v12_k12_gate_accepts_only_nested_and_added_material_cv_gain():
    from scripts.train.train_hyperda_few_shot_adapt import decide_support_gain_v12_gate

    gate = decide_support_gain_v12_gate(
        K=12,
        cv_summary={
            "cv_reference_objective": 1.0,
            "cv_candidate_objective": 0.999994,
            "cv_objective_delta": -6.0e-6,
            "cv_objective_delta_se": 2.0e-6,
            "cv_objective_delta_t": -3.0,
            "cv_cycle_improvement_fraction": 10.0 / 12.0,
            "cv_nested_k4_objective_delta": -4.0e-6,
            "cv_nested_k4_improvement_fraction": 0.75,
            "cv_added_objective_delta": -3.0e-6,
            "cv_added_improvement_fraction": 0.75,
        },
        selected_alpha=0.5,
        min_delta=2e-6,
        min_cycle_improvement_fraction=0.75,
    )

    assert gate["stage3_posterior_decision"] == "accepted"
    assert gate["support_gate_status"] == "support_only_v12_nested_cv_support_gain_accepted"
    assert gate["support_gate_reject_reason"] == []
    assert gate["support_cv_objective_delta"] == pytest.approx(-6.0e-6)
    assert gate["support_cv_objective_delta_se"] == pytest.approx(2.0e-6)
    assert gate["support_cv_objective_delta_t"] == pytest.approx(-3.0)


def test_v13_k12_candidate_pool_ids_and_hyperparameters_are_fixed():
    from scripts.train.train_hyperda_few_shot_adapt import diagnostic_support_gain_v13_k12_calibration_pool

    pool = diagnostic_support_gain_v13_k12_calibration_pool(12)

    assert [candidate["candidate_id"] for candidate in pool] == [
        "k12_v12_global_alpha",
        "k12_alpha2d_fine",
        "k12_global_affine_light",
        "k12_global_affine_stronger",
        "k12_seasonal_affine_light",
        "k12_seasonal_affine_aggressive",
        "k12_alpha2d_plus_global_affine",
    ]
    assert pool[0]["alpha_grid"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert pool[1]["alpha_grid_surface"] == [0.35, 0.45, 0.50, 0.60, 0.75, 0.90, 1.00]
    assert pool[1]["alpha_grid_rootzone"] == [0.35, 0.45, 0.50, 0.60, 0.75, 0.90, 1.00]
    assert pool[2]["ridge_lambda"] == pytest.approx(0.1)
    assert pool[2]["shrinkage_strength"] == pytest.approx(0.25)
    assert pool[3]["ridge_lambda"] == pytest.approx(0.05)
    assert pool[3]["shrinkage_strength"] == pytest.approx(0.10)
    assert pool[4]["season_shrinkage_strength"] == pytest.approx(8.0)
    assert pool[5]["season_shrinkage_strength"] == pytest.approx(4.0)
    assert pool[6]["candidate_type"] == "alpha2d_plus_global_affine"
    assert diagnostic_support_gain_v13_k12_calibration_pool(4) == []


def test_v13_alpha2d_fit_selects_independent_surface_and_rootzone_alpha():
    from scripts.train.train_hyperda_few_shot_adapt import _support_alpha2d_fit

    mask = np.ones((1, 1), dtype=np.float32)
    latw = np.ones((1, 1), dtype=np.float32)
    forecast = np.zeros((1, 1), dtype=np.float32)
    samples_s = [(np.asarray([[2.0]], dtype=np.float32), np.asarray([[1.0]], dtype=np.float32), forecast, mask, latw)]
    samples_r = [(np.asarray([[2.0]], dtype=np.float32), np.asarray([[2.0]], dtype=np.float32), forecast, mask, latw)]

    result = _support_alpha2d_fit(
        samples_s,
        samples_r,
        [0.5, 1.0],
        [0.5, 1.0],
        surface_weight=0.5,
        rootzone_weight=0.5,
    )

    assert result["best_alpha_surface"] == pytest.approx(0.5)
    assert result["best_alpha_rootzone"] == pytest.approx(1.0)


def test_v13_calibrated_score_applies_alpha_before_affine():
    from scripts.train.train_hyperda_few_shot_adapt import _support_calibrated_samples_score

    mask = np.ones((1, 1), dtype=np.float32)
    latw = np.ones((1, 1), dtype=np.float32)
    forecast = np.zeros((1, 1), dtype=np.float32)
    sample = (np.asarray([[2.0]], dtype=np.float32), np.asarray([[3.0]], dtype=np.float32), forecast, mask, latw)
    calibration = {
        "alpha_surface": 0.5,
        "alpha_rootzone": 0.5,
        "support_affine_calibration": {
            "support_affine_coefficients": {
                "surface": {"a": 2.0, "b": 1.0},
                "rootzone": {"a": 2.0, "b": 1.0},
            }
        },
    }

    score = _support_calibrated_samples_score(
        samples_s=[sample],
        samples_r=[sample],
        seasons=["DJF"],
        calibration=calibration,
        surface_weight=0.5,
        rootzone_weight=0.5,
    )

    assert score["standard_support_objective_full_support"] == pytest.approx(0.0)
    assert score["standard_support_surface_loss_full_support"] == pytest.approx(0.0)
    assert score["standard_support_rootzone_loss_full_support"] == pytest.approx(0.0)


def test_v13_selector_accepts_any_positive_cv_gain_with_rootzone_guard():
    from scripts.train.train_hyperda_few_shot_adapt import decide_v13_k12_aggressive_calibration_pool_gate

    gate = decide_v13_k12_aggressive_calibration_pool_gate(
        selected_candidate_id="k12_alpha2d_fine",
        support_candidate_pool=[{"candidate_id": "k12_alpha2d_fine"}],
        cv_summary={
            "cv_reference_objective": 1.0,
            "cv_candidate_objective": 0.999999999999,
            "cv_objective_delta": -1e-12,
            "cv_objective_delta_se": 2e-12,
            "cv_objective_delta_t": -0.5,
            "cv_rootzone_delta": 5e-6,
            "cv_cycle_improvement_fraction": 0.5,
        },
        support_calibration_dof=2.0,
    )

    assert gate["stage3_posterior_decision"] == "accepted"
    assert gate["support_gate_reject_reason"] == []
    assert gate["support_selection_objective"] == "k12_aggressive_nested_cv_calibration_pool_support_only"
    assert gate["target_eval_usage"] == "final_eval_only_no_selection"
    assert gate["k12_vs_k4_cv_objective_delta"] == pytest.approx(-1e-12)
    assert gate["k12_vs_k4_cv_rootzone_delta"] == pytest.approx(5e-6)
    assert gate["support_calibration_dof"] == pytest.approx(2.0)


def test_v13_selector_filters_gate_before_ranking_candidates():
    from scripts.train.train_hyperda_few_shot_adapt import _select_v13_k12_support_candidate_results

    candidate_results = [
        {
            "candidate_id": "k12_alpha2d_fine",
            "candidate_type": "alpha2d_grid",
            "support_calibration_dof": 2.0,
            "cv_summary": {
                "cv_reference_objective": 1.0,
                "cv_candidate_objective": 0.90,
                "cv_objective_delta": -0.10,
                "cv_rootzone_delta": 5.1e-6,
                "cv_cycle_improvement_fraction": 1.0,
            },
        },
        {
            "candidate_id": "k12_v12_global_alpha",
            "candidate_type": "alpha_global_grid",
            "support_calibration_dof": 1.0,
            "cv_summary": {
                "cv_reference_objective": 1.0,
                "cv_candidate_objective": 0.95,
                "cv_objective_delta": -0.05,
                "cv_rootzone_delta": 3.63e-6,
                "cv_cycle_improvement_fraction": 1.0,
            },
        },
    ]

    selection = _select_v13_k12_support_candidate_results(candidate_results)

    assert selection["eligible_support_candidate_count"] == 1
    assert selection["selected_result"]["candidate_id"] == "k12_v12_global_alpha"
    assert selection["gate_target_result"]["candidate_id"] == "k12_v12_global_alpha"
    assert selection["best_rejected_support_candidate"]["candidate_id"] == "k12_alpha2d_fine"
    assert candidate_results[0]["gate_status"] == "support_only_v13_k12_aggressive_calibration_pool_rejected"
    assert "rootzone_cv_regression_gt_5e-6" in candidate_results[0]["gate_reject_reason"]
    assert candidate_results[1]["gate_status"] == "support_only_v13_k12_aggressive_calibration_pool_eligible"
    assert candidate_results[1]["gate_reject_reason"] == []


def test_v13_selector_falls_back_only_when_all_candidates_fail_gate():
    from scripts.train.train_hyperda_few_shot_adapt import (
        _select_v13_k12_support_candidate_results,
        decide_v13_k12_aggressive_calibration_pool_gate,
    )

    candidate_results = [
        {
            "candidate_id": "k12_alpha2d_fine",
            "candidate_type": "alpha2d_grid",
            "support_calibration_dof": 2.0,
            "cv_summary": {
                "cv_reference_objective": 1.0,
                "cv_candidate_objective": 0.90,
                "cv_objective_delta": -0.10,
                "cv_rootzone_delta": 6e-6,
                "cv_cycle_improvement_fraction": 1.0,
            },
        },
        {
            "candidate_id": "k12_global_affine_light",
            "candidate_type": "global_affine",
            "support_calibration_dof": 4.0,
            "cv_summary": {
                "cv_reference_objective": 1.0,
                "cv_candidate_objective": 0.99,
                "cv_objective_delta": -0.01,
                "cv_rootzone_delta": -1e-6,
                "cv_cycle_improvement_fraction": 0.25,
            },
        },
    ]

    selection = _select_v13_k12_support_candidate_results(candidate_results)
    gate_target = selection["gate_target_result"]
    gate = decide_v13_k12_aggressive_calibration_pool_gate(
        selected_candidate_id=gate_target["candidate_id"],
        support_candidate_pool=[{"candidate_id": "k12_alpha2d_fine"}, {"candidate_id": "k12_global_affine_light"}],
        cv_summary=gate_target["cv_summary"],
        support_calibration_dof=gate_target["support_calibration_dof"],
        best_rejected_candidate=selection["best_rejected_support_candidate"],
    )

    assert selection["selected_result"] is None
    assert selection["eligible_support_candidate_count"] == 0
    assert selection["gate_target_result"]["candidate_id"] == "k12_alpha2d_fine"
    assert selection["best_rejected_support_candidate"]["candidate_id"] == "k12_alpha2d_fine"
    assert gate["stage3_posterior_decision"] == "fallback_to_k4_reference"
    assert "rootzone_cv_regression_gt_5e-6" in gate["support_gate_reject_reason"]
    assert gate["best_rejected_support_candidate"]["candidate_id"] == "k12_alpha2d_fine"


def test_v13_selector_accepts_v12_global_alpha_without_extra_margin():
    from scripts.train.train_hyperda_few_shot_adapt import _select_v13_k12_support_candidate_results

    candidate_results = [
        {
            "candidate_id": "k12_v12_global_alpha",
            "candidate_type": "alpha_global_grid",
            "support_calibration_dof": 1.0,
            "cv_summary": {
                "cv_reference_objective": 1.0,
                "cv_candidate_objective": 0.999999,
                "cv_objective_delta": -1e-6,
                "cv_rootzone_delta": 3.63e-6,
                "cv_cycle_improvement_fraction": 1.0,
            },
        }
    ]

    selection = _select_v13_k12_support_candidate_results(candidate_results)

    assert selection["eligible_support_candidate_count"] == 1
    assert selection["selected_result"]["candidate_id"] == "k12_v12_global_alpha"
    assert selection["best_rejected_support_candidate"] == {}
    assert candidate_results[0]["gate_reject_reason"] == []


def test_v13_selector_fallbacks_when_rootzone_cv_regresses_more_than_tolerance():
    from scripts.train.train_hyperda_few_shot_adapt import decide_v13_k12_aggressive_calibration_pool_gate

    gate = decide_v13_k12_aggressive_calibration_pool_gate(
        selected_candidate_id="k12_global_affine_light",
        support_candidate_pool=[{"candidate_id": "k12_global_affine_light"}],
        cv_summary={
            "cv_reference_objective": 1.0,
            "cv_candidate_objective": 0.99,
            "cv_objective_delta": -0.01,
            "cv_rootzone_delta": 5.1e-6,
            "cv_cycle_improvement_fraction": 1.0,
        },
        support_calibration_dof=4.0,
        best_rejected_candidate={"candidate_id": "k12_global_affine_light"},
    )

    assert gate["stage3_posterior_decision"] == "fallback_to_k4_reference"
    assert "rootzone_cv_regression_gt_5e-6" in gate["support_gate_reject_reason"]
    assert gate["support_cv_objective_delta"] == pytest.approx(0.0)
    assert gate["k12_vs_k4_cv_objective_delta"] == pytest.approx(-0.01)
    assert gate["best_rejected_support_candidate"]["candidate_id"] == "k12_global_affine_light"


def test_v13_selector_records_best_rejected_candidate_when_not_better_than_k4():
    from scripts.train.train_hyperda_few_shot_adapt import decide_v13_k12_aggressive_calibration_pool_gate

    gate = decide_v13_k12_aggressive_calibration_pool_gate(
        selected_candidate_id="k12_seasonal_affine_light",
        support_candidate_pool=[{"candidate_id": "k12_seasonal_affine_light"}],
        cv_summary={
            "cv_reference_objective": 1.0,
            "cv_candidate_objective": 1.0001,
            "cv_objective_delta": 0.0001,
            "cv_rootzone_delta": -0.01,
            "cv_cycle_improvement_fraction": 0.75,
        },
        support_calibration_dof=4.0,
        best_rejected_candidate={
            "candidate_id": "k12_seasonal_affine_light",
            "cv_summary": {"cv_objective_delta": 0.0001},
        },
    )

    assert gate["stage3_posterior_decision"] == "fallback_to_k4_reference"
    assert "k12_candidate_not_better_than_k4_reference_cv" in gate["support_gate_reject_reason"]
    assert gate["best_rejected_support_candidate"]["candidate_id"] == "k12_seasonal_affine_light"
    assert gate["support_candidate_pool"] == [{"candidate_id": "k12_seasonal_affine_light"}]


def test_diagnostic_v13_is_non_paper_facing_even_when_accepted():
    from scripts.train.train_hyperda_few_shot_adapt import paper_facing_status_for_stage3

    status = paper_facing_status_for_stage3(
        K=12,
        policy_source="diagnostic_direct_target_support",
        stage3_posterior_decision="accepted",
        stage3_kshot_mode="diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
    )

    assert status["paper_facing_run"] is False
    assert (
        status["diagnostic_run_reason"]
        == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool_target_support_update"
    )


def test_diagnostic_v12_support_gain_is_non_paper_facing_even_when_accepted():
    from scripts.train.train_hyperda_few_shot_adapt import paper_facing_status_for_stage3

    status = paper_facing_status_for_stage3(
        K=12,
        policy_source="diagnostic_direct_target_support",
        stage3_posterior_decision="accepted",
        stage3_kshot_mode="diagnostic_support_gain_v12_nested_cv",
    )

    assert status["paper_facing_run"] is False
    assert (
        status["diagnostic_run_reason"]
        == "diagnostic_support_gain_v12_nested_cv_target_support_update"
    )


def test_diagnostic_v6_is_non_paper_facing_even_when_accepted():
    from scripts.train.train_hyperda_few_shot_adapt import paper_facing_status_for_stage3

    status = paper_facing_status_for_stage3(
        K=12,
        policy_source="diagnostic_direct_target_support",
        stage3_posterior_decision="accepted",
        stage3_kshot_mode="diagnostic_linearized_coeff_ridge_v6_nested",
    )

    assert status["paper_facing_run"] is False
    assert (
        status["diagnostic_run_reason"]
        == "diagnostic_linearized_coeff_ridge_v6_nested_target_support_update"
    )


def test_diagnostic_v7_is_non_paper_facing_even_when_accepted():
    from scripts.train.train_hyperda_few_shot_adapt import paper_facing_status_for_stage3

    status = paper_facing_status_for_stage3(
        K=12,
        policy_source="diagnostic_direct_target_support",
        stage3_posterior_decision="accepted",
        stage3_kshot_mode="diagnostic_linearized_coeff_ridge_v7_balanced_nested",
    )

    assert status["paper_facing_run"] is False
    assert (
        status["diagnostic_run_reason"]
        == "diagnostic_linearized_coeff_ridge_v7_balanced_nested_target_support_update"
    )


def test_diagnostic_v8_hybrid_is_non_paper_facing_even_when_accepted():
    from scripts.train.train_hyperda_few_shot_adapt import paper_facing_status_for_stage3

    status = paper_facing_status_for_stage3(
        K=12,
        policy_source="diagnostic_direct_target_support",
        stage3_posterior_decision="accepted",
        stage3_kshot_mode="diagnostic_linearized_coeff_ridge_v8_hybrid_nested",
    )

    assert status["paper_facing_run"] is False
    assert (
        status["diagnostic_run_reason"]
        == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested_target_support_update"
    )


def test_diagnostic_v9_guarded_is_non_paper_facing_even_when_accepted():
    from scripts.train.train_hyperda_few_shot_adapt import paper_facing_status_for_stage3

    status = paper_facing_status_for_stage3(
        K=12,
        policy_source="diagnostic_direct_target_support",
        stage3_posterior_decision="accepted",
        stage3_kshot_mode="diagnostic_linearized_coeff_ridge_v9_guarded_nested",
    )

    assert status["paper_facing_run"] is False
    assert (
        status["diagnostic_run_reason"]
        == "diagnostic_linearized_coeff_ridge_v9_guarded_nested_target_support_update"
    )


def test_parse_args_paper_safe_requires_source_policy_for_kshot(monkeypatch, tmp_path):
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
            "--stage3_kshot_mode",
            "paper_safe",
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
            "--stage3_kshot_mode",
            "diagnostic_direct_kshot",
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
            "--stage3_kshot_mode",
            "diagnostic_direct_kshot",
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
            "--stage3_kshot_mode",
            "diagnostic_direct_kshot",
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


def test_strict_source_policy_rejects_k4_no_update_for_conservative_stage3(monkeypatch, tmp_path):
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

    with pytest.raises(SystemExit):
        runner.parse_args()


def test_diagnostic_source_policy_can_select_k4_no_update_when_not_required(monkeypatch, tmp_path):
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
            "--stage3_kshot_mode",
            "diagnostic_direct_kshot",
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
    policy_path = tmp_path / "safe_policy_bad_scope.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy_source": "source_side_episode_calibration",
                "target_val_usage": "unused_in_main_protocol",
                "target_eval_usage": "final_eval_only_no_selection",
                "policies": {
                    "few_shot_k4": {
                        "adapt_scope": "safe_operator",
                        "lr": 1e-3,
                        "adaptation_steps": 20,
                        "anchor_alpha": 0.5,
                        "adapt_mix_rho": 1.0,
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
            "--adapt_scope",
            "safe_operator",
            "--safe_policy_json",
            str(policy_path),
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
            "--stage3_kshot_mode",
            "diagnostic_direct_kshot",
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


def test_source_side_safe_policy_default_support_gate_is_diagnostic_off(monkeypatch, tmp_path):
    from scripts.train import train_hyperda_few_shot_adapt as runner

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    policy_path = tmp_path / "safe_policy_nonzero.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy_source": "source_side_episode_calibration",
                "target_val_usage": "unused_in_main_protocol",
                "target_eval_usage": "final_eval_only_no_selection",
                "source_calibration": {
                    "kshot_policy_update_requirement": "nonzero_update",
                    "no_update_candidates_allowed": False,
                },
                "policies": {
                    "few_shot_k4": {
                        "adapt_scope": "coeff_only",
                        "lr": 3e-4,
                        "adaptation_steps": 20,
                        "anchor_alpha": 0.25,
                        "adapt_mix_rho": 1.0,
                        "source_calibrated_candidate_id": "K4_nonzero",
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

    assert args.support_gate == "off"
    assert args.adapt_scope == "coeff_only"
    assert args.adaptation_steps == 20
    assert args.adapt_mix_rho == pytest.approx(1.0)


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


def test_diagnostic_direct_v2_support_risk_guard_rejects_conflict_and_large_prompt_drift():
    from scripts.train.train_hyperda_few_shot_adapt import apply_diagnostic_direct_v2_support_risk_guard

    summary = {
        "support_gate_enabled": False,
        "support_gate_status": "disabled",
        "stage3_posterior_decision": "accepted",
        "support_gate_reject_reason": [],
    }

    guarded = apply_diagnostic_direct_v2_support_risk_guard(
        summary=summary,
        stage3_kshot_mode="diagnostic_direct_kshot_v2",
        support_gradient_diagnostics={
            "support_gradient_negative_fraction": 0.3333333333,
            "support_gradient_cosine_min": -0.287,
        },
        target_parameter_drift={"target_prompt": 2.03, "total": 2.06},
    )

    assert guarded["stage3_posterior_decision"] == "rejected_to_k0_anchor"
    assert guarded["support_gate_status"] == "diagnostic_v2_support_risk_rejected_to_k0_anchor"
    assert "support_gradient_conflict" in guarded["support_gate_reject_reason"]
    assert "target_prompt_drift_exceeds_v2_guard" in guarded["support_gate_reject_reason"]
    assert guarded["diagnostic_v2_support_risk_guard"]["target_eval_usage"] == "final_eval_only_no_selection"


def test_diagnostic_direct_v2_support_risk_guard_leaves_low_risk_update_accepted():
    from scripts.train.train_hyperda_few_shot_adapt import apply_diagnostic_direct_v2_support_risk_guard

    summary = {
        "support_gate_enabled": False,
        "support_gate_status": "disabled",
        "stage3_posterior_decision": "accepted",
        "support_gate_reject_reason": [],
    }

    guarded = apply_diagnostic_direct_v2_support_risk_guard(
        summary=summary,
        stage3_kshot_mode="diagnostic_direct_kshot_v2",
        support_gradient_diagnostics={
            "support_gradient_negative_fraction": 0.0,
            "support_gradient_cosine_min": 0.05,
        },
        target_parameter_drift={"target_prompt": 0.25, "total": 0.4},
    )

    assert guarded["stage3_posterior_decision"] == "accepted"
    assert guarded["support_gate_status"] == "disabled"
    assert guarded["support_gate_reject_reason"] == []
    assert guarded["diagnostic_v2_support_risk_guard"]["status"] == "passed"


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


def test_source_calibrated_no_update_kshot_is_k0_equivalent_diagnostic():
    from scripts.train.train_hyperda_few_shot_adapt import (
        build_stage3_target_posterior_state,
        paper_facing_status_for_stage3,
    )

    status = paper_facing_status_for_stage3(
        K=4,
        policy_source="source_side_episode_calibration",
        stage3_posterior_decision="no_update",
    )

    assert status["paper_facing_run"] is False
    assert status["diagnostic_run_reason"] == "source_policy_selected_no_update_k0_equivalent"

    anchor = {"target_prompt.latent": torch.zeros(2)}
    posterior = build_stage3_target_posterior_state(
        anchor_state=anchor,
        final_state={name: tensor.clone() for name, tensor in anchor.items()},
        K=4,
        adapt_scope="none",
        anchor_alpha=0.0,
        adaptation_steps=0,
        target_labels_loaded=True,
        target_labels_used=False,
        source_prior_hash_before="sourcehash",
        source_prior_hash_after="sourcehash",
        stage3_posterior_policy="safe_operator_ablation",
        stage3_posterior_decision="no_update",
        support_gate_status="source_calibrated_no_update",
        paper_selection_basis="source_side_safe_policy_selected_no_update",
        stage3_acceptance_basis="source_side_safe_policy_selected_no_update_k0_equivalent",
    )
    metadata = posterior["metadata"]

    assert metadata["stage3_no_update_contract"] == "Kshot_source_side_policy_selected_no_update"
    assert metadata["support_gate_policy_role"] == "source_side_policy_selected_no_update"
    assert metadata["target_specific_parameter_updates"] == 0
    assert metadata["stage3_acceptance_basis"] == "source_side_safe_policy_selected_no_update_k0_equivalent"


def test_diagnostic_direct_kshot_is_non_paper_facing_even_when_accepted():
    from scripts.train.train_hyperda_few_shot_adapt import paper_facing_status_for_stage3

    status = paper_facing_status_for_stage3(
        K=4,
        policy_source="diagnostic_direct_target_support",
        stage3_posterior_decision="accepted",
        stage3_kshot_mode="diagnostic_direct_kshot",
    )

    assert status["paper_facing_run"] is False
    assert status["diagnostic_run_reason"] == "diagnostic_direct_kshot_target_support_update"


def test_diagnostic_direct_kshot_v2_is_non_paper_facing_even_when_accepted():
    from scripts.train.train_hyperda_few_shot_adapt import paper_facing_status_for_stage3

    status = paper_facing_status_for_stage3(
        K=12,
        policy_source="diagnostic_direct_target_support",
        stage3_posterior_decision="accepted",
        stage3_kshot_mode="diagnostic_direct_kshot_v2",
    )

    assert status["paper_facing_run"] is False
    assert status["diagnostic_run_reason"] == "diagnostic_direct_kshot_v2_target_support_update"


def test_diagnostic_support_gain_v2_is_non_paper_facing_even_when_accepted():
    from scripts.train.train_hyperda_few_shot_adapt import paper_facing_status_for_stage3

    status = paper_facing_status_for_stage3(
        K=12,
        policy_source="diagnostic_direct_target_support",
        stage3_posterior_decision="accepted",
        stage3_kshot_mode="diagnostic_support_gain_v2",
    )

    assert status["paper_facing_run"] is False
    assert status["diagnostic_run_reason"] == "diagnostic_support_gain_v2_target_support_update"


def test_diagnostic_support_gain_v3_stable_is_non_paper_facing_even_when_accepted():
    from scripts.train.train_hyperda_few_shot_adapt import paper_facing_status_for_stage3

    status = paper_facing_status_for_stage3(
        K=12,
        policy_source="diagnostic_direct_target_support",
        stage3_posterior_decision="accepted",
        stage3_kshot_mode="diagnostic_support_gain_v3_stable",
    )

    assert status["paper_facing_run"] is False
    assert status["diagnostic_run_reason"] == "diagnostic_support_gain_v3_stable_target_support_update"


def test_diagnostic_support_gain_v4_nested_stable_is_non_paper_facing_even_when_accepted():
    from scripts.train.train_hyperda_few_shot_adapt import paper_facing_status_for_stage3

    status = paper_facing_status_for_stage3(
        K=12,
        policy_source="diagnostic_direct_target_support",
        stage3_posterior_decision="accepted",
        stage3_kshot_mode="diagnostic_support_gain_v4_nested_stable",
    )

    assert status["paper_facing_run"] is False
    assert (
        status["diagnostic_run_reason"]
        == "diagnostic_support_gain_v4_nested_stable_target_support_update"
    )


def test_diagnostic_support_affine_v1_nested_is_non_paper_facing_even_when_accepted():
    from scripts.train.train_hyperda_few_shot_adapt import paper_facing_status_for_stage3

    status = paper_facing_status_for_stage3(
        K=12,
        policy_source="diagnostic_direct_target_support",
        stage3_posterior_decision="accepted",
        stage3_kshot_mode="diagnostic_support_affine_v1_nested",
    )

    assert status["paper_facing_run"] is False
    assert (
        status["diagnostic_run_reason"]
        == "diagnostic_support_affine_v1_nested_target_support_update"
    )


def test_support_gain_v3_stable_selector_prefers_larger_alpha_when_tied():
    from hydroda.training.calibration import select_stable_residual_gain_alpha

    result = select_stable_residual_gain_alpha(
        alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
        per_alpha_results={
            0.0: {"min_skill": 0.20, "mean_skill": 0.30},
            0.25: {"min_skill": 0.24, "mean_skill": 0.33},
            0.5: {"min_skill": 0.300, "mean_skill": 0.410},
            0.75: {"min_skill": 0.285, "mean_skill": 0.400},
            1.0: {"min_skill": 0.270, "mean_skill": 0.460},
        },
    )

    assert result["selected_alpha"] == pytest.approx(0.75)
    assert result["best_alpha_raw"] == pytest.approx(0.5)
    assert result["stable_candidate_alphas"] == [0.5, 0.75]
    assert result["selection_margin"] == pytest.approx(0.015)
    assert result["selection_rule"] == "stable_high_alpha_with_mean_skill_guard"


def test_support_affine_calibration_fits_per_variable_coefficients_and_changes_hash():
    from hydroda.training.calibration import (
        apply_residual_affine_calibration_to_increment,
        calibrate_residual_affine,
        prediction_content_hash_for_increments,
    )

    mask = np.ones((2, 2), dtype=np.float32)
    latw = np.ones((2, 2), dtype=np.float32)
    forecast = np.zeros((2, 2), dtype=np.float32)
    pred_s = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    true_s = (2.0 * pred_s + 1.0).astype(np.float32)
    pred_r = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    true_r = np.full((2, 2), 1.0, dtype=np.float32)
    samples_s = [(pred_s, true_s, forecast, mask, latw)]
    samples_r = [(pred_r, true_r, forecast, mask, latw)]

    result = calibrate_residual_affine(
        samples_s,
        samples_r,
        ridge_lambda=0.0,
        shrinkage_strength=0.0,
        K=4,
    )

    assert result["calibration_mode"] == "target_support_residual_affine_v1"
    assert result["label_source"] == "target_support_only"
    assert result["target_eval_usage"] == "final_eval_only_no_selection"
    assert result["support_affine_coefficients"]["surface"]["a"] == pytest.approx(2.0)
    assert result["support_affine_coefficients"]["surface"]["b"] == pytest.approx(1.0)
    assert result["support_affine_coefficients"]["rootzone"]["a"] == pytest.approx(0.0)
    assert result["support_affine_coefficients"]["rootzone"]["b"] == pytest.approx(1.0)
    assert result["effective_calibration_dof"] == pytest.approx(4.0)

    before_hash = prediction_content_hash_for_increments(
        pred_increment_surface=pred_s,
        pred_increment_rootzone=pred_r,
    )
    after_s, after_r = apply_residual_affine_calibration_to_increment(
        pred_increment_surface=pred_s,
        pred_increment_rootzone=pred_r,
        calibration=result,
    )
    after_hash = prediction_content_hash_for_increments(
        pred_increment_surface=after_s,
        pred_increment_rootzone=after_r,
    )

    assert np.allclose(after_s, true_s)
    assert np.allclose(after_r, true_r)
    assert after_hash != before_hash


def test_support_affine_k12_season_coefficients_shrink_toward_global_when_sparse():
    from hydroda.training.calibration import calibrate_residual_affine

    mask = np.ones((1, 1), dtype=np.float32)
    latw = np.ones((1, 1), dtype=np.float32)
    forecast = np.zeros((1, 1), dtype=np.float32)
    samples_s = [
        (np.asarray([[1.0]], dtype=np.float32), np.asarray([[2.0]], dtype=np.float32), forecast, mask, latw),
        (np.asarray([[2.0]], dtype=np.float32), np.asarray([[4.0]], dtype=np.float32), forecast, mask, latw),
    ]
    samples_r = [
        (np.asarray([[1.0]], dtype=np.float32), np.asarray([[1.0]], dtype=np.float32), forecast, mask, latw),
        (np.asarray([[2.0]], dtype=np.float32), np.asarray([[2.0]], dtype=np.float32), forecast, mask, latw),
    ]

    result = calibrate_residual_affine(
        samples_s,
        samples_r,
        seasons=["DJF", "DJF"],
        ridge_lambda=0.0,
        season_shrinkage_strength=10.0,
        K=12,
    )

    assert result["support_affine_coefficients"]["surface"]["a"] == pytest.approx(2.0)
    assert result["seasonal_affine_coefficients"]["DJF"]["surface"]["a"] == pytest.approx(2.0)
    assert result["seasonal_affine_coefficients"]["DJF"]["rootzone"]["a"] == pytest.approx(1.0)
    assert result["effective_calibration_dof"] < 8.0
    assert result["support_affine_coefficients"]["surface"]["fallback_rule"] == "global_per_variable_affine"


def test_diagnostic_safe_operator_v5_k12_fallback_uses_only_nested_support_evidence():
    from scripts.train.train_hyperda_few_shot_adapt import decide_k12_vs_k4_reference_gate

    candidate = {
        "standard_support_objective_full_support": 1.20,
        "standard_support_loss_full_support": 1.10,
        "standard_support_surface_loss_full_support": 0.50,
        "standard_support_rootzone_loss_full_support": 0.60,
    }
    k4_reference = {
        "standard_support_objective_full_support": 1.00,
        "standard_support_loss_full_support": 0.90,
        "standard_support_surface_loss_full_support": 0.40,
        "standard_support_rootzone_loss_full_support": 0.50,
    }

    fallback = decide_k12_vs_k4_reference_gate(
        candidate=candidate,
        k4_reference=k4_reference,
        enabled=True,
        min_delta=0.0,
        k4_reference_adapt_mix_rho=0.50,
        support_nesting_policy="run_local_k12_nested_k4_plus_8_original_k12_nonduplicate",
        nested_support_dates_hash="nested-hash",
    )

    assert fallback["stage3_posterior_decision"] == "fallback_to_k4_reference"
    assert fallback["support_gate_status"] == "support_only_k12_fallback_to_k4_reference"
    assert fallback["support_gate_label_source"] == "target_support_only"
    assert fallback["target_eval_usage"] == "final_eval_only_no_selection"
    assert fallback["support_gate_reject_reason"] == [
        "k12_not_better_than_k4_reference_on_nested_support"
    ]
    assert fallback["k4_reference_support_objective"] == pytest.approx(1.0)
    assert fallback["k12_candidate_support_objective"] == pytest.approx(1.2)
    assert fallback["k4_reference_adapt_mix_rho"] == pytest.approx(0.50)
    assert fallback["support_nesting_policy"] == "run_local_k12_nested_k4_plus_8_original_k12_nonduplicate"
    assert fallback["nested_support_dates_hash"] == "nested-hash"

    accepted = decide_k12_vs_k4_reference_gate(
        candidate={**candidate, "standard_support_objective_full_support": 0.95},
        k4_reference=k4_reference,
        enabled=True,
        min_delta=0.0,
        support_nesting_policy="nested",
        nested_support_dates_hash="hash",
    )
    assert accepted["stage3_posterior_decision"] == "accepted"
    assert accepted["support_gate_status"] == "support_only_k12_beats_k4_reference"
    assert accepted["support_gate_reject_reason"] == []

    weak_margin = decide_k12_vs_k4_reference_gate(
        candidate={**candidate, "standard_support_objective_full_support": 0.998},
        k4_reference=k4_reference,
        enabled=True,
        min_delta=0.003,
        support_nesting_policy="nested",
        nested_support_dates_hash="hash",
    )
    assert weak_margin["stage3_posterior_decision"] == "fallback_to_k4_reference"
    assert weak_margin["support_gate_status"] == "support_only_k12_fallback_to_k4_reference"
    assert weak_margin["k12_vs_k4_support_objective_delta"] == pytest.approx(-0.002)


def test_diagnostic_v9_defers_k0_rejection_to_k4_reference_gate():
    from scripts.train.train_hyperda_few_shot_adapt import (
        defer_k0_anchor_gate_to_k4_reference_gate,
    )

    generic_rejection = {
        "support_gate_enabled": True,
        "support_gate_status": "support_only_rejected_to_k0_anchor",
        "stage3_posterior_decision": "rejected_to_k0_anchor",
        "support_gate_reject_reason": ["objective_not_improved"],
        "support_candidate_objective_delta": 1.0e-4,
    }

    deferred, original = defer_k0_anchor_gate_to_k4_reference_gate(generic_rejection)

    assert deferred["stage3_posterior_decision"] == "accepted"
    assert deferred["support_gate_status"] == "support_only_k12_pending_k4_reference_gate"
    assert deferred["support_only_gate_status"] == "support_only_rejected_to_k0_anchor"
    assert deferred["support_gate_reject_reason"] == []
    assert deferred["k0_anchor_gate_deferred_to_k4_reference"] is True
    assert deferred["k0_anchor_gate"]["support_gate_reject_reason"] == ["objective_not_improved"]
    assert original["stage3_posterior_decision"] == "rejected_to_k0_anchor"


def test_support_gain_v4_nested_stable_selector_uses_dual_variable_guard_and_selects_075():
    from hydroda.training.calibration import select_stable_residual_gain_alpha

    result = select_stable_residual_gain_alpha(
        alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
        per_alpha_results={
            0.0: {"min_skill": 0.000, "mean_skill": 0.010, "surface_skill": 0.010, "rootzone_skill": 0.000},
            0.25: {"min_skill": 0.100, "mean_skill": 0.180, "surface_skill": 0.260, "rootzone_skill": 0.100},
            0.5: {"min_skill": 0.200, "mean_skill": 0.240, "surface_skill": 0.280, "rootzone_skill": 0.200},
            0.75: {"min_skill": 0.185, "mean_skill": 0.232, "surface_skill": 0.279, "rootzone_skill": 0.185},
            1.0: {"min_skill": 0.165, "mean_skill": 0.230, "surface_skill": 0.295, "rootzone_skill": 0.165},
        },
        selection_rule="support_uncertainty_stable_high_alpha_with_dual_guard",
        paired_support_se_capped=0.04,
    )

    assert result["selection_rule"] == "support_uncertainty_stable_high_alpha_with_dual_guard"
    assert result["best_alpha_raw"] == pytest.approx(0.5)
    assert result["selected_alpha"] == pytest.approx(0.75)
    assert result["stable_candidate_alphas"] == [0.5, 0.75]
    assert result["selection_margin"] == pytest.approx(0.015)
    assert result["stability_tolerance"] == pytest.approx(0.04)


def test_support_gain_v3_calibration_metadata_records_support_only_source():
    from hydroda.training.calibration import calibrate_residual_gain

    mask = np.ones((2, 2), dtype=np.float32)
    latw = np.ones((2, 2), dtype=np.float32)
    forecast = np.zeros((2, 2), dtype=np.float32)
    true_inc = np.ones((2, 2), dtype=np.float32)
    pred_inc = (true_inc / 0.5).astype(np.float32)
    samples = [(pred_inc, true_inc, forecast, mask, latw)]

    result = calibrate_residual_gain(
        samples,
        samples,
        [0.0, 0.25, 0.5, 0.75, 1.0],
        selection_rule="stable_high_alpha_with_mean_skill_guard",
    )

    assert result["calibration_mode"] == "target_support_residual_gain_stable_grid"
    assert result["label_source"] == "target_support_only"
    assert result["target_eval_usage"] == "final_eval_only_no_selection"
    assert result["selection_rule"] == "stable_high_alpha_with_mean_skill_guard"
    assert result["best_alpha_raw"] == pytest.approx(0.5)


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


def test_load_source_checkpoint_for_few_shot_preserves_phys_context_operator(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import load_source_checkpoint_for_few_shot

    ckpt_path = tmp_path / "source_phys_context.pt"
    _write_phys_context_source_checkpoint(ckpt_path)
    source = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    assert state.model.hyper_phys_context_modulation is True
    assert state.model.phys_operator_residual is not None
    assert state.model.phys_context_source == "raw_input_side_da_diagnostics"
    assert torch.allclose(
        state.model.state_dict()["phys_operator_residual.delta_head.bias"],
        source["model_state_dict"]["phys_operator_residual.delta_head.bias"],
    )


def test_save_few_shot_checkpoint_config_preserves_phys_context_operator(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        load_source_checkpoint_for_few_shot,
        save_few_shot_checkpoint,
    )

    ckpt_path = tmp_path / "source_phys_context.pt"
    _write_phys_context_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
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
            "eval_month_usage": "month_selects_frozen_target_context_prototype",
        },
    }

    save_few_shot_checkpoint(
        path=out,
        state=state,
        optimizer_state_dict={},
        config={
            "K": 0,
            "adaptation_setting": "zero_shot_context",
            "target_region": "US-R1",
            "seed": 0,
            "source_checkpoint": str(ckpt_path),
            "target_context_dates_hash": "contexthash",
        },
        target_context_prompt_state=prompt_state,
        train_history=[],
    )
    saved = torch.load(out, map_location="cpu", weights_only=False)

    assert saved["config"]["hyper_phys_context_modulation"] is True
    assert saved["config"]["phys_context_source"] == "raw_input_side_da_diagnostics"
    assert saved["config"]["hyper_operator_droppath_p"] == pytest.approx(0.10)
    assert "phys_operator_residual.delta_head.bias" in saved["model_state_dict"]


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


def test_parse_args_accepts_context_tta_mode(monkeypatch, tmp_path):
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
            "--context_tta_mode",
            "prompt_feature_alignment",
        ],
    )

    args = runner.parse_args()

    assert args.context_tta_mode == "prompt_feature_alignment"


def test_paper_facing_status_marks_residual_shift_tta_diagnostic():
    from scripts.train.train_hyperda_few_shot_adapt import paper_facing_status_for_stage3

    residual = paper_facing_status_for_stage3(
        K=0,
        policy_source="preregistered_default",
        stage3_posterior_decision="no_update",
        context_tta_mode="context_prompt_residual_shift",
        context_tta_effective=True,
        context_tta_source_stat_status="target_context_only_no_source_statistics_required",
    )
    aligned = paper_facing_status_for_stage3(
        K=0,
        policy_source="preregistered_default",
        stage3_posterior_decision="no_update",
        context_tta_mode="prompt_feature_alignment",
        context_tta_effective=True,
        context_tta_source_stat_status="source_fit_source_val_only",
    )

    assert residual["paper_facing_run"] is False
    assert residual["diagnostic_run_reason"] == (
        "context_tta_context_prompt_residual_shift_not_source_side_paper_safe"
    )
    assert aligned["paper_facing_run"] is True


def test_evaluate_checkpoint_parse_args_accepts_raw_adapted_diagnostic_flag(monkeypatch, tmp_path):
    text = Path("scripts/eval/evaluate_checkpoint.py").read_text()

    assert "--eval_raw_adapted_before_mix" in text
    assert "raw_adapted_state_dict" in text
    assert "raw_adapted_predictor=raw_adapted_predictor" in text


def test_few_shot_target_context_prompt_state_records_context_tta_hash(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        build_few_shot_target_context_prompt_state,
        load_source_checkpoint_for_few_shot,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    prompt_state = build_few_shot_target_context_prompt_state(
        state=state,
        samples=[
            {
                "x": np.ones((12, 2, 2), dtype=np.float32),
                "month": 1,
                "date_str": "2015-01-15",
                "region_mask": np.ones((2, 2), dtype=np.float32),
            }
        ],
        target_region="US-R1",
        device=torch.device("cpu"),
        context_hash="ctxhash",
        context_tta_mode="prompt_feature_alignment",
    )

    assert prompt_state["context_tta_mode"] == "prompt_feature_alignment"
    assert prompt_state["context_tta_state_hash"]
    assert prompt_state["context_tta_label_usage"] == "none"
    assert prompt_state["metadata"]["context_tta_state_hash"] == prompt_state["context_tta_state_hash"]


def test_prompt_feature_alignment_identity_fallback_is_marked_ineffective(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        build_few_shot_target_context_prompt_state,
        load_source_checkpoint_for_few_shot,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )

    prompt_state = build_few_shot_target_context_prompt_state(
        state=state,
        samples=[
            {
                "x": np.ones((12, 2, 2), dtype=np.float32),
                "month": 1,
                "date_str": "2015-01-15",
                "region_mask": np.ones((2, 2), dtype=np.float32),
            }
        ],
        target_region="US-R1",
        device=torch.device("cpu"),
        context_hash="ctxhash",
        context_tta_mode="prompt_feature_alignment",
    )

    assert prompt_state["context_tta_effective"] is False
    assert prompt_state["context_tta_source_stat_status"] == "identity_fallback_no_source_statistics"
    assert prompt_state["metadata"]["context_tta_effective"] is False
    assert prompt_state["metadata"]["context_tta_source_stat_status"] == "identity_fallback_no_source_statistics"


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
            "--stage3_kshot_mode",
            "diagnostic_direct_kshot",
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
            "target_support_labels_used_for_parameter_update": True,
            "target_support_labels_used_for_optimizer_update": True,
            "target_support_labels_used_for_ridge_solve": False,
            "target_support_labels_used_for_calibration": False,
            "target_support_labels_used_for_support_gate": True,
            "few_shot_update_type": "parameter_update_only",
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
    assert cfg["residual_gain_alpha_surface"] == pytest.approx(1.0)
    assert cfg["residual_gain_alpha_rootzone"] == pytest.approx(1.0)
    assert saved["residual_gain_alpha_surface"] == pytest.approx(1.0)
    assert saved["residual_gain_alpha_rootzone"] == pytest.approx(1.0)
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
    assert cfg["target_support_labels_used_for_parameter_update"] is True
    assert cfg["target_support_labels_used_for_optimizer_update"] is True
    assert cfg["target_support_labels_used_for_ridge_solve"] is False
    assert cfg["target_support_labels_used_for_calibration"] is False
    assert cfg["target_support_labels_used_for_support_gate"] is True
    assert cfg["few_shot_update_type"] == "parameter_update_only"
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


def test_save_few_shot_checkpoint_records_support_gain_calibration(tmp_path):
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
    calibration = {
        "calibration_mode": "target_support_residual_gain_fixed_grid",
        "label_source": "target_support_only",
        "best_alpha_surface": 0.75,
        "best_alpha_rootzone": 0.25,
        "selection_score": 0.1,
        "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
    }

    save_few_shot_checkpoint(
        path=out,
        state=state,
        optimizer_state_dict={},
        config={
            "K": 12,
            "adaptation_setting": "few_shot_k12",
            "stage3_kshot_mode": "diagnostic_support_gain_v2",
            "stage3_posterior_policy": "source_calibrated_mix",
            "stage3_posterior_decision": "accepted",
            "adapt_scope": "none",
            "adapt_solver": "adamw",
            "anchor_alpha": 0.0,
            "adaptation_steps": 0,
            "target_region": "US-R1",
            "source_checkpoint": str(ckpt_path),
            "split_manifest_path": "artifacts/splits/US_loro_zero_few_shot_splits.json",
            "target_context_dates_hash": "contexthash",
            "target_support_dates_hash": "supporthash",
            "target_eval_dates_hash": "evalhash",
            "paper_facing_run": False,
            "policy_source": "diagnostic_direct_target_support",
            "source_checkpoint_sha256": "abc123",
            "target_labels_loaded_for_adaptation": True,
            "target_labels_used_for_adaptation": True,
            "target_support_labels_used_for_parameter_update": False,
            "target_support_labels_used_for_optimizer_update": False,
            "target_support_labels_used_for_ridge_solve": False,
            "target_support_labels_used_for_calibration": True,
            "target_support_labels_used_for_support_gate": False,
            "few_shot_update_type": "support_calibration_only",
            "target_support_count": 12,
            "support_manifest_hash": "support_manifest_sha",
            "residual_gain_alpha_surface": 0.75,
            "residual_gain_alpha_rootzone": 0.25,
            "support_gain_calibration": calibration,
        },
        target_context_prompt_state=prompt_state,
        train_history=[],
    )

    saved = torch.load(out, map_location="cpu", weights_only=False)
    cfg = saved["config"]

    assert saved["residual_gain_alpha_surface"] == pytest.approx(0.75)
    assert saved["residual_gain_alpha_rootzone"] == pytest.approx(0.25)
    assert cfg["residual_gain_alpha_surface"] == pytest.approx(0.75)
    assert cfg["residual_gain_alpha_rootzone"] == pytest.approx(0.25)
    assert cfg["support_gain_calibration"]["calibration_mode"] == "target_support_residual_gain_fixed_grid"
    assert cfg["support_gain_calibration"]["label_source"] == "target_support_only"
    assert cfg["target_labels_loaded_for_adaptation"] is True
    assert cfg["target_labels_used_for_adaptation"] is True
    assert cfg["target_support_labels_used_for_parameter_update"] is False
    assert cfg["target_support_labels_used_for_optimizer_update"] is False
    assert cfg["target_support_labels_used_for_calibration"] is True
    assert cfg["few_shot_update_type"] == "support_calibration_only"


def test_save_few_shot_checkpoint_records_v4_nested_support_gain_metadata(tmp_path):
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
    calibration = {
        "calibration_mode": "target_support_residual_gain_v4_nested_stable_grid",
        "label_source": "target_support_only",
        "target_eval_usage": "final_eval_only_no_selection",
        "selection_rule": "support_uncertainty_stable_high_alpha_with_dual_guard",
        "best_alpha_raw": 0.5,
        "stable_candidate_alphas": [0.5, 0.75],
        "selection_margin": 0.015,
        "stability_tolerance": 0.025,
        "paired_support_se_capped": 0.012,
        "best_alpha_surface": 0.75,
        "best_alpha_rootzone": 0.75,
        "support_nesting_policy": "run_local_k12_nested_k4_plus_8_original_k12_nonduplicate",
        "nested_support_dates_hash": "nestedhash12",
    }

    save_few_shot_checkpoint(
        path=out,
        state=state,
        optimizer_state_dict={},
        config={
            "K": 12,
            "adaptation_setting": "few_shot_k12",
            "stage3_kshot_mode": "diagnostic_support_gain_v4_nested_stable",
            "stage3_posterior_policy": "source_calibrated_mix",
            "stage3_posterior_decision": "accepted",
            "adapt_scope": "none",
            "adapt_solver": "adamw",
            "anchor_alpha": 0.0,
            "adaptation_steps": 0,
            "target_region": "US-R1",
            "source_checkpoint": str(ckpt_path),
            "split_manifest_path": str(tmp_path / "nested_splits.json"),
            "target_context_dates_hash": "contexthash",
            "target_support_dates_hash": "nestedhash12",
            "target_eval_dates_hash": "evalhash",
            "paper_facing_run": False,
            "policy_source": "diagnostic_direct_target_support",
            "source_checkpoint_sha256": "abc123",
            "target_labels_loaded_for_adaptation": True,
            "target_labels_used_for_adaptation": True,
            "target_support_count": 12,
            "support_manifest_hash": "support_manifest_sha",
            "support_nesting_policy": "run_local_k12_nested_k4_plus_8_original_k12_nonduplicate",
            "nested_support_dates_hash": "nestedhash12",
            "nested_support_manifest": str(tmp_path / "nested_support.json"),
            "residual_gain_alpha_surface": 0.75,
            "residual_gain_alpha_rootzone": 0.75,
            "support_gain_calibration": calibration,
        },
        target_context_prompt_state=prompt_state,
        train_history=[],
    )

    saved = torch.load(out, map_location="cpu", weights_only=False)
    cfg = saved["config"]

    assert cfg["support_nesting_policy"] == "run_local_k12_nested_k4_plus_8_original_k12_nonduplicate"
    assert cfg["nested_support_dates_hash"] == "nestedhash12"
    assert cfg["support_gain_calibration"]["selection_rule"] == "support_uncertainty_stable_high_alpha_with_dual_guard"
    assert cfg["support_gain_calibration"]["target_eval_usage"] == "final_eval_only_no_selection"
    assert cfg["support_gain_calibration"]["paired_support_se_capped"] == pytest.approx(0.012)


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


def test_kshot_checkpoint_preserves_raw_adapted_state_after_gate_rollback(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import (
        FewShotAdaptationState,
        extract_target_adapter_state,
        load_source_checkpoint_for_few_shot,
        save_few_shot_checkpoint,
        target_parameter_l2_drift,
    )
    from scripts.train.train_hyperda_target_adapt import apply_target_adapter_state

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
    anchor_state = extract_target_adapter_state(state.model)
    with torch.no_grad():
        state.model.target_adapter_coefficient_residual_b.logit_delta.add_(1.0)
    raw_state = extract_target_adapter_state(state.model)
    raw_drift = target_parameter_l2_drift(anchor_state, raw_state)
    apply_target_adapter_state(state.model, anchor_state)
    final_drift = target_parameter_l2_drift(anchor_state, extract_target_adapter_state(state.model))
    prompt_state = {
        "schema_version": "target_context_prompt_state_v1",
        "prompt_source": "target_context_monthly_prompt_prototypes",
        "label_usage": "none",
        "context_hash": "ctxhash",
        "monthly_counts": {str(i): 0 for i in range(1, 13)},
        "global_prototype": torch.zeros(8),
        "monthly_prototypes": {str(i): None for i in range(1, 13)},
        "metadata": {},
    }

    out = tmp_path / "checkpoint_final_preregistered.pt"
    saved_config = save_few_shot_checkpoint(
        path=out,
        state=state,
        optimizer_state_dict={},
        config={
            "K": 4,
            "adaptation_setting": "few_shot_k4",
            "target_region": "US-R1",
            "target_context_dates_hash": "ctxhash",
            "target_adapter_anchor_state": anchor_state,
            "raw_adapted_adapter_state": raw_state,
            "target_parameter_l2_drift_pre_anchor": raw_drift,
            "target_parameter_l2_drift_post_anchor": final_drift,
            "target_parameter_l2_drift": final_drift,
            "stage3_posterior_policy": "conservative_coeff_posterior",
            "stage3_posterior_decision": "rejected_to_k0_anchor",
            "support_gate_status": "missing_source_policy_rejected_to_k0_anchor",
            "support_gate_reject_reason": ["missing_source_side_safe_policy_json"],
            "policy_source": "preregistered_default",
            "adapt_mix_rho": 0.0,
            "adapt_scope": "coeff_only",
            "adaptation_steps": 5,
            "anchor_alpha": 0.25,
            "support_loss_before": 1.0,
            "support_loss_after": 1.0,
            "support_loss_delta": 0.0,
            "target_labels_loaded_for_adaptation": True,
            "target_labels_used_for_adaptation": True,
        },
        target_context_prompt_state=prompt_state,
        train_history=[],
    )
    ckpt = torch.load(out, map_location="cpu", weights_only=False)

    assert saved_config["raw_adapted_state"]["drift_from_prior"]["total"] > 0.0
    assert saved_config["post_gate_state"]["drift_from_prior"]["total"] == pytest.approx(0.0)
    assert saved_config["raw_adapted_state_hash"] != saved_config["post_gate_state_hash"]
    assert ckpt["raw_adapted_state_dict"]["target_adapter_state_hash"] == saved_config["raw_adapted_state_hash"]
    assert ckpt["post_gate_state_dict"]["target_adapter_state_hash"] == saved_config["post_gate_state_hash"]
    assert ckpt["raw_adapted_state_dict"]["drift_from_prior"]["total"] == pytest.approx(raw_drift["total"])
    assert ckpt["final_eval_mix_state"]["adapt_mix_rho"] == pytest.approx(0.0)
    assert ckpt["config"]["final_eval_mix_state"]["post_gate_state_hash"] == saved_config["post_gate_state_hash"]


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
        ridge_weighting="global_pixel_l2",
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
        ridge_weighting="global_pixel_l2",
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
        ridge_weighting="global_pixel_l2",
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
        ridge_weighting="global_pixel_l2",
        surface_weight=3.0,
        rootzone_weight=1.0,
        use_lat_weighted_loss=False,
    )

    assert diagnostics["masked_pixel_count"] == 512
    assert diagnostics["masked_observation_count"] == 1024
    assert diagnostics["feature_pixel_count"] == 10
    assert diagnostics["feature_observation_count"] == 20


def test_ridge_balanced_huber_weights_balance_cycles_and_downweight_outliers():
    from scripts.train.train_hyperda_few_shot_adapt import _ridge_balanced_huber_observation_weights

    batch = {
        "loss_mask": torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
    }
    pred = torch.zeros(2, 2, 2, 2)
    residual = torch.ones_like(pred) * 0.5
    residual[1, :, 0, 0] = 4.0

    weights = _ridge_balanced_huber_observation_weights(
        batch=batch,
        pred=pred,
        residual=residual,
        surface_weight=3.0,
        rootzone_weight=1.0,
        use_lat_weighted_loss=False,
        delta=1.0,
    )

    assert weights.shape == pred.shape
    assert weights[0, 0].sum().item() == pytest.approx(3.0)
    assert weights[1, 0].sum().item() == pytest.approx(3.0)
    assert weights[0, 1].sum().item() == pytest.approx(1.0)
    assert weights[1, 1].sum().item() == pytest.approx(1.0)
    assert weights[1, 0, 0, 0].item() < weights[1, 0, 0, 1].item()
    assert weights[0, 0, 0, 0].item() == pytest.approx(0.75)


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
        "context_tta_mode": "prompt_feature_alignment",
        "context_tta_state_hash": "tta-state-hash",
        "context_tta_label_usage": "none",
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
        "raw_adapted_state": {
            "schema_version": "hyperda_stage3_raw_adapted_state_v1",
            "stage_name": "raw_adapted",
            "target_adapter_state_hash": "rawhash",
            "drift_from_prior": {"total": 0.2},
        },
        "post_gate_state": {
            "schema_version": "hyperda_stage3_post_gate_state_v1",
            "stage_name": "post_gate",
            "target_adapter_state_hash": "posthash",
            "drift_from_prior": {"total": 0.0},
        },
        "final_eval_mix_state": {
            "schema_version": "hyperda_stage3_final_eval_mix_state_v1",
            "adapt_mix_rho": 1.0,
            "raw_adapted_state_hash": "rawhash",
            "post_gate_state_hash": "posthash",
        },
        "raw_adapted_state_hash": "rawhash",
        "post_gate_state_hash": "posthash",
        "raw_adapted_drift_from_prior": {"total": 0.2},
        "post_gate_drift_from_prior": {"total": 0.0},
        "adapt_scope": "none",
        "adapt_solver": "ridge_coeff",
        "freeze_monthly_gain": True,
        "ridge_lambda": 1.25,
        "ridge_clip_coeff_norm": 0.8,
        "ridge_trust_region_radius": 0.7,
        "ridge_max_feature_pixels": 2000,
        "ridge_standardize_features": True,
        "ridge_weighting": "cycle_variable_balanced_huber",
        "ridge_diagnostics": {
            "status": "solved",
            "ridge_weighting": "cycle_variable_balanced_huber",
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
        "target_support_labels_used_for_parameter_update": False,
        "target_support_labels_used_for_optimizer_update": False,
        "target_support_labels_used_for_ridge_solve": False,
        "target_support_labels_used_for_calibration": False,
        "target_support_labels_used_for_support_gate": False,
        "few_shot_update_type": "no_target_label_update",
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
    assert metadata["context_tta_mode"] == "prompt_feature_alignment"
    assert metadata["context_tta_state_hash"] == "tta-state-hash"
    assert metadata["context_tta_label_usage"] == "none"
    assert metadata["target_support_dates_hash"] == "supporthash"
    assert metadata["target_support_dates"] == []
    assert metadata["support_manifest_hash"] == "support_manifest_sha"
    assert metadata["support_nesting_hash"] == "nesting_sha"
    assert metadata["support_nesting_status"] == "K0_no_support"
    assert metadata["target_eval_dates_hash"] == "evalhash"
    assert metadata["raw_adapted_state"]["target_adapter_state_hash"] == "rawhash"
    assert metadata["post_gate_state"]["target_adapter_state_hash"] == "posthash"
    assert metadata["final_eval_mix_state"]["raw_adapted_state_hash"] == "rawhash"
    assert metadata["raw_adapted_state_hash"] == "rawhash"
    assert metadata["post_gate_state_hash"] == "posthash"
    assert metadata["raw_adapted_drift_from_prior"]["total"] == pytest.approx(0.2)
    assert metadata["post_gate_drift_from_prior"]["total"] == pytest.approx(0.0)
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
    assert metadata["ridge_weighting"] == "cycle_variable_balanced_huber"
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
    assert metadata["target_support_labels_used_for_parameter_update"] is False
    assert metadata["target_support_labels_used_for_optimizer_update"] is False
    assert metadata["target_support_labels_used_for_ridge_solve"] is False
    assert metadata["target_support_labels_used_for_calibration"] is False
    assert metadata["target_support_labels_used_for_support_gate"] is False
    assert metadata["few_shot_update_type"] == "no_target_label_update"
    assert metadata["target_eval_usage"] == "final_eval_only_no_selection"


def test_few_shot_run_metadata_sidecar_records_v14_parameter_update_plus_calibration(tmp_path):
    from scripts.train.train_hyperda_few_shot_adapt import write_run_metadata_sidecar

    checkpoint_path = tmp_path / "checkpoints" / "checkpoint_final_preregistered.pt"
    config = {
        "method": "hyperda_diagnostic_few_shot_k12",
        "adaptation_setting": "few_shot_k12",
        "stage3_kshot_mode": "diagnostic_finetune_support_gain_v14_nested",
        "adapt_recipe": "source_anchor",
        "adapt_scope": "coeff_only",
        "adapt_solver": "adamw",
        "stage3_posterior_policy": "conservative_coeff_posterior",
        "stage3_posterior_decision": "accepted",
        "support_gate_enabled": True,
        "support_gate_status": "accepted",
        "policy_source": "diagnostic_direct_target_support",
        "paper_facing_run": False,
        "K": 12,
        "seed": 0,
        "target_region": "US-R1",
        "anchor_alpha": 0.50,
        "adapt_mix_rho": 1.0,
        "adaptation_steps": 80,
        "actual_optimizer_steps": 80,
        "optimizer_steps_run": 80,
        "support_batch_count": 2,
        "effective_support_passes": 40.0,
        "target_support_count": 12,
        "target_labels_loaded_for_adaptation": True,
        "target_labels_used_for_adaptation": True,
        "target_support_labels_used_for_parameter_update": True,
        "target_support_labels_used_for_optimizer_update": True,
        "target_support_labels_used_for_ridge_solve": False,
        "target_support_labels_used_for_calibration": True,
        "target_support_labels_used_for_support_gate": True,
        "few_shot_update_type": "parameter_update_plus_support_calibration",
        "target_support_dates": ["2019-04-15", "2020-05-16"],
        "target_context_dates_hash": "contexthash",
        "target_support_dates_hash": "supporthash",
        "target_eval_dates_hash": "evalhash",
        "support_manifest_hash": "support_manifest_sha",
        "support_nesting_hash": "nesting_sha",
        "support_nesting_status": "nested_k12_support_manifest",
        "support_nesting_policy": "run_local_k12_nested_k4_plus_8_original_k12_nonduplicate",
        "nested_support_dates_hash": "nestedhash12",
        "split_manifest_path": "artifacts/splits/US_loro_zero_few_shot_splits.json",
        "split_manifest_sha256": "splitsha",
        "target_context_prompt_state": {"schema_version": "target_context_prompt_state_v1"},
        "source_anchor_hyperparameter_source": (
            "diagnostic_finetune_support_gain_v14_nested_support_only_coeff_finetune_plus_gain"
        ),
        "target_eval_usage": "final_eval_only_no_selection",
        "residual_gain_alpha_surface": 1.0,
        "residual_gain_alpha_rootzone": 1.0,
        "support_gain_calibration": {
            "calibration_mode": "target_support_residual_gain_v14_after_finetune_nested_cv",
            "status": "gain_identity_after_finetune",
            "v14_role": "auxiliary_support_gain_after_target_parameter_finetune",
            "gain_gate_decision_is_auxiliary": True,
            "support_gain_gate": {
                "stage3_posterior_decision": "gain_identity_after_finetune",
                "support_gate_status": "support_gain_identity_after_finetune",
            },
        },
    }

    write_run_metadata_sidecar(tmp_path, checkpoint_path, config)

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["stage3_kshot_mode"] == "diagnostic_finetune_support_gain_v14_nested"
    assert metadata["adapt_scope"] == "coeff_only"
    assert metadata["adapt_solver"] == "adamw"
    assert metadata["adaptation_steps"] == 80
    assert metadata["actual_optimizer_steps"] == 80
    assert metadata["target_labels_loaded_for_adaptation"] is True
    assert metadata["target_labels_used_for_adaptation"] is True
    assert metadata["target_support_labels_used_for_parameter_update"] is True
    assert metadata["target_support_labels_used_for_optimizer_update"] is True
    assert metadata["target_support_labels_used_for_ridge_solve"] is False
    assert metadata["target_support_labels_used_for_calibration"] is True
    assert metadata["target_support_labels_used_for_support_gate"] is True
    assert metadata["few_shot_update_type"] == "parameter_update_plus_support_calibration"
    assert metadata["target_eval_usage"] == "final_eval_only_no_selection"
    assert (
        metadata["support_gain_calibration"]["v14_role"]
        == "auxiliary_support_gain_after_target_parameter_finetune"
    )


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
