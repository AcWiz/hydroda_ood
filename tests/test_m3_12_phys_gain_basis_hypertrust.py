from __future__ import annotations

import os
import subprocess

import pytest
import torch

from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.phys_trust import build_phys_gain_source_bank, phys_gain_basis_from_raw_tensor


def _raw_x(vod: float = 0.0) -> torch.Tensor:
    x = torch.zeros(1, 12, 4, 4, dtype=torch.float32)
    x[:, 0] = 0.30
    x[:, 1] = 0.25
    x[:, 2] = 285.0
    x[:, 3] = 280.0
    x[:, 4] = float(vod)
    x[:, 5] = 102.0
    x[:, 6] = 112.0
    x[:, 7] = 1.0
    x[:, 8] = 1.0
    x[:, 9] = 100.0
    x[:, 10] = 110.0
    x[:, 11, :, :2] = 1.0
    return x


def test_phys_gain_basis_sign_convention_and_tau_omega_attenuation():
    low_vod_basis, summary = phys_gain_basis_from_raw_tensor(_raw_x(vod=0.0))
    high_vod_basis, _ = phys_gain_basis_from_raw_tensor(_raw_x(vod=3.0))

    assert summary["formula"]["m_p"].startswith("-d_p")
    assert summary["base_valid_mask_usage"] == "diagnostic_only_not_loss_metric_obs_region_mask_or_gate_mask"
    assert torch.all(low_vod_basis[:, 0] < 0.0)
    assert torch.all(low_vod_basis[:, 1] < 0.0)
    assert high_vod_basis[:, 0].abs().mean() < low_vod_basis[:, 0].abs().mean()
    assert high_vod_basis[:, 1].abs().mean() < low_vod_basis[:, 1].abs().mean()


def test_phys_gain_source_bank_uses_source_fit_only_and_rejects_target_roles():
    sample = {
        "split_role": "source_fit",
        "sample_region_id": "US-R2",
        "month": 1,
        "x": _raw_x(vod=0.2).squeeze(0),
        "increment_surface": torch.full((4, 4), 0.01),
        "increment_rootzone": torch.full((4, 4), 0.005),
        "loss_mask": torch.ones(4, 4, dtype=torch.bool),
    }

    bank = build_phys_gain_source_bank([sample])

    assert bank["source_split_roles"]["bank"] == ["source_fit"]
    assert bank["target_eval_usage"] == "final_eval_only_no_selection"
    assert bank["source_gain_bank_hash"]
    assert bank["base_valid_mask_usage"] == "diagnostic_only_not_loss_metric_obs_region_mask_or_gate_mask"
    poisoned = dict(sample, split_role="target_full_train")
    with pytest.raises(ValueError, match="refuses sample split_role"):
        build_phys_gain_source_bank([poisoned])
    with pytest.raises(ValueError, match="forbids target-side split roles"):
        build_phys_gain_source_bank([sample], source_split_roles=("source_fit", "target_eval"))


def test_phys_gain_basis_zero_init_preserves_hyperda_design_path_output():
    torch.manual_seed(17)
    baseline = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        zero_shot_prior_form="source_base_residual_reliability_gated",
    ).eval()
    torch.manual_seed(17)
    phys = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        zero_shot_prior_form="source_base_residual_reliability_gated",
        hyper_phys_gain_basis_residual=True,
    ).eval()
    x = torch.randn(2, 12, 16, 16)
    z = torch.randn(2, 16)
    reliability = torch.zeros(2, 5)

    with torch.no_grad():
        baseline_pred = baseline(x, z, reliability_features=reliability, x_raw=x)
        phys_pred = phys(x, z, reliability_features=reliability, x_raw=x)

    assert torch.allclose(phys_pred, baseline_pred, atol=1e-7)
    assert phys.last_phys_gain_basis_summary["enabled"] is True
    assert phys.last_phys_gain_basis_summary["residual_abs_mean"] == pytest.approx(0.0)


def test_m3_12_wrapper_refuses_m3_1_warm_start(tmp_path):
    fake_source = tmp_path / "source.pt"
    fake_source.write_text("stub", encoding="utf-8")
    env = {
        **os.environ,
        "ABLATION_ID": "M3_12_phys_gain_basis_hypertrust",
        "RESUME_FROM_M3_1_BEST": "1",
        "DATASET_BACKEND": "netcdf",
    }

    result = subprocess.run(
        ["bash", "run/phase4_hyperda_staged_ablation.sh", str(fake_source), "US-R1", "0", "0", "--dry-run"],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "cannot warm-start from M3_1" in result.stderr


def test_m3_12_wrapper_dry_run_uses_clean_source_stage_flags(tmp_path):
    fake_source = tmp_path / "source.pt"
    fake_source.write_text("stub", encoding="utf-8")
    env = {
        **os.environ,
        "ABLATION_ID": "M3_12_phys_gain_basis_hypertrust",
        "RESUME_FROM_M3_1_BEST": "0",
        "DATASET_BACKEND": "netcdf",
        "TIMESTAMP": "20260101_000000",
    }

    result = subprocess.run(
        ["bash", "run/phase4_hyperda_staged_ablation.sh", str(fake_source), "US-R1", "0", "0", "--dry-run"],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = result.stdout
    assert "ablation_id=M3_12_phys_gain_basis_hypertrust" in stdout
    assert "hyper_phys_gain_basis_residual=1" in stdout
    assert "warm_start_policy=none_clean_source_only_checkpoint_full_hypernetwork_training" in stdout
    assert "--hyper_phys_gain_basis_residual 1" in stdout
    assert f"--init_from_source_base_checkpoint {fake_source}" in stdout
    assert "--init_from_prompt_checkpoint" not in stdout
