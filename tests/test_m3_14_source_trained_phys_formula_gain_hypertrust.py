from __future__ import annotations

import os
import subprocess

import pytest
import torch

from hydroda.models.phys_trust import PHYS_FORMULA_GAIN_SOURCE
from scripts.train.train_prompt_conditioned_shared import PromptConditionedTrainer


def test_m3_14_sign_consistency_regularizer_penalizes_dry_innovation_wet_increment():
    trainer = object.__new__(PromptConditionedTrainer)
    trainer.hyper_phys_consistency_regularization_weight = 0.01
    trainer.phys_context_source = PHYS_FORMULA_GAIN_SOURCE
    trainer.target_increment_normalization = False
    trainer._inc_mean = None
    trainer._inc_std = None

    x_raw = torch.zeros(1, 12, 2, 2, dtype=torch.float32)
    x_raw[:, 5] = 102.0
    x_raw[:, 6] = 112.0
    x_raw[:, 7] = 1.0
    x_raw[:, 8] = 1.0
    x_raw[:, 9] = 100.0
    x_raw[:, 10] = 110.0
    loss_mask = torch.ones(1, 2, 2, dtype=torch.bool)

    wet_increment = torch.full((1, 2, 2, 2), 0.10, dtype=torch.float32)
    dry_increment = torch.full((1, 2, 2, 2), -0.10, dtype=torch.float32)

    conflict = trainer._phys_formula_gain_sign_consistency_penalty(wet_increment, x_raw, loss_mask)
    aligned = trainer._phys_formula_gain_sign_consistency_penalty(dry_increment, x_raw, loss_mask)

    assert conflict.item() > 0.0
    assert aligned.item() == pytest.approx(0.0)


def test_m3_14_wrapper_refuses_m3_1_warm_start(tmp_path):
    fake_source = tmp_path / "source.pt"
    fake_source.write_text("stub", encoding="utf-8")
    env = {
        **os.environ,
        "ABLATION_ID": "M3_14_source_trained_phys_formula_gain_hypertrust",
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


def test_m3_14_wrapper_refuses_deferred_regions_before_freeze_confirmation(tmp_path):
    fake_source = tmp_path / "source.pt"
    fake_source.write_text("stub", encoding="utf-8")
    env = {
        **os.environ,
        "ABLATION_ID": "M3_14_source_trained_phys_formula_gain_hypertrust",
        "RESUME_FROM_M3_1_BEST": "0",
        "DATASET_BACKEND": "netcdf",
    }

    result = subprocess.run(
        ["bash", "run/phase4_hyperda_staged_ablation.sh", str(fake_source), "US-R2", "0", "0", "--dry-run"],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "restricted to US-R1" in result.stderr
    assert "US-R2..US-R6 are deferred" in result.stderr


def test_m3_14_wrapper_refuses_final_output_physics_residual(tmp_path):
    fake_source = tmp_path / "source.pt"
    fake_source.write_text("stub", encoding="utf-8")
    env = {
        **os.environ,
        "ABLATION_ID": "M3_14_source_trained_phys_formula_gain_hypertrust",
        "RESUME_FROM_M3_1_BEST": "0",
        "HYPER_PHYS_GAIN_BASIS_RESIDUAL": "1",
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
    assert "forbids final-output physics residual branches" in result.stderr


def test_m3_14_wrapper_dry_run_uses_clean_source_formula_gain_flags(tmp_path):
    fake_source = tmp_path / "source.pt"
    fake_source.write_text("stub", encoding="utf-8")
    env = {
        **os.environ,
        "ABLATION_ID": "M3_14_source_trained_phys_formula_gain_hypertrust",
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
    assert "ablation_id=M3_14_source_trained_phys_formula_gain_hypertrust" in stdout
    assert "checkpoint_start=source_pooled_global_backbone" in stdout
    assert "current_ablation_policy=US-R1_seed0_K0_only" in stdout
    assert "warm_start_policy=none_clean_source_only_checkpoint_full_hypernetwork_training" in stdout
    assert "phys_context_source=raw_input_side_formula_gain" in stdout
    assert "phys_formula_source=raw_input_side_formula_gain" in stdout
    assert "phys_formula_schema=m3_14_raw_input_side_formula_gain_v1" in stdout
    assert "final_output_residual_allowed=false" in stdout
    assert "hyper_phys_gain_basis_residual=0" in stdout
    assert "hyper_phys_consistency_regularization_weight=0.01" in stdout
    assert "--phys_context_source raw_input_side_formula_gain" in stdout
    assert "--phys_formula_source raw_input_side_formula_gain" in stdout
    assert "--hyper_phys_delta_scale 0.05" in stdout
    assert "--hyper_phys_gate_init 0.50" in stdout
    assert "--hyper_phys_gain_basis_residual 0" in stdout
    assert "--hyper_phys_consistency_regularization_weight 0.01" in stdout
    assert f"--init_from_source_base_checkpoint {fake_source}" in stdout
    assert "--init_from_prompt_checkpoint" not in stdout
