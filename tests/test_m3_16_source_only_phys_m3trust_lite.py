from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_m3_16_wrapper_refuses_m3_1_warm_start(tmp_path: Path):
    fake_source = tmp_path / "source.pt"
    fake_source.write_text("stub", encoding="utf-8")
    env = {
        **os.environ,
        "ABLATION_ID": "M3_16_source_only_phys_m3trust_lite",
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


def test_m3_16_wrapper_refuses_deferred_regions_before_freeze(tmp_path: Path):
    fake_source = tmp_path / "source.pt"
    fake_source.write_text("stub", encoding="utf-8")
    env = {
        **os.environ,
        "ABLATION_ID": "M3_16_source_only_phys_m3trust_lite",
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


def test_m3_16_wrapper_refuses_final_output_physics_residual(tmp_path: Path):
    fake_source = tmp_path / "source.pt"
    fake_source.write_text("stub", encoding="utf-8")
    env = {
        **os.environ,
        "ABLATION_ID": "M3_16_source_only_phys_m3trust_lite",
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


def test_m3_16_wrapper_dry_run_uses_source_only_lite_physics_flags(tmp_path: Path):
    fake_source = tmp_path / "source.pt"
    fake_source.write_text("stub", encoding="utf-8")
    env = {
        **os.environ,
        "ABLATION_ID": "M3_16_source_only_phys_m3trust_lite",
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
    assert "ablation_id=M3_16_source_only_phys_m3trust_lite" in stdout
    assert "diagnostic_status=source_only_phys_m3trust_lite_mainline_candidate" in stdout
    assert "checkpoint_start=source_pooled_global_backbone" in stdout
    assert "stage2_source_only_invariant=true" in stdout
    assert "active_stage2_physics_mainline=true" in stdout
    assert "m3_1_design_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std" in stdout
    assert "trainable_scope=source_base_frozen_adapter_film" in stdout
    assert "phys_context_source=raw_input_side_formula_gain" in stdout
    assert "phys_formula_source=raw_input_side_formula_gain" in stdout
    assert "phys_formula_schema=m3_14_raw_input_side_formula_gain_v1" in stdout
    assert "phys_formula_delta_scale=0.03" in stdout
    assert "phys_formula_gate_init=0.25" in stdout
    assert "phys_formula_operator_droppath_p=0.10" in stdout
    assert "final_output_residual_allowed=false" in stdout
    assert "second_model_forward_allowed=false" in stdout
    assert "hyper_phys_gain_basis_residual=0" in stdout
    assert "hyper_phys_consistency_regularization_weight=0.0" in stdout
    assert "source_fit_gain_bank_forbidden_roles=target_context,target_val,target_eval,target_full_train" in stdout
    assert "current_region_policy=US-R1_seed0_K0_only_R2_to_R6_deferred_until_M3_16_freeze" in stdout
    assert "warm_start_policy=none_clean_source_only_checkpoint_full_hypernetwork_training" in stdout
    assert "--phys_context_source raw_input_side_formula_gain" in stdout
    assert "--phys_formula_source raw_input_side_formula_gain" in stdout
    assert "--hyper_phys_delta_scale 0.03" in stdout
    assert "--hyper_phys_gate_init 0.25" in stdout
    assert "--hyper_phys_gain_basis_residual 0" in stdout
    assert "--hyper_phys_consistency_regularization_weight 0.0" in stdout
    assert f"--init_from_source_base_checkpoint {fake_source}" in stdout
    assert "--init_from_prompt_checkpoint" not in stdout


def test_m3_16_protocol_docs_register_source_only_invariant():
    combined = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in [
            "context/01_RESEARCH_CONTRACT.md",
            "docs/COAUTHOR_CONTEXT.md",
            "tasks/phase5_hyperda_safe_zero_few_shot.md",
            "specs/hyperda_v4.yaml",
        ]
    )

    assert "M3_16_source_only_phys_m3trust_lite" in combined
    assert "Stage 2 invariant" in combined
    assert "source_pooled_global_backbone" in combined
    assert "M3_15_m31_anchored_source_safe_phys_coeff_delta" in combined
    assert "m3_1_warm_start_diagnostic_not_active_stage2_physics_mainline" in combined
    assert "final_output_residual_allowed: false" in combined
