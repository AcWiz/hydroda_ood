import os
import subprocess
from pathlib import Path


def test_source_only_inference_accepts_checkpoint_as_first_argument(tmp_path):
    checkpoint = tmp_path / "phase4_source_only_source_only_US-R1" / "checkpoints" / "checkpoint_best_source_val_safe_score.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"stub checkpoint")

    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    result = subprocess.run(
        ["bash", "run/phase4_source_only_inference.sh", str(checkpoint)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert f"checkpoint={checkpoint}" in result.stdout
    assert "target_region=US-R1" in result.stdout
    assert "DRY_RUN=1" in result.stdout


def test_phase4_dg_inference_dry_run_finds_swad_and_mixstyle_checkpoints():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    result = subprocess.run(
        [
            "bash",
            "run/phase4_dg_baselines_inference_us_r1_seed0.sh",
            "--method-list",
            "swad",
            "mixstyle",
            "--target-region",
            "US-R1",
            "--seed",
            "0",
            "--cuda-device",
            "0",
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "DRY_RUN=1" in result.stdout
    assert "method_id=swad_source_pooled_global_backbone" in result.stdout
    assert "checkpoint_swad.pt" in result.stdout
    assert "method_id=mixstyle_source_pooled_global_backbone" in result.stdout
    assert "checkpoint_best_source_val_safe_score.pt" in result.stdout
    assert "--split_type source_test" in result.stdout
    assert "--split_type target_eval" in result.stdout
    assert "scripts/eval/evaluate_checkpoint.py" in result.stdout


def test_phase4_dg_inference_dry_run_uses_best_checkpoint_for_iu_source_val_loss_run():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    result = subprocess.run(
        [
            "bash",
            "run/phase4_dg_baselines_inference_us_r1_seed0.sh",
            "--method-list",
            "iu",
            "--splits",
            "target_eval",
            "--target-region",
            "US-R1",
            "--seed",
            "0",
            "--cuda-device",
            "0",
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "method_id=identify_unlearn_source_domain_gradient_ascent" in result.stdout
    assert "checkpoints/best.pt" in result.stdout
    assert "checkpoint_best_source_val_safe_score.pt" not in result.stdout


def test_phase4_dg_inference_dry_run_uses_best_checkpoint_for_udim_run():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    result = subprocess.run(
        [
            "bash",
            "run/phase4_dg_baselines_inference_us_r1_seed0.sh",
            "--method-list",
            "udim",
            "--splits",
            "target_eval",
            "--target-region",
            "US-R1",
            "--seed",
            "0",
            "--cuda-device",
            "0",
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "method_id=udim_unknown_domain_inconsistency_minimization" in result.stdout
    assert "checkpoints/best.pt" in result.stdout
    assert "--split_type target_eval" in result.stdout
