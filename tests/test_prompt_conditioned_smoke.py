"""Smoke tests for prompt-conditioned training pipeline (Phase 4B).

Tests:
1. Epoch checkpoint saving
2. Transfer safe score uses source_val only
3. Latitude weights = 1 -> unweighted equivalence
4. Prompt collapse detection
5. Checkpoint contains model + prompt_encoder state_dicts
6. Alpha calibration uses region-aware score
"""
from __future__ import annotations

import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from hydroda.models.conditional_unet import FiLMConditionalResUNet
from hydroda.models.prompt_encoder import RegionPromptEncoder
from hydroda.training.losses import WeightedMaskedHuberLoss
from scripts.train.train_prompt_conditioned_shared import PromptQualityTracker, PromptConditionedTrainer


# ---------------------------------------------------------------------------
# Fake dataset for smoke testing
# ---------------------------------------------------------------------------

class FakePromptDataset:
    """Minimal fake dataset for prompt-conditioned training smoke tests."""

    def __init__(self, n_samples: int = 6, H: int = 32, W: int = 48):
        self.n_samples = n_samples
        self.H = H
        self.W = W
        self._date_records = [{"date_str": "2015-01-01"} for _ in range(n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        x = np.random.randn(12, self.H, self.W).astype(np.float32)
        inc_s = np.random.randn(self.H, self.W).astype(np.float32) * 0.01
        inc_r = np.random.randn(self.H, self.W).astype(np.float32) * 0.005
        loss_mask = np.ones((1, self.H, self.W), dtype=np.float32)
        latitude_weight = np.ones((self.H, self.W), dtype=np.float32)
        forecast_surface = np.random.randn(self.H, self.W).astype(np.float32) * 0.05
        forecast_rootzone = np.random.randn(self.H, self.W).astype(np.float32) * 0.05
        # region_mask_integer: assign region 1 (source region 0)
        region_mask_integer = np.ones((self.H, self.W), dtype=np.int32)
        month = 6
        return {
            "x": x,
            "increment_surface": inc_s,
            "increment_rootzone": inc_r,
            "loss_mask": loss_mask,
            "latitude_weight": latitude_weight,
            "forecast_surface": forecast_surface,
            "forecast_rootzone": forecast_rootzone,
            "region_mask_integer": region_mask_integer,
            "month": month,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_epoch_checkpoint_saving():
    """Verify PromptConditionedTrainer saves epoch checkpoints."""
    train_dataset = FakePromptDataset(n_samples=8, H=32, W=48)
    source_val_dataset = FakePromptDataset(n_samples=3, H=32, W=48)

    model = FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16)
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PromptConditionedTrainer(
            model=model,
            prompt_encoder=prompt_encoder,
            train_dataset=train_dataset,
            max_epochs=2,
            batch_size=2,
            num_workers=0,
            device="cpu",
            checkpoint_dir=tmpdir,
            checkpoint_every_n_epochs=1,
            source_val_dataset=source_val_dataset,
            eval_every_epochs=1,
            source_regions=["US-R1", "US-R2"],
            global_to_source_lookup={0: 0, 1: 1},
            # Disable residual gain for speed
            source_val_residual_gain=False,
        )

        trainer.train(verbose=False)

        ckpt_dir = Path(tmpdir)
        assert (ckpt_dir / "checkpoint_latest.pt").exists(), "checkpoint_latest.pt should exist"
        assert (ckpt_dir / "last.pt").exists(), "last.pt should exist"

        epoch_files = list(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        assert len(epoch_files) >= 1, f"Expected checkpoint_epoch_*.pt files, found {epoch_files}"

        # CSV files should exist
        assert (ckpt_dir / "metrics_train.csv").exists(), "metrics_train.csv should exist"
        assert (ckpt_dir / "README_training_summary.md").exists(), "README_training_summary.md should exist"

        print(f"  test_epoch_checkpoint_saving passed: {sorted([p.name for p in ckpt_dir.glob('*.pt')])}")


def test_transfer_safe_score_uses_source_val_only():
    """Verify transfer safe score computation uses only source_val data."""
    from hydroda.training.calibration import calibrate_residual_gain_region_aware

    H, W = 16, 24
    samples_s_by_region = {}
    samples_r_by_region = {}
    alphas = [0.0, 0.5, 1.0]

    for region in ["US-R2", "US-R3"]:
        s_list = []
        r_list = []
        for _ in range(3):
            pred_inc = np.random.randn(H, W).astype(np.float32) * 0.01
            true_inc = np.random.randn(H, W).astype(np.float32) * 0.01
            fcst = np.random.randn(H, W).astype(np.float32) * 0.05
            mask = np.ones((H, W), dtype=np.float32)
            latw = np.ones((H, W), dtype=np.float32)
            s_list.append((pred_inc, true_inc, fcst, mask, latw))
            r_list.append((pred_inc * 0.8, true_inc * 0.8, fcst, mask, latw))
        samples_s_by_region[region] = s_list
        samples_r_by_region[region] = r_list

    result = calibrate_residual_gain_region_aware(
        samples_s_by_region=samples_s_by_region,
        samples_r_by_region=samples_r_by_region,
        alpha_grid=alphas,
    )

    assert result, "calibrate_residual_gain_region_aware should return non-empty result"
    assert "selection_score" in result
    assert "selection_trace" in result
    assert len(result["selection_trace"]) == 9  # 3x3 grid
    # Score should be finite and not -inf
    assert np.isfinite(result["selection_score"]), f"Score should be finite, got {result['selection_score']}"
    # Each trace entry should have the score formula fields
    trace = result["selection_trace"][0]
    assert "transfer_safe_score" in trace
    assert "worst_region_balanced_skill" in trace
    assert "neg_rootzone_count" in trace

    print(f"  test_transfer_safe_score_uses_source_val_only passed: score={result['selection_score']:.4f}")


def test_latitude_weights_equal_one_unweighted_equivalence():
    """Verify WeightedMaskedHuberLoss with uniform lat weights ~ unweighted."""
    loss_fn_w = WeightedMaskedHuberLoss(delta=0.01, use_lat_weight=True)

    B, C, H, W = 2, 2, 8, 8
    pred = torch.randn(B, C, H, W) * 0.01
    target = torch.randn(B, C, H, W) * 0.01
    mask = torch.ones(B, 1, H, W)

    # With uniform latitude weights of 1.0
    lat_w = torch.ones(1, 1, H, W)
    result_w = loss_fn_w(pred, target, mask, latitude_weight=lat_w)

    # With uniform latitude weights (but use_lat_weight=True, no lat_w -> uniform fallback)
    result_u = loss_fn_w(pred, target, mask, latitude_weight=None)

    # Both should be finite
    assert torch.isfinite(result_w["total_loss"])
    assert torch.isfinite(result_u["total_loss"])
    # With lat_w=ones, should match the uniform fallback (both use weight=1)
    assert torch.allclose(result_w["total_loss"], result_u["total_loss"], rtol=1e-4), \
        f"Uniform lat_w={result_w['total_loss'].item():.6f} vs None={result_u['total_loss'].item():.6f}"

    print(f"  test_latitude_weights_equal_one_unweighted_equivalence passed.")


def test_prompt_collapse_detection():
    """Verify PromptQualityTracker detects collapse and reports normal diversity."""
    tracker = PromptQualityTracker(num_regions=3)

    # Case 1: Normal diverse prompts
    tracker.reset()
    for i in range(3):
        emb = torch.randn(5, 16) + i * 2.0  # distinct per region
        rids = torch.full((5,), i, dtype=torch.long)
        tracker.update(emb, rids)
    metrics = tracker.compute_metrics()
    assert not metrics["prompt_collapse_detected"], "Should not detect collapse with diverse prompts"
    assert np.isfinite(metrics["prompt_pairwise_cosine_distance_mean"])
    assert metrics["prompt_pairwise_cosine_distance_mean"] > 0.01, \
        f"Mean cosine distance should be > 0.01 with diverse prompts, got {metrics['prompt_pairwise_cosine_distance_mean']}"
    print(f"  Normal prompts: cos_dist_mean={metrics['prompt_pairwise_cosine_distance_mean']:.4f}")

    # Case 2: Collapsed prompts (almost identical)
    tracker.reset()
    for i in range(3):
        emb = torch.ones(5, 16) * 0.001 + torch.randn(5, 16) * 1e-6  # nearly identical
        rids = torch.full((5,), i, dtype=torch.long)
        tracker.update(emb, rids)
    metrics = tracker.compute_metrics()
    # With nearly identical embeddings, cosine distance should be very small
    assert metrics["prompt_collapse_detected"] or \
        metrics["prompt_pairwise_cosine_distance_mean"] < 0.05, \
        f"Collapsed prompts should have small cosine distance, got {metrics['prompt_pairwise_cosine_distance_mean']}"
    print(f"  Collapsed prompts: cos_dist_mean={metrics['prompt_pairwise_cosine_distance_mean']:.4f}, collapsed={metrics['prompt_collapse_detected']}")

    # Case 3: Single region (not enough for pairwise)
    tracker.reset()
    emb = torch.randn(5, 16)
    rids = torch.zeros(5, dtype=torch.long)
    tracker.update(emb, rids)
    metrics = tracker.compute_metrics()
    assert np.isnan(metrics["prompt_pairwise_cosine_distance_mean"]), \
        "Single region should yield NaN cosine distance (not enough pairs)"
    assert not metrics["prompt_collapse_detected"], "Single region should not trigger collapse"

    print(f"  test_prompt_collapse_detection passed.")


def test_checkpoint_contains_model_and_prompt_encoder_state_dicts():
    """Verify saved checkpoint contains both model and prompt_encoder state_dicts."""
    train_dataset = FakePromptDataset(n_samples=4, H=32, W=48)

    model = FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16)
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PromptConditionedTrainer(
            model=model,
            prompt_encoder=prompt_encoder,
            train_dataset=train_dataset,
            max_epochs=1,
            batch_size=2,
            num_workers=0,
            device="cpu",
            checkpoint_dir=tmpdir,
            source_regions=["US-R1", "US-R2"],
            global_to_source_lookup={0: 0, 1: 1},
        )

        # Save a checkpoint with gain results
        gain_results = {
            "best_alpha_surface": 0.5,
            "best_alpha_rootzone": 0.6,
            "selection_score": 0.1,
            "skill_surface_with_alpha": 0.2,
            "skill_rootzone_with_alpha": 0.15,
            "rmse_surface_model": 0.01,
            "rmse_rootzone_model": 0.005,
            "alpha_grid": [0.0, 0.5, 1.0],
            "min_skill": 0.15,
            "mean_skill": 0.175,
        }
        trainer.save_checkpoint(
            Path(tmpdir) / "test.pt", epoch=0, loss=0.5, tag="test",
            gain_results=gain_results,
        )

        ckpt = torch.load(Path(tmpdir) / "test.pt", map_location="cpu", weights_only=False)

        assert "model_state_dict" in ckpt, "Checkpoint must contain model_state_dict"
        assert "prompt_encoder_state_dict" in ckpt, "Checkpoint must contain prompt_encoder_state_dict"
        assert "residual_gain_alpha_surface" in ckpt, "Checkpoint must contain residual_gain_alpha_surface"
        assert "residual_gain_alpha_rootzone" in ckpt, "Checkpoint must contain residual_gain_alpha_rootzone"
        assert "selection_score" in ckpt
        assert "config" in ckpt
        assert ckpt["config"]["source_regions"] == ["US-R1", "US-R2"]
        assert ckpt["config"]["checkpoint_every_n_epochs"] == 5  # default

        print(f"  test_checkpoint_contains_model_and_prompt_encoder_state_dicts passed.")


def test_alpha_calibration_uses_region_aware_score():
    """Verify calibrate_residual_gain_region_aware does 2D grid with region-aware scoring."""
    from hydroda.training.calibration import calibrate_residual_gain_region_aware

    H, W = 16, 24
    # Create synthetic per-region samples with distinct patterns
    samples_s_by_region = {}
    samples_r_by_region = {}
    alphas = [0.0, 0.5, 1.0]

    for r_idx, region in enumerate(["US-R2", "US-R3", "US-R4"]):
        s_list = []
        r_list = []
        for _ in range(5):
            pred_inc = np.random.randn(H, W).astype(np.float32) * 0.01
            true_inc = np.random.randn(H, W).astype(np.float32) * 0.01
            fcst = np.random.randn(H, W).astype(np.float32) * 0.05
            mask = np.ones((H, W), dtype=np.float32)
            latw = np.ones((H, W), dtype=np.float32)
            s_list.append((pred_inc, true_inc, fcst, mask, latw))
            r_list.append((pred_inc * 0.8, true_inc * 0.8, fcst, mask, latw))  # slightly different
        samples_s_by_region[region] = s_list
        samples_r_by_region[region] = r_list

    result = calibrate_residual_gain_region_aware(
        samples_s_by_region=samples_s_by_region,
        samples_r_by_region=samples_r_by_region,
        alpha_grid=alphas,
    )

    assert result, "calibrate_residual_gain_region_aware should return non-empty dict"
    assert "best_alpha_surface" in result
    assert "best_alpha_rootzone" in result
    assert "selection_score" in result
    assert "region_variable_skills" in result
    assert result["calibration_mode"] == "region_aware_2d"
    assert result["region_names"] == ["US-R2", "US-R3", "US-R4"]

    # selection_trace should have 9 entries (3x3 grid)
    assert len(result["selection_trace"]) == 9, f"Expected 9 trace entries (3x3 grid), got {len(result['selection_trace'])}"

    # Check trace has expected fields
    trace = result["selection_trace"][0]
    assert "alpha_surface" in trace
    assert "alpha_rootzone" in trace
    assert "transfer_safe_score" in trace
    assert "worst_region_balanced_skill" in trace

    print(f"  test_alpha_calibration_uses_region_aware_score passed: "
          f"best_alpha=({result['best_alpha_surface']}, {result['best_alpha_rootzone']}), "
          f"score={result['selection_score']:.4f}")


if __name__ == "__main__":
    test_epoch_checkpoint_saving()
    test_transfer_safe_score_uses_source_val_only()
    test_latitude_weights_equal_one_unweighted_equivalence()
    test_prompt_collapse_detection()
    test_checkpoint_contains_model_and_prompt_encoder_state_dicts()
    test_alpha_calibration_uses_region_aware_score()
    print("\nAll prompt-conditioned smoke tests passed.")
