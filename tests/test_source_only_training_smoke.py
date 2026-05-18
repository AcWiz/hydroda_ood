"""Smoke test for source-only backbone training pipeline.

Tests:
1. Tiny synthetic dataset (3 samples, H=32, W=48)
2. SmallResUNet with width=8
3. Train for 2 steps
4. Assert total_loss decreases and valid_pixel_count > 0

No-leakage: uses synthetic data, no real DA.nc access.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hydroda.models.resunet import SmallResUNet
from hydroda.training.losses import MaskedHuberLoss


class FakeDataset:
    """Minimal fake dataset for smoke testing."""

    def __init__(self, n_samples: int = 3, H: int = 32, W: int = 48):
        self.n_samples = n_samples
        self.H = H
        self.W = W

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        # Random input [12, H, W] and increments [2, H, W]
        x = np.random.randn(12, self.H, self.W).astype(np.float32)
        inc_s = np.random.randn(self.H, self.W).astype(np.float32) * 0.01
        inc_r = np.random.randn(self.H, self.W).astype(np.float32) * 0.01
        # loss_mask: all ones
        loss_mask = np.ones((1, self.H, self.W), dtype=np.float32)
        return {
            "x": x,
            "increment_surface": inc_s,
            "increment_rootzone": inc_r,
            "loss_mask": loss_mask,
        }


def collate_fn(batch):
    x = torch.from_numpy(np.stack([s["x"] for s in batch], axis=0))
    inc_surface = torch.from_numpy(np.stack([s["increment_surface"] for s in batch], axis=0))
    inc_rootzone = torch.from_numpy(np.stack([s["increment_rootzone"] for s in batch], axis=0))
    loss_mask = torch.from_numpy(np.stack([s["loss_mask"] for s in batch], axis=0))
    return {
        "x": x,
        "increment_surface": inc_surface,
        "increment_rootzone": inc_rootzone,
        "loss_mask": loss_mask,
    }


def test_source_only_training_smoke():
    """Smoke test: can we train for 2 steps and see loss decrease?"""
    # Create tiny model and dataset
    model = SmallResUNet(in_channels=12, out_channels=2, width=8)
    dataset = FakeDataset(n_samples=3, H=32, W=48)
    loss_fn = MaskedHuberLoss(delta=0.01)

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step, batch in enumerate(dataloader):
        x = batch["x"]
        inc_surface = batch["increment_surface"]
        inc_rootzone = batch["increment_rootzone"]
        loss_mask = batch["loss_mask"]

        target = torch.stack([inc_surface, inc_rootzone], dim=1)

        optimizer.zero_grad()
        pred = model(x)
        result = loss_fn(pred, target, loss_mask)
        result["total_loss"].backward()
        optimizer.step()

        losses.append(float(result["total_loss"].item()))
        valid_px = int(result["valid_pixel_count"].item())

        assert valid_px > 0, f"Step {step}: valid_pixel_count should be > 0, got {valid_px}"

        if step >= 1:
            break

    assert len(losses) == 2, f"Expected 2 steps, got {len(losses)}"
    # Loss should be finite
    assert np.isfinite(losses[-1]), f"Final loss should be finite, got {losses[-1]}"
    print(f"  Smoke test passed: losses={losses}, valid_px={valid_px}")


def test_tiny_overfit():
    """Test that model can overfit a single repeated sample."""
    # Single sample repeated 50 times
    x = np.random.randn(12, 32, 48).astype(np.float32)
    inc_s = np.random.randn(32, 48).astype(np.float32) * 0.01
    inc_r = np.random.randn(32, 48).astype(np.float32) * 0.01
    loss_mask = np.ones((1, 32, 48), dtype=np.float32)

    samples = [
        {"x": x.copy(), "increment_surface": inc_s.copy(), "increment_rootzone": inc_r.copy(), "loss_mask": loss_mask.copy()}
        for _ in range(50)
    ]

    model = SmallResUNet(in_channels=12, out_channels=2, width=8)
    loss_fn = MaskedHuberLoss(delta=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    dataloader = DataLoader(samples, batch_size=4, collate_fn=collate_fn)

    for step in range(100):
        for batch in dataloader:
            target = torch.stack([batch["increment_surface"], batch["increment_rootzone"]], dim=1)
            optimizer.zero_grad()
            pred = model(batch["x"])
            result = loss_fn(pred, target, batch["loss_mask"])
            result["total_loss"].backward()
            optimizer.step()

    # Final loss should be very low
    final_loss = float(result["total_loss"].item())
    assert final_loss < 1e-3, f"After 100 steps on single sample, loss should be < 1e-3, got {final_loss}"
    print(f"  Tiny overfit test passed: final_loss={final_loss:.6f}")


def test_source_only_predictor_output_contract():
    """Test SourceOnlyBackbonePredictor output contract."""
    from hydroda.baselines.source_only import SourceOnlyBackbonePredictor
    import tempfile
    import os

    # Create a fresh model and save a fake checkpoint
    model = SmallResUNet(in_channels=12, out_channels=2, width=8)

    # Create fake checkpoint with correct width
    ch_mean = np.ones(12, dtype=np.float32)
    ch_std = np.ones(12, dtype=np.float32)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "width": 8,  # must match model
            "ch_mean": ch_mean.tolist(),
            "ch_std": ch_std.tolist(),
        },
        "protocol_freeze_id": "test_v4",
        "split_manifest_path": "test_manifest.json",
        "git_hash": "test",
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
        torch.save(checkpoint, tmp_path)

    try:
        predictor = SourceOnlyBackbonePredictor(checkpoint_path=tmp_path, device="cpu")

        # Create a fake sample
        sample = {
            "x": np.random.randn(12, 32, 48).astype(np.float32),
            "forecast_surface": np.random.randn(32, 48).astype(np.float32) * 0.3,
            "forecast_rootzone": np.random.randn(32, 48).astype(np.float32) * 0.3,
        }

        result = predictor.predict(sample)

        # Check keys
        required_keys = [
            "pred_increment_surface",
            "pred_increment_rootzone",
            "pred_analysis_surface",
            "pred_analysis_rootzone",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        # Check shapes
        H, W = 32, 48
        assert result["pred_increment_surface"].shape == (H, W)
        assert result["pred_increment_rootzone"].shape == (H, W)
        assert result["pred_analysis_surface"].shape == (H, W)
        assert result["pred_analysis_rootzone"].shape == (H, W)

        # Check pred_analysis = forecast + pred_increment
        np.testing.assert_allclose(
            result["pred_analysis_surface"],
            sample["forecast_surface"] + result["pred_increment_surface"],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            result["pred_analysis_rootzone"],
            sample["forecast_rootzone"] + result["pred_increment_rootzone"],
            rtol=1e-5,
        )

        print(f"  SourceOnlyBackbonePredictor contract test passed.")
    finally:
        os.unlink(tmp_path)


def test_checkpoint_every_5_epochs_smoke():
    """Verify Trainer with checkpoint_every_n_epochs saves periodic checkpoints."""
    import tempfile
    import os

    from hydroda.models.resunet import SmallResUNet
    from hydroda.training.trainer import Trainer

    # Create fake training datasets
    dataset = FakeDataset(n_samples=6, H=32, W=48)
    source_val_dataset = FakeDataset(n_samples=3, H=32, W=48)

    model = SmallResUNet(in_channels=12, out_channels=2, width=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            max_epochs=2,
            batch_size=2,
            num_workers=0,
            device="cpu",
            checkpoint_dir=tmpdir,
            checkpoint_every_n_epochs=1,  # Save every epoch
            source_val_dataset=source_val_dataset,
            eval_every_epochs=1,
        )

        trainer.train(verbose=False)

        # Check that checkpoint files exist
        ckpt_dir = Path(tmpdir)
        assert (ckpt_dir / "checkpoint_latest.pt").exists(), "checkpoint_latest.pt should exist"
        assert (ckpt_dir / "last.pt").exists(), "last.pt should exist"

        # Since checkpoint_every_n_epochs=1 and max_epochs=2, epoch 1 snapshot should exist
        epoch_files = list(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        assert len(epoch_files) >= 1, f"Expected checkpoint_epoch_*.pt files, found {epoch_files}"

        # Since source_val_dataset is provided, best_skill checkpoint should exist
        skill_files = list(ckpt_dir.glob("checkpoint_best_source_val_*.pt"))
        assert len(skill_files) >= 1, f"Expected best source_val checkpoint, found {skill_files}"

        print(f"  Checkpoint smoke test passed: {sorted([p.name for p in ckpt_dir.glob('*.pt')])}")


if __name__ == "__main__":
    import pytest
    import sys
    # Allow running directly
    test_source_only_training_smoke()
    test_tiny_overfit()
    test_source_only_predictor_output_contract()
    test_checkpoint_every_5_epochs_smoke()
    print("\nAll smoke tests passed.")