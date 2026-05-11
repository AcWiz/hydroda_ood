"""Test increment normalization: stats computed from source_train only.

No-leakage declaration:
    - Increment stats computed from source_train split only
    - No target_query labels used in test
"""
from __future__ import annotations

import numpy as np
import pytest

from hydroda.models.resunet import SmallResUNet
from hydroda.training.trainer import Trainer, _compute_channel_stats


class DummyDataset:
    """Minimal dataset for testing increment stats computation."""

    def __init__(self, n_samples, inc_s_mean, inc_s_std, inc_r_mean, inc_r_std):
        self.n_samples = n_samples
        self._inc_s_mean = inc_s_mean
        self._inc_s_std = inc_s_std
        self._inc_r_mean = inc_r_mean
        self._inc_r_std = inc_r_std

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        inc_s = rng.randn(64, 64) * self._inc_s_std + self._inc_s_mean
        inc_r = rng.randn(64, 64) * self._inc_r_std + self._inc_r_mean
        return {
            "x": rng.randn(12, 64, 64).astype(np.float32),
            "increment_surface": inc_s.astype(np.float32),
            "increment_rootzone": inc_r.astype(np.float32),
            "loss_mask": np.ones((64, 64), dtype=np.float32),
            "date_str": f"2015-{idx % 12 + 1:02d}-01",
        }


def test_increment_stats_match_source_distribution():
    """Verify increment stats accurately capture source_train distribution."""
    inc_s_mean = 0.05
    inc_s_std = 0.15
    inc_r_mean = -0.03
    inc_r_std = 0.10

    dataset = DummyDataset(n_samples=100, inc_s_mean=inc_s_mean, inc_s_std=inc_s_std,
                            inc_r_mean=inc_r_mean, inc_r_std=inc_r_std)

    model = SmallResUNet(in_channels=12, out_channels=2, width=8)
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        target_increment_normalization=True,
        max_epochs=1,
        batch_size=4,
    )

    # Trainer should have computed increment stats
    assert trainer._inc_mean is not None, "Increment mean should be computed"
    assert trainer._inc_std is not None, "Increment std should be computed"

    # Stats should be close to true distribution params (with some sampling noise)
    assert abs(trainer._inc_mean[0] - inc_s_mean) < 0.01, f"Surface mean off: {trainer._inc_mean[0]}"
    assert abs(trainer._inc_mean[1] - inc_r_mean) < 0.01, f"Rootzone mean off: {trainer._inc_mean[1]}"
    # Std should be within 20% (with 100 samples, some noise is expected)
    assert abs(trainer._inc_std[0] - inc_s_std) / inc_s_std < 0.2, f"Surface std off: {trainer._inc_std[0]}"
    assert abs(trainer._inc_std[1] - inc_r_std) / inc_r_std < 0.2, f"Rootzone std off: {trainer._inc_std[1]}"


def test_increment_stats_not_used_when_disabled():
    """When target_increment_normalization=False, increment stats should not be computed."""
    dataset = DummyDataset(n_samples=50, inc_s_mean=0.05, inc_s_std=0.15,
                            inc_r_mean=-0.03, inc_r_std=0.10)
    model = SmallResUNet(in_channels=12, out_channels=2, width=8)

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        target_increment_normalization=False,
        max_epochs=1,
        batch_size=4,
    )

    assert trainer._inc_mean is None, "inc_mean should be None when normalization disabled"
    assert trainer._inc_std is None, "inc_std should be None when normalization disabled"


def test_increment_normalization_in_checkpoint():
    """Verify increment stats are saved to and loaded from checkpoint config."""
    dataset = DummyDataset(n_samples=50, inc_s_mean=0.05, inc_s_std=0.15,
                            inc_r_mean=-0.03, inc_r_std=0.10)
    model = SmallResUNet(in_channels=12, out_channels=2, width=8)

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        target_increment_normalization=True,
        max_epochs=1,
        batch_size=4,
        checkpoint_dir="/tmp/test_ckpt_incr",
    )

    # Save checkpoint
    import torch
    ckpt_path = "/tmp/test_ckpt_incr/test.pt"
    trainer.save_checkpoint(trainer.checkpoint_dir / "test.pt", epoch=0, loss=1.0, tag="test")

    # Load and verify
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = loaded["config"]

    assert cfg.get("target_increment_normalization") is True, "target_increment_normalization should be True"
    assert cfg.get("inc_mean") is not None, "inc_mean should be saved"
    assert cfg.get("inc_std") is not None, "inc_std should be saved"
    assert len(cfg["inc_mean"]) == 2, "inc_mean should have 2 elements"
    assert len(cfg["inc_std"]) == 2, "inc_std should have 2 elements"

    # Verify round-trip: loading predictor would use inc_mean/inc_std to denormalize
    # Simulate what SourceOnlyBackbonePredictor does
    inc_mean = np.array(cfg["inc_mean"], dtype=np.float32)
    inc_std = np.array(cfg["inc_std"], dtype=np.float32)

    # Simulate model output (normalized) being denormalized
    pred_norm_s = 0.5  # some normalized output
    pred_raw_s = pred_norm_s * inc_std[0] + inc_mean[0]
    assert np.isfinite(pred_raw_s), "denormalized output should be finite"


def test_checkpoint_round_trip_zero_raw_init():
    """Verify zero_raw_increment_init setting is saved and loaded from checkpoint."""
    dataset = DummyDataset(n_samples=50, inc_s_mean=0.05, inc_s_std=0.15,
                            inc_r_mean=-0.03, inc_r_std=0.10)
    model = SmallResUNet(in_channels=12, out_channels=2, width=8)

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        target_increment_normalization=True,
        zero_raw_increment_init=True,
        max_epochs=1,
        batch_size=4,
        checkpoint_dir="/tmp/test_ckpt_zri",
    )

    # Verify zero_raw_increment_init is set
    assert trainer.zero_raw_increment_init is True

    # Save checkpoint
    import torch
    ckpt_path = "/tmp/test_ckpt_zri/test.pt"
    trainer.save_checkpoint(trainer.checkpoint_dir / "test.pt", epoch=0, loss=1.0, tag="test")

    # Load and verify config
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = loaded["config"]

    assert cfg.get("zero_raw_increment_init") is True, "zero_raw_increment_init should be saved"
    assert cfg.get("inc_mean") is not None, "inc_mean should be saved with zero_raw_init"
    assert cfg.get("inc_std") is not None, "inc_std should be saved with zero_raw_init"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))