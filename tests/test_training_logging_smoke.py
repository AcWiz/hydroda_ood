"""Smoke test for training logging infrastructure (RunManager + JSONLLogger).

This test verifies the Trainer can create JSONL logs and summary.json
without actually training a model. We use a minimal mock setup.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn


def test_run_manager_jsonl_smoke():
    """Test RunManager + JSONLLogger integration smoke test."""
    from hydroda.utils.run_manager import RunManager
    from hydroda.utils.logger import JSONLLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        rm = RunManager(
            phase="test_phase",
            method="test_method",
            target_region="US-R1",
            config={"test": True},
            output_dir=tmpdir,
        )

        # Create JSONL logger and write some data
        jsonl_logger = JSONLLogger(rm.get_log_dir())
        jsonl_logger.log_step({
            "epoch": 0, "step": 0, "total_loss": 1.0,
            "surface_loss": 0.5, "rootzone_loss": 0.5,
            "lr": 0.001,
        })
        jsonl_logger.log_epoch({
            "epoch": 0, "total_loss": 1.0,
            "surface_loss": 0.5, "rootzone_loss": 0.5,
            "lr": 0.001, "elapsed_s": 10.0,
        })

        # Verify file exists and has valid JSON per line
        step_path = rm.get_log_dir() / "train_steps.jsonl"
        assert step_path.exists()
        with open(step_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["epoch"] == 0
        assert data["step"] == 0

        epoch_path = rm.get_log_dir() / "train_epochs.jsonl"
        assert epoch_path.exists()
        with open(epoch_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["epoch"] == 0


def test_trainer_summary_json_fields():
    """Test Trainer.save_summary_json produces required fields."""
    from hydroda.training.trainer import Trainer
    from hydroda.utils.run_manager import RunManager
    import numpy as np

    # We need a minimal dataset - create a mock
    class MockDataset:
        def __init__(self, n=10):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            return {
                "x": np.random.randn(12, 64, 64).astype(np.float32),
                "increment_surface": np.random.randn(64, 64).astype(np.float32),
                "increment_rootzone": np.random.randn(64, 64).astype(np.float32),
                "loss_mask": np.ones((64, 64), dtype=np.float32),
            }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create run manager
        rm = RunManager(
            phase="test_phase",
            method="test_method",
            target_region="US-R1",
            config={"test": True},
            output_dir=tmpdir,
        )

        # Create trainer with mock dataset (trainer initializes but hasn't run train() yet)
        model = nn.Sequential(
            nn.Conv2d(12, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, 3, padding=1),
        )

        mock_ds = MockDataset(4)

        trainer = Trainer(
            model=model,
            train_dataset=mock_ds,
            lr=0.001,
            max_epochs=1,
            batch_size=2,
            num_workers=0,
            device="cpu",
            checkpoint_dir=str(tmpdir),
            run_manager=rm,
            log_every_steps=10,
        )

        # Check train_history exists and has structure (initialized, not yet trained)
        assert isinstance(trainer.train_history, list)
        # best_loss is inf before training starts, that's fine for the smoke test
        assert trainer.best_loss == float("inf")

        # Save summary
        summary_path = rm.summary_json_path()
        trainer.save_summary_json(summary_path)

        # Verify required fields
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["normalization_source"] == "source_fit_only"
        assert summary["early_stopping_source"] == "train_loss_only"
        assert summary["model_selection_source"] == "best_train_loss"
        assert summary["target_query_usage"] == "eval_only_no_early_stopping"
        assert summary["leakage_guard_status"] == "pass"
        assert "experiment_id" in summary
        assert "protocol_freeze_id" in summary
        assert "best_loss" in summary
        assert "git_hash" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])