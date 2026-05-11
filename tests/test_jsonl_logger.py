"""Tests for hydroda/utils/logger.py"""
import json
import tempfile
from pathlib import Path

import pytest


def test_jsonl_logger_step():
    """Test JSONLLogger.log_step writes valid JSON per line."""
    from hydroda.utils.logger import JSONLLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = JSONLLogger(Path(tmpdir))
        logger.log_step({"step": 1, "loss": 0.5, "lr": 0.001})
        logger.log_step({"step": 2, "loss": 0.4, "lr": 0.001})

        step_path = Path(tmpdir) / "train_steps.jsonl"
        assert step_path.exists()

        with open(step_path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "step" in data


def test_jsonl_logger_epoch():
    """Test JSONLLogger.log_epoch writes valid JSON."""
    from hydroda.utils.logger import JSONLLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = JSONLLogger(Path(tmpdir))
        logger.log_epoch({"epoch": 0, "loss": 0.5})
        logger.log_epoch({"epoch": 1, "loss": 0.3})

        epoch_path = Path(tmpdir) / "train_epochs.jsonl"
        assert epoch_path.exists()

        with open(epoch_path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "epoch" in data
            assert "loss" in data


def test_jsonl_logger_eval():
    """Test JSONLLogger.log_eval writes valid JSON."""
    from hydroda.utils.logger import JSONLLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = JSONLLogger(Path(tmpdir))
        logger.log_eval({"rmse": 0.1, "skill": 0.8})

        eval_path = Path(tmpdir) / "eval_metrics.jsonl"
        assert eval_path.exists()

        with open(eval_path) as f:
            content = f.read()
        data = json.loads(content)
        assert data["rmse"] == 0.1


def test_jsonl_logger_appendable():
    """Test JSONLLogger produces appendable files (no premature close)."""
    from hydroda.utils.logger import JSONLLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = JSONLLogger(Path(tmpdir))
        logger.log_step({"step": 1})
        logger.log_step({"step": 2})
        # If close is called, appending should still work
        logger.close()
        logger.log_step({"step": 3})  # Should still work

        step_path = Path(tmpdir) / "train_steps.jsonl"
        with open(step_path) as f:
            lines = f.readlines()
        assert len(lines) == 3


def test_wandb_logger_disabled_mode():
    """Test WandbLogger with mode=disabled doesn't require network."""
    from hydroda.utils.logger import WandbLogger

    # Should not raise even without network
    logger = WandbLogger(mode="disabled", project="test-project")
    assert logger.enabled == False
    assert logger.run_id is None

    # log should be a no-op
    logger.log({"loss": 0.5})
    logger.log_step({"step": 1})
    logger.log_epoch({"epoch": 1})
    logger.log_eval({"rmse": 0.1})
    logger.finish()  # should not raise


def test_wandb_logger_offline_mode_no_network():
    """Test WandbLogger offline mode doesn't require network (wandb.init may still try)."""
    from hydroda.utils.logger import WandbLogger

    # In offline mode, wandb.init may still try to connect but should fallback
    # The logger should handle this gracefully
    logger = WandbLogger(mode="offline", project="test-project")
    # enabled depends on whether wandb installed and init succeeds
    # But it should not crash
    assert logger is not None


def test_console_logger_basic():
    """Test ConsoleLogger can be instantiated."""
    from hydroda.utils.logger import ConsoleLogger

    logger = ConsoleLogger(log_every_steps=10, max_epochs=5)
    assert logger.log_every_steps == 10
    assert logger.max_epochs == 5