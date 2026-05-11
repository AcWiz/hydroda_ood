"""Tests for hydroda/utils/run_manager.py"""
import json
import tempfile
from pathlib import Path

import pytest


def test_run_manager_creates_directories():
    """Test RunManager creates all required subdirectories."""
    from hydroda.utils.run_manager import RunManager

    config = {"test": "value"}
    with tempfile.TemporaryDirectory() as tmpdir:
        rm = RunManager(
            phase="test_phase",
            method="test_method",
            target_region="US-R1",
            config=config,
            output_dir=tmpdir,
            width=16,
            epochs=5,
            lr=0.001,
            norm="norm",
            zero_raw=True,
            seed=0,
        )

        base = rm.get_run_dir()
        assert base.exists()
        assert rm.get_log_dir().exists()
        assert rm.get_checkpoint_dir().exists()
        assert rm.get_results_dir().exists()
        assert rm.get_reports_dir().exists()


def test_run_manager_run_name_auto():
    """Test RunManager auto-generates run_name correctly."""
    from hydroda.utils.run_manager import RunManager

    config = {}
    rm = RunManager(
        phase="phase4",
        method="source_only",
        target_region="US-R1",
        config=config,
        width=16,
        epochs=5,
        lr=0.001,
        norm="norm",
        zero_raw=True,
        seed=42,
    )

    name = rm.get_run_name()
    assert "phase4" in name
    assert "source_only" in name
    assert "US-R1" in name
    assert "w16" in name
    assert "e5" in name
    assert "lr0.001" in name
    assert "norm" in name
    assert "zero" in name
    assert "s42" in name


def test_run_manager_run_name_override():
    """Test RunManager respects run_name override."""
    from hydroda.utils.run_manager import RunManager

    config = {}
    rm = RunManager(
        phase="phase4",
        method="source_only",
        target_region="US-R1",
        config=config,
        run_name="my_custom_run_name",
    )
    assert rm.get_run_name() == "my_custom_run_name"


def test_run_manager_save_config():
    """Test RunManager.save_config writes YAML file."""
    from hydroda.utils.run_manager import RunManager

    config = {"key": "value", "number": 42}
    with tempfile.TemporaryDirectory() as tmpdir:
        rm = RunManager(
            phase="test",
            method="test",
            target_region="US-R1",
            config=config,
            output_dir=tmpdir,
        )
        rm.save_config(config, "config.yaml")
        path = rm.get_run_dir() / "config.yaml"
        assert path.exists()


def test_run_manager_save_environment_info():
    """Test RunManager.save_environment_info writes JSON."""
    from hydroda.utils.run_manager import RunManager

    env_info = {"pytorch_version": "2.0", "gpu_count": 1}
    with tempfile.TemporaryDirectory() as tmpdir:
        rm = RunManager(
            phase="test",
            method="test",
            target_region="US-R1",
            config={},
            output_dir=tmpdir,
        )
        rm.save_environment_info(env_info)
        path = rm.get_run_dir() / "environment.json"
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["pytorch_version"] == "2.0"


def test_run_manager_save_git_info():
    """Test RunManager.save_git_info writes git info JSON."""
    from hydroda.utils.run_manager import RunManager

    with tempfile.TemporaryDirectory() as tmpdir:
        rm = RunManager(
            phase="test",
            method="test",
            target_region="US-R1",
            config={},
            output_dir=tmpdir,
        )
        rm.save_git_info()
        path = rm.get_run_dir() / "git_info.json"
        assert path.exists()
        with open(path) as f:
            git_info = json.load(f)
        assert "git_hash" in git_info


def test_run_manager_checkpoint_paths():
    """Test RunManager returns correct checkpoint paths."""
    from hydroda.utils.run_manager import RunManager

    with tempfile.TemporaryDirectory() as tmpdir:
        rm = RunManager(
            phase="test",
            method="test",
            target_region="US-R1",
            config={},
            output_dir=tmpdir,
        )
        assert str(rm.checkpoint_best_path()).endswith("best.pt")
        assert str(rm.checkpoint_last_path()).endswith("last.pt")
        assert rm.summary_json_path().name == "summary.json"