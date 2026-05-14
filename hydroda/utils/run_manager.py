"""RunManager — Unified run directory management for HydroDA-OOD / HyperDA V4.

Auto-generates run_name: {phase}_{method}_{target_region}_{width}_{epochs}_{lr}_{norm}_{zero_raw}_{seed}_{timestamp}
Creates:
  artifacts/runs/{phase}/{run_name}/
    config.yaml
    config_resolved.yaml
    environment.json
    git_info.json
    protocol.json
    data_manifest.json
    logs_{timestamp}/ (train_steps.jsonl, train_epochs.jsonl, eval_metrics.jsonl, console.log)
    checkpoints/
    results/
    reports/
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from hydroda.utils.runtime import get_git_hash, get_git_status, get_timestamp


class RunManager:
    """Unified run directory manager.

    Args:
        phase: Phase name (e.g. "phase4_source_only")
        method: Method name (e.g. "source_only", "hyperda_zero")
        target_region: Target region (e.g. "US-R1")
        config: Full config dict to save
        output_dir: Override output directory (default: artifacts/runs/{phase}/{run_name})
        run_name: Override auto-generated run_name
        width: Model width
        epochs: Max epochs
        lr: Learning rate
        norm: Normalization string
        zero_raw: Zero-raw init flag
        seed: Seed
        timestamp: Timestamp override (default: now)
    """

    def __init__(
        self,
        phase: str,
        method: str,
        target_region: str,
        config: dict,
        output_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        width: Optional[int] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        norm: Optional[str] = None,
        zero_raw: Optional[bool] = None,
        seed: Optional[int] = None,
        timestamp: Optional[str] = None,
    ):
        self.phase = phase
        self.method = method
        self.target_region = target_region
        self.config = config

        # Build run_name components
        parts = [phase, method, target_region]
        if width is not None:
            parts.append(f"w{width}")
        if epochs is not None:
            parts.append(f"e{epochs}")
        if lr is not None:
            parts.append(f"lr{lr}")
        if norm is not None:
            parts.append(norm)
        if zero_raw is not None:
            parts.append("zero" if zero_raw else "nozero")
        if seed is not None:
            parts.append(f"s{seed}")

        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(ts)

        self._run_name = run_name or "_".join(parts)
        if output_dir:
            self._base_dir = Path(output_dir)
        else:
            self._base_dir = Path("artifacts/runs") / phase / self._run_name

        # Create all subdirectories (logs subdir includes timestamp)
        self._log_dir = self._base_dir / f"logs_{ts}"
        self._checkpoint_dir = self._base_dir / "checkpoints"
        self._results_dir = self._base_dir / "results"
        self._reports_dir = self._base_dir / "reports"

        for d in [self._log_dir, self._checkpoint_dir, self._results_dir, self._reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._console_log_path = self._log_dir / "console.log"
        self._console_log_file: Optional[Any] = None

    def get_run_name(self) -> str:
        return self._run_name

    def get_run_dir(self) -> Path:
        return self._base_dir

    def get_checkpoint_dir(self) -> Path:
        return self._checkpoint_dir

    def get_results_dir(self) -> Path:
        return self._results_dir

    def get_log_dir(self) -> Path:
        return self._log_dir

    def get_reports_dir(self) -> Path:
        return self._reports_dir

    def save_config(self, config: dict, name: str = "config.yaml") -> None:
        """Save config YAML to run directory."""
        path = self._base_dir / name
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def save_environment_info(self, env_info: dict) -> None:
        """Save environment info JSON."""
        path = self._base_dir / "environment.json"
        with open(path, "w") as f:
            json.dump(env_info, f, indent=2)

    def save_git_info(self) -> None:
        """Save git info JSON."""
        git_hash = get_git_hash()
        git_status = get_git_status()
        git_info = {
            "git_hash": git_hash,
            "git_status": git_status,
            "timestamp": get_timestamp(),
        }
        path = self._base_dir / "git_info.json"
        with open(path, "w") as f:
            json.dump(git_info, f, indent=2)

    def save_protocol(self, protocol_info: dict) -> None:
        """Save protocol freeze info JSON."""
        path = self._base_dir / "protocol.json"
        with open(path, "w") as f:
            json.dump(protocol_info, f, indent=2)

    def save_data_manifest(self, manifest: dict) -> None:
        """Save data manifest JSON."""
        path = self._base_dir / "data_manifest.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    def open_console_log(self) -> None:
        """Open console.log for append writing."""
        self._console_log_file = open(self._console_log_path, "a")

    def close_console_log(self) -> None:
        """Close console.log."""
        if self._console_log_file is not None:
            self._console_log_file.close()
            self._console_log_file = None

    def log_console(self, message: str) -> None:
        """Write message to console.log with timestamp."""
        ts = get_timestamp()
        line = f"[{ts}] {message}\n"
        if self._console_log_file is not None:
            self._console_log_file.write(line)
            self._console_log_file.flush()
        # Also print to stdout
        print(message, flush=True)

    def summary_json_path(self) -> Path:
        return self._reports_dir / "summary.json"

    def checkpoint_best_path(self) -> Path:
        return self._checkpoint_dir / "best.pt"

    def checkpoint_last_path(self) -> Path:
        return self._checkpoint_dir / "last.pt"
