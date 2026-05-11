"""Three-layer logging system for HydroDA-OOD / HyperDA V4 experiments.

Layers:
1. ConsoleLogger — readable output every N steps
2. JSONLLogger — machine-readable append-only logs
3. WandbLogger — optional wandb integration (disabled by default)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class ConsoleLogger:
    """Readable console output every `log_every_steps` steps.

    Logs: epoch, step/total_steps, lr, total_loss, surface_loss, rootzone_loss,
    valid_pixel_fraction, grad_norm, pred_increment stats, true_increment stats,
    GPU memory, batches/sec, ETA.
    """

    def __init__(
        self,
        log_every_steps: int = 100,
        max_epochs: int = 30,
        total_steps_per_epoch: Optional[int] = None,
    ):
        self.log_every_steps = log_every_steps
        self.max_epochs = max_epochs
        self.total_steps_per_epoch = total_steps_per_epoch
        self._start_time: Optional[float] = None

    def set_start_time(self, start_time: float) -> None:
        self._start_time = start_time

    def log_step(
        self,
        epoch: int,
        step: int,
        lr: float,
        total_loss: float,
        surface_loss: float,
        rootzone_loss: float,
        valid_pixel_fraction: float,
        grad_norm: Optional[float] = None,
        pred_inc_surface_mean: Optional[float] = None,
        pred_inc_surface_std: Optional[float] = None,
        pred_inc_rootzone_mean: Optional[float] = None,
        pred_inc_rootzone_std: Optional[float] = None,
        true_inc_surface_mean: Optional[float] = None,
        true_inc_surface_std: Optional[float] = None,
        true_inc_rootzone_mean: Optional[float] = None,
        true_inc_rootzone_std: Optional[float] = None,
        gpu_allocated_gb: Optional[float] = None,
        gpu_reserved_gb: Optional[float] = None,
        batches_per_sec: Optional[float] = None,
        total_steps: Optional[int] = None,
    ) -> None:
        if self._start_time is None:
            self._start_time = time.time()
        elapsed = time.time() - self._start_time

        # Calculate ETA
        eta_str = "?"
        if batches_per_sec is not None and batches_per_sec > 0:
            steps_done = epoch * (self.total_steps_per_epoch or 1) + step
            total_steps_val = total_steps or (self.total_steps_per_epoch or 1) * self.max_epochs
            steps_left = total_steps_val - steps_done
            if steps_left > 0:
                eta_s = steps_left / batches_per_sec
                if eta_s < 60:
                    eta_str = f"{eta_s:.0f}s"
                elif eta_s < 3600:
                    eta_str = f"{eta_s/60:.0f}min"
                else:
                    eta_str = f"{eta_s/3600:.1f}h"

        grad_str = f"g={grad_norm:.2e}" if grad_norm is not None else "g=?"
        mem_str = f"gpu={gpu_allocated_gb:.1f}GB" if gpu_allocated_gb is not None else ""

        pred_s = f"pred_s={pred_inc_surface_mean:.3f}/{pred_inc_surface_std:.3f}" if pred_inc_surface_mean is not None else ""
        pred_r = f"pred_r={pred_inc_rootzone_mean:.3f}/{pred_inc_rootzone_std:.3f}" if pred_inc_rootzone_mean is not None else ""
        true_s = f"true_s={true_inc_surface_mean:.3f}/{true_inc_surface_std:.3f}" if true_inc_surface_mean is not None else ""
        true_r = f"true_r={true_inc_rootzone_mean:.3f}/{true_inc_rootzone_std:.3f}" if true_inc_rootzone_mean is not None else ""

        print(
            f"  E{epoch:3d} S{step:5d} | "
            f"loss={total_loss:.4f} surf={surface_loss:.4f} root={rootzone_loss:.4f} | "
            f"valid={valid_pixel_fraction:.3f} {grad_str} | "
            f"{pred_s} {pred_r} | {true_s} {true_r} | "
            f"{mem_str} {batches_per_sec:.1f}b/s ETA={eta_str} | "
            f"lr={lr:.2e}",
            flush=True,
        )

    def log_epoch(
        self,
        epoch: int,
        avg_loss: float,
        avg_surface_loss: float,
        avg_rootzone_loss: float,
        valid_pixel_count: int,
        lr: float,
        elapsed_s: float,
    ) -> None:
        print(
            f"Epoch {epoch:3d} | "
            f"loss={avg_loss:.6f} | "
            f"surface={avg_surface_loss:.6f} | "
            f"rootzone={avg_rootzone_loss:.6f} | "
            f"valid_px={valid_pixel_count:9d} | "
            f"lr={lr:.2e} | "
            f"{elapsed_s:.1f}s",
            flush=True,
        )


class JSONLLogger:
    """Machine-readable append-only JSON logs.

    Produces:
    - logs/train_steps.jsonl — one dict per step
    - logs/train_epochs.jsonl — one dict per epoch
    - logs/eval_metrics.jsonl — one dict per eval

    Uses keep-alive file handles for efficiency.
    """

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._step_path = self.log_dir / "train_steps.jsonl"
        self._epoch_path = self.log_dir / "train_epochs.jsonl"
        self._eval_path = self.log_dir / "eval_metrics.jsonl"
        self._step_file: Optional[Any] = None
        self._epoch_file: Optional[Any] = None
        self._eval_file: Optional[Any] = None

    def _get_step_file(self):
        if self._step_file is None:
            self._step_file = open(self._step_path, "a")
        return self._step_file

    def _get_epoch_file(self):
        if self._epoch_file is None:
            self._epoch_file = open(self._epoch_path, "a")
        return self._epoch_file

    def _get_eval_file(self):
        if self._eval_file is None:
            self._eval_file = open(self._eval_path, "a")
        return self._eval_file

    def log_step(self, data: Dict[str, Any]) -> None:
        """Append a step record to train_steps.jsonl."""
        if self._step_file is not None and self._step_file.closed:
            self._step_file = None
        f = self._get_step_file()
        f.write(json.dumps(data) + "\n")
        f.flush()

    def log_epoch(self, data: Dict[str, Any]) -> None:
        """Append an epoch record to train_epochs.jsonl."""
        if self._epoch_file is not None and self._epoch_file.closed:
            self._epoch_file = None
        f = self._get_epoch_file()
        f.write(json.dumps(data) + "\n")
        f.flush()

    def log_eval(self, data: Dict[str, Any]) -> None:
        """Append an eval record to eval_metrics.jsonl."""
        if self._eval_file is not None and self._eval_file.closed:
            self._eval_file = None
        f = self._get_eval_file()
        f.write(json.dumps(data) + "\n")
        f.flush()

    def close(self) -> None:
        """Close all file handles."""
        for f in [self._step_file, self._epoch_file, self._eval_file]:
            if f is not None and not f.closed:
                f.close()


class WandbLogger:
    """Optional wandb integration (disabled by default).

    Modes: disabled/offline/online

    Records: loss, lr, grad_norm, valid_pixel_fraction, pred_inc_std, gpu_memory, samples_per_sec
    source_val metrics: global_analysis_skill, increment_rmse, increment_corr, increment_bias
    """

    def __init__(
        self,
        mode: str = "disabled",
        project: str = "hydroda-ood",
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        run_name: Optional[str] = None,
    ):
        self.mode = mode
        self.project = project
        self.entity = entity
        self.tags = tags or []
        self.run_name = run_name
        self._run = None

        if mode == "disabled":
            self._enabled = False
            return

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            print("WARNING: wandb not installed, falling back to disabled mode")
            self._enabled = False
            return

        self._enabled = True

        wandb_mode = "disabled" if mode == "disabled" else mode
        self._run = wandb.init(
            project=project,
            entity=entity,
            tags=tags,
            name=run_name,
            mode=wandb_mode,
        )

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log data to wandb."""
        if not self._enabled or self._run is None:
            return
        self._run.log(data, step=step)

    def log_step(self, data: Dict[str, Any]) -> None:
        """Log step-level data."""
        self.log(data)

    def log_epoch(self, data: Dict[str, Any]) -> None:
        """Log epoch-level data."""
        self.log(data)

    def log_eval(self, data: Dict[str, Any]) -> None:
        """Log eval-level data with prefix."""
        if not self._enabled or self._run is None:
            return
        prefixed = {f"eval/{k}": v for k, v in data.items()}
        self._run.log(prefixed)

    def finish(self) -> None:
        if self._enabled and self._run is not None:
            self._run.finish()

    @property
    def run_id(self) -> Optional[str]:
        if self._enabled and self._run is not None:
            return self._run.id
        return None

    @property
    def enabled(self) -> bool:
        return self._enabled