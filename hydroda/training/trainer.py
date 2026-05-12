"""Trainer for source-only backbone in HydroDA-OOD / HyperDA V4.

No-leakage declaration:
    - Training uses source_fit split only (2015-2020, US-R1..R6 excluding target)
    - Normalization stats computed from source_fit only (LeakageGuard check)
    - No target_query labels used in training / normalization / early_stopping
    - No target prompt used
"""
from __future__ import annotations

import gc
import json
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from hydroda.data.dataset import HydroDADataset
from hydroda.data.leakage_guard import LeakageGuard
from hydroda.data.protocol import ProtocolConfig
from hydroda.training.losses import MaskedHuberLoss
from hydroda.utils.device import gpu_health_check
from hydroda.utils.run_manager import RunManager
from hydroda.utils.logger import WandbLogger
from hydroda.utils.runtime import get_git_hash


def _compute_channel_stats(dataset: HydroDADataset, sample_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from a sample of dataset indices.

    Uses robust sampling: at most 50 samples, spread across the dataset.

    Returns:
        mean per channel (12,), std per channel (12,)
    """
    n_samples = min(50, len(sample_indices))
    step = max(1, len(sample_indices) // n_samples)
    indices = sample_indices[::step][:n_samples]

    sums = None
    sq_sums = None
    count = 0

    for idx in indices:
        sample = dataset[idx]
        x = sample["x"]  # (12, H, W)
        # Only consider finite values
        valid = np.isfinite(x)
        # Sum per channel
        for c in range(x.shape[0]):
            ch_data = x[c][valid[c]]
            if ch_data.size == 0:
                continue
            if sums is None:
                sums = np.zeros(12, dtype=np.float64)
                sq_sums = np.zeros(12, dtype=np.float64)
            sums[c] += ch_data.sum()
            sq_sums[c] += (ch_data ** 2).sum()
            count += ch_data.size

    if sums is None:
        # Fallback: return ones (no normalization)
        return np.ones(12, dtype=np.float32), np.ones(12, dtype=np.float32)

    channel_counts = np.zeros(12, dtype=np.float64)
    for idx in indices:
        sample = dataset[idx]
        x = sample["x"]
        valid = np.isfinite(x)
        for c in range(12):
            channel_counts[c] += valid[c].sum()

    means = sums / np.maximum(channel_counts, 1.0)
    variances = (sq_sums / np.maximum(channel_counts, 1.0)) - (means ** 2)
    variances = np.maximum(variances, 0.0)
    stds = np.sqrt(variances) + 1e-6

    return means.astype(np.float32), stds.astype(np.float32)


class Trainer:
    """Trainer for source-only SmallResUNet backbone.

    Handles:
    - DataLoader construction from source_train split
    - Optimization (AdamW, weight_decay)
    - Training loop with loss recording
    - Checkpoint saving (best.pt, last.pt)
    - Source-train-only normalization stats
    - LeakageGuard integration (check_normalization_scope before training)
    - Optional RunManager + JSONL logging + WandbLogger + AMP

    Args:
        model: PyTorch nn.Module (SmallResUNet)
        train_dataset: HydroDADataset for source_train split
        lr: learning rate (default 1e-3)
        weight_decay: weight decay (default 1e-4)
        max_epochs: max training epochs (default 30)
        batch_size: batch size (default 4)
        num_workers: DataLoader num_workers (default 0 for netCDF safety)
        device: device string (default "cuda")
        checkpoint_dir: checkpoint output directory
        experiment_id: experiment identifier
        protocol_freeze_id: protocol freeze identifier
        split_manifest_path: path to split freeze manifest
        grad_clip: gradient clipping value (None = no clipping)
        model_width: model width (default 32)
        target_increment_normalization: normalize target increments (default False)
        zero_raw_increment_init: zero-init output head (default False)
        accum_steps: gradient accumulation steps (default 1)
        run_manager: Optional RunManager for unified run dir + JSONL logging
        use_amp: use automatic mixed precision (default False)
        log_every_steps: log to console/JSONL every N steps (default 100)
        eval_every_epochs: run source_val eval every N epochs (default 1)
        wandb_logger: Optional WandbLogger instance
        source_val_dataset: Optional source_val dataset for eval
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: HydroDADataset,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 30,
        batch_size: int = 4,
        num_workers: int = 0,
        device: str = "cuda",
        checkpoint_dir: str = "artifacts/checkpoints/phase4_source_only",
        experiment_id: str = "phase4_source_only",
        protocol_freeze_id: str = "hyperda_v4_final_2015_2025_context2022_query2023_2025_k0_4_12",
        split_manifest_path: str = "artifacts/protocol/US_region_split_freeze_manifest.json",
        grad_clip: Optional[float] = None,
        model_width: int = 32,
        target_increment_normalization: bool = False,
        zero_raw_increment_init: bool = False,
        accum_steps: int = 1,
        run_manager: Optional[RunManager] = None,
        use_amp: bool = False,
        log_every_steps: int = 100,
        eval_every_epochs: int = 1,
        wandb_logger: Optional[WandbLogger] = None,
        source_val_dataset: Optional[HydroDADataset] = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_dataset = train_dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_id = experiment_id
        self.protocol_freeze_id = protocol_freeze_id
        self.split_manifest_path = split_manifest_path
        self.grad_clip = grad_clip
        self.model_width = model_width
        self.target_increment_normalization = target_increment_normalization
        self.zero_raw_increment_init = zero_raw_increment_init
        self.accum_steps = accum_steps
        self.run_manager = run_manager
        self.use_amp = use_amp and (device == "cuda")
        self.log_every_steps = log_every_steps
        self.eval_every_epochs = eval_every_epochs
        self.wandb_logger = wandb_logger
        self.source_val_dataset = source_val_dataset

        # AMP scaler
        self._amp_scaler: Optional[GradScaler] = None
        if self.use_amp:
            self._amp_scaler = GradScaler('cuda')

        # Leakage guard: check normalization scope with actual training dates
        protocol = ProtocolConfig()
        guard = LeakageGuard(protocol=protocol)
        train_date_strs = [d["date_str"] for d in self.train_dataset._date_records] if hasattr(self.train_dataset, "_date_records") else []
        guard.check_normalization_scope(train_date_strs, scope_name="source_fit_only")

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        self.loss_fn = MaskedHuberLoss(delta=0.01)

        # Compute normalization stats from source_train
        self._ch_mean: Optional[np.ndarray] = None
        self._ch_std: Optional[np.ndarray] = None
        self._compute_normalization_stats()

        # Increment normalization stats (for target increments)
        self._inc_mean: Optional[np.ndarray] = None
        self._inc_std: Optional[np.ndarray] = None
        if self.target_increment_normalization:
            self._compute_increment_stats()

        # Zero-raw-increment initialization
        if self.zero_raw_increment_init:
            if self.target_increment_normalization and self._inc_mean is not None and self._inc_std is not None:
                bias_surface = -self._inc_mean[0] / self._inc_std[0]
                bias_rootzone = -self._inc_mean[1] / self._inc_std[1]
                with torch.no_grad():
                    self.model.head.bias[0] = torch.tensor(bias_surface, device=self.model.head.bias.device)
                    self.model.head.bias[1] = torch.tensor(bias_rootzone, device=self.model.head.bias.device)
                print(f"  zero_raw_increment_init: bias_norm surface={bias_surface:.6f}, rootzone={bias_rootzone:.6f}")
            else:
                print(f"  zero_raw_increment_init: standard zero-init (no inc normalization)")

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []

        # JSONL logger from run_manager
        self._jsonl_logger = None
        if run_manager is not None:
            from hydroda.utils.logger import JSONLLogger
            self._jsonl_logger = JSONLLogger(run_manager.get_log_dir())
            # Open console.log for tee output
            run_manager.open_console_log()

    def _compute_normalization_stats(self) -> None:
        """Compute per-channel mean/std from training dataset (source_fit)."""
        print(f"Computing normalization stats from training dataset (n={len(self.train_dataset)})...")
        indices = list(range(len(self.train_dataset)))
        means, stds = _compute_channel_stats(self.train_dataset, indices)
        self._ch_mean = means
        self._ch_std = stds
        print(f"  Channel means: {means[:4]}... (12 channels)")
        print(f"  Channel stds:  {stds[:4]}... (12 channels)")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise normalization to input tensor.

        Includes NaN/Inf guard: if normalization produces invalid values,
        returns original x with a warning (prevents GPU corruption from
        NaN/Inf propagation through the model).
        """
        if self._ch_mean is None or self._ch_std is None:
            return x
        mean_t = torch.from_numpy(self._ch_mean).to(x.device).view(1, 12, 1, 1)
        std_t = torch.from_numpy(self._ch_std).to(x.device).view(1, 12, 1, 1)
        x_norm = (x - mean_t) / std_t
        if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
            n_nan = torch.isnan(x_norm).sum().item()
            n_inf = torch.isinf(x_norm).sum().item()
            print(f"  WARNING: normalize produced {n_nan} NaN / {n_inf} Inf — returning raw input", flush=True)
            return x
        return x_norm

    def _compute_increment_stats(self) -> None:
        """Compute mean/std of surface and rootzone increments from training dataset (source_fit)."""
        print(f"Computing increment stats from training dataset (n={len(self.train_dataset)})...")
        n_samples = min(200, len(self.train_dataset))
        step = max(1, len(self.train_dataset) // n_samples)
        indices = list(range(0, len(self.train_dataset), step))[:n_samples]

        inc_s_values = []
        inc_r_values = []
        for idx in indices:
            sample = self.train_dataset[idx]
            inc_s = sample["increment_surface"]
            inc_r = sample["increment_rootzone"]
            valid_s = np.isfinite(inc_s)
            valid_r = np.isfinite(inc_r)
            inc_s_values.append(inc_s[valid_s].reshape(-1))
            inc_r_values.append(inc_r[valid_r].reshape(-1))

        inc_s_all = np.concatenate(inc_s_values)
        inc_r_all = np.concatenate(inc_r_values)

        inc_mean = np.array([inc_s_all.mean(), inc_r_all.mean()], dtype=np.float32)
        inc_std = np.array([inc_s_all.std(), inc_r_all.std()], dtype=np.float32)
        inc_std = np.maximum(inc_std, 1e-6)

        self._inc_mean = inc_mean
        self._inc_std = inc_std
        print(f"  Increment means: surface={inc_mean[0]:.6f}, rootzone={inc_mean[1]:.6f}")
        print(f"  Increment stds:  surface={inc_std[0]:.6f}, rootzone={inc_std[1]:.6f}")

    def _build_dataloader(self, dataset: Optional[HydroDADataset] = None) -> DataLoader:
        """Build DataLoader for training or eval."""
        target_dataset = dataset or self.train_dataset

        def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            x = torch.from_numpy(np.stack([s["x"] for s in batch], axis=0))
            increment_surface = torch.from_numpy(
                np.stack([s["increment_surface"] for s in batch], axis=0)
            )
            increment_rootzone = torch.from_numpy(
                np.stack([s["increment_rootzone"] for s in batch], axis=0)
            )
            loss_mask = torch.from_numpy(
                np.stack([s["loss_mask"] for s in batch], axis=0)
            )
            return {
                "x": x,
                "increment_surface": increment_surface,
                "increment_rootzone": increment_rootzone,
                "loss_mask": loss_mask,
            }

        pin_mem = self.device == "cuda"
        return DataLoader(
            target_dataset,
            batch_size=self.batch_size,
            shuffle=(dataset is None),
            num_workers=self.num_workers,
            pin_memory=pin_mem,
            collate_fn=collate_fn,
        )

    def _eval_source_val(self) -> Dict[str, float]:
        """Run evaluation on source_val split and return metrics dict."""
        if self.source_val_dataset is None:
            return {}
        self.model.eval()
        loader = self._build_dataloader(self.source_val_dataset)
        total_loss = 0.0
        total_surface = 0.0
        total_rootzone = 0.0
        total_valid = 0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                inc_surface = batch["increment_surface"].to(self.device)
                inc_rootzone = batch["increment_rootzone"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)

                x_norm = self._normalize(x)
                target = torch.stack([inc_surface, inc_rootzone], dim=1)
                if self.target_increment_normalization and self._inc_mean is not None:
                    inc_mean_t = torch.from_numpy(self._inc_mean).to(x.device).view(1, 2, 1, 1)
                    inc_std_t = torch.from_numpy(self._inc_std).to(x.device).view(1, 2, 1, 1)
                    target = (target - inc_mean_t) / inc_std_t

                if self.use_amp:
                    with autocast('cuda'):
                        pred = self.model(x_norm)
                        losses = self.loss_fn(pred, target, loss_mask)
                else:
                    pred = self.model(x_norm)
                    losses = self.loss_fn(pred, target, loss_mask)

                total_loss += float(losses["total_loss"].item())
                total_surface += float(losses["surface_loss"].item())
                total_rootzone += float(losses["rootzone_loss"].item())
                total_valid += int(losses["valid_pixel_count"].item())
                n_batches += 1

        self.model.train()
        return {
            "source_val_loss": total_loss / max(n_batches, 1),
            "source_val_surface_loss": total_surface / max(n_batches, 1),
            "source_val_rootzone_loss": total_rootzone / max(n_batches, 1),
            "source_val_valid_px": total_valid,
        }

    def train(self, verbose: bool = True) -> List[Dict[str, float]]:
        """Run the training loop.

        Returns:
            train_history: list of dicts with per-epoch metrics
        """
        dataloader = self._build_dataloader()
        self.model.train()
        global_step = 0
        train_start_time = time.time()
        total_steps_per_epoch = len(dataloader)

        # --- Training start header ---
        num_params = sum(p.numel() for p in self.model.parameters())
        header_lines = [
            "=" * 60,
            f"Training Start",
            f"  Experiment:      {self.experiment_id}",
            f"  Protocol:        {self.protocol_freeze_id}",
            f"  Split manifest:  {self.split_manifest_path}",
            f"  Device:          {self.device}",
            f"  Model width:     {self.model_width}",
            f"  Trainable params:{num_params:,}",
            f"  Batch size:      {self.batch_size}",
            f"  Accum steps:     {self.accum_steps}",
            f"  Max epochs:      {self.max_epochs}",
            f"  LR:              {self.lr}",
            f"  Weight decay:    {self.weight_decay}",
            f"  Grad clip:       {self.grad_clip}",
            f"  AMP:             {self.use_amp}",
            f"  Inc norm:        {self.target_increment_normalization}",
            f"  Zero raw init:   {self.zero_raw_increment_init}",
            f"  Train samples:   {len(self.train_dataset)}",
            f"  Steps/epoch:     {total_steps_per_epoch}",
            "=" * 60,
        ]
        for line in header_lines:
            if self.run_manager is not None:
                self.run_manager.log_console(line)
            elif verbose:
                print(line)

        # GPU health check before training (catches dead/flaky GPUs early)
        if self.device == "cuda":
            if not gpu_health_check(torch.device("cuda")):
                raise RuntimeError(
                    "GPU health check FAILED — GPU is unresponsive. "
                    "The device may be in an error state. Try rebooting or using a different GPU."
                )

        for epoch in range(self.current_epoch, self.max_epochs):
            epoch_losses = []
            epoch_surface_losses = []
            epoch_rootzone_losses = []
            epoch_valid_counts = []
            epoch_start = time.time()
            batches_since_eval = 0

            # Zero gradients at start of epoch (gradient accumulation fix)
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(dataloader):
                x = batch["x"].to(self.device)
                inc_surface = batch["increment_surface"].to(self.device)
                inc_rootzone = batch["increment_rootzone"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)

                x_norm = self._normalize(x)

                # NaN/Inf guard on normalized input: skip batch if invalid
                if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
                    n_nan = torch.isnan(x_norm).sum().item()
                    n_inf = torch.isinf(x_norm).sum().item()
                    print(f"  WARNING: E{epoch} S{batch_idx}: normalized input {n_nan} NaN / {n_inf} Inf — skipping batch", flush=True)
                    continue

                target = torch.stack([inc_surface, inc_rootzone], dim=1)

                if self.target_increment_normalization and self._inc_mean is not None:
                    inc_mean_t = torch.from_numpy(self._inc_mean).to(x.device).view(1, 2, 1, 1)
                    inc_std_t = torch.from_numpy(self._inc_std).to(x.device).view(1, 2, 1, 1)
                    target = (target - inc_mean_t) / inc_std_t

                # Forward pass
                ctx = autocast('cuda') if self.use_amp else nullcontext()
                with ctx:
                    pred = self.model(x_norm)
                    losses = self.loss_fn(pred, target, loss_mask)

                # NaN/Inf guard on loss: skip batch if invalid
                if torch.isnan(losses["total_loss"]) or torch.isinf(losses["total_loss"]):
                    print(f"  WARNING: E{epoch} S{batch_idx}: loss is NaN/Inf — skipping batch", flush=True)
                    continue

                # Backward pass
                if self.use_amp:
                    self._amp_scaler.scale(losses["total_loss"]).backward()
                    if (batch_idx + 1) % self.accum_steps == 0:
                        if self.grad_clip is not None:
                            self._amp_scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self._amp_scaler.step(self.optimizer)
                        self._amp_scaler.update()
                        self.optimizer.zero_grad()
                else:
                    losses["total_loss"].backward()
                    if (batch_idx + 1) % self.accum_steps == 0:
                        if self.grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                epoch_losses.append(float(losses["total_loss"].item()))
                epoch_surface_losses.append(float(losses["surface_loss"].item()))
                epoch_rootzone_losses.append(float(losses["rootzone_loss"].item()))
                epoch_valid_counts.append(int(losses["valid_pixel_count"].item()))

                # Per-step logging
                if batch_idx % self.log_every_steps == 0:
                    # Compute grad norm
                    grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5 if grad_norm > 0 else 0.0

                    # Compute pred stats from latest batch
                    pred_s_mean = float(pred[:, 0].mean().item())
                    pred_s_std = float(pred[:, 0].std().item())
                    pred_r_mean = float(pred[:, 1].mean().item())
                    pred_r_std = float(pred[:, 1].std().item())
                    target_s_mean = float(target[:, 0].mean().item())
                    target_s_std = float(target[:, 0].std().item())
                    target_r_mean = float(target[:, 1].mean().item())
                    target_r_std = float(target[:, 1].std().item())

                    # GPU memory
                    gpu_alloc = 0.0
                    gpu_res = 0.0
                    if self.device == "cuda":
                        dev_idx = torch.cuda.current_device()
                        gpu_alloc = torch.cuda.memory_allocated(dev_idx) / 1e9
                        gpu_res = torch.cuda.memory_reserved(dev_idx) / 1e9

                    lr = float(self.optimizer.param_groups[0]["lr"])
                    valid_px = int(losses["valid_pixel_count"].item())
                    total_px = loss_mask.numel()
                    valid_fraction = valid_px / max(total_px, 1)

                    step_data = {
                        "epoch": epoch,
                        "step": batch_idx,
                        "global_step": global_step,
                        "lr": lr,
                        "total_loss": float(losses["total_loss"].item()),
                        "surface_loss": float(losses["surface_loss"].item()),
                        "rootzone_loss": float(losses["rootzone_loss"].item()),
                        "valid_pixel_fraction": valid_fraction,
                        "grad_norm": round(grad_norm, 4),
                        "pred_inc_surface_mean": round(pred_s_mean, 6),
                        "pred_inc_surface_std": round(pred_s_std, 6),
                        "pred_inc_rootzone_mean": round(pred_r_mean, 6),
                        "pred_inc_rootzone_std": round(pred_r_std, 6),
                        "true_inc_surface_mean": round(target_s_mean, 6),
                        "true_inc_surface_std": round(target_s_std, 6),
                        "true_inc_rootzone_mean": round(target_r_mean, 6),
                        "true_inc_rootzone_std": round(target_r_std, 6),
                        "gpu_allocated_gb": round(gpu_alloc, 2),
                        "gpu_reserved_gb": round(gpu_res, 2),
                    }

                    # JSONL step log
                    if self._jsonl_logger is not None:
                        self._jsonl_logger.log_step(step_data)

                    # Console step log
                    if verbose:
                        elapsed = time.time() - train_start_time
                        batches_per_sec = (batch_idx + 1) / max(elapsed, 0.1)
                        print(
                            f"  E{epoch:3d} S{batch_idx:5d} | "
                            f"loss={losses['total_loss'].item():.4f} surf={losses['surface_loss'].item():.4f} "
                            f"root={losses['rootzone_loss'].item():.4f} | "
                            f"valid={valid_fraction:.3f} g={grad_norm:.2e} | "
                            f"pred_s={pred_s_mean:.3f}/{pred_s_std:.3f} pred_r={pred_r_mean:.3f}/{pred_r_std:.3f} | "
                            f"true_s={target_s_mean:.3f}/{target_s_std:.3f} true_r={target_r_mean:.3f}/{target_r_std:.3f} | "
                            f"gpu={gpu_alloc:.1f}GB {batches_per_sec:.1f}b/s | lr={lr:.2e}",
                            flush=True,
                        )

                    # Wandb step log
                    if self.wandb_logger is not None and self.wandb_logger.enabled:
                        wandb_data = {
                            "train/total_loss": float(losses["total_loss"].item()),
                            "train/surface_loss": float(losses["surface_loss"].item()),
                            "train/rootzone_loss": float(losses["rootzone_loss"].item()),
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "train/valid_pixel_fraction": valid_fraction,
                            "train/pred_inc_surface_std": pred_s_std,
                            "train/pred_inc_rootzone_std": pred_r_std,
                            "train/gpu_memory_gb": gpu_alloc,
                        }
                        self.wandb_logger.log_step(wandb_data)

                global_step += 1
                batches_since_eval += 1

            avg_loss = float(np.mean(epoch_losses))
            avg_surface = float(np.mean(epoch_surface_losses))
            avg_rootzone = float(np.mean(epoch_rootzone_losses))
            total_valid = int(np.sum(epoch_valid_counts))
            elapsed = time.time() - epoch_start

            self.scheduler.step(avg_loss)

            # Source val eval every eval_every_epochs
            source_val_metrics = {}
            if self.source_val_dataset is not None and epoch % self.eval_every_epochs == 0:
                source_val_metrics = self._eval_source_val()
                if verbose:
                    sv_loss = source_val_metrics.get("source_val_loss", float("nan"))
                    print(f"  source_val loss={sv_loss:.6f}", flush=True)
                if self._jsonl_logger is not None:
                    self._jsonl_logger.log_eval({"epoch": epoch, **source_val_metrics})

            # Best checkpoint selection: use source_val loss if available, else train loss
            is_best = False
            best_metric = avg_loss
            if source_val_metrics and "source_val_loss" in source_val_metrics:
                sv_loss = source_val_metrics["source_val_loss"]
                if sv_loss < self.best_loss:
                    is_best = True
                    best_metric = sv_loss
            elif avg_loss < self.best_loss:
                is_best = True

            if is_best:
                self.best_loss = best_metric
                self.save_checkpoint(self.checkpoint_dir / "best.pt", epoch, best_metric, "best")

            self.save_checkpoint(self.checkpoint_dir / "last.pt", epoch, avg_loss, "last")

            record = {
                "epoch": epoch,
                "surface_loss": avg_surface,
                "rootzone_loss": avg_rootzone,
                "total_loss": avg_loss,
                "valid_pixel_count": total_valid,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "elapsed_s": elapsed,
            }
            record.update(source_val_metrics)
            self.train_history.append(record)
            if source_val_metrics:
                self.val_history.append({"epoch": epoch, **source_val_metrics})
            self.current_epoch = epoch + 1

            # JSONL epoch log
            if self._jsonl_logger is not None:
                self._jsonl_logger.log_epoch(record)

            # Wandb epoch log
            if self.wandb_logger is not None and self.wandb_logger.enabled:
                wandb_data = {f"train/{k}": v for k, v in record.items()}
                self.wandb_logger.log_epoch(wandb_data)

            if verbose:
                # Per-epoch summary table
                sv_str = ""
                if source_val_metrics:
                    sv_loss = source_val_metrics.get("source_val_loss", float("nan"))
                    sv_str = f"  sv_loss={sv_loss:.6f}"
                epoch_summary = (
                    f"Epoch {epoch:3d} | "
                    f"loss={avg_loss:.6f} | "
                    f"surface={avg_surface:.6f} | "
                    f"rootzone={avg_rootzone:.6f} | "
                    f"valid_px={total_valid:9d} | "
                    f"lr={record['lr']:.2e} | "
                    f"{elapsed:.1f}s{sv_str}"
                )
                if self.run_manager is not None:
                    self.run_manager.log_console(epoch_summary)
                elif verbose:
                    print(epoch_summary)

                # Per-epoch divider every 5 epochs or at epoch 0
                if epoch == 0 or (epoch + 1) % 5 == 0:
                    divider = "  " + "-" * 40
                    if self.run_manager is not None:
                        self.run_manager.log_console(divider)
                    elif verbose:
                        print(divider)

        # --- Training end summary ---
        total_elapsed = time.time() - train_start_time
        end_lines = [
            "=" * 60,
            f"Training Complete",
            f"  Total epochs:     {self.max_epochs}",
            f"  Best loss:        {self.best_loss:.6f}",
            f"  Total time:       {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)",
            f"  Checkpoint dir:   {self.checkpoint_dir}",
            "=" * 60,
        ]
        for line in end_lines:
            if self.run_manager is not None:
                self.run_manager.log_console(line)
            elif verbose:
                print(line)

        # Close console.log if opened
        if self.run_manager is not None:
            self.run_manager.close_console_log()

        return self.train_history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        loss: float,
        tag: str = "",
    ) -> None:
        """Save a checkpoint with config, protocol_freeze_id, split_manifest_path, git_hash."""
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "tag": tag,
            "epoch": epoch,
            "loss": loss,
            "best_loss": self.best_loss,
            "experiment_id": self.experiment_id,
            "protocol_freeze_id": self.protocol_freeze_id,
            "split_manifest_path": self.split_manifest_path,
            "git_hash": get_git_hash(),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_history": self.train_history,
            "config": {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "max_epochs": self.max_epochs,
                "batch_size": self.batch_size,
                "accum_steps": self.accum_steps,
                "effective_batch_size": self.batch_size * self.accum_steps,
                "grad_clip": self.grad_clip,
                "width": self.model_width,
                "num_workers": self.num_workers,
                "target_increment_normalization": self.target_increment_normalization,
                "zero_raw_increment_init": self.zero_raw_increment_init,
                "use_amp": self.use_amp,
                "log_every_steps": self.log_every_steps,
                "ch_mean": self._ch_mean.tolist() if self._ch_mean is not None else None,
                "ch_std": self._ch_std.tolist() if self._ch_std is not None else None,
                "inc_mean": self._inc_mean.tolist() if self._inc_mean is not None else None,
                "inc_std": self._inc_std.tolist() if self._inc_std is not None else None,
            },
        }
        torch.save(checkpoint, path)

    def save_summary_json(self, path: Optional[Path] = None) -> None:
        """Save summary.json with protocol safety fields."""
        has_source_val = self.source_val_dataset is not None
        num_params = sum(p.numel() for p in self.model.parameters())
        summary = {
            "experiment_id": self.experiment_id,
            "protocol_freeze_id": self.protocol_freeze_id,
            "best_loss": self.best_loss,
            "final_epoch": self.current_epoch - 1,
            "total_epochs_completed": self.current_epoch,
            "model_width": self.model_width,
            "trainable_parameters": num_params,
            "batch_size": self.batch_size,
            "accum_steps": self.accum_steps,
            "effective_batch_size": self.batch_size * self.accum_steps,
            "source_val_available": has_source_val,
            "normalization_source": "source_fit_only",
            "early_stopping_source": "source_val_only" if has_source_val else "train_loss_only",
            "model_selection_source": "source_val_only" if has_source_val else "best_train_loss",
            "target_query_usage": "eval_only_no_early_stopping",
            "leakage_guard_status": "pass",
            "git_hash": get_git_hash(),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "train_history": self.train_history,
            "val_history": self.val_history,
        }
        target_path = path or (self.checkpoint_dir / "summary.json")
        with open(target_path, "w") as f:
            json.dump(summary, f, indent=2)

    @staticmethod
    def load_checkpoint(
        path: Path,
        model: nn.Module,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Load a checkpoint and return its metadata."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint