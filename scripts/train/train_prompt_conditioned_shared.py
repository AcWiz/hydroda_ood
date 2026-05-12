#!/usr/bin/env python3
"""Train prompt-conditioned shared backbone (FiLMConditionalResUNet + RegionPromptEncoder).

Multi-region mixed training: source_fit (2015-2020) from all source regions.
Each sample's prompt uses the region embedding for the region with the most
valid pixels in the sample.

Usage:
    PYTHONPATH=. python scripts/train/train_prompt_conditioned_shared.py \\
        --target_region US-R1 --K 0 --seed 0 \\
        --max_epochs 5 --batch_size 1 --accum_steps 4 --lr 3e-4 \\
        --weight_decay 1e-4 --grad_clip 1.0 \\
        --width 32 --prompt_dim 64 \\
        --zero_raw_increment_init --target_increment_normalization \\
        --device cuda --amp \\
        --config configs/model_resunet_main.yaml

No-leakage declaration:
    - Training uses source_fit split only (2015-2020, all source regions)
    - Normalization stats from source_fit only
    - Region prompt uses input-side features only
    - No target_query labels used in training/normalization/early_stopping
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from hydroda.data.dataset import HydroDADataset
from hydroda.data.leakage_guard import LeakageGuard
from hydroda.data.protocol import ProtocolConfig
from hydroda.models.conditional_unet import FiLMConditionalResUNet
from hydroda.models.prompt_encoder import RegionPromptEncoder
from hydroda.training.losses import MaskedHuberLoss
from hydroda.utils.run_manager import RunManager
from hydroda.utils.logger import WandbLogger, ConsoleLogger
from hydroda.utils.device import resolve_device, log_device_summary
from hydroda.utils.runtime import gather_runtime_info, get_git_hash


DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
CHECKPOINT_DIR = "artifacts/checkpoints/phase4_prompt_conditioned"
PROTOCOL_FREEZE_ID = "hyperda_v4_final_2015_2025_context2022_query2023_2025_k0_4_12"
PHASE = "phase4_prompt_conditioned"

_ALL_US_REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
_GLOBAL_REGION_IDX_MAP = {r: i for i, r in enumerate(_ALL_US_REGIONS)}


def _compute_channel_stats(
    dataset: HydroDADataset, sample_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from dataset samples."""
    n_samples = min(50, len(sample_indices))
    step = max(1, len(sample_indices) // n_samples)
    indices = sample_indices[::step][:n_samples]

    sums = np.zeros(12, dtype=np.float64)
    sq_sums = np.zeros(12, dtype=np.float64)
    channel_counts = np.zeros(12, dtype=np.float64)

    for idx in indices:
        sample = dataset[idx]
        x = sample["x"]
        valid = np.isfinite(x)
        for c in range(12):
            ch_data = x[c][valid[c]]
            if ch_data.size == 0:
                continue
            sums[c] += ch_data.sum()
            sq_sums[c] += (ch_data ** 2).sum()
            channel_counts[c] += ch_data.size

    means = sums / np.maximum(channel_counts, 1.0)
    variances = (sq_sums / np.maximum(channel_counts, 1.0)) - (means ** 2)
    variances = np.maximum(variances, 0.0)
    stds = np.sqrt(variances) + 1e-6

    return means.astype(np.float32), stds.astype(np.float32)


def _compute_increment_stats(
    dataset: HydroDADataset, sample_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/std of surface and rootzone increments from dataset."""
    n_samples = min(200, len(sample_indices))
    step = max(1, len(sample_indices) // n_samples)
    indices = list(range(0, len(sample_indices), step))[:n_samples]

    inc_s_values = []
    inc_r_values = []
    for idx in indices:
        sample = dataset[idx]
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

    return inc_mean, inc_std


def _sample_region_from_mask(region_mask_integer: np.ndarray, valid_mask: np.ndarray) -> int:
    """Determine the dominant region from region_mask_integer and valid pixels.

    Args:
        region_mask_integer: [H, W] with region indices (1-6)
        valid_mask: [H, W] boolean valid pixel mask

    Returns:
        region index (0-5)
    """
    for r_idx in range(1, 7):
        count = int(((region_mask_integer == r_idx) & valid_mask).sum())
        if count > 0:
            return r_idx - 1  # 0-indexed

    # Fallback: count per region
    counts = {}
    for r_idx in range(1, 7):
        counts[r_idx] = int((region_mask_integer == r_idx).sum())

    best = max(counts, key=counts.get)
    return best - 1


class PromptConditionedTrainer:
    """Trainer for FiLMConditionalResUNet + RegionPromptEncoder.

    Handles multi-region mixed source_fit training with region-conditioned prompts.
    """

    def __init__(
        self,
        model: FiLMConditionalResUNet,
        prompt_encoder: RegionPromptEncoder,
        train_dataset: HydroDADataset,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 30,
        batch_size: int = 4,
        num_workers: int = 0,
        device: str = "cuda",
        checkpoint_dir: str = "artifacts/checkpoints/phase4_prompt_conditioned",
        experiment_id: str = "phase4_prompt_conditioned",
        protocol_freeze_id: str = PROTOCOL_FREEZE_ID,
        split_manifest_path: str = FREEZE_MANIFEST,
        grad_clip: Optional[float] = None,
        model_width: int = 32,
        prompt_dim: int = 64,
        target_increment_normalization: bool = False,
        zero_raw_increment_init: bool = False,
        accum_steps: int = 1,
        run_manager: Optional[RunManager] = None,
        use_amp: bool = False,
        log_every_steps: int = 100,
        eval_every_epochs: int = 1,
        wandb_logger: Optional[WandbLogger] = None,
        source_val_dataset: Optional[HydroDADataset] = None,
        global_to_source_lookup: Optional[List[int]] = None,
    ) -> None:
        self.model = model.to(device)
        self.prompt_encoder = prompt_encoder.to(device)
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
        self.prompt_dim = prompt_dim
        self.target_increment_normalization = target_increment_normalization
        self.zero_raw_increment_init = zero_raw_increment_init
        self.accum_steps = accum_steps
        self.run_manager = run_manager
        self.use_amp = use_amp and (device == "cuda")
        self.log_every_steps = log_every_steps
        self.eval_every_epochs = eval_every_epochs
        self.wandb_logger = wandb_logger
        self.source_val_dataset = source_val_dataset
        self.global_to_source_lookup = global_to_source_lookup or [0] * 6

        # AMP
        self._amp_scaler: Optional[GradScaler] = None
        if self.use_amp:
            self._amp_scaler = GradScaler('cuda')

        # Leakage guard
        protocol = ProtocolConfig()
        guard = LeakageGuard(protocol=protocol)
        guard.check_normalization_scope([], scope_name="source_fit_only")

        # Optimizer: model + prompt_encoder
        all_params = list(model.parameters()) + list(prompt_encoder.parameters())
        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )
        self.loss_fn = MaskedHuberLoss(delta=0.01)

        # Normalization stats
        self._ch_mean: Optional[np.ndarray] = None
        self._ch_std: Optional[np.ndarray] = None
        self._compute_normalization_stats()

        # Increment stats
        self._inc_mean: Optional[np.ndarray] = None
        self._inc_std: Optional[np.ndarray] = None
        if self.target_increment_normalization:
            self._compute_increment_stats()

        # Zero-raw-init
        if self.zero_raw_increment_init:
            if self.target_increment_normalization and self._inc_mean is not None:
                bias_surface = -self._inc_mean[0] / self._inc_std[0]
                bias_rootzone = -self._inc_mean[1] / self._inc_std[1]
                with torch.no_grad():
                    self.model.head.bias[0] = torch.tensor(
                        bias_surface, device=self.model.head.bias.device
                    )
                    self.model.head.bias[1] = torch.tensor(
                        bias_rootzone, device=self.model.head.bias.device
                    )
                print(f"  zero_raw_increment_init: bias_norm surface={bias_surface:.6f}, rootzone={bias_rootzone:.6f}")
            else:
                print(f"  zero_raw_increment_init: standard zero-init (no inc normalization)")

        # State
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []

        # JSONL logger
        self._jsonl_logger = None
        if run_manager is not None:
            from hydroda.utils.logger import JSONLLogger
            self._jsonl_logger = JSONLLogger(run_manager.get_log_dir())

    def _compute_normalization_stats(self) -> None:
        print(f"Computing normalization stats from training dataset (n={len(self.train_dataset)})...")
        indices = list(range(len(self.train_dataset)))
        means, stds = _compute_channel_stats(self.train_dataset, indices)
        self._ch_mean = means
        self._ch_std = stds
        print(f"  Channel means: {means[:4]}...")
        print(f"  Channel stds:  {stds[:4]}...")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._ch_mean is None or self._ch_std is None:
            return x
        mean_t = torch.from_numpy(self._ch_mean).to(x.device).view(1, 12, 1, 1)
        std_t = torch.from_numpy(self._ch_std).to(x.device).view(1, 12, 1, 1)
        return (x - mean_t) / std_t

    def _compute_increment_stats(self) -> None:
        print(f"Computing increment stats from training dataset (n={len(self.train_dataset)})...")
        indices = list(range(len(self.train_dataset)))
        inc_mean, inc_std = _compute_increment_stats(self.train_dataset, indices)
        self._inc_mean = inc_mean
        self._inc_std = inc_std
        print(f"  Increment means: surface={inc_mean[0]:.6f}, rootzone={inc_mean[1]:.6f}")
        print(f"  Increment stds:  surface={inc_std[0]:.6f}, rootzone={inc_std[1]:.6f}")

    def _build_dataloader(self, dataset: Optional[HydroDADataset] = None) -> DataLoader:
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
            # Determine region per sample from region_mask_integer
            # Map global region index (0..5) to source-only index for prompt encoder
            region_ids = []
            months = []
            for s in batch:
                valid_mask = np.isfinite(s["forecast_surface"]) & np.isfinite(s["forecast_rootzone"])
                global_rid = _sample_region_from_mask(s["region_mask_integer"], valid_mask)
                # Map to source-only index (0..num_source_regions-1)
                src_rid = self.global_to_source_lookup[global_rid] if global_rid < len(self.global_to_source_lookup) else 0
                region_ids.append(src_rid)
                months.append(int(s.get("month", 6)))
            return {
                "x": x,
                "increment_surface": increment_surface,
                "increment_rootzone": increment_rootzone,
                "loss_mask": loss_mask,
                "region_ids": torch.tensor(region_ids, dtype=torch.long),
                "months": torch.tensor(months, dtype=torch.long),
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
        if self.source_val_dataset is None:
            return {}
        self.model.eval()
        self.prompt_encoder.eval()
        loader = self._build_dataloader(self.source_val_dataset)
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                inc_surface = batch["increment_surface"].to(self.device)
                inc_rootzone = batch["increment_rootzone"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)
                region_ids = batch["region_ids"].to(self.device)
                months = batch["months"].to(self.device)

                x_norm = self._normalize(x)
                target = torch.stack([inc_surface, inc_rootzone], dim=1)
                if self.target_increment_normalization and self._inc_mean is not None:
                    inc_mean_t = torch.from_numpy(self._inc_mean).to(x.device).view(1, 2, 1, 1)
                    inc_std_t = torch.from_numpy(self._inc_std).to(x.device).view(1, 2, 1, 1)
                    target = (target - inc_mean_t) / inc_std_t

                z = self.prompt_encoder(x_norm, region_ids, months)
                pred = self.model(x_norm, z)
                losses = self.loss_fn(pred, target, loss_mask)
                total_loss += float(losses["total_loss"].item())
                n_batches += 1

        self.model.train()
        self.prompt_encoder.train()
        return {"source_val_loss": total_loss / max(n_batches, 1)}

    def train(self, verbose: bool = True) -> List[Dict[str, float]]:
        dataloader = self._build_dataloader()
        self.model.train()
        self.prompt_encoder.train()
        global_step = 0
        train_start_time = time.time()

        for epoch in range(self.current_epoch, self.max_epochs):
            epoch_losses = []
            epoch_start = time.time()

            for batch_idx, batch in enumerate(dataloader):
                x = batch["x"].to(self.device)
                inc_surface = batch["increment_surface"].to(self.device)
                inc_rootzone = batch["increment_rootzone"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)
                region_ids = batch["region_ids"].to(self.device)
                months = batch["months"].to(self.device)

                x_norm = self._normalize(x)
                target = torch.stack([inc_surface, inc_rootzone], dim=1)

                if self.target_increment_normalization and self._inc_mean is not None:
                    inc_mean_t = torch.from_numpy(self._inc_mean).to(x.device).view(1, 2, 1, 1)
                    inc_std_t = torch.from_numpy(self._inc_std).to(x.device).view(1, 2, 1, 1)
                    target = (target - inc_mean_t) / inc_std_t

                self.optimizer.zero_grad()

                if self.use_amp:
                    with autocast('cuda'):
                        z = self.prompt_encoder(x_norm, region_ids, months)
                        pred = self.model(x_norm, z)
                        losses = self.loss_fn(pred, target, loss_mask)
                    self._amp_scaler.scale(losses["total_loss"]).backward()
                    if (batch_idx + 1) % self.accum_steps == 0:
                        if self.grad_clip is not None:
                            self._amp_scaler.unscale_(self.optimizer)
                            all_p = list(self.model.parameters()) + list(self.prompt_encoder.parameters())
                            torch.nn.utils.clip_grad_norm_(all_p, self.grad_clip)
                        self._amp_scaler.step(self.optimizer)
                        self._amp_scaler.update()
                else:
                    z = self.prompt_encoder(x_norm, region_ids, months)
                    pred = self.model(x_norm, z)
                    losses = self.loss_fn(pred, target, loss_mask)
                    losses["total_loss"].backward()
                    if (batch_idx + 1) % self.accum_steps == 0:
                        if self.grad_clip is not None:
                            all_p = list(self.model.parameters()) + list(self.prompt_encoder.parameters())
                            torch.nn.utils.clip_grad_norm_(all_p, self.grad_clip)
                        self.optimizer.step()

                epoch_losses.append(float(losses["total_loss"].item()))

                # Per-step logging
                if batch_idx % self.log_every_steps == 0:
                    grad_norm = 0.0
                    for p in list(self.model.parameters()) + list(self.prompt_encoder.parameters()):
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5

                    valid_fraction = float(losses["valid_pixel_count"].item()) / max(loss_mask.numel(), 1)
                    lr_curr = float(self.optimizer.param_groups[0]["lr"])

                    step_data = {
                        "epoch": epoch, "step": batch_idx, "global_step": global_step,
                        "lr": lr_curr,
                        "total_loss": float(losses["total_loss"].item()),
                        "surface_loss": float(losses["surface_loss"].item()),
                        "rootzone_loss": float(losses["rootzone_loss"].item()),
                        "valid_pixel_fraction": valid_fraction,
                        "grad_norm": round(grad_norm, 4),
                    }
                    if self._jsonl_logger is not None:
                        self._jsonl_logger.log_step(step_data)

                    if verbose:
                        elapsed = time.time() - train_start_time
                        batches_per_sec = (batch_idx + 1) / max(elapsed, 0.1)
                        print(
                            f"  E{epoch:3d} S{batch_idx:5d} | "
                            f"loss={losses['total_loss'].item():.4f} "
                            f"surf={losses['surface_loss'].item():.4f} "
                            f"root={losses['rootzone_loss'].item():.4f} | "
                            f"valid={valid_fraction:.3f} g={grad_norm:.2e} | "
                            f"{batches_per_sec:.1f}b/s | lr={lr_curr:.2e}",
                            flush=True,
                        )

                    if self.wandb_logger is not None and self.wandb_logger.enabled:
                        self.wandb_logger.log_step({
                            "train/total_loss": step_data["total_loss"],
                            "train/surface_loss": step_data["surface_loss"],
                            "train/rootzone_loss": step_data["rootzone_loss"],
                            "train/lr": lr_curr,
                            "train/grad_norm": grad_norm,
                            "train/valid_pixel_fraction": valid_fraction,
                        })

                global_step += 1

            avg_loss = float(np.mean(epoch_losses))
            elapsed = time.time() - epoch_start
            self.scheduler.step(avg_loss)

            # Source val eval
            source_val_metrics = {}
            if self.source_val_dataset is not None and epoch % self.eval_every_epochs == 0:
                source_val_metrics = self._eval_source_val()
                if verbose:
                    print(f"  source_val loss={source_val_metrics.get('source_val_loss', float('nan')):.6f}", flush=True)

            # Best checkpoint
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
                "total_loss": avg_loss,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "elapsed_s": elapsed,
            }
            record.update(source_val_metrics)
            self.train_history.append(record)
            if source_val_metrics:
                self.val_history.append({"epoch": epoch, **source_val_metrics})
            self.current_epoch = epoch + 1

            if self._jsonl_logger is not None:
                self._jsonl_logger.log_epoch(record)

            if verbose:
                sv_str = f" sv_loss={source_val_metrics.get('source_val_loss', float('nan')):.6f}" if source_val_metrics else ""
                print(
                    f"Epoch {epoch:3d} | loss={avg_loss:.6f}{sv_str} | "
                    f"lr={record['lr']:.2e} | {elapsed:.1f}s",
                    flush=True,
                )

        return self.train_history

    def save_checkpoint(self, path: Path, epoch: int, loss: float, tag: str = "") -> None:
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
            "prompt_encoder_state_dict": self.prompt_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_history": self.train_history,
            "config": {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "max_epochs": self.max_epochs,
                "batch_size": self.batch_size,
                "grad_clip": self.grad_clip,
                "width": self.model_width,
                "prompt_dim": self.prompt_dim,
                "num_regions": self.prompt_encoder.num_regions,
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
        has_source_val = self.source_val_dataset is not None
        summary = {
            "experiment_id": self.experiment_id,
            "protocol_freeze_id": self.protocol_freeze_id,
            "best_loss": self.best_loss,
            "final_epoch": self.current_epoch - 1,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train prompt-conditioned shared backbone")
    parser.add_argument("--target_region", type=str, required=True)
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--prompt_dim", type=int, default=64)
    parser.add_argument("--zero_raw_increment_init", action="store_true")
    parser.add_argument("--target_increment_normalization", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--require_gpu", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="disabled",
        choices=["disabled", "offline", "online"])
    parser.add_argument("--wandb_project", type=str, default="hydroda-ood")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[])
    parser.add_argument("--log_every_steps", type=int, default=100)
    parser.add_argument("--eval_every_epochs", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    device = resolve_device(args.device, require_gpu=args.require_gpu)

    print("=" * 60)
    print("Phase 4B: Prompt-Conditioned Shared Backbone Training")
    print(f"  target_region={args.target_region}  K={args.K}  seed={args.seed}")
    print(f"  max_epochs={args.max_epochs}  batch_size={args.batch_size}  lr={args.lr}")
    print(f"  device={device}  width={args.width}  prompt_dim={args.prompt_dim}  amp={args.amp}")
    print("=" * 60)

    # Load config
    config = {}
    if args.config and Path(args.config).exists():
        file_config = load_config(args.config)
        for section in ["model", "training", "data", "output"]:
            if section in file_config and isinstance(file_config[section], dict):
                config.update(file_config[section])

    # Run config
    run_config = {
        "target_region": args.target_region, "K": args.K, "seed": args.seed,
        "width": args.width, "prompt_dim": args.prompt_dim,
        "max_epochs": args.max_epochs, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip, "accum_steps": args.accum_steps,
        "num_workers": args.num_workers, "device": str(device),
        "use_amp": args.amp,
        "zero_raw_increment_init": args.zero_raw_increment_init,
        "target_increment_normalization": args.target_increment_normalization,
        "log_every_steps": args.log_every_steps,
        "eval_every_epochs": args.eval_every_epochs,
        "wandb_mode": args.wandb_mode,
        "source_regions": [r for r in _ALL_US_REGIONS if r != args.target_region],
    }

    # RunManager
    run_manager = RunManager(
        phase=PHASE,
        method="prompt_conditioned",
        target_region=args.target_region,
        config=run_config,
        output_dir=args.output_dir,
        run_name=args.run_name,
        width=args.width,
        epochs=args.max_epochs,
        lr=args.lr,
        norm="norm" if args.target_increment_normalization else "nonorm",
        zero_raw=args.zero_raw_increment_init,
        seed=args.seed,
    )
    run_manager.save_config(run_config, "config.yaml")
    run_manager.save_git_info()
    run_manager.save_protocol({
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "split_manifest": FREEZE_MANIFEST,
    })

    # Wandb
    wandb_logger = WandbLogger(
        mode=args.wandb_mode, project=args.wandb_project,
        entity=args.wandb_entity, tags=args.wandb_tags,
        run_name=run_manager.get_run_name(),
    )

    start_time = time.time()

    # Load source_fit dataset (all source regions)
    print(f"\nLoading source_fit dataset...")
    train_dataset = HydroDADataset(
        da_nc_path=DA_NC,
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region=args.target_region,
        split_type="source_fit",
        K=args.K,
        seed=args.seed,
        freeze_manifest=FREEZE_MANIFEST,
    )
    print(f"  source_fit samples: {len(train_dataset)}")
    print(f"  source regions: {run_config['source_regions']}")

    # Build mapping from global region index (0..5) to source-only index (0..num_source-1)
    source_regions = run_config["source_regions"]
    global_to_source_idx = {}
    for src_idx, region_name in enumerate(source_regions):
        global_idx = _GLOBAL_REGION_IDX_MAP[region_name]
        global_to_source_idx[global_idx] = src_idx
    # Also create a tensor-friendly lookup: array of size 6 mapping global_idx -> source_idx (or 0 for target)
    _global_to_source_lookup = [0] * 6
    for global_idx, src_idx in global_to_source_idx.items():
        _global_to_source_lookup[global_idx] = src_idx

    # Load source_val dataset
    print(f"\nLoading source_val dataset...")
    source_val_dataset = HydroDADataset(
        da_nc_path=DA_NC,
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region=args.target_region,
        split_type="source_val",
        K=args.K,
        seed=args.seed,
        freeze_manifest=FREEZE_MANIFEST,
    )
    print(f"  source_val samples: {len(source_val_dataset)}")

    # Init model + prompt encoder
    num_source_regions = len(run_config["source_regions"])
    print(f"\nInitializing FiLMConditionalResUNet (width={args.width}, prompt_dim={args.prompt_dim})...")
    model = FiLMConditionalResUNet(
        in_channels=12, out_channels=2, width=args.width,
        prompt_dim=args.prompt_dim,
        zero_raw_increment_init=args.zero_raw_increment_init,
    )
    prompt_encoder = RegionPromptEncoder(
        num_regions=num_source_regions,
        input_channels=12,
        hidden_dim=args.prompt_dim,
    )

    num_params = sum(p.numel() for p in model.parameters())
    num_pe_params = sum(p.numel() for p in prompt_encoder.parameters())
    print(f"  Model params: {num_params:,}")
    print(f"  Prompt encoder params: {num_pe_params:,}")
    print(f"  Total params: {num_params + num_pe_params:,}")

    # Checkpoint dir
    checkpoint_dir = args.checkpoint_dir or str(run_manager.get_checkpoint_dir())

    # Create trainer
    trainer = PromptConditionedTrainer(
        model=model,
        prompt_encoder=prompt_encoder,
        train_dataset=train_dataset,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=str(device),
        checkpoint_dir=checkpoint_dir,
        experiment_id=run_manager.get_run_name(),
        protocol_freeze_id=PROTOCOL_FREEZE_ID,
        split_manifest_path=FREEZE_MANIFEST,
        grad_clip=args.grad_clip,
        model_width=args.width,
        prompt_dim=args.prompt_dim,
        target_increment_normalization=args.target_increment_normalization,
        zero_raw_increment_init=args.zero_raw_increment_init,
        accum_steps=args.accum_steps,
        run_manager=run_manager,
        use_amp=args.amp,
        log_every_steps=args.log_every_steps,
        eval_every_epochs=args.eval_every_epochs,
        wandb_logger=wandb_logger,
        source_val_dataset=source_val_dataset,
        global_to_source_lookup=_global_to_source_lookup,
    )

    run_manager.save_environment_info(gather_runtime_info())

    # Train
    print(f"\nStarting training...")
    history = trainer.train(verbose=True)

    elapsed = time.time() - start_time

    # Save summary
    summary_path = run_manager.summary_json_path()
    trainer.save_summary_json(summary_path)

    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  best_loss={trainer.best_loss:.6f}")
    print(f"  run_dir={run_manager.get_run_dir()}")
    print(f"  summary={summary_path}")
    print(f"  best_checkpoint={run_manager.checkpoint_best_path()}")
    print(f"  last_checkpoint={run_manager.checkpoint_last_path()}")

    if wandb_logger.enabled:
        wandb_logger.finish()

    # Save history
    history_path = run_manager.get_results_dir() / "train_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
