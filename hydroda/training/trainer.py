"""Trainer for source-only backbone in HydroDA-OOD / HyperDA V4.

No-leakage declaration:
    - Training uses source_fit split only (2015-2021, US-R1..R6 excluding target)
    - Normalization stats computed from source_fit only (LeakageGuard check)
    - No target_eval/query labels used in training / normalization / early_stopping
    - No target prompt used
"""
from __future__ import annotations

import gc
import json
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from hydroda.data.dataset import HydroDADataset, collate_hydroda_samples
from hydroda.data.file_hash import compute_sha256
from hydroda.data.leakage_guard import LeakageGuard
from hydroda.data.protocol import ProtocolConfig
from hydroda.training.calibration import calibrate_residual_gain
from hydroda.training.domain_generalization import (
    InputOnlyTargetContextDataset,
    SAMSharpnessPerturbation,
    SelfBootstrapAugmentation,
    SSARegState,
    SWADState,
    collate_input_only_samples,
    coral_loss,
    domain_loss_variance,
    domain_masked_huber_losses,
    identify_unlearn_loss,
    moment_alignment_loss,
    prediction_consistency_loss,
    region_identify_unlearn_loss,
    region_masked_huber_losses,
    region_moment_alignment_loss,
    subspace_alignment_loss,
    tca_correlation_alignment_loss,
    unknown_domain_inconsistency_loss,
)
from hydroda.training.losses import MaskedHuberLoss, WeightedMaskedHuberLoss
from hydroda.utils.device import gpu_health_check
from hydroda.utils.run_manager import RunManager
from hydroda.utils.logger import WandbLogger
from hydroda.utils.runtime import get_git_hash, get_timestamp


@dataclass(frozen=True)
class CheckpointSelectionDecision:
    is_best: bool
    best_metric: float
    metric_name: str


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
        for physical_sample in _iter_physical_samples(sample):
            x = physical_sample["x"]  # (12, H, W)
            valid = np.isfinite(x)
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
        for physical_sample in _iter_physical_samples(sample):
            x = physical_sample["x"]
            valid = np.isfinite(x)
            for c in range(12):
                channel_counts[c] += valid[c].sum()

    means = sums / np.maximum(channel_counts, 1.0)
    variances = (sq_sums / np.maximum(channel_counts, 1.0)) - (means ** 2)
    variances = np.maximum(variances, 0.0)
    stds = np.sqrt(variances) + 1e-6

    return means.astype(np.float32), stds.astype(np.float32)


def _iter_physical_samples(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [sample]


class Trainer:
    """Trainer for source-only SmallResUNet backbone.

    Handles:
    - DataLoader construction from source_train split
    - Optimization (AdamW, weight_decay)
    - Training loop with loss recording
    - Lat-weighted loss + residual gain calibration on source_val
    - Checkpoint saving (best.pt, last.pt, safe_score, min_skill, epoch snapshots)
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
        use_lat_weighted_loss: use WeightedMaskedHuberLoss (default True)
        checkpoint_every_n_epochs: save epoch checkpoints every N epochs (default 5)
        source_val_gain_grid: list of alpha values for residual gain calibration
        lambda_amp: amplitude penalty weight (0=disabled)
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
        protocol_freeze_id: str = "hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025",
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
        use_lat_weighted_loss: bool = True,
        checkpoint_every_n_epochs: int = 5,
        selection_metric: str = "source_val_safe_score",
        source_val_gain_grid: Optional[List[float]] = None,
        lambda_amp: float = 0.0,
        cuda_sync_debug: bool = False,
        dg_method: str = "none",
        coral_lambda: float = 0.0,
        coral_feature_layer: str = "bottleneck",
        tca_lambda: float = 0.0,
        tca_feature_layer: str = "bottleneck",
        ssa_reg_lambda: float = 0.0,
        ssa_reg_feature_layer: str = "bottleneck",
        ssa_reg_rank: int = 8,
        self_bootstrap_lambda: float = 0.0,
        self_bootstrap_noise_std: float = 0.01,
        self_bootstrap_channel_dropout_p: float = 0.05,
        disam_rho: float = 0.05,
        disam_lambda: float = 0.1,
        udim_rho: float = 0.05,
        udim_lambda: float = 0.1,
        moment_align_lambda: float = 0.01,
        moment_align_feature_layer: str = "bottleneck",
        moment_align_order: int = 2,
        iu_lambda: float = 0.001,
        iu_feature_layer: str = "bottleneck",
        iu_top_fraction: float = 0.25,
        iu_sample_top_fraction: float = 0.5,
        iu_score_cap: float = 10.0,
        target_context_dataset: Optional[HydroDADataset] = None,
        target_context_batch_size: int = 16,
        swad_start_epoch: int = 10,
        swad_tolerance: float = 0.02,
        swad_patience: int = 3,
        extra_checkpoint_metadata: Optional[Dict[str, Any]] = None,
        # Resume: optionally inject pre-computed normalization stats to skip recompute
        _resume_ch_mean: Optional[np.ndarray] = None,
        _resume_ch_std: Optional[np.ndarray] = None,
        _resume_inc_mean: Optional[np.ndarray] = None,
        _resume_inc_std: Optional[np.ndarray] = None,
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
        self.split_manifest_sha256 = ""
        split_manifest_file = Path(split_manifest_path)
        if split_manifest_file.exists():
            self.split_manifest_sha256 = compute_sha256(split_manifest_file)
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
        self.cuda_sync_debug = cuda_sync_debug
        self.use_lat_weighted_loss = use_lat_weighted_loss
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        if selection_metric not in {"source_val_safe_score", "source_val_loss"}:
            raise ValueError(
                "selection_metric must be 'source_val_safe_score' or 'source_val_loss', "
                f"got {selection_metric!r}"
            )
        self.selection_metric = selection_metric
        self.source_val_gain_grid = source_val_gain_grid or [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        self.lambda_amp = lambda_amp
        self.dg_method = str(dg_method or "none")
        self.coral_lambda = float(coral_lambda)
        self.coral_feature_layer = str(coral_feature_layer)
        self.tca_lambda = float(tca_lambda)
        self.tca_feature_layer = str(tca_feature_layer)
        self.ssa_reg_lambda = float(ssa_reg_lambda)
        self.ssa_reg_feature_layer = str(ssa_reg_feature_layer)
        self.ssa_reg_rank = int(ssa_reg_rank)
        self.self_bootstrap_lambda = float(self_bootstrap_lambda)
        self.self_bootstrap_noise_std = float(self_bootstrap_noise_std)
        self.self_bootstrap_channel_dropout_p = float(self_bootstrap_channel_dropout_p)
        self.disam_rho = float(disam_rho)
        self.disam_lambda = float(disam_lambda)
        self.udim_rho = float(udim_rho)
        self.udim_lambda = float(udim_lambda)
        self.moment_align_lambda = float(moment_align_lambda)
        self.moment_align_feature_layer = str(moment_align_feature_layer)
        self.moment_align_order = int(moment_align_order)
        self.iu_lambda = float(iu_lambda)
        self.iu_feature_layer = str(iu_feature_layer)
        self.iu_top_fraction = float(iu_top_fraction)
        self.iu_sample_top_fraction = float(iu_sample_top_fraction)
        self.iu_score_cap = float(iu_score_cap)
        self.target_context_batch_size = int(target_context_batch_size)
        self.extra_checkpoint_metadata = dict(extra_checkpoint_metadata or {})
        self.target_context_dataset = None
        if self.dg_method in {"deep_coral", "tca", "ssa_reg", "self_bootstrap"}:
            if target_context_dataset is None:
                raise ValueError(f"dg_method={self.dg_method!r} requires a target_context_dataset")
            method_name = {
                "deep_coral": "Deep CORAL",
                "tca": "TCA",
                "ssa_reg": "SSA-Reg",
                "self_bootstrap": "Self-Bootstrap",
            }[self.dg_method]
            self.target_context_dataset = (
                target_context_dataset
                if isinstance(target_context_dataset, InputOnlyTargetContextDataset)
                else InputOnlyTargetContextDataset(target_context_dataset, method_name=method_name)
            )
        self.self_bootstrap_augmentation = (
            SelfBootstrapAugmentation(
                noise_std=self.self_bootstrap_noise_std,
                channel_dropout_p=self.self_bootstrap_channel_dropout_p,
            )
            if self.dg_method == "self_bootstrap"
            else None
        )
        self.ssa_reg_state = (
            SSARegState(
                rank=self.ssa_reg_rank,
                lambda_align=self.ssa_reg_lambda,
                feature_layer=self.ssa_reg_feature_layer,
            )
            if self.dg_method == "ssa_reg"
            else None
        )
        self.swad_state = (
            SWADState(
                start_epoch=swad_start_epoch,
                tolerance=swad_tolerance,
                patience=swad_patience,
                mode="min",
            )
            if self.dg_method == "swad"
            else None
        )
        self.sam_perturbation = (
            SAMSharpnessPerturbation(
                self.model,
                rho=self.udim_rho if self.dg_method == "udim" else self.disam_rho,
            )
            if self.dg_method in {"disam", "udim"}
            else None
        )

        # AMP scaler
        self._amp_scaler: Optional[GradScaler] = None
        if self.use_amp:
            self._amp_scaler = GradScaler('cuda', init_scale=256.0)

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
        # Loss function selection
        if self.use_lat_weighted_loss:
            self.loss_fn = WeightedMaskedHuberLoss(
                delta=0.01,
                lambda_amp=self.lambda_amp,
            )
        else:
            self.loss_fn = MaskedHuberLoss(delta=0.01)

        # Compute normalization stats from source_train (or restore from checkpoint on resume)
        self._ch_mean: Optional[np.ndarray] = None
        self._ch_std: Optional[np.ndarray] = None
        if _resume_ch_mean is not None and _resume_ch_std is not None:
            self._ch_mean = _resume_ch_mean
            self._ch_std = _resume_ch_std
            print(f"  [resume] Restored ch_mean from checkpoint")
        else:
            self._compute_normalization_stats()

        # Increment normalization stats (for target increments)
        self._inc_mean: Optional[np.ndarray] = None
        self._inc_std: Optional[np.ndarray] = None
        if self.target_increment_normalization:
            if _resume_inc_mean is not None and _resume_inc_std is not None:
                self._inc_mean = _resume_inc_mean
                self._inc_std = _resume_inc_std
                print(f"  [resume] Restored inc_mean/inc_std from checkpoint")
            else:
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
        self.best_safe_score = float("-inf")
        self._skipped_steps = 0
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
            for physical_sample in _iter_physical_samples(sample):
                inc_s = physical_sample["increment_surface"]
                inc_r = physical_sample["increment_rootzone"]
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
            return collate_hydroda_samples(batch)

        pin_mem = self.device == "cuda"
        return DataLoader(
            target_dataset,
            batch_size=self.batch_size,
            shuffle=(dataset is None),
            num_workers=self.num_workers,
            pin_memory=pin_mem,
            collate_fn=collate_fn,
        )

    def _build_target_context_dataloader(self) -> Optional[DataLoader]:
        if self.target_context_dataset is None:
            return None
        pin_mem = self.device == "cuda"
        return DataLoader(
            self.target_context_dataset,
            batch_size=self.target_context_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_mem,
            collate_fn=collate_input_only_samples,
        )

    def _forward_and_loss(
        self,
        x_norm: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        latitude_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass + loss, handling AMP consistently.

        Returns (pred, losses_dict).
        """
        if self.use_amp:
            with autocast('cuda'):
                pred = self.model(x_norm)
        else:
            pred = self.model(x_norm)

        # Cast to fp32 for numerical stability in loss
        pred = pred.float()

        if self.use_lat_weighted_loss:
            if latitude_weight is None:
                raise ValueError(
                    "use_lat_weighted_loss=True but latitude_weight not provided in batch. "
                    "Ensure dataset returns latitude_weight."
                )
            losses = self.loss_fn(pred, target, loss_mask, latitude_weight=latitude_weight)
        else:
            losses = self.loss_fn(pred, target, loss_mask)
        return pred, losses

    def _forward_and_loss_fp32(
        self,
        x_norm: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        latitude_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        use_amp = self.use_amp
        self.use_amp = False
        try:
            return self._forward_and_loss(x_norm, target, loss_mask, latitude_weight=latitude_weight)
        finally:
            self.use_amp = use_amp

    def _add_disam_domain_variance(
        self,
        losses: Dict[str, torch.Tensor],
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        sample_region_ids: Optional[List[str]],
        region_mask_integer: Optional[torch.Tensor] = None,
        active_region_ids: Optional[List[Any]] = None,
        latitude_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        zero = losses["total_loss"].new_zeros(())
        if self.dg_method != "disam" or self.disam_lambda <= 0.0:
            losses["disam_loss_variance"] = zero
            return losses
        if region_mask_integer is not None:
            per_domain_losses = region_masked_huber_losses(
                pred,
                target,
                loss_mask,
                region_mask_integer,
                active_region_ids=active_region_ids,
                latitude_weight=latitude_weight if self.use_lat_weighted_loss else None,
                delta=float(getattr(self.loss_fn, "delta", 0.01)),
            )
        elif sample_region_ids:
            per_domain_losses = domain_masked_huber_losses(
                pred,
                target,
                loss_mask,
                sample_region_ids,
                latitude_weight=latitude_weight if self.use_lat_weighted_loss else None,
                delta=float(getattr(self.loss_fn, "delta", 0.01)),
            )
        else:
            per_domain_losses = {}
        if len(per_domain_losses) < 2:
            losses["disam_loss_variance"] = zero
            return losses
        variance_loss = domain_loss_variance(per_domain_losses).to(losses["total_loss"].device)
        losses["disam_loss_variance"] = variance_loss
        losses["total_loss"] = losses["total_loss"] + self.disam_lambda * variance_loss
        return losses

    def _compute_source_domain_losses(
        self,
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        sample_region_ids: Optional[List[str]],
        region_mask_integer: Optional[torch.Tensor] = None,
        active_region_ids: Optional[List[Any]] = None,
        latitude_weight: Optional[torch.Tensor] = None,
    ) -> Dict[Any, torch.Tensor]:
        """Compute source-domain losses from spatial region masks or sample IDs."""
        if region_mask_integer is not None:
            return region_masked_huber_losses(
                pred,
                target,
                loss_mask,
                region_mask_integer,
                active_region_ids=active_region_ids,
                latitude_weight=latitude_weight if self.use_lat_weighted_loss else None,
                delta=float(getattr(self.loss_fn, "delta", 0.01)),
            )
        if sample_region_ids:
            return domain_masked_huber_losses(
                pred,
                target,
                loss_mask,
                sample_region_ids,
                latitude_weight=latitude_weight if self.use_lat_weighted_loss else None,
                delta=float(getattr(self.loss_fn, "delta", 0.01)),
            )
        return {}

    def _add_udim_inconsistency(
        self,
        losses: Dict[str, torch.Tensor],
        *,
        clean_domain_losses: Dict[Any, torch.Tensor],
        perturbed_domain_losses: Dict[Any, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        zero = losses["total_loss"].new_zeros(())
        if self.dg_method != "udim" or self.udim_lambda <= 0.0:
            losses["udim_inconsistency_loss"] = zero
            return losses
        if len(set(clean_domain_losses).intersection(perturbed_domain_losses)) < 2:
            losses["udim_inconsistency_loss"] = zero
            return losses
        inconsistency = unknown_domain_inconsistency_loss(clean_domain_losses, perturbed_domain_losses).to(
            losses["total_loss"].device
        )
        losses["udim_inconsistency_loss"] = inconsistency
        losses["total_loss"] = losses["total_loss"] + self.udim_lambda * inconsistency
        return losses

    def _add_moment_alignment(
        self,
        losses: Dict[str, torch.Tensor],
        *,
        x_norm: torch.Tensor,
        sample_region_ids: Optional[List[str]],
        region_mask_integer: Optional[torch.Tensor] = None,
        active_region_ids: Optional[List[Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        zero = losses["total_loss"].new_zeros(())
        if self.dg_method != "moment_align" or self.moment_align_lambda <= 0.0:
            losses["moment_align_loss"] = zero
            return losses
        features = self.model.forward_features(x_norm, return_layer=self.moment_align_feature_layer)
        if region_mask_integer is not None:
            moment_loss = region_moment_alignment_loss(
                features,
                region_mask_integer,
                active_region_ids=active_region_ids,
                order=self.moment_align_order,
            )
        elif sample_region_ids:
            moment_loss = moment_alignment_loss(
                features,
                sample_region_ids,
                order=self.moment_align_order,
            )
        else:
            moment_loss = zero
        losses["moment_align_loss"] = moment_loss
        losses["total_loss"] = losses["total_loss"] + self.moment_align_lambda * moment_loss
        return losses

    def _add_identify_unlearn(
        self,
        losses: Dict[str, torch.Tensor],
        *,
        x_norm: torch.Tensor,
        sample_region_ids: Optional[List[str]],
        region_mask_integer: Optional[torch.Tensor] = None,
        active_region_ids: Optional[List[Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        zero = losses["total_loss"].new_zeros(())
        if self.dg_method != "iu" or self.iu_lambda <= 0.0:
            losses["iu_unlearn_loss"] = zero
            return losses
        features = self.model.forward_features(x_norm, return_layer=self.iu_feature_layer)
        if region_mask_integer is not None:
            unlearn_loss = region_identify_unlearn_loss(
                features,
                region_mask_integer,
                active_region_ids=active_region_ids,
                top_fraction=self.iu_top_fraction,
                sample_top_fraction=self.iu_sample_top_fraction,
                score_cap=self.iu_score_cap,
            )
        elif sample_region_ids:
            unlearn_loss = identify_unlearn_loss(
                features,
                sample_region_ids,
                top_fraction=self.iu_top_fraction,
                sample_top_fraction=self.iu_sample_top_fraction,
                score_cap=self.iu_score_cap,
            )
        else:
            unlearn_loss = zero
        losses["iu_unlearn_loss"] = unlearn_loss
        losses["total_loss"] = losses["total_loss"] + self.iu_lambda * unlearn_loss
        return losses

    def _disam_two_step_update(
        self,
        *,
        x_norm: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        latitude_weight: Optional[torch.Tensor],
        sample_region_ids: Optional[List[str]],
        region_mask_integer: Optional[torch.Tensor] = None,
        active_region_ids: Optional[List[Any]] = None,
    ) -> Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Run one DISAM SAM-style optimizer update and return perturbed losses."""
        if self.sam_perturbation is None:
            raise RuntimeError("DISAM step requested without SAMSharpnessPerturbation")

        pred, losses = self._forward_and_loss_fp32(x_norm, target, loss_mask, latitude_weight=latitude_weight)
        losses = self._add_disam_domain_variance(
            losses,
            pred=pred,
            target=target,
            loss_mask=loss_mask,
            sample_region_ids=sample_region_ids,
            region_mask_integer=region_mask_integer,
            active_region_ids=active_region_ids,
            latitude_weight=latitude_weight,
        )
        if torch.isnan(losses["total_loss"]) or torch.isinf(losses["total_loss"]):
            return None

        losses["total_loss"].backward()
        sam_grad_norm = self.sam_perturbation.perturb()
        self.optimizer.zero_grad()

        pred, losses = self._forward_and_loss_fp32(x_norm, target, loss_mask, latitude_weight=latitude_weight)
        losses = self._add_disam_domain_variance(
            losses,
            pred=pred,
            target=target,
            loss_mask=loss_mask,
            sample_region_ids=sample_region_ids,
            region_mask_integer=region_mask_integer,
            active_region_ids=active_region_ids,
            latitude_weight=latitude_weight,
        )
        losses["disam_sam_grad_norm"] = sam_grad_norm.detach()
        if torch.isnan(losses["total_loss"]) or torch.isinf(losses["total_loss"]):
            self.sam_perturbation.restore()
            self.optimizer.zero_grad()
            return None
        losses["total_loss"].backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.sam_perturbation.restore()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return pred, losses

    def _udim_two_step_update(
        self,
        *,
        x_norm: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        latitude_weight: Optional[torch.Tensor],
        sample_region_ids: Optional[List[str]],
        region_mask_integer: Optional[torch.Tensor] = None,
        active_region_ids: Optional[List[Any]] = None,
    ) -> Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Run one UDIM source-only unknown-domain inconsistency optimizer update."""
        if self.sam_perturbation is None:
            raise RuntimeError("UDIM step requested without SAMSharpnessPerturbation")

        pred, losses = self._forward_and_loss_fp32(x_norm, target, loss_mask, latitude_weight=latitude_weight)
        clean_domain_losses = self._compute_source_domain_losses(
            pred=pred,
            target=target,
            loss_mask=loss_mask,
            sample_region_ids=sample_region_ids,
            region_mask_integer=region_mask_integer,
            active_region_ids=active_region_ids,
            latitude_weight=latitude_weight,
        )
        clean_domain_losses = {key: value.detach() for key, value in clean_domain_losses.items()}
        if torch.isnan(losses["total_loss"]) or torch.isinf(losses["total_loss"]):
            return None

        losses["total_loss"].backward()
        sam_grad_norm = self.sam_perturbation.perturb()
        self.optimizer.zero_grad()

        pred, losses = self._forward_and_loss_fp32(x_norm, target, loss_mask, latitude_weight=latitude_weight)
        perturbed_domain_losses = self._compute_source_domain_losses(
            pred=pred,
            target=target,
            loss_mask=loss_mask,
            sample_region_ids=sample_region_ids,
            region_mask_integer=region_mask_integer,
            active_region_ids=active_region_ids,
            latitude_weight=latitude_weight,
        )
        losses = self._add_udim_inconsistency(
            losses,
            clean_domain_losses=clean_domain_losses,
            perturbed_domain_losses=perturbed_domain_losses,
        )
        losses["udim_sam_grad_norm"] = sam_grad_norm.detach()
        if torch.isnan(losses["total_loss"]) or torch.isinf(losses["total_loss"]):
            self.sam_perturbation.restore()
            self.optimizer.zero_grad()
            return None
        losses["total_loss"].backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.sam_perturbation.restore()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return pred, losses

    @staticmethod
    def _make_source_val_metrics(gain_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract a flat metrics dict from gain calibration results."""
        return {
            "source_val_loss": gain_results["source_val_loss"],
            "source_val_surface_loss": gain_results.get("source_val_surface_loss", float("nan")),
            "source_val_rootzone_loss": gain_results.get("source_val_rootzone_loss", float("nan")),
            "source_val_valid_px": gain_results.get("source_val_valid_px", 0),
            "source_val_rmse_surface": gain_results["rmse_surface_model"],
            "source_val_rmse_rootzone": gain_results["rmse_rootzone_model"],
            "source_val_skill_surface": gain_results["skill_surface_with_alpha"],
            "source_val_skill_rootzone": gain_results["skill_rootzone_with_alpha"],
        }

    def _resolve_checkpoint_selection(
        self,
        *,
        avg_loss: float,
        source_val_metrics: Dict[str, float],
        gain_results: Optional[Dict[str, Any]],
    ) -> CheckpointSelectionDecision:
        if self.selection_metric == "source_val_loss" and source_val_metrics and "source_val_loss" in source_val_metrics:
            sv_loss = float(source_val_metrics["source_val_loss"])
            return CheckpointSelectionDecision(
                is_best=sv_loss < self.best_loss,
                best_metric=sv_loss,
                metric_name="source_val_loss",
            )
        if self.selection_metric == "source_val_safe_score" and gain_results and "selection_score" in gain_results:
            safe_score = float(gain_results["selection_score"])
            return CheckpointSelectionDecision(
                is_best=safe_score > self.best_safe_score,
                best_metric=safe_score,
                metric_name="source_val_safe_score",
            )
        if source_val_metrics and "source_val_loss" in source_val_metrics:
            sv_loss = float(source_val_metrics["source_val_loss"])
            return CheckpointSelectionDecision(
                is_best=sv_loss < self.best_loss,
                best_metric=sv_loss,
                metric_name="source_val_loss",
            )
        return CheckpointSelectionDecision(
            is_best=float(avg_loss) < self.best_loss,
            best_metric=float(avg_loss),
            metric_name="train_loss",
        )

    def _eval_source_val(self) -> Dict[str, float]:
        """Run evaluation on source_val split and return metrics dict.

        Kept for backward compatibility — delegates to gain calibration
        and returns simplified dict. Prefer _calibrate_source_val_residual_gain
        for full results.
        """
        gain_results = self._calibrate_source_val_residual_gain()
        if not gain_results:
            return {}
        return self._make_source_val_metrics(gain_results)

    def _calibrate_source_val_residual_gain(self) -> Dict[str, Any]:
        """Calibrate residual gain alphas on source_val using latitude weighting.

        Iterates over source_val, accumulates per-sample arrays (pred_inc,
        true_inc, forecast, mask, latw) for surface and rootzone. Then scans
        alpha grid, computing analysis skill for each alpha as:

            skill(alpha) = 1 - sqrt(mean(model_mse_alpha)) / sqrt(mean(forecast_mse))

        where mean is over time samples.

        Selection: primary = max(min_skill), tie-break = max(mean_skill),
        tie-break = min(mean_rmse_ratio). alpha=0 is always in the candidate set.

        Returns dict with best_alpha, skills, RMSEs, per_alpha_results, etc.
        """
        if self.source_val_dataset is None:
            return {}

        self.model.eval()
        loader = self._build_dataloader(self.source_val_dataset)
        alphas = self.source_val_gain_grid

        # Per-sample storage for post-hoc alpha scan
        samples_s = []  # each: (pred_inc, true_inc, forecast, mask, latw)
        samples_r = []
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
                latitude_weight = batch.get("latitude_weight")
                if latitude_weight is not None:
                    latitude_weight = latitude_weight.to(self.device)
                forecast_s = batch.get("forecast_surface")
                forecast_r = batch.get("forecast_rootzone")
                if forecast_s is not None:
                    forecast_s = forecast_s.to(self.device)
                if forecast_r is not None:
                    forecast_r = forecast_r.to(self.device)

                x_norm = self._normalize(x)
                target = torch.stack([inc_surface, inc_rootzone], dim=1)
                if self.target_increment_normalization and self._inc_mean is not None:
                    inc_mean_t = torch.from_numpy(self._inc_mean).to(x.device).view(1, 2, 1, 1)
                    inc_std_t = torch.from_numpy(self._inc_std).to(x.device).view(1, 2, 1, 1)
                    target = (target - inc_mean_t) / inc_std_t

                pred, losses = self._forward_and_loss(x_norm, target, loss_mask, latitude_weight=latitude_weight)

                total_loss += float(losses["total_loss"].item())
                total_surface += float(losses.get("surface_loss", 0.0))
                total_rootzone += float(losses.get("rootzone_loss", 0.0))
                total_valid += int(
                    losses.get("valid_pixel_count", losses.get("valid_weight_sum", 0)).item()
                )
                n_batches += 1

                # Denormalize pred for physical-space
                pred_denorm_s = pred[:, 0].float()
                pred_denorm_r = pred[:, 1].float()
                if self.target_increment_normalization and self._inc_mean is not None:
                    pred_denorm_s = pred_denorm_s * self._inc_std[0] + self._inc_mean[0]
                    pred_denorm_r = pred_denorm_r * self._inc_std[1] + self._inc_mean[1]

                # Accumulate per-sample numpy arrays for post-hoc alpha scan
                for b in range(x.size(0)):
                    # Squeeze channel dim from mask if present: [1, H, W] or [H, W] -> [H, W]
                    mask_b = loss_mask[b]
                    if mask_b.ndim == 3:
                        mask_b = mask_b.squeeze(0)
                    mask_np = mask_b.cpu().numpy().astype(np.float32)
                    latw_np = (
                        latitude_weight[b].cpu().numpy().astype(np.float32)
                        if latitude_weight is not None
                        else np.ones(mask_np.shape, dtype=np.float32)
                    )

                    pred_inc_s = pred_denorm_s[b].cpu().numpy().astype(np.float32)
                    true_inc_s = inc_surface[b].cpu().numpy().astype(np.float32)
                    fcst_s = (
                        forecast_s[b].cpu().numpy().astype(np.float32)
                        if forecast_s is not None
                        else np.zeros_like(true_inc_s, dtype=np.float32)
                    )
                    samples_s.append((pred_inc_s, true_inc_s, fcst_s, mask_np, latw_np))

                    pred_inc_r = pred_denorm_r[b].cpu().numpy().astype(np.float32)
                    true_inc_r = inc_rootzone[b].cpu().numpy().astype(np.float32)
                    fcst_r = (
                        forecast_r[b].cpu().numpy().astype(np.float32)
                        if forecast_r is not None
                        else np.zeros_like(true_inc_r, dtype=np.float32)
                    )
                    samples_r.append((pred_inc_r, true_inc_r, fcst_r, mask_np, latw_np))

        self.model.train()

        if not samples_s:
            return {}

        # Delegate alpha scan to shared calibration function
        gain_results = calibrate_residual_gain(samples_s, samples_r, alphas)
        gain_results["source_val_loss"] = total_loss / max(n_batches, 1)
        gain_results["source_val_surface_loss"] = total_surface / max(n_batches, 1)
        gain_results["source_val_rootzone_loss"] = total_rootzone / max(n_batches, 1)
        gain_results["source_val_valid_px"] = total_valid
        return gain_results

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
            f"  Loss fn:         {type(self.loss_fn).__name__}",
            f"  Lat-weighted:    {self.use_lat_weighted_loss}",
            f"  Batch size:      {self.batch_size}",
            f"  Accum steps:     {self.accum_steps}",
            f"  Max epochs:      {self.max_epochs}",
            f"  Checkpoint every:{self.checkpoint_every_n_epochs} epochs",
            f"  Selection metric:{self.selection_metric}",
            f"  LR:              {self.lr}",
            f"  Weight decay:    {self.weight_decay}",
            f"  Grad clip:       {self.grad_clip}",
            f"  AMP:             {self.use_amp}",
            f"  Inc norm:        {self.target_increment_normalization}",
            f"  Zero raw init:   {self.zero_raw_increment_init}",
            f"  Lambda amp:      {self.lambda_amp}",
            f"  DG method:       {self.dg_method}",
            f"  Train samples:   {len(self.train_dataset)}",
            f"  Steps/epoch:     {total_steps_per_epoch}",
        ]
        if self.dg_method == "deep_coral":
            header_lines.append(f"  CORAL lambda:    {self.coral_lambda}")
            header_lines.append(f"  CORAL layer:     {self.coral_feature_layer}")
            header_lines.append(f"  Target context:  {len(self.target_context_dataset)} input-only samples")
        if self.dg_method == "tca":
            header_lines.append(f"  TCA lambda:      {self.tca_lambda}")
            header_lines.append(f"  TCA layer:       {self.tca_feature_layer}")
            header_lines.append(f"  Target context:  {len(self.target_context_dataset)} input-only samples")
        if self.dg_method == "ssa_reg":
            header_lines.append(f"  SSA-Reg lambda:  {self.ssa_reg_lambda}")
            header_lines.append(f"  SSA-Reg layer:   {self.ssa_reg_feature_layer}")
            header_lines.append(f"  SSA-Reg rank:    {self.ssa_reg_rank}")
            header_lines.append(f"  Target context:  {len(self.target_context_dataset)} input-only samples")
        if self.dg_method == "self_bootstrap":
            header_lines.append(f"  Self-bootstrap lambda: {self.self_bootstrap_lambda}")
            header_lines.append(f"  Self-bootstrap noise:  {self.self_bootstrap_noise_std}")
            header_lines.append(f"  Self-bootstrap chdrop: {self.self_bootstrap_channel_dropout_p}")
            header_lines.append(f"  Target context:        {len(self.target_context_dataset)} input-only samples")
        if self.dg_method == "disam":
            header_lines.append(f"  DISAM rho:       {self.disam_rho}")
            header_lines.append(f"  DISAM lambda:    {self.disam_lambda}")
            header_lines.append("  Target context:  not used")
        if self.dg_method == "udim":
            header_lines.append(f"  UDIM rho:        {self.udim_rho}")
            header_lines.append(f"  UDIM lambda:     {self.udim_lambda}")
            header_lines.append("  UDIM objective:  source-only unknown-domain inconsistency")
            header_lines.append("  Target context:  not used")
        if self.dg_method == "moment_align":
            header_lines.append(f"  Moment lambda:   {self.moment_align_lambda}")
            header_lines.append(f"  Moment layer:    {self.moment_align_feature_layer}")
            header_lines.append(f"  Moment order:    {self.moment_align_order}")
            header_lines.append("  Target context:  not used")
        if self.dg_method == "iu":
            header_lines.append(f"  IU lambda:       {self.iu_lambda}")
            header_lines.append(f"  IU layer:        {self.iu_feature_layer}")
            header_lines.append(f"  IU top frac:     {self.iu_top_fraction}")
            header_lines.append(f"  IU sample frac:  {self.iu_sample_top_fraction}")
            header_lines.append(f"  IU score cap:    {self.iu_score_cap}")
            header_lines.append("  IU objective:    bounded domain-specific feature penalty")
            header_lines.append("  Target context:  not used")
        if self.swad_state is not None:
            header_lines.append(
                f"  SWAD:            start={self.swad_state.start_epoch} "
                f"tol={self.swad_state.tolerance} patience={self.swad_state.patience}"
            )
        if self.target_increment_normalization and self._inc_mean is not None:
            header_lines.append(f"  inc_mean:        s={self._inc_mean[0]:.6f} r={self._inc_mean[1]:.6f}")
            header_lines.append(f"  inc_std:         s={self._inc_std[0]:.6f} r={self._inc_std[1]:.6f}")
        header_lines.append("=" * 60)
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

        target_context_loader = self._build_target_context_dataloader()
        target_context_iter = iter(target_context_loader) if target_context_loader is not None else None

        for epoch in range(self.current_epoch, self.max_epochs):
            epoch_losses = []
            epoch_surface_losses = []
            epoch_rootzone_losses = []
            epoch_coral_losses = []
            epoch_tca_losses = []
            epoch_ssa_reg_losses = []
            epoch_self_bootstrap_losses = []
            epoch_disam_losses = []
            epoch_udim_losses = []
            epoch_moment_align_losses = []
            epoch_iu_losses = []
            epoch_valid_counts = []
            epoch_start = time.time()
            batches_since_eval = 0

            # Zero gradients at start of epoch (gradient accumulation fix)
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(dataloader):
                if self.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                x = batch["x"].to(self.device)
                inc_surface = batch["increment_surface"].to(self.device)
                inc_rootzone = batch["increment_rootzone"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)
                latitude_weight = batch.get("latitude_weight")
                if latitude_weight is not None:
                    latitude_weight = latitude_weight.to(self.device)

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

                sample_region_ids = batch.get("sample_region_id")
                region_mask_integer = batch.get("region_mask_integer")
                if region_mask_integer is not None:
                    region_mask_integer = region_mask_integer.to(self.device)
                active_region_ids = batch.get("active_region_ids")

                # Forward pass + loss (AMP handled in _forward_and_loss)
                if self.dg_method == "disam":
                    step_result = self._disam_two_step_update(
                        x_norm=x_norm,
                        target=target,
                        loss_mask=loss_mask,
                        latitude_weight=latitude_weight,
                        sample_region_ids=sample_region_ids,
                        region_mask_integer=region_mask_integer,
                        active_region_ids=active_region_ids,
                    )
                    if step_result is None:
                        print(f"  WARNING: E{epoch} S{batch_idx}: DISAM loss is NaN/Inf — skipping batch", flush=True)
                        continue
                    pred, losses = step_result
                    losses["coral_loss"] = torch.tensor(0.0, device=x.device)
                    losses["tca_loss"] = torch.tensor(0.0, device=x.device)
                    losses["ssa_reg_loss"] = torch.tensor(0.0, device=x.device)
                    losses["self_bootstrap_loss"] = torch.tensor(0.0, device=x.device)
                    losses["udim_inconsistency_loss"] = torch.tensor(0.0, device=x.device)
                    losses["moment_align_loss"] = torch.tensor(0.0, device=x.device)
                    losses["iu_unlearn_loss"] = torch.tensor(0.0, device=x.device)
                    epoch_disam_losses.append(float(losses["disam_loss_variance"].detach().item()))
                elif self.dg_method == "udim":
                    step_result = self._udim_two_step_update(
                        x_norm=x_norm,
                        target=target,
                        loss_mask=loss_mask,
                        latitude_weight=latitude_weight,
                        sample_region_ids=sample_region_ids,
                        region_mask_integer=region_mask_integer,
                        active_region_ids=active_region_ids,
                    )
                    if step_result is None:
                        print(f"  WARNING: E{epoch} S{batch_idx}: UDIM loss is NaN/Inf — skipping batch", flush=True)
                        continue
                    pred, losses = step_result
                    losses["coral_loss"] = torch.tensor(0.0, device=x.device)
                    losses["tca_loss"] = torch.tensor(0.0, device=x.device)
                    losses["ssa_reg_loss"] = torch.tensor(0.0, device=x.device)
                    losses["self_bootstrap_loss"] = torch.tensor(0.0, device=x.device)
                    losses["disam_loss_variance"] = torch.tensor(0.0, device=x.device)
                    losses["moment_align_loss"] = torch.tensor(0.0, device=x.device)
                    losses["iu_unlearn_loss"] = torch.tensor(0.0, device=x.device)
                    epoch_udim_losses.append(float(losses["udim_inconsistency_loss"].detach().item()))
                else:
                    pred, losses = self._forward_and_loss(x_norm, target, loss_mask, latitude_weight=latitude_weight)
                    losses["disam_loss_variance"] = torch.tensor(0.0, device=x.device)
                    losses["udim_inconsistency_loss"] = torch.tensor(0.0, device=x.device)
                if self.dg_method == "deep_coral" and self.coral_lambda > 0.0 and target_context_iter is not None:
                    try:
                        target_context_batch = next(target_context_iter)
                    except StopIteration:
                        target_context_iter = iter(target_context_loader)
                        target_context_batch = next(target_context_iter)
                    target_x = target_context_batch["x"].to(self.device)
                    target_x_norm = self._normalize(target_x)
                    source_features = self.model.forward_features(x_norm, return_layer=self.coral_feature_layer)
                    target_features = self.model.forward_features(target_x_norm, return_layer=self.coral_feature_layer)
                    coral = coral_loss(source_features, target_features)
                    losses["coral_loss"] = coral
                    losses["total_loss"] = losses["total_loss"] + self.coral_lambda * coral
                    epoch_coral_losses.append(float(coral.detach().item()))
                else:
                    losses["coral_loss"] = torch.tensor(0.0, device=x.device)
                if self.dg_method == "tca" and self.tca_lambda > 0.0 and target_context_iter is not None:
                    try:
                        target_context_batch = next(target_context_iter)
                    except StopIteration:
                        target_context_iter = iter(target_context_loader)
                        target_context_batch = next(target_context_iter)
                    target_x = target_context_batch["x"].to(self.device)
                    target_x_norm = self._normalize(target_x)
                    source_features = self.model.forward_features(x_norm, return_layer=self.tca_feature_layer)
                    target_features = self.model.forward_features(target_x_norm, return_layer=self.tca_feature_layer)
                    tca_loss = tca_correlation_alignment_loss(source_features, target_features)
                    losses["tca_loss"] = tca_loss
                    losses["total_loss"] = losses["total_loss"] + self.tca_lambda * tca_loss
                    epoch_tca_losses.append(float(tca_loss.detach().item()))
                else:
                    losses["tca_loss"] = torch.tensor(0.0, device=x.device)
                if self.dg_method == "ssa_reg" and self.ssa_reg_lambda > 0.0 and target_context_iter is not None:
                    try:
                        target_context_batch = next(target_context_iter)
                    except StopIteration:
                        target_context_iter = iter(target_context_loader)
                        target_context_batch = next(target_context_iter)
                    target_x = target_context_batch["x"].to(self.device)
                    target_x_norm = self._normalize(target_x)
                    source_features = self.model.forward_features(x_norm, return_layer=self.ssa_reg_feature_layer)
                    target_features = self.model.forward_features(target_x_norm, return_layer=self.ssa_reg_feature_layer)
                    ssa_reg_loss = subspace_alignment_loss(
                        source_features,
                        target_features,
                        rank=self.ssa_reg_rank,
                    )
                    losses["ssa_reg_loss"] = ssa_reg_loss
                    losses["total_loss"] = losses["total_loss"] + self.ssa_reg_lambda * ssa_reg_loss
                    epoch_ssa_reg_losses.append(float(ssa_reg_loss.detach().item()))
                else:
                    losses["ssa_reg_loss"] = torch.tensor(0.0, device=x.device)
                if (
                    self.dg_method == "self_bootstrap"
                    and self.self_bootstrap_lambda > 0.0
                    and target_context_iter is not None
                    and self.self_bootstrap_augmentation is not None
                ):
                    try:
                        target_context_batch = next(target_context_iter)
                    except StopIteration:
                        target_context_iter = iter(target_context_loader)
                        target_context_batch = next(target_context_iter)
                    target_x = target_context_batch["x"].to(self.device)
                    target_x_norm = self._normalize(target_x)
                    target_x_aug = self.self_bootstrap_augmentation(target_x_norm)
                    clean_pred = self.model(target_x_norm)
                    aug_pred = self.model(target_x_aug)
                    consistency = prediction_consistency_loss(clean_pred, aug_pred)
                    losses["self_bootstrap_loss"] = consistency
                    losses["total_loss"] = losses["total_loss"] + self.self_bootstrap_lambda * consistency
                    epoch_self_bootstrap_losses.append(float(consistency.detach().item()))
                else:
                    losses["self_bootstrap_loss"] = torch.tensor(0.0, device=x.device)
                if self.dg_method == "moment_align":
                    losses = self._add_moment_alignment(
                        losses,
                        x_norm=x_norm,
                        sample_region_ids=sample_region_ids,
                        region_mask_integer=region_mask_integer,
                        active_region_ids=active_region_ids,
                    )
                    epoch_moment_align_losses.append(float(losses["moment_align_loss"].detach().item()))
                else:
                    losses["moment_align_loss"] = torch.tensor(0.0, device=x.device)
                if self.dg_method == "iu":
                    losses = self._add_identify_unlearn(
                        losses,
                        x_norm=x_norm,
                        sample_region_ids=sample_region_ids,
                        region_mask_integer=region_mask_integer,
                        active_region_ids=active_region_ids,
                    )
                    epoch_iu_losses.append(float(losses["iu_unlearn_loss"].detach().item()))
                else:
                    losses["iu_unlearn_loss"] = torch.tensor(0.0, device=x.device)
                batch_stats: Dict[str, float] = {
                    "pred_s_mean": float(pred[:, 0].mean().item()),
                    "pred_s_std": float(pred[:, 0].std().item()),
                    "pred_r_mean": float(pred[:, 1].mean().item()),
                    "pred_r_std": float(pred[:, 1].std().item()),
                    "target_s_mean": float(target[:, 0].mean().item()),
                    "target_s_std": float(target[:, 0].std().item()),
                    "target_r_mean": float(target[:, 1].mean().item()),
                    "target_r_std": float(target[:, 1].std().item()),
                    "loss_mask_numel": float(loss_mask.numel()),
                    "region_forward_count": 1.0,
                    "logical_batch_size": float(x.size(0)),
                }

                # Optional CUDA sync for precise error attribution (debug only)
                if self.cuda_sync_debug and self.device == "cuda":
                    torch.cuda.synchronize()

                if self.dg_method not in {"disam", "udim"}:
                    # NaN/Inf guard on loss: skip batch if invalid
                    if torch.isnan(losses["total_loss"]) or torch.isinf(losses["total_loss"]):
                        print(f"  WARNING: E{epoch} S{batch_idx}: loss is NaN/Inf — skipping batch", flush=True)
                        continue

                    # Backward pass
                    if self.use_amp:
                        self._amp_scaler.scale(losses["total_loss"]).backward()
                        if (batch_idx + 1) % self.accum_steps == 0:
                            prev_scale = self._amp_scaler.get_scale()
                            if self.grad_clip is not None:
                                self._amp_scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                            self._amp_scaler.step(self.optimizer)
                            self._amp_scaler.update()
                            self.optimizer.zero_grad()
                            # Track gradient overflow skips (scale reduction = Inf/NaN detected)
                            if self._amp_scaler.get_scale() < prev_scale:
                                self._skipped_steps += 1
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
                valid_count = int(
                    losses.get("valid_pixel_count", losses.get("valid_weight_sum", 0)).item()
                )
                epoch_valid_counts.append(valid_count)

                # Per-step logging
                if batch_idx % self.log_every_steps == 0:
                    # Compute grad norm
                    grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5 if grad_norm > 0 else 0.0

                    # Compute pred stats from latest logical batch.
                    pred_s_mean = batch_stats["pred_s_mean"]
                    pred_s_std = batch_stats["pred_s_std"]
                    pred_r_mean = batch_stats["pred_r_mean"]
                    pred_r_std = batch_stats["pred_r_std"]
                    target_s_mean = batch_stats["target_s_mean"]
                    target_s_std = batch_stats["target_s_std"]
                    target_r_mean = batch_stats["target_r_mean"]
                    target_r_std = batch_stats["target_r_std"]

                    # GPU memory
                    gpu_alloc = 0.0
                    gpu_res = 0.0
                    if self.device == "cuda":
                        dev_idx = torch.cuda.current_device()
                        gpu_alloc = torch.cuda.memory_allocated(dev_idx) / 1e9
                        gpu_res = torch.cuda.memory_reserved(dev_idx) / 1e9
                        gpu_peak_res = torch.cuda.max_memory_reserved(dev_idx) / 1e9
                    else:
                        gpu_peak_res = 0.0

                    lr = float(self.optimizer.param_groups[0]["lr"])
                    valid_px = int(
                        losses.get("valid_pixel_count", losses.get("valid_weight_sum", 0)).item()
                    )
                    total_px = int(batch_stats["loss_mask_numel"])
                    valid_fraction = valid_px / max(total_px, 1)
                    amp_scale = self._amp_scaler.get_scale() if self.use_amp else 0.0

                    step_data = {
                        "epoch": epoch,
                        "step": batch_idx,
                        "global_step": global_step,
                        "lr": lr,
                        "total_loss": float(losses["total_loss"].item()),
                        "surface_loss": float(losses["surface_loss"].item()),
                        "rootzone_loss": float(losses["rootzone_loss"].item()),
                        "coral_loss": float(losses.get("coral_loss", torch.tensor(0.0)).item()),
                        "tca_loss": float(losses.get("tca_loss", torch.tensor(0.0)).item()),
                        "ssa_reg_loss": float(losses.get("ssa_reg_loss", torch.tensor(0.0)).item()),
                        "self_bootstrap_loss": float(losses.get("self_bootstrap_loss", torch.tensor(0.0)).item()),
                        "disam_loss_variance": float(losses.get("disam_loss_variance", torch.tensor(0.0)).item()),
                        "udim_inconsistency_loss": float(
                            losses.get("udim_inconsistency_loss", torch.tensor(0.0)).item()
                        ),
                        "moment_align_loss": float(losses.get("moment_align_loss", torch.tensor(0.0)).item()),
                        "iu_unlearn_loss": float(losses.get("iu_unlearn_loss", torch.tensor(0.0)).item()),
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
                        "gpu_peak_reserved_gb": round(gpu_peak_res, 2),
                        "region_forward_count": int(batch_stats["region_forward_count"]),
                        "logical_batch_size": int(batch_stats["logical_batch_size"]),
                        "amp_scale": amp_scale,
                        "skipped_steps": self._skipped_steps,
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
                            f"gpu={gpu_alloc:.1f}GB peak={gpu_peak_res:.1f}GB {batches_per_sec:.1f}b/s | lr={lr:.2e} "
                            f"amp_scale={amp_scale:.0f} skip={self._skipped_steps}",
                            flush=True,
                        )

                    # Wandb step log
                    if self.wandb_logger is not None and self.wandb_logger.enabled:
                        wandb_data = {
                            "train/total_loss": float(losses["total_loss"].item()),
                            "train/surface_loss": float(losses["surface_loss"].item()),
                            "train/rootzone_loss": float(losses["rootzone_loss"].item()),
                            "train/coral_loss": float(losses.get("coral_loss", torch.tensor(0.0)).item()),
                            "train/tca_loss": float(losses.get("tca_loss", torch.tensor(0.0)).item()),
                            "train/ssa_reg_loss": float(losses.get("ssa_reg_loss", torch.tensor(0.0)).item()),
                            "train/self_bootstrap_loss": float(losses.get("self_bootstrap_loss", torch.tensor(0.0)).item()),
                            "train/disam_loss_variance": float(
                                losses.get("disam_loss_variance", torch.tensor(0.0)).item()
                            ),
                            "train/udim_inconsistency_loss": float(
                                losses.get("udim_inconsistency_loss", torch.tensor(0.0)).item()
                            ),
                            "train/moment_align_loss": float(
                                losses.get("moment_align_loss", torch.tensor(0.0)).item()
                            ),
                            "train/iu_unlearn_loss": float(
                                losses.get("iu_unlearn_loss", torch.tensor(0.0)).item()
                            ),
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "train/valid_pixel_fraction": valid_fraction,
                            "train/pred_inc_surface_std": pred_s_std,
                            "train/pred_inc_rootzone_std": pred_r_std,
                            "train/gpu_memory_gb": gpu_alloc,
                            "train/gpu_peak_reserved_gb": gpu_peak_res,
                            "train/region_forward_count": int(batch_stats["region_forward_count"]),
                            "train/logical_batch_size": int(batch_stats["logical_batch_size"]),
                            "train/skipped_steps": self._skipped_steps,
                        }
                        self.wandb_logger.log_step(wandb_data)

                global_step += 1
                batches_since_eval += 1

            avg_loss = float(np.mean(epoch_losses))
            avg_surface = float(np.mean(epoch_surface_losses))
            avg_rootzone = float(np.mean(epoch_rootzone_losses))
            avg_coral = float(np.mean(epoch_coral_losses)) if epoch_coral_losses else 0.0
            avg_tca = float(np.mean(epoch_tca_losses)) if epoch_tca_losses else 0.0
            avg_ssa_reg = float(np.mean(epoch_ssa_reg_losses)) if epoch_ssa_reg_losses else 0.0
            avg_self_bootstrap = (
                float(np.mean(epoch_self_bootstrap_losses)) if epoch_self_bootstrap_losses else 0.0
            )
            avg_disam = float(np.mean(epoch_disam_losses)) if epoch_disam_losses else 0.0
            avg_udim = float(np.mean(epoch_udim_losses)) if epoch_udim_losses else 0.0
            avg_moment_align = float(np.mean(epoch_moment_align_losses)) if epoch_moment_align_losses else 0.0
            avg_iu = float(np.mean(epoch_iu_losses)) if epoch_iu_losses else 0.0
            total_valid = int(np.sum(epoch_valid_counts))
            elapsed = time.time() - epoch_start

            self.scheduler.step(avg_loss)

            # Source val eval + gain calibration every eval_every_epochs
            source_val_metrics = {}
            gain_results = {}
            if self.source_val_dataset is not None and epoch % self.eval_every_epochs == 0:
                gain_results = self._calibrate_source_val_residual_gain()
                if gain_results:
                    source_val_metrics = self._make_source_val_metrics(gain_results)
                if verbose and gain_results:
                    sv_loss = gain_results.get("source_val_loss", float("nan"))
                    sv_rmse_s = gain_results.get("rmse_surface_model", float("nan"))
                    sv_rmse_r = gain_results.get("rmse_rootzone_model", float("nan"))
                    sv_skill_s = gain_results.get("skill_surface_with_alpha", float("nan"))
                    sv_skill_r = gain_results.get("skill_rootzone_with_alpha", float("nan"))
                    alpha_s = gain_results.get("best_alpha_surface", 1.0)
                    alpha_r = gain_results.get("best_alpha_rootzone", 1.0)
                    sel_score = gain_results.get("selection_score", float("-inf"))
                    print(
                        f"  source_val loss={sv_loss:.6f}"
                        f"  rmse_s={sv_rmse_s:.6f} r={sv_rmse_r:.6f}"
                        f"  skill_s={sv_skill_s:.4f} r={sv_skill_r:.4f}"
                        f"  alpha_s={alpha_s:.3f} alpha_r={alpha_r:.3f}"
                        f"  sel_score={sel_score:.4f}",
                        flush=True,
                    )
                if self._jsonl_logger is not None:
                    log_data = {"epoch": epoch}
                    log_data.update(source_val_metrics)
                    if gain_results:
                        log_data["selection_score"] = gain_results.get("selection_score", float("nan"))
                        log_data["best_alpha_surface"] = gain_results.get("best_alpha_surface", 1.0)
                        log_data["best_alpha_rootzone"] = gain_results.get("best_alpha_rootzone", 1.0)
                    self._jsonl_logger.log_eval(log_data)

            if self.swad_state is not None and source_val_metrics:
                swad_added = self.swad_state.update(
                    epoch=epoch,
                    source_val_metric=source_val_metrics["source_val_loss"],
                    model=self.model,
                )
                source_val_metrics["swad_n_averaged"] = float(self.swad_state.n_averaged)
                source_val_metrics["swad_window_updated"] = float(1 if swad_added else 0)

            selection = self._resolve_checkpoint_selection(
                avg_loss=avg_loss,
                source_val_metrics=source_val_metrics,
                gain_results=gain_results,
            )

            if selection.is_best:
                self.best_loss = selection.best_metric
                ckpt_path = self.checkpoint_dir / "best.pt"
                self.save_checkpoint(
                    ckpt_path,
                    epoch,
                    selection.best_metric,
                    "best",
                    gain_results=gain_results,
                    selection_metric_name=selection.metric_name,
                )

            # Save safe_score checkpoint (selection_score == min_skill, the primary criterion)
            if gain_results:
                if gain_results["selection_score"] > self.best_safe_score:
                    self.best_safe_score = gain_results["selection_score"]
                    self.save_checkpoint(
                        self.checkpoint_dir / "checkpoint_best_source_val_safe_score.pt",
                        epoch, gain_results["selection_score"], "best_safe_score",
                        gain_results=gain_results,
                        selection_metric_name="source_val_safe_score",
                    )

            # Always save last.pt
            self.save_checkpoint(
                self.checkpoint_dir / "last.pt", epoch, avg_loss, "last",
                gain_results=gain_results,
                selection_metric_name="train_loss",
            )

            # Epoch checkpoint
            self.save_checkpoint(
                self.checkpoint_dir / "checkpoint_latest.pt", epoch, avg_loss, "latest",
                gain_results=gain_results,
                selection_metric_name="train_loss",
            )
            if (epoch + 1) % self.checkpoint_every_n_epochs == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                    epoch, avg_loss, f"epoch_{epoch:03d}",
                    gain_results=gain_results,
                    selection_metric_name="train_loss",
                )

            record = {
                "epoch": epoch,
                "surface_loss": avg_surface,
                "rootzone_loss": avg_rootzone,
                "coral_loss": avg_coral,
                "tca_loss": avg_tca,
                "ssa_reg_loss": avg_ssa_reg,
                "self_bootstrap_loss": avg_self_bootstrap,
                "disam_loss_variance": avg_disam,
                "udim_inconsistency_loss": avg_udim,
                "moment_align_loss": avg_moment_align,
                "iu_unlearn_loss": avg_iu,
                "total_loss": avg_loss,
                "valid_pixel_count": total_valid,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "elapsed_s": elapsed,
                "skipped_steps": self._skipped_steps,
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
                    sv_rmse_s = source_val_metrics.get("source_val_rmse_surface", float("nan"))
                    sv_rmse_r = source_val_metrics.get("source_val_rmse_rootzone", float("nan"))
                    sv_skill_s = source_val_metrics.get("source_val_skill_surface", float("nan"))
                    sv_skill_r = source_val_metrics.get("source_val_skill_rootzone", float("nan"))
                    sv_str = (f"  sv_loss={sv_loss:.6f}"
                              f"  sv_rmse_s={sv_rmse_s:.6f} r={sv_rmse_r:.6f}"
                              f"  sv_skill_s={sv_skill_s:.4f} r={sv_skill_r:.4f}")
                epoch_summary = (
                    f"Epoch {epoch:3d} | "
                    f"loss={avg_loss:.6f} | "
                    f"surface={avg_surface:.6f} | "
                    f"rootzone={avg_rootzone:.6f} | "
                    f"valid_px={total_valid:9d} | "
                    f"lr={record['lr']:.2e} | "
                    f"{elapsed:.1f}s | skip={self._skipped_steps}"
                    f"{sv_str}"
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
        if self.swad_state is not None and self.swad_state.n_averaged > 0:
            swad_path = self.checkpoint_dir / "checkpoint_swad.pt"
            swad_metadata = {
                "experiment_id": self.experiment_id,
                "protocol_freeze_id": self.protocol_freeze_id,
                "split_manifest_path": self.split_manifest_path,
                "split_manifest_sha256": self.split_manifest_sha256,
                "git_hash": get_git_hash(),
                "timestamp": get_timestamp(),
            }
            swad_metadata.update(self.extra_checkpoint_metadata)
            self.swad_state.save_checkpoint(swad_path, metadata=swad_metadata)
            end_lines.insert(5, f"  SWAD checkpoint: {swad_path}")
        if self.ssa_reg_state is not None:
            ssa_reg_path = self.checkpoint_dir / "checkpoint_ssa_reg.pt"
            ssa_reg_metadata = {
                "experiment_id": self.experiment_id,
                "protocol_freeze_id": self.protocol_freeze_id,
                "split_manifest_path": self.split_manifest_path,
                "split_manifest_sha256": self.split_manifest_sha256,
                "git_hash": get_git_hash(),
                "timestamp": get_timestamp(),
            }
            ssa_reg_metadata.update(self.extra_checkpoint_metadata)
            self.ssa_reg_state.save_checkpoint(
                ssa_reg_path,
                model_state_dict=self.model.state_dict(),
                metadata=ssa_reg_metadata,
            )
            end_lines.insert(5, f"  SSA-Reg checkpoint: {ssa_reg_path}")
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
        gain_results: Optional[Dict[str, Any]] = None,
        selection_metric_name: Optional[str] = None,
    ) -> None:
        """Save a checkpoint with config, protocol_freeze_id, split_manifest_path, git_hash."""
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: Dict[str, Any] = {
            "tag": tag,
            "epoch": epoch,
            "loss": loss,
            "best_loss": self.best_loss,
            "experiment_id": self.experiment_id,
            "protocol_freeze_id": self.protocol_freeze_id,
            "split_manifest_path": self.split_manifest_path,
            "split_manifest_sha256": self.split_manifest_sha256,
            "git_hash": get_git_hash(),
            "timestamp": get_timestamp(),
            "selection_metric": self.selection_metric,
            "checkpoint_selection_metric": selection_metric_name or self.selection_metric,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_history": self.train_history,
            "val_history": self.val_history,
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
                "use_lat_weighted_loss": self.use_lat_weighted_loss,
                "checkpoint_every_n_epochs": self.checkpoint_every_n_epochs,
                "selection_metric": self.selection_metric,
                "checkpoint_selection_metric": selection_metric_name or self.selection_metric,
                "lambda_amp": self.lambda_amp,
                "dg_method": self.dg_method,
                "coral_lambda": self.coral_lambda,
                "coral_feature_layer": self.coral_feature_layer,
                "tca_lambda": self.tca_lambda,
                "tca_feature_layer": self.tca_feature_layer,
                "ssa_reg_lambda": self.ssa_reg_lambda,
                "ssa_reg_feature_layer": self.ssa_reg_feature_layer,
                "ssa_reg_rank": self.ssa_reg_rank,
                "self_bootstrap_lambda": self.self_bootstrap_lambda,
                "self_bootstrap_noise_std": self.self_bootstrap_noise_std,
                "self_bootstrap_channel_dropout_p": self.self_bootstrap_channel_dropout_p,
                "disam_rho": self.disam_rho,
                "disam_lambda": self.disam_lambda,
                "udim_rho": self.udim_rho,
                "udim_lambda": self.udim_lambda,
                "udim_objective": "source_only_unknown_domain_inconsistency",
                "moment_align_lambda": self.moment_align_lambda,
                "moment_align_feature_layer": self.moment_align_feature_layer,
                "moment_align_order": self.moment_align_order,
                "iu_lambda": self.iu_lambda,
                "iu_feature_layer": self.iu_feature_layer,
                "iu_top_fraction": self.iu_top_fraction,
                "iu_sample_top_fraction": self.iu_sample_top_fraction,
                "iu_score_cap": self.iu_score_cap,
                "iu_objective": "bounded_domain_specific_feature_penalty",
                "target_context_batch_size": self.target_context_batch_size,
                "ch_mean": self._ch_mean.tolist() if self._ch_mean is not None else None,
                "ch_std": self._ch_std.tolist() if self._ch_std is not None else None,
                "inc_mean": self._inc_mean.tolist() if self._inc_mean is not None else None,
                "inc_std": self._inc_std.tolist() if self._inc_std is not None else None,
            },
        }
        checkpoint.update(self.extra_checkpoint_metadata)
        checkpoint["config"].update(self.extra_checkpoint_metadata)
        # Attach gain calibration results
        if gain_results:
            checkpoint["residual_gain_alpha_surface"] = gain_results.get("best_alpha_surface", 1.0)
            checkpoint["residual_gain_alpha_rootzone"] = gain_results.get("best_alpha_rootzone", 1.0)
            checkpoint["source_val_safe_metrics"] = gain_results
            checkpoint["source_val_gain_grid"] = gain_results.get("alpha_grid", self.source_val_gain_grid)
            checkpoint["selection_score"] = gain_results.get("selection_score", float("nan"))
            checkpoint["min_skill"] = gain_results.get("min_skill", float("nan"))
            checkpoint["mean_skill"] = gain_results.get("mean_skill", float("nan"))
        torch.save(checkpoint, path)

    def save_summary_json(self, path: Optional[Path] = None) -> None:
        """Save summary.json with protocol safety fields."""
        has_source_val = self.source_val_dataset is not None
        num_params = sum(p.numel() for p in self.model.parameters())
        summary = {
            "experiment_id": self.experiment_id,
            "protocol_freeze_id": self.protocol_freeze_id,
            "split_manifest_path": self.split_manifest_path,
            "split_manifest_sha256": self.split_manifest_sha256,
            "best_loss": self.best_loss,
            "final_epoch": self.current_epoch - 1,
            "total_epochs_completed": self.current_epoch,
            "model_width": self.model_width,
            "trainable_parameters": num_params,
            "batch_size": self.batch_size,
            "accum_steps": self.accum_steps,
            "effective_batch_size": self.batch_size * self.accum_steps,
            "source_val_available": has_source_val,
            "selection_metric": self.selection_metric,
            "normalization_source": "source_fit_only",
            "early_stopping_source": "source_val_only" if has_source_val else "train_loss_only",
            "model_selection_source": "source_val_only" if has_source_val else "best_train_loss",
            "target_adaptation_source": "main_zero_few_shot_target_support_only",
            "target_eval_usage": "eval_only_no_early_stopping",
            "target_query_usage": "eval_only_no_early_stopping",  # deprecated alias
            "leakage_guard_status": "pass",
            "git_hash": get_git_hash(),
            "timestamp": get_timestamp(),
            "train_history": self.train_history,
            "val_history": self.val_history,
        }
        summary.update(self.extra_checkpoint_metadata)
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

    def load_state(self, checkpoint: Dict[str, Any]) -> int:
        """Restore full training state from a checkpoint. Returns resumed epoch."""
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.best_safe_score = checkpoint.get("selection_score", float("-inf"))
        self.train_history = checkpoint.get("train_history", [])
        self.val_history = checkpoint.get("val_history", [])

        return self.current_epoch
