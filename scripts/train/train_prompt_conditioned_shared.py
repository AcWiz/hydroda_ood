#!/usr/bin/env python3
"""Train prompt-conditioned shared backbone (FiLMConditionalResUNet + RegionPromptEncoder).

Multi-region mixed training: source_fit (2015-2021) from all source regions.
Each sample's prompt uses the region embedding for the region with the most
valid pixels in the sample.

Usage:
    PYTHONPATH=. python scripts/train/train_prompt_conditioned_shared.py \\
        --target_region US-R1 --adaptation_setting target_full_train --seed 0 \\
        --max_epochs 5 --batch_size 1 --accum_steps 4 --lr 3e-4 \\
        --weight_decay 1e-4 --grad_clip 1.0 \\
        --width 32 --prompt_dim 64 \\
        --zero_raw_increment_init --target_increment_normalization \\
        --device cuda --amp \\
        --config configs/model_resunet_main.yaml

No-leakage declaration:
    - Training uses source_fit split only (2015-2021, all source regions)
    - Normalization stats from source_fit only
    - Region prompt uses input-side features only
    - No target_eval/target_query labels used in training/normalization/early_stopping
"""
from __future__ import annotations

import argparse
import csv
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
from hydroda.data.file_hash import compute_sha256
from hydroda.data.leakage_guard import LeakageGuard
from hydroda.data.protocol import ProtocolConfig
from hydroda.models.conditional_unet import FiLMConditionalResUNet
from hydroda.models.prompt_encoder import RegionPromptEncoder
from hydroda.training.calibration import calibrate_residual_gain, calibrate_residual_gain_region_aware
from hydroda.training.losses import MaskedHuberLoss, WeightedMaskedHuberLoss
from hydroda.metrics.skill import weighted_analysis_skill_components
from hydroda.utils.run_manager import RunManager
from hydroda.utils.logger import WandbLogger, ConsoleLogger
from hydroda.utils.device import resolve_device, log_device_summary, gpu_health_check
from hydroda.utils.runtime import gather_runtime_info, get_git_hash, get_timestamp


DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_target_train_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
CHECKPOINT_DIR = "artifacts/checkpoints/phase4_prompt_conditioned"
PROTOCOL_FREEZE_ID = "hyperda_v4_3_historical_target_adapt_2015_2025_train2015_2021_val2022_test2023_2025"
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


def _sample_region_from_mask(
    region_mask_integer: np.ndarray,
    valid_mask: np.ndarray,
    active_global_indices: set | None = None,
) -> int:
    """Determine the dominant region from region_mask_integer and valid pixels.

    Args:
        region_mask_integer: [H, W] with region indices (1-6)
        valid_mask: [H, W] boolean valid pixel mask
        active_global_indices: optional set of 0-indexed global region indices
            to restrict sampling to (e.g. source-only regions). If None,
            all regions 1-6 are considered.

    Returns:
        region index (0-5)
    """
    if active_global_indices is not None:
        # Convert 0-indexed global indices to 1-indexed region numbers
        candidate_nums = sorted(r + 1 for r in active_global_indices)
    else:
        candidate_nums = list(range(1, 7))

    for r_idx in candidate_nums:
        count = int(((region_mask_integer == r_idx) & valid_mask).sum())
        if count > 0:
            return r_idx - 1  # 0-indexed

    # Fallback: count per region (restricted to candidates)
    counts = {}
    for r_idx in candidate_nums:
        counts[r_idx] = int((region_mask_integer == r_idx).sum())

    best = max(counts, key=counts.get)
    return best - 1


class PromptQualityTracker:
    """Track prompt embedding quality across regions to detect collapse.

    Accumulates prompt embeddings per region during evaluation and computes
    pairwise cosine distances to measure prompt diversity.
    """

    def __init__(self, num_regions: int):
        self.num_regions = num_regions
        self.reset()

    def reset(self) -> None:
        self._embeddings: Dict[int, List[np.ndarray]] = {i: [] for i in range(self.num_regions)}

    def update(self, prompt_emb: torch.Tensor, region_ids: torch.Tensor) -> None:
        """Accumulate prompt embeddings keyed by region id.

        Args:
            prompt_emb: [B, prompt_dim] prompt embeddings
            region_ids: [B] region indices (0-indexed)
        """
        emb_np = prompt_emb.detach().cpu().numpy()
        rids = region_ids.detach().cpu().numpy()
        for b in range(emb_np.shape[0]):
            rid = int(rids[b])
            if rid in self._embeddings:
                self._embeddings[rid].append(emb_np[b].copy())

    def compute_metrics(self) -> Dict[str, float]:
        """Compute prompt quality metrics from accumulated embeddings.

        Returns dict with:
            prompt_pairwise_cosine_distance_mean
            prompt_pairwise_cosine_distance_min
            prompt_collapse_detected
        """
        region_means = []
        for rid in sorted(self._embeddings.keys()):
            embs = self._embeddings[rid]
            if embs:
                region_means.append(np.mean(np.stack(embs, axis=0), axis=0))

        n = len(region_means)
        if n < 2:
            return {
                "prompt_pairwise_cosine_distance_mean": float("nan"),
                "prompt_pairwise_cosine_distance_min": float("nan"),
                "prompt_collapse_detected": False,
            }

        cos_distances = []
        for i in range(n):
            for j in range(i + 1, n):
                a = region_means[i]
                b = region_means[j]
                a_norm = a / (np.linalg.norm(a) + 1e-8)
                b_norm = b / (np.linalg.norm(b) + 1e-8)
                cos_sim = float(np.dot(a_norm, b_norm))
                cos_distances.append(1.0 - cos_sim)

        mean_cos_dist = float(np.mean(cos_distances)) if cos_distances else float("nan")
        min_cos_dist = float(np.min(cos_distances)) if cos_distances else float("nan")
        collapsed = bool(mean_cos_dist < 0.01) if np.isfinite(mean_cos_dist) else False

        return {
            "prompt_pairwise_cosine_distance_mean": mean_cos_dist,
            "prompt_pairwise_cosine_distance_min": min_cos_dist,
            "prompt_collapse_detected": collapsed,
        }


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
        use_lat_weighted_loss: bool = True,
        source_val_gain_grid: Optional[List[float]] = None,
        source_regions: Optional[List[str]] = None,
        checkpoint_every_n_epochs: int = 5,
        lambda_amp: float = 0.0,
        selection_metric: str = "source_val_transfer_safe_score",
        source_val_residual_gain: bool = True,
        cuda_sync_debug: bool = False,
        target_region: Optional[str] = None,
        adaptation_setting: str = "target_full_train",
        K: Optional[int] = None,
        # Resume: optionally inject pre-computed stats to skip recompute
        _resume_ch_mean: Optional[np.ndarray] = None,
        _resume_ch_std: Optional[np.ndarray] = None,
        _resume_inc_mean: Optional[np.ndarray] = None,
        _resume_inc_std: Optional[np.ndarray] = None,
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
        split_manifest_file = Path(split_manifest_path) if split_manifest_path else None
        self.split_manifest_sha256 = (
            compute_sha256(split_manifest_file)
            if split_manifest_file is not None and split_manifest_file.exists()
            else ""
        )
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
        self.global_to_source_lookup = global_to_source_lookup if global_to_source_lookup is not None else {i: i for i in range(6)}
        self.use_lat_weighted_loss = use_lat_weighted_loss
        self.source_val_gain_grid = source_val_gain_grid or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.source_regions = source_regions or []
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.lambda_amp = lambda_amp
        self.selection_metric = selection_metric
        self.source_val_residual_gain = source_val_residual_gain
        self.cuda_sync_debug = cuda_sync_debug
        self.target_region = target_region
        self.adaptation_setting = adaptation_setting
        self.K = None if adaptation_setting == "target_full_train" else K
        self._source_region_global_indices = sorted(global_to_source_lookup.keys()) if global_to_source_lookup else []

        # AMP
        self._amp_scaler: Optional[GradScaler] = None
        if self.use_amp:
            self._amp_scaler = GradScaler('cuda', init_scale=256.0)

        # Leakage guard
        protocol = ProtocolConfig()
        guard = LeakageGuard(protocol=protocol)
        train_date_strs = [d["date_str"] for d in self.train_dataset._date_records] if hasattr(self.train_dataset, "_date_records") else []
        guard.check_normalization_scope(train_date_strs, scope_name="source_fit_only")

        # Optimizer: model + prompt_encoder
        all_params = list(model.parameters()) + list(prompt_encoder.parameters())
        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )
        # Loss function selection
        if self.use_lat_weighted_loss:
            self.loss_fn = WeightedMaskedHuberLoss(
                delta=0.01,
                lambda_amp=self.lambda_amp,
            )
        else:
            self.loss_fn = MaskedHuberLoss(delta=0.01)

        # Normalization stats
        self._ch_mean: Optional[np.ndarray] = None
        self._ch_std: Optional[np.ndarray] = None
        if _resume_ch_mean is not None and _resume_ch_std is not None:
            self._ch_mean = _resume_ch_mean
            self._ch_std = _resume_ch_std
            print(f"  [resume] Restored ch_mean from checkpoint")
        else:
            self._compute_normalization_stats()

        # Increment stats
        self._inc_mean: Optional[np.ndarray] = None
        self._inc_std: Optional[np.ndarray] = None
        if self.target_increment_normalization:
            if _resume_inc_mean is not None and _resume_inc_std is not None:
                self._inc_mean = _resume_inc_mean
                self._inc_std = _resume_inc_std
                print(f"  [resume] Restored inc_mean/inc_std from checkpoint")
            else:
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
        self.best_safe_score = float("-inf")
        self.best_selection_metric = self.selection_metric
        self.best_selection_value = (
            float("-inf")
            if self.selection_metric == "source_val_transfer_safe_score"
            else float("inf")
        )
        self._skipped_steps = 0
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []
        self.prompt_quality_history: List[Dict[str, float]] = []

        # Prompt quality tracker
        num_src_regions = len(self.source_regions) if self.source_regions else prompt_encoder.num_regions
        self._prompt_quality_tracker = PromptQualityTracker(num_regions=num_src_regions)

        # JSONL logger
        self._jsonl_logger = None
        if run_manager is not None:
            from hydroda.utils.logger import JSONLLogger
            self._jsonl_logger = JSONLLogger(run_manager.get_log_dir())
            # Open console.log for tee output
            run_manager.open_console_log()

    def _selection_value(
        self,
        train_loss: float,
        source_val_metrics: Dict[str, float],
        gain_results: Dict[str, Any],
    ) -> Tuple[float, bool]:
        """Return (value, maximize) for the configured checkpoint-selection metric."""
        if self.selection_metric == "source_val_transfer_safe_score":
            if not gain_results or "selection_score" not in gain_results:
                return float("-inf"), True
            return float(gain_results["selection_score"]), True
        if self.selection_metric == "source_val_loss":
            if not source_val_metrics or "source_val_loss" not in source_val_metrics:
                return float("inf"), False
            return float(source_val_metrics["source_val_loss"]), False
        if self.selection_metric == "train_loss":
            return float(train_loss), False
        raise ValueError(f"Unsupported selection_metric: {self.selection_metric}")

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
        x_norm = (x - mean_t) / std_t
        # NaN/Inf guard: if normalization produces invalid values, return raw input
        nan_mask = torch.isnan(x_norm)
        inf_mask = torch.isinf(x_norm)
        if nan_mask.any() or inf_mask.any():
            n_nan = nan_mask.sum().item()
            n_inf = inf_mask.sum().item()
            print(f"  WARNING: normalize produced {n_nan} NaN / {n_inf} Inf — returning raw input", flush=True)
            return x
        return x_norm

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
                global_rid = _sample_region_from_mask(
                    s["region_mask_integer"], valid_mask,
                    active_global_indices=set(self.global_to_source_lookup.keys()),
                )
                # Safety: global_rid must be in source region set
                assert global_rid in self.global_to_source_lookup, \
                    f"collate_fn: global_rid={global_rid} not in lookup (target data in source split?)"
                src_rid = self.global_to_source_lookup[global_rid]
                region_ids.append(src_rid)
                months.append(int(s.get("month", 6)))
            result = {
                "x": x,
                "increment_surface": increment_surface,
                "increment_rootzone": increment_rootzone,
                "loss_mask": loss_mask,
                "region_ids": torch.tensor(region_ids, dtype=torch.long),
                "months": torch.tensor(months, dtype=torch.long),
            }
            # Add latitude_weight for lat-weighted loss
            if "latitude_weight" in batch[0]:
                latitude_weight = torch.from_numpy(
                    np.stack([s["latitude_weight"] for s in batch], axis=0)
                )
                result["latitude_weight"] = latitude_weight
            # Add forecast fields for gain calibration
            for key in ["forecast_surface", "forecast_rootzone"]:
                if key in batch[0]:
                    result[key] = torch.from_numpy(
                        np.stack([s[key] for s in batch], axis=0)
                    )
            return result

        pin_mem = self.device == "cuda"
        return DataLoader(
            target_dataset,
            batch_size=self.batch_size,
            shuffle=(dataset is None),
            num_workers=self.num_workers,
            pin_memory=pin_mem,
            collate_fn=collate_fn,
        )

    def _get_increment_scale(self) -> Optional[torch.Tensor]:
        """Return per-channel increment scale [2] from source_fit stats."""
        if self._inc_std is not None:
            return torch.from_numpy(self._inc_std.astype(np.float32))
        return None

    def _forward_and_loss(
        self,
        x_norm: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        region_ids: torch.Tensor,
        months: torch.Tensor,
        latitude_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass + loss for prompt-conditioned model.

        Handles AMP consistently. Returns (pred, losses_dict).
        """
        if self.use_amp:
            with autocast('cuda'):
                z = self.prompt_encoder(x_norm, region_ids, months)
                pred = self.model(x_norm, z)
        else:
            z = self.prompt_encoder(x_norm, region_ids, months)
            pred = self.model(x_norm, z)

        # Cast to fp32 for numerical stability in loss
        pred = pred.float()

        if self.use_lat_weighted_loss:
            if latitude_weight is None:
                raise ValueError(
                    "use_lat_weighted_loss=True but latitude_weight not provided in batch. "
                    "Ensure dataset returns latitude_weight."
                )
            inc_scale = self._get_increment_scale()
            losses = self.loss_fn(pred, target, loss_mask, latitude_weight=latitude_weight, increment_scale=inc_scale)
        else:
            losses = self.loss_fn(pred, target, loss_mask)
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

    def _eval_source_val(self) -> Dict[str, float]:
        """Run evaluation on source_val split with gain calibration.

        Returns metrics dict with skill, rmse, best_alpha, etc.
        """
        gain_results = self._calibrate_source_val_residual_gain()
        if not gain_results:
            return {}
        return {
            "source_val_loss": gain_results["source_val_loss"],
            "source_val_rmse_surface": gain_results["rmse_surface_model"],
            "source_val_rmse_rootzone": gain_results["rmse_rootzone_model"],
            "source_val_skill_surface": gain_results["skill_surface_with_alpha"],
            "source_val_skill_rootzone": gain_results["skill_rootzone_with_alpha"],
        }

    def _calibrate_source_val_residual_gain(self) -> Dict[str, Any]:
        """Calibrate residual gain alphas on source_val.

        Uses region-aware 2D alpha grid search when source_val_residual_gain=True
        and source_regions are available. Otherwise falls back to shared-alpha
        calibration via calibrate_residual_gain().
        """
        if self.source_val_dataset is None:
            return {}

        self.model.eval()
        self.prompt_encoder.eval()
        loader = self._build_dataloader(self.source_val_dataset)
        alphas = self.source_val_gain_grid

        # Per-region sample storage
        samples_s_by_region: Dict[str, List] = {}
        samples_r_by_region: Dict[str, List] = {}
        for rname in self.source_regions:
            samples_s_by_region[rname] = []
            samples_r_by_region[rname] = []

        total_loss = 0.0
        total_surface = 0.0
        total_rootzone = 0.0
        total_valid = 0
        n_batches = 0

        # Reset prompt quality tracker
        self._prompt_quality_tracker.reset()

        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                inc_surface = batch["increment_surface"].to(self.device)
                inc_rootzone = batch["increment_rootzone"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)
                region_ids = batch["region_ids"].to(self.device)
                months = batch["months"].to(self.device)
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

                z = self.prompt_encoder(x_norm, region_ids, months)
                pred = self.model(x_norm, z)

                # Track prompt embeddings for quality
                self._prompt_quality_tracker.update(z, region_ids)

                # Loss
                if self.use_lat_weighted_loss:
                    inc_scale = self._get_increment_scale()
                    losses = self.loss_fn(pred, target, loss_mask, latitude_weight=latitude_weight, increment_scale=inc_scale)
                else:
                    losses = self.loss_fn(pred, target, loss_mask)

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

                # Accumulate per-sample arrays by region
                for b in range(x.size(0)):
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

                    pred_inc_r = pred_denorm_r[b].cpu().numpy().astype(np.float32)
                    true_inc_r = inc_rootzone[b].cpu().numpy().astype(np.float32)
                    fcst_r = (
                        forecast_r[b].cpu().numpy().astype(np.float32)
                        if forecast_r is not None
                        else np.zeros_like(true_inc_r, dtype=np.float32)
                    )

                    # Map source region id to region name
                    src_rid = int(region_ids[b].item())
                    region_name = self.source_regions[src_rid] if src_rid < len(self.source_regions) else f"region_{src_rid}"

                    if region_name in samples_s_by_region:
                        samples_s_by_region[region_name].append((pred_inc_s, true_inc_s, fcst_s, mask_np, latw_np))
                        samples_r_by_region[region_name].append((pred_inc_r, true_inc_r, fcst_r, mask_np, latw_np))

        self.model.train()
        self.prompt_encoder.train()

        # Compute prompt quality
        prompt_quality = self._prompt_quality_tracker.compute_metrics()

        # Check if any samples were accumulated
        total_s = sum(len(v) for v in samples_s_by_region.values())
        if total_s == 0:
            return {}

        # Choose calibration mode
        if self.source_val_residual_gain and len(self.source_regions) >= 2:
            # Region-aware 2D alpha grid
            trace_path = self.checkpoint_dir / "alpha_selection_trace.csv"
            gain_results = calibrate_residual_gain_region_aware(
                samples_s_by_region=samples_s_by_region,
                samples_r_by_region=samples_r_by_region,
                alpha_grid=alphas,
                prompt_quality_metrics=prompt_quality,
                trace_path=trace_path,
            )
            gain_results["prompt_quality"] = prompt_quality
            gain_results["source_val_loss"] = total_loss / max(n_batches, 1)
            gain_results["source_val_surface_loss"] = total_surface / max(n_batches, 1)
            gain_results["source_val_rootzone_loss"] = total_rootzone / max(n_batches, 1)
            gain_results["source_val_valid_px"] = total_valid
            return gain_results
        else:
            # Fallback: shared-alpha calibration
            all_s = []
            all_r = []
            for region in samples_s_by_region:
                all_s.extend(samples_s_by_region[region])
                all_r.extend(samples_r_by_region[region])

            gain_results = calibrate_residual_gain(all_s, all_r, alphas)
            gain_results["prompt_quality"] = prompt_quality
            gain_results["source_val_loss"] = total_loss / max(n_batches, 1)
            gain_results["source_val_surface_loss"] = total_surface / max(n_batches, 1)
            gain_results["source_val_rootzone_loss"] = total_rootzone / max(n_batches, 1)
            gain_results["source_val_valid_px"] = total_valid
            return gain_results

    def train(self, verbose: bool = True) -> List[Dict[str, float]]:
        dataloader = self._build_dataloader()
        self.model.train()
        self.prompt_encoder.train()
        global_step = 0
        train_start_time = time.time()
        total_steps_per_epoch = len(dataloader)

        # Training start header
        num_model_params = sum(p.numel() for p in self.model.parameters())
        num_pe_params = sum(p.numel() for p in self.prompt_encoder.parameters())
        header_lines = [
            "=" * 60,
            f"Training Start — Prompt-Conditioned Shared Backbone",
            f"  Experiment:      {self.experiment_id}",
            f"  Protocol:        {self.protocol_freeze_id}",
            f"  Split manifest:  {self.split_manifest_path}",
            f"  Device:          {self.device}",
            f"  Model params:    {num_model_params:,}",
            f"  Prompt enc params: {num_pe_params:,}",
            f"  Total params:    {num_model_params + num_pe_params:,}",
            f"  Model width:     {self.model_width}",
            f"  Prompt dim:      {self.prompt_dim}",
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
            f"  Res gain:        {self.source_val_residual_gain}",
            f"  Train samples:   {len(self.train_dataset)}",
            f"  Source regions:  {self.source_regions}",
            f"  Steps/epoch:     {total_steps_per_epoch}",
        ]
        if self.target_increment_normalization and self._inc_mean is not None:
            header_lines.append(f"  inc_mean:        s={self._inc_mean[0]:.6f} r={self._inc_mean[1]:.6f}")
            header_lines.append(f"  inc_std:         s={self._inc_std[0]:.6f} r={self._inc_std[1]:.6f}")
        header_lines.append("=" * 60)
        for line in header_lines:
            if self.run_manager is not None:
                self.run_manager.log_console(line)
            elif verbose:
                print(line, flush=True)

        # GPU health check before training
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

            # Zero gradients at start of epoch (gradient accumulation fix)
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(dataloader):
                x = batch["x"].to(self.device)
                inc_surface = batch["increment_surface"].to(self.device)
                inc_rootzone = batch["increment_rootzone"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)
                region_ids = batch["region_ids"].to(self.device)
                months = batch["months"].to(self.device)
                latitude_weight = batch.get("latitude_weight")
                if latitude_weight is not None:
                    latitude_weight = latitude_weight.to(self.device)

                x_norm = self._normalize(x)

                # NaN/Inf guard on normalized input: skip batch if invalid
                if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
                    n_nan = torch.isnan(x_norm).sum().item()
                    n_inf = torch.isinf(x_norm).sum().item()
                    line = f"  WARNING: E{epoch} S{batch_idx}: normalized input {n_nan} NaN / {n_inf} Inf — skipping batch"
                    if self.run_manager is not None:
                        self.run_manager.log_console(line)
                    else:
                        print(line, flush=True)
                    continue

                target = torch.stack([inc_surface, inc_rootzone], dim=1)

                if self.target_increment_normalization and self._inc_mean is not None:
                    inc_mean_t = torch.from_numpy(self._inc_mean).to(x.device).view(1, 2, 1, 1)
                    inc_std_t = torch.from_numpy(self._inc_std).to(x.device).view(1, 2, 1, 1)
                    target = (target - inc_mean_t) / inc_std_t

                # Forward + loss via unified helper
                pred, losses = self._forward_and_loss(
                    x_norm, target, loss_mask, region_ids, months,
                    latitude_weight=latitude_weight,
                )

                # Optional CUDA sync for precise error attribution (debug only)
                if self.cuda_sync_debug and self.device == "cuda":
                    torch.cuda.synchronize()

                # NaN/Inf guard on loss: skip batch if invalid
                if torch.isnan(losses["total_loss"]) or torch.isinf(losses["total_loss"]):
                    line = f"  WARNING: E{epoch} S{batch_idx}: loss is NaN/Inf — skipping batch"
                    if self.run_manager is not None:
                        self.run_manager.log_console(line)
                    else:
                        print(line, flush=True)
                    continue

                # Backward pass
                if self.use_amp:
                    self._amp_scaler.scale(losses["total_loss"]).backward()
                    if (batch_idx + 1) % self.accum_steps == 0:
                        prev_scale = self._amp_scaler.get_scale()
                        if self.grad_clip is not None:
                            self._amp_scaler.unscale_(self.optimizer)
                            all_p = list(self.model.parameters()) + list(self.prompt_encoder.parameters())
                            torch.nn.utils.clip_grad_norm_(all_p, self.grad_clip)
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
                            all_p = list(self.model.parameters()) + list(self.prompt_encoder.parameters())
                            torch.nn.utils.clip_grad_norm_(all_p, self.grad_clip)
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
                    for p in list(self.model.parameters()) + list(self.prompt_encoder.parameters()):
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5 if grad_norm > 0 else 0.0

                    # Compute pred/target stats
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

                    lr_curr = float(self.optimizer.param_groups[0]["lr"])
                    valid_px = int(
                        losses.get("valid_pixel_count", losses.get("valid_weight_sum", 0)).item()
                    )
                    total_px = loss_mask.numel()
                    valid_fraction = valid_px / max(total_px, 1)
                    amp_scale = self._amp_scaler.get_scale() if self.use_amp else 0.0

                    step_data = {
                        "epoch": epoch,
                        "step": batch_idx,
                        "global_step": global_step,
                        "lr": lr_curr,
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
                        line = (
                            f"  E{epoch:3d} S{batch_idx:5d} | "
                            f"loss={losses['total_loss'].item():.4f} surf={losses['surface_loss'].item():.4f} "
                            f"root={losses['rootzone_loss'].item():.4f} | "
                            f"valid={valid_fraction:.3f} g={grad_norm:.2e} | "
                            f"pred_s={pred_s_mean:.3f}/{pred_s_std:.3f} pred_r={pred_r_mean:.3f}/{pred_r_std:.3f} | "
                            f"true_s={target_s_mean:.3f}/{target_s_std:.3f} true_r={target_r_mean:.3f}/{target_r_std:.3f} | "
                            f"gpu={gpu_alloc:.1f}GB {batches_per_sec:.1f}b/s | lr={lr_curr:.2e} "
                            f"amp_scale={amp_scale:.0f} skip={self._skipped_steps}"
                        )
                        if self.run_manager is not None:
                            self.run_manager.log_console(line)
                        else:
                            print(line, flush=True)

                    # Wandb step log
                    if self.wandb_logger is not None and self.wandb_logger.enabled:
                        self.wandb_logger.log_step({
                            "train/total_loss": float(losses["total_loss"].item()),
                            "train/surface_loss": float(losses["surface_loss"].item()),
                            "train/rootzone_loss": float(losses["rootzone_loss"].item()),
                            "train/lr": lr_curr,
                            "train/grad_norm": grad_norm,
                            "train/valid_pixel_fraction": valid_fraction,
                            "train/pred_inc_surface_std": pred_s_std,
                            "train/pred_inc_rootzone_std": pred_r_std,
                            "train/gpu_memory_gb": gpu_alloc,
                            "train/skipped_steps": self._skipped_steps,
                        })

                global_step += 1

            avg_loss = float(np.mean(epoch_losses))
            avg_surface = float(np.mean(epoch_surface_losses))
            avg_rootzone = float(np.mean(epoch_rootzone_losses))
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
                    # Track prompt quality
                    if "prompt_quality" in gain_results:
                        pq = gain_results["prompt_quality"]
                        pq["epoch"] = epoch
                        self.prompt_quality_history.append(pq)
                if verbose and gain_results:
                    sv_loss = gain_results.get("source_val_loss", float("nan"))
                    sv_rmse_s = gain_results.get("rmse_surface_model", float("nan"))
                    sv_rmse_r = gain_results.get("rmse_rootzone_model", float("nan"))
                    sv_skill_s = gain_results.get("skill_surface_with_alpha", float("nan"))
                    sv_skill_r = gain_results.get("skill_rootzone_with_alpha", float("nan"))
                    alpha_s = gain_results.get("best_alpha_surface", 1.0)
                    alpha_r = gain_results.get("best_alpha_rootzone", 1.0)
                    sel_score = gain_results.get("selection_score", float("-inf"))
                    calib_mode = gain_results.get("calibration_mode", "shared")
                    print(
                        f"  source_val loss={sv_loss:.6f}"
                        f"  rmse_s={sv_rmse_s:.6f} r={sv_rmse_r:.6f}"
                        f"  skill_s={sv_skill_s:.4f} r={sv_skill_r:.4f}"
                        f"  alpha_s={alpha_s:.3f} alpha_r={alpha_r:.3f}"
                        f"  sel_score={sel_score:.4f} [{calib_mode}]",
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

            # Best checkpoint selection based on selection_metric
            selected_value, maximize_selection = self._selection_value(
                train_loss=avg_loss,
                source_val_metrics=source_val_metrics,
                gain_results=gain_results,
            )
            current_best = self.best_selection_value
            is_best = (
                selected_value > current_best
                if maximize_selection
                else selected_value < current_best
            )
            safe_score = (
                float(gain_results["selection_score"])
                if gain_results and "selection_score" in gain_results
                else float("-inf")
            )
            is_best_safe_score = safe_score > self.best_safe_score

            if is_best:
                self.best_selection_metric = self.selection_metric
                self.best_selection_value = selected_value
                if maximize_selection:
                    self.best_safe_score = selected_value
                else:
                    self.best_loss = selected_value
                ckpt_path = self.checkpoint_dir / "best.pt"
                self.save_checkpoint(
                    ckpt_path,
                    epoch,
                    selected_value,
                    "best",
                    gain_results=gain_results,
                    selection_value=selected_value,
                )

            # Save safe_score / transfer_safe_score checkpoint
            if is_best_safe_score:
                self.best_safe_score = safe_score
                tag = "best_source_val_transfer_safe_score"
                self.save_checkpoint(
                    self.checkpoint_dir / f"checkpoint_{tag}.pt",
                    epoch,
                    safe_score,
                    tag,
                    gain_results=gain_results,
                    selection_value=safe_score,
                )

            # Always save last.pt
            self.save_checkpoint(
                self.checkpoint_dir / "last.pt", epoch, avg_loss, "last",
                gain_results=gain_results,
                selection_value=selected_value,
            )

            # Epoch checkpoints
            self.save_checkpoint(
                self.checkpoint_dir / "checkpoint_latest.pt", epoch, avg_loss, "latest",
                gain_results=gain_results,
                selection_value=selected_value,
            )
            if (epoch + 1) % self.checkpoint_every_n_epochs == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                    epoch, avg_loss, f"epoch_{epoch:03d}",
                    gain_results=gain_results,
                    selection_value=selected_value,
                )

            record = {
                "epoch": epoch,
                "surface_loss": avg_surface,
                "rootzone_loss": avg_rootzone,
                "total_loss": avg_loss,
                "valid_pixel_count": total_valid,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "elapsed_s": elapsed,
                "skipped_steps": self._skipped_steps,
            }
            record.update(source_val_metrics)
            if gain_results:
                record["selection_score"] = gain_results.get("selection_score", float("nan"))
                record["best_alpha_surface"] = gain_results.get("best_alpha_surface", 1.0)
                record["best_alpha_rootzone"] = gain_results.get("best_alpha_rootzone", 1.0)
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
                else:
                    print(epoch_summary, flush=True)

                # Per-epoch divider every 5 epochs or at epoch 0
                if epoch == 0 or (epoch + 1) % 5 == 0:
                    divider = "  " + "-" * 40
                    if self.run_manager is not None:
                        self.run_manager.log_console(divider)
                    else:
                        print(divider, flush=True)

        # --- Training end summary ---
        total_elapsed = time.time() - train_start_time
        end_lines = [
            "=" * 60,
            f"Training Complete",
            f"  Total epochs:     {self.current_epoch}",
            f"  Best selection:   {self.best_selection_metric}={self.best_selection_value:.6f}",
            f"  Best loss:        {self.best_loss:.6f}",
            f"  Best safe score:  {self.best_safe_score:.6f}",
            f"  Skipped steps:    {self._skipped_steps}",
            f"  Total time:       {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)",
            f"  Checkpoint dir:   {self.checkpoint_dir}",
            "=" * 60,
        ]
        for line in end_lines:
            if self.run_manager is not None:
                self.run_manager.log_console(line)
            elif verbose:
                print(line, flush=True)

        # Save CSV output files
        self._save_output_csvs()

        # Close console.log if opened
        if self.run_manager is not None:
            self.run_manager.close_console_log()

        return self.train_history

    def _save_output_csvs(self) -> None:
        """Save all CSV output files to checkpoint_dir."""
        ckpt_dir = self.checkpoint_dir

        # metrics_train.csv
        if self.train_history:
            train_path = ckpt_dir / "metrics_train.csv"
            with open(train_path, "w", newline="") as f:
                fieldnames = list(self.train_history[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.train_history)

        # metrics_source_val_by_epoch.csv
        if self.val_history:
            val_path = ckpt_dir / "metrics_source_val_by_epoch.csv"
            with open(val_path, "w", newline="") as f:
                fieldnames = list(self.val_history[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.val_history)

        # prompt_quality_by_epoch.csv
        if self.prompt_quality_history:
            pq_path = ckpt_dir / "prompt_quality_by_epoch.csv"
            with open(pq_path, "w", newline="") as f:
                fieldnames = list(self.prompt_quality_history[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.prompt_quality_history)

        # README_training_summary.md
        self._save_readme()

    def _save_readme(self) -> None:
        """Save a human-readable README with training summary."""
        num_model_params = sum(p.numel() for p in self.model.parameters())
        num_pe_params = sum(p.numel() for p in self.prompt_encoder.parameters())
        has_source_val = self.source_val_dataset is not None

        lines = [
            f"# Prompt-Conditioned Training Summary",
            f"",
            f"- **Experiment**: {self.experiment_id}",
            f"- **Protocol**: {self.protocol_freeze_id}",
            f"- **Split manifest**: {self.split_manifest_path}",
            f"- **Model width**: {self.model_width}",
            f"- **Prompt dim**: {self.prompt_dim}",
            f"- **Model params**: {num_model_params:,}",
            f"- **Prompt encoder params**: {num_pe_params:,}",
            f"- **Total params**: {num_model_params + num_pe_params:,}",
            f"- **Batch size**: {self.batch_size}",
            f"- **Accum steps**: {self.accum_steps}",
            f"- **Effective batch size**: {self.batch_size * self.accum_steps}",
            f"- **Max epochs**: {self.max_epochs}",
            f"- **LR**: {self.lr}",
            f"- **Weight decay**: {self.weight_decay}",
            f"- **Grad clip**: {self.grad_clip}",
            f"- **AMP**: {self.use_amp}",
            f"- **Loss fn**: {type(self.loss_fn).__name__}",
            f"- **Lat-weighted**: {self.use_lat_weighted_loss}",
            f"- **Lambda amp**: {self.lambda_amp}",
            f"- **Inc normalization**: {self.target_increment_normalization}",
            f"- **Zero raw init**: {self.zero_raw_increment_init}",
            f"- **Selection metric**: {self.selection_metric}",
            f"- **Residual gain**: {self.source_val_residual_gain}",
            f"- **Source regions**: {self.source_regions}",
            f"- **Train samples**: {len(self.train_dataset)}",
            f"- **Source val available**: {has_source_val}",
            f"- **Normalization source**: source_fit_only",
            f"- **Early stopping source**: {'source_val_only' if has_source_val else 'train_loss_only'}",
            f"- **Model selection source**: {'source_val_only' if has_source_val else 'best_train_loss'}",
            f"- **Target eval usage**: eval_only_no_early_stopping",
            f"- **Leakage guard**: pass",
            f"- **Best selection metric**: {self.best_selection_metric}",
            f"- **Best selection value**: {self.best_selection_value:.6f}",
            f"- **Best loss**: {self.best_loss:.6f}",
            f"- **Best safe score**: {self.best_safe_score:.6f}",
            f"- **Skipped steps**: {self._skipped_steps}",
            f"- **Epochs completed**: {self.current_epoch}",
            f"- **Git hash**: {get_git_hash()}",
            f"- **Timestamp**: {get_timestamp()}",
            f"",
            f"## Output Files",
            f"",
            f"- `metrics_train.csv` — per-epoch training metrics",
            f"- `metrics_source_val_by_epoch.csv` — per-epoch source_val metrics",
            f"- `prompt_quality_by_epoch.csv` — per-epoch prompt quality metrics",
            f"- `alpha_selection_trace.csv` — 2D alpha grid selection trace",
            f"- `summary.json` — training summary",
            f"- `checkpoint_latest.pt` — latest epoch checkpoint",
            f"- `checkpoint_best_source_val_transfer_safe_score.pt` — best transfer-safe checkpoint",
            f"",
        ]
        readme_path = self.checkpoint_dir / "README_training_summary.md"
        with open(readme_path, "w") as f:
            f.write("\n".join(lines))

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        loss: float,
        tag: str = "",
        gain_results: Optional[Dict[str, Any]] = None,
        selection_value: Optional[float] = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved_selection_value = float(loss if selection_value is None else selection_value)
        checkpoint: Dict[str, Any] = {
            "tag": tag,
            "epoch": epoch,
            "loss": loss,
            "best_loss": self.best_loss,
            "best_safe_score": self.best_safe_score,
            "best_selection_metric": self.best_selection_metric,
            "best_selection_value": self.best_selection_value,
            "selection_metric": self.selection_metric,
            "selection_value": resolved_selection_value,
            "experiment_id": self.experiment_id,
            "protocol_freeze_id": self.protocol_freeze_id,
            "split_manifest_path": self.split_manifest_path,
            "git_hash": get_git_hash(),
            "timestamp": get_timestamp(),
            "model_state_dict": self.model.state_dict(),
            "prompt_encoder_state_dict": self.prompt_encoder.state_dict(),
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
                "prompt_dim": self.prompt_dim,
                "num_regions": self.prompt_encoder.num_regions,
                "num_workers": self.num_workers,
                "target_increment_normalization": self.target_increment_normalization,
                "zero_raw_increment_init": self.zero_raw_increment_init,
                "use_amp": self.use_amp,
                "use_lat_weighted_loss": self.use_lat_weighted_loss,
                "log_every_steps": self.log_every_steps,
                "checkpoint_every_n_epochs": self.checkpoint_every_n_epochs,
                "lambda_amp": self.lambda_amp,
                "selection_metric": self.selection_metric,
                "source_val_residual_gain": self.source_val_residual_gain,
                "target_region": self.target_region,
                "adaptation_setting": self.adaptation_setting,
                "K": self.K,
                "protocol_freeze_id": self.protocol_freeze_id,
                "selection_value": resolved_selection_value,
                "best_selection_metric": self.best_selection_metric,
                "best_selection_value": self.best_selection_value,
                "ch_mean": self._ch_mean.tolist() if self._ch_mean is not None else None,
                "ch_std": self._ch_std.tolist() if self._ch_std is not None else None,
                "inc_mean": self._inc_mean.tolist() if self._inc_mean is not None else None,
                "inc_std": self._inc_std.tolist() if self._inc_std is not None else None,
                "source_regions": self.source_regions,
                "source_region_global_indices": self._source_region_global_indices,
            },
        }
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

    def load_state(self, checkpoint: Dict[str, Any]) -> int:
        """Restore full training state from a checkpoint. Returns resumed epoch."""
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.prompt_encoder.load_state_dict(checkpoint["prompt_encoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_loss = float(checkpoint.get("best_loss", float("inf")))
        self.best_safe_score = checkpoint.get(
            "best_safe_score",
            checkpoint.get("selection_score", float("-inf")),
        )
        self.best_selection_metric = checkpoint.get("best_selection_metric", self.selection_metric)
        if "best_selection_value" in checkpoint:
            self.best_selection_value = float(checkpoint["best_selection_value"])
        elif self.best_selection_metric == "source_val_transfer_safe_score":
            self.best_selection_value = float(self.best_safe_score)
        else:
            self.best_selection_value = float(self.best_loss)
        self._skipped_steps = checkpoint.get("config", {}).get("skipped_steps", 0)
        self.train_history = checkpoint.get("train_history", [])
        self.val_history = checkpoint.get("val_history", [])
        self.prompt_quality_history = checkpoint.get("prompt_quality_history", [])
        resumed_epoch = checkpoint["epoch"] + 1
        self.current_epoch = resumed_epoch

        # Restore normalization stats from checkpoint
        if checkpoint["config"].get("ch_mean") is not None:
            self._ch_mean = np.array(checkpoint["config"]["ch_mean"], dtype=np.float32)
        if checkpoint["config"].get("ch_std") is not None:
            self._ch_std = np.array(checkpoint["config"]["ch_std"], dtype=np.float32)
        if checkpoint["config"].get("inc_mean") is not None:
            self._inc_mean = np.array(checkpoint["config"]["inc_mean"], dtype=np.float32)
        if checkpoint["config"].get("inc_std") is not None:
            self._inc_std = np.array(checkpoint["config"]["inc_std"], dtype=np.float32)

        return resumed_epoch

    def save_summary_json(self, path: Optional[Path] = None) -> None:
        has_source_val = self.source_val_dataset is not None
        num_params = sum(p.numel() for p in self.model.parameters()) + sum(p.numel() for p in self.prompt_encoder.parameters())
        summary: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "protocol_freeze_id": self.protocol_freeze_id,
            "best_loss": self.best_loss,
            "best_safe_score": self.best_safe_score,
            "best_selection_metric": self.best_selection_metric,
            "best_selection_value": self.best_selection_value,
            "final_epoch": self.current_epoch - 1,
            "total_epochs_completed": self.current_epoch,
            "model_width": self.model_width,
            "prompt_dim": self.prompt_dim,
            "trainable_parameters": num_params,
            "batch_size": self.batch_size,
            "accum_steps": self.accum_steps,
            "effective_batch_size": self.batch_size * self.accum_steps,
            "source_val_available": has_source_val,
            "use_lat_weighted_loss": self.use_lat_weighted_loss,
            "loss_function": type(self.loss_fn).__name__,
            "lambda_amp": self.lambda_amp,
            "selection_metric": self.selection_metric,
            "source_val_residual_gain": self.source_val_residual_gain,
            "source_regions": self.source_regions,
            "split_manifest_sha256": self.split_manifest_sha256,
            "normalization_source": "source_fit_only",
            "early_stopping_source": "source_val_only" if has_source_val else "train_loss_only",
            "model_selection_source": "source_val_only" if has_source_val else "best_train_loss",
            "target_eval_usage": "eval_only_no_early_stopping",
            "target_query_usage": "eval_only_no_early_stopping",
            "leakage_guard_status": "pass",
            "skipped_steps": self._skipped_steps,
            "git_hash": get_git_hash(),
            "timestamp": get_timestamp(),
            "train_history": self.train_history,
            "val_history": self.val_history,
        }
        target_path = path or (self.checkpoint_dir / "summary.json")
        with open(target_path, "w") as f:
            json.dump(summary, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train prompt-conditioned shared backbone")
    parser.add_argument("--target_region", type=str, required=True)
    parser.add_argument("--adaptation_setting", type=str, default="target_full_train",
        help="Split adaptation setting (default: target_full_train; legacy example: legacy_few_shot_k4)")
    parser.add_argument("--K", type=int, default=None,
        help="Legacy few-shot K value. Ignored for target_full_train.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--prompt_dim", type=int, default=64)
    parser.add_argument("--zero_raw_increment_init", action="store_true")
    parser.add_argument("--target_increment_normalization", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
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
    parser.add_argument("--use_lat_weighted_loss", action="store_true", default=True,
        help="Use WeightedMaskedHuberLoss with latitude weighting (default True)")
    parser.add_argument("--no_lat_weighted_loss", action="store_false", dest="use_lat_weighted_loss",
        help="Disable latitude-weighted loss (use plain MaskedHuberLoss)")
    parser.add_argument("--resume_from", type=str, default=None,
        help="Path to checkpoint.pt to resume from (last.pt or best.pt). "
             "When provided, training continues from the saved epoch and "
             "normalization stats are restored from the checkpoint.")
    parser.add_argument("--checkpoint_every", type=int, default=5,
        help="Save epoch checkpoint every N epochs (default 5)")
    parser.add_argument("--selection_metric", type=str, default="source_val_transfer_safe_score",
        choices=["source_val_transfer_safe_score", "source_val_loss", "train_loss"],
        help="Metric for checkpoint selection (default source_val_transfer_safe_score)")
    parser.add_argument("--lambda_amp", type=float, default=0.0,
        help="Amplitude penalty weight for WeightedMaskedHuberLoss (default 0.0 = disabled)")
    parser.add_argument("--no_source_val_residual_gain", action="store_true",
        help="Disable residual gain calibration (use alpha=1.0 for both variables)")
    parser.add_argument("--cuda_sync_debug", action="store_true",
        help="Enable CUDA synchronize after each forward pass (debug only)")
    args = parser.parse_args()
    if args.adaptation_setting == "target_full_train":
        args.K = None
    elif args.K is None:
        args.K = 0
    return args


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    device = resolve_device(args.device, require_gpu=args.require_gpu)

    print("=" * 60)
    print("Phase 4B: Prompt-Conditioned Shared Backbone Training")
    print(f"  target_region={args.target_region}  adaptation_setting={args.adaptation_setting}  K={args.K}  seed={args.seed}")
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
        "target_region": args.target_region,
        "adaptation_setting": args.adaptation_setting,
        "K": args.K,
        "seed": args.seed,
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
        "use_lat_weighted_loss": args.use_lat_weighted_loss,
        "wandb_mode": args.wandb_mode,
        "checkpoint_every": args.checkpoint_every,
        "selection_metric": args.selection_metric,
        "lambda_amp": args.lambda_amp,
        "source_val_residual_gain": not args.no_source_val_residual_gain,
        "source_regions": [r for r in _ALL_US_REGIONS if r != args.target_region],
    }

    # Resolve output_dir for RunManager BEFORE creating it.
    # If resuming, auto-derive from checkpoint path so run_name is consistent.
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume_from checkpoint not found: {resume_path}")
        if args.output_dir is None:
            args.output_dir = str(resume_path.parent.parent)
            print(f"[resume] output_dir auto-derived from checkpoint: {args.output_dir}")

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
        adaptation_setting=args.adaptation_setting,
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
        adaptation_setting=args.adaptation_setting,
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

    # Optional resume: pre-load checkpoint before creating Trainer
    resumed_epoch = 0
    ckpt = None

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume_from checkpoint not found: {resume_path}")
        print(f"\nResuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        resumed_epoch = ckpt["epoch"] + 1
        print(f"  checkpoint epoch={ckpt['epoch']}  best_loss={ckpt.get('best_loss', 'N/A')}")
        print(f"  resuming from epoch {resumed_epoch} ({resumed_epoch} already completed)")

        # Auto-derive output_dir from checkpoint path (already done before RunManager creation)

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
        global_to_source_lookup=global_to_source_idx,
        use_lat_weighted_loss=args.use_lat_weighted_loss,
        source_regions=source_regions,
        checkpoint_every_n_epochs=args.checkpoint_every,
        lambda_amp=args.lambda_amp,
        selection_metric=args.selection_metric,
        source_val_residual_gain=not args.no_source_val_residual_gain,
        cuda_sync_debug=args.cuda_sync_debug,
        target_region=args.target_region,
        adaptation_setting=args.adaptation_setting,
        K=args.K,
    )

    # Resume: restore full training state after Trainer creation
    if resumed_epoch > 0 and ckpt is not None:
        print(f"\nRestoring training state from checkpoint (resuming from epoch {resumed_epoch})...")
        trainer.load_state(ckpt)
        print(f"  Restored: optimizer, scheduler, epoch, best_loss, train_history")
        print(f"  train_history entries so far: {len(trainer.train_history)}")
        print(f"  val_history entries so far: {len(trainer.val_history)}")

    run_manager.save_environment_info(gather_runtime_info())

    # Train
    print(f"\nStarting training...")
    history = trainer.train(verbose=True)

    elapsed = time.time() - start_time

    # Save summary
    summary_path = run_manager.summary_json_path()
    trainer.save_summary_json(summary_path)

    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  best_selection_metric={trainer.best_selection_metric}")
    print(f"  best_selection_value={trainer.best_selection_value:.6f}")
    print(f"  best_loss={trainer.best_loss:.6f}")
    print(f"  best_safe_score={trainer.best_safe_score:.6f}")
    print(f"  skipped_steps={trainer._skipped_steps}")
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
