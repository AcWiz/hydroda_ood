#!/usr/bin/env python3
"""Train prompt-conditioned shared backbone (FiLMConditionalResUNet + RegionPromptEncoder).

Multi-region mixed training: source_fit (2015-2021) from all source regions.
Each sample's prompt uses the region embedding for the region with the most
valid pixels in the sample.

Usage:
    PYTHONPATH=. python scripts/train/train_prompt_conditioned_shared.py \\
        --target_region US-R1 --adaptation_setting zero_shot_context --K 0 --seed 0 \\
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
import hashlib
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
from torch.utils.data import BatchSampler, DataLoader

from hydroda.data.dataset import HydroDADataset, build_hydroda_dataset, collate_hydroda_samples
from hydroda.data.file_hash import compute_sha256
from hydroda.data.leakage_guard import LeakageGuard
from hydroda.data.protocol import ProtocolConfig
from hydroda.models.conditional_unet import FiLMConditionalResUNet
from hydroda.models.hyper_conditional_unet import (
    HyperAdapterConditionalResUNet,
    SOURCE_SALIENCY_PRIOR_APPLICATIONS,
)
from hydroda.models.source_saliency import ADAPTER_LAYER_NAMES, load_source_saliency_prior, tensor_sha256
from hydroda.baselines.prompt_conditioned import (
    RELIABILITY_FEATURE_TRANSFORM,
    bounded_reliability_features,
    build_target_context_prompt_state,
    prompt_channel_11_usage,
    prompt_diagnostic_input_domain,
    prompt_diagnostic_tensor,
    prompt_domain_metadata,
    prompt_input_feature_source,
    prompt_normalized_input_used,
    masked_input_embedding_and_coverage,
)
from hydroda.models.prompt_encoder import RegionPromptEncoder, RobustInputSideDAPromptEncoder
from hydroda.training.calibration import calibrate_residual_gain, calibrate_residual_gain_region_aware
from hydroda.training.losses import MaskedHuberLoss, WeightedMaskedHuberLoss
from hydroda.metrics.skill import weighted_analysis_skill_components
from hydroda.utils.run_manager import RunManager
from hydroda.utils.logger import WandbLogger, ConsoleLogger
from hydroda.utils.device import resolve_device, log_device_summary, gpu_health_check
from hydroda.utils.runtime import gather_runtime_info, get_git_hash, get_timestamp


DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_zero_few_shot_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
CHECKPOINT_DIR = "artifacts/checkpoints/phase4_prompt_conditioned"
PROTOCOL_FREEZE_ID = "hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025"
PHASE = "phase4_prompt_conditioned"

_ALL_US_REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
_GLOBAL_REGION_IDX_MAP = {r: i for i, r in enumerate(_ALL_US_REGIONS)}
CONTEXT_ENCODERS = (
    "current_mean_std",
    "robust_input_side_da_diagnostics",
    "robust_input_side_da_diagnostics_raw",
)
TRAINABLE_SCOPES = ("all", "source_base_frozen_adapter_film")
SOURCE_BASE_FROZEN_MODULES = ("enc1", "enc2", "enc3", "bottleneck", "dec2", "dec1", "head")
SOURCE_BASE_TRAINABLE_MODULES = (
    "film1",
    "film2",
    "film3",
    "film_b",
    "hyper_adapter_b",
    "hyper_adapter_d2",
    "hyper_adapter_d1",
    "shared_coeff_generator",
    "reliability_gate",
    "residual_head",
    "source_residual_gate_net",
)
ZERO_SHOT_PRIOR_FORMS = (
    "direct_hyper",
    "source_residual_prior",
    "source_base_residual_reliability_gated",
)
SOURCE_RESIDUAL_GATES = ("none", "prompt_reliability_scalar")
SOURCE_EPISODE_PROMPT_POLICIES = ("current_region_prompt", "context_monthly_prototype")
SOURCE_REGION_EPISODE_POLICY = "per_source_region_active_mask_episode_v1"
SOURCE_PROTOTYPE_CACHE_MODES = ("off", "read_write", "refresh")
SOURCE_PROTOTYPE_CACHE_SCHEMA_VERSION = "source_context_monthly_prototype_cache_v1"
TRAIN_BATCH_SAMPLERS = ("random", "source_region_year_grouped")
TENSOR_CACHE_LOAD_MODES = ("eager", "mmap")


class SourceRegionYearGroupedBatchSampler(BatchSampler):
    """Batch source episodes by `(sample_region_id, year)` while covering all indices."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        *,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        records = getattr(dataset, "_date_records", None)
        if records is None:
            raise ValueError(
                "source_region_year_grouped sampler requires dataset._date_records "
                "with sample_region_id/date_str metadata"
            )
        if len(records) != len(dataset):
            raise ValueError(
                "source_region_year_grouped sampler requires one _date_records entry per sample; "
                f"got records={len(records)} samples={len(dataset)}"
            )
        self._groups: Dict[Tuple[str, int], List[int]] = {}
        for idx, record in enumerate(records):
            region_id = record.get("sample_region_id")
            if not region_id:
                active_ids = record.get("active_region_ids") or []
                region_id = active_ids[0] if len(active_ids) == 1 else None
            date_str = str(record.get("date_str", ""))
            if not region_id or len(date_str) < 4:
                raise ValueError(
                    "source_region_year_grouped sampler records must include "
                    f"sample_region_id and date_str year; bad record at idx={idx}: {record}"
                )
            self._groups.setdefault((str(region_id), int(date_str[:4])), []).append(idx)
        self._length = 0
        for indices in self._groups.values():
            full, remainder = divmod(len(indices), self.batch_size)
            self._length += full
            if remainder and not self.drop_last:
                self._length += 1

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        keys = list(self._groups.keys())
        rng.shuffle(keys)
        for key in keys:
            indices = list(self._groups[key])
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch
        self.epoch += 1

    def __len__(self) -> int:
        return self._length


def build_prompt_encoder(
    *,
    context_encoder: str,
    num_regions: int,
    input_channels: int,
    hidden_dim: int,
) -> RegionPromptEncoder:
    """Instantiate the configured source-stage prompt context encoder."""
    if context_encoder == "current_mean_std":
        return RegionPromptEncoder(
            num_regions=num_regions,
            input_channels=input_channels,
            hidden_dim=hidden_dim,
        )
    if context_encoder in {"robust_input_side_da_diagnostics", "robust_input_side_da_diagnostics_raw"}:
        return RobustInputSideDAPromptEncoder(
            num_regions=num_regions,
            input_channels=input_channels,
            hidden_dim=hidden_dim,
        )
    raise ValueError(f"Unsupported context_encoder: {context_encoder}")


def resolve_context_encoder_from_checkpoint(checkpoint: Dict[str, Any]) -> str:
    """Return checkpoint context encoder, defaulting old checkpoints safely."""
    config = checkpoint.get("config", {})
    context_encoder = config.get("context_encoder", "current_mean_std")
    if context_encoder not in CONTEXT_ENCODERS:
        raise ValueError(f"Unsupported checkpoint context_encoder: {context_encoder}")
    return str(context_encoder)


def _as_float_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(v) for v in value]


def _load_source_saliency_prior_for_model(
    *,
    path: Optional[str],
    n_basis: int,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    if not path:
        return None, {
            "enabled": False,
            "path": "",
            "beta": 0.0,
            "prior_shape": [len(ADAPTER_LAYER_NAMES), int(n_basis)],
            "source": "none",
        }
    prior, metadata = load_source_saliency_prior(
        path,
        expected_n_layers=len(ADAPTER_LAYER_NAMES),
        expected_n_basis=int(n_basis),
    )
    metadata = dict(metadata)
    metadata.update(
        {
            "enabled": True,
            "path": str(path),
            "prior_shape": list(prior.shape),
            "prior_sha256": metadata.get("prior_sha256") or tensor_sha256(prior),
        }
    )
    return prior, metadata


def _source_saliency_metadata_for_config(
    *,
    path: str,
    beta: float,
    prior: Optional[torch.Tensor],
    application: str = "soft_regularization_metadata",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if application not in SOURCE_SALIENCY_PRIOR_APPLICATIONS:
        raise ValueError(
            "hyper_source_saliency_prior_application must be one of "
            f"{SOURCE_SALIENCY_PRIOR_APPLICATIONS}"
        )
    meta = dict(metadata or {})
    enabled = prior is not None and float(beta) > 0.0
    hard_routing = enabled and application == "legacy_gate_logit_bias_before_topk"
    meta.update(
        {
            "enabled": bool(enabled),
            "path": str(path or ""),
            "beta": float(beta),
            "prior_shape": list(prior.shape) if prior is not None else meta.get("prior_shape", []),
            "prior_sha256": tensor_sha256(prior) if prior is not None else meta.get("prior_sha256", ""),
            "application": str(application) if enabled else "disabled",
            "hard_routing_effect": "legacy_gate_logit_bias_before_topk" if hard_routing else "none",
            "regularization_role": (
                "source_side_soft_regularization_metadata"
                if enabled and application == "soft_regularization_metadata"
                else "legacy_diagnostic_hard_routing" if hard_routing else "disabled"
            ),
            "target_val_usage": meta.get("target_val_usage", "unused_in_main_protocol"),
            "target_eval_usage": meta.get("target_eval_usage", "final_eval_only_no_selection"),
        }
    )
    return meta


def _checkpoint_config_value(checkpoint: Dict[str, Any], key: str) -> Any:
    config = checkpoint.get("config", {})
    if isinstance(config, dict) and key in config:
        return config[key]
    return checkpoint.get(key)


def _json_sha256(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _array_sha256(*arrays: Optional[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        if array is None:
            digest.update(b"none")
            continue
        arr = np.asarray(array, dtype=np.float32)
        digest.update(str(tuple(arr.shape)).encode("utf-8"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _prompt_input_branch_sha256(prompt_encoder: nn.Module) -> str:
    """Hash the prompt input-summary branch that defines prototype embedding space."""
    digest = hashlib.sha256()
    digest.update(prompt_encoder.__class__.__name__.encode("utf-8"))
    digest.update(str(getattr(prompt_encoder, "input_channels", "")).encode("utf-8"))
    diagnostic_schema = getattr(prompt_encoder, "diagnostic_schema", None)
    if diagnostic_schema is not None:
        digest.update(json.dumps(list(diagnostic_schema), sort_keys=False).encode("utf-8"))
    input_proj = getattr(prompt_encoder, "input_proj", None)
    if input_proj is not None:
        digest.update(input_proj.__class__.__name__.encode("utf-8"))
        for name, tensor in sorted(input_proj.state_dict().items()):
            arr = tensor.detach().cpu().contiguous().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(arr.shape)).encode("utf-8"))
            digest.update(arr.tobytes())
    return digest.hexdigest()


def set_training_seed(seed: int) -> None:
    """Seed CPU/GPU RNGs so run metadata and cache keys are reproducible."""
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(name.startswith("module.") for name in state_dict):
        return state_dict
    return {
        (name[len("module."):] if name.startswith("module.") else name): tensor
        for name, tensor in state_dict.items()
    }


def validate_source_base_checkpoint_for_staged_init(
    *,
    checkpoint_path: str,
    expected_width: int,
    require_increment_stats: bool,
) -> Dict[str, Any]:
    """Validate staged HyperDA source-base checkpoint before run artifacts are created."""
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--init_from_source_base_checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    source_config = dict(checkpoint.get("config", {}))
    source_model_type = source_config.get("model_type")
    if source_model_type not in {None, "source_only", "source_only_backbone", "source_pooled_global_backbone"}:
        raise ValueError(
            "--init_from_source_base_checkpoint must point to a source-only SmallResUNet "
            f"checkpoint, got config.model_type={source_model_type!r}"
        )
    if "prompt_encoder_state_dict" in checkpoint:
        raise ValueError(
            "--init_from_source_base_checkpoint must point to a source-only checkpoint; "
            "prompt-conditioned checkpoints include prompt_encoder_state_dict"
        )
    if "model_state_dict" not in checkpoint:
        raise ValueError("source base checkpoint is missing model_state_dict")

    source_width = source_config.get("width", source_config.get("model_width"))
    if source_width is not None and int(source_width) != int(expected_width):
        raise ValueError(
            f"source base checkpoint width mismatch: checkpoint width={source_width}, "
            f"requested HyperDA width={expected_width}"
        )

    source_state = _strip_module_prefix(checkpoint["model_state_dict"])
    if any(name.startswith(("film", "hyper_adapter", "target_")) for name in source_state):
        raise ValueError(
            "--init_from_source_base_checkpoint must be a SmallResUNet source-only checkpoint, "
            "not a conditional/HyperDA checkpoint"
        )

    if _checkpoint_config_value(checkpoint, "ch_mean") is None or _checkpoint_config_value(checkpoint, "ch_std") is None:
        raise ValueError("source base checkpoint must contain ch_mean/ch_std normalization stats")
    if require_increment_stats and (
        _checkpoint_config_value(checkpoint, "inc_mean") is None
        or _checkpoint_config_value(checkpoint, "inc_std") is None
    ):
        raise ValueError(
            "source base checkpoint must contain inc_mean/inc_std when target_increment_normalization is enabled"
        )

    return checkpoint


def load_source_base_checkpoint_into_hyperda(
    *,
    model: HyperAdapterConditionalResUNet,
    checkpoint_path: str,
    expected_width: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Load SmallResUNet source-base weights into the shared HyperDA modules."""
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--init_from_source_base_checkpoint not found: {ckpt_path}")

    checkpoint = validate_source_base_checkpoint_for_staged_init(
        checkpoint_path=str(ckpt_path),
        expected_width=expected_width,
        require_increment_stats=False,
    )
    source_config = dict(checkpoint.get("config", {}))

    source_state = _strip_module_prefix(checkpoint["model_state_dict"])

    target_state = model.state_dict()
    shared_names = [
        name
        for name in target_state
        if name.startswith(tuple(f"{module}." for module in SOURCE_BASE_FROZEN_MODULES))
    ]
    loaded_state = dict(target_state)
    loaded_names: List[str] = []
    missing_names: List[str] = []
    shape_mismatches: List[str] = []
    for name in shared_names:
        if name not in source_state:
            missing_names.append(name)
            continue
        if tuple(source_state[name].shape) != tuple(target_state[name].shape):
            shape_mismatches.append(
                f"{name}: checkpoint {tuple(source_state[name].shape)} != model {tuple(target_state[name].shape)}"
            )
            continue
        loaded_state[name] = source_state[name]
        loaded_names.append(name)

    if missing_names:
        preview = ", ".join(missing_names[:5])
        raise ValueError(
            "source base checkpoint is missing shared SmallResUNet parameters "
            f"({len(missing_names)} missing, e.g. {preview})"
        )
    if shape_mismatches:
        preview = "; ".join(shape_mismatches[:5])
        raise ValueError(
            "source base checkpoint shared-parameter shape mismatch; check width. "
            f"{preview}"
        )

    model.load_state_dict(loaded_state, strict=True)
    resolved_path = ckpt_path.resolve()
    return {
        "checkpoint_path": str(resolved_path),
        "checkpoint_sha256": compute_sha256(resolved_path),
        "source_config": source_config,
        "loaded_parameter_names": loaded_names,
        "ch_mean": _as_float_list(_checkpoint_config_value(checkpoint, "ch_mean")),
        "ch_std": _as_float_list(_checkpoint_config_value(checkpoint, "ch_std")),
        "inc_mean": _as_float_list(_checkpoint_config_value(checkpoint, "inc_mean")),
        "inc_std": _as_float_list(_checkpoint_config_value(checkpoint, "inc_std")),
    }


def trainable_parameter_names(
    model: nn.Module,
    prompt_encoder: nn.Module,
) -> List[str]:
    names = [f"model.{name}" for name, param in model.named_parameters() if param.requires_grad]
    names.extend(
        f"prompt_encoder.{name}"
        for name, param in prompt_encoder.named_parameters()
        if param.requires_grad
    )
    return names


def trainable_parameters(model: nn.Module, prompt_encoder: nn.Module) -> List[nn.Parameter]:
    return [
        param
        for param in list(model.parameters()) + list(prompt_encoder.parameters())
        if param.requires_grad
    ]


def apply_trainable_scope(
    *,
    model: nn.Module,
    prompt_encoder: nn.Module,
    trainable_scope: str,
) -> Dict[str, Any]:
    """Apply source-stage trainability policy and return metadata."""
    if trainable_scope not in TRAINABLE_SCOPES:
        raise ValueError(f"Unsupported trainable_scope: {trainable_scope}")

    if trainable_scope == "all":
        for param in list(model.parameters()) + list(prompt_encoder.parameters()):
            param.requires_grad_(True)
        frozen_modules: List[str] = []
    else:
        if not isinstance(model, HyperAdapterConditionalResUNet):
            raise ValueError(
                "trainable_scope=source_base_frozen_adapter_film requires "
                "model_type=hyperda_basis_adapter"
            )
        if getattr(model, "enable_target_adaptation", False):
            raise ValueError("source-stage staged HyperDA must not enable target adaptation modules")
        for param in list(model.parameters()) + list(prompt_encoder.parameters()):
            param.requires_grad_(False)
        for param in prompt_encoder.parameters():
            param.requires_grad_(True)
        if hasattr(model, "source_stage_trainable_modules"):
            trainable_modules = model.source_stage_trainable_modules()
        else:
            trainable_modules = [
                (module_name, getattr(model, module_name, None))
                for module_name in SOURCE_BASE_TRAINABLE_MODULES
            ]
        for module_name, module in trainable_modules:
            if module is None:
                continue
            if (
                module_name.startswith("hyper_adapter")
                and getattr(model, "hyper_coeff_generator", "per_adapter")
                in {
                    "shared_layer_aware",
                    "shared_layer_aware_rank_gated",
                    "shared_layer_aware_rank_gated_stable",
                }
                and hasattr(module, "bases")
            ):
                for param in module.bases.parameters():
                    param.requires_grad_(True)
                basis_gain_delta = getattr(module, "basis_gain_delta", None)
                if basis_gain_delta is not None:
                    basis_gain_delta.requires_grad_(True)
                continue
            for param in module.parameters():
                param.requires_grad_(True)
        frozen_modules = list(SOURCE_BASE_FROZEN_MODULES)

    names = trainable_parameter_names(model, prompt_encoder)
    count = int(sum(param.numel() for param in trainable_parameters(model, prompt_encoder)))
    return {
        "trainable_scope": trainable_scope,
        "trainable_parameter_count": count,
        "trainable_parameter_names": names,
        "frozen_source_base_modules": frozen_modules,
    }


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
        adaptation_setting: str = "zero_shot_context",
        K: Optional[int] = None,
        context_encoder: str = "current_mean_std",
        model_type: str = "prompt_conditioned",
        hyper_n_basis: int = 8,
        hyper_adapter_bottleneck: Optional[int] = None,
        hyper_adapter_scale: float = 1.0,
        hyper_coeff_generator: str = "per_adapter",
        hyper_rank_gate_top_k: int = 4,
        hyper_rank_gate_temperature_init: float = 1.0,
        hyper_adapter_param_style: str = "basis_1x1",
        hyper_reliability_gate: str = "none",
        hyper_reliability_init: float = 0.95,
        hyper_source_saliency_prior: Optional[torch.Tensor] = None,
        hyper_source_saliency_prior_beta: float = 0.0,
        hyper_source_saliency_prior_path: str = "",
        hyper_source_saliency_prior_application: str = "soft_regularization_metadata",
        hyper_source_saliency_prior_metadata: Optional[Dict[str, Any]] = None,
        hyper_prompt_manifold_reliability: bool = False,
        hyper_prompt_manifold_reliability_strength: float = 0.0,
        hyper_enable_film: bool = True,
        hyper_enable_adapters: bool = True,
        hyper_residual_magnitude_penalty: float = 0.0,
        hyper_coeff_entropy_floor: float = 0.0,
        hyper_coeff_entropy_penalty: float = 0.0,
        zero_shot_prior_form: str = "direct_hyper",
        source_residual_rho: float = 1.0,
        source_residual_gate: str = "prompt_reliability_scalar",
        source_residual_gate_init: float = 0.95,
        source_residual_reliability_dim: int = 5,
        trainable_scope: str = "all",
        source_episode_prompt_policy: str = "current_region_prompt",
        source_anchor_blend_calibration: bool = False,
        hyper_output_head_residual: bool = False,
        init_from_source_base_checkpoint: Optional[str] = None,
        source_base_checkpoint_sha256: str = "",
        source_base_checkpoint_config: Optional[Dict[str, Any]] = None,
        source_base_loaded_parameter_names: Optional[List[str]] = None,
        dataset_backend: str = "netcdf",
        tensor_cache_load_mode: str = "eager",
        train_batch_sampler: str = "random",
        prefetch_factor: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        amp_init_scale: float = 256.0,
        amp_min_scale: float = 1.0,
        amp_skip_abort_threshold: int = 16,
        source_prototype_cache_dir: Optional[str] = None,
        source_prototype_cache_mode: str = "off",
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
        self.amp_init_scale = float(amp_init_scale)
        self.amp_min_scale = float(amp_min_scale)
        self.amp_skip_abort_threshold = int(amp_skip_abort_threshold)
        if self.use_amp:
            if self.amp_init_scale <= 0.0:
                raise ValueError("amp_init_scale must be positive")
            if self.amp_min_scale < 0.0:
                raise ValueError("amp_min_scale must be non-negative")
            if self.amp_skip_abort_threshold < 1:
                raise ValueError("amp_skip_abort_threshold must be >= 1")
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
        if context_encoder not in CONTEXT_ENCODERS:
            raise ValueError(f"Unsupported context_encoder: {context_encoder}")
        self.context_encoder = context_encoder
        if adaptation_setting == "target_full_train":
            self.K = None
        elif K is None and adaptation_setting == "zero_shot_context":
            self.K = 0
        else:
            self.K = K
        self.model_type = model_type
        self.hyper_n_basis = int(hyper_n_basis)
        self.hyper_adapter_bottleneck = hyper_adapter_bottleneck
        self.hyper_adapter_scale = float(hyper_adapter_scale)
        self.hyper_coeff_generator = str(hyper_coeff_generator)
        self.hyper_rank_gate_top_k = int(hyper_rank_gate_top_k)
        self.hyper_rank_gate_temperature_init = float(hyper_rank_gate_temperature_init)
        self.hyper_adapter_param_style = str(hyper_adapter_param_style)
        self.hyper_reliability_gate = str(hyper_reliability_gate)
        self.hyper_reliability_init = float(hyper_reliability_init)
        self.hyper_source_saliency_prior = (
            hyper_source_saliency_prior.detach().cpu()
            if hyper_source_saliency_prior is not None
            else None
        )
        self.hyper_source_saliency_prior_beta = float(hyper_source_saliency_prior_beta)
        self.hyper_source_saliency_prior_path = str(hyper_source_saliency_prior_path or "")
        if hyper_source_saliency_prior_application not in SOURCE_SALIENCY_PRIOR_APPLICATIONS:
            raise ValueError(
                "Unsupported hyper_source_saliency_prior_application: "
                f"{hyper_source_saliency_prior_application}"
            )
        self.hyper_source_saliency_prior_application = str(hyper_source_saliency_prior_application)
        self.hyper_source_saliency_prior_metadata = _source_saliency_metadata_for_config(
            path=self.hyper_source_saliency_prior_path,
            beta=self.hyper_source_saliency_prior_beta,
            prior=self.hyper_source_saliency_prior,
            application=self.hyper_source_saliency_prior_application,
            metadata=hyper_source_saliency_prior_metadata,
        )
        self.hyper_prompt_manifold_reliability = bool(hyper_prompt_manifold_reliability)
        self.hyper_prompt_manifold_reliability_strength = float(hyper_prompt_manifold_reliability_strength)
        self.hyper_enable_film = bool(hyper_enable_film)
        self.hyper_enable_adapters = bool(hyper_enable_adapters)
        self.hyper_residual_magnitude_penalty = float(hyper_residual_magnitude_penalty)
        self.hyper_coeff_entropy_floor = float(hyper_coeff_entropy_floor)
        self.hyper_coeff_entropy_penalty = float(hyper_coeff_entropy_penalty)
        if self.hyper_residual_magnitude_penalty < 0.0:
            raise ValueError("hyper_residual_magnitude_penalty must be non-negative")
        if self.hyper_coeff_entropy_floor < 0.0:
            raise ValueError("hyper_coeff_entropy_floor must be non-negative")
        if self.hyper_coeff_entropy_penalty < 0.0:
            raise ValueError("hyper_coeff_entropy_penalty must be non-negative")
        if zero_shot_prior_form not in ZERO_SHOT_PRIOR_FORMS:
            raise ValueError(f"Unsupported zero_shot_prior_form: {zero_shot_prior_form}")
        if source_residual_gate not in SOURCE_RESIDUAL_GATES:
            raise ValueError(f"Unsupported source_residual_gate: {source_residual_gate}")
        self.zero_shot_prior_form = str(zero_shot_prior_form)
        self.source_residual_rho = float(source_residual_rho)
        self.source_residual_gate = str(source_residual_gate)
        self.source_residual_gate_init = float(source_residual_gate_init)
        self.source_residual_reliability_dim = int(source_residual_reliability_dim)
        self.trainable_scope = trainable_scope
        if source_episode_prompt_policy not in SOURCE_EPISODE_PROMPT_POLICIES:
            raise ValueError(f"Unsupported source_episode_prompt_policy: {source_episode_prompt_policy}")
        self.source_episode_prompt_policy = source_episode_prompt_policy
        self.source_anchor_blend_calibration = bool(source_anchor_blend_calibration)
        self.hyper_output_head_residual = bool(hyper_output_head_residual)
        self.init_from_source_base_checkpoint = init_from_source_base_checkpoint
        self.source_base_checkpoint_sha256 = source_base_checkpoint_sha256
        self.source_base_checkpoint_config = source_base_checkpoint_config or {}
        self.source_base_loaded_parameter_names = source_base_loaded_parameter_names or []
        self.dataset_backend = str(dataset_backend)
        if tensor_cache_load_mode not in TENSOR_CACHE_LOAD_MODES:
            raise ValueError(f"Unsupported tensor_cache_load_mode: {tensor_cache_load_mode}")
        if train_batch_sampler not in TRAIN_BATCH_SAMPLERS:
            raise ValueError(f"Unsupported train_batch_sampler: {train_batch_sampler}")
        self.tensor_cache_load_mode = str(tensor_cache_load_mode)
        self.train_batch_sampler = str(train_batch_sampler)
        self.prefetch_factor = int(prefetch_factor) if prefetch_factor is not None else (2 if self.num_workers > 0 else None)
        self.pin_memory = bool(pin_memory) if pin_memory is not None else (self.device == "cuda")
        self.persistent_workers = self.num_workers > 0
        if source_prototype_cache_mode not in SOURCE_PROTOTYPE_CACHE_MODES:
            raise ValueError(f"Unsupported source_prototype_cache_mode: {source_prototype_cache_mode}")
        self.source_prototype_cache_mode = str(source_prototype_cache_mode)
        self.source_prototype_cache_dir = (
            Path(source_prototype_cache_dir)
            if source_prototype_cache_dir
            else None
        )
        self.source_prototype_cache_hit = False
        self.source_prototype_cache_path = ""
        self.source_prototype_cache_key = ""
        self.source_prototype_cache_metadata: Dict[str, Any] = {}
        self._source_region_global_indices = sorted(global_to_source_lookup.keys()) if global_to_source_lookup else []
        self.leakage_policy = {
            "train_split": "source_fit",
            "selection_split": "source_val",
            "normalization_source": (
                "source_fit_only_from_source_checkpoint"
                if self.init_from_source_base_checkpoint
                else "source_fit_only"
            ),
            "model_selection_source": "source_val_only" if source_val_dataset is not None else "best_train_loss",
            "target_context_usage": "input_side_only",
            "target_val_usage": "unused_in_main_protocol",
            "target_eval_usage": "final_eval_only_no_selection",
            "forbidden_fields": [
                "analysis_*",
                "increment_*",
                "prediction_error_*",
                "target_val",
                "target_eval",
                "channel_11_as_observation_or_region_mask",
            ],
        }

        # AMP
        self._amp_scaler: Optional[GradScaler] = None
        if self.use_amp:
            self._amp_scaler = GradScaler('cuda', init_scale=self.amp_init_scale)

        # Leakage guard
        protocol = ProtocolConfig()
        guard = LeakageGuard(protocol=protocol)
        train_date_strs = [d["date_str"] for d in self.train_dataset._date_records] if hasattr(self.train_dataset, "_date_records") else []
        guard.check_normalization_scope(train_date_strs, scope_name="source_fit_only")

        self._trainable_scope_metadata = apply_trainable_scope(
            model=self.model,
            prompt_encoder=self.prompt_encoder,
            trainable_scope=self.trainable_scope,
        )
        optimizer_params = trainable_parameters(self.model, self.prompt_encoder)
        if not optimizer_params:
            raise ValueError(f"trainable_scope={self.trainable_scope!r} produced no trainable parameters")
        self.optimizer = torch.optim.AdamW(optimizer_params, lr=lr, weight_decay=weight_decay)
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

        self._source_context_monthly_prototypes: Optional[Dict[str, Any]] = None
        needs_source_reliability = (
            isinstance(self.model, HyperAdapterConditionalResUNet)
            and getattr(self.model, "uses_source_residual_prior", False)
        )
        if self.source_episode_prompt_policy == "context_monthly_prototype" or needs_source_reliability:
            self._load_or_build_source_context_monthly_prototypes()

        # Zero-raw-init applies only to scratch training. A staged source-base
        # checkpoint already contains the trained frozen head.
        if self.zero_raw_increment_init and self.init_from_source_base_checkpoint:
            print("  zero_raw_increment_init: skipped because source base checkpoint is loaded")
        elif self.zero_raw_increment_init:
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
        self._consecutive_amp_skips = 0
        self._amp_failure_reason = ""
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []
        self.prompt_quality_history: List[Dict[str, float]] = []
        self._last_zero_shot_rho_selection: Dict[str, Any] = {
            "zero_shot_rho": float(self.source_residual_rho),
            "zero_shot_rho_grid": [float(v) for v in self.source_val_gain_grid],
            "zero_shot_rho_selection_source": "not_yet_evaluated",
            "zero_shot_rho_selection_reason": "not_yet_evaluated",
            "zero_shot_rho_trace": [],
        }

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

    def _resolved_config_metadata(self) -> Dict[str, Any]:
        """Return the protocol-relevant resolved config saved in artifacts."""
        return {
            "target_region": self.target_region,
            "adaptation_setting": self.adaptation_setting,
            "K": self.K,
            "model_type": self.model_type,
            "context_encoder": self.context_encoder,
            "prompt_diagnostic_schema": list(getattr(self.prompt_encoder, "diagnostic_schema", [])),
            "prompt_input_feature_source": prompt_input_feature_source(self.context_encoder),
            "prompt_channel_11_usage": prompt_channel_11_usage(self.context_encoder),
            "prompt_diagnostic_input_domain": prompt_diagnostic_input_domain(self.context_encoder),
            "normalized_input_used_for_prompt_diagnostics": prompt_normalized_input_used(self.context_encoder),
            "width": self.model_width,
            "prompt_dim": self.prompt_dim,
            "hyper_n_basis": self.hyper_n_basis,
            "hyper_adapter_bottleneck": self.hyper_adapter_bottleneck,
            "hyper_adapter_scale": self.hyper_adapter_scale,
            "hyper_coeff_generator": self.hyper_coeff_generator,
            "hyper_rank_gate_top_k": self.hyper_rank_gate_top_k,
            "hyper_rank_gate_temperature_init": self.hyper_rank_gate_temperature_init,
            "hyper_adapter_param_style": self.hyper_adapter_param_style,
            "hyper_reliability_gate": self.hyper_reliability_gate,
            "hyper_reliability_init": self.hyper_reliability_init,
            "hyper_source_saliency_prior_beta": self.hyper_source_saliency_prior_beta,
            "hyper_source_saliency_prior_path": self.hyper_source_saliency_prior_path,
            "hyper_source_saliency_prior_application": self.hyper_source_saliency_prior_application,
            "hyper_source_saliency_prior_metadata": self.hyper_source_saliency_prior_metadata,
            "hyper_source_saliency_prior": (
                self.hyper_source_saliency_prior.tolist()
                if self.hyper_source_saliency_prior is not None
                else None
            ),
            "hyper_prompt_manifold_reliability": self.hyper_prompt_manifold_reliability,
            "hyper_prompt_manifold_reliability_strength": self.hyper_prompt_manifold_reliability_strength,
            "hyper_enable_film": self.hyper_enable_film,
            "hyper_enable_adapters": self.hyper_enable_adapters,
            "hyper_residual_magnitude_penalty": self.hyper_residual_magnitude_penalty,
            "hyper_coeff_entropy_floor": self.hyper_coeff_entropy_floor,
            "hyper_coeff_entropy_penalty": self.hyper_coeff_entropy_penalty,
            "zero_shot_prior_form": self.zero_shot_prior_form,
            "source_residual_prior_mode": self.zero_shot_prior_form != "direct_hyper",
            "source_residual_rho": self.source_residual_rho,
            "source_residual_gate": self.source_residual_gate,
            "source_residual_gate_init": self.source_residual_gate_init,
            "source_residual_reliability_dim": self.source_residual_reliability_dim,
            "zero_shot_rho": self.source_residual_rho,
            "zero_shot_rho_grid": self.source_val_gain_grid,
            "zero_shot_rho_selection_source": self._last_zero_shot_rho_selection.get(
                "zero_shot_rho_selection_source",
                "source_val_regionwise_safe_episode_only",
            ),
            "zero_shot_rho_selection_reason": self._last_zero_shot_rho_selection.get(
                "zero_shot_rho_selection_reason",
                "",
            ),
            "zero_shot_rho_safe_policy": self._last_zero_shot_rho_selection.get(
                "zero_shot_rho_safe_policy",
                "source_safe_regionwise_non_degradation_surface_rootzone_vs_rho0",
            ),
            "zero_shot_rho_trace": self._last_zero_shot_rho_selection.get("zero_shot_rho_trace", []),
            "zero_shot_residual_formula": (
                "pred = source_base + rho * reliability_gate(prompt, context) * hyper_residual"
                if self.zero_shot_prior_form != "direct_hyper"
                else "direct_hyper"
            ),
            "reliability_feature_schema": getattr(self.model, "reliability_feature_schema", []),
            "reliability_feature_transform": RELIABILITY_FEATURE_TRANSFORM,
            "target_labels_used_for_adaptation": False,
            "target_val_usage": "unused_in_main_protocol",
            "target_eval_usage": "final_eval_only_no_selection",
            "target_eval_input_stats_used_for_update": False,
            "trainable_scope": self.trainable_scope,
            "source_episode_prompt_policy": self.source_episode_prompt_policy,
            "source_context_monthly_prototype_summary": self._source_context_monthly_prototype_summary(),
            "source_prototype_cache_mode": self.source_prototype_cache_mode,
            "source_prototype_cache_hit": self.source_prototype_cache_hit,
            "source_prototype_cache_path": self.source_prototype_cache_path,
            "source_prototype_cache_key": self.source_prototype_cache_key,
            "source_anchor_blend_calibration": self.source_anchor_blend_calibration,
            "hyper_output_head_residual": self.hyper_output_head_residual,
            "trainable_parameter_count": self._trainable_scope_metadata["trainable_parameter_count"],
            "trainable_parameter_names": self._trainable_scope_metadata["trainable_parameter_names"],
            "frozen_source_base_modules": self._trainable_scope_metadata["frozen_source_base_modules"],
            "init_from_source_base_checkpoint": self.init_from_source_base_checkpoint,
            "source_base_checkpoint_sha256": self.source_base_checkpoint_sha256,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "accum_steps": self.accum_steps,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "selection_metric": self.selection_metric,
            "source_val_residual_gain": self.source_val_residual_gain,
            "source_regions": self.source_regions,
            "source_region_global_indices": self._source_region_global_indices,
            "split_manifest_path": self.split_manifest_path,
            "protocol_freeze_id": self.protocol_freeze_id,
            "dataset_backend": self.dataset_backend,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "pin_memory": self.pin_memory,
            "tensor_cache_load_mode": self.tensor_cache_load_mode,
            "train_batch_sampler": self.train_batch_sampler,
            "eval_every_epochs": self.eval_every_epochs,
            "use_amp": self.use_amp,
            "amp_init_scale": self.amp_init_scale,
            "amp_min_scale": self.amp_min_scale,
            "amp_skip_abort_threshold": self.amp_skip_abort_threshold,
            "amp_failure_reason": self._amp_failure_reason,
        }

    def _compute_normalization_stats(self) -> None:
        print(f"Computing normalization stats from training dataset (n={len(self.train_dataset)})...")
        indices = list(range(len(self.train_dataset)))
        means, stds = _compute_channel_stats(self.train_dataset, indices)
        self._ch_mean = means
        self._ch_std = stds
        print(f"  Channel means: {means[:4]}...")
        print(f"  Channel stds:  {stds[:4]}...")

    def _source_prototype_cache_key_payload(self) -> Dict[str, Any]:
        dataset_split_sha = getattr(self.train_dataset, "split_manifest_sha256", "") or self.split_manifest_sha256
        return {
            "schema_version": SOURCE_PROTOTYPE_CACHE_SCHEMA_VERSION,
            "target_region": self.target_region,
            "source_regions": list(self.source_regions),
            "source_region_global_indices": list(self._source_region_global_indices),
            "split_manifest_sha256": dataset_split_sha,
            "protocol_freeze_id": self.protocol_freeze_id,
            "dataset_backend": self.dataset_backend,
            "context_encoder": self.context_encoder,
            "prompt_diagnostic_input_domain": prompt_diagnostic_input_domain(self.context_encoder),
            "normalized_input_used_for_prompt_diagnostics": prompt_normalized_input_used(self.context_encoder),
            "normalization_stats_hash": _array_sha256(self._ch_mean, self._ch_std),
            "prompt_encoder_input_branch_hash": _prompt_input_branch_sha256(self.prompt_encoder),
            "source_episode_prompt_policy": self.source_episode_prompt_policy,
            "episode_policy": SOURCE_REGION_EPISODE_POLICY,
        }

    def _source_prototype_cache_path_for_key(self, key_payload: Dict[str, Any]) -> Optional[Path]:
        if self.source_prototype_cache_dir is None:
            return None
        cache_key = _json_sha256(key_payload)
        self.source_prototype_cache_key = cache_key
        return self.source_prototype_cache_dir / f"{cache_key}.pt"

    def _load_or_build_source_context_monthly_prototypes(self) -> None:
        key_payload = self._source_prototype_cache_key_payload()
        cache_path = self._source_prototype_cache_path_for_key(key_payload)
        if self.source_prototype_cache_mode != "off" and cache_path is not None:
            self.source_prototype_cache_path = str(cache_path)
            if self.source_prototype_cache_mode != "refresh" and cache_path.exists():
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                metadata = dict(payload.get("metadata", {}))
                if metadata.get("key_payload") != key_payload:
                    raise ValueError(
                        f"source prototype cache key payload mismatch in {cache_path}; "
                        "refusing to load stale cache"
                    )
                self._source_context_monthly_prototypes = payload["prototypes"]
                self.source_prototype_cache_metadata = metadata
                self.source_prototype_cache_hit = True
                print(
                    "  source_context_monthly_prototype_cache: "
                    f"hit path={cache_path}",
                    flush=True,
                )
                return

        self.source_prototype_cache_hit = False
        self._build_source_context_monthly_prototypes()
        if self.source_prototype_cache_mode != "off" and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            summary = self._source_context_monthly_prototype_summary()
            metadata = {
                "schema_version": SOURCE_PROTOTYPE_CACHE_SCHEMA_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "key": self.source_prototype_cache_key,
                "key_payload": key_payload,
                "target_region": self.target_region,
                "source_regions": list(self.source_regions),
                "region_counts": summary.get("region_counts", []),
                "global_count": summary.get("global_count", 0),
                "region_month_prototype_count": summary.get("region_month_prototype_count", 0),
                "source": "source_fit_input_side_only_no_labels",
                **prompt_domain_metadata(self.context_encoder),
                "forbidden_fields_not_read": [
                    "target",
                    "analysis_surface",
                    "analysis_rootzone",
                    "increment_surface",
                    "increment_rootzone",
                    "target_val",
                    "target_eval",
                ],
            }
            torch.save(
                {
                    "metadata": metadata,
                    "prototypes": self._source_context_monthly_prototypes,
                },
                cache_path,
            )
            self.source_prototype_cache_path = str(cache_path)
            self.source_prototype_cache_metadata = metadata
            print(
                "  source_context_monthly_prototype_cache: "
                f"wrote path={cache_path}",
                flush=True,
            )

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

    def _build_source_context_monthly_prototypes(self) -> None:
        """Cache source_fit input-side monthly prompt summaries.

        The cached tensors contain only prompt input statistics derived from
        source_fit inputs and deployment-known month/region metadata. Labels,
        target_val, and target_eval are not read here.
        """
        num_regions = int(self.prompt_encoder.num_regions)
        input_emb_dim = int(self.prompt_encoder.input_proj.out_features)
        monthly_sums = torch.zeros(num_regions, 13, input_emb_dim, dtype=torch.float64)
        monthly_counts = torch.zeros(num_regions, 13, dtype=torch.long)
        monthly_coverage_sums = torch.zeros(num_regions, 13, dtype=torch.float64)
        region_sums = torch.zeros(num_regions, input_emb_dim, dtype=torch.float64)
        region_coverage_sums = torch.zeros(num_regions, dtype=torch.float64)
        region_counts = torch.zeros(num_regions, dtype=torch.long)
        global_sum = torch.zeros(input_emb_dim, dtype=torch.float64)
        global_count = 0
        global_coverage_sum = 0.0

        was_training = self.prompt_encoder.training
        self.prompt_encoder.eval()
        input_side_getter = getattr(self.train_dataset, "get_input_side_sample", None)
        with torch.no_grad():
            for idx in range(len(self.train_dataset)):
                sample = input_side_getter(idx) if callable(input_side_getter) else self.train_dataset[idx]
                sample_region_id = sample.get("sample_region_id")
                if sample_region_id:
                    global_rid = _GLOBAL_REGION_IDX_MAP[str(sample_region_id)]
                else:
                    active_ids = sample.get("active_region_ids") or []
                    if len(active_ids) != 1:
                        raise ValueError(
                            "source context prototype builder requires per-source-region episodes; "
                            f"got active_region_ids={active_ids!r}"
                        )
                    global_rid = _GLOBAL_REGION_IDX_MAP[str(active_ids[0])]
                if int(global_rid) not in self.global_to_source_lookup:
                    raise ValueError(
                        "source context prototype builder encountered non-source region "
                        f"global_rid={global_rid}"
                    )
                src_rid = int(self.global_to_source_lookup[global_rid])
                if src_rid < 0 or src_rid >= num_regions:
                    raise ValueError(
                        f"source context prototype compact region id {src_rid} outside "
                        f"prompt_encoder.num_regions={num_regions}"
                    )
                month = int(sample.get("month", 6))
                month = min(12, max(1, month))

                x = torch.as_tensor(sample["x"], dtype=torch.float32, device=self.device).unsqueeze(0)
                x_norm = self._normalize(x)
                x_prompt = prompt_diagnostic_tensor(
                    self.prompt_encoder,
                    context_encoder=self.context_encoder,
                    x_norm=x_norm,
                    x_raw=x,
                )
                region_mask = torch.as_tensor(
                    np.asarray(sample.get("region_mask", sample.get("active_region_mask"))) > 0.5,
                    dtype=torch.bool,
                    device=self.device,
                )
                input_emb, finite_coverage = masked_input_embedding_and_coverage(
                    self.prompt_encoder,
                    x_prompt,
                    region_mask,
                )
                stats = input_emb.detach().float().cpu().squeeze(0)
                if int(stats.numel()) != input_emb_dim:
                    raise ValueError(
                        "source context prototype input embedding dimension mismatch: "
                        f"got {int(stats.numel())}, expected {input_emb_dim}"
                    )
                stats64 = stats.to(torch.float64)
                monthly_sums[src_rid, month] += stats64
                monthly_counts[src_rid, month] += 1
                monthly_coverage_sums[src_rid, month] += float(finite_coverage)
                region_sums[src_rid] += stats64
                region_counts[src_rid] += 1
                region_coverage_sums[src_rid] += float(finite_coverage)
                global_sum += stats64
                global_coverage_sum += float(finite_coverage)
                global_count += 1

        if was_training:
            self.prompt_encoder.train()

        monthly_stats = torch.zeros_like(monthly_sums, dtype=torch.float32)
        region_stats = torch.zeros_like(region_sums, dtype=torch.float32)
        monthly_coverage = torch.zeros_like(monthly_coverage_sums, dtype=torch.float32)
        region_coverage = torch.zeros_like(region_coverage_sums, dtype=torch.float32)
        for rid in range(num_regions):
            if int(region_counts[rid]) > 0:
                region_stats[rid] = (region_sums[rid] / float(region_counts[rid])).float()
                region_coverage[rid] = float(region_coverage_sums[rid] / float(region_counts[rid]))
            for month in range(1, 13):
                if int(monthly_counts[rid, month]) > 0:
                    monthly_stats[rid, month] = (
                        monthly_sums[rid, month] / float(monthly_counts[rid, month])
                    ).float()
                    monthly_coverage[rid, month] = float(
                        monthly_coverage_sums[rid, month] / float(monthly_counts[rid, month])
                    )
                else:
                    monthly_stats[rid, month] = region_stats[rid]
                    monthly_coverage[rid, month] = region_coverage[rid]

        global_stats = (
            (global_sum / float(global_count)).float()
            if global_count > 0
            else torch.zeros(input_emb_dim, dtype=torch.float32)
        )
        global_coverage = float(global_coverage_sum / float(global_count)) if global_count > 0 else 0.0
        self._source_context_monthly_prototypes = {
            "monthly_input_emb": monthly_stats,
            "monthly_counts": monthly_counts,
            "monthly_coverage": monthly_coverage,
            "region_input_emb": region_stats,
            "region_counts": region_counts,
            "region_coverage": region_coverage,
            "global_input_emb": global_stats,
            "global_count": int(global_count),
            "global_coverage": global_coverage,
            "input_emb_dim": input_emb_dim,
            "source": "source_fit_source_region_episode_input_side_only",
            "episode_policy": SOURCE_REGION_EPISODE_POLICY,
            "input_summary_mask": "active_source_region_mask",
            "reliability_feature_transform": RELIABILITY_FEATURE_TRANSFORM,
            **prompt_domain_metadata(self.context_encoder),
        }
        print(
            "  source_context_monthly_prototype: "
            f"built {int((monthly_counts[:, 1:] > 0).sum().item())} region-month prototypes "
            f"from {global_count} source_fit input-side samples",
            flush=True,
        )

    def _source_context_monthly_prototype_summary(self) -> Dict[str, Any]:
        cache = self._source_context_monthly_prototypes
        if not cache:
            return {
                "enabled": False,
                "source": "not_applicable",
                "global_count": 0,
                "region_month_prototype_count": 0,
                "monthly_counts_by_region": {},
            }
        monthly_counts = cache["monthly_counts"]
        region_counts = cache["region_counts"]
        counts_by_region: Dict[str, Dict[str, int]] = {}
        for rid in range(monthly_counts.shape[0]):
            region_name = self.source_regions[rid] if rid < len(self.source_regions) else f"region_{rid}"
            counts_by_region[region_name] = {
                str(month): int(monthly_counts[rid, month].item())
                for month in range(1, 13)
            }
        return {
            "enabled": True,
            "source": cache["source"],
            "episode_policy": cache.get("episode_policy", SOURCE_REGION_EPISODE_POLICY),
            "input_summary_mask": cache.get("input_summary_mask", "active_source_region_mask"),
            "reliability_feature_transform": cache.get("reliability_feature_transform", RELIABILITY_FEATURE_TRANSFORM),
            "prompt_diagnostic_input_domain": cache.get(
                "prompt_diagnostic_input_domain",
                prompt_diagnostic_input_domain(self.context_encoder),
            ),
            "normalized_input_used_for_prompt_diagnostics": cache.get(
                "normalized_input_used_for_prompt_diagnostics",
                prompt_normalized_input_used(self.context_encoder),
            ),
            "global_count": int(cache["global_count"]),
            "region_counts": [int(v.item()) for v in region_counts],
            "region_month_prototype_count": int((monthly_counts[:, 1:] > 0).sum().item()),
            "monthly_counts_by_region": counts_by_region,
        }

    def _lookup_source_context_input_stats(
        self,
        region_ids: torch.Tensor,
        months: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cache = self._source_context_monthly_prototypes
        if not cache:
            raise RuntimeError("source context monthly prototypes have not been built")
        monthly_stats = cache["monthly_input_emb"]
        monthly_counts = cache["monthly_counts"]
        region_stats = cache["region_input_emb"]
        region_counts = cache["region_counts"]
        global_stats = cache["global_input_emb"]
        stats: List[torch.Tensor] = []
        for rid_tensor, month_tensor in zip(region_ids.detach().cpu(), months.detach().cpu()):
            rid = int(rid_tensor.item())
            month = min(12, max(1, int(month_tensor.item())))
            if 0 <= rid < monthly_stats.shape[0] and int(monthly_counts[rid, month].item()) > 0:
                stats.append(monthly_stats[rid, month])
            elif 0 <= rid < region_stats.shape[0] and int(region_counts[rid].item()) > 0:
                stats.append(region_stats[rid])
            else:
                stats.append(global_stats)
        return torch.stack(stats, dim=0).to(device=device, dtype=dtype)

    def _lookup_source_context_reliability_features(
        self,
        region_ids: torch.Tensor,
        months: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        schema_len = int(getattr(self.model, "source_residual_reliability_dim", 5))
        cache = self._source_context_monthly_prototypes
        if not cache:
            return torch.zeros(region_ids.shape[0], schema_len, dtype=dtype, device=device)
        monthly_counts = cache["monthly_counts"]
        monthly_coverage = cache["monthly_coverage"]
        region_counts = cache["region_counts"]
        region_coverage = cache["region_coverage"]
        global_count = int(cache.get("global_count", 0))
        global_coverage = float(cache.get("global_coverage", 0.0))
        monthly_stats = cache["monthly_input_emb"]
        region_stats = cache["region_input_emb"]
        global_stats = cache["global_input_emb"]
        rows: List[List[float]] = []
        for rid_tensor, month_tensor in zip(region_ids.detach().cpu(), months.detach().cpu()):
            rid = int(rid_tensor.item())
            month = min(12, max(1, int(month_tensor.item())))
            monthly_count = 0
            finite_coverage = global_coverage
            distance = 0.0
            if 0 <= rid < monthly_counts.shape[0] and int(monthly_counts[rid, month].item()) > 0:
                monthly_count = int(monthly_counts[rid, month].item())
                finite_coverage = float(monthly_coverage[rid, month].item())
                distance = float(torch.linalg.vector_norm(monthly_stats[rid, month] - global_stats).item())
            elif 0 <= rid < region_counts.shape[0] and int(region_counts[rid].item()) > 0:
                finite_coverage = float(region_coverage[rid].item())
                distance = float(torch.linalg.vector_norm(region_stats[rid] - global_stats).item())
            rows.append(
                (
                    bounded_reliability_features(
                    monthly_count=float(monthly_count),
                    has_monthly_prototype=1.0 if monthly_count > 0 else 0.0,
                    global_context_count=float(global_count),
                    finite_input_coverage=finite_coverage,
                    prompt_to_source_manifold_distance=distance,
                    )
                    + [0.0] * schema_len
                )[:schema_len]
            )
        return torch.as_tensor(rows, dtype=dtype, device=device)

    def _encode_prompt_from_input_stats(
        self,
        input_emb: torch.Tensor,
        region_ids: torch.Tensor,
        months: torch.Tensor,
    ) -> torch.Tensor:
        r_emb = self.prompt_encoder.region_embed(region_ids)
        t_enc = self.prompt_encoder._temporal_encoding(months)
        t_emb = self.prompt_encoder.temporal_proj(t_enc)
        return self.prompt_encoder.mlp(torch.cat([r_emb, input_emb, t_emb], dim=1))

    def _source_stage_prompt(
        self,
        x_norm: torch.Tensor,
        region_ids: torch.Tensor,
        months: torch.Tensor,
        x_raw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.source_episode_prompt_policy != "context_monthly_prototype":
            x_prompt = prompt_diagnostic_tensor(
                self.prompt_encoder,
                context_encoder=self.context_encoder,
                x_norm=x_norm,
                x_raw=x_raw,
            )
            return self.prompt_encoder(x_prompt, region_ids, months)
        input_stats = self._lookup_source_context_input_stats(
            region_ids,
            months,
            device=x_norm.device,
            dtype=x_norm.dtype,
        )
        return self._encode_prompt_from_input_stats(input_stats, region_ids, months)

    def _source_stage_prompt_and_reliability(
        self,
        x_norm: torch.Tensor,
        region_ids: torch.Tensor,
        months: torch.Tensor,
        x_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z = self._source_stage_prompt(x_norm, region_ids, months, x_raw=x_raw)
        reliability_features: Optional[torch.Tensor] = None
        if (
            isinstance(self.model, HyperAdapterConditionalResUNet)
            and getattr(self.model, "uses_source_residual_prior", False)
        ):
            reliability_features = self._lookup_source_context_reliability_features(
                region_ids,
                months,
                device=x_norm.device,
                dtype=x_norm.dtype,
            )
        return z, reliability_features

    def refresh_source_context_monthly_prototypes(self) -> None:
        """Rebuild source-fit prompt prototypes after restored normalization stats."""
        needs_source_reliability = (
            isinstance(self.model, HyperAdapterConditionalResUNet)
            and getattr(self.model, "uses_source_residual_prior", False)
        )
        if self.source_episode_prompt_policy == "context_monthly_prototype" or needs_source_reliability:
            self._load_or_build_source_context_monthly_prototypes()

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
            # Determine region per sample from region_mask_integer
            # Map global region index (0..5) to source-only index for prompt encoder
            region_ids = []
            months = []
            for s in batch:
                if s.get("sample_region_id"):
                    global_rid = _GLOBAL_REGION_IDX_MAP[str(s["sample_region_id"])]
                else:
                    active_ids = s.get("active_region_ids") or []
                    if len(active_ids) == 1:
                        global_rid = _GLOBAL_REGION_IDX_MAP[str(active_ids[0])]
                    else:
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
            result = collate_hydroda_samples(batch)
            result["region_ids"] = torch.tensor(region_ids, dtype=torch.long)
            result["months"] = torch.tensor(months, dtype=torch.long)
            return result

        loader_kwargs: Dict[str, Any] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": collate_fn,
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        if dataset is None and self.train_batch_sampler == "source_region_year_grouped":
            loader_kwargs["batch_sampler"] = SourceRegionYearGroupedBatchSampler(
                target_dataset,
                batch_size=self.batch_size,
                seed=int(getattr(target_dataset, "seed", 0)),
            )
        else:
            loader_kwargs["batch_size"] = self.batch_size
            loader_kwargs["shuffle"] = (dataset is None)
        return DataLoader(target_dataset, **loader_kwargs)

    def _get_increment_scale(self) -> Optional[torch.Tensor]:
        """Return per-channel increment scale [2] from source_fit stats."""
        if self.target_increment_normalization:
            return torch.ones(2, dtype=torch.float32)
        if self._inc_std is not None:
            return torch.from_numpy(self._inc_std.astype(np.float32))
        return None

    def _loss_for_prediction(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        latitude_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.use_lat_weighted_loss:
            if latitude_weight is None:
                raise ValueError(
                    "use_lat_weighted_loss=True but latitude_weight not provided in batch. "
                    "Ensure dataset returns latitude_weight."
                )
            inc_scale = self._get_increment_scale()
            return self.loss_fn(pred, target, loss_mask, latitude_weight=latitude_weight, increment_scale=inc_scale)
        return self.loss_fn(pred, target, loss_mask)

    def _coefficient_entropy_penalty(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Return entropy-floor penalty for HyperDA basis coefficients."""
        zero = z.new_zeros(())
        if self.hyper_coeff_entropy_penalty <= 0.0 or self.hyper_coeff_entropy_floor <= 0.0:
            return zero
        if not isinstance(self.model, HyperAdapterConditionalResUNet):
            return zero
        if not getattr(self.model, "hyper_enable_adapters", False):
            return zero

        penalties: List[torch.Tensor] = []
        for layer_name in ("bottleneck", "dec2", "dec1"):
            logits = self.model.adapter_coefficient_logits(z, layer_name)
            coeffs = self.model._adapter_module(layer_name).coefficients(z, coeff_logits=logits)
            entropy = -(coeffs.clamp_min(1e-8) * coeffs.clamp_min(1e-8).log()).sum(dim=-1)
            penalties.append((float(self.hyper_coeff_entropy_floor) - entropy).clamp_min(0.0).mean())
        if not penalties:
            return zero
        return torch.stack(penalties).mean() * float(self.hyper_coeff_entropy_penalty)

    def _residual_magnitude_penalty(
        self,
        pred: torch.Tensor,
        x_norm: torch.Tensor,
        z: torch.Tensor,
        reliability_features: Optional[torch.Tensor],
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize source-safe residual magnitude relative to frozen source base."""
        zero = pred.new_zeros(())
        if self.hyper_residual_magnitude_penalty <= 0.0:
            return zero
        if not (
            isinstance(self.model, HyperAdapterConditionalResUNet)
            and getattr(self.model, "uses_source_residual_prior", False)
        ):
            return zero
        source_base = self.model.source_base_forward(x_norm).detach()
        residual = pred - source_base.to(device=pred.device, dtype=pred.dtype)
        mask = loss_mask
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(device=pred.device, dtype=pred.dtype).expand_as(residual)
        denom = mask.sum().clamp_min(1.0)
        return residual.square().mul(mask).sum().div(denom) * float(self.hyper_residual_magnitude_penalty)

    def _apply_hyperda_regularization(
        self,
        losses: Dict[str, torch.Tensor],
        *,
        pred: torch.Tensor,
        x_norm: torch.Tensor,
        z: torch.Tensor,
        reliability_features: Optional[torch.Tensor],
        loss_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        residual_penalty = self._residual_magnitude_penalty(
            pred,
            x_norm,
            z,
            reliability_features,
            loss_mask,
        )
        entropy_penalty = self._coefficient_entropy_penalty(z)
        total_extra = residual_penalty + entropy_penalty
        if torch.is_tensor(total_extra) and total_extra.requires_grad:
            losses = dict(losses)
            losses["hyper_residual_magnitude_loss"] = residual_penalty
            losses["hyper_coeff_entropy_floor_loss"] = entropy_penalty
            losses["total_loss"] = losses["total_loss"] + total_extra
        else:
            losses = dict(losses)
            losses["hyper_residual_magnitude_loss"] = residual_penalty.detach()
            losses["hyper_coeff_entropy_floor_loss"] = entropy_penalty.detach()
        return losses

    def _abort_amp_failure(
        self,
        *,
        reason: str,
        epoch: int,
        batch_idx: int,
        global_step: int,
        prev_scale: float,
        new_scale: float,
    ) -> None:
        """Record AMP numerical failure and abort the run with a non-zero exit."""
        self._amp_failure_reason = str(reason)
        payload = {
            "status": "failed",
            "failure_type": "amp_numerical_failure",
            "reason": self._amp_failure_reason,
            "epoch": int(epoch),
            "batch_idx": int(batch_idx),
            "global_step": int(global_step),
            "prev_amp_scale": float(prev_scale),
            "amp_scale": float(new_scale),
            "amp_min_scale": self.amp_min_scale,
            "amp_skip_abort_threshold": self.amp_skip_abort_threshold,
            "skipped_steps": self._skipped_steps,
            "consecutive_amp_skips": self._consecutive_amp_skips,
            "timestamp": get_timestamp(),
        }
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        failure_paths = [self.checkpoint_dir / "training_failure.json"]
        if self.run_manager is not None:
            failure_paths.append(self.run_manager.get_reports_dir() / "training_failure.json")
        for path in failure_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        line = (
            "AMP numerical failure: "
            f"{reason} (epoch={epoch}, step={batch_idx}, global_step={global_step}, "
            f"prev_scale={prev_scale}, new_scale={new_scale}, "
            f"consecutive_skips={self._consecutive_amp_skips})"
        )
        if self.run_manager is not None:
            self.run_manager.log_console(line)
            self.run_manager.close_console_log()
        else:
            print(line, flush=True)
        raise FloatingPointError(line)

    def _forward_and_loss(
        self,
        x_norm: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        region_ids: torch.Tensor,
        months: torch.Tensor,
        latitude_weight: Optional[torch.Tensor] = None,
        *,
        x_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass + loss for prompt-conditioned model.

        Handles AMP consistently. Returns (pred, losses_dict).
        """
        if x_raw is None:
            x_raw = x_norm

        if self.use_amp:
            with autocast('cuda'):
                z, reliability_features = self._source_stage_prompt_and_reliability(
                    x_norm,
                    region_ids,
                    months,
                    x_raw=x_raw,
                )
                if reliability_features is not None:
                    pred = self.model(x_norm, z, reliability_features=reliability_features)
                else:
                    pred = self.model(x_norm, z)
        else:
            z, reliability_features = self._source_stage_prompt_and_reliability(
                x_norm,
                region_ids,
                months,
                x_raw=x_raw,
            )
            if reliability_features is not None:
                pred = self.model(x_norm, z, reliability_features=reliability_features)
            else:
                pred = self.model(x_norm, z)

        # Cast to fp32 for numerical stability in loss
        pred = pred.float()

        losses = self._loss_for_prediction(pred, target, loss_mask, latitude_weight=latitude_weight)
        losses = self._apply_hyperda_regularization(
            losses,
            pred=pred,
            x_norm=x_norm,
            z=z,
            reliability_features=reliability_features,
            loss_mask=loss_mask,
        )
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

    def _select_source_val_rho(self, loader: DataLoader) -> Dict[str, Any]:
        """Select source-residual prior rho with source-region non-degradation."""
        if not (
            isinstance(self.model, HyperAdapterConditionalResUNet)
            and getattr(self.model, "uses_source_residual_prior", False)
        ):
            return {
                "zero_shot_rho": float(self.source_residual_rho),
                "zero_shot_rho_grid": [float(v) for v in self.source_val_gain_grid],
                "zero_shot_rho_selection_source": "not_applicable_non_source_residual_prior",
                "zero_shot_rho_trace": [],
            }

        region_names = list(self.source_regions) or [f"region_{idx}" for idx in range(self.prompt_encoder.num_regions)]
        rho_loss_sums = {
            float(rho): {
                region: {"surface": 0.0, "rootzone": 0.0, "total": 0.0}
                for region in region_names
            }
            for rho in self.source_val_gain_grid
        }
        rho_weight_sums = {
            float(rho): {
                region: {"surface": 0.0, "rootzone": 0.0, "total": 0.0}
                for region in region_names
            }
            for rho in self.source_val_gain_grid
        }
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

                x_norm = self._normalize(x)
                target = torch.stack([inc_surface, inc_rootzone], dim=1)
                if self.target_increment_normalization and self._inc_mean is not None:
                    inc_mean_t = torch.from_numpy(self._inc_mean).to(x.device).view(1, 2, 1, 1)
                    inc_std_t = torch.from_numpy(self._inc_std).to(x.device).view(1, 2, 1, 1)
                    target = (target - inc_mean_t) / inc_std_t

                z, reliability_features = self._source_stage_prompt_and_reliability(
                    x_norm,
                    region_ids,
                    months,
                    x_raw=x,
                )
                source_base_pred = self.model(
                    x_norm,
                    z,
                    rho=0.0,
                    reliability_features=reliability_features,
                )
                full_residual_pred = self.model(
                    x_norm,
                    z,
                    rho=1.0,
                    reliability_features=reliability_features,
                )
                residual_delta = full_residual_pred - source_base_pred
                for rho_candidate in self.source_val_gain_grid:
                    rho_value = float(rho_candidate)
                    mixed_pred = source_base_pred + rho_value * residual_delta
                    for b in range(x.shape[0]):
                        src_rid = int(region_ids[b].detach().cpu().item())
                        region_name = region_names[src_rid] if src_rid < len(region_names) else f"region_{src_rid}"
                        rho_losses = self._loss_for_prediction(
                            mixed_pred[b : b + 1].float(),
                            target[b : b + 1],
                            loss_mask[b : b + 1],
                            latitude_weight=latitude_weight[b : b + 1] if latitude_weight is not None else None,
                        )
                        surface_loss = float(rho_losses["surface_loss"].detach().cpu().item())
                        rootzone_loss = float(rho_losses["rootzone_loss"].detach().cpu().item())
                        total_loss_value = float(rho_losses["total_loss"].detach().cpu().item())
                        weight_tensor = rho_losses.get(
                            "valid_weight_sum",
                            rho_losses.get("valid_pixel_count", torch.ones((), device=x.device)),
                        )
                        weight = max(float(weight_tensor.detach().cpu().item()), 1.0)
                        for variable, loss_value in (
                            ("surface", surface_loss),
                            ("rootzone", rootzone_loss),
                            ("total", total_loss_value),
                        ):
                            rho_loss_sums[rho_value][region_name][variable] += loss_value * weight
                            rho_weight_sums[rho_value][region_name][variable] += weight

        trace: List[Dict[str, Any]] = []
        mean_loss_by_rho: Dict[float, float] = {}
        baseline_losses: Dict[str, Dict[str, float]] = {}
        epsilon = 1e-12
        for rho_value in sorted(rho_loss_sums):
            per_region: Dict[str, Dict[str, float]] = {}
            total_loss_sum = 0.0
            total_weight_sum = 0.0
            for region_name in region_names:
                region_metrics: Dict[str, float] = {}
                for variable in ("surface", "rootzone", "total"):
                    weight = rho_weight_sums[rho_value][region_name][variable]
                    loss_value = rho_loss_sums[rho_value][region_name][variable] / max(weight, 1.0)
                    region_metrics[f"{variable}_loss"] = float(loss_value)
                    region_metrics[f"{variable}_weight"] = float(weight)
                    if variable == "total":
                        total_loss_sum += rho_loss_sums[rho_value][region_name][variable]
                        total_weight_sum += max(weight, 1.0) if weight > 0.0 else 0.0
                per_region[region_name] = region_metrics
            mean_loss = total_loss_sum / max(total_weight_sum, 1.0)
            mean_loss_by_rho[rho_value] = float(mean_loss)
            if abs(rho_value) < epsilon:
                baseline_losses = {
                    region: {
                        "surface": metrics["surface_loss"],
                        "rootzone": metrics["rootzone_loss"],
                    }
                    for region, metrics in per_region.items()
                }
            trace.append(
                {
                    "rho": float(rho_value),
                    "source_val_loss": float(mean_loss),
                    "per_region": per_region,
                }
            )

        safe_rhos: List[float] = []
        for item in trace:
            rho_value = float(item["rho"])
            unsafe_cells: List[Dict[str, Any]] = []
            if abs(rho_value) < epsilon:
                safe_rhos.append(rho_value)
                item["safe_relative_to_rho0"] = True
                item["unsafe_cells"] = []
                continue
            for region_name in region_names:
                for variable in ("surface", "rootzone"):
                    candidate = float(item["per_region"][region_name][f"{variable}_loss"])
                    baseline = float(baseline_losses.get(region_name, {}).get(variable, float("inf")))
                    if candidate > baseline + epsilon:
                        unsafe_cells.append(
                            {
                                "region": region_name,
                                "variable": variable,
                                "candidate_loss": candidate,
                                "rho0_loss": baseline,
                                "delta": candidate - baseline,
                            }
                        )
            item["safe_relative_to_rho0"] = not unsafe_cells
            item["unsafe_cells"] = unsafe_cells
            if not unsafe_cells:
                safe_rhos.append(rho_value)

        if not safe_rhos:
            safe_rhos = [0.0]
        best_rho = min(safe_rhos, key=lambda rho_value: mean_loss_by_rho.get(float(rho_value), float("inf")))
        if abs(best_rho) < epsilon:
            if any(float(item["rho"]) > 0.0 and item.get("unsafe_cells") for item in trace):
                selection_reason = "fallback_rho0_due_to_source_region_variable_degradation"
            else:
                selection_reason = "rho0_minimum_safe_source_val_loss"
        else:
            selection_reason = "nonzero_rho_safe_all_source_regions_and_lower_mean_loss"

        self.source_residual_rho = best_rho
        self.model.source_residual_rho = best_rho
        selection = {
            "zero_shot_rho": best_rho,
            "zero_shot_rho_grid": [float(v) for v in self.source_val_gain_grid],
            "zero_shot_rho_selection_source": "source_val_regionwise_safe_episode_only",
            "zero_shot_rho_selection_reason": selection_reason,
            "zero_shot_rho_safe_policy": "source_safe_regionwise_non_degradation_surface_rootzone_vs_rho0",
            "zero_shot_rho_trace": trace,
        }
        self._last_zero_shot_rho_selection = dict(selection)
        return selection

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
        rho_selection = self._select_source_val_rho(loader)

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

                z, reliability_features = self._source_stage_prompt_and_reliability(
                    x_norm,
                    region_ids,
                    months,
                    x_raw=x,
                )
                if reliability_features is not None:
                    pred = self.model(x_norm, z, reliability_features=reliability_features)
                else:
                    pred = self.model(x_norm, z)

                # Track prompt embeddings for quality
                self._prompt_quality_tracker.update(z, region_ids)

                # Loss
                losses = self._loss_for_prediction(pred, target, loss_mask, latitude_weight=latitude_weight)

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
            gain_results.update(rho_selection)
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
            gain_results.update(rho_selection)
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
        trainable_count = self._trainable_scope_metadata["trainable_parameter_count"]
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
            f"  Trainable scope: {self.trainable_scope}",
            f"  Trainable params:{trainable_count:,}",
            f"  Model width:     {self.model_width}",
            f"  Prompt dim:      {self.prompt_dim}",
            f"  Model type:      {self.model_type}",
            f"  Hyper n_basis:   {self.hyper_n_basis}",
            f"  Hyper bottleneck:{self.hyper_adapter_bottleneck}",
            f"  Hyper scale:     {self.hyper_adapter_scale}",
            f"  Hyper coeff gen: {self.hyper_coeff_generator}",
            f"  Hyper rel gate:  {self.hyper_reliability_gate}",
            f"  Hyper rel init:  {self.hyper_reliability_init}",
            f"  Hyper saliency:  beta={self.hyper_source_saliency_prior_beta} application={self.hyper_source_saliency_prior_application} path={self.hyper_source_saliency_prior_path or 'none'}",
            f"  Hyper manifold:  enabled={self.hyper_prompt_manifold_reliability} strength={self.hyper_prompt_manifold_reliability_strength}",
            f"  Hyper FiLM:      {self.hyper_enable_film}",
            f"  Hyper adapters:  {self.hyper_enable_adapters}",
            f"  Hyper residual penalty: {self.hyper_residual_magnitude_penalty}",
            f"  Hyper coeff entropy: floor={self.hyper_coeff_entropy_floor} penalty={self.hyper_coeff_entropy_penalty}",
            f"  Prior form:      {self.zero_shot_prior_form}",
            f"  Source res rho:  {self.source_residual_rho}",
            f"  Source res gate: {self.source_residual_gate}",
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
            f"  Dataset backend: {self.dataset_backend}",
            f"  Tensor load:     {self.tensor_cache_load_mode}",
            f"  Batch sampler:   {self.train_batch_sampler}",
            f"  Num workers:     {self.num_workers}",
            f"  Prefetch factor: {self.prefetch_factor if self.prefetch_factor is not None else 'none'}",
            f"  Persistent wrk:  {self.persistent_workers}",
            f"  Pin memory:      {self.pin_memory}",
            f"  Train samples:   {len(self.train_dataset)}",
            f"  Source regions:  {self.source_regions}",
            f"  Steps/epoch:     {total_steps_per_epoch}",
        ]
        if self.init_from_source_base_checkpoint:
            header_lines.append(f"  Source base ckpt:{self.init_from_source_base_checkpoint}")
            header_lines.append(f"  Source base sha: {self.source_base_checkpoint_sha256}")
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

            next_batch_start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                iter_start = time.time()
                data_wait_sec = iter_start - next_batch_start
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
                    next_batch_start = time.time()
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
                    x_raw=x,
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
                    next_batch_start = time.time()
                    continue

                # Backward pass
                if self.use_amp:
                    self._amp_scaler.scale(losses["total_loss"]).backward()
                    if (batch_idx + 1) % self.accum_steps == 0:
                        prev_scale = self._amp_scaler.get_scale()
                        if self.grad_clip is not None:
                            self._amp_scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                trainable_parameters(self.model, self.prompt_encoder),
                                self.grad_clip,
                            )
                        self._amp_scaler.step(self.optimizer)
                        self._amp_scaler.update()
                        self.optimizer.zero_grad()
                        new_scale = self._amp_scaler.get_scale()
                        # Track gradient overflow skips (scale reduction = Inf/NaN detected)
                        if new_scale < prev_scale:
                            self._skipped_steps += 1
                            self._consecutive_amp_skips += 1
                        else:
                            self._consecutive_amp_skips = 0
                        if new_scale < self.amp_min_scale:
                            self._abort_amp_failure(
                                reason="amp_scale_below_min_scale",
                                epoch=epoch,
                                batch_idx=batch_idx,
                                global_step=global_step,
                                prev_scale=float(prev_scale),
                                new_scale=float(new_scale),
                            )
                        if self._consecutive_amp_skips >= self.amp_skip_abort_threshold:
                            self._abort_amp_failure(
                                reason="consecutive_amp_overflow_skips_exceeded",
                                epoch=epoch,
                                batch_idx=batch_idx,
                                global_step=global_step,
                                prev_scale=float(prev_scale),
                                new_scale=float(new_scale),
                            )
                else:
                    losses["total_loss"].backward()
                    if (batch_idx + 1) % self.accum_steps == 0:
                        if self.grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(
                                trainable_parameters(self.model, self.prompt_encoder),
                                self.grad_clip,
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                compute_end = time.time()
                compute_sec = compute_end - iter_start
                iter_wall_sec = compute_end - next_batch_start
                samples_per_sec = float(x.shape[0]) / max(iter_wall_sec, 1e-12)

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
                        "data_wait_sec": round(data_wait_sec, 6),
                        "compute_sec": round(compute_sec, 6),
                        "iter_wall_sec": round(iter_wall_sec, 6),
                        "samples_per_sec": round(samples_per_sec, 6),
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
                            f"gpu={gpu_alloc:.1f}GB {batches_per_sec:.1f}b/s "
                            f"{samples_per_sec:.1f}samples/s wait={data_wait_sec:.3f}s compute={compute_sec:.3f}s | "
                            f"lr={lr_curr:.2e} "
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
                next_batch_start = time.time()

            avg_loss = float(np.mean(epoch_losses))
            avg_surface = float(np.mean(epoch_surface_losses))
            avg_rootzone = float(np.mean(epoch_rootzone_losses))
            total_valid = int(np.sum(epoch_valid_counts))
            elapsed = time.time() - epoch_start

            self.scheduler.step(avg_loss)

            # Source val eval + gain calibration every eval_every_epochs
            source_val_metrics = {}
            gain_results = {}
            ran_source_val_eval = (
                self.source_val_dataset is not None
                and self.eval_every_epochs > 0
                and epoch % self.eval_every_epochs == 0
            )
            if ran_source_val_eval:
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
            has_selection_evidence = (
                self.selection_metric == "train_loss"
                or (ran_source_val_eval and bool(source_val_metrics or gain_results))
            )
            current_best = self.best_selection_value
            is_best = False
            if has_selection_evidence:
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
            is_best_safe_score = ran_source_val_eval and safe_score > self.best_safe_score

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
        trainable_count = self._trainable_scope_metadata["trainable_parameter_count"]
        has_source_val = self.source_val_dataset is not None

        lines = [
            f"# Prompt-Conditioned Training Summary",
            f"",
            f"- **Experiment**: {self.experiment_id}",
            f"- **Protocol**: {self.protocol_freeze_id}",
            f"- **Split manifest**: {self.split_manifest_path}",
            f"- **Model type**: {self.model_type}",
            f"- **Context encoder**: {self.context_encoder}",
            f"- **Prompt input feature source**: {prompt_input_feature_source(self.context_encoder)}",
            f"- **Prompt channel 11 usage**: {prompt_channel_11_usage(self.context_encoder)}",
            f"- **Prompt diagnostic input domain**: {prompt_diagnostic_input_domain(self.context_encoder)}",
            f"- **Normalized input used for prompt diagnostics**: {prompt_normalized_input_used(self.context_encoder)}",
            f"- **Model width**: {self.model_width}",
            f"- **Prompt dim**: {self.prompt_dim}",
            f"- **Hyper n basis**: {self.hyper_n_basis}",
            f"- **Hyper adapter bottleneck**: {self.hyper_adapter_bottleneck}",
            f"- **Hyper adapter scale**: {self.hyper_adapter_scale}",
            f"- **Hyper coeff generator**: {self.hyper_coeff_generator}",
            f"- **Hyper reliability gate**: {self.hyper_reliability_gate}",
            f"- **Hyper reliability init**: {self.hyper_reliability_init}",
            f"- **Hyper source saliency prior beta**: {self.hyper_source_saliency_prior_beta}",
            f"- **Hyper source saliency prior application**: {self.hyper_source_saliency_prior_application}",
            f"- **Hyper source saliency prior path**: {self.hyper_source_saliency_prior_path or 'none'}",
            f"- **Hyper prompt manifold reliability**: {self.hyper_prompt_manifold_reliability}",
            f"- **Hyper prompt manifold reliability strength**: {self.hyper_prompt_manifold_reliability_strength}",
            f"- **Hyper FiLM enabled**: {self.hyper_enable_film}",
            f"- **Hyper adapters enabled**: {self.hyper_enable_adapters}",
            f"- **Hyper residual magnitude penalty**: {self.hyper_residual_magnitude_penalty}",
            f"- **Hyper coefficient entropy floor**: {self.hyper_coeff_entropy_floor}",
            f"- **Hyper coefficient entropy penalty**: {self.hyper_coeff_entropy_penalty}",
            f"- **Zero-shot prior form**: {self.zero_shot_prior_form}",
            f"- **Zero-shot rho**: {self.source_residual_rho}",
            f"- **Source residual gate**: {self.source_residual_gate}",
            f"- **Model params**: {num_model_params:,}",
            f"- **Prompt encoder params**: {num_pe_params:,}",
            f"- **Total params**: {num_model_params + num_pe_params:,}",
            f"- **Trainable scope**: {self.trainable_scope}",
            f"- **Trainable params**: {trainable_count:,}",
            f"- **Frozen source base modules**: {', '.join(self._trainable_scope_metadata['frozen_source_base_modules']) or 'none'}",
            f"- **Source base checkpoint**: {self.init_from_source_base_checkpoint or 'none'}",
            f"- **Source base checkpoint SHA256**: {self.source_base_checkpoint_sha256 or 'none'}",
            f"- **Dataset backend**: {self.dataset_backend}",
            f"- **Tensor cache load mode**: {self.tensor_cache_load_mode}",
            f"- **Train batch sampler**: {self.train_batch_sampler}",
            f"- **Num workers**: {self.num_workers}",
            f"- **Persistent workers**: {self.persistent_workers}",
            f"- **Prefetch factor**: {self.prefetch_factor if self.prefetch_factor is not None else 'none'}",
            f"- **Pin memory**: {self.pin_memory}",
            f"- **Eval every epochs**: {self.eval_every_epochs}",
            f"- **Source prototype cache mode**: {self.source_prototype_cache_mode}",
            f"- **Source prototype cache hit**: {self.source_prototype_cache_hit}",
            f"- **Source prototype cache path**: {self.source_prototype_cache_path or 'none'}",
            f"- **Batch size**: {self.batch_size}",
            f"- **Accum steps**: {self.accum_steps}",
            f"- **Effective batch size**: {self.batch_size * self.accum_steps}",
            f"- **Max epochs**: {self.max_epochs}",
            f"- **LR**: {self.lr}",
            f"- **Weight decay**: {self.weight_decay}",
            f"- **Grad clip**: {self.grad_clip}",
            f"- **AMP**: {self.use_amp}",
            f"- **AMP init scale**: {self.amp_init_scale}",
            f"- **AMP min scale**: {self.amp_min_scale}",
            f"- **AMP skip abort threshold**: {self.amp_skip_abort_threshold}",
            f"- **AMP failure reason**: {self._amp_failure_reason or 'none'}",
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
            f"- **Normalization source**: {self.leakage_policy['normalization_source']}",
            f"- **Early stopping source**: {'source_val_only' if has_source_val else 'train_loss_only'}",
            f"- **Model selection source**: {'source_val_only' if has_source_val else 'best_train_loss'}",
            f"- **Target labels used for adaptation**: false",
            f"- **Target val usage**: unused_in_main_protocol",
            f"- **Target eval usage**: final_eval_only_no_selection",
            f"- **Target eval input stats used for update**: false",
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
                "model_type": self.model_type,
                "width": self.model_width,
                "prompt_dim": self.prompt_dim,
                "hyper_n_basis": self.hyper_n_basis,
                "hyper_adapter_bottleneck": self.hyper_adapter_bottleneck,
                "hyper_adapter_scale": self.hyper_adapter_scale,
                "hyper_coeff_generator": self.hyper_coeff_generator,
                "hyper_rank_gate_top_k": self.hyper_rank_gate_top_k,
                "hyper_rank_gate_temperature_init": self.hyper_rank_gate_temperature_init,
                "hyper_adapter_param_style": self.hyper_adapter_param_style,
                "hyper_reliability_gate": self.hyper_reliability_gate,
                "hyper_reliability_init": self.hyper_reliability_init,
                "hyper_source_saliency_prior_beta": self.hyper_source_saliency_prior_beta,
                "hyper_source_saliency_prior_path": self.hyper_source_saliency_prior_path,
                "hyper_source_saliency_prior_application": self.hyper_source_saliency_prior_application,
                "hyper_source_saliency_prior_metadata": self.hyper_source_saliency_prior_metadata,
                "hyper_source_saliency_prior": (
                    self.hyper_source_saliency_prior.tolist()
                    if self.hyper_source_saliency_prior is not None
                    else None
                ),
                "hyper_prompt_manifold_reliability": self.hyper_prompt_manifold_reliability,
                "hyper_prompt_manifold_reliability_strength": self.hyper_prompt_manifold_reliability_strength,
                "hyper_enable_film": self.hyper_enable_film,
                "hyper_enable_adapters": self.hyper_enable_adapters,
                "hyper_residual_magnitude_penalty": self.hyper_residual_magnitude_penalty,
                "hyper_coeff_entropy_floor": self.hyper_coeff_entropy_floor,
                "hyper_coeff_entropy_penalty": self.hyper_coeff_entropy_penalty,
                "zero_shot_prior_form": self.zero_shot_prior_form,
                "source_residual_prior_mode": self.zero_shot_prior_form != "direct_hyper",
                "source_residual_rho": self.source_residual_rho,
                "source_residual_gate": self.source_residual_gate,
                "source_residual_gate_init": self.source_residual_gate_init,
                "source_residual_reliability_dim": self.source_residual_reliability_dim,
                "zero_shot_rho": self.source_residual_rho,
                "zero_shot_rho_grid": self.source_val_gain_grid,
                "zero_shot_rho_selection_source": self._last_zero_shot_rho_selection.get(
                    "zero_shot_rho_selection_source",
                    "source_val_regionwise_safe_episode_only",
                ),
                "zero_shot_rho_selection_reason": self._last_zero_shot_rho_selection.get(
                    "zero_shot_rho_selection_reason",
                    "",
                ),
                "zero_shot_rho_safe_policy": self._last_zero_shot_rho_selection.get(
                    "zero_shot_rho_safe_policy",
                    "source_safe_regionwise_non_degradation_surface_rootzone_vs_rho0",
                ),
                "zero_shot_rho_trace": self._last_zero_shot_rho_selection.get("zero_shot_rho_trace", []),
                "zero_shot_residual_formula": (
                    "pred = source_base + rho * reliability_gate(prompt, context) * hyper_residual"
                    if self.zero_shot_prior_form != "direct_hyper"
                    else "direct_hyper"
                ),
                "reliability_feature_schema": getattr(self.model, "reliability_feature_schema", []),
                "reliability_feature_transform": RELIABILITY_FEATURE_TRANSFORM,
                "target_labels_used_for_adaptation": False,
                "target_val_usage": "unused_in_main_protocol",
                "target_eval_usage": "final_eval_only_no_selection",
                "target_eval_input_stats_used_for_update": False,
                "trainable_scope": self.trainable_scope,
                "source_episode_prompt_policy": self.source_episode_prompt_policy,
                "source_context_monthly_prototype_summary": self._source_context_monthly_prototype_summary(),
                "source_prototype_cache_mode": self.source_prototype_cache_mode,
                "source_prototype_cache_hit": self.source_prototype_cache_hit,
                "source_prototype_cache_path": self.source_prototype_cache_path,
                "source_prototype_cache_key": self.source_prototype_cache_key,
                "source_anchor_blend_calibration": self.source_anchor_blend_calibration,
                "hyper_output_head_residual": self.hyper_output_head_residual,
                "trainable_parameter_count": self._trainable_scope_metadata["trainable_parameter_count"],
                "trainable_parameter_names": self._trainable_scope_metadata["trainable_parameter_names"],
                "frozen_source_base_modules": self._trainable_scope_metadata["frozen_source_base_modules"],
                "init_from_source_base_checkpoint": self.init_from_source_base_checkpoint,
                "source_base_checkpoint_sha256": self.source_base_checkpoint_sha256,
                "source_base_checkpoint_config": self.source_base_checkpoint_config,
                "source_base_loaded_parameter_names": self.source_base_loaded_parameter_names,
                "context_encoder": self.context_encoder,
                "prompt_diagnostic_schema": list(getattr(self.prompt_encoder, "diagnostic_schema", [])),
                "prompt_input_feature_source": prompt_input_feature_source(self.context_encoder),
                "prompt_channel_11_usage": prompt_channel_11_usage(self.context_encoder),
                "prompt_diagnostic_input_domain": prompt_diagnostic_input_domain(self.context_encoder),
                "normalized_input_used_for_prompt_diagnostics": prompt_normalized_input_used(self.context_encoder),
                "num_regions": self.prompt_encoder.num_regions,
                "num_workers": self.num_workers,
                "persistent_workers": self.persistent_workers,
                "prefetch_factor": self.prefetch_factor,
                "pin_memory": self.pin_memory,
                "tensor_cache_load_mode": self.tensor_cache_load_mode,
                "train_batch_sampler": self.train_batch_sampler,
                "target_increment_normalization": self.target_increment_normalization,
                "zero_raw_increment_init": self.zero_raw_increment_init,
                "use_amp": self.use_amp,
                "amp_init_scale": self.amp_init_scale,
                "amp_min_scale": self.amp_min_scale,
                "amp_skip_abort_threshold": self.amp_skip_abort_threshold,
                "amp_failure_reason": self._amp_failure_reason,
                "use_lat_weighted_loss": self.use_lat_weighted_loss,
                "log_every_steps": self.log_every_steps,
                "eval_every_epochs": self.eval_every_epochs,
                "checkpoint_every_n_epochs": self.checkpoint_every_n_epochs,
                "lambda_amp": self.lambda_amp,
                "selection_metric": self.selection_metric,
                "source_val_residual_gain": self.source_val_residual_gain,
                "target_region": self.target_region,
                "adaptation_setting": self.adaptation_setting,
                "K": self.K,
                "protocol_freeze_id": self.protocol_freeze_id,
                "dataset_backend": self.dataset_backend,
                "selection_value": resolved_selection_value,
                "best_selection_metric": self.best_selection_metric,
                "best_selection_value": self.best_selection_value,
                "ch_mean": self._ch_mean.tolist() if self._ch_mean is not None else None,
                "ch_std": self._ch_std.tolist() if self._ch_std is not None else None,
                "inc_mean": self._inc_mean.tolist() if self._inc_mean is not None else None,
                "inc_std": self._inc_std.tolist() if self._inc_std is not None else None,
                "source_regions": self.source_regions,
                "source_region_global_indices": self._source_region_global_indices,
                "skipped_steps": self._skipped_steps,
                "consecutive_amp_skips": self._consecutive_amp_skips,
                "leakage_policy": self.leakage_policy,
                "resolved_config": self._resolved_config_metadata(),
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
        self._consecutive_amp_skips = checkpoint.get("config", {}).get("consecutive_amp_skips", 0)
        self._amp_failure_reason = checkpoint.get("config", {}).get("amp_failure_reason", "")
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
        trainable_count = self._trainable_scope_metadata["trainable_parameter_count"]
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
            "model_type": self.model_type,
            "context_encoder": self.context_encoder,
            "prompt_diagnostic_input_domain": prompt_diagnostic_input_domain(self.context_encoder),
            "normalized_input_used_for_prompt_diagnostics": prompt_normalized_input_used(self.context_encoder),
            "prompt_input_feature_source": prompt_input_feature_source(self.context_encoder),
            "prompt_channel_11_usage": prompt_channel_11_usage(self.context_encoder),
            "hyper_n_basis": self.hyper_n_basis,
            "hyper_adapter_bottleneck": self.hyper_adapter_bottleneck,
            "hyper_adapter_scale": self.hyper_adapter_scale,
            "hyper_coeff_generator": self.hyper_coeff_generator,
            "hyper_rank_gate_top_k": self.hyper_rank_gate_top_k,
            "hyper_rank_gate_temperature_init": self.hyper_rank_gate_temperature_init,
            "hyper_adapter_param_style": self.hyper_adapter_param_style,
            "hyper_reliability_gate": self.hyper_reliability_gate,
            "hyper_reliability_init": self.hyper_reliability_init,
            "hyper_source_saliency_prior_beta": self.hyper_source_saliency_prior_beta,
            "hyper_source_saliency_prior_path": self.hyper_source_saliency_prior_path,
            "hyper_source_saliency_prior_application": self.hyper_source_saliency_prior_application,
            "hyper_source_saliency_prior_metadata": self.hyper_source_saliency_prior_metadata,
            "hyper_source_saliency_prior": (
                self.hyper_source_saliency_prior.tolist()
                if self.hyper_source_saliency_prior is not None
                else None
            ),
            "hyper_prompt_manifold_reliability": self.hyper_prompt_manifold_reliability,
            "hyper_prompt_manifold_reliability_strength": self.hyper_prompt_manifold_reliability_strength,
            "hyper_enable_film": self.hyper_enable_film,
            "hyper_enable_adapters": self.hyper_enable_adapters,
            "hyper_residual_magnitude_penalty": self.hyper_residual_magnitude_penalty,
            "hyper_coeff_entropy_floor": self.hyper_coeff_entropy_floor,
            "hyper_coeff_entropy_penalty": self.hyper_coeff_entropy_penalty,
            "zero_shot_prior_form": self.zero_shot_prior_form,
            "source_residual_prior_mode": self.zero_shot_prior_form != "direct_hyper",
            "source_residual_rho": self.source_residual_rho,
            "source_residual_gate": self.source_residual_gate,
            "source_residual_gate_init": self.source_residual_gate_init,
            "source_residual_reliability_dim": self.source_residual_reliability_dim,
            "zero_shot_rho": self.source_residual_rho,
            "zero_shot_rho_grid": self.source_val_gain_grid,
            "zero_shot_rho_selection_source": self._last_zero_shot_rho_selection.get(
                "zero_shot_rho_selection_source",
                "source_val_regionwise_safe_episode_only",
            ),
            "zero_shot_rho_selection_reason": self._last_zero_shot_rho_selection.get(
                "zero_shot_rho_selection_reason",
                "",
            ),
            "zero_shot_rho_safe_policy": self._last_zero_shot_rho_selection.get(
                "zero_shot_rho_safe_policy",
                "source_safe_regionwise_non_degradation_surface_rootzone_vs_rho0",
            ),
            "zero_shot_rho_trace": self._last_zero_shot_rho_selection.get("zero_shot_rho_trace", []),
            "zero_shot_residual_formula": (
                "pred = source_base + rho * reliability_gate(prompt, context) * hyper_residual"
                if self.zero_shot_prior_form != "direct_hyper"
                else "direct_hyper"
            ),
            "reliability_feature_schema": getattr(self.model, "reliability_feature_schema", []),
            "reliability_feature_transform": RELIABILITY_FEATURE_TRANSFORM,
            "trainable_scope": self.trainable_scope,
            "source_episode_prompt_policy": self.source_episode_prompt_policy,
            "source_context_monthly_prototype_summary": self._source_context_monthly_prototype_summary(),
            "source_prototype_cache_mode": self.source_prototype_cache_mode,
            "source_prototype_cache_hit": self.source_prototype_cache_hit,
            "source_prototype_cache_path": self.source_prototype_cache_path,
            "source_prototype_cache_key": self.source_prototype_cache_key,
            "source_prototype_cache_metadata": self.source_prototype_cache_metadata,
            "source_anchor_blend_calibration": self.source_anchor_blend_calibration,
            "hyper_output_head_residual": self.hyper_output_head_residual,
            "trainable_parameters": trainable_count,
            "trainable_parameter_count": trainable_count,
            "trainable_parameter_names": self._trainable_scope_metadata["trainable_parameter_names"],
            "frozen_source_base_modules": self._trainable_scope_metadata["frozen_source_base_modules"],
            "init_from_source_base_checkpoint": self.init_from_source_base_checkpoint,
            "source_base_checkpoint_sha256": self.source_base_checkpoint_sha256,
            "batch_size": self.batch_size,
            "accum_steps": self.accum_steps,
            "effective_batch_size": self.batch_size * self.accum_steps,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "pin_memory": self.pin_memory,
            "tensor_cache_load_mode": self.tensor_cache_load_mode,
            "train_batch_sampler": self.train_batch_sampler,
            "eval_every_epochs": self.eval_every_epochs,
            "dataset_backend": self.dataset_backend,
            "source_val_available": has_source_val,
            "use_lat_weighted_loss": self.use_lat_weighted_loss,
            "loss_function": type(self.loss_fn).__name__,
            "lambda_amp": self.lambda_amp,
            "selection_metric": self.selection_metric,
            "source_val_residual_gain": self.source_val_residual_gain,
            "source_regions": self.source_regions,
            "split_manifest_sha256": self.split_manifest_sha256,
            "use_amp": self.use_amp,
            "amp_init_scale": self.amp_init_scale,
            "amp_min_scale": self.amp_min_scale,
            "amp_skip_abort_threshold": self.amp_skip_abort_threshold,
            "amp_failure_reason": self._amp_failure_reason,
            "normalization_source": self.leakage_policy["normalization_source"],
            "early_stopping_source": "source_val_only" if has_source_val else "train_loss_only",
            "model_selection_source": "source_val_only" if has_source_val else "best_train_loss",
            "target_labels_used_for_adaptation": False,
            "target_val_usage": "unused_in_main_protocol",
            "target_eval_usage": "final_eval_only_no_selection",
            "target_query_usage": "final_eval_only_no_selection",
            "target_eval_input_stats_used_for_update": False,
            "leakage_guard_status": "pass",
            "leakage_policy": self.leakage_policy,
            "resolved_config": self._resolved_config_metadata(),
            "skipped_steps": self._skipped_steps,
            "consecutive_amp_skips": self._consecutive_amp_skips,
            "git_hash": get_git_hash(),
            "timestamp": get_timestamp(),
            "train_history": self.train_history,
            "val_history": self.val_history,
        }
        target_path = path or (self.checkpoint_dir / "summary.json")
        with open(target_path, "w") as f:
            json.dump(summary, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train prompt-conditioned or HyperDA shared backbone")
    parser.add_argument("--target_region", type=str, required=True)
    parser.add_argument("--source_regions", type=str, default=None,
        help="Comma-separated source region ids to record/use for source-stage prompts. Defaults to all non-target US regions.")
    parser.add_argument("--adaptation_setting", type=str, default="zero_shot_context",
        help="Split adaptation setting (default: zero_shot_context; main examples: zero_shot_context, few_shot_k4, few_shot_k12)")
    parser.add_argument("--K", type=int, default=None,
        help="Zero/few-shot K value for the main protocol.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--prompt_dim", type=int, default=64)
    parser.add_argument("--model_type", type=str, default="prompt_conditioned",
        choices=["prompt_conditioned", "hyperda_basis_adapter"],
        help="Conditional model type to train")
    parser.add_argument("--context_encoder", type=str, default="current_mean_std",
        choices=list(CONTEXT_ENCODERS),
        help="Prompt context encoder (default current_mean_std)")
    parser.add_argument("--hyper_n_basis", type=int, default=8,
        help="Number of generated adapter bases for model_type=hyperda_basis_adapter")
    parser.add_argument("--hyper_adapter_bottleneck", type=int, default=None,
        help="Bottleneck channels for generated HyperDA adapter")
    parser.add_argument("--hyper_adapter_scale", type=float, default=1.0,
        help="Residual scale for generated HyperDA adapter")
    parser.add_argument("--hyper_coeff_generator", type=str, default="per_adapter",
        choices=[
            "per_adapter",
            "shared_layer_aware",
            "shared_layer_aware_rank_gated",
            "shared_layer_aware_rank_gated_stable",
        ],
        help="Basis coefficient generator for HyperDA adapters")
    parser.add_argument("--hyper_rank_gate_top_k", type=int, default=4,
        help="Top-k adapter basis budget for shared_layer_aware_rank_gated")
    parser.add_argument("--hyper_rank_gate_temperature_init", type=float, default=1.0,
        help="Initial coefficient temperature for shared_layer_aware_rank_gated")
    parser.add_argument("--hyper_adapter_param_style", type=str, default="basis_1x1",
        choices=["basis_1x1", "dora_like_gain", "dora_like_gain_bounded"],
        help="Adapter parameterization style for HyperDA basis residuals")
    parser.add_argument("--hyper_reliability_gate", type=str, default="none",
        choices=["none", "prompt_scalar"],
        help="Bounded reliability gate for HyperDA adapter residuals")
    parser.add_argument("--hyper_reliability_init", type=float, default=0.95,
        help="Initial prompt-scalar reliability gate value")
    parser.add_argument("--hyper_source_saliency_prior_path", type=str, default="",
        help="Optional source_fit/source-episode adapter layer x basis saliency prior artifact")
    parser.add_argument("--hyper_source_saliency_prior_beta", type=float, default=0.0,
        help="Source-side saliency prior strength. Default application is soft metadata/regularization and does not alter hard top-k routing.")
    parser.add_argument("--hyper_source_saliency_prior_application", type=str, default="soft_regularization_metadata",
        choices=list(SOURCE_SALIENCY_PRIOR_APPLICATIONS),
        help="How to apply source-side saliency. soft_regularization_metadata keeps M2.1 hard routing; legacy_gate_logit_bias_before_topk is diagnostic only.")
    parser.add_argument("--hyper_prompt_manifold_reliability", type=int, default=0,
        choices=[0, 1],
        help="Conservatively scale adapter residuals by input-side prompt-manifold reliability")
    parser.add_argument("--hyper_prompt_manifold_reliability_strength", type=float, default=0.0,
        help="Strength for prompt-manifold reliability scaling; 0 preserves existing behavior")
    parser.add_argument("--hyper_enable_film", type=int, default=1,
        choices=[0, 1],
        help="Enable FiLM conditioning in HyperDA model")
    parser.add_argument("--hyper_enable_adapters", type=int, default=1,
        choices=[0, 1],
        help="Enable HyperDA adapter residuals in forward and staged optimizer")
    parser.add_argument("--hyper_residual_magnitude_penalty", type=float, default=0.0,
        help="Optional source-stage penalty on source-base residual magnitude for M2.3-style conservative residual priors.")
    parser.add_argument("--hyper_coeff_entropy_floor", type=float, default=0.0,
        help="Optional entropy floor for adapter basis coefficients; 0 disables the floor.")
    parser.add_argument("--hyper_coeff_entropy_penalty", type=float, default=0.0,
        help="Optional penalty weight for coefficient entropy below --hyper_coeff_entropy_floor.")
    parser.add_argument("--zero_shot_prior_form", type=str, default="direct_hyper",
        choices=list(ZERO_SHOT_PRIOR_FORMS),
        help="HyperDA zero-shot prior form. source_base_residual_reliability_gated uses frozen source base plus gated residual.")
    parser.add_argument("--source_residual_rho", type=float, default=1.0,
        help="Source-residual prior blend rho selected from source_val/source episodes")
    parser.add_argument("--source_residual_gate", type=str, default="prompt_reliability_scalar",
        choices=list(SOURCE_RESIDUAL_GATES),
        help="Reliability gate for source_base_residual_reliability_gated")
    parser.add_argument("--source_residual_gate_init", type=float, default=0.95,
        help="Initial source-residual reliability gate value")
    parser.add_argument("--source_residual_reliability_dim", type=int, default=5,
        help="Number of input-side reliability features")
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
    parser.add_argument("--amp_init_scale", type=float, default=256.0,
        help="Initial torch GradScaler scale when --amp is enabled")
    parser.add_argument("--amp_min_scale", type=float, default=1.0,
        help="Abort AMP training if GradScaler scale drops below this value")
    parser.add_argument("--amp_skip_abort_threshold", type=int, default=16,
        help="Abort AMP training after this many consecutive overflow scale reductions")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--splits_json", type=str, default=SPLITS_JSON,
        help="Split artifact for source/target dates")
    parser.add_argument("--protocol_freeze_id", type=str, default=PROTOCOL_FREEZE_ID,
        help="Protocol freeze id to record in run metadata")
    parser.add_argument("--split_manifest_path", type=str, default=FREEZE_MANIFEST,
        help="Split/protocol manifest path to record in run metadata")
    parser.add_argument("--dataset_backend", type=str, default="netcdf",
        choices=["netcdf", "tensor_cache"],
        help="Dataset backend for source_fit/source_val. tensor_cache uses prebuilt per-region crops.")
    parser.add_argument("--tensor_cache_dir", type=str, default="artifacts/region_crops/US",
        help="Directory containing prebuilt tensor-cache region crops.")
    parser.add_argument("--max_year_cache_entries", type=int, default=1,
        help="Per-region tensor-cache year tensors to keep in memory (0 disables year cache).")
    parser.add_argument("--tensor_cache_load_mode", type=str, default="eager",
        choices=list(TENSOR_CACHE_LOAD_MODES),
        help="Tensor-cache torch.load mode: eager or mmap.")
    parser.add_argument("--train_batch_sampler", type=str, default="random",
        choices=list(TRAIN_BATCH_SAMPLERS),
        help="Training batch sampler policy.")
    parser.add_argument("--prefetch_factor", type=int, default=2,
        help="DataLoader prefetch_factor when num_workers > 0.")
    parser.add_argument("--source_prototype_cache_dir", type=str, default=None,
        help="Optional directory for source_fit input-side monthly prototype cache.")
    parser.add_argument("--source_prototype_cache_mode", type=str, default="off",
        choices=list(SOURCE_PROTOTYPE_CACHE_MODES),
        help="Source prototype cache mode: off, read_write, or refresh.")
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
    parser.add_argument("--init_from_prompt_checkpoint", type=str, default=None,
        help="For HyperDA, initialize shared FiLM backbone and prompt encoder from a prompt-conditioned checkpoint.")
    parser.add_argument("--init_from_source_base_checkpoint", type=str, default=None,
        help="For staged HyperDA, initialize frozen source-base enc/dec/head weights from a source-only SmallResUNet checkpoint.")
    parser.add_argument("--trainable_scope", type=str, default="all", choices=list(TRAINABLE_SCOPES),
        help="Trainability policy. 'all' preserves existing behavior; "
             "'source_base_frozen_adapter_film' freezes source base enc/dec/head and trains prompt/FiLM/HyperDA adapters.")
    parser.add_argument("--source_episode_prompt_policy", type=str, default="current_region_prompt",
        choices=["current_region_prompt", "context_monthly_prototype"],
        help="Source-stage prompt policy metadata. context_monthly_prototype matches zero/few-shot deployment episodes.")
    parser.add_argument("--source_anchor_blend_calibration", type=int, default=0,
        choices=[0, 1],
        help="Record whether source-side anchor blend calibration is enabled for staged HyperDA.")
    parser.add_argument("--hyper_output_head_residual", type=int, default=0,
        choices=[0, 1],
        help="Record whether the staged HyperDA prior includes output-head residual support.")
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
    if args.hyper_source_saliency_prior_beta < 0.0:
        parser.error("--hyper_source_saliency_prior_beta must be non-negative")
    if args.hyper_prompt_manifold_reliability_strength < 0.0:
        parser.error("--hyper_prompt_manifold_reliability_strength must be non-negative")
    if args.hyper_residual_magnitude_penalty < 0.0:
        parser.error("--hyper_residual_magnitude_penalty must be non-negative")
    if args.hyper_coeff_entropy_floor < 0.0:
        parser.error("--hyper_coeff_entropy_floor must be non-negative")
    if args.hyper_coeff_entropy_penalty < 0.0:
        parser.error("--hyper_coeff_entropy_penalty must be non-negative")
    return args


def _resolve_source_regions(target_region: str, source_regions_arg: Optional[str]) -> List[str]:
    if not source_regions_arg:
        return [r for r in _ALL_US_REGIONS if r != target_region]
    source_regions = [r.strip() for r in source_regions_arg.split(",") if r.strip()]
    if not source_regions:
        raise ValueError("--source_regions was provided but no region ids were parsed")
    unknown = [r for r in source_regions if r not in _ALL_US_REGIONS]
    if unknown:
        raise ValueError(f"Unsupported source region ids in --source_regions: {unknown}")
    if target_region in source_regions:
        raise ValueError("--source_regions must exclude --target_region")
    return source_regions


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_prompt_checkpoint_initialization(
    trainer: PromptConditionedTrainer,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Initialize compatible model/prompt weights from a prompt-conditioned checkpoint."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--init_from_prompt_checkpoint not found: {ckpt_path}")

    print(f"\nInitializing from prompt checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_state = ckpt.get("model_state_dict", {})
    model_result = trainer.model.load_state_dict(model_state, strict=False)
    print(
        "  model init: "
        f"missing={len(model_result.missing_keys)} unexpected={len(model_result.unexpected_keys)}"
    )

    prompt_state = ckpt.get("prompt_encoder_state_dict")
    if prompt_state is not None:
        prompt_result = trainer.prompt_encoder.load_state_dict(prompt_state, strict=False)
        print(
            "  prompt init: "
            f"missing={len(prompt_result.missing_keys)} unexpected={len(prompt_result.unexpected_keys)}"
        )

    cfg = ckpt.get("config", {})
    if cfg.get("ch_mean") is not None and cfg.get("ch_std") is not None:
        trainer._ch_mean = np.array(cfg["ch_mean"], dtype=np.float32)
        trainer._ch_std = np.array(cfg["ch_std"], dtype=np.float32)
        print("  restored input normalization stats from prompt checkpoint")
    if cfg.get("inc_mean") is not None and cfg.get("inc_std") is not None:
        trainer._inc_mean = np.array(cfg["inc_mean"], dtype=np.float32)
        trainer._inc_std = np.array(cfg["inc_std"], dtype=np.float32)
        print("  restored increment normalization stats from prompt checkpoint")


def main():
    args = parse_args()
    set_training_seed(args.seed)
    device = resolve_device(args.device, require_gpu=args.require_gpu)
    source_regions = _resolve_source_regions(args.target_region, args.source_regions)
    if args.init_from_source_base_checkpoint:
        if args.model_type != "hyperda_basis_adapter":
            raise ValueError("--init_from_source_base_checkpoint requires --model_type hyperda_basis_adapter")
        validate_source_base_checkpoint_for_staged_init(
            checkpoint_path=args.init_from_source_base_checkpoint,
            expected_width=args.width,
            require_increment_stats=args.target_increment_normalization,
        )

    print("=" * 60)
    print("Phase 4B: Prompt-Conditioned / HyperDA Shared Backbone Training")
    print(f"  target_region={args.target_region}  adaptation_setting={args.adaptation_setting}  K={args.K}  seed={args.seed}")
    print(f"  max_epochs={args.max_epochs}  batch_size={args.batch_size}  lr={args.lr}")
    print(f"  device={device}  width={args.width}  prompt_dim={args.prompt_dim}  model_type={args.model_type}  amp={args.amp}")
    print(
        "  source_saliency_prior="
        f"{args.hyper_source_saliency_prior_path or 'none'} beta={args.hyper_source_saliency_prior_beta} "
        f"application={args.hyper_source_saliency_prior_application}"
    )
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
        "model_type": args.model_type,
        "context_encoder": args.context_encoder,
        "prompt_diagnostic_input_domain": prompt_diagnostic_input_domain(args.context_encoder),
        "normalized_input_used_for_prompt_diagnostics": prompt_normalized_input_used(args.context_encoder),
        "prompt_input_feature_source": prompt_input_feature_source(args.context_encoder),
        "prompt_channel_11_usage": prompt_channel_11_usage(args.context_encoder),
        "width": args.width, "prompt_dim": args.prompt_dim,
        "hyper_n_basis": args.hyper_n_basis,
        "hyper_adapter_bottleneck": args.hyper_adapter_bottleneck,
        "hyper_adapter_scale": args.hyper_adapter_scale,
        "hyper_coeff_generator": args.hyper_coeff_generator,
        "hyper_rank_gate_top_k": args.hyper_rank_gate_top_k,
        "hyper_rank_gate_temperature_init": args.hyper_rank_gate_temperature_init,
        "hyper_adapter_param_style": args.hyper_adapter_param_style,
        "hyper_reliability_gate": args.hyper_reliability_gate,
        "hyper_reliability_init": args.hyper_reliability_init,
        "hyper_source_saliency_prior_path": args.hyper_source_saliency_prior_path,
        "hyper_source_saliency_prior_beta": args.hyper_source_saliency_prior_beta,
        "hyper_source_saliency_prior_application": args.hyper_source_saliency_prior_application,
        "hyper_prompt_manifold_reliability": bool(args.hyper_prompt_manifold_reliability),
        "hyper_prompt_manifold_reliability_strength": args.hyper_prompt_manifold_reliability_strength,
        "hyper_enable_film": bool(args.hyper_enable_film),
        "hyper_enable_adapters": bool(args.hyper_enable_adapters),
        "hyper_residual_magnitude_penalty": args.hyper_residual_magnitude_penalty,
        "hyper_coeff_entropy_floor": args.hyper_coeff_entropy_floor,
        "hyper_coeff_entropy_penalty": args.hyper_coeff_entropy_penalty,
        "zero_shot_prior_form": args.zero_shot_prior_form,
        "source_residual_prior_mode": args.zero_shot_prior_form != "direct_hyper",
        "source_residual_rho": args.source_residual_rho,
        "source_residual_gate": args.source_residual_gate,
        "source_residual_gate_init": args.source_residual_gate_init,
        "source_residual_reliability_dim": args.source_residual_reliability_dim,
        "zero_shot_rho": args.source_residual_rho,
        "zero_shot_rho_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "zero_shot_rho_selection_source": "source_val_regionwise_safe_episode_only",
        "zero_shot_rho_safe_policy": "source_safe_regionwise_non_degradation_surface_rootzone_vs_rho0",
        "reliability_feature_transform": RELIABILITY_FEATURE_TRANSFORM,
        "target_labels_used_for_adaptation": False,
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "final_eval_only_no_selection",
        "target_eval_input_stats_used_for_update": False,
        "max_epochs": args.max_epochs, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip, "accum_steps": args.accum_steps,
        "num_workers": args.num_workers, "device": str(device),
        "use_amp": args.amp,
        "amp_init_scale": args.amp_init_scale,
        "amp_min_scale": args.amp_min_scale,
        "amp_skip_abort_threshold": args.amp_skip_abort_threshold,
        "zero_raw_increment_init": args.zero_raw_increment_init,
        "target_increment_normalization": args.target_increment_normalization,
        "log_every_steps": args.log_every_steps,
        "eval_every_epochs": args.eval_every_epochs,
        "use_lat_weighted_loss": args.use_lat_weighted_loss,
        "wandb_mode": args.wandb_mode,
        "checkpoint_every": args.checkpoint_every,
        "init_from_prompt_checkpoint": args.init_from_prompt_checkpoint,
        "init_from_source_base_checkpoint": args.init_from_source_base_checkpoint,
        "trainable_scope": args.trainable_scope,
        "source_episode_prompt_policy": args.source_episode_prompt_policy,
        "source_anchor_blend_calibration": bool(args.source_anchor_blend_calibration),
        "hyper_output_head_residual": bool(args.hyper_output_head_residual),
        "selection_metric": args.selection_metric,
        "lambda_amp": args.lambda_amp,
        "source_val_residual_gain": not args.no_source_val_residual_gain,
        "splits_json": args.splits_json,
        "split_manifest_path": args.split_manifest_path,
        "protocol_freeze_id": args.protocol_freeze_id,
        "dataset_backend": args.dataset_backend,
        "tensor_cache_dir": args.tensor_cache_dir,
        "max_year_cache_entries": args.max_year_cache_entries,
        "tensor_cache_load_mode": args.tensor_cache_load_mode,
        "train_batch_sampler": args.train_batch_sampler,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "persistent_workers": args.num_workers > 0,
        "pin_memory": str(device) == "cuda",
        "source_prototype_cache_dir": args.source_prototype_cache_dir,
        "source_prototype_cache_mode": args.source_prototype_cache_mode,
        "source_regions": source_regions,
        "source_region_episode_policy": SOURCE_REGION_EPISODE_POLICY,
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
        resume_ckpt_for_config = torch.load(resume_path, map_location="cpu", weights_only=False)
        args.context_encoder = resolve_context_encoder_from_checkpoint(resume_ckpt_for_config)
        run_config["context_encoder"] = args.context_encoder
        print(f"[resume] context_encoder restored from checkpoint: {args.context_encoder}")

    source_saliency_prior, source_saliency_metadata = _load_source_saliency_prior_for_model(
        path=args.hyper_source_saliency_prior_path,
        n_basis=args.hyper_n_basis,
    )
    source_saliency_metadata = _source_saliency_metadata_for_config(
        path=args.hyper_source_saliency_prior_path,
        beta=args.hyper_source_saliency_prior_beta,
        prior=source_saliency_prior,
        application=args.hyper_source_saliency_prior_application,
        metadata=source_saliency_metadata,
    )
    run_config["hyper_source_saliency_prior_metadata"] = source_saliency_metadata
    run_config["hyper_source_saliency_prior"] = (
        source_saliency_prior.tolist() if source_saliency_prior is not None else None
    )

    # RunManager
    run_manager = RunManager(
        phase=PHASE,
        method=args.model_type,
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
    run_manager.save_config(run_config, "config_resolved.yaml")
    run_manager.save_git_info()
    run_manager.save_protocol({
        "protocol_freeze_id": args.protocol_freeze_id,
        "split_manifest": args.split_manifest_path,
        "splits_json": args.splits_json,
    })

    # Wandb
    wandb_logger = WandbLogger(
        mode=args.wandb_mode, project=args.wandb_project,
        entity=args.wandb_entity, tags=args.wandb_tags,
        run_name=run_manager.get_run_name(),
    )

    start_time = time.time()

    # Load source_fit dataset as one episode per source region.
    print(f"\nLoading source_fit dataset...")
    train_dataset = build_hydroda_dataset(
        da_nc_path=DA_NC,
        region_masks_nc=REGION_MASKS_NC,
        splits_json=args.splits_json,
        target_region=args.target_region,
        split_type="source_fit",
        K=args.K,
        seed=args.seed,
        adaptation_setting=args.adaptation_setting,
        freeze_manifest=FREEZE_MANIFEST,
        dataset_backend=args.dataset_backend,
        active_region_ids=source_regions,
        tensor_cache_dir=args.tensor_cache_dir,
        max_year_cache_entries=args.max_year_cache_entries,
        tensor_cache_load_mode=args.tensor_cache_load_mode,
    )
    print(f"  source_fit samples: {len(train_dataset)}")
    print(f"  source regions: {run_config['source_regions']}")

    # Build mapping from global region index (0..5) to source-only index (0..num_source-1)
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
    source_val_dataset = build_hydroda_dataset(
        da_nc_path=DA_NC,
        region_masks_nc=REGION_MASKS_NC,
        splits_json=args.splits_json,
        target_region=args.target_region,
        split_type="source_val",
        K=args.K,
        seed=args.seed,
        adaptation_setting=args.adaptation_setting,
        freeze_manifest=FREEZE_MANIFEST,
        dataset_backend=args.dataset_backend,
        active_region_ids=source_regions,
        tensor_cache_dir=args.tensor_cache_dir,
        max_year_cache_entries=args.max_year_cache_entries,
        tensor_cache_load_mode=args.tensor_cache_load_mode,
    )
    print(f"  source_val samples: {len(source_val_dataset)}")

    # Init model + prompt encoder
    num_source_regions = len(run_config["source_regions"])
    if args.model_type == "hyperda_basis_adapter":
        print(
            f"\nInitializing HyperAdapterConditionalResUNet "
            f"(width={args.width}, prompt_dim={args.prompt_dim}, "
            f"n_basis={args.hyper_n_basis}, adapter_bottleneck={args.hyper_adapter_bottleneck})..."
        )
        model = HyperAdapterConditionalResUNet(
            in_channels=12, out_channels=2, width=args.width,
            prompt_dim=args.prompt_dim,
            hyper_n_basis=args.hyper_n_basis,
            hyper_adapter_bottleneck=args.hyper_adapter_bottleneck,
            hyper_adapter_scale=args.hyper_adapter_scale,
            hyper_coeff_generator=args.hyper_coeff_generator,
            hyper_rank_gate_top_k=args.hyper_rank_gate_top_k,
            hyper_rank_gate_temperature_init=args.hyper_rank_gate_temperature_init,
            hyper_adapter_param_style=args.hyper_adapter_param_style,
            hyper_reliability_gate=args.hyper_reliability_gate,
            hyper_reliability_init=args.hyper_reliability_init,
            hyper_source_saliency_prior=source_saliency_prior,
            hyper_source_saliency_prior_beta=args.hyper_source_saliency_prior_beta,
            hyper_source_saliency_prior_path=args.hyper_source_saliency_prior_path,
            hyper_source_saliency_prior_application=args.hyper_source_saliency_prior_application,
            hyper_prompt_manifold_reliability=bool(args.hyper_prompt_manifold_reliability),
            hyper_prompt_manifold_reliability_strength=args.hyper_prompt_manifold_reliability_strength,
            hyper_enable_film=bool(args.hyper_enable_film),
            hyper_enable_adapters=bool(args.hyper_enable_adapters),
            zero_shot_prior_form=args.zero_shot_prior_form,
            source_residual_rho=args.source_residual_rho,
            source_residual_gate=args.source_residual_gate,
            source_residual_gate_init=args.source_residual_gate_init,
            source_residual_reliability_dim=args.source_residual_reliability_dim,
            zero_raw_increment_init=args.zero_raw_increment_init,
        )
    else:
        print(f"\nInitializing FiLMConditionalResUNet (width={args.width}, prompt_dim={args.prompt_dim})...")
        model = FiLMConditionalResUNet(
            in_channels=12, out_channels=2, width=args.width,
            prompt_dim=args.prompt_dim,
            zero_raw_increment_init=args.zero_raw_increment_init,
        )
    prompt_encoder = build_prompt_encoder(
        context_encoder=args.context_encoder,
        num_regions=num_source_regions,
        input_channels=12,
        hidden_dim=args.prompt_dim,
    )

    source_base_init_metadata: Dict[str, Any] = {}
    source_base_ch_mean = None
    source_base_ch_std = None
    source_base_inc_mean = None
    source_base_inc_std = None
    if args.init_from_source_base_checkpoint:
        if args.model_type != "hyperda_basis_adapter":
            raise ValueError("--init_from_source_base_checkpoint requires --model_type hyperda_basis_adapter")
        source_base_init_metadata = load_source_base_checkpoint_into_hyperda(
            model=model,
            checkpoint_path=args.init_from_source_base_checkpoint,
            expected_width=args.width,
            device=device,
        )
        source_base_ch_mean = (
            np.array(source_base_init_metadata["ch_mean"], dtype=np.float32)
            if source_base_init_metadata.get("ch_mean") is not None
            else None
        )
        source_base_ch_std = (
            np.array(source_base_init_metadata["ch_std"], dtype=np.float32)
            if source_base_init_metadata.get("ch_std") is not None
            else None
        )
        source_base_inc_mean = (
            np.array(source_base_init_metadata["inc_mean"], dtype=np.float32)
            if source_base_init_metadata.get("inc_mean") is not None
            else None
        )
        source_base_inc_std = (
            np.array(source_base_init_metadata["inc_std"], dtype=np.float32)
            if source_base_init_metadata.get("inc_std") is not None
            else None
        )
        if source_base_ch_mean is None or source_base_ch_std is None:
            raise ValueError("source base checkpoint must contain ch_mean/ch_std normalization stats")
        if args.target_increment_normalization and (
            source_base_inc_mean is None or source_base_inc_std is None
        ):
            raise ValueError(
                "source base checkpoint must contain inc_mean/inc_std when target_increment_normalization is enabled"
            )
        run_config["init_from_source_base_checkpoint"] = source_base_init_metadata["checkpoint_path"]
        run_config["source_base_checkpoint_sha256"] = source_base_init_metadata["checkpoint_sha256"]
        run_manager.save_config(run_config, "config.yaml")
        run_manager.save_config(run_config, "config_resolved.yaml")

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
        protocol_freeze_id=args.protocol_freeze_id,
        split_manifest_path=args.split_manifest_path,
        grad_clip=args.grad_clip,
        model_width=args.width,
        prompt_dim=args.prompt_dim,
        target_increment_normalization=args.target_increment_normalization,
        zero_raw_increment_init=args.zero_raw_increment_init,
        accum_steps=args.accum_steps,
        run_manager=run_manager,
        use_amp=args.amp,
        amp_init_scale=args.amp_init_scale,
        amp_min_scale=args.amp_min_scale,
        amp_skip_abort_threshold=args.amp_skip_abort_threshold,
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
        context_encoder=args.context_encoder,
        model_type=args.model_type,
        hyper_n_basis=args.hyper_n_basis,
        hyper_adapter_bottleneck=args.hyper_adapter_bottleneck,
        hyper_adapter_scale=args.hyper_adapter_scale,
        hyper_coeff_generator=args.hyper_coeff_generator,
        hyper_rank_gate_top_k=args.hyper_rank_gate_top_k,
        hyper_rank_gate_temperature_init=args.hyper_rank_gate_temperature_init,
        hyper_adapter_param_style=args.hyper_adapter_param_style,
        hyper_reliability_gate=args.hyper_reliability_gate,
        hyper_reliability_init=args.hyper_reliability_init,
        hyper_source_saliency_prior=source_saliency_prior,
        hyper_source_saliency_prior_beta=args.hyper_source_saliency_prior_beta,
        hyper_source_saliency_prior_path=args.hyper_source_saliency_prior_path,
        hyper_source_saliency_prior_application=args.hyper_source_saliency_prior_application,
        hyper_source_saliency_prior_metadata=source_saliency_metadata,
        hyper_prompt_manifold_reliability=bool(args.hyper_prompt_manifold_reliability),
        hyper_prompt_manifold_reliability_strength=args.hyper_prompt_manifold_reliability_strength,
        hyper_enable_film=bool(args.hyper_enable_film),
        hyper_enable_adapters=bool(args.hyper_enable_adapters),
        hyper_residual_magnitude_penalty=args.hyper_residual_magnitude_penalty,
        hyper_coeff_entropy_floor=args.hyper_coeff_entropy_floor,
        hyper_coeff_entropy_penalty=args.hyper_coeff_entropy_penalty,
        zero_shot_prior_form=args.zero_shot_prior_form,
        source_residual_rho=args.source_residual_rho,
        source_residual_gate=args.source_residual_gate,
        source_residual_gate_init=args.source_residual_gate_init,
        source_residual_reliability_dim=args.source_residual_reliability_dim,
        trainable_scope=args.trainable_scope,
        source_episode_prompt_policy=args.source_episode_prompt_policy,
        source_anchor_blend_calibration=bool(args.source_anchor_blend_calibration),
        hyper_output_head_residual=bool(args.hyper_output_head_residual),
        init_from_source_base_checkpoint=source_base_init_metadata.get("checkpoint_path"),
        source_base_checkpoint_sha256=source_base_init_metadata.get("checkpoint_sha256", ""),
        source_base_checkpoint_config=source_base_init_metadata.get("source_config", {}),
        source_base_loaded_parameter_names=source_base_init_metadata.get("loaded_parameter_names", []),
        dataset_backend=args.dataset_backend,
        tensor_cache_load_mode=args.tensor_cache_load_mode,
        train_batch_sampler=args.train_batch_sampler,
        prefetch_factor=args.prefetch_factor,
        source_prototype_cache_dir=args.source_prototype_cache_dir,
        source_prototype_cache_mode=args.source_prototype_cache_mode,
        _resume_ch_mean=source_base_ch_mean,
        _resume_ch_std=source_base_ch_std,
        _resume_inc_mean=source_base_inc_mean,
        _resume_inc_std=source_base_inc_std,
    )

    # Resume: restore full training state after Trainer creation
    if resumed_epoch > 0 and ckpt is not None:
        print(f"\nRestoring training state from checkpoint (resuming from epoch {resumed_epoch})...")
        trainer.load_state(ckpt)
        trainer.refresh_source_context_monthly_prototypes()
        print(f"  Restored: optimizer, scheduler, epoch, best_loss, train_history")
        print(f"  train_history entries so far: {len(trainer.train_history)}")
        print(f"  val_history entries so far: {len(trainer.val_history)}")
    elif args.init_from_prompt_checkpoint:
        _load_prompt_checkpoint_initialization(
            trainer=trainer,
            checkpoint_path=args.init_from_prompt_checkpoint,
            device=device,
        )
        trainer.refresh_source_context_monthly_prototypes()

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
