#!/usr/bin/env python3
"""HyperDA zero/few-shot target adaptation with preregistered final checkpoints.

No-leakage declaration:
    - Loads a source-trained HyperDA checkpoint.
    - Freezes source backbone, prompt encoder, hypernetwork, and basis bank.
    - K=0 builds target context prompt state only; it performs no target-label
      training.
    - K=4/12 trains only lightweight target-specific variables on target_support.
    - Does not construct target_val and does not use target-side early stopping.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hydroda.data.dataset import HydroDADataset, collate_hydroda_samples
from hydroda.data.file_hash import compute_sha256
from hydroda.data.protocol import ProtocolConfig
from hydroda.baselines.prompt_conditioned import (
    build_target_context_prompt_state,
    compose_target_context_prompt_from_state,
    normalize_target_context_prompt_state,
    target_context_prompt_metadata,
)
from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.prompt_encoder import RegionPromptEncoder
from hydroda.training.losses import MaskedHuberLoss, WeightedMaskedHuberLoss
from hydroda.utils.run_manager import RunManager
from hydroda.utils.runtime import get_git_hash, get_timestamp

from scripts.train.train_hyperda_target_adapt import (
    TargetAdaptationState,
    _analysis_loss,
    _as_list,
    _collate_target_batch,
    _denormalize_increment,
    _model_forward,
    _normalize_x,
    _target_region_embedding,
    _target_tensor,
    adaptation_regularization,
    apply_target_adaptation_stage,
    apply_target_adapter_state,
    extract_target_adapter_state,
    interpolate_target_adapter_state,
)


DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_zero_few_shot_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
PROTOCOL = ProtocolConfig()
PROTOCOL_FREEZE_ID = PROTOCOL.protocol_freeze_id
PHASE = "phase5_hyperda_zero_few_shot"


@dataclass
class FewShotAdaptationState(TargetAdaptationState):
    """State container for the zero/few-shot runner."""


def default_steps_for_K(K: int) -> int:
    if int(K) == 0:
        return 0
    if int(K) == 4:
        return 100
    if int(K) == 12:
        return 80
    raise ValueError(f"unsupported K={K}; expected one of {list(PROTOCOL.main_K_values)}")


def default_lr_for_K(K: int) -> float:
    if int(K) == 12:
        return 3e-4
    return 1e-3


def default_anchor_alpha_for_K(K: int) -> float:
    if int(K) == 0:
        return 0.0
    if int(K) == 4:
        return 0.75
    if int(K) == 12:
        return 0.25
    raise ValueError(f"unsupported K={K}; expected one of {list(PROTOCOL.main_K_values)}")


def default_anchor_alpha_grid_for_K(K: int) -> List[float]:
    if int(K) == 4:
        return [0.25, 0.5, 0.75, 1.0]
    if int(K) == 12:
        return [0.1, 0.25, 0.5, 0.75, 1.0]
    return [0.0]


def build_dataset_plan(K: int) -> List[str]:
    PROTOCOL.assert_supported_K(K)
    if int(K) == 0:
        return ["target_context"]
    return ["target_context", "target_support"]


def method_id_for_adaptation_setting(adaptation_setting: str, K: int) -> str:
    """Return paper-facing method IDs for the main HyperDA K-axis."""
    if adaptation_setting == "zero_shot_context" or int(K) == 0:
        return "hyperda_zero_shot_context"
    if adaptation_setting == "few_shot_k4" or int(K) == 4:
        return "hyperda_few_shot_k4"
    if adaptation_setting == "few_shot_k12" or int(K) == 12:
        return "hyperda_few_shot_k12"
    raise ValueError(f"unsupported HyperDA adaptation setting: {adaptation_setting!r}, K={K}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HyperDA zero/few-shot target variables")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--target_region", type=str, required=True)
    parser.add_argument("--K", type=int, required=True, choices=list(PROTOCOL.main_K_values))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adaptation_setting", type=str, default=None)
    parser.add_argument("--allow_legacy_full_target_train", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--da_nc", type=str, default=DA_NC)
    parser.add_argument("--region_masks_nc", type=str, default=REGION_MASKS_NC)
    parser.add_argument("--splits_json", type=str, default=SPLITS_JSON)
    parser.add_argument("--freeze_manifest", type=str, default=FREEZE_MANIFEST)
    parser.add_argument("--target_latent_dim", type=int, default=32)
    parser.add_argument("--adaptation_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_lat_weighted_loss", action="store_true", default=True)
    parser.add_argument("--no_lat_weighted_loss", action="store_false", dest="use_lat_weighted_loss")
    parser.add_argument("--lambda_prior", type=float, default=1e-3)
    parser.add_argument("--lambda_latent", type=float, default=1e-3)
    parser.add_argument("--lambda_gain", type=float, default=1e-2)
    parser.add_argument("--lambda_gain_smooth", type=float, default=1e-3)
    parser.add_argument("--lambda_analysis", type=float, default=0.25)
    parser.add_argument("--surface_weight", type=float, default=3.0)
    parser.add_argument("--rootzone_weight", type=float, default=1.0)
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument(
        "--adapt_recipe",
        type=str,
        default="source_anchor",
        choices=["conservative", "source_anchor", "episode_prior"],
        help="Preregistered few-shot adaptation recipe; target labels never select this value.",
    )
    parser.add_argument(
        "--anchor_alpha",
        type=float,
        default=None,
        help="Fixed source-anchor interpolation alpha. Defaults by K from source-side episodic validation.",
    )
    parser.add_argument(
        "--source_anchor_hyperparameter_source",
        type=str,
        default="source_side_episodic_validation_preregistered",
        help="Metadata field naming the non-target source of alpha/lr/step choices.",
    )
    parser.add_argument("--enable_target_spatial_refine", action="store_true",
        help="Legacy/internal ablation; disabled by default for the main protocol.")

    args = parser.parse_args()
    PROTOCOL.assert_supported_K(args.K)
    if args.adaptation_setting is None:
        args.adaptation_setting = PROTOCOL.adaptation_setting_for_K(args.K)
    try:
        PROTOCOL.assert_supported_adaptation_setting(
            args.adaptation_setting,
            allow_legacy_full_target_train=args.allow_legacy_full_target_train,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.adaptation_setting == "target_full_train" and not args.allow_legacy_full_target_train:
        parser.error("--adaptation_setting target_full_train requires --allow_legacy_full_target_train")
    if args.adaptation_steps is None:
        args.adaptation_steps = default_steps_for_K(args.K)
    if args.lr is None:
        args.lr = default_lr_for_K(args.K)
    if args.anchor_alpha is None:
        args.anchor_alpha = default_anchor_alpha_for_K(args.K)
    if not 0.0 <= float(args.anchor_alpha) <= 1.0:
        parser.error("--anchor_alpha must be in [0, 1]")
    if int(args.K) == 0 and int(args.adaptation_steps) != 0:
        parser.error("K=0 must use adaptation_steps=0")
    if int(args.K) == 0 and abs(float(args.anchor_alpha)) > 1e-12:
        parser.error("K=0 must use anchor_alpha=0")
    args.target_val_usage = "unused_in_main_protocol"
    args.model_selection_source = "source_val_preregistered"
    return args


def load_source_checkpoint_for_few_shot(
    checkpoint_path: str,
    device: torch.device,
    target_latent_dim: int = 32,
    enable_target_spatial_refine: bool = False,
) -> FewShotAdaptationState:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"source checkpoint not found: {ckpt_path}")
    source_checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    source_config = dict(source_checkpoint.get("config", {}))
    model_type = source_config.get("model_type", "prompt_conditioned")
    if model_type != "hyperda_basis_adapter":
        raise ValueError(
            "train_hyperda_few_shot_adapt.py requires config.model_type="
            f"'hyperda_basis_adapter', got {model_type!r}"
        )

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=int(source_config.get("width", 32)),
        prompt_dim=int(source_config.get("prompt_dim", 64)),
        hyper_n_basis=int(source_config.get("hyper_n_basis", 8)),
        hyper_adapter_bottleneck=source_config.get("hyper_adapter_bottleneck"),
        hyper_adapter_scale=float(source_config.get("hyper_adapter_scale", 1.0)),
        zero_raw_increment_init=bool(source_config.get("zero_raw_increment_init", False)),
        enable_target_adaptation=True,
        target_latent_dim=target_latent_dim,
        enable_target_spatial_refine=enable_target_spatial_refine,
    )
    load_result = model.load_state_dict(source_checkpoint["model_state_dict"], strict=False)
    unexpected = [k for k in load_result.unexpected_keys if not k.startswith("target_")]
    if unexpected:
        raise RuntimeError(f"unexpected source checkpoint model keys: {unexpected[:8]}")
    model.to(device)
    model.freeze_source_prior_for_target_adaptation()

    prompt_encoder = RegionPromptEncoder(
        num_regions=int(source_config.get("num_regions", len(source_config.get("source_regions", [])) or 6)),
        input_channels=12,
        hidden_dim=int(source_config.get("prompt_dim", 64)),
    )
    prompt_state = source_checkpoint.get("prompt_encoder_state_dict")
    if prompt_state is not None:
        prompt_encoder.load_state_dict(prompt_state)
    prompt_encoder.to(device).eval()
    for param in prompt_encoder.parameters():
        param.requires_grad_(False)

    normalization = {
        "ch_mean": _as_list(source_config.get("ch_mean"), 12, 0.0),
        "ch_std": _as_list(source_config.get("ch_std"), 12, 1.0),
        "inc_mean": _as_list(source_config.get("inc_mean"), 2, 0.0) if source_config.get("inc_mean") is not None else None,
        "inc_std": _as_list(source_config.get("inc_std"), 2, 1.0) if source_config.get("inc_std") is not None else None,
    }
    return FewShotAdaptationState(
        model=model,
        prompt_encoder=prompt_encoder,
        source_checkpoint=source_checkpoint,
        source_config=source_config,
        normalization=normalization,
    )


def _loader(dataset: HydroDADataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=_collate_target_batch,
    )


def build_few_shot_target_context_prompt_state(
    state: FewShotAdaptationState,
    samples: Iterable[Dict[str, Any]],
    target_region: str,
    device: torch.device,
    context_hash: str = "",
) -> Dict[str, Any]:
    return build_target_context_prompt_state(
        samples=samples,
        prompt_encoder=state.prompt_encoder,
        normalize_x=lambda x: _normalize_x(x, state.normalization),
        target_region_embedding=_target_region_embedding(state, target_region, device),
        device=device,
        context_hash=context_hash,
    )


def few_shot_batch_loss(
    state: FewShotAdaptationState,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    loss_fn: nn.Module,
    normalize_increment: bool,
    lambda_prior: float,
    lambda_latent: float,
    lambda_gain: float,
    lambda_gain_smooth: float,
    lambda_analysis: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Compute few-shot loss with the frozen target-context monthly prompt state.

    Support batch inputs provide fields and supervised losses only; they do not
    update or summarize the target-context prompt state.
    """
    x = batch["x"].to(device)
    months = batch["months"].to(device)
    x_norm = _normalize_x(x, state.normalization)
    z = compose_target_context_prompt_from_state(target_context_prompt_state, months, device=device)
    pred = _model_forward(state.model, x_norm, z, months, x)
    target = _target_tensor(
        batch["increment_surface"].to(device),
        batch["increment_rootzone"].to(device),
        state.normalization,
        normalize_increment=normalize_increment,
    )
    loss_mask = batch["loss_mask"].to(device)
    latitude_weight = batch.get("latitude_weight")
    if latitude_weight is not None:
        latitude_weight = latitude_weight.to(device)
    if isinstance(loss_fn, WeightedMaskedHuberLoss):
        inc_scale = torch.ones(2, dtype=torch.float32, device=device) if normalize_increment else None
        losses = loss_fn(pred, target, loss_mask, latitude_weight=latitude_weight, increment_scale=inc_scale)
    else:
        losses = loss_fn(pred, target, loss_mask)
    analysis_losses = _analysis_loss(
        pred=pred,
        target=target,
        batch=batch,
        normalization=state.normalization,
        normalize_increment=normalize_increment,
        loss_fn=loss_fn,
        loss_mask=loss_mask,
        latitude_weight=latitude_weight,
    )
    if analysis_losses is not None:
        losses["analysis_loss"] = analysis_losses["total_loss"]
        losses["analysis_surface_loss"] = analysis_losses["surface_loss"].detach()
        losses["analysis_rootzone_loss"] = analysis_losses["rootzone_loss"].detach()
    else:
        losses["analysis_loss"] = torch.zeros((), dtype=losses["total_loss"].dtype, device=losses["total_loss"].device)
    if "forecast_surface" in batch and "forecast_rootzone" in batch:
        forecast = torch.stack(
            [
                batch["forecast_surface"].to(pred.device),
                batch["forecast_rootzone"].to(pred.device),
            ],
            dim=1,
        )
        pred_analysis = forecast + _denormalize_increment(pred, state.normalization, normalize_increment)
        true_analysis = forecast + _denormalize_increment(target, state.normalization, normalize_increment)
        losses["pred_analysis_physical"] = pred_analysis.detach()
        losses["true_analysis_physical"] = true_analysis.detach()
    reg = adaptation_regularization(state.model, lambda_prior, lambda_latent, lambda_gain, lambda_gain_smooth)
    losses["regularization_loss"] = reg.detach()
    losses["objective"] = losses["total_loss"] + lambda_analysis * losses["analysis_loss"] + reg
    return losses


def train_fixed_steps(
    state: FewShotAdaptationState,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    loss_fn: nn.Module,
    normalize_increment: bool,
    adaptation_steps: int,
    grad_clip: Optional[float],
    lambda_prior: float,
    lambda_latent: float,
    lambda_gain: float,
    lambda_gain_smooth: float,
    lambda_analysis: float,
    log_every_steps: int = 50,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    if adaptation_steps <= 0:
        return history
    state.model.train()
    data_iter = iter(loader)
    for step in range(1, adaptation_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        optimizer.zero_grad(set_to_none=True)
        losses = few_shot_batch_loss(
            state,
            batch,
            device,
            target_context_prompt_state,
            loss_fn,
            normalize_increment,
            lambda_prior,
            lambda_latent,
            lambda_gain,
            lambda_gain_smooth,
            lambda_analysis=lambda_analysis,
        )
        losses["objective"].backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([p for p in state.model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        row = {
            "step": float(step),
            "objective": float(losses["objective"].detach().cpu()),
            "total_loss": float(losses["total_loss"].detach().cpu()),
            "analysis_loss": float(losses["analysis_loss"].detach().cpu()),
            "regularization_loss": float(losses["regularization_loss"].detach().cpu()),
        }
        history.append(row)
        if log_every_steps > 0 and (step == 1 or step % log_every_steps == 0 or step == adaptation_steps):
            print(
                f"step={step}/{adaptation_steps} "
                f"objective={row['objective']:.6f} total={row['total_loss']:.6f} "
                f"analysis={row['analysis_loss']:.6f}",
                flush=True,
            )
    return history


def _drift_group_for_key(name: str) -> str:
    if name.startswith("target_prompt."):
        return "target_prompt"
    if name.startswith("target_adapter_coefficient_residual_"):
        return "adapter_coefficient_residuals"
    if name.startswith("residual_gain."):
        return "monthly_residual_gain"
    if name.startswith("target_spatial_refine."):
        return "target_spatial_refine"
    return "other_target_parameters"


def target_parameter_l2_drift(
    anchor_state: Dict[str, torch.Tensor],
    adapted_state: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Return L2 drift from source/prior initialization for target-only state."""
    if set(anchor_state) != set(adapted_state):
        missing = sorted(set(anchor_state) - set(adapted_state))
        extra = sorted(set(adapted_state) - set(anchor_state))
        raise ValueError(f"adapter state keys differ; missing={missing[:5]} extra={extra[:5]}")
    group_sq: Dict[str, float] = {}
    total_sq = 0.0
    for name, anchor_tensor in anchor_state.items():
        adapted_tensor = adapted_state[name]
        if tuple(anchor_tensor.shape) != tuple(adapted_tensor.shape):
            raise ValueError(
                f"shape mismatch for {name}: anchor={tuple(anchor_tensor.shape)} adapted={tuple(adapted_tensor.shape)}"
            )
        diff = adapted_tensor.detach().cpu().float() - anchor_tensor.detach().cpu().float()
        sq = float(diff.square().sum().item())
        total_sq += sq
        group = _drift_group_for_key(name)
        group_sq[group] = group_sq.get(group, 0.0) + sq
    drift = {group: float(value ** 0.5) for group, value in sorted(group_sq.items())}
    drift["total"] = float(total_sq ** 0.5)
    return drift


def apply_source_anchor_interpolation(
    model: nn.Module,
    anchor_state: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Apply fixed ``theta_init + alpha * (theta_adapt - theta_init)`` to target tensors only."""
    adapted_state = extract_target_adapter_state(model)
    interpolated_state = interpolate_target_adapter_state(anchor_state, adapted_state, float(alpha))
    apply_target_adapter_state(model, interpolated_state)
    return interpolated_state


def support_loss_summary(train_history: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not train_history:
        return {"support_final_loss": None, "support_loss_delta": None}
    first = train_history[0]
    last = train_history[-1]
    first_loss = first.get("total_loss", first.get("objective"))
    final_loss = last.get("total_loss", last.get("objective"))
    if first_loss is None or final_loss is None:
        return {"support_final_loss": None, "support_loss_delta": None}
    return {
        "support_final_loss": float(final_loss),
        "support_loss_delta": float(final_loss) - float(first_loss),
    }


def _date_str_records(dataset: Optional[HydroDADataset], date_key: str) -> List[str]:
    if dataset is None:
        return []
    records = getattr(dataset, "_split_entry", {}).get(date_key, [])
    if not isinstance(records, list):
        return []
    return [str(record.get("date_str", "")) for record in records if isinstance(record, dict) and record.get("date_str")]


def save_few_shot_checkpoint(
    path: Path,
    state: FewShotAdaptationState,
    optimizer_state_dict: Dict[str, Any],
    config: Dict[str, Any],
    target_context_prompt_state: Dict[str, Any],
    train_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_names = state.model.target_trainable_parameter_names()
    prompt_state = normalize_target_context_prompt_state(target_context_prompt_state)
    prompt_metadata = target_context_prompt_metadata(prompt_state)
    full_config = dict(config)
    method_id = method_id_for_adaptation_setting(
        str(full_config.get("adaptation_setting", "")),
        int(full_config.get("K", 0)),
    )
    full_config.update(
        {
            "method": method_id,
            "model_type": "hyperda_basis_adapter_target_adapt",
            "protocol_freeze_id": PROTOCOL_FREEZE_ID,
            "target_context_period": "2015-2021",
            "target_support_period": "2015-2021",
            "target_val_period": "unused_in_main_protocol",
            "target_eval_period": "2023-2025",
            "frozen_modules": ["source_backbone", "prompt_encoder", "hypernetwork", "adapter_basis_bank"],
            "trainable_modules": [
                "target_prompt",
                "adapter_coefficient_residuals",
                "monthly_residual_gain",
            ],
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": int(sum(p.numel() for p in state.model.parameters() if p.requires_grad)),
            "adapt_recipe": full_config.get("adapt_recipe", "source_anchor"),
            "anchor_alpha": float(full_config.get("anchor_alpha", default_anchor_alpha_for_K(int(full_config.get("K", 0))))),
            "anchor_alpha_grid_preregistered": full_config.get(
                "anchor_alpha_grid_preregistered",
                default_anchor_alpha_grid_for_K(int(full_config.get("K", 0))),
            ),
            "source_anchor_hyperparameter_source": full_config.get(
                "source_anchor_hyperparameter_source",
                "source_side_episodic_validation_preregistered",
            ),
            "model_selection_source": "source_val_preregistered",
            "target_val_usage": "unused_in_main_protocol",
            "checkpoint_selection": "fixed_preregistered_final_step",
            "target_eval_usage": "final_eval_only_no_training_no_selection",
            "target_context_prompt_state": prompt_state,
            "target_context_prompt_state_summary": prompt_metadata,
            "prompt_policy": prompt_metadata["prompt_source"],
            "prompt_label_usage": prompt_metadata["label_usage"],
            "eval_input_usage": prompt_metadata["eval_input_usage"],
            "eval_month_usage": prompt_metadata["eval_month_usage"],
            "normalization_source": "source_fit_only_from_source_checkpoint",
            "leakage_guard_status": "pass",
            "git_hash": get_git_hash(),
            "timestamp": get_timestamp(),
        }
    )
    checkpoint = {
        "tag": "final_preregistered",
        "epoch": 0,
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "model_state_dict": state.model.state_dict(),
        "prompt_encoder_state_dict": state.prompt_encoder.state_dict(),
        "target_context_prompt_state": prompt_state,
        "optimizer_state_dict": optimizer_state_dict,
        "source_checkpoint_config": state.source_config,
        "train_history": train_history,
        "config": full_config,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
    }
    torch.save(checkpoint, path)
    return full_config


def write_run_metadata_sidecar(output_dir: Path, checkpoint_path: Path, config: Dict[str, Any]) -> None:
    """Write JSON-safe run metadata mirroring required checkpoint metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_state = config.get("target_context_prompt_state")
    prompt_state_summary: Dict[str, Any] = {}
    if isinstance(prompt_state, dict):
        try:
            prompt_state_summary = target_context_prompt_metadata(prompt_state)
        except Exception:
            prompt_state_summary = {
                "schema_version": prompt_state.get("schema_version", ""),
                "prompt_source": prompt_state.get("prompt_source", ""),
                "label_usage": prompt_state.get("label_usage", ""),
                "context_hash": prompt_state.get("context_hash", ""),
                "context_date_hash": prompt_state.get("context_date_hash", prompt_state.get("context_hash", "")),
            }
    metadata = {
        "checkpoint": str(checkpoint_path),
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "method": config.get("method", ""),
        "adaptation_setting": config.get("adaptation_setting", ""),
        "K": config.get("K", None),
        "seed": config.get("seed", None),
        "target_region": config.get("target_region", ""),
        "split_manifest_path": config.get("split_manifest_path", ""),
        "split_manifest_sha256": config.get("split_manifest_sha256", ""),
        "target_context_dates_hash": config.get("target_context_dates_hash", ""),
        "target_support_dates_hash": config.get("target_support_dates_hash", ""),
        "target_support_dates": list(config.get("target_support_dates", [])),
        "target_eval_dates_hash": config.get("target_eval_dates_hash", ""),
        "target_context_prompt_state": prompt_state_summary,
        "trainable_parameter_count": config.get("trainable_parameter_count", 0),
        "trainable_parameter_names": list(config.get("trainable_parameter_names", [])),
        "adaptation_steps": config.get("adaptation_steps", 0),
        "lr": config.get("lr", None),
        "adapt_recipe": config.get("adapt_recipe", ""),
        "anchor_alpha": config.get("anchor_alpha", None),
        "anchor_alpha_grid_preregistered": list(config.get("anchor_alpha_grid_preregistered", [])),
        "source_anchor_hyperparameter_source": config.get("source_anchor_hyperparameter_source", ""),
        "support_final_loss": config.get("support_final_loss", None),
        "support_loss_delta": config.get("support_loss_delta", None),
        "target_parameter_l2_drift": dict(config.get("target_parameter_l2_drift", {})),
        "normalization_source": config.get("normalization_source", ""),
        "model_selection_source": config.get("model_selection_source", ""),
        "target_val_usage": config.get("target_val_usage", ""),
        "target_eval_usage": config.get("target_eval_usage", ""),
        "checkpoint_selection": config.get("checkpoint_selection", ""),
        "prompt_policy": config.get("prompt_policy", ""),
        "prompt_label_usage": config.get("prompt_label_usage", ""),
        "eval_input_usage": config.get("eval_input_usage", ""),
        "eval_month_usage": config.get("eval_month_usage", ""),
        "frozen_modules": list(config.get("frozen_modules", [])),
        "trainable_modules": list(config.get("trainable_modules", [])),
        "git_hash": config.get("git_hash", ""),
        "timestamp": config.get("timestamp", ""),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _split_hashes(dataset: HydroDADataset) -> Dict[str, str]:
    entry = dataset._split_entry
    return {
        "target_context_dates_hash": entry.get("target_context_dates_hash", entry.get("target_train_dates_hash", "")),
        "target_support_dates_hash": entry.get("target_support_dates_hash", entry.get("support_dates_hash", "")),
        "target_eval_dates_hash": entry.get("target_eval_dates_hash", entry.get("target_query_dates_hash", "")),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"hyperda_zero_few_shot_{args.target_region}_K{args.K}_s{args.seed}"
    output_dir = Path(args.output_dir) if args.output_dir else RunManager(PHASE).create_run_dir(run_name)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=args.source_checkpoint,
        device=device,
        target_latent_dim=args.target_latent_dim,
        enable_target_spatial_refine=args.enable_target_spatial_refine,
    )
    apply_target_adaptation_stage(state.model, epoch=0, stage1_epochs=0)
    anchor_adapter_state = extract_target_adapter_state(state.model)

    target_context_dataset = HydroDADataset(
        da_nc_path=args.da_nc,
        region_masks_nc=args.region_masks_nc,
        splits_json=args.splits_json,
        target_region=args.target_region,
        split_type="target_context",
        K=args.K,
        seed=args.seed,
        adaptation_setting=args.adaptation_setting,
        freeze_manifest=args.freeze_manifest,
    )
    support_dataset = None
    if args.K > 0:
        support_dataset = HydroDADataset(
            da_nc_path=args.da_nc,
            region_masks_nc=args.region_masks_nc,
            splits_json=args.splits_json,
            target_region=args.target_region,
            split_type="target_support",
            K=args.K,
            seed=args.seed,
            adaptation_setting=args.adaptation_setting,
            freeze_manifest=args.freeze_manifest,
        )

    split_hashes = _split_hashes(target_context_dataset)
    target_context_samples = (
        target_context_dataset.get_input_side_sample(i)
        for i in range(len(target_context_dataset))
    )
    target_context_prompt_state = build_few_shot_target_context_prompt_state(
        state=state,
        samples=target_context_samples,
        target_region=args.target_region,
        device=device,
        context_hash=split_hashes.get("target_context_dates_hash", ""),
    )
    prompt_metadata = target_context_prompt_metadata(target_context_prompt_state)
    print(
        "Target-context monthly prompt prototypes: "
        f"n={prompt_metadata['n_samples']} "
        f"dates={prompt_metadata['date_start']}..{prompt_metadata['date_end']} "
        f"labels={prompt_metadata['label_usage']}",
        flush=True,
    )
    train_history: List[Dict[str, Any]] = []
    optimizer = torch.optim.AdamW(
        [p for p in state.model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    try:
        if args.K > 0:
            assert support_dataset is not None
            loader = _loader(support_dataset, args.batch_size, args.num_workers, shuffle=True)
            loss_fn: nn.Module
            if args.use_lat_weighted_loss:
                loss_fn = WeightedMaskedHuberLoss(
                    delta=1.0,
                    surface_weight=args.surface_weight,
                    rootzone_weight=args.rootzone_weight,
                )
            else:
                loss_fn = MaskedHuberLoss(
                    delta=1.0,
                    surface_weight=args.surface_weight,
                    rootzone_weight=args.rootzone_weight,
                )
            started = time.time()
            train_history = train_fixed_steps(
                state=state,
                loader=loader,
                optimizer=optimizer,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                loss_fn=loss_fn,
                normalize_increment=state.normalization.get("inc_mean") is not None,
                adaptation_steps=args.adaptation_steps,
                grad_clip=args.grad_clip,
                lambda_prior=args.lambda_prior,
                lambda_latent=args.lambda_latent,
                lambda_gain=args.lambda_gain,
                lambda_gain_smooth=args.lambda_gain_smooth,
                lambda_analysis=args.lambda_analysis,
                log_every_steps=args.log_every_steps,
            )
            print(f"Fixed-step few-shot training finished in {time.time() - started:.1f}s")
        else:
            print("K=0: skipping target-label training; saving source prior with target-context metadata.")

        if args.K > 0:
            if args.adapt_recipe in {"source_anchor", "conservative", "episode_prior"}:
                apply_source_anchor_interpolation(state.model, anchor_adapter_state, alpha=args.anchor_alpha)
                print(
                    "Applied source-anchor interpolation: "
                    f"recipe={args.adapt_recipe} alpha={args.anchor_alpha:.4f}",
                    flush=True,
                )
        else:
            apply_source_anchor_interpolation(state.model, anchor_adapter_state, alpha=0.0)

        final_adapter_state = extract_target_adapter_state(state.model)
        drift = target_parameter_l2_drift(anchor_adapter_state, final_adapter_state)
        loss_summary = support_loss_summary(train_history)
        split_manifest_sha256 = compute_sha256(args.splits_json) if Path(args.splits_json).exists() else ""
        config = {
            "K": args.K,
            "adaptation_setting": args.adaptation_setting,
            "adapt_recipe": args.adapt_recipe,
            "anchor_alpha": float(args.anchor_alpha),
            "anchor_alpha_grid_preregistered": default_anchor_alpha_grid_for_K(args.K),
            "source_anchor_hyperparameter_source": args.source_anchor_hyperparameter_source,
            "target_region": args.target_region,
            "seed": args.seed,
            "source_checkpoint": args.source_checkpoint,
            "split_manifest_path": args.splits_json,
            "split_manifest_sha256": split_manifest_sha256,
            "adaptation_steps": args.adaptation_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_prior": args.lambda_prior,
            "lambda_latent": args.lambda_latent,
            "lambda_gain": args.lambda_gain,
            "lambda_gain_smooth": args.lambda_gain_smooth,
            **loss_summary,
            "target_parameter_l2_drift": drift,
            "target_support_dates": _date_str_records(support_dataset, "target_support_dates"),
            "target_latent_dim": args.target_latent_dim,
            "enable_target_spatial_refine": args.enable_target_spatial_refine,
            **split_hashes,
        }
        final_path = checkpoints_dir / "checkpoint_final_preregistered.pt"
        saved_config = save_few_shot_checkpoint(
            path=final_path,
            state=state,
            optimizer_state_dict=optimizer.state_dict(),
            config=config,
            target_context_prompt_state=target_context_prompt_state,
            train_history=train_history,
        )
        write_run_metadata_sidecar(output_dir, final_path, saved_config)
        print(f"Saved: {final_path}")
    finally:
        target_context_dataset.close()
        if support_dataset is not None:
            support_dataset.close()


if __name__ == "__main__":
    main()
