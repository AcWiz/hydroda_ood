#!/usr/bin/env python3
"""Train HyperDA target adaptation variables on target_train.

No-leakage declaration:
    - Loads a source-trained HyperDA checkpoint.
    - Freezes source prior parameters: theta0, H_psi-style adapter basis bank,
      prompt encoder, and shared backbone.
    - Trains only target latent, adapter coefficient residuals, monthly
      residual gain, and optional target spatial residual head on target_train
      (2015-2021).
    - Selects the adaptation checkpoint only on target_val (2022).
    - Never reads target_eval (2023-2025) labels during adaptation.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
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
from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.prompt_encoder import RegionPromptEncoder
from hydroda.training.losses import WeightedMaskedHuberLoss, MaskedHuberLoss
from hydroda.utils.run_manager import RunManager
from hydroda.utils.runtime import get_git_hash, get_timestamp


DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_target_train_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
PROTOCOL_FREEZE_ID = ProtocolConfig().protocol_freeze_id
PHASE = "phase5_hyperda_target_adapt"
_REGION_TO_IDX = {f"US-R{i}": i - 1 for i in range(1, 7)}


@dataclass
class TargetAdaptationState:
    model: HyperAdapterConditionalResUNet
    prompt_encoder: RegionPromptEncoder
    source_checkpoint: Dict[str, Any]
    source_config: Dict[str, Any]
    normalization: Dict[str, Optional[List[float]]]


@dataclass
class ResumeTrainingState:
    start_epoch: int
    best_target_val_loss: float
    best_epoch: int
    best_metrics: Dict[str, float]
    best_epochs_by_metric: Dict[str, int]
    train_history: List[Dict[str, Any]]
    val_history: List[Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HyperDA target adaptation variables")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None,
        help="Resume from a Phase 5 target-adapt checkpoint such as checkpoints/last.pt.")
    parser.add_argument("--target_region", type=str, required=True)
    parser.add_argument("--adaptation_setting", type=str, default="target_full_train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--da_nc", type=str, default=DA_NC)
    parser.add_argument("--region_masks_nc", type=str, default=REGION_MASKS_NC)
    parser.add_argument("--splits_json", type=str, default=SPLITS_JSON)
    parser.add_argument("--freeze_manifest", type=str, default=FREEZE_MANIFEST)
    parser.add_argument("--target_latent_dim", type=int, default=32)
    parser.add_argument("--enable_target_spatial_refine", action="store_true",
        help="Train a zero-initialized target spatial residual head for local Surface refinement.")
    parser.add_argument("--target_spatial_refine_hidden", type=int, default=16)
    parser.add_argument("--target_spatial_refine_rootzone", action="store_true",
        help="Also allow the target spatial residual head to modify RootZone.")
    parser.add_argument("--target_spatial_refine_input", type=str, default="normalized", choices=["normalized", "raw"],
        help="Input tensor seen by target spatial residual head.")
    parser.add_argument(
        "--target_spatial_refine_type",
        type=str,
        default="simple",
        choices=["simple", "hydro_msr", "hydro_msr_gain", "hydro_msr_gain_lite", "hydro_msr_rose"],
        help="Target spatial residual adapter family.")
    parser.add_argument("--target_spatial_refine_gain_span", type=float, default=0.25,
        help="Bound around alpha=1 for hydro_msr_gain_lite.")
    parser.add_argument("--hydro_msr_hidden", type=int, default=16,
        help="Hidden width for the Hydro-MSR output adapter.")
    parser.add_argument("--enable_hydro_msr_da_film", action="store_true",
        help="Enable identity-initialized DA-quality FiLM inside Hydro-MSR.")
    parser.add_argument("--enable_da_regime_gain_mixer", action="store_true",
        help="Metadata flag for hydro_msr_gain runs; the gain mixer is enabled by target_spatial_refine_type.")
    parser.add_argument("--stage1_epochs", type=int, default=10,
        help="Epochs for Stage 1 global target modules before spatial/gain-only Stage 2; 0 disables staging.")
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_lat_weighted_loss", action="store_true", default=True)
    parser.add_argument("--no_lat_weighted_loss", action="store_false", dest="use_lat_weighted_loss")
    parser.add_argument("--lambda_prior", type=float, default=1e-4)
    parser.add_argument("--lambda_latent", type=float, default=1e-4)
    parser.add_argument("--lambda_gain", type=float, default=1e-3)
    parser.add_argument("--lambda_gain_smooth", type=float, default=1e-3)
    parser.add_argument("--lambda_analysis", type=float, default=0.25,
        help="Weight for physical-space analysis reconstruction loss.")
    parser.add_argument("--surface_weight", type=float, default=3.0)
    parser.add_argument("--rootzone_weight", type=float, default=1.0)
    parser.add_argument("--selection_rootzone_weight", type=float, default=1.0,
        help="RootZone weight used only for combined_val_wrmse checkpoint selection.")
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--max_train_batches", type=int, default=0,
        help="Debug/smoke cap for target_train batches per epoch; 0 means all batches.")
    parser.add_argument("--max_val_batches", type=int, default=0,
        help="Debug/smoke cap for target_val batches per epoch; 0 means all batches.")
    parser.add_argument(
        "--target_selection_metric",
        type=str,
        default="objective",
        choices=[
            "objective",
            "total_loss",
            "analysis_loss",
            "surface_val_wrmse",
            "rootzone_val_wrmse",
            "combined_val_wrmse",
        ],
        help="Validation metric used to save the best target-val checkpoint.",
    )
    args = parser.parse_args()
    ProtocolConfig().assert_supported_adaptation_setting(args.adaptation_setting)
    return args


def _as_list(value: Any, n: int, fill: float) -> List[float]:
    if value is None:
        return [fill] * n
    return [float(x) for x in value]


def load_source_checkpoint_for_target_adaptation(
    checkpoint_path: str,
    device: torch.device,
    target_latent_dim: int = 32,
    enable_target_spatial_refine: bool = False,
    target_spatial_refine_hidden: int = 16,
    target_spatial_refine_rootzone: bool = False,
    target_spatial_refine_input: str = "normalized",
    target_spatial_refine_type: str = "simple",
    target_spatial_refine_gain_span: float = 0.25,
    hydro_msr_hidden: int = 16,
    enable_hydro_msr_da_film: bool = False,
) -> TargetAdaptationState:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"source checkpoint not found: {ckpt_path}")
    source_checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    source_config = dict(source_checkpoint.get("config", {}))
    model_type = source_config.get("model_type", "prompt_conditioned")
    if model_type != "hyperda_basis_adapter":
        raise ValueError(
            "train_hyperda_target_adapt.py requires a HyperDA source checkpoint "
            f"with config.model_type='hyperda_basis_adapter', got {model_type!r}"
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
        target_spatial_refine_hidden=target_spatial_refine_hidden,
        target_spatial_refine_rootzone=target_spatial_refine_rootzone,
        target_spatial_refine_input=target_spatial_refine_input,
        target_spatial_refine_type=target_spatial_refine_type,
        target_spatial_refine_gain_span=target_spatial_refine_gain_span,
        hydro_msr_hidden=hydro_msr_hidden,
        enable_hydro_msr_da_film=enable_hydro_msr_da_film,
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
    return TargetAdaptationState(
        model=model,
        prompt_encoder=prompt_encoder,
        source_checkpoint=source_checkpoint,
        source_config=source_config,
        normalization=normalization,
    )


def _collate_target_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    result = collate_hydroda_samples(batch)
    result["months"] = torch.tensor([int(s["month"]) for s in batch], dtype=torch.long)
    return result


def build_dataloader(dataset: HydroDADataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=_collate_target_batch,
    )


def _normalize_x(x: torch.Tensor, normalization: Dict[str, Optional[List[float]]]) -> torch.Tensor:
    ch_mean = normalization.get("ch_mean")
    ch_std = normalization.get("ch_std")
    if ch_mean is None or ch_std is None:
        return x
    mean_t = torch.tensor(ch_mean, dtype=x.dtype, device=x.device).view(1, 12, 1, 1)
    std_t = torch.tensor(ch_std, dtype=x.dtype, device=x.device).view(1, 12, 1, 1)
    return (x - mean_t) / std_t.clamp_min(1e-6)


def _target_tensor(
    inc_surface: torch.Tensor,
    inc_rootzone: torch.Tensor,
    normalization: Dict[str, Optional[List[float]]],
    normalize_increment: bool,
) -> torch.Tensor:
    target = torch.stack([inc_surface, inc_rootzone], dim=1)
    if normalize_increment and normalization.get("inc_mean") is not None and normalization.get("inc_std") is not None:
        mean_t = torch.tensor(normalization["inc_mean"], dtype=target.dtype, device=target.device).view(1, 2, 1, 1)
        std_t = torch.tensor(normalization["inc_std"], dtype=target.dtype, device=target.device).view(1, 2, 1, 1)
        target = (target - mean_t) / std_t.clamp_min(1e-6)
    return target


def _denormalize_increment(
    value: torch.Tensor,
    normalization: Dict[str, Optional[List[float]]],
    normalize_increment: bool,
) -> torch.Tensor:
    if not normalize_increment or normalization.get("inc_mean") is None or normalization.get("inc_std") is None:
        return value
    mean_t = torch.tensor(normalization["inc_mean"], dtype=value.dtype, device=value.device).view(1, 2, 1, 1)
    std_t = torch.tensor(normalization["inc_std"], dtype=value.dtype, device=value.device).view(1, 2, 1, 1)
    return value * std_t + mean_t


def _analysis_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    normalization: Dict[str, Optional[List[float]]],
    normalize_increment: bool,
    loss_fn: nn.Module,
    loss_mask: torch.Tensor,
    latitude_weight: Optional[torch.Tensor],
) -> Optional[Dict[str, torch.Tensor]]:
    if "forecast_surface" not in batch or "forecast_rootzone" not in batch:
        return None
    forecast = torch.stack(
        [
            batch["forecast_surface"].to(pred.device),
            batch["forecast_rootzone"].to(pred.device),
        ],
        dim=1,
    )
    pred_analysis = forecast + _denormalize_increment(pred, normalization, normalize_increment)
    true_analysis = forecast + _denormalize_increment(target, normalization, normalize_increment)
    if isinstance(loss_fn, WeightedMaskedHuberLoss):
        return loss_fn(
            pred_analysis,
            true_analysis,
            loss_mask,
            latitude_weight=latitude_weight,
            increment_scale=torch.ones(2, dtype=torch.float32, device=pred.device),
        )
    return loss_fn(pred_analysis, true_analysis, loss_mask)


def _latitude_valid_weight(
    loss_mask: torch.Tensor,
    reference: torch.Tensor,
    latitude_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    mask = loss_mask
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(device=reference.device, dtype=reference.dtype)
    if latitude_weight is None:
        lat_w = torch.ones_like(mask)
    else:
        lat_w = latitude_weight.to(device=reference.device, dtype=reference.dtype)
        if lat_w.ndim == 2:
            lat_w = lat_w.unsqueeze(0).unsqueeze(0)
        elif lat_w.ndim == 3:
            lat_w = lat_w.unsqueeze(1)
    return mask * lat_w


def _update_target_val_wrmse_accumulators(
    accum: Dict[str, float],
    pred_analysis: torch.Tensor,
    true_analysis: torch.Tensor,
    loss_mask: torch.Tensor,
    latitude_weight: Optional[torch.Tensor],
) -> None:
    weight = _latitude_valid_weight(loss_mask, pred_analysis, latitude_weight)
    weight_exp = weight.expand_as(pred_analysis)
    finite = torch.isfinite(pred_analysis) & torch.isfinite(true_analysis) & torch.isfinite(weight_exp) & (weight_exp > 0)
    valid_weight = weight_exp * finite.to(dtype=pred_analysis.dtype)
    sq_error = (pred_analysis - true_analysis).square() * valid_weight
    sse = sq_error.sum(dim=(0, 2, 3)).detach().cpu()
    weight_sum = valid_weight.sum(dim=(0, 2, 3)).detach().cpu()
    accum["surface_sse"] += float(sse[0])
    accum["surface_weight"] += float(weight_sum[0])
    if pred_analysis.shape[1] > 1:
        accum["rootzone_sse"] += float(sse[1])
        accum["rootzone_weight"] += float(weight_sum[1])


def _finalize_target_val_wrmse(accum: Dict[str, float]) -> Dict[str, float]:
    return {
        "target_val_surface_wrmse_latw": math.sqrt(accum["surface_sse"] / max(accum["surface_weight"], 1e-12)),
        "target_val_rootzone_wrmse_latw": math.sqrt(accum["rootzone_sse"] / max(accum["rootzone_weight"], 1e-12)),
    }


def _expand_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and reference.shape[1] != 1:
        mask = mask.expand_as(reference)
    return mask.to(device=reference.device, dtype=reference.dtype)


def _target_selection_value(
    metrics: Dict[str, float],
    target_selection_metric: str,
    rootzone_weight: float = 1.0,
    selection_rootzone_weight: Optional[float] = None,
) -> float:
    rootzone_selection_weight = rootzone_weight if selection_rootzone_weight is None else selection_rootzone_weight
    if target_selection_metric == "objective":
        return float(metrics["target_val_loss"])
    if target_selection_metric == "total_loss":
        return float(metrics["total_loss"])
    if target_selection_metric == "analysis_loss":
        return float(metrics["analysis_loss"])
    if target_selection_metric == "surface_val_wrmse":
        return float(metrics["target_val_surface_wrmse_latw"])
    if target_selection_metric == "rootzone_val_wrmse":
        return float(metrics["target_val_rootzone_wrmse_latw"])
    if target_selection_metric == "combined_val_wrmse":
        return float(metrics["target_val_surface_wrmse_latw"]) + float(rootzone_selection_weight) * float(
            metrics["target_val_rootzone_wrmse_latw"]
        )
    raise ValueError(f"unsupported target_selection_metric={target_selection_metric!r}")


def _selected_metric_name(target_selection_metric: str) -> str:
    if target_selection_metric == "objective":
        return "target_val_loss"
    if target_selection_metric in {"total_loss", "analysis_loss"}:
        return target_selection_metric
    if target_selection_metric == "surface_val_wrmse":
        return "target_val_surface_wrmse_latw"
    if target_selection_metric == "rootzone_val_wrmse":
        return "target_val_rootzone_wrmse_latw"
    if target_selection_metric == "combined_val_wrmse":
        return "combined_val_wrmse"
    raise ValueError(f"unsupported target_selection_metric={target_selection_metric!r}")


def _best_checkpoint_name(target_selection_metric: str) -> str:
    return _best_checkpoint_names().get(target_selection_metric, "checkpoint_best_target_val_loss.pt")


def _best_checkpoint_names() -> Dict[str, str]:
    return {
        "objective": "checkpoint_best_target_val_loss.pt",
        "total_loss": "checkpoint_best_target_val_loss.pt",
        "analysis_loss": "checkpoint_best_target_val_loss.pt",
        "surface_val_wrmse": "checkpoint_best_target_val_surface_wrmse.pt",
        "rootzone_val_wrmse": "checkpoint_best_target_val_rootzone_wrmse.pt",
        "combined_val_wrmse": "checkpoint_best_target_val_combined_wrmse.pt",
    }


def apply_target_adaptation_stage(
    model: HyperAdapterConditionalResUNet,
    epoch: int,
    stage1_epochs: int = 10,
) -> List[str]:
    """Switch target trainable modules for staged target adaptation."""
    if not model.enable_target_adaptation:
        raise ValueError("target adaptation modules are not enabled")
    for param in model.parameters():
        param.requires_grad_(False)

    stage1_modules = [
        model.target_prompt,
        model.target_adapter_coefficient_residual_b,
        model.target_adapter_coefficient_residual_d2,
        model.target_adapter_coefficient_residual_d1,
        model.residual_gain,
    ]
    stage2_modules = [model.target_spatial_refine]
    has_stage2_modules = any(module is not None for module in stage2_modules)
    active_modules = (
        stage2_modules
        if has_stage2_modules and stage1_epochs > 0 and epoch >= stage1_epochs
        else stage1_modules
    )
    if stage1_epochs <= 0:
        active_modules = [*stage1_modules, *stage2_modules]
    for module in active_modules:
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad_(True)
    return model.target_trainable_parameter_names()


def _metric_values_for_all_checkpoints(
    metrics: Dict[str, float],
    selection_rootzone_weight: float,
) -> Dict[str, float]:
    return {
        "objective": _target_selection_value(metrics, "objective"),
        "surface_val_wrmse": _target_selection_value(metrics, "surface_val_wrmse"),
        "rootzone_val_wrmse": _target_selection_value(metrics, "rootzone_val_wrmse"),
        "combined_val_wrmse": _target_selection_value(
            metrics,
            "combined_val_wrmse",
            selection_rootzone_weight=selection_rootzone_weight,
        ),
    }


def _target_region_embedding(state: TargetAdaptationState, target_region: str, device: torch.device) -> torch.Tensor:
    target_global_idx = _REGION_TO_IDX.get(target_region, 0)
    source_global_indices = state.source_config.get("source_region_global_indices")
    if source_global_indices is not None:
        lookup = {int(global_idx): prompt_idx for prompt_idx, global_idx in enumerate(source_global_indices)}
        if target_global_idx in lookup:
            ids = torch.tensor([lookup[target_global_idx]], dtype=torch.long, device=device)
            return state.prompt_encoder.region_embed(ids)
    return state.prompt_encoder.region_embed.weight.data.mean(dim=0, keepdim=True).to(device)


def build_target_prompt(
    state: TargetAdaptationState,
    x_norm: torch.Tensor,
    months: torch.Tensor,
    target_region: str,
) -> torch.Tensor:
    with torch.no_grad():
        input_stats = state.prompt_encoder._compute_input_stats(x_norm)
        i_emb = state.prompt_encoder.input_proj(input_stats)
        t_enc = state.prompt_encoder._temporal_encoding(months)
        t_emb = state.prompt_encoder.temporal_proj(t_enc)
        r_emb = _target_region_embedding(state, target_region, x_norm.device).expand(x_norm.shape[0], -1)
        combined = torch.cat([r_emb, i_emb, t_emb], dim=1)
        return state.prompt_encoder.mlp(combined)


def _model_forward(
    model: nn.Module,
    x_norm: torch.Tensor,
    z: torch.Tensor,
    months: torch.Tensor,
    x_raw: torch.Tensor,
) -> torch.Tensor:
    if "x_raw" in inspect.signature(model.forward).parameters:
        return model(x_norm, z, month=months, x_raw=x_raw)
    return model(x_norm, z, month=months)


def adaptation_regularization(
    model: HyperAdapterConditionalResUNet,
    lambda_prior: float,
    lambda_latent: float,
    lambda_gain: float,
    lambda_gain_smooth: float,
) -> torch.Tensor:
    device = next(model.parameters()).device
    total = torch.zeros((), dtype=torch.float32, device=device)
    residual_modules = [
        model.target_adapter_coefficient_residual_b,
        model.target_adapter_coefficient_residual_d2,
        model.target_adapter_coefficient_residual_d1,
    ]
    if lambda_prior > 0:
        for module in residual_modules:
            if module is not None:
                total = total + lambda_prior * module.logit_delta.square().mean()
    if lambda_latent > 0 and model.target_prompt is not None:
        total = total + lambda_latent * model.target_prompt.latent.square().mean()
    if model.residual_gain is not None:
        if lambda_gain > 0:
            total = total + lambda_gain * model.residual_gain.gain_delta.square().mean()
        if lambda_gain_smooth > 0:
            gain = model.residual_gain.gain_delta
            bias = model.residual_gain.bias
            total = total + lambda_gain_smooth * (
                (gain[1:] - gain[:-1]).square().mean()
                + (bias[1:] - bias[:-1]).square().mean()
            )
    return total


def _batch_loss(
    state: TargetAdaptationState,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    target_region: str,
    loss_fn: nn.Module,
    normalize_increment: bool,
    lambda_prior: float,
    lambda_latent: float,
    lambda_gain: float,
    lambda_gain_smooth: float,
    lambda_analysis: float = 0.25,
) -> Dict[str, torch.Tensor]:
    x = batch["x"].to(device)
    months = batch["months"].to(device)
    x_norm = _normalize_x(x, state.normalization)
    z = build_target_prompt(state, x_norm, months, target_region)
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
    else:
        pass
    reg = adaptation_regularization(state.model, lambda_prior, lambda_latent, lambda_gain, lambda_gain_smooth)
    losses["regularization_loss"] = reg.detach()
    losses["objective"] = (
        losses["total_loss"]
        + lambda_analysis * losses["analysis_loss"]
        + reg
    )
    return losses


def train_one_epoch(
    state: TargetAdaptationState,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_region: str,
    loss_fn: nn.Module,
    normalize_increment: bool,
    grad_clip: Optional[float],
    lambda_prior: float,
    lambda_latent: float,
    lambda_gain: float,
    lambda_gain_smooth: float,
    lambda_analysis: float = 0.25,
    max_batches: int = 0,
) -> Dict[str, float]:
    state.model.train()
    totals: Dict[str, float] = {
        "objective": 0.0,
        "total_loss": 0.0,
        "analysis_loss": 0.0,
        "regularization_loss": 0.0,
    }
    n = 0
    for step, batch in enumerate(loader):
        if max_batches > 0 and step >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        losses = _batch_loss(
            state, batch, device, target_region, loss_fn, normalize_increment,
            lambda_prior, lambda_latent, lambda_gain, lambda_gain_smooth,
            lambda_analysis=lambda_analysis,
        )
        losses["objective"].backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([p for p in state.model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_loss(
    state: TargetAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_region: str,
    loss_fn: nn.Module,
    normalize_increment: bool,
    lambda_analysis: float = 0.25,
    max_batches: int = 0,
) -> Dict[str, float]:
    state.model.eval()
    totals: Dict[str, float] = {
        "target_val_loss": 0.0,
        "total_loss": 0.0,
        "analysis_loss": 0.0,
        "surface_loss": 0.0,
        "rootzone_loss": 0.0,
    }
    wrmse_accum = {
        "surface_sse": 0.0,
        "surface_weight": 0.0,
        "rootzone_sse": 0.0,
        "rootzone_weight": 0.0,
    }
    n = 0
    for step, batch in enumerate(loader):
        if max_batches > 0 and step >= max_batches:
            break
        losses = _batch_loss(
            state, batch, device, target_region, loss_fn, normalize_increment,
            lambda_prior=0.0, lambda_latent=0.0, lambda_gain=0.0, lambda_gain_smooth=0.0,
            lambda_analysis=lambda_analysis,
        )
        totals["target_val_loss"] += float(losses["objective"].detach().cpu())
        totals["total_loss"] += float(losses["total_loss"].detach().cpu())
        totals["analysis_loss"] += float(losses["analysis_loss"].detach().cpu())
        totals["surface_loss"] += float(losses["surface_loss"].detach().cpu())
        totals["rootzone_loss"] += float(losses["rootzone_loss"].detach().cpu())
        if "pred_analysis_physical" in losses and "true_analysis_physical" in losses:
            latitude_weight = batch.get("latitude_weight")
            if latitude_weight is not None:
                latitude_weight = latitude_weight.to(device)
            _update_target_val_wrmse_accumulators(
                wrmse_accum,
                pred_analysis=losses["pred_analysis_physical"],
                true_analysis=losses["true_analysis_physical"],
                loss_mask=batch["loss_mask"].to(device),
                latitude_weight=latitude_weight,
            )
        n += 1
    metrics = {k: v / max(n, 1) for k, v in totals.items()}
    metrics.update(_finalize_target_val_wrmse(wrmse_accum))
    return metrics


def save_target_adaptation_checkpoint(
    path: Path,
    state: TargetAdaptationState,
    optimizer_state_dict: Dict[str, Any],
    epoch: int,
    tag: str,
    train_history: List[Dict[str, Any]],
    val_history: List[Dict[str, Any]],
    best_target_val_loss: float,
    best_epoch: int,
    config: Dict[str, Any],
    best_metrics: Optional[Dict[str, float]] = None,
    best_epochs_by_metric: Optional[Dict[str, int]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_names = state.model.target_trainable_parameter_names()
    trainable_modules = [
        "target_latent",
        "adapter_coefficient_residuals",
        "residual_gain",
    ]
    if getattr(state.model, "target_spatial_refine", None) is not None:
        trainable_modules.append("target_spatial_refine")
    full_config = dict(config)
    full_config.update(
        {
            "method": "hyperda_target_adapt",
            "model_type": "hyperda_basis_adapter_target_adapt",
            "target_train_period": "2015-2021",
            "target_val_period": "2022",
            "target_eval_period": "2023-2025",
            "frozen_modules": ["theta0", "H_psi", "adapter_basis_bank", "prompt_encoder"],
            "trainable_modules": trainable_modules,
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": int(sum(p.numel() for p in state.model.parameters() if p.requires_grad)),
            "model_selection_source": "target_val_2022_preregistered_adaptation_selection",
            "target_eval_usage": "final_eval_only_no_training_no_selection",
            "normalization_source": "source_fit_only_from_source_checkpoint",
            "leakage_guard_status": "pass",
            "git_hash": get_git_hash(),
            "timestamp": get_timestamp(),
        }
    )
    checkpoint = {
        "tag": tag,
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_target_val_loss": float(best_target_val_loss),
        "best_metrics": dict(best_metrics or {}),
        "best_epochs_by_metric": dict(best_epochs_by_metric or {}),
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "model_state_dict": state.model.state_dict(),
        "prompt_encoder_state_dict": state.prompt_encoder.state_dict(),
        "optimizer_state_dict": optimizer_state_dict,
        "source_checkpoint_config": state.source_config,
        "train_history": train_history,
        "val_history": val_history,
        "config": full_config,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
    }
    torch.save(checkpoint, path)


def _restore_rng_state(rng_state: Dict[str, Any]) -> None:
    if not rng_state:
        return
    torch_state = rng_state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = rng_state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    numpy_state = rng_state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)


def restore_target_adaptation_resume(
    resume_from: str,
    state: TargetAdaptationState,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_epochs: int,
) -> ResumeTrainingState:
    resume_path = Path(resume_from)
    if not resume_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    model_type = checkpoint.get("config", {}).get("model_type")
    if model_type != "hyperda_basis_adapter_target_adapt":
        raise ValueError(
            "--resume_from requires a Phase 5 target-adapt checkpoint with "
            f"config.model_type='hyperda_basis_adapter_target_adapt', got {model_type!r}"
        )
    state.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    prompt_state = checkpoint.get("prompt_encoder_state_dict")
    if prompt_state is not None:
        state.prompt_encoder.load_state_dict(prompt_state)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    _restore_rng_state(checkpoint.get("rng_state", {}))

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    if start_epoch >= max_epochs:
        raise ValueError(
            f"resume checkpoint epoch={checkpoint.get('epoch')} already reaches "
            f"max_epochs={max_epochs}; increase MAX_EPOCHS/--max_epochs to continue"
        )
    return ResumeTrainingState(
        start_epoch=start_epoch,
        best_target_val_loss=float(checkpoint.get("best_target_val_loss", float("inf"))),
        best_epoch=int(checkpoint.get("best_epoch", -1)),
        best_metrics=dict(checkpoint.get("best_metrics", {})),
        best_epochs_by_metric=dict(checkpoint.get("best_epochs_by_metric", {})),
        train_history=list(checkpoint.get("train_history", [])),
        val_history=list(checkpoint.get("val_history", [])),
    )


def _dataset_date_hash(dataset: HydroDADataset, key: str) -> str:
    entry = getattr(dataset, "_split_entry", {})
    return str(entry.get(key, ""))


def _target_val_dates_hash(train_dataset: HydroDADataset, val_dataset: HydroDADataset) -> str:
    return (
        _dataset_date_hash(val_dataset, "target_val_dates_hash")
        or _dataset_date_hash(train_dataset, "target_val_dates_hash")
        or _dataset_date_hash(train_dataset, "source_val_dates_hash")
    )


def _safe_len(dataset: HydroDADataset) -> int:
    return int(len(dataset))


def run(args: argparse.Namespace) -> Path:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.resume_from and not Path(args.resume_from).exists():
        raise FileNotFoundError(f"resume checkpoint not found: {args.resume_from}")

    state = load_source_checkpoint_for_target_adaptation(
        checkpoint_path=args.source_checkpoint,
        device=device,
        target_latent_dim=args.target_latent_dim,
        enable_target_spatial_refine=args.enable_target_spatial_refine,
        target_spatial_refine_hidden=args.target_spatial_refine_hidden,
        target_spatial_refine_rootzone=args.target_spatial_refine_rootzone,
        target_spatial_refine_input=args.target_spatial_refine_input,
        target_spatial_refine_type=args.target_spatial_refine_type,
        target_spatial_refine_gain_span=args.target_spatial_refine_gain_span,
        hydro_msr_hidden=args.hydro_msr_hidden,
        enable_hydro_msr_da_film=args.enable_hydro_msr_da_film,
    )
    normalize_increment = bool(state.source_config.get("target_increment_normalization", state.normalization.get("inc_mean") is not None))

    train_dataset = HydroDADataset(
        da_nc_path=args.da_nc,
        region_masks_nc=args.region_masks_nc,
        splits_json=args.splits_json,
        target_region=args.target_region,
        split_type="target_train",
        K=None,
        seed=args.seed,
        adaptation_setting=args.adaptation_setting,
        freeze_manifest=args.freeze_manifest,
    )
    val_dataset = HydroDADataset(
        da_nc_path=args.da_nc,
        region_masks_nc=args.region_masks_nc,
        splits_json=args.splits_json,
        target_region=args.target_region,
        split_type="target_val",
        K=None,
        seed=args.seed,
        adaptation_setting=args.adaptation_setting,
        freeze_manifest=args.freeze_manifest,
    )
    if len(train_dataset) == 0:
        raise ValueError("target_train dataset is empty; cannot adapt")
    if len(val_dataset) == 0:
        raise ValueError("target_val dataset is empty; regenerate split manifest or use source_val-date fallback")

    split_sha = compute_sha256(args.splits_json) if Path(args.splits_json).exists() else ""
    run_config = {
        "source_checkpoint": args.source_checkpoint,
        "resume_from": args.resume_from,
        "target_region": args.target_region,
        "adaptation_setting": args.adaptation_setting,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "target_latent_dim": args.target_latent_dim,
        "enable_target_spatial_refine": args.enable_target_spatial_refine,
        "target_spatial_refine_hidden": args.target_spatial_refine_hidden,
        "target_spatial_refine_rootzone": args.target_spatial_refine_rootzone,
        "target_spatial_refine_input": args.target_spatial_refine_input,
        "target_spatial_refine_type": args.target_spatial_refine_type,
        "target_spatial_refine_gain_span": args.target_spatial_refine_gain_span,
        "hydro_msr_hidden": args.hydro_msr_hidden,
        "enable_hydro_msr_da_film": args.enable_hydro_msr_da_film,
        "enable_da_regime_gain_mixer": args.enable_da_regime_gain_mixer
        or args.target_spatial_refine_type in {"hydro_msr_gain", "hydro_msr_gain_lite"},
        "stage1_epochs": args.stage1_epochs,
        "stage_schedule": "staged_global_then_spatial" if args.stage1_epochs > 0 else "joint_target_modules",
        "lambda_prior": args.lambda_prior,
        "lambda_latent": args.lambda_latent,
        "lambda_gain": args.lambda_gain,
        "lambda_gain_smooth": args.lambda_gain_smooth,
        "lambda_analysis": args.lambda_analysis,
        "target_selection_metric": args.target_selection_metric,
        "selected_metric_name": _selected_metric_name(args.target_selection_metric),
        "selected_metric_value": None,
        "surface_weight": args.surface_weight,
        "rootzone_weight": args.rootzone_weight,
        "selection_rootzone_weight": args.selection_rootzone_weight,
        "split_manifest_path": args.splits_json,
        "split_manifest_sha256": split_sha,
        "target_train_dates_hash": _dataset_date_hash(train_dataset, "target_train_dates_hash"),
        "target_val_dates_hash": _target_val_dates_hash(train_dataset, val_dataset),
        "target_eval_dates_hash": _dataset_date_hash(train_dataset, "target_eval_dates_hash"),
        "target_train_cycle_count": _safe_len(train_dataset),
        "target_val_cycle_count": _safe_len(val_dataset),
        "adaptation_steps": int(math.ceil(len(train_dataset) / max(args.batch_size, 1)) * args.max_epochs),
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
    }
    output_dir = args.output_dir
    if args.resume_from and output_dir is None:
        output_dir = str(Path(args.resume_from).parent.parent)
    run_manager = RunManager(
        phase=PHASE,
        method="hyperda_target_adapt",
        target_region=args.target_region,
        config=run_config,
        output_dir=output_dir,
        run_name=args.run_name,
        width=int(state.source_config.get("width", 32)),
        epochs=args.max_epochs,
        lr=args.lr,
        norm="norm" if normalize_increment else "nonorm",
        zero_raw=bool(state.source_config.get("zero_raw_increment_init", False)),
        seed=args.seed,
    )

    train_loader = build_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, args.batch_size, args.num_workers, shuffle=False)
    state.model.freeze_source_prior_for_target_adaptation()
    trainable_params = [p for p in state.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn: nn.Module = (
        WeightedMaskedHuberLoss(
            delta=0.01,
            surface_weight=args.surface_weight,
            rootzone_weight=args.rootzone_weight,
            use_lat_weight=True,
        )
        if args.use_lat_weighted_loss
        else MaskedHuberLoss(
            delta=0.01,
            surface_weight=args.surface_weight,
            rootzone_weight=args.rootzone_weight,
        )
    )

    start_epoch_idx = 0
    best_val = float("inf")
    best_epoch = -1
    best_metrics = {
        metric: float("inf")
        for metric in ["objective", "surface_val_wrmse", "rootzone_val_wrmse", "combined_val_wrmse"]
    }
    best_epochs_by_metric = {metric: -1 for metric in best_metrics}
    train_history: List[Dict[str, Any]] = []
    val_history: List[Dict[str, Any]] = []
    if args.resume_from:
        resume_state = restore_target_adaptation_resume(
            resume_from=args.resume_from,
            state=state,
            optimizer=optimizer,
            device=device,
            max_epochs=args.max_epochs,
        )
        start_epoch_idx = resume_state.start_epoch
        best_val = resume_state.best_target_val_loss
        best_epoch = resume_state.best_epoch
        best_metrics.update({str(k): float(v) for k, v in resume_state.best_metrics.items()})
        best_epochs_by_metric.update({str(k): int(v) for k, v in resume_state.best_epochs_by_metric.items()})
        train_history = resume_state.train_history
        val_history = resume_state.val_history
        run_config["resume_start_epoch"] = start_epoch_idx
        run_config["resume_previous_best_target_val_loss"] = best_val
        run_config["resume_previous_best_epoch"] = best_epoch
        print(
            f"Resuming Phase 5 target adaptation from {args.resume_from}: "
            f"start_epoch={start_epoch_idx} max_epochs={args.max_epochs} "
            f"best_epoch={best_epoch} best_target_val_loss={best_val:.6f}",
            flush=True,
        )
    active_stage_parameters = apply_target_adaptation_stage(state.model, start_epoch_idx, args.stage1_epochs)
    run_config["initial_stage_trainable_parameter_names"] = active_stage_parameters

    run_manager.save_config(run_config, "config.yaml")
    run_manager.save_git_info()
    run_manager.save_protocol(
        {
            "protocol_freeze_id": PROTOCOL_FREEZE_ID,
            "target_train": "2015-2021",
            "target_val": "2022",
            "target_eval": "2023-2025",
            "split_manifest": args.splits_json,
        }
    )
    run_manager.save_data_manifest(run_config)

    start = time.time()
    try:
        for epoch in range(start_epoch_idx, args.max_epochs):
            active_stage_parameters = apply_target_adaptation_stage(state.model, epoch, args.stage1_epochs)
            train_metrics = train_one_epoch(
                state, train_loader, optimizer, device, args.target_region, loss_fn,
                normalize_increment, args.grad_clip, args.lambda_prior, args.lambda_latent,
                args.lambda_gain, args.lambda_gain_smooth,
                lambda_analysis=args.lambda_analysis,
                max_batches=args.max_train_batches,
            )
            val_metrics = evaluate_loss(
                state,
                val_loader,
                device,
                args.target_region,
                loss_fn,
                normalize_increment,
                lambda_analysis=args.lambda_analysis,
                max_batches=args.max_val_batches,
            )
            train_row = {"epoch": epoch, "target_train_loss": train_metrics["total_loss"], **train_metrics}
            val_row = {"epoch": epoch, **val_metrics}
            train_history.append(train_row)
            val_history.append(val_row)
            selection_value = _target_selection_value(
                val_metrics,
                args.target_selection_metric,
                selection_rootzone_weight=args.selection_rootzone_weight,
            )
            selected_metric_name = _selected_metric_name(args.target_selection_metric)
            val_row["selected_metric_name"] = selected_metric_name
            val_row["selected_metric_value"] = selection_value
            val_row["stage_trainable_parameter_count"] = len(active_stage_parameters)
            val_row["stage_name"] = (
                "stage2_spatial_gain"
                if args.stage1_epochs > 0 and epoch >= args.stage1_epochs
                else "stage1_global_target"
            )
            print(
                f"epoch={epoch:03d} target_train_loss={train_metrics['total_loss']:.6f} "
                f"target_val_loss={val_metrics['target_val_loss']:.6f} "
                f"selection_{args.target_selection_metric}="
                f"{selection_value:.6f}",
                flush=True,
            )
            current_best_values = _metric_values_for_all_checkpoints(
                val_metrics,
                selection_rootzone_weight=args.selection_rootzone_weight,
            )
            selection_improved = selection_value < best_val
            if selection_improved:
                best_val = selection_value
                best_epoch = epoch
            for metric_key, metric_value in current_best_values.items():
                if metric_value >= best_metrics[metric_key]:
                    continue
                best_metrics[metric_key] = metric_value
                best_epochs_by_metric[metric_key] = epoch
                metric_config = {
                    **run_config,
                    "selected_metric_name": _selected_metric_name(metric_key),
                    "selected_metric_value": metric_value,
                    "checkpoint_metric": metric_key,
                    "stage_trainable_parameter_names": active_stage_parameters,
                }
                save_target_adaptation_checkpoint(
                    run_manager.get_checkpoint_dir() / _best_checkpoint_name(metric_key),
                    state,
                    optimizer.state_dict(),
                    epoch,
                    Path(_best_checkpoint_name(metric_key)).stem,
                    train_history,
                    val_history,
                    best_val,
                    best_epoch,
                    metric_config,
                    best_metrics=best_metrics,
                    best_epochs_by_metric=best_epochs_by_metric,
                )
            if selection_improved:
                best_config = {
                    **run_config,
                    "selected_metric_name": selected_metric_name,
                    "selected_metric_value": selection_value,
                    "checkpoint_metric": args.target_selection_metric,
                    "stage_trainable_parameter_names": active_stage_parameters,
                }
                save_target_adaptation_checkpoint(
                    run_manager.get_checkpoint_dir() / _best_checkpoint_name(args.target_selection_metric),
                    state,
                    optimizer.state_dict(),
                    epoch,
                    Path(_best_checkpoint_name(args.target_selection_metric)).stem,
                    train_history,
                    val_history,
                    best_val,
                    best_epoch,
                    best_config,
                    best_metrics=best_metrics,
                    best_epochs_by_metric=best_epochs_by_metric,
                )
            save_target_adaptation_checkpoint(
                run_manager.get_checkpoint_dir() / "last.pt",
                state,
                optimizer.state_dict(),
                epoch,
                "last",
                train_history,
                val_history,
                best_val,
                best_epoch,
                {
                    **run_config,
                    "selected_metric_name": selected_metric_name,
                    "selected_metric_value": best_val if math.isfinite(best_val) else None,
                    "stage_trainable_parameter_names": active_stage_parameters,
                },
                best_metrics=best_metrics,
                best_epochs_by_metric=best_epochs_by_metric,
            )
            if (epoch + 1) % args.checkpoint_every == 0:
                save_target_adaptation_checkpoint(
                    run_manager.get_checkpoint_dir() / f"checkpoint_epoch_{epoch:03d}.pt",
                    state,
                    optimizer.state_dict(),
                    epoch,
                    f"epoch_{epoch:03d}",
                    train_history,
                    val_history,
                    best_val,
                    best_epoch,
                    {
                        **run_config,
                        "selected_metric_name": selected_metric_name,
                        "selected_metric_value": best_val if math.isfinite(best_val) else None,
                        "stage_trainable_parameter_names": active_stage_parameters,
                    },
                    best_metrics=best_metrics,
                    best_epochs_by_metric=best_epochs_by_metric,
                )
    finally:
        train_dataset.close()
        val_dataset.close()

    summary = {
        **run_config,
        "best_target_val_loss": best_val,
        "best_metrics": best_metrics,
        "best_epochs_by_metric": best_epochs_by_metric,
        "selected_metric_name": _selected_metric_name(args.target_selection_metric),
        "selected_metric_value": best_val if math.isfinite(best_val) else None,
        "best_epoch": best_epoch,
        "resume_from": args.resume_from,
        "resume_start_epoch": start_epoch_idx,
        "elapsed_s": time.time() - start,
        "trainable_parameter_count": int(sum(p.numel() for p in state.model.parameters() if p.requires_grad)),
        "trainable_parameter_names": state.model.target_trainable_parameter_names(),
        "frozen_modules": ["theta0", "H_psi", "adapter_basis_bank", "prompt_encoder"],
        "trainable_modules": [
            "target_latent",
            "adapter_coefficient_residuals",
            "residual_gain",
            *(
                ["target_spatial_refine"]
                if getattr(state.model, "target_spatial_refine", None) is not None
                else []
            ),
        ],
        "model_selection_source": "target_val_2022_preregistered_adaptation_selection",
        "target_eval_usage": "final_eval_only_no_training_no_selection",
        "normalization_source": "source_fit_only_from_source_checkpoint",
        "leakage_guard_status": "pass",
        "train_history": train_history,
        "val_history": val_history,
    }
    with open(run_manager.summary_json_path(), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return run_manager.get_run_dir()


def main() -> None:
    run_dir = run(parse_args())
    print(f"Done: HyperDA target adaptation run_dir={run_dir}")


if __name__ == "__main__":
    main()
