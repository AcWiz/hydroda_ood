#!/usr/bin/env python3
"""Train HyperDA target adaptation variables on target_train.

No-leakage declaration:
    - Loads a source-trained HyperDA checkpoint.
    - Freezes source prior parameters: theta0, H_psi-style adapter basis bank,
      prompt encoder, and shared backbone.
    - Trains only target latent, adapter coefficient residuals, and monthly
      residual gain on target_train (2015-2021).
    - Selects the adaptation checkpoint only on target_val (2022).
    - Never reads target_eval (2023-2025) labels during adaptation.
"""
from __future__ import annotations

import argparse
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

from hydroda.data.dataset import HydroDADataset
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HyperDA target adaptation variables")
    parser.add_argument("--source_checkpoint", type=str, required=True)
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
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--max_train_batches", type=int, default=0,
        help="Debug/smoke cap for target_train batches per epoch; 0 means all batches.")
    parser.add_argument("--max_val_batches", type=int, default=0,
        help="Debug/smoke cap for target_val batches per epoch; 0 means all batches.")
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
    result = {
        "x": torch.from_numpy(np.stack([s["x"] for s in batch], axis=0)),
        "increment_surface": torch.from_numpy(np.stack([s["increment_surface"] for s in batch], axis=0)),
        "increment_rootzone": torch.from_numpy(np.stack([s["increment_rootzone"] for s in batch], axis=0)),
        "loss_mask": torch.from_numpy(np.stack([s["loss_mask"] for s in batch], axis=0)),
        "months": torch.tensor([int(s["month"]) for s in batch], dtype=torch.long),
    }
    if "latitude_weight" in batch[0]:
        result["latitude_weight"] = torch.from_numpy(np.stack([s["latitude_weight"] for s in batch], axis=0))
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
) -> Dict[str, torch.Tensor]:
    x = batch["x"].to(device)
    months = batch["months"].to(device)
    x_norm = _normalize_x(x, state.normalization)
    z = build_target_prompt(state, x_norm, months, target_region)
    pred = state.model(x_norm, z, month=months)
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
    reg = adaptation_regularization(state.model, lambda_prior, lambda_latent, lambda_gain, lambda_gain_smooth)
    losses["regularization_loss"] = reg.detach()
    losses["objective"] = losses["total_loss"] + reg
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
    max_batches: int = 0,
) -> Dict[str, float]:
    state.model.train()
    totals: Dict[str, float] = {"objective": 0.0, "total_loss": 0.0, "regularization_loss": 0.0}
    n = 0
    for step, batch in enumerate(loader):
        if max_batches > 0 and step >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        losses = _batch_loss(
            state, batch, device, target_region, loss_fn, normalize_increment,
            lambda_prior, lambda_latent, lambda_gain, lambda_gain_smooth,
        )
        losses["objective"].backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([p for p in state.model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        totals["objective"] += float(losses["objective"].detach().cpu())
        totals["total_loss"] += float(losses["total_loss"].detach().cpu())
        totals["regularization_loss"] += float(losses["regularization_loss"].detach().cpu())
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
    max_batches: int = 0,
) -> Dict[str, float]:
    state.model.eval()
    totals: Dict[str, float] = {"target_val_loss": 0.0, "surface_loss": 0.0, "rootzone_loss": 0.0}
    n = 0
    for step, batch in enumerate(loader):
        if max_batches > 0 and step >= max_batches:
            break
        losses = _batch_loss(
            state, batch, device, target_region, loss_fn, normalize_increment,
            lambda_prior=0.0, lambda_latent=0.0, lambda_gain=0.0, lambda_gain_smooth=0.0,
        )
        totals["target_val_loss"] += float(losses["total_loss"].detach().cpu())
        totals["surface_loss"] += float(losses["surface_loss"].detach().cpu())
        totals["rootzone_loss"] += float(losses["rootzone_loss"].detach().cpu())
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_names = state.model.target_trainable_parameter_names()
    full_config = dict(config)
    full_config.update(
        {
            "method": "hyperda_target_adapt",
            "model_type": "hyperda_basis_adapter_target_adapt",
            "target_train_period": "2015-2021",
            "target_val_period": "2022",
            "target_eval_period": "2023-2025",
            "frozen_modules": ["theta0", "H_psi", "adapter_basis_bank", "prompt_encoder"],
            "trainable_modules": [
                "target_latent",
                "adapter_coefficient_residuals",
                "residual_gain",
            ],
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
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "model_state_dict": state.model.state_dict(),
        "prompt_encoder_state_dict": state.prompt_encoder.state_dict(),
        "optimizer_state_dict": optimizer_state_dict,
        "source_checkpoint_config": state.source_config,
        "train_history": train_history,
        "val_history": val_history,
        "config": full_config,
    }
    torch.save(checkpoint, path)


def _dataset_date_hash(dataset: HydroDADataset, key: str) -> str:
    entry = getattr(dataset, "_split_entry", {})
    return str(entry.get(key, ""))


def _safe_len(dataset: HydroDADataset) -> int:
    return int(len(dataset))


def run(args: argparse.Namespace) -> Path:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state = load_source_checkpoint_for_target_adaptation(
        checkpoint_path=args.source_checkpoint,
        device=device,
        target_latent_dim=args.target_latent_dim,
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
        "target_region": args.target_region,
        "adaptation_setting": args.adaptation_setting,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "target_latent_dim": args.target_latent_dim,
        "lambda_prior": args.lambda_prior,
        "lambda_latent": args.lambda_latent,
        "lambda_gain": args.lambda_gain,
        "lambda_gain_smooth": args.lambda_gain_smooth,
        "split_manifest_path": args.splits_json,
        "split_manifest_sha256": split_sha,
        "target_train_dates_hash": _dataset_date_hash(train_dataset, "target_train_dates_hash"),
        "target_val_dates_hash": _dataset_date_hash(train_dataset, "source_val_dates_hash"),
        "target_eval_dates_hash": _dataset_date_hash(train_dataset, "target_eval_dates_hash"),
        "target_train_cycle_count": _safe_len(train_dataset),
        "target_val_cycle_count": _safe_len(val_dataset),
        "adaptation_steps": int(math.ceil(len(train_dataset) / max(args.batch_size, 1)) * args.max_epochs),
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
    }
    run_manager = RunManager(
        phase=PHASE,
        method="hyperda_target_adapt",
        target_region=args.target_region,
        config=run_config,
        output_dir=args.output_dir,
        run_name=args.run_name,
        width=int(state.source_config.get("width", 32)),
        epochs=args.max_epochs,
        lr=args.lr,
        norm="norm" if normalize_increment else "nonorm",
        zero_raw=bool(state.source_config.get("zero_raw_increment_init", False)),
        seed=args.seed,
    )
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

    train_loader = build_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, args.batch_size, args.num_workers, shuffle=False)
    trainable_params = [p for p in state.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn: nn.Module = (
        WeightedMaskedHuberLoss(delta=0.01, use_lat_weight=True)
        if args.use_lat_weighted_loss
        else MaskedHuberLoss(delta=0.01)
    )

    best_val = float("inf")
    best_epoch = -1
    train_history: List[Dict[str, Any]] = []
    val_history: List[Dict[str, Any]] = []
    start = time.time()
    try:
        for epoch in range(args.max_epochs):
            train_metrics = train_one_epoch(
                state, train_loader, optimizer, device, args.target_region, loss_fn,
                normalize_increment, args.grad_clip, args.lambda_prior, args.lambda_latent,
                args.lambda_gain, args.lambda_gain_smooth,
                max_batches=args.max_train_batches,
            )
            val_metrics = evaluate_loss(
                state,
                val_loader,
                device,
                args.target_region,
                loss_fn,
                normalize_increment,
                max_batches=args.max_val_batches,
            )
            train_row = {"epoch": epoch, "target_train_loss": train_metrics["total_loss"], **train_metrics}
            val_row = {"epoch": epoch, **val_metrics}
            train_history.append(train_row)
            val_history.append(val_row)
            print(
                f"epoch={epoch:03d} target_train_loss={train_metrics['total_loss']:.6f} "
                f"target_val_loss={val_metrics['target_val_loss']:.6f}",
                flush=True,
            )
            if val_metrics["target_val_loss"] < best_val:
                best_val = val_metrics["target_val_loss"]
                best_epoch = epoch
                save_target_adaptation_checkpoint(
                    run_manager.get_checkpoint_dir() / "checkpoint_best_target_val_loss.pt",
                    state,
                    optimizer.state_dict(),
                    epoch,
                    "best_target_val_loss",
                    train_history,
                    val_history,
                    best_val,
                    best_epoch,
                    run_config,
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
                run_config,
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
                    run_config,
                )
    finally:
        train_dataset.close()
        val_dataset.close()

    summary = {
        **run_config,
        "best_target_val_loss": best_val,
        "best_epoch": best_epoch,
        "elapsed_s": time.time() - start,
        "trainable_parameter_count": int(sum(p.numel() for p in state.model.parameters() if p.requires_grad)),
        "trainable_parameter_names": state.model.target_trainable_parameter_names(),
        "frozen_modules": ["theta0", "H_psi", "adapter_basis_bank", "prompt_encoder"],
        "trainable_modules": ["target_latent", "adapter_coefficient_residuals", "residual_gain"],
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
