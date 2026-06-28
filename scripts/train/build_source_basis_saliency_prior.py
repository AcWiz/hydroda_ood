#!/usr/bin/env python3
"""Build source-side adapter layer x basis saliency prior for rank-gated HyperDA.

The artifact is computed only from source_fit/source-side pseudo episodes and
is intended as a gate-logit bias for the stable rank-gated DoRA path. It does
not read target_val or target_eval labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from hydroda.data.dataset import build_hydroda_dataset
from hydroda.data.file_hash import compute_sha256
from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.prompt_encoder import RegionPromptEncoder, RobustInputSideDAPromptEncoder
from hydroda.models.source_saliency import (
    ADAPTER_LAYER_NAMES,
    FORBIDDEN_SALIENCY_SOURCE_SPLITS,
    make_saliency_artifact,
    tensor_sha256,
)
from hydroda.utils.device import resolve_device

from scripts.train.train_prompt_conditioned_shared import (
    DA_NC,
    FREEZE_MANIFEST,
    REGION_MASKS_NC,
    SPLITS_JSON,
    PromptConditionedTrainer,
    _GLOBAL_REGION_IDX_MAP,
    _resolve_source_regions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build source-side HyperDA adapter basis saliency prior."
    )
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--target_region", required=True)
    parser.add_argument("--source_regions", default=None)
    parser.add_argument("--source_split", default="source_fit", choices=["source_fit", "source_episode", "source_side_episode"])
    parser.add_argument("--da_nc", default=DA_NC)
    parser.add_argument("--region_masks_nc", default=REGION_MASKS_NC)
    parser.add_argument("--splits_json", default=SPLITS_JSON)
    parser.add_argument("--freeze_manifest", default=FREEZE_MANIFEST)
    parser.add_argument("--adaptation_setting", default="zero_shot_context")
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_backend", default="netcdf", choices=["netcdf", "tensor_cache"])
    parser.add_argument("--tensor_cache_dir", default="artifacts/region_crops/US")
    parser.add_argument("--max_year_cache_entries", type=int, default=1)
    parser.add_argument("--tensor_cache_load_mode", default="eager", choices=["eager", "mmap"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require_gpu", action="store_true")
    parser.add_argument("--use_lat_weighted_loss", action="store_true", default=True)
    parser.add_argument("--no_lat_weighted_loss", action="store_false", dest="use_lat_weighted_loss")
    args = parser.parse_args()
    if args.source_split in FORBIDDEN_SALIENCY_SOURCE_SPLITS:
        parser.error("--source_split must be source-side only")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers must be non-negative")
    if args.max_batches < 0:
        parser.error("--max_batches must be non-negative")
    return args


def _prompt_encoder_from_config(config: Dict[str, Any]):
    kwargs = {
        "num_regions": int(config.get("num_regions", len(config.get("source_regions", [])) or 6)),
        "input_channels": 12,
        "hidden_dim": int(config.get("prompt_dim", 64)),
    }
    context_encoder = config.get("context_encoder", "current_mean_std")
    if context_encoder == "current_mean_std":
        return RegionPromptEncoder(**kwargs)
    if context_encoder == "robust_input_side_da_diagnostics":
        return RobustInputSideDAPromptEncoder(**kwargs)
    raise ValueError(f"Unsupported source checkpoint context_encoder: {context_encoder}")


def _model_from_config(config: Dict[str, Any]) -> HyperAdapterConditionalResUNet:
    if config.get("model_type") != "hyperda_basis_adapter":
        raise ValueError(
            "source saliency prior builder requires config.model_type='hyperda_basis_adapter'"
        )
    if config.get("hyper_coeff_generator") != "shared_layer_aware_rank_gated_stable":
        raise ValueError(
            "source saliency prior builder targets M2.1 stable rank-gated HyperDA; "
            f"got hyper_coeff_generator={config.get('hyper_coeff_generator')!r}"
        )
    return HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=int(config.get("width", 32)),
        prompt_dim=int(config.get("prompt_dim", 64)),
        hyper_n_basis=int(config.get("hyper_n_basis", 8)),
        hyper_adapter_bottleneck=config.get("hyper_adapter_bottleneck"),
        hyper_adapter_scale=float(config.get("hyper_adapter_scale", 1.0)),
        hyper_coeff_generator=config.get("hyper_coeff_generator", "shared_layer_aware_rank_gated_stable"),
        hyper_rank_gate_top_k=int(config.get("hyper_rank_gate_top_k", 4)),
        hyper_rank_gate_temperature_init=float(config.get("hyper_rank_gate_temperature_init", 2.0)),
        hyper_adapter_param_style=config.get("hyper_adapter_param_style", "dora_like_gain_bounded"),
        hyper_reliability_gate=config.get("hyper_reliability_gate", "prompt_scalar"),
        hyper_reliability_init=float(config.get("hyper_reliability_init", 0.95)),
        hyper_source_saliency_prior=config.get("hyper_source_saliency_prior"),
        hyper_source_saliency_prior_beta=0.0,
        hyper_source_saliency_prior_application=config.get(
            "hyper_source_saliency_prior_application",
            "soft_regularization_metadata",
        ),
        hyper_prompt_manifold_reliability=bool(config.get("hyper_prompt_manifold_reliability", False)),
        hyper_prompt_manifold_reliability_strength=float(
            config.get("hyper_prompt_manifold_reliability_strength", 0.0)
        ),
        hyper_enable_film=bool(config.get("hyper_enable_film", True)),
        hyper_enable_adapters=bool(config.get("hyper_enable_adapters", True)),
        zero_shot_prior_form=config.get("zero_shot_prior_form", "direct_hyper"),
        source_residual_rho=float(config.get("source_residual_rho", config.get("zero_shot_rho", 1.0))),
        source_residual_gate=config.get("source_residual_gate", "prompt_reliability_scalar"),
        source_residual_gate_init=float(config.get("source_residual_gate_init", 0.95)),
        source_residual_reliability_dim=int(config.get("source_residual_reliability_dim", 5)),
        zero_raw_increment_init=bool(config.get("zero_raw_increment_init", False)),
    )


def _basis_gain_modules(model: HyperAdapterConditionalResUNet) -> list[tuple[str, torch.nn.Parameter]]:
    modules = [
        ("bottleneck", model.hyper_adapter_b),
        ("dec2", model.hyper_adapter_d2),
        ("dec1", model.hyper_adapter_d1),
    ]
    params: list[tuple[str, torch.nn.Parameter]] = []
    for layer_name, module in modules:
        param = getattr(module, "basis_gain_delta", None)
        if param is None:
            raise ValueError(
                "source saliency prior builder requires dora_like_gain_bounded adapter bases"
            )
        params.append((layer_name, param))
    return params


def build_saliency_scores(trainer: PromptConditionedTrainer, *, max_batches: int) -> tuple[torch.Tensor, int]:
    model = trainer.model
    if not isinstance(model, HyperAdapterConditionalResUNet):
        raise ValueError("trainer.model must be HyperAdapterConditionalResUNet")
    gain_params = _basis_gain_modules(model)
    for param in model.parameters():
        param.requires_grad_(False)
    for _, param in gain_params:
        param.requires_grad_(True)

    model.train()
    trainer.prompt_encoder.eval()
    score_sum = torch.zeros(len(ADAPTER_LAYER_NAMES), trainer.hyper_n_basis, dtype=torch.float64)
    n_batches = 0
    loader = trainer._build_dataloader()
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        trainer.prompt_encoder.zero_grad(set_to_none=True)

        x = batch["x"].to(trainer.device)
        inc_surface = batch["increment_surface"].to(trainer.device)
        inc_rootzone = batch["increment_rootzone"].to(trainer.device)
        loss_mask = batch["loss_mask"].to(trainer.device)
        region_ids = batch["region_ids"].to(trainer.device)
        months = batch["months"].to(trainer.device)
        latitude_weight = batch.get("latitude_weight")
        if latitude_weight is not None:
            latitude_weight = latitude_weight.to(trainer.device)

        x_norm = trainer._normalize(x)
        if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
            continue
        target = torch.stack([inc_surface, inc_rootzone], dim=1)
        if trainer.target_increment_normalization and trainer._inc_mean is not None:
            inc_mean_t = torch.from_numpy(trainer._inc_mean).to(x.device).view(1, 2, 1, 1)
            inc_std_t = torch.from_numpy(trainer._inc_std).to(x.device).view(1, 2, 1, 1)
            target = (target - inc_mean_t) / inc_std_t

        _pred, losses = trainer._forward_and_loss(
            x_norm,
            target,
            loss_mask,
            region_ids,
            months,
            latitude_weight=latitude_weight,
        )
        loss = losses["total_loss"]
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        for layer_idx, (_layer_name, param) in enumerate(gain_params):
            if param.grad is None:
                continue
            if trainer.hyper_adapter_param_style == "dora_like_gain_bounded":
                gain = 1.0 + 0.25 * torch.tanh(param.detach())
            else:
                gain = 1.0 + param.detach()
            score_sum[layer_idx] += (gain * param.grad.detach()).abs().double().cpu()
        n_batches += 1
    if n_batches <= 0:
        raise RuntimeError("no finite source-side batches were available for saliency scoring")
    return (score_sum / float(n_batches)).float(), n_batches


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device, require_gpu=args.require_gpu)
    ckpt_path = Path(args.source_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--source_checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config", {}))
    model = _model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    prompt_encoder = _prompt_encoder_from_config(config)
    if checkpoint.get("prompt_encoder_state_dict") is not None:
        prompt_encoder.load_state_dict(checkpoint["prompt_encoder_state_dict"])
    prompt_encoder.to(device)

    source_regions = _resolve_source_regions(
        args.target_region,
        args.source_regions or ",".join(config.get("source_regions", [])),
    )
    global_to_source_idx = {
        _GLOBAL_REGION_IDX_MAP[region_name]: idx
        for idx, region_name in enumerate(source_regions)
    }
    dataset = build_hydroda_dataset(
        da_nc_path=args.da_nc,
        region_masks_nc=args.region_masks_nc,
        splits_json=args.splits_json,
        target_region=args.target_region,
        split_type="source_fit",
        K=args.K,
        seed=args.seed,
        adaptation_setting=args.adaptation_setting,
        freeze_manifest=args.freeze_manifest,
        dataset_backend=args.dataset_backend,
        active_region_ids=source_regions,
        tensor_cache_dir=args.tensor_cache_dir,
        max_year_cache_entries=args.max_year_cache_entries,
        tensor_cache_load_mode=args.tensor_cache_load_mode,
    )
    trainer = PromptConditionedTrainer(
        model=model,
        prompt_encoder=prompt_encoder,
        train_dataset=dataset,
        max_epochs=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=str(device),
        checkpoint_dir=str(Path(args.output_path).parent / "_saliency_tmp"),
        source_regions=source_regions,
        global_to_source_lookup=global_to_source_idx,
        use_lat_weighted_loss=args.use_lat_weighted_loss,
        source_val_residual_gain=False,
        target_increment_normalization=bool(
            config.get("target_increment_normalization", config.get("inc_mean") is not None)
        ),
        model_type="hyperda_basis_adapter",
        hyper_n_basis=int(config.get("hyper_n_basis", 8)),
        hyper_adapter_bottleneck=config.get("hyper_adapter_bottleneck"),
        hyper_adapter_scale=float(config.get("hyper_adapter_scale", 1.0)),
        hyper_coeff_generator=config.get("hyper_coeff_generator", "shared_layer_aware_rank_gated_stable"),
        hyper_rank_gate_top_k=int(config.get("hyper_rank_gate_top_k", 4)),
        hyper_rank_gate_temperature_init=float(config.get("hyper_rank_gate_temperature_init", 2.0)),
        hyper_adapter_param_style=config.get("hyper_adapter_param_style", "dora_like_gain_bounded"),
        hyper_reliability_gate=config.get("hyper_reliability_gate", "prompt_scalar"),
        hyper_reliability_init=float(config.get("hyper_reliability_init", 0.95)),
        hyper_prompt_manifold_reliability=bool(config.get("hyper_prompt_manifold_reliability", False)),
        hyper_prompt_manifold_reliability_strength=float(
            config.get("hyper_prompt_manifold_reliability_strength", 0.0)
        ),
        zero_shot_prior_form=config.get("zero_shot_prior_form", "direct_hyper"),
        source_residual_rho=float(config.get("source_residual_rho", config.get("zero_shot_rho", 1.0))),
        source_residual_gate=config.get("source_residual_gate", "prompt_reliability_scalar"),
        source_residual_gate_init=float(config.get("source_residual_gate_init", 0.95)),
        source_residual_reliability_dim=int(config.get("source_residual_reliability_dim", 5)),
        source_episode_prompt_policy=config.get("source_episode_prompt_policy", "current_region_prompt"),
        source_prototype_cache_mode="off",
        dataset_backend=args.dataset_backend,
        tensor_cache_load_mode=args.tensor_cache_load_mode,
        _resume_ch_mean=np.array(config["ch_mean"], dtype=np.float32) if config.get("ch_mean") is not None else None,
        _resume_ch_std=np.array(config["ch_std"], dtype=np.float32) if config.get("ch_std") is not None else None,
        _resume_inc_mean=np.array(config["inc_mean"], dtype=np.float32) if config.get("inc_mean") is not None else None,
        _resume_inc_std=np.array(config["inc_std"], dtype=np.float32) if config.get("inc_std") is not None else None,
    )
    scores, batches = build_saliency_scores(trainer, max_batches=args.max_batches)
    artifact = make_saliency_artifact(
        scores,
        score_type="dora_gain_snip_abs_gain_grad",
        source_split=args.source_split,
        metadata={
            "source_checkpoint_path": str(ckpt_path),
            "source_checkpoint_sha256": compute_sha256(ckpt_path),
            "target_region": args.target_region,
            "source_regions": source_regions,
            "n_batches": int(batches),
            "max_batches": int(args.max_batches),
            "dataset_split_type_read": "source_fit",
            "score_estimator": "SNIP-style abs(dora_gain * dL/d_dora_gain_delta)",
            "target_val_usage": "unused_in_main_protocol",
            "target_eval_usage": "final_eval_only_no_selection",
        },
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "prior_sha256": artifact["prior_sha256"],
                "shape": list(artifact["prior"].shape),
                "n_batches": batches,
                "source_split": args.source_split,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
