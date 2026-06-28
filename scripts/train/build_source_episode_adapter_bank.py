#!/usr/bin/env python3
"""Build a source-side HyperDA adapter coefficient episode bank.

This is the first P4 artifact stage only. It adapts source pseudo-target
episodes in lightweight adapter coefficient space and saves provenance-rich
artifacts for later prior-generator work. It does not train a generator, use
diffusion, generate full U-Net parameters, or read target_eval labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from hydroda.baselines.prompt_conditioned import target_context_prompt_metadata
from hydroda.data.dataset import HydroDADataset
from hydroda.data.file_hash import compute_sha256
from hydroda.data.protocol import ProtocolConfig
from hydroda.evaluation.harness import evaluate_split, summarize_metric_rows

from scripts.train.train_hyperda_few_shot_adapt import (
    COEFF_RESIDUAL_PARAMETER_NAMES,
    DA_NC,
    FREEZE_MANIFEST,
    REGION_MASKS_NC,
    SPLITS_JSON,
    PROTOCOL_FREEZE_ID,
    FewShotAdaptationState,
    _loader,
    _normalize_x,
    apply_adapt_scope,
    build_few_shot_target_context_prompt_state,
    coefficient_residual_vector,
    load_source_checkpoint_for_few_shot,
    run_ridge_coeff_adaptation,
)


BANK_SCHEMA_VERSION = "hyperda_source_episode_adapter_bank_v1"
EPISODE_SCHEMA_VERSION = "hyperda_source_episode_adapter_episode_v1"
ADAPTER_SPACE = "hyperda_adapter_coefficient_residual_logits"
ALL_US_REGIONS = ("US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6")
REQUIRED_EPISODE_METADATA_FIELDS = (
    "schema_version",
    "episode_id",
    "pseudo_target_region",
    "source_regions_used",
    "K",
    "seed",
    "context_dates",
    "context_dates_hash",
    "support_dates",
    "support_dates_hash",
    "support_count",
    "query_dates",
    "query_dates_hash",
    "prompt_context_stats",
    "adapter_space",
    "adapter_parameter_names",
    "adapter_vector_dim",
    "adapter_delta_norm",
    "query_skill",
    "source_checkpoint_path",
    "checkpoint_hash",
    "split_manifest_path",
    "split_manifest_hash",
    "normalizer_provenance",
    "leakage_metadata",
)


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _parse_k_list(value: str) -> List[int]:
    values = [int(item) for item in _split_csv(value)]
    allowed = set(ProtocolConfig().main_K_values)
    bad = [k for k in values if k not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"K_list contains unsupported values {bad}; expected subset of {sorted(allowed)}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build source-side HyperDA adapter coefficient episode bank."
    )
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--output_dir", default="artifacts/operator_bank/source_episode_adapter_bank")
    parser.add_argument("--da_nc", default=DA_NC)
    parser.add_argument("--region_masks_nc", default=REGION_MASKS_NC)
    parser.add_argument("--splits_json", default=SPLITS_JSON)
    parser.add_argument("--freeze_manifest", default=FREEZE_MANIFEST)
    parser.add_argument("--K_list", type=_parse_k_list, default=[0, 4, 12])
    parser.add_argument("--pseudo_target_regions", type=_split_csv, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_latent_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--max_query_samples", type=int, default=0)
    parser.add_argument("--ridge_lambda", type=float, default=1.0)
    parser.add_argument("--ridge_clip_coeff_norm", type=float, default=1.0)
    parser.add_argument("--ridge_trust_region_radius", type=float, default=1.0)
    parser.add_argument(
        "--ridge_max_feature_pixels",
        type=int,
        default=20000,
        help="Deterministic cap on support feature observations used by ridge coefficient adaptation; 0 means all.",
    )
    parser.add_argument(
        "--ridge_standardize_features",
        action="store_true",
        help="Scale ridge design columns by their support RMS before solving.",
    )
    parser.add_argument("--surface_weight", type=float, default=3.0)
    parser.add_argument("--rootzone_weight", type=float, default=1.0)
    parser.add_argument("--use_lat_weighted_loss", action="store_true", default=True)
    parser.add_argument("--no_lat_weighted_loss", action="store_false", dest="use_lat_weighted_loss")
    parser.add_argument(
        "--allow_regions_not_in_checkpoint",
        action="store_true",
        help="Internal development override. By default episodes must be in checkpoint config.source_regions.",
    )
    args = parser.parse_args()
    if args.max_episodes < 0:
        parser.error("--max_episodes must be non-negative")
    if args.max_query_samples < 0:
        parser.error("--max_query_samples must be non-negative")
    if args.ridge_lambda < 0:
        parser.error("--ridge_lambda must be non-negative")
    if args.ridge_clip_coeff_norm < 0:
        parser.error("--ridge_clip_coeff_norm must be non-negative")
    if args.ridge_trust_region_radius < 0:
        parser.error("--ridge_trust_region_radius must be non-negative")
    if args.ridge_max_feature_pixels < 0:
        parser.error("--ridge_max_feature_pixels must be non-negative")
    return args


def _date_hash(dates: Sequence[str]) -> str:
    payload = json.dumps(list(dates), separators=(",", ":"), sort_keys=True).encode("utf-8")
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _date_str_records(dataset: HydroDADataset, date_key: str) -> List[str]:
    records = getattr(dataset, "_split_entry", {}).get(date_key, [])
    if not isinstance(records, list):
        return []
    return [str(record.get("date_str", "")) for record in records if isinstance(record, dict) and record.get("date_str")]


def _split_entry_hash(dataset: HydroDADataset, key: str, fallback_dates: Sequence[str]) -> str:
    value = getattr(dataset, "_split_entry", {}).get(key, "")
    return str(value) if value else _date_hash(fallback_dates)


def _checkpoint_source_regions(checkpoint: Dict[str, Any]) -> List[str]:
    config = dict(checkpoint.get("config", {}))
    source_regions = config.get("source_regions")
    if source_regions:
        return [str(region) for region in source_regions]
    global_indices = config.get("source_region_global_indices")
    if global_indices:
        return [f"US-R{int(idx) + 1}" for idx in global_indices]
    num_regions = int(config.get("num_regions", len(ALL_US_REGIONS)))
    return list(ALL_US_REGIONS[:num_regions])


def resolve_pseudo_target_regions(
    checkpoint: Dict[str, Any],
    requested_regions: Optional[Sequence[str]],
    *,
    allow_regions_not_in_checkpoint: bool,
) -> List[str]:
    """Resolve pseudo-target regions from checkpoint source-region metadata."""
    source_regions = _checkpoint_source_regions(checkpoint)
    candidates = [str(region) for region in (requested_regions or source_regions)]
    unknown = [region for region in candidates if region not in source_regions]
    if unknown and not allow_regions_not_in_checkpoint:
        raise ValueError(
            "Requested pseudo_target_regions are not present in checkpoint source_regions: "
            f"{unknown}. checkpoint source_regions={source_regions}"
        )
    for region in candidates:
        if region not in ALL_US_REGIONS:
            raise ValueError(f"Unsupported pseudo_target_region={region!r}; expected one of {list(ALL_US_REGIONS)}")
    return candidates


def source_regions_used_for_episode(checkpoint_source_regions: Sequence[str], pseudo_target_region: str) -> List[str]:
    return [region for region in checkpoint_source_regions if region != pseudo_target_region]


def validate_episode_metadata(metadata: Dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_EPISODE_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"episode metadata missing required fields: {missing}")
    if metadata["schema_version"] != EPISODE_SCHEMA_VERSION:
        raise ValueError(
            f"episode schema_version={metadata['schema_version']!r}; expected {EPISODE_SCHEMA_VERSION!r}"
        )
    if metadata["adapter_space"] != ADAPTER_SPACE:
        raise ValueError(f"episode adapter_space={metadata['adapter_space']!r}; expected {ADAPTER_SPACE!r}")
    leakage = metadata.get("leakage_metadata", {})
    if leakage.get("target_eval_used") is not False:
        raise ValueError("episode leakage_metadata.target_eval_used must be false")
    if leakage.get("target_eval_labels_loaded") is not False:
        raise ValueError("episode leakage_metadata.target_eval_labels_loaded must be false")
    if leakage.get("query_role") != "source_val":
        raise ValueError("episode leakage_metadata.query_role must be source_val")
    if int(metadata.get("K", -1)) == 0 and bool(leakage.get("support_labels_used")):
        raise ValueError("K=0 episode must not use support labels")


def build_episode_metadata(
    *,
    episode_id: str,
    pseudo_target_region: str,
    source_regions_used: Sequence[str],
    K: int,
    seed: int,
    context_dates: Sequence[str],
    context_dates_hash: str,
    support_dates: Sequence[str],
    support_dates_hash: str,
    query_dates: Sequence[str],
    query_dates_hash: str,
    prompt_context_stats: Dict[str, Any],
    adapter_parameter_names: Sequence[str],
    adapter_vector_dim: int,
    adapter_delta_norm: float,
    query_skill: Dict[str, Any],
    checkpoint_path: str,
    checkpoint_hash: str,
    split_manifest_path: str,
    split_manifest_hash: str,
    normalizer_provenance: Dict[str, Any],
    ridge_diagnostics: Dict[str, Any],
    checkpoint_source_regions: Sequence[str],
    allowed_region_override: bool,
) -> Dict[str, Any]:
    metadata = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "episode_id": episode_id,
        "pseudo_target_region": pseudo_target_region,
        "source_regions_used": list(source_regions_used),
        "K": int(K),
        "seed": int(seed),
        "context_dates": list(context_dates),
        "context_dates_hash": str(context_dates_hash),
        "support_dates": list(support_dates),
        "support_dates_hash": str(support_dates_hash),
        "support_count": len(support_dates),
        "query_dates": list(query_dates),
        "query_dates_hash": str(query_dates_hash),
        "prompt_context_stats": dict(prompt_context_stats),
        "adapter_space": ADAPTER_SPACE,
        "adapter_parameter_names": list(adapter_parameter_names),
        "adapter_vector_dim": int(adapter_vector_dim),
        "adapter_delta_norm": float(adapter_delta_norm),
        "query_skill": dict(query_skill),
        "source_checkpoint_path": str(checkpoint_path),
        "checkpoint_hash": str(checkpoint_hash),
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_hash": str(split_manifest_hash),
        "normalizer_provenance": dict(normalizer_provenance),
        "ridge_diagnostics": dict(ridge_diagnostics),
        "leakage_metadata": {
            "target_eval_used": False,
            "target_eval_labels_loaded": False,
            "target_eval_features_loaded": False,
            "target_val_used": False,
            "query_role": "source_val",
            "support_role": "source_fit_pseudo_support",
            "context_role": "source_fit_input_side_pseudo_context",
            "support_labels_loaded": bool(int(K) > 0 and len(support_dates) > 0),
            "support_labels_used": bool(int(K) > 0 and len(support_dates) > 0),
            "label_use": "source_support_only_for_K_gt_0" if int(K) > 0 else "none_for_K0",
            "normalization_source": normalizer_provenance.get("normalization_source", ""),
            "real_target_region_excluded_by_checkpoint_source_regions": pseudo_target_region in set(checkpoint_source_regions),
            "allow_regions_not_in_checkpoint": bool(allowed_region_override),
        },
    }
    validate_episode_metadata(metadata)
    return metadata


def save_adapter_coefficient_artifact(
    path: Path,
    model: torch.nn.Module,
    *,
    base_coeff_vector: torch.Tensor,
) -> Dict[str, Any]:
    """Save only low-dimensional adapter coefficient vectors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    adapted = coefficient_residual_vector(model)
    base = base_coeff_vector.detach().float().cpu()
    if adapted.numel() != base.numel():
        raise ValueError(f"adapter coefficient vector length changed: base={base.numel()} adapted={adapted.numel()}")
    delta = adapted - base
    payload = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "space": ADAPTER_SPACE,
        "parameter_names": list(COEFF_RESIDUAL_PARAMETER_NAMES),
        "parameter_shapes": {
            name: list(dict(model.named_parameters())[name].shape)
            for name in COEFF_RESIDUAL_PARAMETER_NAMES
        },
        "base_coeff_vector": base,
        "adapted_coeff_vector": adapted.detach().float().cpu(),
        "delta_coeff_vector": delta.detach().float().cpu(),
        "adapter_vector_dim": int(delta.numel()),
        "adapter_delta_norm": float(torch.linalg.vector_norm(delta).item()),
    }
    torch.save(payload, path)
    return {
        "space": payload["space"],
        "parameter_names": payload["parameter_names"],
        "adapter_vector_dim": payload["adapter_vector_dim"],
        "adapter_delta_norm": payload["adapter_delta_norm"],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def write_bank_manifest(
    *,
    output_dir: Path,
    episodes: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    checkpoint_hash: str,
    split_manifest_hash: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": BANK_SCHEMA_VERSION,
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "n_episodes": len(episodes),
        "checkpoint_hash": checkpoint_hash,
        "split_manifest_hash": split_manifest_hash,
        "config": dict(config),
        "artifact_policy": {
            "adapter_space": ADAPTER_SPACE,
            "full_parameter_generation": False,
            "diffusion": False,
            "generator_training": False,
            "target_eval_usage": "none",
            "query_role": "source_val",
        },
        "episodes": list(episodes),
    }
    write_json(output_dir / "manifest.json", manifest)
    with open(output_dir / "episodes_index.jsonl", "w", encoding="utf-8") as f:
        for row in episodes:
            f.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
    summary_fields = [
        "episode_id",
        "pseudo_target_region",
        "K",
        "adapter_delta_norm",
        "surface_skill_primary",
        "rootzone_skill_primary",
        "metadata_path",
        "adapter_path",
        "query_metrics_path",
    ]
    with open(output_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        for row in episodes:
            writer.writerow(row)
    return manifest


class _StatePredictor:
    method_name = "hyperda_source_episode_adapter"

    def __init__(
        self,
        state: FewShotAdaptationState,
        prompt_state: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.state = state
        self.prompt_state = prompt_state
        self.device = device

    def predict(self, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
        from scripts.train.train_hyperda_few_shot_adapt import (
            _denormalize_increment,
            _model_forward,
        )
        from hydroda.baselines.prompt_conditioned import compose_target_context_prompt_from_state

        x = torch.from_numpy(np.asarray(sample["x"], dtype=np.float32)).unsqueeze(0).to(self.device)
        month_value = int(sample.get("month", 6))
        months = torch.tensor([month_value], dtype=torch.long, device=self.device)
        x_norm = _normalize_x(x, self.state.normalization)
        z = compose_target_context_prompt_from_state(self.prompt_state, months, device=self.device)
        with torch.no_grad():
            pred = _model_forward(self.state.model, x_norm, z, months, x)
        pred = _denormalize_increment(
            pred,
            self.state.normalization,
            normalize_increment=self.state.normalization.get("inc_mean") is not None,
        )
        pred_inc_s = pred[0, 0].detach().cpu().numpy().astype(np.float32)
        pred_inc_r = pred[0, 1].detach().cpu().numpy().astype(np.float32)
        forecast_surface = np.asarray(sample["forecast_surface"], dtype=np.float32)
        forecast_rootzone = np.asarray(sample["forecast_rootzone"], dtype=np.float32)
        return {
            "pred_increment_surface": pred_inc_s,
            "pred_increment_rootzone": pred_inc_r,
            "pred_analysis_surface": forecast_surface + pred_inc_s,
            "pred_analysis_rootzone": forecast_rootzone + pred_inc_r,
        }


class _PseudoQueryDataset:
    """Metadata wrapper that marks source_val as pseudo-query, not target_eval."""

    def __init__(self, dataset: HydroDADataset, *, target_eval_dates_hash: str = "not_used_source_val_query") -> None:
        self.dataset = dataset
        self.target_eval_dates_hash = target_eval_dates_hash

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(self.dataset[idx])
        sample["split_role"] = "source_val_pseudo_query"
        sample["target_eval_dates_hash"] = self.target_eval_dates_hash
        sample["target_region_id"] = sample.get("target_region_id", "")
        sample["active_region_ids"] = list(sample.get("active_region_ids", []))
        return sample

    def preload(self) -> Dict[int, Dict[str, Any]]:
        return {idx: self[idx] for idx in range(len(self))}


def _query_skill_from_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        import pandas as pd

        return summarize_metric_rows(pd.DataFrame(rows))
    except Exception as exc:
        return {"status": "summary_failed", "error": str(exc)}


def _extract_skill_value(query_skill: Dict[str, Any], variable: str) -> Optional[float]:
    value = query_skill.get(variable, {}).get("skill_primary")
    return float(value) if value is not None else None


def _build_episode(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    pseudo_target_region: str,
    K: int,
    checkpoint_source_regions: Sequence[str],
    checkpoint_hash: str,
    split_manifest_hash: str,
) -> Dict[str, Any]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=args.source_checkpoint,
        device=device,
        target_latent_dim=args.target_latent_dim,
    )
    apply_adapt_scope(state.model, "coeff_only", freeze_monthly_gain=True)
    base_coeff = coefficient_residual_vector(state.model)
    adaptation_setting = ProtocolConfig().adaptation_setting_for_K(int(K))

    context_dataset = HydroDADataset(
        da_nc_path=args.da_nc,
        region_masks_nc=args.region_masks_nc,
        splits_json=args.splits_json,
        target_region=pseudo_target_region,
        split_type="target_context",
        K=int(K),
        seed=args.seed,
        adaptation_setting=adaptation_setting,
        freeze_manifest=args.freeze_manifest,
    )
    context_dataset.set_active_region(pseudo_target_region)
    support_dataset = None
    query_dataset = None
    try:
        context_dates = _date_str_records(context_dataset, "target_context_dates")
        context_hash = _split_entry_hash(context_dataset, "target_context_dates_hash", context_dates)
        prompt_state = build_few_shot_target_context_prompt_state(
            state=state,
            samples=(context_dataset.get_input_side_sample(i) for i in range(len(context_dataset))),
            target_region=pseudo_target_region,
            device=device,
            context_hash=context_hash,
        )
        prompt_stats = target_context_prompt_metadata(prompt_state)

        support_dates: List[str] = []
        support_hash = _split_entry_hash(context_dataset, "target_support_dates_hash", support_dates)
        ridge_diagnostics: Dict[str, Any] = {}
        if int(K) > 0:
            support_dataset = HydroDADataset(
                da_nc_path=args.da_nc,
                region_masks_nc=args.region_masks_nc,
                splits_json=args.splits_json,
                target_region=pseudo_target_region,
                split_type="target_support",
                K=int(K),
                seed=args.seed,
                adaptation_setting=adaptation_setting,
                freeze_manifest=args.freeze_manifest,
            )
            support_dataset.set_active_region(pseudo_target_region)
            support_dates = _date_str_records(support_dataset, "target_support_dates")
            support_hash = _split_entry_hash(support_dataset, "target_support_dates_hash", support_dates)
            support_loader = _loader(support_dataset, args.batch_size, args.num_workers, shuffle=False)
            ridge_diagnostics = run_ridge_coeff_adaptation(
                state=state,
                loader=support_loader,
                device=device,
                target_context_prompt_state=prompt_state,
                normalize_increment=state.normalization.get("inc_mean") is not None,
                ridge_lambda=args.ridge_lambda,
                ridge_clip_coeff_norm=args.ridge_clip_coeff_norm,
                ridge_trust_region_radius=args.ridge_trust_region_radius,
                ridge_max_feature_pixels=args.ridge_max_feature_pixels,
                ridge_standardize_features=args.ridge_standardize_features,
                ridge_weighting="global_pixel_l2",
                surface_weight=args.surface_weight,
                rootzone_weight=args.rootzone_weight,
                use_lat_weighted_loss=args.use_lat_weighted_loss,
            )

        query_dataset = HydroDADataset(
            da_nc_path=args.da_nc,
            region_masks_nc=args.region_masks_nc,
            splits_json=args.splits_json,
            target_region=pseudo_target_region,
            split_type="source_val",
            K=int(K),
            seed=args.seed,
            adaptation_setting=adaptation_setting,
            freeze_manifest=args.freeze_manifest,
        )
        query_dataset.set_active_region(pseudo_target_region)
        query_dates = _date_str_records(query_dataset, "source_val_dates")
        query_hash = _split_entry_hash(query_dataset, "source_val_dates_hash", query_dates)
        predictor = _StatePredictor(state=state, prompt_state=prompt_state, device=device)
        pseudo_query_dataset = _PseudoQueryDataset(query_dataset)
        rows = evaluate_split(
            dataset=pseudo_query_dataset,
            predictor=predictor,
            split_role="source_val_pseudo_query",
            experiment_id=f"p4_source_episode_{pseudo_target_region}_K{K}_S{args.seed}",
            protocol_freeze_id=PROTOCOL_FREEZE_ID,
            method=predictor.method_name,
            split_file=args.splits_json,
            mask_file=args.region_masks_nc,
            target_context_dates_hash=context_hash,
            target_support_dates_hash=support_hash,
            support_dates_hash=support_hash,
            target_train_dates_hash=context_hash,
            target_eval_dates_hash="not_used_source_val_query",
            split_manifest_sha256=split_manifest_hash,
            preloaded=False,
            max_samples=args.max_query_samples if args.max_query_samples > 0 else None,
        )
        query_skill = _query_skill_from_rows(rows)

        episode_id = f"{pseudo_target_region}_K{int(K)}_S{args.seed}"
        episode_dir = output_dir / "episodes" / episode_id
        adapter_info = save_adapter_coefficient_artifact(
            episode_dir / "adapter_coefficients.pt",
            state.model,
            base_coeff_vector=base_coeff,
        )
        normalizer_provenance = {
            "normalization_source": "source_fit_only_from_source_checkpoint",
            "ch_mean_present": state.normalization.get("ch_mean") is not None,
            "ch_std_present": state.normalization.get("ch_std") is not None,
            "inc_mean_present": state.normalization.get("inc_mean") is not None,
            "inc_std_present": state.normalization.get("inc_std") is not None,
        }
        metadata = build_episode_metadata(
            episode_id=episode_id,
            pseudo_target_region=pseudo_target_region,
            source_regions_used=source_regions_used_for_episode(checkpoint_source_regions, pseudo_target_region),
            K=int(K),
            seed=args.seed,
            context_dates=context_dates,
            context_dates_hash=context_hash,
            support_dates=support_dates,
            support_dates_hash=support_hash,
            query_dates=query_dates,
            query_dates_hash=query_hash,
            prompt_context_stats=prompt_stats,
            adapter_parameter_names=adapter_info["parameter_names"],
            adapter_vector_dim=adapter_info["adapter_vector_dim"],
            adapter_delta_norm=adapter_info["adapter_delta_norm"],
            query_skill=query_skill,
            checkpoint_path=args.source_checkpoint,
            checkpoint_hash=checkpoint_hash,
            split_manifest_path=args.splits_json,
            split_manifest_hash=split_manifest_hash,
            normalizer_provenance=normalizer_provenance,
            ridge_diagnostics=ridge_diagnostics,
            checkpoint_source_regions=checkpoint_source_regions,
            allowed_region_override=bool(args.allow_regions_not_in_checkpoint),
        )
        write_json(episode_dir / "metadata.json", metadata)
        write_json(episode_dir / "query_metrics.json", {"rows": rows, "summary": query_skill})
        return {
            "episode_id": episode_id,
            "pseudo_target_region": pseudo_target_region,
            "K": int(K),
            "metadata_path": str((episode_dir / "metadata.json").relative_to(output_dir)),
            "adapter_path": str((episode_dir / "adapter_coefficients.pt").relative_to(output_dir)),
            "query_metrics_path": str((episode_dir / "query_metrics.json").relative_to(output_dir)),
            "adapter_delta_norm": adapter_info["adapter_delta_norm"],
            "surface_skill_primary": _extract_skill_value(query_skill, "surface"),
            "rootzone_skill_primary": _extract_skill_value(query_skill, "rootzone"),
        }
    finally:
        context_dataset.close()
        if support_dataset is not None:
            support_dataset.close()
        if query_dataset is not None:
            query_dataset.close()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_source_regions = _checkpoint_source_regions(checkpoint)
    pseudo_regions = resolve_pseudo_target_regions(
        checkpoint,
        args.pseudo_target_regions,
        allow_regions_not_in_checkpoint=bool(args.allow_regions_not_in_checkpoint),
    )
    checkpoint_hash = compute_sha256(args.source_checkpoint) if Path(args.source_checkpoint).exists() else ""
    split_manifest_hash = compute_sha256(args.splits_json) if Path(args.splits_json).exists() else ""

    planned = [(region, int(K)) for region in pseudo_regions for K in args.K_list]
    if args.max_episodes > 0:
        planned = planned[: args.max_episodes]
    episodes = []
    for region, K in planned:
        print(f"Building source episode: pseudo_target_region={region} K={K}", flush=True)
        episodes.append(
            _build_episode(
                args=args,
                output_dir=output_dir,
                pseudo_target_region=region,
                K=K,
                checkpoint_source_regions=checkpoint_source_regions,
                checkpoint_hash=checkpoint_hash,
                split_manifest_hash=split_manifest_hash,
            )
        )

    write_bank_manifest(
        output_dir=output_dir,
        episodes=episodes,
        config={
            "source_checkpoint": args.source_checkpoint,
            "splits_json": args.splits_json,
            "K_list": list(args.K_list),
            "pseudo_target_regions": list(pseudo_regions),
            "seed": int(args.seed),
            "ridge_lambda": float(args.ridge_lambda),
            "ridge_clip_coeff_norm": float(args.ridge_clip_coeff_norm),
            "ridge_trust_region_radius": float(args.ridge_trust_region_radius),
            "ridge_max_feature_pixels": int(args.ridge_max_feature_pixels),
            "ridge_standardize_features": bool(args.ridge_standardize_features),
            "max_query_samples": int(args.max_query_samples),
            "allow_regions_not_in_checkpoint": bool(args.allow_regions_not_in_checkpoint),
            "checkpoint_source_regions": list(checkpoint_source_regions),
        },
        checkpoint_hash=checkpoint_hash,
        split_manifest_hash=split_manifest_hash,
    )
    print(f"Saved source episode adapter bank: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
