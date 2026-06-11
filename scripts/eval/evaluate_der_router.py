#!/usr/bin/env python3
"""Evaluate HydroDA-DER variable-wise dual-expert routing.

Legacy/secondary note:
    This router uses target_val=2022 for variable-wise expert selection and is
    therefore not part of the V4.4 zero/few-shot main protocol. Keep it only as
    an explicit secondary/internal analysis path.

Two modes are intentionally separated:

``select``
    Evaluate fixed candidate experts on ``target_val`` and write a
    ``router_config.json`` with per-variable expert choices.

``eval``
    Evaluate ``target_eval`` by reading an existing ``router_config.json``.
    This mode never searches candidates or changes routing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from hydroda.data.dataset import HydroDADataset
from hydroda.data.file_hash import compute_sha256
from hydroda.evaluation.der_router import (
    DEFAULT_SELECTION_METRIC,
    MODEL_SELECTION_SOURCE,
    ROUTER_SELECTION_SOURCE,
    DualExpertRouterPredictor,
    build_router_config,
    create_predictor,
    dataset_date_hash,
    load_router_config,
    select_variable_experts,
    target_val_dates_hash,
    validate_eval_uses_router_config,
    write_router_config,
)
from hydroda.evaluation.harness import evaluate_split, summarize_metric_rows
from hydroda.utils.device import resolve_device


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_target_train_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
PROTOCOL_FREEZE_ID = "hyperda_v4_3_historical_target_adapt_2015_2025_train2015_2021_val2022_test2023_2025"
PROTOCOL_STATUS = "legacy_secondary_target_val_router_not_main_protocol"


def _make_dataset(
    *,
    target_region: str,
    split_type: str,
    adaptation_setting: str,
    seed: int,
    da_nc_path: str,
    region_masks_nc: str,
    splits_json: str,
    freeze_manifest: str,
) -> Any:
    return HydroDADataset(
        da_nc_path=da_nc_path,
        region_masks_nc=region_masks_nc,
        splits_json=splits_json,
        target_region=target_region,
        split_type=split_type,
        K=None,
        seed=seed,
        adaptation_setting=adaptation_setting,
        freeze_manifest=freeze_manifest,
    )


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {}
    return summarize_metric_rows(pd.DataFrame(rows))


def _write_metrics_outputs(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Dict[str, Dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "metrics_long.csv", index=False)
    summary = summarize_metric_rows(df)
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _evaluate_predictor(
    *,
    dataset: Any,
    predictor: Any,
    split_role: str,
    experiment_id: str,
    method: str,
    splits_json: str,
    region_masks_nc: str,
    split_manifest_sha256: str,
    max_samples: int | None,
) -> List[Dict[str, Any]]:
    return evaluate_split(
        dataset=dataset,
        predictor=predictor,
        split_role=split_role,
        experiment_id=experiment_id,
        protocol_freeze_id=PROTOCOL_FREEZE_ID,
        method=method,
        split_file=splits_json,
        mask_file=region_masks_nc,
        split_manifest_sha256=split_manifest_sha256,
        preloaded=False,
        max_samples=max_samples,
    )


def _candidate_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
) -> List[Dict[str, Any]]:
    tagged: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["candidate_id"] = candidate_id
        tagged.append(out)
    return tagged


def _build_dual_expert_predictor_from_config(
    router_config: Mapping[str, Any],
    *,
    device: str,
    target_region: str,
) -> DualExpertRouterPredictor:
    selected = router_config.get("selected_experts", {})
    surface_cfg = dict(selected["surface"])
    rootzone_cfg = dict(selected["rootzone"])
    surface_expert = create_predictor(
        checkpoint=surface_cfg["checkpoint"],
        predictor_type=surface_cfg["predictor_type"],
        device=device,
        target_region=target_region,
    )
    rootzone_expert = create_predictor(
        checkpoint=rootzone_cfg["checkpoint"],
        predictor_type=rootzone_cfg["predictor_type"],
        device=device,
        target_region=target_region,
    )
    return DualExpertRouterPredictor(
        surface_expert=surface_expert,
        rootzone_expert=rootzone_expert,
        surface_metadata=surface_cfg,
        rootzone_metadata=rootzone_cfg,
    )


def run_select(
    *,
    candidates: Iterable[Mapping[str, Any]],
    target_region: str,
    adaptation_setting: str,
    seed: int,
    output_dir: str | Path,
    router_config_path: str | Path,
    device: str,
    max_samples: int | None,
    da_nc_path: str,
    region_masks_nc: str,
    splits_json: str,
    freeze_manifest: str,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> Dict[str, Any]:
    """Evaluate candidates on target_val and write router_config.json."""
    candidates = [dict(c) for c in candidates]
    if len(candidates) < 2:
        raise ValueError("HydroDA-DER selection requires at least two candidate experts")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_sha256 = compute_sha256(splits_json) if Path(splits_json).exists() else ""

    val_dataset = _make_dataset(
        target_region=target_region,
        split_type="target_val",
        adaptation_setting=adaptation_setting,
        seed=seed,
        da_nc_path=da_nc_path,
        region_masks_nc=region_masks_nc,
        splits_json=splits_json,
        freeze_manifest=freeze_manifest,
    )
    train_dataset = _make_dataset(
        target_region=target_region,
        split_type="target_train",
        adaptation_setting=adaptation_setting,
        seed=seed,
        da_nc_path=da_nc_path,
        region_masks_nc=region_masks_nc,
        splits_json=splits_json,
        freeze_manifest=freeze_manifest,
    )

    all_rows: List[Dict[str, Any]] = []
    candidate_summaries: Dict[str, Any] = {}
    try:
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            predictor = create_predictor(
                checkpoint=candidate["checkpoint"],
                predictor_type=candidate["predictor_type"],
                device=device,
                target_region=target_region,
            )
            rows = _evaluate_predictor(
                dataset=val_dataset,
                predictor=predictor,
                split_role="target_val",
                experiment_id=f"hydroda_der_select_{target_region}_{candidate_id}_S{seed}",
                method=getattr(predictor, "method_name", str(candidate["predictor_type"])),
                splits_json=splits_json,
                region_masks_nc=region_masks_nc,
                split_manifest_sha256=split_manifest_sha256,
                max_samples=max_samples,
            )
            tagged_rows = _candidate_metric_rows(rows, candidate_id=candidate_id)
            all_rows.extend(tagged_rows)
            candidate_summaries[candidate_id] = _metric_summary(rows)
    finally:
        if hasattr(val_dataset, "close"):
            val_dataset.close()
        if hasattr(train_dataset, "close"):
            train_dataset.close()

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(output_dir / "target_val_candidate_metrics_long.csv", index=False)
    selection = select_variable_experts(metrics_df, metric=selection_metric, split_role="target_val")
    config = build_router_config(
        candidates=candidates,
        selection=selection,
        target_region=target_region,
        adaptation_setting=adaptation_setting,
        seed=seed,
        split_manifest_path=splits_json,
        split_manifest_sha256=split_manifest_sha256,
        target_val_dates_hash=target_val_dates_hash(train_dataset, val_dataset),
        target_train_dates_hash=dataset_date_hash(train_dataset, "target_train_dates_hash"),
        target_eval_dates_hash=dataset_date_hash(train_dataset, "target_eval_dates_hash"),
        protocol_freeze_id=PROTOCOL_FREEZE_ID,
        selection_metric=selection_metric,
    )
    write_router_config(config, router_config_path)

    summary = {
        "method": "HydroDA-DER",
        "router_config": str(router_config_path),
        "router_selection_source": ROUTER_SELECTION_SOURCE,
        "model_selection_source": MODEL_SELECTION_SOURCE,
        "target_region": target_region,
        "adaptation_setting": adaptation_setting,
        "seed": seed,
        "split_type": "target_val",
        "selection_metric": selection_metric,
        "selected_experts": config["selected_experts"],
        "candidate_summaries": candidate_summaries,
        "split_manifest_sha256": split_manifest_sha256,
        "target_val_dates_hash": config["target_val_dates_hash"],
        "no_leakage_declaration": config["no_leakage_declaration"],
    }
    with (output_dir / "selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_eval(
    *,
    router_config_path: str | Path,
    target_region: str,
    adaptation_setting: str,
    seed: int,
    split_type: str,
    output_dir: str | Path,
    device: str,
    max_samples: int | None,
    da_nc_path: str,
    region_masks_nc: str,
    splits_json: str,
    freeze_manifest: str,
) -> Dict[str, Any]:
    """Evaluate DER using an existing router config."""
    router_config = validate_eval_uses_router_config(
        split_type=split_type,
        router_config_path=router_config_path,
    )
    if not router_config:
        router_config = load_router_config(router_config_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_sha256 = compute_sha256(splits_json) if Path(splits_json).exists() else ""

    dataset = _make_dataset(
        target_region=target_region,
        split_type=split_type,
        adaptation_setting=adaptation_setting,
        seed=seed,
        da_nc_path=da_nc_path,
        region_masks_nc=region_masks_nc,
        splits_json=splits_json,
        freeze_manifest=freeze_manifest,
    )
    predictor = _build_dual_expert_predictor_from_config(
        router_config,
        device=device,
        target_region=target_region,
    )
    n_samples_evaluated = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    try:
        started = time.time()
        rows = _evaluate_predictor(
            dataset=dataset,
            predictor=predictor,
            split_role=split_type,
            experiment_id=f"hydroda_der_{target_region}_{split_type}_S{seed}",
            method=predictor.method_name,
            splits_json=splits_json,
            region_masks_nc=region_masks_nc,
            split_manifest_sha256=split_manifest_sha256,
            max_samples=max_samples,
        )
    finally:
        if hasattr(dataset, "close"):
            dataset.close()

    metric_summary = _write_metrics_outputs(rows, output_dir)
    summary = {
        "method": predictor.method_name,
        "router_method": router_config["method"],
        "router_config": str(router_config_path),
        "router_selection_source": router_config["router_selection_source"],
        "model_selection_source": router_config.get("model_selection_source", MODEL_SELECTION_SOURCE),
        "selected_experts": router_config["selected_experts"],
        "target_region": target_region,
        "adaptation_setting": adaptation_setting,
        "seed": seed,
        "split_type": split_type,
        "n_samples_evaluated": n_samples_evaluated,
        "n_metric_rows": len(rows),
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "split_manifest_sha256": split_manifest_sha256,
        "target_train_dates_hash": router_config.get("target_train_dates_hash", ""),
        "target_val_dates_hash": router_config.get("target_val_dates_hash", ""),
        "target_eval_dates_hash": router_config.get("target_eval_dates_hash", ""),
        "surface": metric_summary.get("surface", {}),
        "rootzone": metric_summary.get("rootzone", {}),
        "no_leakage_declaration": router_config.get("no_leakage_declaration", {}),
        "eval_time_s": time.time() - started,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _parse_candidate_specs(specs: Sequence[str]) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) < 3:
            raise ValueError(
                "candidate specs must be candidate_id:predictor_type:checkpoint "
                f"(got {spec!r})"
            )
        candidate_id, predictor_type = parts[0], parts[1]
        checkpoint = ":".join(parts[2:])
        candidates.append(
            {
                "candidate_id": candidate_id,
                "predictor_type": predictor_type,
                "checkpoint": checkpoint,
            }
        )
    return candidates


def candidates_from_args(args: argparse.Namespace) -> List[Dict[str, str]]:
    candidates = _parse_candidate_specs(args.candidate or [])
    has_named_surface = bool(getattr(args, "surface_checkpoint", None))
    has_named_rootzone = bool(getattr(args, "rootzone_checkpoint", None))
    if has_named_surface or has_named_rootzone:
        if not (has_named_surface and has_named_rootzone):
            raise ValueError("--surface_checkpoint and --rootzone_checkpoint must be provided together")
        candidates.extend(
            [
                {
                    "candidate_id": "surface_expert",
                    "predictor_type": str(args.surface_predictor_type),
                    "checkpoint": str(args.surface_checkpoint),
                },
                {
                    "candidate_id": "rootzone_expert",
                    "predictor_type": str(args.rootzone_predictor_type),
                    "checkpoint": str(args.rootzone_checkpoint),
                },
            ]
        )
    if not candidates:
        raise ValueError(
            "No DER candidate experts provided. Use --candidate or "
            "--surface_checkpoint/--rootzone_checkpoint."
        )
    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HydroDA-DER variable-wise dual-expert router")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--target_region", required=True)
    common.add_argument("--adaptation_setting", default="target_full_train")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--output_dir", required=True)
    common.add_argument("--device", default="cuda")
    common.add_argument("--require_gpu", action="store_true")
    common.add_argument("--max_samples", type=int, default=0)
    common.add_argument("--da_nc", default=f"{DATA_DIR}/DA.nc")
    common.add_argument("--region_masks_nc", default=REGION_MASKS_NC)
    common.add_argument("--splits_json", default=SPLITS_JSON)
    common.add_argument("--freeze_manifest", default=FREEZE_MANIFEST)

    select_parser = subparsers.add_parser("select", parents=[common], help="Select DER experts on target_val")
    select_parser.add_argument(
        "--candidate",
        action="append",
        help="candidate_id:predictor_type:checkpoint. Repeat for each candidate.",
    )
    select_parser.add_argument("--surface_checkpoint", default=None)
    select_parser.add_argument("--surface_predictor_type", default="source_only")
    select_parser.add_argument("--rootzone_checkpoint", default=None)
    select_parser.add_argument("--rootzone_predictor_type", default="hyperda_target_adapt")
    select_parser.add_argument("--router_config", required=True)
    select_parser.add_argument("--selection_metric", default=DEFAULT_SELECTION_METRIC)

    eval_parser = subparsers.add_parser("eval", parents=[common], help="Evaluate DER with existing router_config")
    eval_parser.add_argument("--router_config", required=True)
    eval_parser.add_argument("--split_type", default="target_eval")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = resolve_device(args.device, require_gpu=args.require_gpu)
    max_samples = args.max_samples if args.max_samples > 0 else None

    if args.command == "select":
        summary = run_select(
            candidates=candidates_from_args(args),
            target_region=args.target_region,
            adaptation_setting=args.adaptation_setting,
            seed=args.seed,
            output_dir=args.output_dir,
            router_config_path=args.router_config,
            device=str(device),
            max_samples=max_samples,
            da_nc_path=args.da_nc,
            region_masks_nc=args.region_masks_nc,
            splits_json=args.splits_json,
            freeze_manifest=args.freeze_manifest,
            selection_metric=args.selection_metric,
        )
        print(json.dumps(summary, indent=2))
        return

    summary = run_eval(
        router_config_path=args.router_config,
        target_region=args.target_region,
        adaptation_setting=args.adaptation_setting,
        seed=args.seed,
        split_type=args.split_type,
        output_dir=args.output_dir,
        device=str(device),
        max_samples=max_samples,
        da_nc_path=args.da_nc,
        region_masks_nc=args.region_masks_nc,
        splits_json=args.splits_json,
        freeze_manifest=args.freeze_manifest,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
