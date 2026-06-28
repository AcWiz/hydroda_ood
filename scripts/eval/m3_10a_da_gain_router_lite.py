#!/usr/bin/env python3
"""M3_10a DA gain consistency router.

This script keeps the HyperDA-TRUST checkpoint frozen and adds only a
source-side DA gain bank plus a bounded post-hoc prediction blend.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor
from hydroda.data.dataset import HydroDADataset
from hydroda.data.file_hash import compute_sha256
from hydroda.evaluation.da_gain_router import (
    DA_GAIN_METHOD_ID,
    DAGainBankAccumulator,
    DAGainRouterPredictor,
    SOURCE_REGIONS,
    build_source_records_from_predictor,
    load_gain_bank,
    save_gain_bank,
    select_eta_from_source_val,
    validate_router_metadata_no_target_selection,
)
from hydroda.evaluation.harness import (
    evaluate_split,
    metric_rows_content_hash,
    metric_values_content_hash,
    summarize_metric_rows,
)
from hydroda.utils.device import resolve_device
from scripts.eval.evaluate_checkpoint import (
    FREEZE_MANIFEST,
    REGION_MASKS_NC,
    ZERO_FEW_SHOT_PROTOCOL_FREEZE_ID,
    ZERO_FEW_SHOT_SPLITS_JSON,
    dataset_date_hash,
)


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"


def _load_dataset(
    *,
    target_region: str,
    split_type: str,
    seed: int,
    K: int,
    splits_json: str,
) -> HydroDADataset:
    return HydroDADataset(
        da_nc_path=f"{DATA_DIR}/DA.nc",
        region_masks_nc=REGION_MASKS_NC,
        splits_json=splits_json,
        target_region=target_region,
        split_type=split_type,
        K=K,
        seed=seed,
        adaptation_setting="zero_shot_context" if int(K) == 0 else f"few_shot_k{int(K)}",
        freeze_manifest=FREEZE_MANIFEST,
    )


def _build_predictor(
    *,
    checkpoint: str,
    device: str,
    target_region: str,
    target_context_prompt: bool,
    K: int,
    seed: int,
    splits_json: str,
) -> PromptConditionedBackbonePredictor:
    predictor = PromptConditionedBackbonePredictor(
        checkpoint_path=checkpoint,
        device=device,
        target_region=target_region,
    )
    if target_context_prompt:
        target_context_dataset = _load_dataset(
            target_region=target_region,
            split_type="target_context",
            seed=seed,
            K=K,
            splits_json=splits_json,
        )
        try:
            predictor.set_target_context_prompt_from_samples(
                target_context_dataset.get_input_side_sample(i)
                for i in range(len(target_context_dataset))
            )
        finally:
            target_context_dataset.close()
    return predictor


def _write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _materialize_source_records(
    *,
    checkpoint: str,
    device: str,
    target_region: str,
    split_type: str,
    seed: int,
    K: int,
    splits_json: str,
    max_samples: int,
    include_predictions: bool,
) -> list[dict[str, Any]]:
    predictor = None
    source_regions: list[str] = []
    if include_predictions:
        predictor = _build_predictor(
            checkpoint=checkpoint,
            device=device,
            target_region=target_region,
            target_context_prompt=False,
            K=K,
            seed=seed,
            splits_json=splits_json,
        )
        source_regions = list(getattr(predictor, "source_regions", []) or [])
    if not source_regions:
        source_regions = [region for region in SOURCE_REGIONS if region != target_region]
    dataset = _load_dataset(
        target_region=target_region,
        split_type=split_type,
        seed=seed,
        K=K,
        splits_json=splits_json,
    )
    try:
        records: list[dict[str, Any]] = []
        for source_region in source_regions:
            dataset.set_active_region(source_region)
            records.extend(
                build_source_records_from_predictor(
                    dataset=dataset,
                    predictor=predictor,
                    max_samples=max_samples if int(max_samples) > 0 else None,
                )
            )
        return records
    finally:
        dataset.close()


def _source_regions_for_bank(target_region: str) -> list[str]:
    return [region for region in SOURCE_REGIONS if region != target_region]


def cmd_build_bank(args: argparse.Namespace) -> None:
    accumulator = DAGainBankAccumulator(
        ridge_lambda=args.ridge_lambda,
        eps=args.eps,
        source_checkpoint=args.checkpoint,
        split_manifest=args.splits_json,
        exploratory_after_us_r1_target_eval_seen=True,
    )
    dataset = _load_dataset(
        target_region=args.target_region,
        split_type="source_fit",
        seed=args.seed,
        K=args.K,
        splits_json=args.splits_json,
    )
    source_regions = _source_regions_for_bank(args.target_region)
    try:
        for source_region in source_regions:
            dataset.set_active_region(source_region)
            n = len(dataset) if int(args.max_samples) <= 0 else min(len(dataset), int(args.max_samples))
            for idx in range(n):
                sample = dataset[idx]
                accumulator.update(sample)
                if args.progress_every > 0 and accumulator.n_records_seen % int(args.progress_every) == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "build-bank",
                                "source_region": source_region,
                                "records_seen": accumulator.n_records_seen,
                                "records_used": accumulator.n_records_used,
                                "max_samples_per_region": int(args.max_samples),
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        dataset.close()
    bank = accumulator.finalize()
    save_gain_bank(bank, args.bank_out)
    print(json.dumps({
        "status": "ok",
        "bank_out": args.bank_out,
        "n_source_records_seen": accumulator.n_records_seen,
        "n_source_records_used": accumulator.n_records_used,
        "bank_content_hash": bank.get("bank_content_hash", ""),
        "source_label_usage": bank.get("source_label_usage"),
        "target_eval_usage": bank.get("target_eval_usage"),
        "exploratory_after_us_r1_target_eval_seen": bank.get("exploratory_after_us_r1_target_eval_seen"),
    }, indent=2, sort_keys=True))


def cmd_select_eta(args: argparse.Namespace) -> None:
    device = str(resolve_device(args.device, require_gpu=args.require_gpu))
    bank = load_gain_bank(args.bank)
    records = _materialize_source_records(
        checkpoint=args.checkpoint,
        device=device,
        target_region=args.target_region,
        split_type="source_val",
        seed=args.seed,
        K=args.K,
        splits_json=args.splits_json,
        max_samples=args.max_samples,
        include_predictions=True,
    )
    eta_grid = [float(item) for item in str(args.eta_grid).split(",") if item.strip()]
    selection = select_eta_from_source_val(
        records,
        bank,
        eta_grid=eta_grid,
        proposal_clip_scale=args.proposal_clip_scale,
        min_dual_cvar_delta=args.min_dual_cvar_delta,
        max_rmse_relative_degrade=args.max_rmse_relative_degrade,
    )
    validate_router_metadata_no_target_selection(selection)
    _write_json(args.selection_out, selection)
    print(json.dumps({
        "status": "ok",
        "selection_out": args.selection_out,
        "selected_eta_surface": selection["selected_eta_surface"],
        "selected_eta_rootzone": selection["selected_eta_rootzone"],
        "source_gate_pass": selection["source_gate_pass"],
        "selection_source": selection["selection_source"],
        "target_eval_usage": selection["target_eval_usage"],
        "exploratory_after_us_r1_target_eval_seen": selection["exploratory_after_us_r1_target_eval_seen"],
    }, indent=2, sort_keys=True))


def _eta_from_args(args: argparse.Namespace) -> tuple[float, float, dict[str, Any]]:
    if args.selection:
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        validate_router_metadata_no_target_selection(selection)
        return (
            float(selection.get("selected_eta_surface", 0.0)),
            float(selection.get("selected_eta_rootzone", selection.get("selected_eta_surface", 0.0))),
            selection,
        )
    eta_surface = float(args.eta)
    return eta_surface, float(args.eta_rootzone if args.eta_rootzone is not None else eta_surface), {}


def cmd_evaluate(args: argparse.Namespace) -> None:
    start = time.time()
    device = str(resolve_device(args.device, require_gpu=args.require_gpu))
    bank = load_gain_bank(args.bank)
    eta_surface, eta_rootzone, selection = _eta_from_args(args)
    target_context_prompt = bool(args.target_context_prompt)
    if args.split_type in {"target_eval", "target_query"} and not args.no_target_context_prompt:
        target_context_prompt = True
    base_predictor = _build_predictor(
        checkpoint=args.checkpoint,
        device=device,
        target_region=args.target_region,
        target_context_prompt=target_context_prompt,
        K=args.K,
        seed=args.seed,
        splits_json=args.splits_json,
    )
    predictor = DAGainRouterPredictor(
        base_predictor,
        bank,
        eta_surface=eta_surface,
        eta_rootzone=eta_rootzone,
        proposal_clip_scale=args.proposal_clip_scale,
    )
    dataset = _load_dataset(
        target_region=args.target_region,
        split_type=args.split_type,
        seed=args.seed,
        K=args.K,
        splits_json=args.splits_json,
    )
    output_dir = Path(args.output_dir) / args.target_region
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_sha256 = compute_sha256(args.splits_json) if Path(args.splits_json).exists() else ""
    try:
        rows, hashes = evaluate_split(
            dataset=dataset,
            predictor=predictor,
            split_role=args.split_type,
            experiment_id=f"{DA_GAIN_METHOD_ID}_{args.target_region}_K{args.K}_S{args.seed}",
            protocol_freeze_id=ZERO_FEW_SHOT_PROTOCOL_FREEZE_ID,
            method=predictor.method_name,
            split_file=args.splits_json,
            mask_file=REGION_MASKS_NC,
            target_context_dates_hash=dataset_date_hash(dataset, "target_context_dates_hash"),
            target_support_dates_hash=dataset_date_hash(dataset, "target_support_dates_hash"),
            support_dates_hash=dataset_date_hash(dataset, "support_dates_hash"),
            target_train_dates_hash=dataset_date_hash(dataset, "target_train_dates_hash"),
            target_eval_dates_hash=dataset_date_hash(dataset, "target_eval_dates_hash"),
            split_manifest_sha256=split_manifest_sha256,
            preloaded=False,
            max_samples=args.max_samples if int(args.max_samples) > 0 else None,
            return_hashes=True,
        )
    finally:
        dataset.close()
    elapsed = time.time() - start
    df = pd.DataFrame(rows)
    if args.output_level in {"long", "full"}:
        df.to_csv(output_dir / "metrics_long.csv", index=False)
    by_region = (
        df.groupby(["target_region_id", "variable", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    by_region.to_csv(output_dir / "metrics_by_region.csv", index=False)
    by_season = (
        df.groupby(["season", "variable", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    by_season.to_csv(output_dir / "metrics_by_season.csv", index=False)
    metric_summary = summarize_metric_rows(df)
    summary = {
        "method": predictor.method_name,
        "method_id": DA_GAIN_METHOD_ID,
        "base_method": getattr(base_predictor, "method_name", "unknown"),
        "checkpoint": args.checkpoint,
        "target_region": args.target_region,
        "K": args.K,
        "seed": args.seed,
        "split_type": args.split_type,
        "split_file": args.splits_json,
        "split_manifest_sha256": split_manifest_sha256,
        "protocol_freeze_id": ZERO_FEW_SHOT_PROTOCOL_FREEZE_ID,
        "eta_surface": eta_surface,
        "eta_rootzone": eta_rootzone,
        "proposal_clip_scale": args.proposal_clip_scale,
        "bank_path": args.bank,
        "bank_content_hash": bank.get("bank_content_hash", ""),
        "selection": selection,
        "router_metadata": predictor.metadata,
        "target_context_dates_hash": dataset_date_hash(dataset, "target_context_dates_hash"),
        "target_support_dates_hash": dataset_date_hash(dataset, "target_support_dates_hash"),
        "target_eval_dates_hash": dataset_date_hash(dataset, "target_eval_dates_hash"),
        "source_label_usage": "source_fit_labels_only_for_bank",
        "eta_selection_source": "source_val_only" if selection else "manual_or_eta_zero",
        "target_val_usage": "unused",
        "target_eval_usage": (
            "final_eval_only_no_selection" if args.split_type in {"target_eval", "target_query"} else "not_target_eval"
        ),
        "exploratory_after_us_r1_target_eval_seen": True,
        "neural_training_epochs": 0,
        "neural_parameter_updates": 0,
        "n_metric_rows": len(rows),
        "n_samples_evaluated": hashes.get("prediction_record_count", 0),
        "prediction_content_hash": hashes.get("prediction_content_hash", ""),
        "metric_content_hash": metric_rows_content_hash(rows),
        "metric_values_content_hash": metric_values_content_hash(rows),
        "surface": metric_summary.get("surface", {}),
        "rootzone": metric_summary.get("rootzone", {}),
        "eval_time_s": elapsed,
    }
    _write_json(output_dir / "summary.json", summary)
    diagnostics = {
        key: summary[key]
        for key in (
            "method_id",
            "base_method",
            "checkpoint",
            "target_region",
            "K",
            "seed",
            "split_type",
            "eta_surface",
            "eta_rootzone",
            "bank_path",
            "bank_content_hash",
            "target_val_usage",
            "target_eval_usage",
            "exploratory_after_us_r1_target_eval_seen",
            "neural_training_epochs",
            "neural_parameter_updates",
            "prediction_content_hash",
            "metric_content_hash",
            "metric_values_content_hash",
        )
    }
    _write_json(output_dir / "diagnostics.json", diagnostics)
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "eta_surface": eta_surface,
        "eta_rootzone": eta_rootzone,
        "surface_rmse_latw_mean": summary["surface"].get("rmse_latw_mean"),
        "rootzone_rmse_latw_mean": summary["rootzone"].get("rmse_latw_mean"),
        "target_eval_usage": summary["target_eval_usage"],
        "exploratory_after_us_r1_target_eval_seen": True,
    }, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M3_10a DA gain router lite")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--target_region", default="US-R1")
        p.add_argument("--K", type=int, default=0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--splits_json", default=ZERO_FEW_SHOT_SPLITS_JSON)
        p.add_argument("--device", default="cuda")
        p.add_argument("--require_gpu", action="store_true")
        p.add_argument("--max_samples", type=int, default=0)

    build = sub.add_parser("build-bank", help="Build source_fit DA gain bank")
    add_common(build)
    build.add_argument("--bank_out", required=True)
    build.add_argument("--ridge_lambda", type=float, default=1e-3)
    build.add_argument("--eps", type=float, default=1e-6)
    build.add_argument("--progress_every", type=int, default=200)
    build.set_defaults(func=cmd_build_bank)

    select = sub.add_parser("select-eta", help="Select eta on source_val")
    add_common(select)
    select.add_argument("--bank", required=True)
    select.add_argument("--selection_out", required=True)
    select.add_argument("--eta_grid", default="0,0.025,0.05,0.10")
    select.add_argument("--proposal_clip_scale", type=float, default=1.0)
    select.add_argument("--min_dual_cvar_delta", type=float, default=-0.005)
    select.add_argument("--max_rmse_relative_degrade", type=float, default=0.002)
    select.set_defaults(func=cmd_select_eta)

    evaluate = sub.add_parser("evaluate", help="Evaluate post-hoc DA gain router")
    add_common(evaluate)
    evaluate.add_argument("--bank", required=True)
    evaluate.add_argument("--selection", default="")
    evaluate.add_argument("--eta", type=float, default=0.0)
    evaluate.add_argument("--eta_rootzone", type=float, default=None)
    evaluate.add_argument("--proposal_clip_scale", type=float, default=1.0)
    evaluate.add_argument("--split_type", default="target_eval")
    evaluate.add_argument("--output_dir", default="artifacts/runs/M3_10a_da_gain_router_lite/eval")
    evaluate.add_argument("--output_level", choices=["compact", "long", "full"], default="compact")
    evaluate.add_argument("--target_context_prompt", action="store_true")
    evaluate.add_argument("--no_target_context_prompt", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
