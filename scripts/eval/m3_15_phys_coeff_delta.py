#!/usr/bin/env python3
"""M3_15 source-safe physics coefficient-delta runner."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor
from hydroda.data.dataset import HydroDADataset
from hydroda.data.file_hash import compute_sha256
from hydroda.evaluation.harness import (
    evaluate_split,
    metric_rows_content_hash,
    metric_values_content_hash,
    summarize_metric_rows,
)
from hydroda.evaluation.m3_15_phys_coeff_delta import (
    M3_15_METHOD_ID,
    M3_15_SOURCE_GATE_REPORT_SCHEMA,
    M315PhysCoeffDeltaPredictor,
    source_gate_report_from_selection,
    select_eta_from_source_val,
    validate_router_metadata_no_target_selection,
    validate_source_gate_for_target_eval,
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
SOURCE_REGIONS = ("US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6")


def _write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_regions_for_selection(target_region: str) -> list[str]:
    return [region for region in SOURCE_REGIONS if region != target_region]


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


def _record_from_sample(sample: Mapping[str, Any], *, sample_idx: int) -> dict[str, Any]:
    return {
        "sample_idx": sample_idx,
        "split_role": sample.get("split_role", ""),
        "query_time_index": int(sample.get("time_index", -1)),
        "query_date": sample.get("date_str", ""),
        "month": sample.get("month"),
        "season": sample.get("season", ""),
        "country_id": sample.get("country_id", ""),
        "target_region_id": sample.get("target_region_id", ""),
        "sample_region_id": sample.get("sample_region_id", ""),
        "active_region_ids": list(sample.get("active_region_ids", [])),
        "adaptation_setting": sample.get("adaptation_setting", "zero_shot_context"),
        "K": sample.get("K", 0),
        "seed": int(sample.get("seed", -1)),
        "x": sample["x"],
        "forecast_surface": sample["forecast_surface"],
        "forecast_rootzone": sample["forecast_rootzone"],
        "analysis_surface": sample["analysis_surface"],
        "analysis_rootzone": sample["analysis_rootzone"],
        "increment_surface": sample["increment_surface"],
        "increment_rootzone": sample["increment_rootzone"],
        "metric_mask": sample["metric_mask"],
        "region_mask": sample.get("region_mask", sample["metric_mask"]),
        "latitude_weight": sample["latitude_weight"],
    }


def _iter_source_records(
    *,
    m3_1_checkpoint: str,
    phys_coeff_checkpoint: str,
    device: str,
    target_region: str,
    split_type: str,
    seed: int,
    K: int,
    splits_json: str,
    max_samples: int,
):
    m3_1_predictor = _build_predictor(
        checkpoint=m3_1_checkpoint,
        device=device,
        target_region=target_region,
        target_context_prompt=False,
        K=K,
        seed=seed,
        splits_json=splits_json,
    )
    phys_predictor = _build_predictor(
        checkpoint=phys_coeff_checkpoint,
        device=device,
        target_region=target_region,
        target_context_prompt=False,
        K=K,
        seed=seed,
        splits_json=splits_json,
    )
    source_regions = list(getattr(m3_1_predictor, "source_regions", []) or [])
    if not source_regions:
        source_regions = _source_regions_for_selection(target_region)
    dataset = _load_dataset(
        target_region=target_region,
        split_type=split_type,
        seed=seed,
        K=K,
        splits_json=splits_json,
    )
    try:
        for source_region in source_regions:
            dataset.set_active_region(source_region)
            n = len(dataset) if int(max_samples) <= 0 else min(len(dataset), int(max_samples))
            for idx in range(n):
                sample = dataset[idx]
                record = _record_from_sample(sample, sample_idx=idx)
                m3_1_pred = m3_1_predictor.predict(sample)
                phys_pred = phys_predictor.predict(sample)
                record["pred_m3_1_increment_surface"] = m3_1_pred["pred_increment_surface"]
                record["pred_m3_1_increment_rootzone"] = m3_1_pred["pred_increment_rootzone"]
                record["pred_phys_coeff_increment_surface"] = phys_pred["pred_increment_surface"]
                record["pred_phys_coeff_increment_rootzone"] = phys_pred["pred_increment_rootzone"]
                yield record
    finally:
        dataset.close()


def cmd_select_eta(args: argparse.Namespace) -> None:
    device = str(resolve_device(args.device, require_gpu=args.require_gpu))
    records = _iter_source_records(
        m3_1_checkpoint=args.m3_1_checkpoint,
        phys_coeff_checkpoint=args.phys_coeff_checkpoint,
        device=device,
        target_region=args.target_region,
        split_type="source_val",
        seed=args.seed,
        K=args.K,
        splits_json=args.splits_json,
        max_samples=args.max_samples,
    )
    eta_grid = [float(item) for item in str(args.eta_grid).split(",") if item.strip()]
    selection = select_eta_from_source_val(
        records,
        eta_grid=eta_grid,
        anchor_dual_cvar=args.anchor_dual_cvar,
        min_dual_cvar_delta=args.min_dual_cvar_delta,
        min_best_variable_rmse_relative_improve=args.min_best_variable_rmse_relative_improve,
        max_other_variable_rmse_relative_degrade=args.max_other_variable_rmse_relative_degrade,
        max_region_rmse_relative_degrade=args.max_region_rmse_relative_degrade,
        max_season_rmse_relative_degrade=args.max_season_rmse_relative_degrade,
    )
    validate_router_metadata_no_target_selection(selection)
    _write_json(args.selection_out, selection)
    print(
        json.dumps(
            {
                "status": "ok",
                "selection_out": args.selection_out,
                "selected_eta_surface": selection["selected_eta_surface"],
                "selected_eta_rootzone": selection["selected_eta_rootzone"],
                "source_gate_pass": selection["source_gate_pass"],
                "identity_diagnostic": selection["identity_diagnostic"],
                "selection_hash": selection["selection_hash"],
                "selection_source": selection["selection_source"],
                "target_eval_usage": selection["target_eval_usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def cmd_write_gate_report(args: argparse.Namespace) -> None:
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    report = source_gate_report_from_selection(
        selection,
        target_region=args.target_region,
        K=args.K,
        seed=args.seed,
        selection_path=args.selection,
    )
    _write_json(args.gate_report_out, report)
    print(json.dumps({"status": "ok", **report}, indent=2, sort_keys=True))


def _selection_from_args(args: argparse.Namespace) -> tuple[float, float, dict[str, Any]]:
    if args.source_gate:
        gate = json.loads(Path(args.source_gate).read_text(encoding="utf-8"))
        if gate.get("schema_version") != M3_15_SOURCE_GATE_REPORT_SCHEMA:
            raise ValueError(f"Unsupported M3_15 source gate report: {gate.get('schema_version')!r}")
        if gate.get("method_id") != M3_15_METHOD_ID:
            raise ValueError("M3_15 source gate report method_id mismatch")
        if args.split_type in {"target_eval", "target_query"} and not bool(gate.get("target_eval_allowed", False)):
            raise ValueError("M3_15 target_eval refused: source_gate.json does not allow target_eval")
    if args.selection:
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        validate_router_metadata_no_target_selection(selection)
        if args.split_type in {"target_eval", "target_query"}:
            validate_source_gate_for_target_eval(selection)
        return (
            float(selection.get("selected_eta_surface", 0.0)),
            float(selection.get("selected_eta_rootzone", selection.get("selected_eta_surface", 0.0))),
            selection,
        )
    if args.split_type in {"target_eval", "target_query"}:
        raise ValueError("M3_15 target_eval requires --selection and --source_gate from source_val")
    eta_surface = float(args.eta)
    return eta_surface, float(args.eta_rootzone if args.eta_rootzone is not None else eta_surface), {}


def cmd_evaluate(args: argparse.Namespace) -> None:
    start = time.time()
    device = str(resolve_device(args.device, require_gpu=args.require_gpu))
    eta_surface, eta_rootzone, selection = _selection_from_args(args)
    target_context_prompt = bool(args.target_context_prompt)
    if args.split_type in {"target_eval", "target_query"} and not args.no_target_context_prompt:
        target_context_prompt = True
    m3_1_predictor = _build_predictor(
        checkpoint=args.m3_1_checkpoint,
        device=device,
        target_region=args.target_region,
        target_context_prompt=target_context_prompt,
        K=args.K,
        seed=args.seed,
        splits_json=args.splits_json,
    )
    phys_predictor = _build_predictor(
        checkpoint=args.phys_coeff_checkpoint,
        device=device,
        target_region=args.target_region,
        target_context_prompt=target_context_prompt,
        K=args.K,
        seed=args.seed,
        splits_json=args.splits_json,
    )
    predictor = M315PhysCoeffDeltaPredictor(
        m3_1_predictor,
        phys_predictor,
        eta_surface=eta_surface,
        eta_rootzone=eta_rootzone,
        selection=selection,
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
            experiment_id=f"{M3_15_METHOD_ID}_{args.target_region}_K{args.K}_S{args.seed}",
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
        df.groupby(["target_region_id", "sample_region_id", "variable", "metric"])["value"]
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
    per_sample_df = df[df["query_date"] != "global"] if "query_date" in df.columns else df
    for variable in ("surface", "rootzone"):
        metric_summary.setdefault(variable, {})
        match = per_sample_df[
            (per_sample_df["variable"] == variable)
            & (per_sample_df["metric"] == "analysis_rmse_latw")
        ]
        metric_summary[variable]["analysis_rmse_latw_mean"] = (
            float(match["value"].mean()) if len(match) else float("nan")
        )
    summary = {
        "method": predictor.method_name,
        "method_id": M3_15_METHOD_ID,
        "base_anchor": "M3_1_hyperda_trust_medium",
        "m3_1_checkpoint": args.m3_1_checkpoint,
        "phys_coeff_checkpoint": args.phys_coeff_checkpoint,
        "target_region": args.target_region,
        "K": args.K,
        "seed": args.seed,
        "split_type": args.split_type,
        "split_file": args.splits_json,
        "split_manifest_sha256": split_manifest_sha256,
        "protocol_freeze_id": ZERO_FEW_SHOT_PROTOCOL_FREEZE_ID,
        "eta_surface": eta_surface,
        "eta_rootzone": eta_rootzone,
        "source_gate_path": args.source_gate,
        "selection": selection,
        "router_metadata": predictor.metadata,
        "output_dir": str(output_dir),
        "target_context_dates_hash": dataset_date_hash(dataset, "target_context_dates_hash"),
        "target_support_dates_hash": dataset_date_hash(dataset, "target_support_dates_hash"),
        "target_eval_dates_hash": dataset_date_hash(dataset, "target_eval_dates_hash"),
        "eta_selection_source": "source_val_only" if selection else "manual_or_eta_zero_source_eval_only",
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": (
            "final_eval_only_no_selection" if args.split_type in {"target_eval", "target_query"} else "not_target_eval"
        ),
        "neural_training_scope": "phys_coeff_delta_only",
        "final_output_residual_allowed": False,
        "prediction_interpolation": "pred_v=(1-eta_v)*pred_m3_1_v+eta_v*pred_phys_coeff_v",
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
            "base_anchor",
            "m3_1_checkpoint",
            "phys_coeff_checkpoint",
            "target_region",
            "K",
            "seed",
            "split_type",
            "eta_surface",
            "eta_rootzone",
            "source_gate_path",
            "target_val_usage",
            "target_eval_usage",
            "prediction_content_hash",
            "metric_content_hash",
            "metric_values_content_hash",
            "prediction_interpolation",
            "final_output_residual_allowed",
        )
    }
    _write_json(output_dir / "diagnostics.json", diagnostics)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "eta_surface": eta_surface,
                "eta_rootzone": eta_rootzone,
                "surface_rmse_latw_mean": summary["surface"].get("analysis_rmse_latw_mean"),
                "rootzone_rmse_latw_mean": summary["rootzone"].get("analysis_rmse_latw_mean"),
                "target_eval_usage": summary["target_eval_usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M3_15 source-safe physics coefficient-delta")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--m3_1_checkpoint", required=True)
        p.add_argument("--phys_coeff_checkpoint", required=True)
        p.add_argument("--target_region", default="US-R1")
        p.add_argument("--K", type=int, default=0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--splits_json", default=ZERO_FEW_SHOT_SPLITS_JSON)
        p.add_argument("--device", default="cuda")
        p.add_argument("--require_gpu", action="store_true")
        p.add_argument("--max_samples", type=int, default=0)

    select = sub.add_parser("select-eta", help="Select interpolation eta pair on source_val")
    add_common(select)
    select.add_argument("--selection_out", required=True)
    select.add_argument("--eta_grid", default="0,0.1,0.25,0.5,1.0")
    select.add_argument("--anchor_dual_cvar", type=float, default=0.446573390549)
    select.add_argument("--min_dual_cvar_delta", type=float, default=0.001)
    select.add_argument("--min_best_variable_rmse_relative_improve", type=float, default=0.001)
    select.add_argument("--max_other_variable_rmse_relative_degrade", type=float, default=0.0005)
    select.add_argument("--max_region_rmse_relative_degrade", type=float, default=0.003)
    select.add_argument("--max_season_rmse_relative_degrade", type=float, default=0.003)
    select.set_defaults(func=cmd_select_eta)

    gate = sub.add_parser("write-gate-report", help="Write source_gate.json from source_val selection")
    gate.add_argument("--target_region", default="US-R1")
    gate.add_argument("--K", type=int, default=0)
    gate.add_argument("--seed", type=int, default=0)
    gate.add_argument("--selection", required=True)
    gate.add_argument("--gate_report_out", required=True)
    gate.set_defaults(func=cmd_write_gate_report)

    evaluate = sub.add_parser("evaluate", help="Evaluate M3_15 source-selected interpolation")
    add_common(evaluate)
    evaluate.add_argument("--selection", default="")
    evaluate.add_argument("--source_gate", default="")
    evaluate.add_argument("--eta", type=float, default=0.0)
    evaluate.add_argument("--eta_rootzone", type=float, default=None)
    evaluate.add_argument("--split_type", default="target_eval")
    evaluate.add_argument("--output_dir", default="artifacts/runs/M3_15_m31_anchored_source_safe_phys_coeff_delta/eval")
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
