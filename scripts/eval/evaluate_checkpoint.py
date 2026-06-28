#!/usr/bin/env python3
"""Evaluate trained neural backbone checkpoints (source-only or prompt-conditioned).

Usage:
    # Source-only backbone
    PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \\
        --checkpoint artifacts/checkpoints/phase4_source_only/US-R1/best.pt \\
        --target_region US-R1 --adaptation_setting zero_shot_context --K 0 --seed 0 \\
        --split_type target_eval --predictor_type source_only

    # Prompt-conditioned shared backbone
    PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \\
        --checkpoint artifacts/checkpoints/phase4_prompt_conditioned/US-R1/best.pt \\
        --target_region US-R1 --adaptation_setting zero_shot_context --K 0 --seed 0 \\
        --split_type target_eval --predictor_type prompt_conditioned

    # Legacy/internal full-target reproduction remains explicit:
    #   --adaptation_setting target_full_train

No-leakage declaration:
    - Evaluation uses target_eval split (target_query alias accepted; post-prediction label use only)
    - No target_eval/target_query labels used in training/normalization/prompt
    - Reuses evaluate_split() from harness.py
    - Metrics computed post-prediction with LeakageGuard protection
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hydroda.baselines.source_only import SourceOnlyBackbonePredictor
from hydroda.data.dataset import HydroDADataset
from hydroda.data.file_hash import compute_sha256
from hydroda.evaluation.harness import (
    evaluate_split,
    metric_rows_content_hash,
    metric_values_content_hash,
    summarize_metric_rows,
)
from scripts.train.train_hyperda_target_adapt import apply_target_adapter_state
from hydroda.training.calibration import calibrate_residual_gain
from hydroda.utils.device import resolve_device


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
ZERO_FEW_SHOT_SPLITS_JSON = "artifacts/splits/US_loro_zero_few_shot_splits.json"
LEGACY_TARGET_TRAIN_SPLITS_JSON = "artifacts/splits/US_loro_target_train_splits.json"
SPLITS_JSON = ZERO_FEW_SHOT_SPLITS_JSON
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
ZERO_FEW_SHOT_PROTOCOL_FREEZE_ID = "hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025"
LEGACY_TARGET_TRAIN_PROTOCOL_FREEZE_ID = "hyperda_v4_3_historical_target_adapt_2015_2025_train2015_2021_val2022_test2023_2025"
PROTOCOL_FREEZE_ID = ZERO_FEW_SHOT_PROTOCOL_FREEZE_ID

_PREDICTOR_OUTPUT_DIRS = {
    "source_only": Path("artifacts/results/phase4_source_only"),
    "prompt_conditioned": Path("artifacts/results/phase4_prompt_conditioned"),
    "hyperda_target_adapt": Path("artifacts/results/phase5_hyperda_target_adapt"),
}


def resolve_split_protocol_for_adaptation(adaptation_setting: str) -> tuple[str, str]:
    """Return split manifest and freeze id for the requested adaptation setting."""
    if adaptation_setting == "target_full_train":
        return LEGACY_TARGET_TRAIN_SPLITS_JSON, LEGACY_TARGET_TRAIN_PROTOCOL_FREEZE_ID
    return ZERO_FEW_SHOT_SPLITS_JSON, ZERO_FEW_SHOT_PROTOCOL_FREEZE_ID


def dataset_date_hash(dataset: HydroDADataset, key: str) -> str:
    """Return a split-date hash recorded on the dataset manifest entry."""
    return str(getattr(dataset, "_split_entry", {}).get(key, ""))


def validate_target_context_prompt_hash(
    *,
    predictor: object,
    dataset: HydroDADataset,
    predictor_type: str,
    split_type: str,
) -> None:
    """Ensure eval uses the same target-context monthly prompt state as adaptation."""
    if predictor_type != "hyperda_target_adapt" or split_type not in ("target_eval", "target_query"):
        return
    split_hash = dataset_date_hash(dataset, "target_context_dates_hash")
    prompt_metadata = dict(getattr(predictor, "_target_prompt_metadata", {}) or {})
    prompt_hash = str(
        prompt_metadata.get("context_hash")
        or prompt_metadata.get("context_date_hash")
        or ""
    )
    if split_hash and prompt_hash and split_hash != prompt_hash:
        raise ValueError(
            "target_context_dates_hash mismatch between eval split manifest and "
            f"checkpoint target_context_prompt_state.context_hash: {split_hash!r} != {prompt_hash!r}"
        )


def aggregate_results(rows):
    """Aggregate metrics by region, season, variable."""
    if not rows:
        return {}
    df = pd.DataFrame(rows)

    by_region = (
        df.groupby(["target_region_id", "variable", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    by_region.columns = ["target_region_id", "variable", "metric", "mean", "std", "count"]

    by_season = (
        df.groupby(["season", "variable", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    by_season.columns = ["season", "variable", "metric", "mean", "std", "count"]

    return {
        "by_region": by_region.to_dict(orient="records"),
        "by_season": by_season.to_dict(orient="records"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate neural backbone checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--target_region", type=str, required=True)
    parser.add_argument("--adaptation_setting", type=str, default=None,
        help="Split adaptation setting. Defaults from K: zero_shot_context/few_shot_k4/few_shot_k12.")
    parser.add_argument("--K", type=int, default=0,
        help="Zero/few-shot K value for the main protocol.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_type", type=str, default="target_eval")
    parser.add_argument(
        "--active_region_override",
        type=str,
        default=None,
        help="Source-side pseudo-query override; valid for source_* splits only.",
    )
    parser.add_argument("--splits_json", type=str, default=None,
        help="Override split manifest. Diagnostics only; defaults from adaptation_setting.")
    parser.add_argument("--max_samples", type=int, default=0,
        help="Max samples to evaluate (0 = no limit, evaluate all)")
    parser.add_argument("--batch_size", type=int, default=8,
        help="Evaluation batch-size metadata. Current predictor path evaluates samples sequentially.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--require_gpu", action="store_true",
        help="Exit with error if CUDA unavailable")
    parser.add_argument("--predictor_type", type=str, default="source_only",
        choices=["source_only", "prompt_conditioned", "hyperda_target_adapt"],
        help="Type of predictor to load")
    parser.add_argument("--output_dir", type=str, default=None,
        help="Override output directory")
    parser.add_argument("--target_context_prompt", action="store_true",
        help="Legacy path for older checkpoints: build target-context monthly prompt prototypes from 2015-2021 input-side fields")
    parser.add_argument("--target_prompt_from_target_train", action="store_true",
        help="Legacy alias for --target_context_prompt")
    parser.add_argument("--no_target_context_prompt", action="store_true",
        help="Ablation: keep checkpoint prompt state/fallback without on-the-fly context construction")
    parser.add_argument("--no_target_prompt_from_target_train", action="store_true",
        help="Legacy alias for --no_target_context_prompt")
    parser.add_argument("--target_train_residual_gain_calibration", action="store_true",
        help="Legacy/internal: calibrate residual gain on target support/train labels")
    parser.add_argument("--allow_legacy_target_label_calibration", action="store_true",
        help="Required with --target_train_residual_gain_calibration.")
    parser.add_argument(
        "--adapt_mix_rho",
        type=float,
        default=1.0,
        help="Fixed output mixture with same-context K0 prediction; diagnostic only, never selected on target_eval.",
    )
    parser.add_argument(
        "--eval_raw_adapted_before_mix",
        action="store_true",
        help="Diagnostic Stage 3 eval: hash raw-adapted predictions from raw_adapted_state_dict before gate/mix.",
    )
    parser.add_argument(
        "--prediction_record_path",
        type=str,
        default="",
        help="Optional JSONL path for source-safe prediction records used by offline calibration mixing.",
    )
    parser.add_argument(
        "--output_level",
        type=str,
        default=os.environ.get("EVAL_OUTPUT_LEVEL", "compact"),
        choices=["compact", "long", "full"],
        help=(
            "Artifact volume profile: compact writes summary and aggregate CSVs, "
            "long also writes metrics_long.csv.gz, full writes metrics_long.csv "
            "and honors --prediction_record_path."
        ),
    )
    args = parser.parse_args()

    if args.adaptation_setting is None:
        args.adaptation_setting = "zero_shot_context" if int(args.K) == 0 else f"few_shot_k{int(args.K)}"
    if args.adaptation_setting == "target_full_train":
        args.K = None
    splits_json, protocol_freeze_id = resolve_split_protocol_for_adaptation(args.adaptation_setting)
    if args.splits_json:
        splits_json = args.splits_json
    if args.target_prompt_from_target_train:
        args.target_context_prompt = True
    if args.no_target_prompt_from_target_train:
        args.no_target_context_prompt = True
    if args.target_train_residual_gain_calibration and not args.allow_legacy_target_label_calibration:
        raise ValueError(
            "--target_train_residual_gain_calibration is legacy/internal. "
            "Pass --allow_legacy_target_label_calibration to opt in explicitly."
        )
    if not 0.0 <= float(args.adapt_mix_rho) <= 1.0:
        raise ValueError("--adapt_mix_rho must be in [0, 1]")

    if (
        args.predictor_type in ("prompt_conditioned", "hyperda_target_adapt")
        and args.split_type in ("target_eval", "target_query")
        and args.predictor_type != "hyperda_target_adapt"
        and not args.no_target_context_prompt
    ):
        args.target_context_prompt = True

    # Resolve device
    device = resolve_device(args.device, require_gpu=args.require_gpu)

    ckpt_path = Path(args.checkpoint)

    # Determine output directory
    if args.output_dir:
        region_output_dir = Path(args.output_dir) / args.target_region
    else:
        base_dir = _PREDICTOR_OUTPUT_DIRS.get(args.predictor_type, Path("artifacts/results/phase4"))
        region_output_dir = base_dir / args.target_region
    region_output_dir.mkdir(parents=True, exist_ok=True)

    if args.predictor_type == "source_only":
        phase_label = "Phase 4A"
    elif args.predictor_type == "hyperda_target_adapt":
        phase_label = "Phase 5"
    else:
        phase_label = "Phase 4B"
    print("=" * 60)
    print(f"{phase_label}: Neural Backbone Evaluation")
    print(f"  predictor_type={args.predictor_type}")
    print(f"  checkpoint={ckpt_path}")
    print(f"  target_region={args.target_region}  adaptation_setting={args.adaptation_setting}  K={args.K}  seed={args.seed}")
    print(f"  split_type={args.split_type}  max_samples={args.max_samples if args.max_samples > 0 else 'all'}")
    print(f"  eval_batch_size={args.batch_size} (metadata; sample-wise predictor path)")
    print(f"  device={device}")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset ({args.split_type})...")
    dataset = HydroDADataset(
        da_nc_path=f"{DATA_DIR}/DA.nc",
        region_masks_nc=REGION_MASKS_NC,
        splits_json=splits_json,
        target_region=args.target_region,
        split_type=args.split_type,
        K=args.K,
        seed=args.seed,
        adaptation_setting=args.adaptation_setting,
        freeze_manifest=FREEZE_MANIFEST,
    )
    if args.active_region_override:
        if args.split_type not in ("source_train", "source_fit", "source_val", "source_test"):
            raise ValueError("--active_region_override is valid only for source_* splits")
        dataset.set_active_region(args.active_region_override)
        print(f"  active_region_override={args.active_region_override} (source-side pseudo-query)")

    total_samples = len(dataset)
    n_samples = min(total_samples, args.max_samples) if args.max_samples > 0 else total_samples
    print(f"  dataset size: {total_samples}, evaluating {n_samples} samples")

    # Load predictor
    print(f"\nLoading checkpoint...")
    zero_shot_predictor = None
    raw_adapted_predictor = None
    no_tta_predictor = None
    no_tta_zero_shot_predictor = None
    if args.predictor_type in ("prompt_conditioned", "hyperda_target_adapt"):
        from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

        predictor = PromptConditionedBackbonePredictor(
            checkpoint_path=str(ckpt_path),
            device=str(device),
            target_region=args.target_region,
        )
        raw_checkpoint = None
        context_tta_effective_for_comparator = bool(
            predictor.stage3_protocol_metadata.get("context_tta_effective", False)
            or getattr(predictor, "_target_prompt_metadata", {}).get("context_tta_effective", False)
        )
        if context_tta_effective_for_comparator:
            try:
                no_tta_predictor = PromptConditionedBackbonePredictor(
                    checkpoint_path=str(ckpt_path),
                    device=str(device),
                    target_region=args.target_region,
                )
                no_tta_predictor.load_no_tta_target_context_prompt_state_from_current()
            except (RuntimeError, ValueError) as exc:
                print(f"  no-TTA comparator unavailable for this TTA mode: {exc}")
                no_tta_predictor = None
        if args.predictor_type == "hyperda_target_adapt" and int(args.K or 0) > 0:
            zero_shot_predictor = PromptConditionedBackbonePredictor(
                checkpoint_path=str(ckpt_path),
                device=str(device),
                target_region=args.target_region,
                apply_support_affine_calibration=False,
            )
            import torch

            raw_checkpoint = torch.load(ckpt_path, map_location=str(device), weights_only=False)
            anchor_state = raw_checkpoint.get("target_adapter_anchor_state")
            if not anchor_state:
                raise ValueError(
                    "K-shot hyperda_target_adapt evaluation requires checkpoint "
                    "target_adapter_anchor_state to reconstruct same-context K0 "
                    "predictions for Stage 3 hash/delta diagnostics."
                )
            apply_target_adapter_state(zero_shot_predictor.model, anchor_state)
            if no_tta_predictor is not None:
                no_tta_zero_shot_predictor = PromptConditionedBackbonePredictor(
                    checkpoint_path=str(ckpt_path),
                    device=str(device),
                    target_region=args.target_region,
                    apply_support_affine_calibration=False,
                )
                no_tta_zero_shot_predictor.load_no_tta_target_context_prompt_state_from_current()
                apply_target_adapter_state(no_tta_zero_shot_predictor.model, anchor_state)
        if args.predictor_type == "hyperda_target_adapt" and args.eval_raw_adapted_before_mix:
            raw_adapted_predictor = PromptConditionedBackbonePredictor(
                checkpoint_path=str(ckpt_path),
                device=str(device),
                target_region=args.target_region,
            )
            import torch

            if raw_checkpoint is None:
                raw_checkpoint = torch.load(ckpt_path, map_location=str(device), weights_only=False)
            raw_state = (raw_checkpoint.get("raw_adapted_state_dict") or {}).get("target_adapter_state_dict")
            if not raw_state:
                raise ValueError(
                    "--eval_raw_adapted_before_mix requires checkpoint raw_adapted_state_dict.target_adapter_state_dict"
                )
            apply_target_adapter_state(raw_adapted_predictor.model, raw_state)
        target_context_dataset = None
        target_train_dataset = None
        if args.target_context_prompt:
            if args.split_type not in ("target_eval", "target_query"):
                raise ValueError(
                    "--target_context_prompt is only valid for target_eval/target_query"
                )
            target_context_dataset = HydroDADataset(
                da_nc_path=f"{DATA_DIR}/DA.nc",
                region_masks_nc=REGION_MASKS_NC,
                splits_json=splits_json,
                target_region=args.target_region,
                split_type="target_context",
                K=args.K,
                seed=args.seed,
                adaptation_setting=args.adaptation_setting,
                freeze_manifest=FREEZE_MANIFEST,
            )
        if args.target_train_residual_gain_calibration:
            if args.split_type not in ("target_eval", "target_query"):
                raise ValueError(
                    "--target_train_residual_gain_calibration is only valid for target_eval/target_query"
                )
            target_train_dataset = HydroDADataset(
                da_nc_path=f"{DATA_DIR}/DA.nc",
                region_masks_nc=REGION_MASKS_NC,
                splits_json=splits_json,
                target_region=args.target_region,
                split_type="target_support",
                K=args.K,
                seed=args.seed,
                adaptation_setting=args.adaptation_setting,
                freeze_manifest=FREEZE_MANIFEST,
            )

        if args.target_context_prompt:
            print("  Building target-context monthly prompt prototypes from target_context inputs...")
            prompt_metadata = predictor.set_target_context_prompt_from_samples(
                target_context_dataset.get_input_side_sample(i)
                for i in range(len(target_context_dataset))
            )
            if no_tta_predictor is not None:
                no_tta_predictor.set_target_context_prompt_from_samples(
                    (
                        target_context_dataset.get_input_side_sample(i)
                        for i in range(len(target_context_dataset))
                    ),
                    context_tta_mode="none",
                )
            if no_tta_zero_shot_predictor is not None:
                no_tta_zero_shot_predictor.set_target_context_prompt_from_samples(
                    (
                        target_context_dataset.get_input_side_sample(i)
                        for i in range(len(target_context_dataset))
                    ),
                    context_tta_mode="none",
                )
            print(
                "  target-context prompt state: "
                f"n={prompt_metadata['n_samples']} "
                f"dates={prompt_metadata['date_start']}..{prompt_metadata['date_end']} "
                f"labels={prompt_metadata['label_usage']}"
            )

        target_train_calibration = {}
        if args.target_train_residual_gain_calibration:
            print("  Calibrating residual gain on legacy target support labels...")
            samples_s = []
            samples_r = []
            alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
            for i in range(len(target_train_dataset)):
                sample = target_train_dataset[i]
                pred = predictor.predict(sample)
                mask = sample["metric_mask"]
                latw = sample.get("latitude_weight")
                if latw is None:
                    latw = np.ones(mask.shape, dtype=np.float32)
                samples_s.append((
                    pred["pred_increment_surface"],
                    sample["increment_surface"],
                    sample["forecast_surface"],
                    mask,
                    latw,
                ))
                samples_r.append((
                    pred["pred_increment_rootzone"],
                    sample["increment_rootzone"],
                    sample["forecast_rootzone"],
                    mask,
                    latw,
                ))
            target_train_calibration = calibrate_residual_gain(samples_s, samples_r, alpha_grid)
            if target_train_calibration:
                predictor.alpha_surface = target_train_calibration["best_alpha_surface"]
                predictor.alpha_rootzone = target_train_calibration["best_alpha_rootzone"]
                predictor.apply_residual_gain = True
                print(
                    "  target_train alphas: "
                    f"surface={predictor.alpha_surface:.3f} "
                    f"rootzone={predictor.alpha_rootzone:.3f}"
                )

        if target_context_dataset is not None:
            target_context_dataset.close()
        if target_train_dataset is not None:
            target_train_dataset.close()
    else:
        predictor = SourceOnlyBackbonePredictor(
            checkpoint_path=str(ckpt_path),
            device=str(device),
        )
    print(f"  method: {predictor.method_name}")
    validate_target_context_prompt_hash(
        predictor=predictor,
        dataset=dataset,
        predictor_type=args.predictor_type,
        split_type=args.split_type,
    )

    # Run evaluation
    print(f"\nRunning evaluation...")
    start_time = time.time()

    split_manifest_sha256 = compute_sha256(splits_json) if Path(splits_json).exists() else ""
    experiment_suffix = args.adaptation_setting if args.K is None else f"{args.adaptation_setting}_K{args.K}"
    eval_kwargs = {
        "split_role": args.split_type,
        "experiment_id": f"phase4_{args.predictor_type}_{args.target_region}_{experiment_suffix}_S{args.seed}",
        "protocol_freeze_id": protocol_freeze_id,
        "method": predictor.method_name,
        "split_file": splits_json,
        "mask_file": REGION_MASKS_NC,
        "target_context_dates_hash": dataset_date_hash(dataset, "target_context_dates_hash"),
        "target_support_dates_hash": dataset_date_hash(dataset, "target_support_dates_hash"),
        "support_dates_hash": dataset_date_hash(dataset, "support_dates_hash"),
        "target_train_dates_hash": dataset_date_hash(dataset, "target_train_dates_hash"),
        "target_eval_dates_hash": dataset_date_hash(dataset, "target_eval_dates_hash"),
        "split_manifest_sha256": split_manifest_sha256,
        "preloaded": False,
        "max_samples": args.max_samples if args.max_samples > 0 else None,
    }
    rows = []
    n_samples_effective = n_samples
    if args.predictor_type in ("prompt_conditioned", "hyperda_target_adapt") and args.split_type == "source_test":
        source_regions = getattr(predictor, "source_regions", [])
        if not source_regions:
            raise ValueError("Prompt-conditioned source_test requires source_regions metadata in checkpoint")
        n_samples_effective = n_samples * len(source_regions)
        for source_region in source_regions:
            print(f"  source_test active source region: {source_region}")
            dataset.set_active_region(source_region)
            rows.extend(evaluate_split(dataset=dataset, predictor=predictor, **eval_kwargs))
        dataset.set_active_all_regions()
        eval_hashes = {
            "prediction_content_hash": "",
            "prediction_record_count": n_samples_effective,
            "metric_content_hash": metric_rows_content_hash(rows),
            "metric_values_content_hash": metric_values_content_hash(rows),
            "metric_row_count": len(rows),
        }
    else:
        rows, eval_hashes = evaluate_split(
            dataset=dataset,
            predictor=predictor,
            return_hashes=True,
            zero_shot_predictor=zero_shot_predictor,
            raw_adapted_predictor=raw_adapted_predictor,
            no_tta_predictor=no_tta_predictor,
            no_tta_zero_shot_predictor=no_tta_zero_shot_predictor,
            adapt_mix_rho=float(args.adapt_mix_rho),
            prediction_record_path=(
                args.prediction_record_path or None
                if args.output_level == "full"
                else None
            ),
            **eval_kwargs,
        )

    elapsed = time.time() - start_time
    print(f"  Evaluation done in {elapsed:.1f}s — {len(rows)} metric rows")

    dataset.close()

    if not rows:
        print("  WARNING: No rows generated. Check dataset/predictor.")
        summary = {"status": "no_rows", "checkpoint": str(ckpt_path)}
        with open(region_output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return

    # Save long-form results according to the selected artifact volume profile.
    df = pd.DataFrame(rows)
    long_path = region_output_dir / "metrics_long.csv"
    long_gz_path = region_output_dir / "metrics_long.csv.gz"
    if args.output_level == "full":
        df.to_csv(long_path, index=False)
        if long_gz_path.exists():
            long_gz_path.unlink()
        print(f"  Saved {len(rows)} rows to {long_path}")
    elif args.output_level == "long":
        df.to_csv(long_gz_path, index=False, compression="gzip")
        if long_path.exists():
            long_path.unlink()
        print(f"  Saved {len(rows)} compressed rows to {long_gz_path}")
    else:
        if long_path.exists():
            long_path.unlink()
        if long_gz_path.exists():
            long_gz_path.unlink()
        print("  EVAL_OUTPUT_LEVEL=compact: skipped metrics_long.csv")

    # Aggregate
    agg = aggregate_results(rows)

    if agg.get("by_region"):
        by_region_df = pd.DataFrame(agg["by_region"])
        by_region_path = region_output_dir / "metrics_by_region.csv"
        by_region_df.to_csv(by_region_path, index=False)

    if agg.get("by_season"):
        by_season_df = pd.DataFrame(agg["by_season"])
        by_season_path = region_output_dir / "metrics_by_season.csv"
        by_season_df.to_csv(by_season_path, index=False)

    metric_summary = summarize_metric_rows(df)
    stage3_protocol_metadata = dict(getattr(predictor, "stage3_protocol_metadata", {}) or {})
    target_prompt_metadata = dict(getattr(predictor, "_target_prompt_metadata", {}) or {})
    prompt_l2_delta_mean = float(
        target_prompt_metadata.get(
            "prompt_l2_delta_mean",
            stage3_protocol_metadata.get("prompt_l2_delta_mean", 0.0),
        )
        or 0.0
    )
    prediction_delta_vs_no_tta = float(
        eval_hashes.get(
            "prediction_delta_vs_no_tta",
            stage3_protocol_metadata.get("prediction_delta_vs_no_tta", 0.0),
        )
        or 0.0
    )
    prediction_max_abs_delta_vs_no_tta = float(
        eval_hashes.get("prediction_max_abs_delta_vs_no_tta", 0.0) or 0.0
    )

    summary = {
        "method": predictor.method_name,
        "checkpoint": str(ckpt_path),
        "target_region": args.target_region,
        "adaptation_setting": args.adaptation_setting,
        "K": args.K,
        "seed": args.seed,
        "split_type": args.split_type,
        "split_file": splits_json,
        "n_samples_evaluated": n_samples_effective,
        "n_metric_rows": len(rows),
        "eval_batch_size": args.batch_size,
        "output_level": args.output_level,
        "protocol_freeze_id": protocol_freeze_id,
        "split_manifest_sha256": split_manifest_sha256,
        "target_context_dates_hash": dataset_date_hash(dataset, "target_context_dates_hash"),
        "target_support_dates_hash": dataset_date_hash(dataset, "target_support_dates_hash"),
        "support_dates_hash": dataset_date_hash(dataset, "support_dates_hash"),
        "target_train_dates_hash": dataset_date_hash(dataset, "target_train_dates_hash"),
        "target_eval_dates_hash": dataset_date_hash(dataset, "target_eval_dates_hash"),
        "target_prompt": target_prompt_metadata,
        "stage3_protocol": stage3_protocol_metadata,
        "context_tta_effective": bool(
            stage3_protocol_metadata.get(
                "context_tta_effective",
                target_prompt_metadata.get("context_tta_effective", False),
            )
        ),
        "context_tta_source_stat_status": stage3_protocol_metadata.get(
            "context_tta_source_stat_status",
            target_prompt_metadata.get("context_tta_source_stat_status", "not_requested"),
        ),
        "prompt_l2_delta_mean": prompt_l2_delta_mean,
        "prediction_delta_vs_no_tta": prediction_delta_vs_no_tta,
        "prediction_max_abs_delta_vs_no_tta": prediction_max_abs_delta_vs_no_tta,
        "target_train_residual_gain_calibration": target_train_calibration if args.predictor_type in ("prompt_conditioned", "hyperda_target_adapt") else {},
        "prediction_content_hash": eval_hashes.get("prediction_content_hash", ""),
        "prediction_record_count": eval_hashes.get("prediction_record_count", 0),
        "prediction_records_written": bool(args.output_level == "full" and args.prediction_record_path),
        "metric_content_hash": eval_hashes.get("metric_content_hash", ""),
        "metric_row_content_hash": eval_hashes.get("metric_content_hash", ""),
        "metric_values_content_hash": eval_hashes.get("metric_values_content_hash", ""),
        "metric_hash_source": "in_memory_metric_rows_before_csv_write",
        "adapt_mix_rho": float(args.adapt_mix_rho),
        "eval_raw_adapted_before_mix": bool(args.eval_raw_adapted_before_mix),
        "zero_shot_prediction_content_hash": eval_hashes.get("zero_shot_prediction_content_hash", ""),
        "adapted_pre_mix_prediction_content_hash": eval_hashes.get("adapted_pre_mix_prediction_content_hash", ""),
        "raw_adapted_prediction_content_hash": eval_hashes.get("raw_adapted_prediction_content_hash", ""),
        "post_gate_prediction_content_hash": eval_hashes.get("post_gate_prediction_content_hash", ""),
        "final_mixed_prediction_content_hash": eval_hashes.get("final_mixed_prediction_content_hash", ""),
        "no_tta_prediction_content_hash": eval_hashes.get("no_tta_prediction_content_hash", ""),
        "raw_to_k0_mean_abs_delta": eval_hashes.get("raw_to_k0_mean_abs_delta", 0.0),
        "post_gate_to_k0_mean_abs_delta": eval_hashes.get("post_gate_to_k0_mean_abs_delta", 0.0),
        "final_mix_to_k0_mean_abs_delta": eval_hashes.get("final_mix_to_k0_mean_abs_delta", 0.0),
        "mix_mean_abs_change_from_k0": eval_hashes.get("mix_mean_abs_change_from_k0", 0.0),
        "mix_max_abs_change_from_k0": eval_hashes.get("mix_max_abs_change_from_k0", 0.0),
        "mix_mean_abs_change_from_adapted": eval_hashes.get("mix_mean_abs_change_from_adapted", 0.0),
        "mix_max_abs_change_from_adapted": eval_hashes.get("mix_max_abs_change_from_adapted", 0.0),
        "surface": metric_summary.get("surface", {}),
        "rootzone": metric_summary.get("rootzone", {}),
        "eval_time_s": elapsed,
    }

    summary_path = region_output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to {summary_path}")
    print(f"\n  Surface  skill primary global   ={summary['surface']['skill_primary']:.10f}")
    print(f"  Surface  skill primary latw     ={summary['surface']['skill_latw_primary']:.10f}")
    print(f"  Surface  skill per-sample med   ={summary['surface']['skill_median']:.10f}")
    print(f"  Rootzone skill primary global   ={summary['rootzone']['skill_primary']:.10f}")
    print(f"  Rootzone skill primary latw     ={summary['rootzone']['skill_latw_primary']:.10f}")
    print(f"  Rootzone skill per-sample med   ={summary['rootzone']['skill_median']:.10f}")
    print(f"  Surface  WRMSE={summary['surface']['rmse_latw_mean']:.10f}")
    print(f"  Rootzone WRMSE={summary['rootzone']['rmse_latw_mean']:.10f}")
    print(f"  Surface  Corr_latw={summary['surface']['corr_latw_mean']:.10f}")
    print(f"  Rootzone Corr_latw={summary['rootzone']['corr_latw_mean']:.10f}")

    # Diagnostics
    diagnostics = {
        "checkpoint": str(ckpt_path),
        "target_region": args.target_region,
        "adaptation_setting": args.adaptation_setting,
        "split_type": args.split_type,
        "split_file": splits_json,
        "predictor_type": args.predictor_type,
        "n_samples_total": total_samples,
        "n_samples_evaluated": n_samples_effective,
        "n_metric_rows": len(rows),
        "eval_batch_size": args.batch_size,
        "output_level": args.output_level,
        "target_context_dates_hash": dataset_date_hash(dataset, "target_context_dates_hash"),
        "target_support_dates_hash": dataset_date_hash(dataset, "target_support_dates_hash"),
        "target_eval_dates_hash": dataset_date_hash(dataset, "target_eval_dates_hash"),
        "split_manifest_sha256": split_manifest_sha256,
        "metrics_computed": sorted(df["metric"].unique().tolist()),
        "variables": sorted(df["variable"].unique().tolist()),
        "seasonal_breakdown": sorted(df["season"].unique().tolist()) if "season" in df.columns else [],
        "stage3_protocol": stage3_protocol_metadata,
        "context_tta_effective": bool(
            stage3_protocol_metadata.get(
                "context_tta_effective",
                target_prompt_metadata.get("context_tta_effective", False),
            )
        ),
        "context_tta_source_stat_status": stage3_protocol_metadata.get(
            "context_tta_source_stat_status",
            target_prompt_metadata.get("context_tta_source_stat_status", "not_requested"),
        ),
        "prompt_l2_delta_mean": prompt_l2_delta_mean,
        "prediction_delta_vs_no_tta": prediction_delta_vs_no_tta,
        "prediction_max_abs_delta_vs_no_tta": prediction_max_abs_delta_vs_no_tta,
        "prediction_content_hash": eval_hashes.get("prediction_content_hash", ""),
        "prediction_record_count": eval_hashes.get("prediction_record_count", 0),
        "prediction_records_written": bool(args.output_level == "full" and args.prediction_record_path),
        "metric_content_hash": eval_hashes.get("metric_content_hash", ""),
        "metric_values_content_hash": eval_hashes.get("metric_values_content_hash", ""),
        "metric_hash_source": "in_memory_metric_rows_before_csv_write",
        "adapt_mix_rho": float(args.adapt_mix_rho),
        "eval_raw_adapted_before_mix": bool(args.eval_raw_adapted_before_mix),
        "zero_shot_prediction_content_hash": eval_hashes.get("zero_shot_prediction_content_hash", ""),
        "adapted_pre_mix_prediction_content_hash": eval_hashes.get("adapted_pre_mix_prediction_content_hash", ""),
        "raw_adapted_prediction_content_hash": eval_hashes.get("raw_adapted_prediction_content_hash", ""),
        "post_gate_prediction_content_hash": eval_hashes.get("post_gate_prediction_content_hash", ""),
        "final_mixed_prediction_content_hash": eval_hashes.get("final_mixed_prediction_content_hash", ""),
        "no_tta_prediction_content_hash": eval_hashes.get("no_tta_prediction_content_hash", ""),
        "raw_to_k0_mean_abs_delta": eval_hashes.get("raw_to_k0_mean_abs_delta", 0.0),
        "post_gate_to_k0_mean_abs_delta": eval_hashes.get("post_gate_to_k0_mean_abs_delta", 0.0),
        "final_mix_to_k0_mean_abs_delta": eval_hashes.get("final_mix_to_k0_mean_abs_delta", 0.0),
        "mix_mean_abs_change_from_k0": eval_hashes.get("mix_mean_abs_change_from_k0", 0.0),
        "mix_max_abs_change_from_k0": eval_hashes.get("mix_max_abs_change_from_k0", 0.0),
        "mix_mean_abs_change_from_adapted": eval_hashes.get("mix_mean_abs_change_from_adapted", 0.0),
        "mix_max_abs_change_from_adapted": eval_hashes.get("mix_max_abs_change_from_adapted", 0.0),
    }
    diag_path = region_output_dir / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"{phase_label} Evaluation Complete")
    print(f"  Output: {region_output_dir}/")
    print(f"  output_level={args.output_level}")
    print(f"  metrics_by_region.csv | metrics_by_season.csv")
    print(f"  summary.json | diagnostics.json")


if __name__ == "__main__":
    main()
