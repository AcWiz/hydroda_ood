#!/usr/bin/env python3
"""Evaluate trained neural backbone checkpoints (source-only or prompt-conditioned).

Usage:
    # Source-only backbone
    PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \\
        --checkpoint artifacts/checkpoints/phase4_source_only/US-R1/best.pt \\
        --target_region US-R1 --adaptation_setting target_full_train --seed 0 \\
        --split_type target_eval --predictor_type source_only

    # Prompt-conditioned shared backbone
    PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \\
        --checkpoint artifacts/checkpoints/phase4_prompt_conditioned/US-R1/best.pt \\
        --target_region US-R1 --adaptation_setting target_full_train --seed 0 \\
        --split_type target_eval --predictor_type prompt_conditioned

No-leakage declaration:
    - Evaluation uses target_eval split (target_query alias accepted; post-prediction label use only)
    - No target_eval/target_query labels used in training/normalization/prompt
    - Reuses evaluate_split() from harness.py
    - Metrics computed post-prediction with LeakageGuard protection
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hydroda.baselines.source_only import SourceOnlyBackbonePredictor
from hydroda.data.dataset import HydroDADataset
from hydroda.data.file_hash import compute_sha256
from hydroda.evaluation.harness import evaluate_split, summarize_metric_rows
from hydroda.training.calibration import calibrate_residual_gain
from hydroda.utils.device import resolve_device


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_target_train_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
PROTOCOL_FREEZE_ID = "hyperda_v4_3_historical_target_adapt_2015_2025_train2015_2021_val2022_test2023_2025"

_PREDICTOR_OUTPUT_DIRS = {
    "source_only": Path("artifacts/results/phase4_source_only"),
    "prompt_conditioned": Path("artifacts/results/phase4_prompt_conditioned"),
    "hyperda_target_adapt": Path("artifacts/results/phase5_hyperda_target_adapt"),
}


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
    parser.add_argument("--adaptation_setting", type=str, default="target_full_train",
        help="Split adaptation setting (default: target_full_train; legacy example: legacy_few_shot_k4)")
    parser.add_argument("--K", type=int, default=None,
        help="Legacy few-shot K value. Ignored for target_full_train.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_type", type=str, default="target_eval")
    parser.add_argument("--max_samples", type=int, default=0,
        help="Max samples to evaluate (0 = no limit, evaluate all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--require_gpu", action="store_true",
        help="Exit with error if CUDA unavailable")
    parser.add_argument("--predictor_type", type=str, default="source_only",
        choices=["source_only", "prompt_conditioned", "hyperda_target_adapt"],
        help="Type of predictor to load")
    parser.add_argument("--output_dir", type=str, default=None,
        help="Override output directory")
    parser.add_argument("--target_prompt_from_target_train", action="store_true",
        help="For prompt-conditioned target_eval, build a fixed prompt from target_train input-side fields only")
    parser.add_argument("--no_target_prompt_from_target_train", action="store_true",
        help="Ablation: for prompt-conditioned target_eval, keep the checkpoint fallback/mean target prompt")
    parser.add_argument("--target_train_residual_gain_calibration", action="store_true",
        help="For prompt-conditioned target_eval, calibrate residual gain on target_train labels only")
    args = parser.parse_args()

    if args.adaptation_setting == "target_full_train":
        args.K = None
    elif args.K is None:
        args.K = 0

    if (
        args.predictor_type in ("prompt_conditioned", "hyperda_target_adapt")
        and args.split_type in ("target_eval", "target_query")
        and args.predictor_type != "hyperda_target_adapt"
        and not args.no_target_prompt_from_target_train
    ):
        args.target_prompt_from_target_train = True

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
    print(f"  device={device}")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset ({args.split_type})...")
    dataset = HydroDADataset(
        da_nc_path=f"{DATA_DIR}/DA.nc",
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region=args.target_region,
        split_type=args.split_type,
        K=args.K,
        seed=args.seed,
        adaptation_setting=args.adaptation_setting,
        freeze_manifest=FREEZE_MANIFEST,
    )

    total_samples = len(dataset)
    n_samples = min(total_samples, args.max_samples) if args.max_samples > 0 else total_samples
    print(f"  dataset size: {total_samples}, evaluating {n_samples} samples")

    # Load predictor
    print(f"\nLoading checkpoint...")
    if args.predictor_type in ("prompt_conditioned", "hyperda_target_adapt"):
        from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

        predictor = PromptConditionedBackbonePredictor(
            checkpoint_path=str(ckpt_path),
            device=str(device),
            target_region=args.target_region,
        )
        target_train_dataset = None
        if args.target_prompt_from_target_train or args.target_train_residual_gain_calibration:
            if args.split_type not in ("target_eval", "target_query"):
                raise ValueError(
                    "--target_prompt_from_target_train and "
                    "--target_train_residual_gain_calibration are only valid for target_eval/target_query"
                )
            target_train_dataset = HydroDADataset(
                da_nc_path=f"{DATA_DIR}/DA.nc",
                region_masks_nc=REGION_MASKS_NC,
                splits_json=SPLITS_JSON,
                target_region=args.target_region,
                split_type="target_train",
                K=args.K,
                seed=args.seed,
                adaptation_setting=args.adaptation_setting,
                freeze_manifest=FREEZE_MANIFEST,
            )

        if args.target_prompt_from_target_train:
            print("  Building fixed target prompt from target_train inputs...")
            prompt_metadata = predictor.set_target_prompt_from_samples(
                target_train_dataset[i] for i in range(len(target_train_dataset))
            )
            print(
                "  target prompt: "
                f"n={prompt_metadata['n_samples']} "
                f"dates={prompt_metadata['date_start']}..{prompt_metadata['date_end']} "
                f"labels={prompt_metadata['label_usage']}"
            )

        target_train_calibration = {}
        if args.target_train_residual_gain_calibration:
            print("  Calibrating residual gain on target_train labels...")
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

        if target_train_dataset is not None:
            target_train_dataset.close()
    else:
        predictor = SourceOnlyBackbonePredictor(
            checkpoint_path=str(ckpt_path),
            device=str(device),
        )
    print(f"  method: {predictor.method_name}")

    # Run evaluation
    print(f"\nRunning evaluation...")
    start_time = time.time()

    split_manifest_sha256 = compute_sha256(SPLITS_JSON) if Path(SPLITS_JSON).exists() else ""
    experiment_suffix = args.adaptation_setting if args.K is None else f"{args.adaptation_setting}_K{args.K}"
    eval_kwargs = {
        "split_role": args.split_type,
        "experiment_id": f"phase4_{args.predictor_type}_{args.target_region}_{experiment_suffix}_S{args.seed}",
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "method": predictor.method_name,
        "split_file": SPLITS_JSON,
        "mask_file": REGION_MASKS_NC,
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
    else:
        rows = evaluate_split(dataset=dataset, predictor=predictor, **eval_kwargs)

    elapsed = time.time() - start_time
    print(f"  Evaluation done in {elapsed:.1f}s — {len(rows)} metric rows")

    dataset.close()

    if not rows:
        print("  WARNING: No rows generated. Check dataset/predictor.")
        summary = {"status": "no_rows", "checkpoint": str(ckpt_path)}
        with open(region_output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return

    # Save long-form results
    df = pd.DataFrame(rows)
    long_path = region_output_dir / "metrics_long.csv"
    df.to_csv(long_path, index=False)
    print(f"  Saved {len(rows)} rows to {long_path}")

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

    summary = {
        "method": predictor.method_name,
        "checkpoint": str(ckpt_path),
        "target_region": args.target_region,
        "adaptation_setting": args.adaptation_setting,
        "K": args.K,
        "seed": args.seed,
        "split_type": args.split_type,
        "n_samples_evaluated": n_samples_effective,
        "n_metric_rows": len(rows),
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "split_manifest_sha256": split_manifest_sha256,
        "target_prompt": getattr(predictor, "_target_prompt_metadata", {}),
        "target_train_residual_gain_calibration": target_train_calibration if args.predictor_type in ("prompt_conditioned", "hyperda_target_adapt") else {},
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
        "predictor_type": args.predictor_type,
        "n_samples_total": total_samples,
        "n_samples_evaluated": n_samples_effective,
        "n_metric_rows": len(rows),
        "metrics_computed": sorted(df["metric"].unique().tolist()),
        "variables": sorted(df["variable"].unique().tolist()),
        "seasonal_breakdown": sorted(df["season"].unique().tolist()) if "season" in df.columns else [],
    }
    diag_path = region_output_dir / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"{phase_label} Evaluation Complete")
    print(f"  Output: {region_output_dir}/")
    print(f"  metrics_long.csv | metrics_by_region.csv | metrics_by_season.csv")
    print(f"  summary.json | diagnostics.json")


if __name__ == "__main__":
    main()
