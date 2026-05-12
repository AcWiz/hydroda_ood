#!/usr/bin/env python3
"""Train source-only SmallResUNet backbone with full experiment infrastructure.

Usage:
    PYTHONPATH=. python scripts/train/train_source_only_backbone.py \\
        --target_region US-R1 --K 0 --seed 0 \\
        --max_epochs 30 --batch_size 4 --lr 1e-3 \\
        --device cuda --amp \\
        --wandb_mode disabled \\
        --config configs/model_resunet_main.yaml

No-leakage declaration:
    - Only source_train split used for training
    - Normalization stats from source_train only
    - No target_query labels used in training/normalization/early_stopping
    - No target prompt used
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml
import torch

from hydroda.data.dataset import HydroDADataset
from hydroda.models.resunet import SmallResUNet
from hydroda.training.trainer import Trainer
from hydroda.utils.run_manager import RunManager
from hydroda.utils.logger import WandbLogger, ConsoleLogger, JSONLLogger
from hydroda.utils.device import resolve_device, log_device_summary
from hydroda.utils.runtime import gather_runtime_info


DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
CHECKPOINT_DIR = "artifacts/checkpoints/phase4_source_only"
PROTOCOL_FREEZE_ID = "hyperda_v4_final_2015_2025_context2022_query2023_2025_k0_4_12"
PHASE = "phase4_source_only"


def parse_args():
    parser = argparse.ArgumentParser(description="Train source-only backbone with experiment tracking")
    # Data
    parser.add_argument("--target_region", type=str, required=True, help="Target region, e.g. US-R1")
    parser.add_argument("--K", type=int, default=None, help="K value for split (default from YAML or 0)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for split (default from YAML or 0)")
    # Model
    parser.add_argument("--width", type=int, default=None,
        help="SmallResUNet width: 16 for development, 32 for full model (default from YAML or 32)")
    parser.add_argument("--zero_raw_increment_init", action="store_true",
        help="Zero-initialize output head so pred_inc_raw ≈ 0 at init")
    parser.add_argument("--target_increment_normalization", action="store_true",
        help="Normalize target increments during training")
    # Training
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--accum_steps", type=int, default=None,
        help="Gradient accumulation steps for larger effective batch size")
    # Data loading
    parser.add_argument("--num_workers", type=int, default=None,
        help="DataLoader num_workers (default 0, avoid >0 due to netCDF threading issues)")
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--require_gpu", action="store_true",
        help="Exit with error if CUDA unavailable")
    parser.add_argument("--amp", action="store_true",
        help="Enable automatic mixed precision")
    # Config
    parser.add_argument("--config", type=str, default=None,
        help="Path to YAML config file (CLI args override YAML values)")
    # Run management
    parser.add_argument("--run_name", type=str, default=None,
        help="Override run name")
    parser.add_argument("--output_dir", type=str, default=None,
        help="Override output directory")
    # Logging
    parser.add_argument("--wandb_mode", type=str, default="disabled",
        choices=["disabled", "offline", "online"],
        help="Wandb mode (default: disabled)")
    parser.add_argument("--wandb_project", type=str, default="hydroda-ood")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[])
    parser.add_argument("--log_every_steps", type=int, default=100)
    parser.add_argument("--eval_every_epochs", type=int, default=1)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)

    # First pass: check if --config is provided
    preliminary_args, _ = parser.parse_known_args()

    # Load YAML config as defaults (CLI will override)
    yaml_defaults = {}
    if preliminary_args.config and Path(preliminary_args.config).exists():
        file_config = load_config(preliminary_args.config)
        for section in ["model", "training", "data", "output"]:
            if section in file_config and isinstance(file_config[section], dict):
                yaml_defaults.update(file_config[section])

    # Set YAML values as argparse defaults for key parameters
    yaml_to_arg_map = {
        "width": "width",
        "max_epochs": "max_epochs",
        "batch_size": "batch_size",
        "accum_steps": "accum_steps",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "grad_clip": "grad_clip",
        "num_workers": "num_workers",
        "K": "K",
        "seed": "seed",
        "zero_raw_increment_init": "zero_raw_increment_init",
        "target_increment_normalization": "target_increment_normalization",
    }
    for yaml_key, arg_key in yaml_to_arg_map.items():
        if yaml_key in yaml_defaults and parser.get_default(arg_key) is None:
            parser.set_defaults(**{arg_key: yaml_defaults[yaml_key]})

    # Set hard-coded fallback defaults for required params not in YAML
    if parser.get_default("K") is None:
        parser.set_defaults(K=0)
    if parser.get_default("seed") is None:
        parser.set_defaults(seed=0)
    if parser.get_default("width") is None:
        parser.set_defaults(width=32)
    if parser.get_default("max_epochs") is None:
        parser.set_defaults(max_epochs=30)
    if parser.get_default("batch_size") is None:
        parser.set_defaults(batch_size=2)
    if parser.get_default("lr") is None:
        parser.set_defaults(lr=1e-3)
    if parser.get_default("weight_decay") is None:
        parser.set_defaults(weight_decay=1e-4)
    if parser.get_default("accum_steps") is None:
        parser.set_defaults(accum_steps=1)
    if parser.get_default("num_workers") is None:
        parser.set_defaults(num_workers=0)

    return parser.parse_args()


def compute_source_val_shrinkage(trainer, source_val_dataset, device):
    """Evaluate model on source_val with shrinkage alphas and select best alpha.

    Returns:
        dict with best_alpha, alpha_results, summary
    """
    import numpy as np
    from hydroda.metrics.skill import _valid_flat, _rmse

    model = trainer.model
    model.eval()

    if len(source_val_dataset) == 0:
        print("  WARNING: source_val_dataset is empty, returning null shrinkage result")
        return {
            "selection_source": "source_val",
            "best_alpha": None,
            "best_combined_skill": float("nan"),
            "alpha_results": {},
            "note_zero_alpha": None,
            "warning": "source_val_dataset_is_empty",
        }

    alphas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    alpha_results = {a: {"surface": [], "rootzone": []} for a in alphas}
    global_results = {}

    with torch.no_grad():
        for idx in range(len(source_val_dataset)):
            sample = source_val_dataset[idx]
            x = torch.from_numpy(np.asarray(sample["x"], dtype=np.float32)).unsqueeze(0).to(device)
            x_norm = trainer._normalize(x)
            pred = model(x_norm)
            pred_inc_s = pred[0, 0].cpu().numpy().astype(np.float32)
            pred_inc_r = pred[0, 1].cpu().numpy().astype(np.float32)

            # Denormalize if model was trained with increment normalization
            if trainer.target_increment_normalization and trainer._inc_mean is not None:
                pred_inc_s = pred_inc_s * trainer._inc_std[0] + trainer._inc_mean[0]
                pred_inc_r = pred_inc_r * trainer._inc_std[1] + trainer._inc_mean[1]

            mask = sample["metric_mask"]
            forecast_s = sample["forecast_surface"]
            forecast_r = sample["forecast_rootzone"]
            analysis_s = sample["analysis_surface"]
            analysis_r = sample["analysis_rootzone"]

            for alpha in alphas:
                for var in ["surface", "rootzone"]:
                    if var == "surface":
                        pred_an = forecast_s + alpha * pred_inc_s
                        true_an = analysis_s
                        fcst = forecast_s
                    else:
                        pred_an = forecast_r + alpha * pred_inc_r
                        true_an = analysis_r
                        fcst = forecast_r

                    p, t, f = _valid_flat(pred_an, true_an, fcst, mask=mask)
                    if p.size >= 2:
                        rmse_pred = _rmse(p, t)
                        rmse_fcst = _rmse(f, t)
                        if np.isfinite(rmse_fcst) and rmse_fcst > 0:
                            skill = float(1.0 - rmse_pred / rmse_fcst)
                        else:
                            skill = np.nan
                    else:
                        skill = np.nan

                    alpha_results[alpha][var].append(skill)

    # Compute mean skill per alpha (global pooled across all source_val cycles)
    for alpha in alphas:
        global_results[f"alpha_{alpha}"] = {}
        for var in ["surface", "rootzone"]:
            valid = [v for v in alpha_results[alpha][var] if np.isfinite(v)]
            if valid:
                global_results[f"alpha_{alpha}"][f"{var}_skill_mean"] = float(np.mean(valid))
                global_results[f"alpha_{alpha}"][f"{var}_skill_median"] = float(np.median(valid))
                global_results[f"alpha_{alpha}"][f"{var}_skill_std"] = float(np.std(valid))
                global_results[f"alpha_{alpha}"][f"{var}_n"] = len(valid)
            else:
                global_results[f"alpha_{alpha}"][f"{var}_skill_mean"] = np.nan
                global_results[f"alpha_{alpha}"][f"{var}_skill_median"] = np.nan
                global_results[f"alpha_{alpha}"][f"{var}_skill_std"] = np.nan
                global_results[f"alpha_{alpha}"][f"{var}_n"] = 0

    # Select best alpha based on mean of surface + rootzone skill
    best_alpha = None
    best_combined_skill = float("nan")

    n_finite = 0
    for alpha in alphas:
        s_finite = np.isfinite(global_results[f"alpha_{alpha}"].get("surface_skill_mean", np.nan))
        r_finite = np.isfinite(global_results[f"alpha_{alpha}"].get("rootzone_skill_mean", np.nan))
        if s_finite or r_finite:
            n_finite += 1

    if n_finite == 0:
        return {
            "selection_source": "source_val",
            "best_alpha": None,
            "best_combined_skill": float("nan"),
            "alpha_results": global_results,
            "note_zero_alpha": None,
            "warning": "no_finite_skills_computed_all_alphas_nan",
        }

    best_combined_skill = -np.inf
    for alpha in alphas:
        s_mean = global_results[f"alpha_{alpha}"].get("surface_skill_mean", np.nan)
        r_mean = global_results[f"alpha_{alpha}"].get("rootzone_skill_mean", np.nan)
        combined = (s_mean + r_mean) / 2.0 if np.isfinite(s_mean) and np.isfinite(r_mean) else np.nan
        global_results[f"alpha_{alpha}"]["combined_skill_mean"] = combined
        if np.isfinite(combined) and combined > best_combined_skill:
            best_combined_skill = combined
            best_alpha = alpha

    result = {
        "selection_source": "source_val",
        "best_alpha": best_alpha,
        "best_combined_skill": best_combined_skill,
        "alpha_results": global_results,
        "note_zero_alpha": (
            "source_val believes model should not modify forecast"
            if best_alpha == 0.0 else None
        ),
    }
    return result


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # Resolve device
    device = resolve_device(args.device, require_gpu=args.require_gpu)

    print("=" * 60)
    print("Phase 4A: Source-only Backbone Training")
    print(f"  target_region={args.target_region}  K={args.K}  seed={args.seed}")
    print(f"  max_epochs={args.max_epochs}  batch_size={args.batch_size}  lr={args.lr}")
    print(f"  device={device}  width={args.width}  amp={args.amp}")
    print("=" * 60)

    # Build config for RunManager (args already resolved with YAML defaults + CLI overrides)
    run_config = {
        "target_region": args.target_region,
        "K": args.K,
        "seed": args.seed,
        "width": args.width,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "accum_steps": args.accum_steps,
        "num_workers": args.num_workers,
        "device": str(device),
        "use_amp": args.amp,
        "zero_raw_increment_init": args.zero_raw_increment_init,
        "target_increment_normalization": args.target_increment_normalization,
        "log_every_steps": args.log_every_steps,
        "eval_every_epochs": args.eval_every_epochs,
        "wandb_mode": args.wandb_mode,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_tags": args.wandb_tags,
    }

    # Create RunManager
    run_manager = RunManager(
        phase=PHASE,
        method="source_only",
        target_region=args.target_region,
        config=run_config,
        output_dir=args.output_dir,
        run_name=args.run_name,
        width=args.width,
        epochs=args.max_epochs,
        lr=args.lr,
        norm="norm" if args.target_increment_normalization else "nonorm",
        zero_raw=args.zero_raw_increment_init,
        seed=args.seed,
    )

    # Save configs
    run_manager.save_config(run_config, "config.yaml")
    run_manager.save_git_info()

    # Log protocol freeze id
    run_manager.save_protocol({
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "split_manifest": FREEZE_MANIFEST,
    })

    # Setup logging
    console_logger = ConsoleLogger(
        log_every_steps=args.log_every_steps,
        max_epochs=args.max_epochs,
    )

    # Wandb logger
    wandb_logger = WandbLogger(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        tags=args.wandb_tags,
        run_name=run_manager.get_run_name(),
    )

    start_time = time.time()

    # Create source_fit dataset (2015-2020, excluding target region)
    print(f"\nLoading source_fit dataset...")
    train_dataset = HydroDADataset(
        da_nc_path=DA_NC,
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region=args.target_region,
        split_type="source_fit",
        K=args.K,
        seed=args.seed,
        freeze_manifest=FREEZE_MANIFEST,
    )
    print(f"  source_fit samples: {len(train_dataset)}")

    # Create source_val dataset (2021, excluding target region)
    print(f"\nLoading source_val dataset...")
    source_val_dataset = HydroDADataset(
        da_nc_path=DA_NC,
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region=args.target_region,
        split_type="source_val",
        K=args.K,
        seed=args.seed,
        freeze_manifest=FREEZE_MANIFEST,
    )
    print(f"  source_val samples: {len(source_val_dataset)}")

    if len(source_val_dataset) == 0:
        raise RuntimeError(
            f"source_val_dataset is empty for target={args.target_region}, K={args.K}, seed={args.seed}. "
            f"The split manifest must contain source_val_dates (2021 dates for source regions). "
            f"Re-run build_kdate_splits.py with the updated code that populates source_val_dates."
        )

    # Init model
    print(f"\nInitializing SmallResUNet (width={args.width})...")
    model = SmallResUNet(
        in_channels=12,
        out_channels=2,
        width=args.width,
        zero_raw_increment_init=args.zero_raw_increment_init,
    )

    # Get checkpoint dir from run_manager
    checkpoint_dir = args.checkpoint_dir or str(run_manager.get_checkpoint_dir())

    # Create Trainer with run_manager
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=str(device),
        checkpoint_dir=checkpoint_dir,
        experiment_id=run_manager.get_run_name(),
        protocol_freeze_id=PROTOCOL_FREEZE_ID,
        split_manifest_path=FREEZE_MANIFEST,
        grad_clip=args.grad_clip,
        model_width=args.width,
        target_increment_normalization=args.target_increment_normalization,
        zero_raw_increment_init=args.zero_raw_increment_init,
        accum_steps=args.accum_steps,
        run_manager=run_manager,
        use_amp=args.amp,
        log_every_steps=args.log_every_steps,
        eval_every_epochs=args.eval_every_epochs,
        wandb_logger=wandb_logger,
        source_val_dataset=source_val_dataset,
    )

    # Save environment info AFTER model is on GPU for accurate memory stats
    run_manager.save_environment_info(gather_runtime_info())

    # Train
    print(f"\nStarting training...")
    history = trainer.train(verbose=True)

    elapsed = time.time() - start_time

    # Save summary.json
    summary_path = run_manager.summary_json_path()
    trainer.save_summary_json(summary_path)

    # Source-val shrinkage computation
    print(f"\n{'=' * 60}")
    print(f"Computing source_val shrinkage...")
    print(f"  alphas: [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]")
    shrinkage = compute_source_val_shrinkage(trainer, source_val_dataset, str(device))
    print(f"  best_alpha: {shrinkage['best_alpha']}")
    print(f"  best_combined_skill: {shrinkage['best_combined_skill']:.6f}")
    if shrinkage.get("note_zero_alpha"):
        print(f"  NOTE: {shrinkage['note_zero_alpha']}")
    shrinkage_path = run_manager.get_results_dir() / "shrinkage_source_val.json"
    with open(shrinkage_path, "w") as f:
        json.dump(shrinkage, f, indent=2)
    print(f"  saved to {shrinkage_path}")

    # Print results
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  best_loss={trainer.best_loss:.6f}")
    print(f"  run_dir={run_manager.get_run_dir()}")
    print(f"  summary={summary_path}")

    best_ckpt = run_manager.checkpoint_best_path()
    last_ckpt = run_manager.checkpoint_last_path()
    print(f"  best_checkpoint={best_ckpt}")
    print(f"  last_checkpoint={last_ckpt}")

    # Save history
    history_path = run_manager.get_results_dir() / "train_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Close wandb
    if wandb_logger.enabled:
        wandb_logger.finish()
        print(f"  wandb_run_id={wandb_logger.run_id}")


if __name__ == "__main__":
    main()