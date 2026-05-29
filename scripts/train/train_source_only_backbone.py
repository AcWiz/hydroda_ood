#!/usr/bin/env python3
"""Train source-only SmallResUNet backbone with full experiment infrastructure.

Usage:
    PYTHONPATH=. python scripts/train/train_source_only_backbone.py \\
        --target_region US-R1 --adaptation_setting target_full_train --seed 0 \\
        --max_epochs 30 --batch_size 4 --lr 1e-3 \\
        --device cuda --amp \\
        --wandb_mode disabled \\
        --config configs/model_resunet_main.yaml

No-leakage declaration:
    - Only source_train split used for training
    - Normalization stats from source_train only
    - No target_eval/target_query labels used in training/normalization/early_stopping
    - No target prompt or target adaptation labels used by this source-only model
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
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
SPLITS_JSON = "artifacts/splits/US_loro_target_train_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
CHECKPOINT_DIR = "artifacts/checkpoints/phase4_source_only"
PROTOCOL_FREEZE_ID = "hyperda_v4_3_historical_target_adapt_2015_2025_train2015_2021_val2022_test2023_2025"
PHASE = "phase4_source_only"


def parse_args():
    parser = argparse.ArgumentParser(description="Train source-only backbone with experiment tracking")
    # Data
    parser.add_argument("--target_region", type=str, required=True, help="Target region, e.g. US-R1")
    parser.add_argument("--adaptation_setting", type=str, default=None,
        help="Split adaptation setting (default: target_full_train; legacy example: legacy_few_shot_k4)")
    parser.add_argument("--K", type=int, default=None,
        help="Legacy few-shot K value. Ignored for target_full_train.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for split (default from YAML or 0)")
    # Model
    parser.add_argument("--width", type=int, default=None,
        help="SmallResUNet width: 16 for development, 32 for full model (default from YAML or 32)")
    parser.add_argument("--zero_raw_increment_init", action="store_true", default=None,
        help="Zero-initialize output head so pred_inc_raw ≈ 0 at init")
    parser.add_argument("--no_zero_raw_increment_init", action="store_false", dest="zero_raw_increment_init",
        help="Disable zero-raw-increment init")
    parser.add_argument("--target_increment_normalization", action="store_true", default=None,
        help="Normalize target increments during training")
    parser.add_argument("--no_target_increment_normalization", action="store_false", dest="target_increment_normalization",
        help="Disable target increment normalization")
    parser.add_argument("--target_normalization_mode", type=str, default=None,
        choices=["none", "per_variable_increment_std"],
        help="Convenience: set target normalization mode. "
             "'none' = raw MSE (no target norm). "
             "'per_variable_increment_std' = normalize each variable to unit variance "
             "(maps to --target_increment_normalization --zero_raw_increment_init). "
             "Overrides explicit --target_increment_normalization / --zero_raw_increment_init flags.")
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
    parser.add_argument("--resume_from", type=str, default=None,
        help="Path to checkpoint.pt to resume from (last.pt or best.pt). "
             "When provided, training continues from the saved epoch and "
             "normalization stats are restored from the checkpoint.")
    # Lat-weighted loss and gain calibration
    parser.add_argument("--use_lat_weighted_loss", action="store_true", default=True,
        help="Use WeightedMaskedHuberLoss with latitude weighting (default: True)")
    parser.add_argument("--no_lat_weighted_loss", action="store_false", dest="use_lat_weighted_loss",
        help="Disable latitude-weighted loss")
    parser.add_argument("--checkpoint_every", type=int, default=5,
        help="Save epoch checkpoint every N epochs (default: 5)")
    parser.add_argument("--selection_metric", type=str, default="source_val_safe_score",
        choices=["source_val_safe_score", "source_val_loss"],
        help="Metric for best checkpoint selection (default: source_val_safe_score)")
    parser.add_argument("--apply_source_val_residual_gain", action="store_true", default=True,
        help="Apply residual gain calibration on source_val (default: True)")
    parser.add_argument("--no_source_val_residual_gain", action="store_false", dest="apply_source_val_residual_gain",
        help="Disable residual gain calibration")
    parser.add_argument("--lambda_amp", type=float, default=0.0,
        help="Amplitude penalty weight (0=disabled, default: 0.0)")

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
        "adaptation_setting": "adaptation_setting",
        "K": "K",
        "seed": "seed",
        "zero_raw_increment_init": "zero_raw_increment_init",
        "target_increment_normalization": "target_increment_normalization",
    }
    for yaml_key, arg_key in yaml_to_arg_map.items():
        if yaml_key in yaml_defaults and parser.get_default(arg_key) is None:
            parser.set_defaults(**{arg_key: yaml_defaults[yaml_key]})

    # Set hard-coded fallback defaults for required params not in YAML
    if parser.get_default("adaptation_setting") is None:
        parser.set_defaults(adaptation_setting="target_full_train")
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

    args = parser.parse_args()

    if args.adaptation_setting == "target_full_train":
        args.K = None
    elif args.K is None:
        args.K = 0

    # Map --target_normalization_mode to underlying flags
    if args.target_normalization_mode is not None:
        if args.target_normalization_mode == "per_variable_increment_std":
            args.target_increment_normalization = True
            args.zero_raw_increment_init = True
            print(f"  [target_normalization_mode] per_variable_increment_std → "
                  f"target_increment_normalization=True, zero_raw_increment_init=True")
        elif args.target_normalization_mode == "none":
            args.target_increment_normalization = False
            args.zero_raw_increment_init = False
            print(f"  [target_normalization_mode] none → "
                  f"target_increment_normalization=False, zero_raw_increment_init=False")

    return args


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
    print(f"  target_region={args.target_region}  adaptation_setting={args.adaptation_setting}  K={args.K}  seed={args.seed}")
    print(f"  max_epochs={args.max_epochs}  batch_size={args.batch_size}  lr={args.lr}")
    print(f"  device={device}  width={args.width}  amp={args.amp}")
    print("=" * 60)

    # Auto-derive output_dir from checkpoint when resuming without explicit output_dir
    if args.resume_from and args.output_dir is None:
        ckpt_path = Path(args.resume_from)
        # checkpoint is at .../run_name/checkpoints/last.pt or best.pt
        # run directory is checkpoints' parent
        args.output_dir = str(ckpt_path.parent.parent)
        print(f"  output_dir auto-derived from checkpoint: {args.output_dir}")

    # Build config for RunManager (args already resolved with YAML defaults + CLI overrides)
    run_config = {
        "target_region": args.target_region,
        "adaptation_setting": args.adaptation_setting,
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
        "target_normalization_mode": args.target_normalization_mode or (
            "per_variable_increment_std" if args.target_increment_normalization else "none"
        ),
        "log_every_steps": args.log_every_steps,
        "eval_every_epochs": args.eval_every_epochs,
        "use_lat_weighted_loss": args.use_lat_weighted_loss,
        "checkpoint_every": args.checkpoint_every,
        "selection_metric": args.selection_metric,
        "apply_source_val_residual_gain": args.apply_source_val_residual_gain,
        "lambda_amp": args.lambda_amp,
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

    # Optional resume: load checkpoint before creating Trainer
    resumed_epoch = 0
    checkpoint_config = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume_from checkpoint not found: {resume_path}")
        print(f"\nResuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        checkpoint_config = ckpt["config"]
        resumed_epoch = ckpt["epoch"] + 1
        print(f"  checkpoint epoch={ckpt['epoch']}  best_loss={ckpt.get('best_loss', 'N/A')}")
        print(f"  resuming from epoch {resumed_epoch} ({resumed_epoch} already completed)")

    # Create source_fit dataset (2015-2021, excluding target region)
    print(f"\nLoading source_fit dataset...")
    train_dataset = HydroDADataset(
        da_nc_path=DA_NC,
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region=args.target_region,
        split_type="source_fit",
        K=args.K,
        seed=args.seed,
        adaptation_setting=args.adaptation_setting,
        freeze_manifest=FREEZE_MANIFEST,
    )
    print(f"  source_fit samples: {len(train_dataset)}")

    # Create source_val dataset (2022, excluding target region)
    print(f"\nLoading source_val dataset...")
    source_val_dataset = HydroDADataset(
        da_nc_path=DA_NC,
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region=args.target_region,
        split_type="source_val",
        K=args.K,
        seed=args.seed,
        adaptation_setting=args.adaptation_setting,
        freeze_manifest=FREEZE_MANIFEST,
    )
    print(f"  source_val samples: {len(source_val_dataset)}")

    if len(source_val_dataset) == 0:
        raise RuntimeError(
            f"source_val_dataset is empty for target={args.target_region}, K={args.K}, seed={args.seed}. "
            f"The split manifest must contain source_val_dates (2022 dates for source regions). "
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

    # Resume: pre-compute normalization stats from checkpoint so Trainer.__init__
    # skips recomputation (avoids dataset re-scan and ensures exact stats match)
    resume_ch_mean = None
    resume_ch_std = None
    resume_inc_mean = None
    resume_inc_std = None
    if resumed_epoch > 0 and checkpoint_config is not None:
        if checkpoint_config.get("ch_mean") is not None:
            resume_ch_mean = np.array(checkpoint_config["ch_mean"], dtype=np.float32)
        if checkpoint_config.get("ch_std") is not None:
            resume_ch_std = np.array(checkpoint_config["ch_std"], dtype=np.float32)
        if checkpoint_config.get("inc_mean") is not None:
            resume_inc_mean = np.array(checkpoint_config["inc_mean"], dtype=np.float32)
        if checkpoint_config.get("inc_std") is not None:
            resume_inc_std = np.array(checkpoint_config["inc_std"], dtype=np.float32)

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
        use_lat_weighted_loss=args.use_lat_weighted_loss,
        checkpoint_every_n_epochs=args.checkpoint_every,
        lambda_amp=args.lambda_amp,
        source_val_gain_grid=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        # Resume: inject pre-computed stats so Trainer.__init__ skips recompute
        _resume_ch_mean=resume_ch_mean,
        _resume_ch_std=resume_ch_std,
        _resume_inc_mean=resume_inc_mean,
        _resume_inc_std=resume_inc_std,
    )

    # Resume: restore full training state
    if resumed_epoch > 0:
        print(f"\nRestoring training state from checkpoint (resuming from epoch {resumed_epoch})...")
        trainer.load_state(ckpt)
        print(f"  Restored: optimizer, scheduler, epoch, best_loss, train_history")
        print(f"  train_history entries so far: {len(trainer.train_history)}")
        print(f"  val_history entries so far: {len(trainer.val_history)}")

    # Print normalization mode summary
    print(f"\n{'=' * 40}")
    print(f"Normalization Mode Summary")
    print(f"  target_increment_normalization: {trainer.target_increment_normalization}")
    print(f"  zero_raw_increment_init:        {trainer.zero_raw_increment_init}")
    if trainer.target_increment_normalization and trainer._inc_mean is not None:
        print(f"  inc_mean (surface, rootzone):   {trainer._inc_mean[0]:.6f}, {trainer._inc_mean[1]:.6f}")
        print(f"  inc_std  (surface, rootzone):   {trainer._inc_std[0]:.6f}, {trainer._inc_std[1]:.6f}")
        print(f"  → targets normalized to ~N(0,1) per variable; loss in normalized space")
    else:
        print(f"  → raw MSE loss (no target normalization)")
    print(f"{'=' * 40}")

    # Save environment info AFTER model is on GPU for accurate memory stats
    run_manager.save_environment_info(gather_runtime_info())

    # Train
    print(f"\nStarting training...")
    history = trainer.train(verbose=True)

    elapsed = time.time() - start_time

    # Save summary.json
    summary_path = run_manager.summary_json_path()
    trainer.save_summary_json(summary_path)

    # Print results
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  best_loss={trainer.best_loss:.6f}")
    print(f"  best_safe_score={trainer.best_safe_score:.4f}")
    print(f"  run_dir={run_manager.get_run_dir()}")
    print(f"  summary={summary_path}")

    best_ckpt = run_manager.checkpoint_best_path()
    last_ckpt = run_manager.checkpoint_last_path()
    safe_ckpt = trainer.checkpoint_dir / "checkpoint_best_source_val_safe_score.pt"
    print(f"  best_checkpoint={best_ckpt}")
    print(f"  last_checkpoint={last_ckpt}")
    if safe_ckpt.exists():
        print(f"  safe_score_checkpoint={safe_ckpt}")

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
