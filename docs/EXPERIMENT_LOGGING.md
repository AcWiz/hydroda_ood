# Experiment Logging System

## Three-Layer Logging

HydroDA-OOD uses a three-layer logging system:

### 1. ConsoleLogger — Human-Readable Output

Readable output every `log_every_steps` steps.

**Logged fields per step:**
- epoch, step, lr
- total_loss, surface_loss, rootzone_loss
- valid_pixel_fraction, grad_norm
- pred_increment_surface_mean/std, pred_increment_rootzone_mean/std
- true_increment_surface_mean/std, true_increment_rootzone_mean/std
- GPU memory allocated/reserved
- batches/sec, ETA

**Logged fields per epoch:**
- epoch, total_loss, surface_loss, rootzone_loss
- valid_pixel_count, lr, elapsed_s
- source_val metrics (if source_val_dataset provided)

### 2. JSONLLogger — Machine-Readable Logs

Append-only JSON logs in `logs/` directory:
- `logs/train_steps.jsonl` — one JSON dict per training step
- `logs/train_epochs.jsonl` — one JSON dict per epoch
- `logs/eval_metrics.jsonl` — one JSON dict per eval run

Each line is a valid JSON object. Files can be appended to without rewriting.

### 3. WandbLogger — Optional Experiment Tracking

Weights & Biases integration. Default: **disabled**.

Modes:
- `disabled` (default): no network calls, no wandb init
- `offline`: local wandb run, no network needed
- `online`: full wandb sync to cloud

Enable via CLI:
```bash
python scripts/train_source_only_backbone.py \
  --wandb_mode online \
  --wandb_project hydroda-ood \
  --wandb_entity your-team \
  --wandb_tags phase4 source_only US-R1
```

**Logged to Wandb:**
- train/total_loss, train/lr, train/grad_norm
- train/valid_pixel_fraction, train/pred_inc_std
- train/gpu_memory_gb
- eval/* metrics (source_val evaluation)

## Run Directory Structure

Each run creates:
```
artifacts/runs/{phase}/{run_name}/
  config.yaml           # CLI/config parameters
  environment.json      # Runtime environment info
  git_info.json         # Git hash and status
  protocol.json         # Protocol freeze ID
  data_manifest.json    # Data split info
  logs/
    train_steps.jsonl   # Per-step metrics
    train_epochs.jsonl  # Per-epoch metrics
    eval_metrics.jsonl  # Eval results
    console.log         # Console output
  checkpoints/
    best.pt             # Best model checkpoint
    last.pt             # Latest checkpoint
  results/
    train_history.json  # Full training history
  reports/
    summary.json        # Final summary with protocol safety fields
```

## Protocol Safety Rules

**Critical rules to prevent target query leakage:**

1. **Normalization**: Only source_train/source_fit stats used. Never target_query.
2. **Early stopping**: Only source_val or train_loss. Never target_query.
3. **Model selection**: Only source_val or train_loss. Never target_query.
4. **Target query**: ONLY for post-prediction final evaluation, logged as `target_query_eval_only/*`.

**Summary.json fields confirm protocol safety:**
```json
{
  "normalization_source": "source_train_only",
  "early_stopping_source": "train_loss_only",
  "model_selection_source": "best_train_loss",
  "target_query_usage": "eval_only_no_early_stopping",
  "leakage_guard_status": "pass"
}
```

## wandb_mode=disabled Safety

When `--wandb_mode disabled`:
- No network calls on import
- No wandb.init() called
- Logger is a no-op for all logging calls
- Safe to run on machines without internet access