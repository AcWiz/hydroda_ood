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

## HyperDA v1 Experiment Record

HyperDA v1 is the first generated-operator method after the prompt-conditioned
FiLM baseline. It reuses the Phase 4 target-full-train protocol and changes
only the conditional model class.

### Default Development Run

Use one region and one seed first:

```bash
bash run/phase4_hyperda.sh US-R1 0 1
```

Default settings:

- `model_type=hyperda_basis_adapter`
- `width=32`
- `prompt_dim=64`
- `hyper_n_basis=8`
- `hyper_adapter_bottleneck=32`
- `hyper_adapter_scale=1.0`
- `batch_size=16`
- `accum_steps=4`
- `max_epochs=50`
- `selection_metric=source_val_transfer_safe_score`

Protocol invariants:

- train split: `source_fit`
- model selection: `source_val`
- target evaluation: `target_eval` only after training
- split artifact: `artifacts/splits/US_loro_target_train_splits.json`
- adaptation setting: `target_full_train`
- no `--K` argument

### Artifact Paths

Training writes under:

```text
artifacts/runs/phase4_prompt_conditioned/phase4_prompt_conditioned_hyperda_basis_adapter_*/
```

The checkpoint summary and config include:

- `model_type`
- `hyper_n_basis`
- `hyper_adapter_bottleneck`
- `hyper_adapter_scale`
- `best_selection_metric`
- `best_selection_value`
- source/target protocol safety fields

### Evaluation

After training, evaluate the best transfer-safe checkpoint:

```bash
bash run/phase4_prompt_conditioned_inference.sh \
  artifacts/runs/phase4_prompt_conditioned/<run_name>/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt \
  US-R1 0 1
```

The existing prompt-conditioned predictor auto-loads HyperDA checkpoints when
`config.model_type` is `hyperda_basis_adapter`.

### Required Comparison Before Paper Claims

For any table or claim, compare under the same current protocol:

- `source_only_backbone`
- `prompt_conditioned_shared_backbone`
- `hyperda_basis_adapter`

The initial development scope is `US-R1`, seed `0`. Multi-region and multi-seed
runs should be added only after HyperDA v1 is stable.
