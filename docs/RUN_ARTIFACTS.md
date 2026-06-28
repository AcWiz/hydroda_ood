# Run Artifacts

## Directory Structure

Each experiment run creates a dated directory under `artifacts/runs/{phase}/`:

```
artifacts/runs/{phase}/
  {run_name}/
    config.yaml              # All CLI + config parameters
    config_resolved.yaml     # Resolved config after defaults
    environment.json         # Runtime environment snapshot
    git_info.json            # Git hash + status at start
    protocol.json            # Protocol freeze ID + split manifest
    data_manifest.json       # Data source info
    logs/
      train_steps.jsonl      # Per-step: epoch, step, loss, lr, grad_norm, ...
      train_epochs.jsonl     # Per-epoch: epoch, loss, source_val metrics
      eval_metrics.jsonl     # Eval: rmse, skill, corr, bias by region
      console.log           # Timestamp-prefixed console output
    checkpoints/
      best.pt                # Best model (lowest train_loss)
      last.pt                # Most recent checkpoint
    results/
      train_history.json     # Full epoch-by-epoch history
      metrics_long.csv        # Regenerable long-format metrics table
    reports/
      summary.json           # Final summary (see below)
      {method}_report.md     # Human-readable report
```

## Run Naming Convention

Auto-generated run_name format:
```
{phase}_{method}_{target_region}_w{width}_e{epochs}_lr{lr}_{norm}_{zero_raw}_s{seed}_{timestamp}
```

Example:
```
phase4_source_only_US-R1_w16_e5_lr0.001_norm_zero_s42_20250511_143052
```

Components:
- `phase`: e.g., "phase4_source_only"
- `method`: e.g., "source_only", "hyperda_zero"
- `target_region`: e.g., "US-R1"
- `w{width}`: model width (16 or 32)
- `e{epochs}`: max epochs
- `lr{lr}`: learning rate
- `norm`/`nonorm`: target_increment_normalization on/off
- `zero`/`nozero`: zero_raw_increment_init on/off
- `s{seed}`: random seed
- `timestamp`: UTC datetime

## Artifact Checklist

For each completed V4.4 zero-shot HyperDA-TRUST run or SAFE diagnostic run, verify:
- [ ] `config.yaml` — all parameters recorded
- [ ] `git_info.json` — git hash recorded (reproducibility)
- [ ] `protocol.json` — protocol_freeze_id matches experiment
- [ ] `logs/train_steps.jsonl` — not empty, valid JSON per line
- [ ] `logs/train_epochs.jsonl` — not empty, valid JSON per line
- [ ] `checkpoints/best.pt` — exists, model loads correctly
- [ ] `checkpoints/last.pt` — exists
- [ ] `reports/summary.json` or `overview.json` — compact provenance and metrics present
- [ ] `reports/summary.json` — leakage_guard_status = "pass"
- [ ] `protocol.json` / metadata — `target_val_usage = unused_in_main_protocol`
- [ ] SAFE K=4/K=12 diagnostics — `safe_policy.json` provenance and policy hash present
- [ ] SAFE K=4/K=12 diagnostics — `stage3_posterior_decision` is reported; `rejected_to_k0_anchor` rows are treated as K0-equivalent fallback
- [ ] SAFE K=4/K=12 diagnostics — target_eval was final-only and not used for selection

`metrics_long.csv` is a local regenerable payload when compact summaries,
metadata, config, and protocol files are present. It may be removed during
medium cleanup; do not remove summaries, configs, protocol metadata, safe
policies, checkpoints, split manifests, or audit artifacts.

## Summary.json Required Fields

```json
{
  "experiment_id": "...",
  "protocol_freeze_id": "...",
  "best_loss": 0.xxx,
  "final_epoch": 4,
  "normalization_source": "source_train_only",
  "early_stopping_source": "train_loss_only",
  "model_selection_source": "best_train_loss",
  "target_eval_usage": "final_eval_only_no_selection",
  "target_query_usage": "deprecated_alias_eval_only_no_early_stopping",
  "leakage_guard_status": "pass",
  "git_hash": "abc1234",
  "timestamp": "2025-05-11T14:30:52Z",
  "train_history": [...]
}
```

## Overriding Run Directory

Use `--run_name` to set a custom identifier, or `--output_dir` to set the base directory:

```bash
python scripts/train_source_only_backbone.py \
  --run_name my_debug_run \
  --output_dir /tmp/hydroda_runs
```

## Checkpoint Contents

Each checkpoint (best.pt, last.pt) contains:
- `tag`: "best" or "last"
- `epoch`, `loss`, `best_loss` or source-val selection metric
- `experiment_id`, `protocol_freeze_id`, `split_manifest_path`
- `adaptation_setting`, `K`, `model_selection_source`
- SAFE K-shot diagnostic metadata such as `safe_policy_json_sha256`,
  `policy_source`, `anchor_alpha`, and `adapt_mix_rho`
- `git_hash`, `timestamp`
- `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`
- `train_history`
- `config` (lr, weight_decay, max_epochs, batch_size, width, normalization params, etc.)

## Loading Checkpoints

```python
from hydroda.training.trainer import Trainer

checkpoint_path = "artifacts/runs/phase4_source_only/.../checkpoints/best.pt"
metadata = Trainer.load_checkpoint(Path(checkpoint_path), model, device="cuda")
print(metadata["epoch"], metadata["loss"], metadata["git_hash"])
```
