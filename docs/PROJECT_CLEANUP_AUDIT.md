# Project Cleanup Audit

Date: 2026-05-28

This audit records the conservative cleanup policy for HydroDA-OOD after the
protocol migration from legacy K-shot support adaptation to V4.2
full-target-training-period adaptation.

## Active Research Contract

The active protocol is:

```text
protocol_version: v4.2-full-target-train
adaptation_setting: target_full_train
source_fit/train: 2015-2021
source_val:   2022
target_train/adaptation: 2015-2021
target_eval: 2023-2025
```

Any document, script, test, or report that treats `K=0/4/12`, `target_support`,
or `target_context_k12` as the main protocol is historical or legacy.

## File Classes

Active source-of-truth files:

```text
CLAUDE.md
完整研究计划方案.md
docs/COAUTHOR_CONTEXT.md
context/00_EXECUTABLE_CONTEXT_MAP.md
context/01_RESEARCH_CONTRACT.md
context/02_DATA_AND_LEAKAGE_CONTRACT.md
specs/protocol_v4.yaml
specs/experiment_schema.yaml
specs/hyperda_v4.yaml
specs/baselines.yaml
checklists/no_leakage_checklist.md
```

Legacy but retained:

```text
specs/kdate_protocol.yaml
context/04_KDATE_SPLIT_PROTOCOL.md
scripts/legacy/
reports/forecast_only_latw_audit/target_context_k12/
artifacts/splits/US_loro_kdate_splits.json
```

Generated or local-only:

```text
__pycache__/
.pytest_cache/
wandb/
reports/**/metrics_long*.csv
artifacts/runs/
artifacts/metrics/
artifacts/experiments/
artifacts/checkpoints/
```

Canonical region artifacts currently tracked for US development:

```text
artifacts/regions/US_region_masks.nc
artifacts/regions/US_region_mask_tensor.pt
artifacts/regions/US_region_masks_manifest.json
artifacts/regions/US_region_stats.json
```

Do not remove these region artifacts from git tracking in this cleanup pass.
They are small enough for the current repository and are used by real-data
tests. Revisit this only if the project moves to external artifact storage.

## Cleanup Decisions

1. Keep old K-shot assets for secondary ablations, but rename references so
   they cannot be mistaken for the main protocol.
2. Keep summary reports and small diagnostics; remove or ignore long metrics
   tables that can be regenerated from scripts.
3. Remove Python bytecode and W&B run payloads from git tracking.
4. Prefer `target_train` and `target_eval` in all new docs, scripts, result
   schemas, and test names.
5. Leave compatibility aliases in code paths that must read old manifests.

## Risks To Monitor

Legacy leakage risk:

Old reports may contain 2022 `target_context` evaluations or `K=12` fields.
They must not be used as main-protocol evidence.

Artifact drift risk:

The new target-train split file is not generated until
`scripts/data/build_target_train_splits.py` is run. Tests that require this
artifact should skip or use synthetic manifests until the artifact is frozen.

Review risk:

Reviewers may object that full target-train adaptation is easier than few-shot.
The response is to state the exact label budget, preserve held-out
`target_eval`, and report legacy K-shot only as a secondary ablation.
