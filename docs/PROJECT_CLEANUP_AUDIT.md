# Project Cleanup Audit

Date: 2026-05-28

This audit records the medium cleanup policy for HydroDA-OOD after the protocol
migration to V4.4 zero/few-shot target generalization and HyperDA-SAFE.

## Active Research Contract

The active protocol is:

```text
protocol_version: v4.4-zero-few-shot-generalization
adaptation_setting: zero_shot_context | few_shot_k4 | few_shot_k12
source_fit/train: 2015-2021
source_val:   2022
target_context: 2015-2021 input-side only
target_support: K labeled cycles, K in {0,4,12}
target_val: unused in main protocol
target_eval: 2023-2025
```

The only active paper-facing protocol surface is V4.4 zero/few-shot plus
HyperDA-SAFE: Source-Anchored Few-Shot Operator Refinement. Any document,
script, test, or report that treats HyRAO, K-date selection,
`target_full_train`, target_val-based target selection, phase6 residual ridge /
BORA adapters, or phase7 APO as the main protocol is historical, legacy, or
retired failed exploration.

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
scripts/legacy/
reports/forecast_only_latw_audit/target_context_k12/
artifacts/splits/US_loro_zero_few_shot_splits.json
artifacts/splits/US_loro_kdate_splits.json
artifacts/splits/US_loro_target_train_splits.json
```

`artifacts/splits/US_loro_zero_few_shot_splits.json` is the canonical active
split artifact and must be retained. The target-train and K-date manifests are
retained only for historical reproduction and compatibility.

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
artifacts/tmp/
artifacts/cache/source_context_monthly_prototypes/
```

Delete local `metrics_long.csv` only when the same run or parent directory has
a compact `summary*`, `overview*`, `metadata.json`, or `config.yaml` sufficient
to identify provenance and reproduce the table. Keep checkpoints, run configs,
protocol metadata, safe policies, compact summaries, audits, split manifests,
geolocation files, and region masks.

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

1. Keep old full-target and K-date assets for reproduction/ablations, but
   rename references so they cannot be mistaken for the main protocol.
2. Keep summary reports and small diagnostics; remove or ignore long metrics
   tables that can be regenerated from scripts.
3. Remove Python bytecode and W&B run payloads from git tracking.
4. Prefer `target_context`, `target_support`, and `target_eval` in all new docs,
   scripts, result schemas, and test names.
5. Leave compatibility aliases in code paths that must read old manifests.
6. Do not restore retired active entrypoints for HyperDA+ source-prior matrix,
   P2.6-P2.8 probe wrappers, episode-prior eval wrappers, phase6 residual
   ridge/BORA adapters, or phase7 APO.

## Risks To Monitor

Legacy leakage risk:

Old reports may contain 2022 `target_context` evaluations or `K=12` fields.
They must not be used as main-protocol evidence.

Retired-method confusion risk:

HyRAO, K-date, `target_full_train`, phase6 residual ridge/BORA, and phase7 APO
may still appear in compatibility notes, old artifact names, or retained tests.
Those mentions must be labeled legacy, diagnostic, internal, or retired; they
must not appear as active wrappers or paper-main method IDs.

Artifact drift risk:

The canonical split artifact is
`artifacts/splits/US_loro_zero_few_shot_splits.json`. Nested support manifests
may be removed only when their SHA256 matches this canonical file exactly.

Review risk:

Reviewers may object that target-side labels leak into model selection. The
response is to state the exact K-cycle label budget, preserve held-out
`target_eval`, and keep target_val/full-target runs out of the main protocol.
