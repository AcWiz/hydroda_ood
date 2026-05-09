# Experiment Card

## Basic

```yaml
experiment_id:
benchmark_id:
phase:
status:
method_id:
country_id:
target_region_id:
source_region_ids:
K:
seed:
split_manifest_path:
config_path:
created_utc:
```

## Research question

本实验回答什么问题？

## Inputs

- Data:
- Region masks:
- Split manifest:
- Normalization stats:

## Method

简述方法，不超过 10 行。

## Commands

```bash
# training / baseline command
# evaluation command
```

## Outputs

- Metrics:
- Predictions:
- Checkpoints:
- Report:

## No-leakage declaration

- [ ] No target query labels for support selection.
- [ ] No target query labels for normalization.
- [ ] No target query labels for early stopping/model selection.
- [ ] Region definition independent of increments/errors.

## Results

| metric | surface | rootzone | notes |
|---|---:|---:|---|

## Failure / notes

