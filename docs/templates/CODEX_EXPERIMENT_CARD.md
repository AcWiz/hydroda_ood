# Codex Experiment Card

Use one card per run or tightly coupled run group.

## Identity

- Experiment name:
- Date:
- Owner:
- Phase:
- Method:
- Baseline group:
- Seed:

## Scientific Question

- Question:
- Falsifiable comparison:
- Reviewer-facing claim:

## Protocol

- Protocol version:
- Source domains:
- Target domain:
- Source fit/train period:
- Source validation period:
- Target context period:
- Target support K:
- Target evaluation period:
- Adaptation setting:
- Split artifact:
- Region artifact:

## Training Configuration

- Model:
- Loss:
- Loss mask:
- Metric mask:
- Latitude weighting:
- Target normalization:
- Input normalization source:
- Optimizer:
- Learning rate:
- LR schedule:
- Weight decay:
- Batch size:
- Accumulation steps:
- Max epochs:
- Selection metric:
- Early-stopping source:
- Checkpoint:

## Command

```bash

```

## Artifacts

- Run directory:
- Config:
- Environment:
- Git info:
- Protocol metadata:
- Data manifest:
- Training logs:
- Evaluation logs:
- Metrics:
- Reports:

## Results

| Split | Region | Variable | Metric | Value | Notes |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

## Interpretation

- Main observation:
- Compared with:
- Supports:
- Does not support:
- Failure mode or anomaly:

## Safety And Reproducibility

- Target-eval labels used only for final evaluation:
- Normalization excludes target evaluation:
- Model selection excludes target evaluation:
- Region definitions frozen before evaluation:
- Exact checkpoint recoverable:
- Metrics reproducible from logs:

## Next Experiment

- Next action:
- Reason:
- Minimal command:
