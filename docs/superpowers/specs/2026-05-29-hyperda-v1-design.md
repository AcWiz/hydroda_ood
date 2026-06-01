# HyperDA v1 Design

## Goal

Build the first executable HyperDA method for the Phase 4 target-region protocol. The first run is intentionally small: one target region and one seed, using the same no-leakage split protocol and source-val checkpoint selection as the prompt-conditioned baseline.

## Current Baseline

`run/phase4_prompt_conditioned.sh` trains a shared conditional baseline:

- training split: `source_fit`
- checkpoint selection: `source_val`
- target labels: not used for prompt-conditioned shared baseline training or selection
- target eval: post-training evaluation only
- model: `RegionPromptEncoder` plus `FiLMConditionalResUNet`
- selection metric: `source_val_transfer_safe_score`

The observed local result for `US-R1`, seed `0`, shows prompt-conditioned improvement over the older source-only result. Because those artifacts used different historical naming/freeze ids in places, the reusable experiment record should always compare methods under the current `target_full_train` protocol.

## HyperDA v1 Architecture

HyperDA v1 uses the same safe prompt source as the prompt-conditioned baseline:

```text
RegionPromptEncoder(x, source_region_id, month) -> z
```

Instead of using `z` only for FiLM feature modulation, HyperDA v1 uses `z` to mix a small set of learned bottleneck adapter bases:

```text
enc1 -> enc2 -> enc3 -> bottleneck -> basis-generated adapter residual -> dec2 -> dec1 -> head
```

The adapter is inserted only at the UNet bottleneck in v1:

```text
adapter(b, z) = sum_m softmax(head(z))_m * AdapterBasis_m(b)
output = b + adapter_scale * adapter(b, z)
```

Each adapter basis is a lightweight residual block:

```text
1x1 conv: C -> adapter_bottleneck
GELU
1x1 conv: adapter_bottleneck -> C
```

Defaults:

- `width=32`
- bottleneck channels `C=width*4=128`
- `prompt_dim=64`
- `hyper_n_basis=8`
- `hyper_adapter_bottleneck=32`
- `hyper_adapter_scale=1.0`

## Why Bottleneck-Only First

Bottleneck-only adapter generation is the lowest-risk HyperDA implementation:

- it keeps the shared UNet topology unchanged;
- it adds a clear generated-operator residual rather than a large generated model;
- it keeps compute manageable because the bottleneck feature map has lower spatial resolution;
- it gives a clean ablation against FiLM prompt-conditioning.

The first implementation intentionally does not stack FiLM and generated adapters. This makes the method attribution cleaner: `prompt_conditioned` is the FiLM baseline; `hyperda_basis_adapter` is the generated-adapter method.

## Training Protocol

The first HyperDA run should mirror `phase4_prompt_conditioned`:

- wrapper: `run/phase4_hyperda.sh`
- default target region: `US-R1`
- default seed: `0`
- adaptation setting: `target_full_train`
- split artifact: `artifacts/splits/US_loro_target_train_splits.json`
- no `--K` argument
- training uses `source_fit`
- checkpoint selection uses `source_val`
- default selection metric: `source_val_transfer_safe_score`
- target eval is not used for training, early stopping, or model selection

## Implementation Boundaries

Implement HyperDA v1 by adding a model type to the existing prompt-conditioned trainer rather than copying the trainer. This keeps data loading, normalization, checkpointing, source-val evaluation, and summary metadata identical across the FiLM and HyperDA variants.

New or modified files:

- `hydroda/models/hyper_adapters.py`: basis-generated bottleneck adapter modules.
- `hydroda/models/hyper_conditional_unet.py`: HyperDA bottleneck-adapter UNet.
- `scripts/train/train_prompt_conditioned_shared.py`: CLI/model factory metadata additions.
- `hydroda/baselines/prompt_conditioned.py`: checkpoint auto-load support for HyperDA model type.
- `run/phase4_hyperda.sh`: one-region/one-seed training wrapper.
- `docs/EXPERIMENT_LOGGING.md`: reusable experiment record entry.
- tests under `tests/`.

## Success Criteria

- Unit tests verify adapter coefficient shape, output shape, and parameter gradients.
- Smoke tests verify trainer summary/checkpoint metadata records `model_type=hyperda_basis_adapter`.
- Protocol text tests verify `run/phase4_hyperda.sh` uses `target_full_train`, `source_val_transfer_safe_score`, current split JSON, and no `--K`.
- Compile checks pass for the changed Python modules.

## Initial Experiment Record

Initial run command:

```bash
bash run/phase4_hyperda.sh US-R1 0 1
```

Follow-up evaluation command after training:

```bash
bash run/phase4_prompt_conditioned_inference.sh <hyperda_checkpoint> US-R1 0 1
```

The inference script can be reused because `PromptConditionedBackbonePredictor` auto-loads the checkpoint model type.
