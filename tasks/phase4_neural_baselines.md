# Phase 4 — Neural Baselines

## 目标

在 simple baselines 稳定之后，实现透明的 neural DA increment baselines。

## 优先级

```text
1. small_conv_da_operator
2. small_unet_da_operator
3. source_only neural training
4. full fine-tuning
5. head / bias / AdaBN / adapter / LoRA
6. pooled_sformer_wrapper
```

## 需要实现

```text
hydroda/models/small_unet.py
hydroda/models/adapters.py
hydroda/experiments/runner.py
hydroda/experiments/evaluate.py
scripts/train_neural.py
scripts/evaluate_checkpoint.py
tests/test_neural_forward.py
tests/test_overfit_tiny_batch.py
```

## 禁止

```text
不要绕过 HydroDADataset
不要绕过 split manifest
不要使用 query labels early stopping
不要直接使用 collaborator training loop
```

## 验收标准

```text
1. forward pass shape correct。
2. tiny batch overfit 可以降低 loss。
3. evaluation 使用同一 metrics pipeline。
4. checkpoints 保存 config、git hash、split manifest path。
```
