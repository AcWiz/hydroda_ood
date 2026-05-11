# Phase 4 — Strong Shared Neural Baselines

## 目标

实现能真正挑战 HyperDA 的强 neural baselines，而不是堆 heuristic baseline。Phase 4 的核心是：

```text
Source-only backbone
Prompt-conditioned shared backbone
```

Prompt-conditioned shared backbone 是最关键对手。如果 HyperDA 不能超过它，就不能证明 parameter generation 必要。

## 必读

```text
context/01_RESEARCH_CONTRACT.md
specs/baselines.yaml
specs/hyperda_v4.yaml
specs/protocol_v4.yaml
checklists/no_leakage_checklist.md
```

## 需要实现

```text
hydroda/models/resunet.py
hydroda/models/conditional_unet.py
hydroda/models/prompt_encoder.py
hydroda/training/losses.py
hydroda/experiments/runner.py
scripts/train_source_backbone.py
scripts/train_prompt_conditioned_shared.py
scripts/evaluate_checkpoint.py
tests/test_neural_forward.py
tests/test_tiny_overfit.py
```

## Backbone 策略

第一版优先：

```text
Small ResUNet
ConvNeXt-UNet optional
```

暂不优先：

```text
Sformer wrapper
large transformer backbone
```

原因：当前要证明 parameter generation，而不是 backbone capacity。

## Baselines

```text
source_only_backbone:
  train: source regions / continents, 2015-2020
  target: no prompt, no target labels

prompt_conditioned_shared:
  train: source regions / continents, 2015-2020
  input: x + region prompt token/feature map
  target K=0: target 2022 input-side prompt only
  target K>0: input-side prompt + calibration summary prompt
```

## 禁止

```text
不要绕过 HydroDADataset
不要绕过 ProtocolConfig / LeakageGuard
不要使用 target query labels early stopping
不要复用 collaborator training loop 中任何未审计 split 逻辑
不要在 source-only 中注入 target prompt
```

## 验收标准

```text
1. forward pass shape: [B, C_in, H, W] -> [B, 2, H, W]
2. loss mask 正确应用。
3. tiny batch overfit loss 可下降。
4. checkpoint 保存 config、protocol freeze id、split manifest path、git hash。
5. evaluation 使用 Phase 3 同一 metrics pipeline。
```
