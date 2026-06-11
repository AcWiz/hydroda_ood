# 阶段 5：HyperDA zero/few-shot adaptation

## 目标

在强 baseline 完成后，实现 target-specific HyperDA / adapter / LoRA adaptation。

## 组件

```text
Prompt / target_context summary encoder
Generated lightweight operator
Adapter tuning
Target latent
Adapter coefficient residuals
Residual gain
Gradient Top-K block tuning
FISA
HISA optional
```

## 任务

1. 从 target_context 2015-2021 input stream 构建 prompt；K=4/12 只使用 K 个
   target_support labels；target 2022 主协议不用于 adaptation selection。
2. 实现 target-specific generated operator initialization。
3. 实现 zero/few-shot lightweight adaptation loop。
4. 实现 adapter tuning。
5. 实现 block-level gradient scores。
6. 实现 FISA score。
7. 记录 trainable parameter ratio。

## 验收标准

- K=0 不使用 target labels；K=4/12 只使用 K 个 target_support labels。
- target 阶段冻结 Hψ / θ0 / adapter basis bank，只训练 target latent / adapter
  coefficient residuals / residual gain / registered lightweight residuals。
- target_val=2022 主协议不用于 checkpoint / step / residual gain 选择。
- 2023-2025 target_eval labels 只用于最终评估。
- 所有方法使用相同 target_context/support/eval manifest。
- Adaptation logs 记录 support loss，但不能用 target_eval metrics 做选择。
- 报告 run seed 和 support seed 方差。
