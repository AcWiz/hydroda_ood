# 阶段 5：HyperDA full-target-train adaptation

## 目标

在强 baseline 完成后，实现 target-specific HyperDA / adapter / LoRA adaptation。

## 组件

```text
Prompt / target_train summary encoder
Generated lightweight operator
Adapter tuning
Gradient Top-K block tuning
FISA
HISA optional
```

## 任务

1. 从 target 2022 input stream 和 labeled target_train summaries 构建 prompt。
2. 实现 target-specific generated operator initialization。
3. 实现 full target_train adaptation loop。
4. 实现 adapter tuning。
5. 实现 block-level gradient scores。
6. 实现 FISA score。
7. 记录 trainable parameter ratio。

## 验收标准

- target_full_train 只使用 2015-2021 target_train labels。
- 2023-2025 target_eval labels 只用于最终评估。
- 所有方法使用相同 target_train/eval manifest。
- Adaptation logs 记录 target_train loss，但不能用 target_eval metrics 做选择。
- 报告 run seed 方差；legacy K-shot 才报告 support seed 方差。
