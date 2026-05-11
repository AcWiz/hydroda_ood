# 阶段 5：HyRAO-Meta 与稀疏适配

## 目标

在强 baseline 完成后，实现 region-conditioned adaptation。

## 组件

```text
Region descriptor encoder
Region latent tuning
Adapter tuning
Gradient Top-K block tuning
FISA
HISA optional
```

## 任务

1. 从 target 2022 input-only stream 构建 descriptors。
2. 实现 region latent initialization。
3. 实现 support adaptation loop。
4. 实现 adapter tuning。
5. 实现 block-level gradient scores。
6. 实现 FISA score。
7. 记录 trainable parameter ratio。

## 验收标准

- K=0 不使用 target analysis labels。
- K>0 只使用 2022 support labels。
- 所有方法使用相同 support dates。
- Adaptation logs 同时记录 support loss 和 query metrics。
- 报告 support seeds 方差。
