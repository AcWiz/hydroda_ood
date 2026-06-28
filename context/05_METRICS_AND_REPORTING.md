# 05 Metrics and Reporting

## 1. 主指标

Analysis-space skill relative to forecast：

```text
Skill = 1 - RMSE(pred_analysis, analysis) / RMSE(forecast, analysis)
```

其中：

```text
pred_analysis = forecast + pred_increment
```

必须分别报告：

```text
surface
rootzone
```

---

## 2. Increment-space metrics

必须报告：

```text
increment_rmse
increment_mae
increment_bias
increment_corr
sign_accuracy_deadzone
top20_increment_rmse
top20_increment_skill
```

Dead-zone sign accuracy：当 `abs(true_increment) < epsilon` 时不计入 sign accuracy。

---

## 3. Adaptation metrics

主协议当前按 K=0 zero-shot 报告 HyperDA Operator Generator 和 HyperDA-TRUST，
并与 source-only / prompt-conditioned shared baseline 比较。K=4/K=12 SAFE
结果保留为 diagnostic / future extension；若为 `rejected_to_k0_anchor`，不得
计入 few-shot improvement。Full-target 结果只用于 legacy/internal reproduction：

```text
AdaptationGain(K) = Metric(adapted, K) - Metric(source_only, K=0)
CE12 = legacy calibration efficiency from K=0 to K=12
AUC-K = area under K-performance curve for K in {0,4,12}
```

具体方向由 metric 决定。对于 skill 越大越好；对于 RMSE 越小越好。

---

## 4. Aggregation rules

默认主表使用 region-balanced aggregation：

```text
先对每个 region 求平均，再对 region 求平均。
```

不要让大区域在主表中支配小区域。

同时输出 pixel-weighted 结果作为附录。

---

## 5. Required result tables

Phase 3 起必须逐步产生：

```text
Table 1: Forecast-only and source-only sanity by held-out target
Table 2: Main K=0 comparison across source-only, prompt-conditioned, HyperDA Operator Generator, HyperDA-TRUST
Table 3: SAFE K=4/K=12 diagnostic table with rejected_to_k0_anchor status when applicable
Table 4: Legacy full-target reproduction (secondary/internal)
Table 5: Event/high-update analysis after final model selection
```

---

## 6. Event analysis rules

High-update event analysis 只能在模型训练和选择完成后进行。

允许用于 event selection：

```text
query true increment magnitude after final model selection
rainfall-pulse proxy if input-only
seasonal windows
predefined scientific event types
```

不能用 event results 重新选模型或调参。
