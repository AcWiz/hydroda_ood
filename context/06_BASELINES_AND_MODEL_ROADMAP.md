# 06 Baselines and Model Roadmap

## 1. Baseline-first 原则

在 Forecast-only、source-only、prompt-conditioned shared 和 HyperDA K=0/4/12
没有在同一 V4.4 zero/few-shot split 上跑通前，不要声称 HyperDA 有主表优势。
mean increment、monthly mean、ridge 和 full-target/adapter/LoRA 结果保留为
internal sanity 或 appendix ablation。

顶会审稿人会先问：

```text
这个任务是否比简单 bias correction 难？
少量 target_support labels 是否已经足够？
Ridge 或简单校准是否能吃掉大部分收益？
```

---

## 2. Phase 3/4 baseline ladder

### forecast

```text
pred_increment = 0
pred_analysis = forecast
```

### source_mean_increment

用 source fit regions 的平均 increment。

### target_support_mean_increment

Internal sanity only。只允许使用当前 split 的 K 个 target_support labeled cycles。
完整 2015-2021 target_train mean 属于 legacy/full-target reproduction，不进入主表。

### target_monthly_support_increment

Internal sanity only。只允许使用当前 split 的 target_support labeled cycles；完整
target_train monthly mean 属于 legacy/full-target reproduction。

### ridge

输入 features：

```text
surface/rootzone forecast
TB-H / TB-V
TB polarization difference
vegopacity
obs_mask
sin/cos day-of-year
optional lat/lon or row/col encoding
```

目标：surface/rootzone increment。

---

## 3. Phase 4 neural baselines

优先级：

```text
forecast_only
source_pooled_global_backbone  # paper-facing role: Source-only backbone
prompt_conditioned_shared_backbone
source_regime_specialist_bank  # final cross-continent same-regime specialist bank
HyperDA_K0_zero_shot_context
HyperDA_K4_lightweight_target_adaptation
HyperDA_K12_lightweight_target_adaptation
```

`source_pooled_global_backbone` trains only source domains on 2015-2021 and
uses source_val 2022 for checkpoint/model selection. In the final
leave-one-continent-out protocol, source domains are the two non-target
continents. In the current US-only transition, the existing leave-one-region-out
source-only runner is the correct global baseline if it excludes the target US
region from training.

`source_regime_specialist_bank` is the region-specific baseline under the new
protocol: for target continent C and regime Ri, train the Ri specialist only on
source continents' Ri regions, then route by preregistered region/regime ID or
input-only target_context. US-only has no cross-continent same-regime
counterpart, so source-region expert routing/ensemble is internal sanity only.

Legacy/internal neural baselines:

```text
legacy_all_regions_sanity:
  previous wrapper: phase4_source_only_all_regions
  trains all US-R1..R6, including the held-out target region's 2015-2021 labels;
  do not report as OOD global.

target_full_history_region_oracle:
  previous wrapper: phase4_source_only_region_specific
  trains each target region's own 2015-2021 labels; appendix/internal upper
  bound only, not V4.4 zero/few-shot target generalization.
```

Sformer 只能接入我们的 dataset/split/metric，不能沿用合作方 split 或 training logic。

---

## 4. Phase 5 HyperDA zero/few-shot target generalization

HyperDA 定义：

```text
ζ_R = H_ψ(P_R)
pred_increment = f_{θ0, ζ_R}(x_R)
```

其中 `P_R` 是 target-context monthly prompt prototypes，只能包含 2015-2021
target_context input-side descriptor、target region embedding/fallback embedding
和部署时已知的 month-of-year seasonal phase，例如：

```text
forecast climatology
TB statistics
vegopacity statistics
mask / missingness statistics
time coverage statistics
optional static covariates
```

主协议：

```text
K=0: target-context monthly prompt prototypes from input-side context only, no target labels.
K=4/12: fixed K labeled target_support cycles, fixed preregistered steps,
        no target_val early stopping or selection.
```

短期 few-shot adaptation 默认使用 source-anchor recipe：先在 support 上优化轻量
target-specific variables，再保存固定
`theta_init + alpha * (theta_adapt - theta_init)` 候选，不用 target_val 或
target_eval 选择 alpha。当前保守默认值为 K4 `steps=100, lr=1e-3, alpha=0.75`，
K12 `steps=80, lr=3e-4, alpha=0.25`；这些超参只允许来自 source-side episodic
validation / preregistration，并必须写入 metadata。

Legacy/internal reproduction：

```text
target_full_train / full target_train adapters / target_val selection 只作为
explicit opt-in reproduction，不进入主表。
```

---

## 5. Sparse adaptation roadmap

先做 block-level，而不是 scalar parameter-level。

候选 block：

```text
input embedding
observation fusion block
vegopacity/TB gate
encoder stage 1/2
decoder head
normalization affine params
region adapter
```

第一版 score：

```text
score_b = ||grad_b||_2^2
```

第二版 FISA：

```text
score_b = ||grad_b||_2^2 / (fisher_b + lambda)
```

HISA 只有在 FISA 稳定且计算可承受后再做。
