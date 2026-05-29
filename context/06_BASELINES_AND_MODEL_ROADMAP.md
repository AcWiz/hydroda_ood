# 06 Baselines and Model Roadmap

## 1. Baseline-first 原则

在 Forecast-only、source-only、prompt-conditioned shared、adapter/LoRA 没跑通前，不要声称 HyperDA 有主表优势。mean increment、monthly mean、ridge 保留为 internal sanity。

顶会审稿人会先问：

```text
这个任务是否比简单 bias correction 难？
完整 target_train mean 是否已经足够？
Ridge 是否能吃掉大部分收益？
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

### target_train_mean_increment

Internal sanity only。用完整 2015-2021 target_train dates 的平均 increment。

### target_monthly_train_increment

Internal sanity only。用完整 2015-2021 target_train dates 的 monthly increment mean。

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
small_conv_da_operator
small_unet_da_operator
source_only_neural
full_finetune
head_tuning
bias_only
AdaBN
adapter
LoRA
pooled_sformer_wrapper
```

Sformer 只能接入我们的 dataset/split/metric，不能沿用合作方 split 或 training logic。

---

## 4. Phase 5 HyperDA target adaptation

HyperDA 定义：

```text
ζ_R = H_ψ(P_R)
pred_increment = f_{θ0, ζ_R}(x_R)
```

其中 `P_R` 是 target prompt，可包含 input-side descriptor 和 target_train
adaptation summaries，包括：

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
使用完整 target_train=2015-2021 构造 target-specific operator / prompt / adapter。
```

Legacy few-shot ablation：

```text
K=0/4/12 只作为 secondary ablation，不进入主表。
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
