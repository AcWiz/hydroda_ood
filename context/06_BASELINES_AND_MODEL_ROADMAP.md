# 06 Baselines and Model Roadmap

## 1. Baseline-first 原则

在 Forecast-only、mean increment、monthly mean、ridge 没跑通前，不要实现复杂模型。

顶会审稿人会先问：

```text
这个任务是否比简单 bias correction 难？
K-date support mean 是否已经足够？
Ridge 是否能吃掉大部分收益？
```

---

## 2. Phase 3 必须实现的简单 baseline

### forecast

```text
pred_increment = 0
pred_analysis = forecast
```

### source_mean_increment

用 source train regions 的平均 increment。

### target_support_mean_increment

K>0 时，用 target support dates 的平均 increment。

### target_monthly_support_increment

K=12/24 时，用 target support dates 的 monthly increment mean。

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

## 4. Phase 5 HyRAO

HyRAO 定义：

```text
z_r = g_phi(d_r)
pred_increment = f_theta(x, z_r)
```

其中 `d_r` 是 input-only region descriptor，包括：

```text
forecast climatology
TB statistics
vegopacity statistics
mask / missingness statistics
time coverage statistics
optional static covariates
```

K=0：

```text
只用 input-only descriptor 初始化或调制模型。
```

K>0：

```text
使用 target support labels 适配 z_r 和少量 region-specific modules。
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
