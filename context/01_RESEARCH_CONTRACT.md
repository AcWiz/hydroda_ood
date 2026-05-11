# 01_RESEARCH_CONTRACT.md — HyperDA V4 冻结研究契约

本文件是 `hydroda_ood` 的 V4 研究契约。若旧文档仍出现 HyRAO、K=24、ridge 主表、source mean 主表、sparse Hessian/Fisher 主方法等设定，以本文件为准。

## 1. 论文定位

本文研究的是 neural land DA analysis-increment emulation，而不是 ordinary soil moisture prediction。

核心表述：

```text
HydroDA-OOD is a leakage-controlled cross-continental benchmark for neural land DA increment emulation.
HyperDA is a hydroclimatic spatio-temporal prompt-conditioned hypernetwork for generating target-specific lightweight neural DA increment operators.
```

## 2. 任务定义

对于每个 DA analysis cycle：

```text
ΔSM_surface  = SM_surface_analysis  - SM_surface_forecast
ΔSM_rootzone = SM_rootzone_analysis - SM_rootzone_forecast
```

模型输出 increment，并通过 forecast + increment 重建 estimated analysis。

## 3. 主实验协议

最终主实验是 US/CN/AU leave-one-continent-out：

```text
US + CN -> AU
US + AU -> CN
CN + AU -> US
```

时间协议：

```text
source_fit:      2015-2020
source_val:      2021
target_context:  2022
target_query:    2023-2025
```

K-cycle calibration：

```text
K ∈ {0, 4, 12}
```

K 表示 labeled target DA analysis cycles，不是 patches/pixels/mini-batches。

## 4. 方法契约

HyperDA 的核心不是 feature-level conditioning，而是 parameter-space transfer：

```text
ζ_R = H_ψ(P_R)
f_{θ0, ζ_R}(x_R) -> ΔSM_hat_R
```

只生成 lightweight parameters：adapter、output-head residual、optional FiLM。

第一版使用 deterministic basis-factorized generation：

```text
ζ_R,l = ζ_0,l + Σ_m α_R,l,m B_l,m
```

## 5. Baseline 契约

论文主表只保留：

```text
Forecast-only
Source-only backbone
Prompt-conditioned shared backbone
Adapter tuning
LoRA tuning
HyperDA-Zero
HyperDA-Calib
HyperDA-Refine
```

以下只允许作为 internal sanity check，不进入论文主表：

```text
source_mean_increment
target_support_mean_increment
monthly_mean_increment
ridge_calibration
nearest-source specialist
prompt-weighted specialist
kNN parameter interpolation
linear prompt-to-parameter
```

## 6. 零泄漏契约

禁止 target query labels 参与 prompt、normalization、support selection、early stopping、model selection、threshold calibration 或 prompt feature tuning。

所有涉及时间、region、split、metric 的代码必须通过 `ProtocolConfig` / `LeakageGuard` 或等价机制进行检查。
