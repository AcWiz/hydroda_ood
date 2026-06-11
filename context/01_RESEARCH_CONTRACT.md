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
source_fit:      2015-2021
source_val:      2022
target_context:  2015-2021 input-side only
target_support:  K labeled target DA cycles, K in {0,4,12}
target_val:      unused in main protocol
target_eval:     2023-2025
```

主协议：

```text
K ∈ {0,4,12}
adaptation_setting ∈ {zero_shot_context, few_shot_k4, few_shot_k12}
```

目标域 2015-2021 只作为 input-side `target_context` 构造 target-context
monthly prompt prototypes。K=0 不使用目标域标签；K=4/12 只允许在 K 个 labeled
target DA cycles 上更新轻量 target-specific 变量，并在 forward 中继续使用同一
monthly context prototype 策略。主协议不使用 `target_val` 做 checkpoint
selection、early stopping 或 gain calibration。2023-2025 target_eval labels
只用于最终离线评估。

`month` 是部署时已知的 month-of-year seasonal phase，只用于选择 monthly
prototype；它不是绝对日期标签，也不是 target_eval 模型选择信号。

`target_full_train` 仅作为 legacy/internal reproduction 路径，必须显式 opt-in。
K 表示 labeled target DA analysis cycles，不是 patches/pixels/mini-batches。

主协议 source 覆盖范围保持 `source_fit=2015-2021`、`source_val=2022`。
2022 继续用于 source-domain checkpoint / early stopping / hyperparameter selection；
更长 source fit 只能作为 expanded-source secondary ablation。

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
Source-only backbone (`source_pooled_global_backbone`)
Prompt-conditioned shared backbone
Source-regime specialist bank (`source_regime_specialist_bank`, final
cross-continent source-side same-regime specialists)
HyperDA K=0 zero-shot context prompt
HyperDA K=4 lightweight target adaptation
HyperDA K=12 lightweight target adaptation
```

以下只允许作为 internal sanity check，不进入论文主表：

```text
source_mean_increment
target_train_mean_increment
target_monthly_train_increment
ridge_calibration
nearest-source specialist
prompt-weighted specialist
kNN parameter interpolation
linear prompt-to-parameter
Adapter/LoRA K-shot ablations
legacy full-target HyperDA-Calib/Refine
legacy_all_regions_sanity
target_full_history_region_oracle
```

当前 US-only 过渡实验中，旧 `train_source_only_backbone.py` 的
leave-one-region-out source-only 训练可以作为 `source_pooled_global_backbone`
的开发版 global baseline；`phase4_source_only_all_regions` 因训练全部 US
regions，包含 target region 2015-2021 labels，只能作为 legacy sanity。
region-specific 主 baseline 必须是 source-trained same-regime specialist bank；
在 CN/AU 未落地前，US-only 只能做 source-region expert routing/ensemble sanity，
target-region supervised expert 属于 `target_full_history_region_oracle`。

## 6. 零泄漏契约

禁止 target_eval/query labels 参与 prompt、normalization、target adaptation sample selection、training、early stopping、model selection、threshold calibration 或 prompt feature tuning。
主协议还禁止 target_val/target_eval 参与 target-side selection、early stopping
或 residual-gain calibration。

所有涉及时间、region、split、metric 的代码必须通过 `ProtocolConfig` / `LeakageGuard` 或等价机制进行检查。
