# 01_RESEARCH_CONTRACT.md — HyperDA V4 冻结研究契约

本文件是 `hydroda_ood` 的 V4 研究契约。若旧文档仍出现 HyRAO、K=24、ridge 主表、source mean 主表、sparse Hessian/Fisher 主方法等设定，以本文件为准。

## 1. 论文定位

本文研究的是 neural land DA analysis-increment emulation，而不是 ordinary soil moisture prediction。

核心表述：

```text
HydroDA-OOD is a leakage-controlled cross-continental benchmark for neural land DA increment emulation.
HyperDA-SAFE is a hydroclimatic spatio-temporal prompt-conditioned hypernetwork with Source-Anchored Few-Shot Operator Refinement for generating target-specific lightweight neural DA increment operators.
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

HyperDA-SAFE 的核心不是 feature-level conditioning，也不是 final-output
residual patch，而是 parameter-space transfer with a source-anchored target
operator update：

```text
ζ_R = H_ψ(P_R)
f_{θ0, ζ_R}(x_R) -> ΔSM_hat_R
```

只生成 lightweight parameters：adapter、output-head residual、optional FiLM。

第一版使用 deterministic basis-factorized generation：

```text
ζ_R,l = ζ_0,l + Σ_m α_R,l,m B_l,m
```

当前 ablation-capable source-stage 实现允许在不生成 full backbone
parameters 的前提下使用 layer-aware rank-gated basis coefficients 和
identity-centered DoRA-like basis gain。该能力仍属于 lightweight
parameter-space operator generation；是否进入主方法必须由 source_val
selection 和 target_eval final evidence 支撑，不能用 target_val/target_eval
调参。

当前固定的 HyperDA-SAFE source prior 主线是：

```text
M2_1_rank_gated_dora_stable
stable rank-gated bounded-DoRA HyperDA prior + SAFE refinement
shared_layer_aware_rank_gated_stable
dora_like_gain_bounded
temperature `2.0`
`USE_AMP=0`
`LR=2e-4`
```

原始 `M2_rank_gated_dora` 记录为
`retired_failed_exploration_not_paper_main`，原因是 AMP skip/numerical
failure；不得作为论文主线候选。`M2_2_source_saliency_prior` 暂降级为
secondary diagnostic；其 source saliency prior 只能作为 legacy diagnostic
gate-logit bias 或 soft metadata，不得默认改变 M2.1 hard top-k routing。
`M2_3_source_safe_residual_hyperda` 是 source-safe residual diagnostic
ablation，从 M2.1 启动并使用
`pred = source_base + rho * reliability_gate(prompt, context) * hyper_residual`；
`rho` 与 safe score 只能由 source_val/source-side episodes 选择，K=0 不使用
target labels，target_val 仍为 `unused_in_main_protocol`。当前 US-R1 K=0
证据显示其 source-side safe score 提升但 target_eval RMSE 差于 M2.1，因此
M2.3 记录为 negative/diagnostic ablation，不替换 M2.1。
`M2_5a_da_aware_prompt_only` 保留为
`negative_diagnostic_non_strict_prompt_only`：已有 artifact 显示 source_val
提升但 US-R1 K=0 target_eval 变差，且该运行使用 `direct_hyper + rho=0.0`，
并非相对 M2.1 `source_base_residual_reliability_gated + rho=1.0` 的严格
prompt-only 对照；其 DA-aware diagnostics 还在 normalized tensor domain 上计算，
不再是物理 raw TB O-F innovation / contrast。`M2_5b_da_aware_conservative_router`
是后续 source-stage diagnostic：保持 M2.1 的
`shared_layer_aware_rank_gated_stable`、`dora_like_gain_bounded`、top-k=4、
temperature `2.0`、`USE_AMP=0` 和 `LR=2e-4`，使用
`context_encoder=robust_input_side_da_diagnostics_raw`，prompt diagnostics
从 raw input-side `x` 计算，backbone 继续使用 `x_norm`；同时恢复
`zero_shot_prior_form=source_base_residual_reliability_gated`、
`source_residual_rho=1.0`、`SOURCE_RESIDUAL_GATE_INIT=0.90`，并启用
`hyper_prompt_manifold_reliability=1`、strength `0.25`。允许特征只来自
input-side `target_context`/当前输入字段：TB H/V innovation、observation-error
confidence、H/V polarization contrast、vegetation opacity、soil/surface
temperature、surface/rootzone forecast state、finite coverage，以及 bounded
`base_valid_mask` diagnostic coverage。禁止读取 target labels、target_val、
target_eval statistics；channel 11 不得作为 loss/metric/obs/region hard mask。
`M2_4_target_context_conservative_hyperda` 是 Stage 3 K=0 target-context
conservative shrinkage diagnostic，不是 source-stage ablation：冻结 M2.1
source prior，不继续 source fine-tune，用 target_context input-only
reliability 做 post-hoc residual shrinkage，并通过
leave-one-source-region pseudo-target episodes 选择 worst-case
non-degradation vs M2.1；必须记录
`target_labels_used_for_adaptation=false`、`target_val_usage=unused_in_main_protocol`、
`target_eval_usage=final_eval_only_no_selection`、
`target_eval_input_stats_used_for_update=false`。Hessian/Fisher/top-parameter
selection 不进入主线，保留为 future source-side ablation。

当前 active source-stage 训练入口是 staged HyperDA：

```text
run/phase4_hyperda_staged.sh
```

Stage 1 使用 `source_pooled_global_backbone` source-only checkpoint。Stage 2
把该 source base backbone/head 冻结，只训练 target/context prompt encoder、
FiLM 和 basis-adapter generation modules（`trainable_scope =
source_base_frozen_adapter_film`）。旧 scratch `phase4_hyperda.sh` 仅作为兼容
wrapper 转发到 staged 主线。

K=0 使用 source-trained HyperDA prior 和 target_context monthly prompt
prototypes。K=4/12 只在 K 个 labeled target support DA cycles 上更新
target-specific lightweight variables，并保存 source-anchored refinement：

```text
θ_SAFE = θ_prior + α_K (θ_adapt - θ_prior)
```

`α_K`、steps、learning rate、rootzone non-degradation guard 只能来自
source_val/source-side preregistered rules，不能使用 target_val 或
target_eval labels 选择。论文主协议的 K=4/K=12 wrapper 必须读取 source-side
episode calibration 导出的 `safe_policy.json`，并记录
`policy_source=source_side_episode_calibration`、source episode regions、
policy hash、support manifest hash 和 output blend `adapt_mix_rho`。缺失该
policy 的 K-shot run 只能作为 explicit diagnostic，不进入 paper-facing
HyperDA-SAFE 结果。该方法在论文中命名为：

```text
HyperDA-SAFE: Source-Anchored Few-Shot Operator Refinement
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

以下失败探索已退休，状态为
`retired_failed_exploration_not_paper_main`，不得作为论文主方法、主表
baseline 或默认 run entrypoint：

```text
phase6_surface_residual_ridge
phase6_bora_residual_adapter
phase7_hyperda_apo
```

这些 artifacts 可作为内部失败证据保留，但它们是 not a paper-main method。

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
