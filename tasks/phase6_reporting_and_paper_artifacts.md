# Phase 6/7/8 — HyperDA full-target-train adaptation 与论文产物

## 目标

实现 HyperDA full-target-train generation/adaptation/refinement，并与 adapter / LoRA
在完整 2015-2021 target_train 下公平比较。主方法不是 zero-shot；目标阶段训练
target-specific latent / operator residual / residual gain，2022 target_val 只用于
预注册 adaptation selection，2023-2025 target_eval 只用于最终评估。旧
HyperDA-Zero/Calib/Refine K=0/4/12 只作为 secondary ablation。随后生成 US-only
development report 和最终 LOCO paper artifacts。

## 需要实现

```text
hydroda/models/hyperda.py
hydroda/models/parameter_basis.py
hydroda/models/hyperda_decoder.py
hydroda/models/prompt_encoder.py
hydroda/adaptation/adapter_tuning.py
hydroda/adaptation/lora_tuning.py
hydroda/adaptation/hyperda_refine.py
hydroda/models/target_adaptation.py
scripts/train/train_hyperda_target_adapt.py
run/phase5_hyperda_target_adapt.sh
scripts/run_kcycle_comparison.py
scripts/make_paper_tables.py
scripts/make_paper_figures.py
```

## HyperDA variants

```text
HyperDA-FullTargetTrain:
  prompt = target_train 2015-2021 input-side prompt + full target_train labeled summaries
  labels = all available 2015-2021 target_train cycles

HyperDA-Adapt-FullTargetTrain:
  initialize zeta_R = H_psi(P_R)
  freeze theta0, H_psi, adapter basis bank
  train target latent, adapter coefficient residuals, residual gain on target_train
  select adaptation step / checkpoint only with target_val=2022

HyperDA-Refine-FullTargetTrain:
  initialize ζ_R = H_ψ(P_R)
  freeze θ0, H_ψ, and adapter basis bank
  update target latent, adapter coefficient residuals, lightweight operator residual,
    output-head residual, and residual gain for pre-registered steps on target_train
```

## 主比较

target_full_train：

```text
Forecast-only
Source-only backbone
Adapter tuning
LoRA tuning
Prompt-conditioned shared + calibration prompt
HyperDA-FullTargetTrain
HyperDA-Adapt-FullTargetTrain
HyperDA-Refine-FullTargetTrain
```

## 报告要求

```text
Surface Skill
Rootzone Skill
Increment RMSE
Increment correlation
High-update Skill
trainable parameter count
adaptation steps
wall-clock time
target_train dates hash / target_eval dates hash / split manifest hash
seed mean ± std / CI
```

## 验收标准

```text
1. HyperDA full-target-train 只使用 2015-2021 target_train labels 进行 target-specific operator 构造。
2. 2023-2025 target_eval labels 只用于最终评估。
3. HyperDA-Adapt / Refine 不更新 θ0、Hψ 或 adapter basis bank。
4. adapter / LoRA 使用相同 target_train dates、steps、seed、normalization。
5. 所有表格能从 metrics_long.csv 自动生成。
```
