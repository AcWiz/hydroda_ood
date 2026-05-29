# Phase 6/7/8 — HyperDA full-target-train adaptation 与论文产物

## 目标

实现 HyperDA full-target-train generation/refinement，并与 adapter / LoRA 在完整 2015-2021 target_train 下公平比较。旧 HyperDA-Zero/Calib/Refine K=0/4/12 只作为 secondary ablation。随后生成 US-only development report 和最终 LOCO paper artifacts。

## 需要实现

```text
hydroda/models/hyperda.py
hydroda/models/parameter_basis.py
hydroda/models/hyperda_decoder.py
hydroda/models/prompt_encoder.py
hydroda/adaptation/adapter_tuning.py
hydroda/adaptation/lora_tuning.py
hydroda/adaptation/hyperda_refine.py
scripts/run_hyperda_zero.py
scripts/run_hyperda_calib.py
scripts/run_hyperda_refine.py
scripts/run_kcycle_comparison.py
scripts/make_paper_tables.py
scripts/make_paper_figures.py
```

## HyperDA variants

```text
HyperDA-FullTargetTrain:
  prompt = target 2022 input-side prompt + full target_train labeled summaries
  labels = all available 2015-2021 target_train cycles

HyperDA-Refine-FullTargetTrain:
  initialize ζ_R = H_ψ(P_R)
  freeze θ0 and H_ψ
  update only ζ_R for pre-registered steps on full target_train cycles
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
3. HyperDA-Refine 只更新 ζ_R，不更新 θ0 或 Hψ。
4. adapter / LoRA 使用相同 target_train dates、steps、seed、normalization。
5. 所有表格能从 metrics_long.csv 自动生成。
```
