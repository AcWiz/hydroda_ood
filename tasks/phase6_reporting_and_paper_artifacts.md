# Phase 6/7/8 — HyperDA zero/few-shot adaptation 与论文产物

## 目标

实现 HyperDA zero/few-shot generation/adaptation，并在 K=0/4/12 标签预算下公平比较。
目标阶段只训练轻量 target-specific prompt / adapter coefficient residuals / monthly
residual gain；2022 target_val 主协议不使用，2023-2025 target_eval 只用于最终评估。
full-target 结果只作为 legacy/internal reproduction。随后生成 US-only
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
scripts/train/train_hyperda_few_shot_adapt.py
run/phase5_hyperda_zero_few_shot.sh
scripts/run_kcycle_comparison.py
scripts/make_paper_tables.py
scripts/make_paper_figures.py
```

## HyperDA variants

```text
HyperDA-ZeroShot-Context:
  prompt = target-context monthly prompt prototypes from 2015-2021 input-side context
  labels = none

HyperDA-FewShot-K4/K12:
  initialize zeta_R = H_psi(P_R)
  freeze theta0, H_psi, adapter basis bank
  train target_prompt, adapter coefficient residuals, monthly residual gain
  use fixed preregistered steps, no target_val early stopping
```

## 主比较

zero/few-shot main：

```text
Forecast-only
Source-only backbone
Prompt-conditioned shared backbone
HyperDA K=0
HyperDA K=4
HyperDA K=12
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
target_context/support dates hash / target_eval dates hash / split manifest hash
seed mean ± std / CI
```

## 验收标准

```text
1. HyperDA K=0 不使用 target labels；K=4/12 只使用 K 个 target_support cycles。
2. 2023-2025 target_eval labels 只用于最终评估。
3. HyperDA-Adapt / Refine 不更新 θ0、Hψ 或 adapter basis bank。
4. adapter / LoRA ablation 使用相同 target_support dates、steps、seed、normalization。
5. 所有表格能从 metrics_long.csv 自动生成。
```
