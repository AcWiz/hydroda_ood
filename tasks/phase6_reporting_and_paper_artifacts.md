# Phase 6/7/8 — HyperDA-TRUST zero-shot 与论文产物

## 目标

实现 HyperDA K=0 operator generation 与 HyperDA-TRUST K=0 trust routing，并与
forecast/source-only/prompt-conditioned baseline 公平比较。SAFE K=4/K=12 保留为
diagnostic / future extension；当前若为 `rejected_to_k0_anchor`，必须报告为
K0-equivalent fallback，不是 accepted few-shot adaptation。2022 target_val 主协议不使用，2023-2025 target_eval 只用于最终评估。
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

HyperDA-TRUST-K0:
  source-only trust bank + nearest-source coefficient consensus
  route target-prompt operator coefficients without target labels

SAFE-FewShot-K4/K12 diagnostic:
  report only with stage3_posterior_decision
  rejected_to_k0_anchor means K0-equivalent fallback
```

## 主比较

zero-shot main：

```text
Forecast-only
Source-only backbone
Prompt-conditioned shared backbone
HyperDA Operator Generator K=0
HyperDA-TRUST K=0
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
1. HyperDA K=0 / HyperDA-TRUST K=0 不使用 target labels；K=4/12 SAFE 只作为 diagnostic / future extension。
2. 2023-2025 target_eval labels 只用于最终评估。
3. SAFE diagnostic rows expose `stage3_posterior_decision`; `rejected_to_k0_anchor` 不写成 few-shot gain。
4. adapter / LoRA ablation 使用相同 target_support dates、steps、seed、normalization。
5. 所有表格能从 metrics_long.csv 自动生成。
```
