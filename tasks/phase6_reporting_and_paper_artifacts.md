# Phase 6/7/8 — HyperDA、K-cycle Calibration 与论文产物

## 目标

实现 HyperDA-Zero、HyperDA-Calib、HyperDA-Refine，并与 adapter / LoRA 在 K=4/K=12 下公平比较。随后生成 US-only development report 和最终 LOCO paper artifacts。

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
HyperDA-Zero:
  prompt = target 2021 input-side prompt
  labels = none

HyperDA-Calib:
  prompt = target 2021 input-side prompt + K labeled calibration summaries
  K ∈ {4, 12}

HyperDA-Refine:
  initialize ζ_R = H_ψ(P_R)
  freeze θ0 and H_ψ
  update only ζ_R for 10-50 steps on K calibration cycles
```

## 主比较

K=0：

```text
Forecast-only
Source-only backbone
Prompt-conditioned shared backbone
HyperDA-Zero
```

K=4/K=12：

```text
Forecast-only
Source-only backbone
Adapter tuning
LoRA tuning
Prompt-conditioned shared + calibration prompt
HyperDA-Calib
HyperDA-Refine
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
support seed mean ± std / CI
```

## 验收标准

```text
1. HyperDA-Zero 不使用 target labels。
2. HyperDA-Calib 只使用 2021 K-cycle support summaries。
3. HyperDA-Refine 只更新 ζ_R，不更新 θ0 或 Hψ。
4. adapter / LoRA 使用相同 support dates、steps、seed、normalization。
5. 所有表格能从 metrics_long.csv 自动生成。
```
