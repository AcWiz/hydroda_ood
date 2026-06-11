# 12_PROTOCOL_V4_FINAL_UPDATE.md — Protocol V4.4 zero/few-shot update

本文件记录最新冻结协议修订。若旧文档、旧任务、旧 artifacts 中仍出现
V4.1 K-shot / few-cycle 或 V4.2/V4.3 full-target 主协议引用，以本文件、`CLAUDE.md`、
`完整研究计划方案.md`、`specs/protocol_v4.yaml` 为准。

## 新冻结协议

```text
source_fit:     2015-01-01 to 2021-12-31  source continents only
source_val:     2022-01-01 to 2022-12-31  source continents only
target_context: 2015-01-01 to 2021-12-31  held-out target input-side monthly prompt prototype context
target_support: K labeled cycles from 2015-2021, K in {0,4,12}
target_val:     unused in main protocol
target_eval:    2023-01-01 to 2025-12-31  final offline evaluation only
```

## 与 V4.1 的差异

V4.1 把目标域泛化定义为 K-cycle few-shot calibration：`K ∈ {0,4,12}`，
2022 是少量 support/context，2023-2025 是 query/evaluation。

V4.4 主协议改为：

```text
adaptation_setting ∈ {zero_shot_context, few_shot_k4, few_shot_k12}
target_context_dates = 2015-2021 input-side target context
target_support_dates = K labeled target cycles from 2015-2021
target_eval_dates = held-out 2023-2025
```

K=0 不使用 target labels；K=4/12 只允许轻量 target-specific 变量在 K 个 support
cycles 上更新。Source/HyperDA 训练完成后，source backbone、prompt encoder、
HyperDA basis/hypernetwork 冻结，然后严格在 2023-2025 target_eval 上评估。

## Source 覆盖范围决定

主协议保持 `source_fit=2015-2021`、`source_val=2022`。2022 保留用于 source-domain
checkpoint selection、early stopping、hyperparameter selection 和 architecture selection。
纳入 source 2022 的 refit 只能作为明确标注的 expanded-source secondary ablation，
不能混入主表。

## 关键纪律

`target_eval=2023-2025` labels 不能用于 training、prompt construction、
normalization、target adaptation sample selection、early stopping、model selection、
hyperparameter selection、threshold calibration、prompt feature tuning 或 region
definition。

`target_support=2015-2021` labels 只按 K 预算用于 few-shot lightweight adaptation。
`target_val=2022` 不参与主协议 target-side selection、early stopping 或 gain
calibration。主协议 global normalization 仍保持 `source_fit_only`，checkpoint /
hyperparameter 选择来自 source_val。

## Legacy K-shot

`target_full_train` 只保留为 legacy/internal reproduction，必须显式 opt-in。
新表格和实验名必须使用 `adaptation_setting`，并记录
`target_context_dates_hash`、`target_support_dates_hash`、`target_eval_dates_hash`
和 `split_manifest_sha256`。
