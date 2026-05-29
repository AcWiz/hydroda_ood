# 12_PROTOCOL_V4_FINAL_UPDATE.md — Protocol V4.3 historical-target-adaptation update

本文件记录最新冻结协议修订。若旧文档、旧任务、旧 artifacts 中仍出现
V4.1 K-shot / few-cycle 或 V4.2 target_train=2022 主协议引用，以本文件、`CLAUDE.md`、
`完整研究计划方案.md`、`specs/protocol_v4.yaml` 为准。

## 新冻结协议

```text
source_fit:   2015-01-01 to 2021-12-31  source continents only
source_val:   2022-01-01 to 2022-12-31  source continents only
target_train: 2015-01-01 to 2021-12-31  held-out target continent; full historical training/adaptation
target_val:   2022-01-01 to 2022-12-31  held-out target continent; preregistered adaptation validation only
target_eval:  2023-01-01 to 2025-12-31  final offline evaluation only
```

## 与 V4.1 的差异

V4.1 把目标域泛化定义为 K-cycle few-shot calibration：`K ∈ {0,4,12}`，
2022 是少量 support/context，2023-2025 是 query/evaluation。

V4.3 主协议改为：

```text
adaptation_setting = target_full_train
target_adaptation_dates = full target_train 2015-2021
target_val_dates = held-out target 2022, only if preregistered for adaptation validation
target_eval_dates = held-out 2023-2025
```

目标域不再只使用少量 support samples。HyperDA / prompt / adapter / LoRA /
generated parameters 可以在 source/HyperDA 训练完成后使用 held-out target domain
完整 2015-2021 historical target_train 样本构造 target-specific operator，然后
严格在 2023-2025 target_eval 上评估。

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

`target_train=2015-2021` labels 可以用于 target-specific adaptation/generalization。
`target_val=2022` 是否用于 adaptation step / regularization / early stopping 选择
必须预注册。主协议 global normalization 仍保持 `source_fit_only`，主要 checkpoint/hyperparameter
选择仍来自 `source_val_only`，除非方法变体提前注册。

## Legacy K-shot

旧 `K ∈ {0,4,12}` 设定只保留为 secondary few-shot ablation。`target_support` 字段
只作为旧代码 alias；新表格和实验名必须使用 `adaptation_setting`，并记录
`target_train_dates_hash`、`target_eval_dates_hash` 和 `split_manifest_sha256`。
