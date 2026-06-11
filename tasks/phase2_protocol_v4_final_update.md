# phase2_protocol_v4_final_update.md — Superseded Protocol V4.2 full-target-train update

Status: superseded by Protocol V4.4 zero/few-shot generalization. Current
main protocol is documented in `specs/protocol_v4.yaml`,
`context/01_RESEARCH_CONTRACT.md`, and `context/12_PROTOCOL_V4_FINAL_UPDATE.md`.
This file is retained only as historical migration context.

## 目标

将项目从 V4.1 K-cycle few-shot 主协议迁移到 V4.2 target full-training-set
adaptation 主协议。

## 新冻结协议

```text
source_fit:   2015-2021
source_val:   2022
target_train: 2015-2021
target_eval:  2023-2025
```

主实验变量：

```text
adaptation_setting = target_full_train
```

`K=0/4/12` 不再是主实验变量，只能作为 `legacy_few_shot_ablation`。

## 设计决定

主协议不扩大 source fit。`source_fit=2015-2021` 保持训练用，`source_val=2022`
保持 checkpoint / early stopping / hyperparameter selection gate。纳入 2021/2022
的 source refit 只能作为 expanded-source secondary ablation。

## 必改文件

- `CLAUDE.md`
- `完整研究计划方案.md`
- `context/01_RESEARCH_CONTRACT.md`
- `context/04_KDATE_SPLIT_PROTOCOL.md`
- `context/12_PROTOCOL_V4_FINAL_UPDATE.md`
- `context/00_EXECUTABLE_CONTEXT_MAP.md`
- `specs/protocol_v4.yaml`
- `specs/kdate_protocol.yaml`
- `specs/baselines.yaml`
- `specs/experiment_schema.yaml`
- `specs/metrics.yaml`
- `hydroda/data/protocol.py`
- `hydroda/data/leakage_guard.py`
- `hydroda/splits/kdate.py`
- `hydroda/splits/manifest.py`
- `hydroda/data/dataset.py`
- `hydroda/evaluation/harness.py`
- 相关 tests：protocol/leakage/split/evaluation/baseline tests

## 验收标准

1. `ProtocolConfig().role_for_date("2020-06-01") == "source_fit"`。
2. `ProtocolConfig().role_for_date("2021-06-01") == "source_fit"`。
3. `ProtocolConfig().role_for_date("2022-06-01") == "source_val"`。
4. `ProtocolConfig().role_for_date("2023-06-01") == "target_eval"`。
5. split builder 默认生成 `adaptation_setting ∈ {zero_shot_context, few_shot_k4, few_shot_k12}`。
6. target_eval labels 不能用于 training、prompt、normalization、early stopping、model selection 或 hyperparameter selection。
7. metrics_long.csv 包含 `adaptation_setting`、`target_context_dates_hash`、`target_support_dates_hash`、`target_eval_dates_hash`、`split_manifest_sha256`。
