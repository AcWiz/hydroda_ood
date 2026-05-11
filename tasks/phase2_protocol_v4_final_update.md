# phase2_protocol_v4_final_update.md — 更新 Protocol V4-final ✅ 已完成

## 目标

将项目从旧时间协议更新为新冻结协议（迁移已于 2026-05-11 完成）。

## 新冻结协议

```text
source_fit:     2015-2020
source_val:     2021
target_context: 2022
target_query:   2023-2025
```

## 必改文件（全部已完成）

- `CLAUDE.md`
- `完整研究计划方案.md`
- `context/01_RESEARCH_CONTRACT.md`
- `context/04_KDATE_SPLIT_PROTOCOL.md`
- `context/00_EXECUTABLE_CONTEXT_MAP.md`
- `specs/protocol_v4.yaml`
- `specs/kdate_protocol.yaml`
- `hydroda/data/protocol.py`
- `scripts/build_kdate_splits.py`
- `hydroda/splits/kdate.py`
- 相关 tests：`test_protocol_leakage_guard.py`、`test_kdate_splits_no_leakage.py`、`test_target_query_evaluation_only.py`

## 验收标准

1. `ProtocolConfig().role_for_date("2020-06-01") == "source_fit"`。
2. `ProtocolConfig().role_for_date("2021-06-01") == "source_val"`。
3. `ProtocolConfig().role_for_date("2022-06-01") == "target_context"`。
4. `ProtocolConfig().role_for_date("2023-06-01") == "target_query"`。
5. split builder 使用 2022 年作为 support/context year，2023-2025 作为 query years。
6. target context labels 不能用于 checkpoint、early stopping、hyperparameter 或 model selection。
7. target query labels 仍然只能用于最终 offline evaluation。

## 建议测试

```bash
pytest tests/test_protocol_leakage_guard.py        tests/test_kdate_splits_no_leakage.py        tests/test_target_query_evaluation_only.py -q
```

如果没有真实数据或 split artifacts，数据依赖型测试可以 skip；`test_protocol_leakage_guard.py` 必须通过。
