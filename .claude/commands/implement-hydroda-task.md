# /implement-hydroda-task

实现一个明确的 HydroDA-OOD / HyperDA V4 工程任务。

输入参数：

```text
phase:
task:
```

执行规则：

1. 读取 `CLAUDE.md`、`完整研究计划方案.md`、`context/01_RESEARCH_CONTRACT.md`。
2. 读取对应 phase task 与相关 `specs/*.yaml`。
3. 明确该任务是否可能触碰 data / region / split / metric / normalization / prompt / target labels。
4. 如果可能触碰，先读取 `checklists/no_leakage_checklist.md`，并在实现中加入 guard 或 smoke test。
5. 先写最小 smoke test 或单元测试，再实现最小可运行路径。
6. 运行测试；如无法运行真实数据，至少运行 import / shape / synthetic-data smoke test。
7. 更新报告或 artifact manifest。
8. 用 `CLAUDE.md` 的固定中文格式汇报。

禁止：

```text
扩大任务范围
把旧 HyRAO / simple baseline 主表逻辑重新作为论文主线
绕过 ProtocolConfig / LeakageGuard
使用 target query labels 做任何训练、选择、归一化或调参
```

如果发现需要协议决策，写入 `notes/decision_log.md`，不要自行改变研究协议。
