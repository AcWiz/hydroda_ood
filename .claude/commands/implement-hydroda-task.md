# /implement-hydroda-task

实现一个明确的 HydroDA-OOD 工程任务。

输入参数：

```text
phase:
task:
```

执行规则：

1. 读取对应 phase task。
2. 读取相关 specs。
3. 先写 tests 或 smoke test。
4. 实现最小可运行路径。
5. 运行测试。
6. 更新报告或 artifact。
7. 用 CLAUDE.md 的固定格式汇报。

禁止扩大任务范围；如果发现需要协议决策，写入 notes/decision_log.md，而不是自行改变协议。
