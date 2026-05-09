# /plan-hydroda-phase

请为 HydroDA-OOD 的指定 phase 制定具体执行计划。

参数：

```text
phase: 0|1|2|3|4|5|6
```

执行要求：

1. 读取 `CLAUDE.md`。
2. 读取 `context/00_EXECUTABLE_CONTEXT_MAP.md`。
3. 读取对应 `tasks/phase*_*.md`。
4. 列出必须实现的文件、CLI、tests、artifacts。
5. 明确 phase gate 和 no-leakage 风险。
6. 不要写代码，先给 plan。

输出格式：

```text
Phase 目标：
需要读取的上下文：
实现文件：
测试文件：
命令：
预期 artifacts：
风险：
验收标准：
```
