# 09 Iteration Workflow

## 1. 每次 Claude Code 任务的固定流程

```text
1. 读 CLAUDE.md 和 phase task。
2. 复述当前任务目标，不要泛化。
3. 检查需要哪些 specs/checklists。
4. 先实现最小可运行路径。
5. 运行 smoke test。
6. 生成 artifact/report。
7. 更新 decision log 或风险登记。
8. 用固定格式汇报。
```

---

## 2. 不允许的行为

```text
先写复杂模型再补数据契约
看到报错就绕过 mask
把 query years 加入 normalization
用所有 target dates 做 early stopping
直接修改 region 定义来改善结果
在没有 audit 的情况下假设维度
```

---

## 3. Bug 修复顺序

当结果异常，按顺序检查：

```text
1. DA.nc variable/channel mapping
2. forecast-analysis-increment reconstruction
3. loss_mask / region_mask / obs_mask
4. date split manifest
5. normalization stats source
6. metric denominator and aggregation
7. patch crop/padding/reconstruction
8. model implementation
```

---

## 4. 决策日志

任何协议级变化必须写入：

```text
notes/decision_log.md
```

格式见：

```text
templates/decision_log_entry_template.md
```
