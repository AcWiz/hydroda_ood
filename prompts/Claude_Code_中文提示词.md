# Claude Code 中文提示词集合

## 1. 初始读取上下文

```text
请先阅读 CLAUDE.md 以及它引用的所有文件。暂时不要写代码。

请总结：
1. 本项目的科学任务；
2. 数据契约；
3. 信息泄漏禁止规则；
4. 当前可用数据；
5. 第一个实现里程碑；
6. 你需要先检查哪些仓库文件。

总结后等待我的下一步指令。
```

## 2. 阶段 0：NetCDF 审计

```text
请实现 tasks/阶段0_仓库与NetCDF审计.md。

约束：
- 不要实现模型。
- 不要假设 DA.nc 一定有 lat/lon；必须审计。
- 如果 expected variables 或 channels 缺失，必须 fail loudly。
- 需要同时输出 JSON 和 Markdown audit reports。
- 在报告中给出一个最小 smoke test 命令。
```

## 3. 阶段 1：数据契约

```text
请实现 tasks/阶段1_数据契约.md。

重点：
- 本任务是 DA analysis-increment emulation。
- Dataset 必须分别返回 forecast、analysis、increment、obs_mask、region_mask、loss_mask。
- 不要混淆 obs_mask 和 loss_mask。
- 不要使用 target query years 做 normalization。
- 添加测试，验证 increment = analysis - forecast。
```

## 4. 阶段 2：区域与 K-date split

```text
请实现 tasks/阶段2_区域与KDate划分.md。

重点：
- 使用 specs/regions_v1.yaml。
- 如果 DA.nc 没有 lat/lon coordinates，请停止并报告需要 grid mapping metadata。
- 当前只生成 US-R1 到 US-R6。
- 生成 K=0,4,12,24 和 seeds 0..4 的 support splits。
- 生成 split leakage report。
```

## 5. 阶段 3：简单 baseline

```text
请实现 tasks/阶段3_简单基线.md。

先从 Forecast-only 和 mean increment baselines 开始。
Forecast-only 的 Skill 必须按定义精确等于 0。
不要实现 neural models。
输出 CSV metrics 和简短 Markdown result report。
```

## 6. 代码审查

```text
请审查当前实现，重点检查：
1. temporal leakage；
2. target query label leakage；
3. hard-coded US assumptions；
4. mask misuse；
5. normalized/raw unit confusion；
6. missing tests；
7. reproducibility gaps。

请先返回优先级 patch plan，然后只实现最高优先级修复。
```

## 7. 实验失败诊断

```text
某个 run 失败了。请只根据 logs 和 configs 诊断。

不要为了让 run 通过而改变科学协议。

请判断失败原因属于：
- data shape mismatch；
- missing variables；
- invalid region mask；
- insufficient support dates；
- normalization issue；
- model shape requirement；
- metric bug。

然后提出并实现最小修复。
```

## 8. 结果解读

```text
给定 metrics CSV，请写一段 research-style analysis：

1. 哪些区域最难；
2. K-date calibration 是否有效；
3. monthly increment 是否是强危险 baseline；
4. neural methods 是否超过 simple baselines；
5. failure modes 暗示了哪些 DA update behavior。

不要过度声称真实 soil moisture accuracy。
```

## 9. 让 Claude Code 执行下一阶段

```text
请阅读当前阶段任务卡和之前阶段产物，先总结：
1. 已完成内容；
2. 当前阶段目标；
3. 需要修改或新增的文件；
4. 可能的信息泄漏风险；
5. 验收标准。

然后给出实现计划。等我确认后再写代码。
```

## 10. 让 Claude Code 生成状态报告

```text
请生成当前项目状态报告，包含：
1. 已完成阶段；
2. 已生成 artifacts；
3. 已通过 tests；
4. 当前失败项；
5. 下一步最重要任务；
6. 哪些内容可能影响论文可信度。

报告请用中文。
```
