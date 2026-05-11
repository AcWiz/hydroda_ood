# HyperDA V4 update bundle

这个包用于把当前 `hydroda_ood` 从旧版 HyRAO / simple-baseline 主线升级为 HyperDA V4 主线。

## 推荐使用方式

在仓库根目录执行：

```bash
# 1. 备份旧文件
mkdir -p _backup_before_hyperda_v4
cp CLAUDE.md 完整研究计划方案.md _backup_before_hyperda_v4/ 2>/dev/null || true
cp -r context tasks specs .claude hydroda tests _backup_before_hyperda_v4/ 2>/dev/null || true

# 2. 解压本包后，从 bundle 根目录复制文件
rsync -av hyperda_v4_update_bundle/ ./

# 3. 运行最小 smoke tests
python -m pytest tests/test_evaluation_harness_metric_routing.py tests/test_protocol_leakage_guard.py tests/test_neural_forward.py -q
```

如果当前仓库还没有 `hydroda/models` 或 `hydroda/operator_bank`，`rsync` 会自动创建。

## 这次更新做了什么

1. 更新 `CLAUDE.md`，把项目主线冻结为 HydroDA-OOD / HyperDA V4。
2. 更新 `完整研究计划方案.md`，删除旧 HyRAO 主线，明确 HyperDA parameter-space transfer。
3. 修改 tasks：Phase 3 不再主推 source_mean / ridge；Phase 4 强化 source-only 和 prompt-conditioned shared；Phase 5 改为 source operator episode bank；Phase 6/7/8 改为 HyperDA 与 K-cycle calibration。
4. 新增 `specs/protocol_v4.yaml`、`specs/hyperda_v4.yaml`，更新 `specs/baselines.yaml`。
5. 修复 evaluation harness 的核心 metric routing 风险：increment metrics 必须使用 pred_increment / true_increment。
6. 修复旧 harness 中 `SPLITS_JSON` 未定义与 method 硬编码问题。
7. Dataset 增加 `month`、`season` metadata。
8. 新增 `ProtocolConfig` 与 `LeakageGuard`，为后续 prompt、normalization、support sampler、evaluator 统一防泄漏。
9. 新增最小 `SmallResUNet`、`HyperDA`、`ParameterBasis`、`ZetaPacker` 骨架，方便 Claude Code 后续继续扩展。

## 交给 Claude Code 的建议指令

```text
请先阅读 CLAUDE.md、完整研究计划方案.md、context/01_RESEARCH_CONTRACT.md、specs/protocol_v4.yaml、specs/hyperda_v4.yaml。
然后运行 tests/test_evaluation_harness_metric_routing.py、tests/test_protocol_leakage_guard.py、tests/test_neural_forward.py。
如果测试失败，只修复本次 V4 更新涉及的最小问题，不要扩大范围。
完成后汇报：修改文件、运行命令、生成 artifacts、测试结果、泄漏风险、下一步。
```

## 注意

这个包不假装已经运行真实数据实验。真实实验仍需在能访问 `/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc` 的环境中执行。
