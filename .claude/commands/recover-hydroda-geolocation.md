---
description: Run HydroDA-OOD Phase 0.5 directory-level geolocation recovery
argument-hint: [search_root=/fastersharefiles2/fenglonghan/dataset/SMAP]
---

# Claude Code Command: recover-hydroda-geolocation

请执行 Phase 0.5: Geolocation and Dataset-Contract Resolution。

## 必读上下文

请先阅读：

```text
CLAUDE.md
context/00_EXECUTABLE_CONTEXT_MAP.md
context/11_GEOLOCATION_RECOVERY_PROTOCOL.md
tasks/phase0_5_geolocation_and_contract_resolution.md
specs/geolocation_sources.yaml
specs/hydroda_dataset_contract.yaml
reports/audits/geolocation_recovery_US.md
artifacts/audits/geolocation_recovery_US.json
```

## 任务

不要只检查 `DA.nc`。请递归搜索整个目录：

```text
/fastersharefiles2/fenglonghan/dataset/SMAP
```

寻找 lat/lon、grid mapping、projection、coordinate lookup、domain mask、原始处理脚本或配置文件。

请实现或更新：

```text
scripts/scan_smap_geolocation_sources.py
scripts/geolocation_recovery.py
hydroda/data/geolocation.py
tests/test_geolocation_candidate_scan.py
```

并生成：

```text
artifacts/audits/smap_directory_manifest.json
reports/audits/smap_directory_manifest.md
artifacts/geolocation/geolocation_candidates_US.json
reports/audits/geolocation_lookup_US.md
```

如果成功恢复坐标，还要生成：

```text
artifacts/geolocation/US_latlon.nc
artifacts/geolocation/US_grid_mapping.json  # 如果需要投影转换记录
```

## 强约束

- 不要训练模型。
- 不要构建正式 US-R1..US-R6 region masks，除非 lat/lon 已恢复并验证。
- 不要使用 target query labels、analysis increment statistics 或 model errors。
- input channel 11 继续称为 `base_valid_mask`，不要改成 obs_mask 或 region_mask。

## 完成后

请用中文报告：

```text
完成内容：
- 新增文件：
- 修改文件：
- 运行命令：
- 生成 artifacts：
- 候选 geolocation sources：
- lat/lon 恢复结果：
- Phase 1 / Phase 2 状态：
- 泄漏风险：
- 下一步：
```
