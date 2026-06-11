# 阶段 2：区域 Mask 与 Target-Train Split

## 目标

构建美国区域 mask，并生成 leave-one-region-out zero/few-shot split 文件。

## 必须实现的模块

```text
hydroda/data/region_masks.py
hydroda/data/split_builder.py
hydroda/regions/quality_report.py
```

## 任务

1. 读取 `specs/regions_v1.yaml`。
2. 将 US bbox 映射到 grid indices。
3. 如果 DA.nc 没有 lat/lon coords，停止并报告缺失 grid mapping metadata。
4. 构建 US-R1 到 US-R6 的 masks。
5. 生成 region quality report。
6. 生成 US leave-one-region-out splits。
7. 为 `adaptation_setting ∈ {zero_shot_context, few_shot_k4, few_shot_k12}` 生成 2015-2021 target_context 与 K 个 target_support dates；K=24 只作为 internal ablation。

## 输出

```text
artifacts/regions/masks/US_R*.npy
artifacts/regions/region_metadata_us.json
reports/region_quality_us.md
artifacts/splits/us_loro/*.json
reports/split_leakage_report.md
```

## 验收标准

- 六个 US masks 都非空。
- 每个区域有 2015-2021 target_context / target_support dates 和 2023-2025 target_eval dates。
- split report 证明 target_eval/query dates 没有进入 target_context/support/adaptation。
- 同一 `adaptation_setting, seed, region` 的 target_context/support/eval dates 与 method 无关。
