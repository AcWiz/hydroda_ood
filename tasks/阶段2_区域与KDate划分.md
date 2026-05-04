# 阶段 2：区域 Mask 与 K-Date Split

## 目标

构建美国区域 mask，并生成 leave-one-region-out split 文件。

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
7. 为 K=0/4/12/24 和 seeds 0..4 生成 support dates。

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
- 每个区域有足够 valid dates 支持 K=24。
- split report 证明 target query dates 没有进入 support。
- 同一 `K, seed, region` 的 support dates 与 method 无关。
