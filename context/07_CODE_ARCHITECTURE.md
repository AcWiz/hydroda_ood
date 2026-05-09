# 07 Code Architecture

## 1. 推荐目录

```text
hydroda/
  data/
    netcdf_audit.py
    dataset.py
    normalization.py
    patch_sampler.py
  regions/
    masks.py
    descriptors.py
    quality_report.py
  splits/
    kdate.py
    manifest.py
    leakage_tests.py
  metrics/
    skill.py
    increment.py
    aggregation.py
  baselines/
    forecast.py
    mean_increment.py
    monthly_mean.py
    ridge.py
  models/
    small_unet.py
    sformer_wrapper.py
    adapters.py
    hyrao.py
  adaptation/
    finetune.py
    sparse_blocks.py
    fisa.py
  experiments/
    runner.py
    registry.py
    evaluate.py
  reports/
    tables.py
    figures.py
  utils/
    config.py
    seed.py
    io.py
    logging.py
scripts/
  audit_netcdf.py
  build_region_masks.py
  build_splits.py
  run_baseline.py
  train_neural.py
  evaluate.py
configs/
tests/
artifacts/
reports/
```

---

## 2. Configuration-driven rule

任何 country、region、K、seed、time range、variable mapping、metric 都必须配置驱动。

禁止：

```python
if region == "US-R1": ...
if year == 2021: ...  # 除非来自 protocol config
```

允许：

```python
cfg.protocol.target_support.start
cfg.regions[region_id].bbox
```

---

## 3. Artifact naming

```text
artifacts/audits/netcdf_audit_{country}.json
reports/audits/netcdf_audit_{country}.md
artifacts/regions/{country}/region_quality.csv
artifacts/splits/{benchmark_id}/{country}/{region}/K{K}_seed{seed}.json
artifacts/predictions/{experiment_id}/{country}/{region}/K{K}_seed{seed}.zarr
artifacts/metrics/{experiment_id}/metrics_long.csv
reports/experiments/{experiment_id}.md
```

---

## 4. Test policy

每个 phase 最少测试：

```text
Phase 0: audit script smoke test with metadata-only open
Phase 1: dataset sample contract test
Phase 2: no-leakage split tests
Phase 3: metric sanity + forecast baseline test
Phase 4: neural forward pass + overfit tiny batch
Phase 5: adaptation changes only allowed params
Phase 6: tables generated from metrics_long.csv only
```

---

## 5. External collaborator code

`external/collaborator_code/` 只读参考。

禁止复制进 `hydroda/` 主代码。允许写 wrapper，但 wrapper 必须通过我们的 dataset、split、metric pipeline。
