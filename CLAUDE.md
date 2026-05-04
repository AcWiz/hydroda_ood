# CLAUDE.md — HydroDA-OOD 中文可执行上下文

你正在参与 HydroDA-OOD 项目。这是一个面向 ICLR / NeurIPS / ICML / AISTATS / JMLR / Scientific ML 顶级会议的研究代码库。请把这个仓库当作论文级实验系统，而不是普通脚本项目。

## 一、项目使命

构建一个可复现的 benchmark 和实验管线，用于研究：

> 跨大陆 neural land data assimilation analysis-increment emulation under few-cycle target calibration。

当前只有美国 `DA.nc` 可用。中国和澳大利亚的 `DA.nc` 之后会加入。因此所有代码必须做到：

```text
当前可运行：US-only within-country hydroclimatic OOD
未来可扩展：US / China / Australia cross-continental OOD
```

添加中国和澳大利亚时，应该主要是新增配置、区域 mask 和数据路径，而不是重写代码。

## 二、不可改变的科学表述

不要把任务说成“预测真实土壤湿度”。

正确任务是：

```text
输入：
  forecast soil moisture
  microwave brightness temperature observations
  vegetation opacity
  observation / validity masks
  time encoding
  optional input-only region descriptors

预测：
  DA analysis increment

其中：
  ΔSM_surface  = SM_surface_analysis  - SM_surface_forecast
  ΔSM_rootzone = SM_rootzone_analysis - SM_rootzone_forecast

输出：
  SM_hat = SM_forecast + ΔSM_hat
```

请使用：

```text
DA analysis-increment emulation
reference DA analysis
neural DA operator
few-cycle calibration
```

避免使用：

```text
true soil moisture prediction
ground-truth soil moisture estimation
```

## 三、已知合作方代码事实

合作方代码显示，DA 任务的 active input variables 是：

```text
sm_surface_forecast
sm_rootzone_forecast
mwrtm_vegopacity
tb_h_obs
tb_v_obs
mask
```

DA target variables 是：

```text
sm_surface_analysis
sm_rootzone_analysis
```

合作方 dataloader 使用 `xarray.open_dataset()` lazy-load NetCDF，根据 JSON 中的 `include=true` 选择 input/target 通道，做 mean/std 归一化，并返回 input、target、time_code、year、month、day、time。

合作方 DA 模型是 Sformer / Swin-like backbone。当前 DA config 关键参数包括：

```text
img_size = [256, 640]
patch_size = 4
window_size = 8
num_vars = 6
embed_dim = 128
depths = [2, 2]
```

合作方模型风格本质上是学习 masked update，然后加回 forecast 得到 analysis。这与我们自己的 increment-emulation 任务完全兼容。

合作方 Sformer 可以作为强 neural baseline，但不要让它决定我们的 benchmark、split、metric 或区域协议。

### 合作方代码使用规则

合作方代码放在 external/collaborator_code/ 下，只读参考。

允许 Claude Code：
1. 阅读合作方 dataloader，理解 DA.nc 的 input/target 结构；
2. 阅读 Config_Vars_DA.json，确认变量名、mean/std、include 字段；
3. 阅读 Config_Train_DA.json，确认 Sformer baseline 的模型参数；
4. 在后续阶段通过 wrapper 接入合作方 Sformer 作为 pooled neural baseline。

禁止 Claude Code：
1. 直接把 external/collaborator_code/ 复制进 hydroda/ 主代码；
2. 直接修改合作方原始文件；
3. 用合作方 dataloader 替代 HydroDADataset；
4. 用合作方训练逻辑绕过我们的 region split、K-date split 和 no-leakage protocol；
5. 在 Phase 0–3 阶段实现或调试 Sformer，必须先完成 NetCDF audit、dataset contract、region mask、K-date split 和 simple baselines。


## 四、每次写代码前必须阅读的文件

请优先读取：

```text
context/00_项目总览.md
context/01_研究协议.md
context/02_数据契约.md
context/03_区域划分_v1.md
context/04_实验矩阵.md
context/05_模型与基线.md
context/06_指标与验证.md
context/07_工程架构.md
context/08_代码规范.md
context/09_迭代工作流.md
context/10_风险登记.md
specs/hydroda_schema.yaml
specs/regions_v1.yaml
specs/kdate_protocol.yaml
specs/baselines.yaml
specs/metrics.yaml
checklists/可复现性检查清单.md
```

## 五、推荐代码结构

请把项目实现为配置驱动的模块化系统：

```text
hydroda/
  data/
    netcdf_audit.py
    dataset.py
    normalization.py
    patch_sampler.py
    region_masks.py
    split_builder.py
  regions/
    descriptors.py
    predefined_regions.py
    quality_report.py
  metrics/
    increment_metrics.py
    skill.py
    aggregation.py
  models/
    baseline_linear.py
    resunet.py
    sformer_wrapper.py
    adapters.py
    hyrao_meta.py
    sparse_tuning.py
  experiments/
    run_baseline.py
    run_adaptation.py
    evaluate.py
  utils/
    config.py
    io.py
    seed.py
    logging.py
configs/
  data/us_da.yaml
  regions/regions_v1.yaml
  experiments/*.yaml
scripts/
  audit_netcdf.py
  build_region_masks.py
  build_splits.py
  train_baseline.py
  evaluate_checkpoint.py
tests/
  test_dataset_contract.py
  test_split_no_leakage.py
  test_metrics.py
```

## 六、实现优先级

严格按下面顺序推进：

```text
1. 审计 DA.nc。
2. 建立数据契约和 HydroDADataset。
3. 构建美国 6 个科学区域 mask 和质量报告。
4. 构建 K-date support/query split。
5. 实现 Forecast、mean increment、monthly increment、ridge 等简单 baseline。
6. 实现 patch-based neural baseline。
7. 在数据、split、metric 全部稳定后，再包装合作方 Sformer。
8. 最后实现 HyRAO-Meta / adapter / sparse tuning。
```

不要在 NetCDF audit、dataset contract、region split、metric sanity check 完成前实现复杂模型。

## 七、设计规则

- 不要把 US-only 逻辑硬编码到 dataset 或 metric 中。
- country、region、regime、split、K-date 必须配置驱动。
- source train 固定为 2015–2020。
- target support 固定为 2021。
- target query 固定为 2022–2025。
- K 固定为 `0, 4, 12, 24`。
- K 表示 historical DA analysis dates/cycles，不是 patch 或 pixel 数。
- 必须记录 effective support budget：`K dates × patches/date × valid pixels`。
- 主表默认 region-balanced aggregation。
- surface 和 rootzone 必须分开报告。

## 八、信息泄漏禁止规则

禁止：

```text
使用 target query labels 选择 support dates
使用 target query labels 做 early stopping
使用 target query distribution 做 normalization
K=0 使用任何 target analysis statistics
根据模型性能重新定义 region
使用 analysis increments 选择区域
```

K=0 允许使用：

```text
target 2021 input-only stream
forecast statistics
TB statistics
vegopacity statistics
mask / missingness statistics
static descriptors
```

## 九、新功能完成标准

任何新功能必须满足：

```text
1. 有 config entry 或明确默认值；
2. 有 unit test 或 smoke test；
3. 有最小 CLI / script 路径；
4. 记录关键假设；
5. 没有 temporal leakage；
6. 没有 target query leakage；
7. 没有不必要的 hard-coded region；
8. 如果改变协议语义，必须更新文档。
```


## 十、每次任务完成后的汇报格式

每次完成任务后，必须用中文汇报：

```text
1. 新增文件；
2. 修改文件；
3. 运行命令；
4. 生成文件；
5. 测试或 smoke check 结果；
6. 主要发现；
7. 数据风险；
8. 是否存在信息泄漏风险；
9. 下一步建议。
```

不要只输出代码 diff。不要只说“已完成”。