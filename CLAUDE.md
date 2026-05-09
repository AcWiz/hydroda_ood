# CLAUDE.md — HydroDA-OOD Executable Context System v2.2

你正在参与 `hydroda_ood`。请把本仓库当成**论文级 Scientific ML 实验系统**，而不是一次性脚本项目。

本项目目标：构建并验证 `HydroDA-OOD`，用于研究 **cross-region / cross-continent neural land data assimilation analysis-increment emulation under few-cycle target calibration**。

---

## 0. 启动规则

写任何代码前必须读取：

```text
context/00_EXECUTABLE_CONTEXT_MAP.md
context/01_RESEARCH_CONTRACT.md
当前 phase 对应的 tasks/*.md
相关 specs/*.yaml
如果涉及数据、region、split、metric，读取 checklists/no_leakage_checklist.md
```

如果上下文冲突，优先级为：

```text
CLAUDE.md
> tasks/phase*_*.md
> specs/*.yaml
> context/*.md
> checklists/*.md
> prompts/*.md
> notes/*.md
```

不要把临时规则粗暴追加到本文件末尾。新增协议必须进入对应 context/spec/task，并在 `context/00_EXECUTABLE_CONTEXT_MAP.md` 中注册。

---

## 1. 科学任务定义

本项目不是 ordinary soil moisture prediction。

正确任务：

```text
reference DA analysis-increment emulation
```

目标变量：

```text
ΔSM_surface  = sm_surface_analysis  - sm_surface_forecast
ΔSM_rootzone = sm_rootzone_analysis - sm_rootzone_forecast
```

模型输出 increment，再加回 forecast 得到 estimated analysis。

禁止写成：

```text
predict true soil moisture
ground-truth soil moisture estimation
```

---

## 2. 当前数据现实

当前 US 数据：

```text
/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc
```

已知结构：

```text
input[T, 12, H, W]
target[T, 4, H, W]
H=256, W=640
```

US geolocation 已恢复：

```text
artifacts/geolocation/US_latlon.nc
recovery_method = directory_latlon_vector_lookup
```

`input` channel 11 暂定为：

```text
base_valid_mask
```

不得把它冒充：

```text
obs_mask
region_mask
static_land_mask
```

---

## 3. 固定三国区域协议

三国 18 个区域已预注册在：

```text
specs/regions_v2.yaml
region_protocol_version = fixed_bbox_v1
```

每个国家固定 6 个 regime slots：

```text
R1 dryland_sparse_vegetation
R2 semi_arid_transition
R3 irrigated_managed_agriculture
R4 rainfed_agriculture
R5 humid_high_vegetation
R6 mountain_cold_terrain_stress
```

区域边界必须来自 `specs/regions_v2.yaml`。禁止根据 analysis increments、target query labels、model errors 或实验结果修改区域。

US 已可生成正式 scientific masks。CN/AU 区域范围已固定，但必须等待各自 geolocation artifact 后才能生成正式 masks。

---

## 4. 阶段顺序

```text
Phase 0    NetCDF audit
Phase 0.5  Geolocation and dataset-contract resolution
Phase 1    HydroDADataset and sample contract
Phase 2A   Fixed region masks
Phase 2B   K-date support/query splits
Phase 3    Simple baselines
Phase 4    Neural baselines
Phase 5    HyRAO + sparse adaptation
Phase 6    Tables, figures, paper artifacts
```

在 Phase 0–3 完成前，禁止投入复杂 Transformer、Sformer wrapper、LoRA、Hessian/Fisher 稀疏适配。

---

## 5. 零泄漏规则

禁止：

```text
target query labels 用于 support selection
target query labels 用于 normalization
target query labels 用于 early stopping
target query labels 用于 model selection
analysis increments 用于定义 region
model errors 用于重新定义 region
high-update masks 用于训练或 support selection
```

K 表示 target DA dates/cycles 数，不是 patches、pixels 或 mini-batches 数。

---

## 6. 完成汇报格式

每次任务完成后用中文输出：

```text
完成内容：
- 新增文件：
- 修改文件：
- 运行命令：
- 生成 artifacts：
- 测试 / smoke check：
- 关键发现：
- 泄漏风险：
- 下一步：
```
