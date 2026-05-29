# CLAUDE.md — HydroDA-OOD / HyperDA V4.2 Executable Research System

你正在参与 `hydroda_ood`。请把本仓库当成**论文级 Scientific ML / Geoscience ML / Data Assimilation 实验系统**，而不是一次性脚本项目。

本项目冻结主线：

```text
HydroDA-OOD:
  leakage-controlled cross-continental neural land DA increment emulation protocol.

HyperDA:
  hydroclimatic spatio-temporal prompt-conditioned, basis-factorized hypernetwork
  for generating target-specific lightweight neural DA increment operators.
```

本项目不是 ordinary soil moisture prediction。正确任务是：

```text
neural land DA analysis-increment emulation
```

模型预测 DA increment，然后加回 forecast 得到 estimated analysis。

---

## 0. 启动规则

写任何代码前必须读取：

```text
context/00_EXECUTABLE_CONTEXT_MAP.md
context/01_RESEARCH_CONTRACT.md
docs/COAUTHOR_CONTEXT.md
完整研究计划方案.md
当前 phase 对应的 tasks/*.md
相关 specs/*.yaml
checklists/no_leakage_checklist.md
```

如果上下文冲突，优先级为：

```text
CLAUDE.md
> 完整研究计划方案.md（HyperDA V4 冻结版）
> tasks/phase*_*.md
> specs/*.yaml
> context/*.md
> checklists/*.md
> prompts/*.md
> notes/*.md
```

不要把临时规则粗暴追加到本文件末尾。新增协议必须进入对应 context/spec/task，并在 `context/00_EXECUTABLE_CONTEXT_MAP.md` 中注册。

项目清理、legacy/active 文件分类和共同一作协作入口见：

```text
docs/COAUTHOR_CONTEXT.md
docs/PROJECT_CLEANUP_AUDIT.md
```

---

## 环境管理

conda 环境名称：
- GPU：`hydroda-ood`（CUDA 12.4，RTX 5090 Blackwell）
- CPU fallback：`hydroda-ood-cpu`

激活方式：
```bash
conda activate hydroda-ood      # GPU
conda activate hydroda-ood-cpu  # CPU fallback
```

不要在 base 环境安装任何东西。GPU 环境使用 `environment-gpu.yml`，CPU 环境使用 `environment.yml`。

---

## 1. 科学任务定义

禁止把本项目写成：

```text
predict true soil moisture
generic soil moisture forecasting
ground-truth soil moisture estimation
```

必须写成：

```text
We emulate the analysis-increment correction operator induced by a physical / operational land DA system.
```

目标变量：

```text
ΔSM_surface  = SM_surface_analysis  - SM_surface_forecast
ΔSM_rootzone = SM_rootzone_analysis - SM_rootzone_forecast
```

模型输出：

```text
pred_increment_surface
pred_increment_rootzone
```

重建 analysis：

```text
pred_analysis_surface  = forecast_surface  + pred_increment_surface
pred_analysis_rootzone = forecast_rootzone + pred_increment_rootzone
```

---

## 2. 最新 V4 论文主线

论文主问题固定为：

```text
Can hydroclimatic spatio-temporal prompts and full target-training-period
adaptation generate target-specific neural data assimilation increment operators
that outperform shared conditional models under cross-continental hydroclimatic shift?
```

必须围绕以下可证伪比较组织实验：

```text
1. Forecast-only
2. Source-only backbone
3. Prompt-conditioned shared backbone
4. Adapter tuning
5. LoRA tuning
6. HyperDA generated operator from full target train
7. HyperDA generated operator + full target train calibration
8. HyperDA-Refine with full target train
```

论文主表**不再**包含以下无明确学术定位或容易分散主线的 heuristic baseline：

```text
source_mean_increment
target_train_mean_increment
monthly_mean_increment
ridge_calibration
nearest-source specialist
prompt-weighted specialist
kNN parameter interpolation
linear prompt-to-parameter
random-prompt HyperDA
```

这些方法最多作为 internal sanity check，不进入论文主表，不作为当前工程优先级。

---

## 3. 数据现实与审计纪律

当前 US 数据路径：

```text
/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc
```

历史上下文提示过：

```text
input[T, 12, H, W]
target[T, 4, H, W]
H = 256, W = 640
```

但工程实现必须以 NetCDF audit 结果为准，不能硬编码未审计的 channel 语义。

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

## 4. 区域与主实验协议

最终主实验使用：

```text
United States
China
Australia
```

每个大陆固定 6 个 hydroclimatic regime slots：

```text
R1 dryland / sparse vegetation
R2 semi-arid grassland / agro-pastoral transition
R3 irrigated / intensively managed agriculture
R4 rainfed agriculture
R5 humid high-vegetation
R6 mountain / snow / cold-terrain stress
```

总计：

```text
3 continents × 6 regions = 18 hydroclimatic regions
```

区域边界必须来自 `specs/regions_v2.yaml` 或其后续冻结版。禁止根据 analysis increments、target query labels、model errors 或实验结果修改区域。

最终主实验为 Leave-One-Continent-Out：

```text
Source: US + CN -> Target: AU
Source: US + AU -> Target: CN
Source: CN + AU -> Target: US
```

当前 US-only 只作为 development subset，用于 pipeline debugging、method validation 和 risk diagnosis。

---

## 5. 时间协议与 target full-training-set adaptation

冻结时间协议：

```text
Source fit/train:             2015-01-01 to 2020-12-31
Source validation:            2021-01-01 to 2021-12-31
Target train/adaptation:      2022-01-01 to 2022-12-31
Target held-out evaluation:   2023-01-01 to 2025-12-31
```

主协议不再是 few-shot / K-shot target adaptation。HyperDA / adapter / LoRA /
prompt 构造在训练完成后，可以使用 held-out target domain 的完整 2022
target training period 来构造 target-specific operator / prompt / adapter /
generated parameters，然后只在严格 held-out 的 2023-2025 target evaluation
period 上评估。

主实验适配设置：

```text
adaptation_setting = target_full_train
```

旧 K-cycle 设定仅保留为 secondary legacy ablation：

```text
legacy_few_shot_K ∈ {0, 4, 12}
```

旧 K 定义仍是：

```text
K = number of labeled target DA analysis cycles from target train year 2022
```

K 不是 patches、pixels 或 mini-batches 数。同一天切出多少 spatial patches，都只算一个 DA calibration cycle。

主协议中不再单独保留 2022 support set；`target_support` 只作为旧代码 /
legacy few-shot ablation 的别名。论文主表必须按 `adaptation_setting` 区分，
不能把 `K` 作为主实验列。

Source 覆盖范围主协议保持 `source_fit=2015-2020`、`source_val=2021`。
不要把 2021 混入 source fit，因为它承担 checkpoint / early stopping /
hyperparameter selection。source 2022 或 2015-2021 refit 只能作为明确标注的
expanded-source secondary ablation，不能混入主表。

---

## 6. Leakage Guard

严格禁止：

```text
target evaluation labels 用于 prompt construction
target evaluation labels 用于 normalization
target evaluation labels 用于 target-train/adaptation sample selection
target evaluation labels 用于 early stopping
target evaluation labels 用于 model selection
target evaluation labels 用于 threshold calibration
target evaluation labels 用于 prompt feature tuning
analysis increments 用于定义 region
model errors 用于重新定义 region 或 target-train/adaptation dates
```

以下模块必须走同一套协议对象或 guard：

```text
PromptBuilder
Normalizer
TargetAdaptationSampler
Dataset
Evaluator
ExperimentRunner
```

不能让每个文件各自写时间切分逻辑。

---

## 7. V4 阶段顺序

当前执行顺序固定为：

```text
Phase 0: DA.nc audit
Phase 1: Region cropping and region artifact contract
Phase 2: HydroDADataset + ProtocolConfig + LeakageGuard
Phase 3: Forecast-only + metrics/evaluation sanity
Phase 4: Source-only backbone + prompt-conditioned shared backbone
Phase 5: Source operator episode bank
Phase 6: HyperDA target-full-train generation / calibration / refinement
Phase 7: adapter and LoRA full-target-train comparison; K=0/4/12 only as legacy ablation
Phase 8: US-only development report
Phase 9: CN/AU expansion and leave-one-continent-out
```

在 Phase 3 未完全通过之前，不要实现 HyperDA、LoRA、复杂 Transformer 或 Sformer wrapper。

在 Phase 4 未通过之前，不要训练 source operator episode bank。

---

## 8. HyperDA 实现约束

HyperDA 包含：

```text
1. Shared DA increment backbone
2. Source operator episode bank
3. Hydroclimatic spatio-temporal prompt encoder
4. Basis-factorized hypernetwork parameter generator
```

总体形式：

```text
ζ_R = H_ψ(P_R)
pred_increment_R = f_{θ0, ζ_R}(x_R)
```

第一版不生成 full backbone weights，只生成 lightweight target-specific parameters：

```text
generated adapter parameters
generated output-head residual parameters
optional generated FiLM parameters
```

第一版 backbone 使用 Small ResUNet 或 ConvNeXt-UNet。不要把贡献混到大型 backbone capacity 上。

---

## 9. 文件与报告输出纪律

所有实验必须保存：

```text
config snapshot
split manifest path
protocol freeze id
method name
adaptation_setting
legacy K only for few-shot ablations
target_train_dates_hash
target_eval_dates_hash
split_manifest_sha256
trainable parameter count
adaptation steps
metrics_long.csv
summary.json
```

每个任务完成后用中文输出：

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
