# 00 Executable Context Map

本文件定义 Claude Code 在不同任务中应该加载哪些上下文，避免一次性读入所有文档导致注意力发散。

---

## A. 总入口

所有任务必读：

```text
CLAUDE.md
context/00_EXECUTABLE_CONTEXT_MAP.md
context/01_RESEARCH_CONTRACT.md
```

如果任务涉及数据、split、metric、baseline、region 或 geolocation，还必须读取：

```text
checklists/no_leakage_checklist.md
```

---

## B. Phase-specific context loading

### Phase 0: NetCDF audit

读取：

```text
tasks/phase0_netcdf_audit.md
context/02_DATA_AND_LEAKAGE_CONTRACT.md
specs/hydroda_dataset_contract.yaml
checklists/phase_gate_checklist.md
```

目标：只审计 `DA.nc`，不训练模型。

---

### Phase 0.5: Geolocation and dataset-contract resolution

读取：

```text
tasks/phase0_5_geolocation_and_contract_resolution.md
context/02_DATA_AND_LEAKAGE_CONTRACT.md
context/11_GEOLOCATION_RECOVERY_PROTOCOL.md
specs/hydroda_dataset_contract.yaml
specs/geolocation_sources.yaml
checklists/no_leakage_checklist.md
checklists/phase_gate_checklist.md
```

目标：

```text
1. 固化 multi-channel dataset contract；
2. 将 input channel 11 暂定为 base_valid_mask；
3. 不只检查 DA.nc，而要递归搜索 /fastersharefiles2/fenglonghan/dataset/SMAP；
4. 尝试恢复 DA.nc grid -> lat/lon 或 projection mapping；
5. 决定 Phase 2 scientific region masks 是否解锁。
```

---

### Phase 1: HydroDADataset and sample contract

读取：

```text
tasks/phase1_dataset_contract.md
context/02_DATA_AND_LEAKAGE_CONTRACT.md
context/07_CODE_ARCHITECTURE.md
specs/hydroda_dataset_contract.yaml
```

目标：实现 `HydroDADataset`，返回 contract-defined sample。

说明：Phase 1 可以在 grid-only 下进行，但 `region_mask` 必须作为外部 artifact 接口保留，不得由 `base_valid_mask` 冒充。

---

### Phase 2: Region masks + K-date splits

读取：

```text
tasks/phase2_regions_and_kdate_splits.md
context/03_REGION_PROTOCOL.md
context/04_KDATE_SPLIT_PROTOCOL.md
context/11_GEOLOCATION_RECOVERY_PROTOCOL.md
specs/regions_v2.yaml
specs/kdate_protocol.yaml
specs/geolocation_sources.yaml
artifacts/geolocation/US_latlon.nc
checklists/no_leakage_checklist.md
```

目标：生成 masks、quality report、support/query split manifest。

**Artifacts**（Phase 2A 完成后冻结）：

```text
artifacts/regions/US_region_masks.nc              # canonical NC（不可修改）
artifacts/regions/US_region_mask_tensor.pt         # fast training mirror
artifacts/regions/US_region_masks_manifest.json    # artifact contract & SHA256
artifacts/protocol/US_region_split_freeze_manifest.json  # 已更新引用新 artifact
hydroda/data/region_artifacts.py                   # unified loader utility
reports/regions/US_region_artifact_contract.md     # contract 文档
```

Gate：

```text
Temporal K-date splits: 可在 Phase 1 dataset 完成后先实现。
Scientific US-R1..US-R6 masks: UNLOCKED (US_latlon.nc recovered 2026-05-06)
Grid-only masks: ALLOWED for development only
```

---

### Phase 3: Simple baselines

读取：

```text
tasks/phase3_simple_baselines.md
context/05_METRICS_AND_REPORTING.md
context/06_BASELINES_AND_MODEL_ROADMAP.md
specs/baselines.yaml
specs/metrics.yaml
checklists/no_leakage_checklist.md
```

目标：Forecast-only、source mean、target support mean、monthly support mean、ridge。

---

### Phase 4: Neural baselines

读取：

```text
tasks/phase4_neural_baselines.md
context/06_BASELINES_AND_MODEL_ROADMAP.md
context/07_CODE_ARCHITECTURE.md
specs/experiment_schema.yaml
checklists/no_leakage_checklist.md
```

目标：small Conv/UNet baseline；Sformer 只作为 wrapper，不能绕过 split/metric pipeline。

---

### Phase 5: HyRAO and sparse adaptation

读取：

```text
tasks/phase5_hyrao_sparse_adaptation.md
context/01_RESEARCH_CONTRACT.md
context/06_BASELINES_AND_MODEL_ROADMAP.md
context/10_REVIEWER_RISK_CONTROL.md
specs/experiment_schema.yaml
checklists/no_leakage_checklist.md
```

目标：region descriptor、region latent、adapter、sparse block adaptation。

---

### Phase 6: Reporting

读取：

```text
tasks/phase6_reporting_and_paper_artifacts.md
context/05_METRICS_AND_REPORTING.md
context/08_EXPERIMENT_REGISTRY.md
templates/result_report_template.md
templates/experiment_card_template.md
checklists/reproducibility_checklist.md
```

目标：paper-ready tables、K-curves、event maps、risk notes。

---

## C. 不确定时的行为

如果 metadata 不清楚，不要猜测变量维度、时间范围、坐标系统或 mask 语义。先写 audit 代码并报告。

如果研究协议和实际数据冲突，不要静默修改协议。输出：

```text
发现的冲突
影响范围
建议修订
需要用户确认的问题
```

如果结果异常，不要先调模型。先检查：

```text
normalization
mask
increment sign
forecast + increment reconstruction
split leakage
region crop / padding
metric aggregation
geolocation / region mapping
```

---

## D. Conda Environment

本项目使用 `environment.yml` 作为 canonical 依赖规范。

```text
conda env create -f environment.yml
conda activate hydroda-ood
python scripts/check_environment.py   # 验证
```

文档：`docs/conda_environment.md`

**注意**：默认安装 CPU-only PyTorch。GPU 用户需额外安装：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**不要**将 data/artifact 文件放入 conda environment。
