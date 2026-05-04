# 阶段 0：仓库与 NetCDF 审计

## 目标

在实现任何模型之前，先审计当前仓库和美国 `DA.nc`。

## 输入

```text
/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc

合作方代码：
  Dataset_api_netcdf_lazyload_time.py
  Config_Vars_DA.json
  Config_Train_DA.json
  arch_DA_time.py
```

## 任务

1. 检查当前仓库结构。
2. 创建 `scripts/audit_netcdf.py`。
3. 创建 `hydroda/data/netcdf_audit.py`。
4. 审计：
   - dims
   - coords
   - data_vars
   - attrs
   - time range
   - time frequency
   - `input` 和 `target` shapes
   - variable / channel availability
   - mask value distribution
   - NaN / Inf counts
   - coordinate / projection availability
5. 保存：
   - `artifacts/audits/netcdf_audit_us.json`
   - `reports/netcdf_audit_us.md`

## 验收标准

- 能运行：

```bash
python scripts/audit_netcdf.py \
  --path /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc \
  --out artifacts/audits/netcdf_audit_us.json
```

- 报告必须明确说明是否存在 lat/lon coordinates。
- 报告必须明确说明 time coverage 是否包含 2015–2025。
- 如果 expected variables 或 channels 缺失，脚本必须给出可操作的错误信息。
