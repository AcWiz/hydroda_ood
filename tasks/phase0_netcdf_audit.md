# Phase 0 — NetCDF Audit

## 目标

在写 dataset、split、baseline 之前，审计当前美国 `DA.nc` 的真实结构。

数据路径：

```text
/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc
```

## 禁止

```text
不要训练模型
不要构建 splits
不要假设变量维度
不要静默修复数据问题
```

## 需要实现

```text
hydroda/data/netcdf_audit.py
scripts/audit_netcdf.py
tests/test_netcdf_audit_smoke.py
```

## CLI

```bash
python scripts/audit_netcdf.py \
  --data /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc \
  --country US \
  --out-json artifacts/audits/netcdf_audit_US.json \
  --out-md reports/audits/netcdf_audit_US.md
```

## audit 内容

必须输出：

```text
dims
coords
data_vars
time range and frequency
missing dates
variable shapes and dtypes
input/target channel mapping if available
NaN/Inf ratio
mask unique values
finite forecast/analysis overlap
coordinate availability
memory estimate
```

## 验收标准

```text
1. audit script 可以 metadata-only 打开 DA.nc。
2. 输出 json 和 markdown report。
3. report 中明确是否存在 lat/lon coords。
4. report 中明确是否能识别 input/target channels。
5. 不产生任何 model checkpoint。
```

## 完成汇报

用 CLAUDE.md 中固定格式汇报，并说明：

```text
是否可进入 Phase 1
阻塞问题
需要用户确认的问题
```
