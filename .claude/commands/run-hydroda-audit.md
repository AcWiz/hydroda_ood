# /run-hydroda-audit

执行 HydroDA-OOD Phase 0 NetCDF audit。

要求：

1. 读取：
   - `CLAUDE.md`
   - `tasks/phase0_netcdf_audit.md`
   - `context/02_DATA_AND_LEAKAGE_CONTRACT.md`
   - `specs/hydroda_dataset_contract.yaml`
2. 实现或检查：
   - `hydroda/data/netcdf_audit.py`
   - `scripts/audit_netcdf.py`
   - `tests/test_netcdf_audit_smoke.py`
3. 运行 audit 命令。
4. 输出 json 和 markdown report。
5. 汇报是否可进入 Phase 1。

不要训练模型，不要构建 splits。
