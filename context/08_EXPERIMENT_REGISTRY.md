# 08 Experiment Registry

本文件定义实验命名、优先级和最小完成标准。

---

## 1. Experiment ID format

```text
{benchmark}_{phase}_{method}_{target}_{K}_{seed}_{version}
```

例：

```text
usloo_p3_forecast_US-R1_K0_s0_v1
usloo_p3_ridge_US-R3_K12_s2_v1
usloo_p5_hyrao_US-R5_K4_s0_v1
```

---

## 2. Phase A: US-only development

### A0 audit

```text
country = US
data = DA.nc
outputs = audit json + markdown report
```

### A1 region quality

```text
US-R1..US-R6 masks
region_quality_us.csv
```

### A2 simple baselines

Grid：

```text
target_region: US-R1..US-R6
K: 0,4,12,24
seed: 0..4
method: forecast, source_mean, target_mean, monthly_mean, ridge
```

---

## 3. Phase B: Full cross-continent benchmark

当 CN/AU 到位：

```text
Train US+CN -> AU
Train US+AU -> CN
Train CN+AU -> US
```

---

## 4. Phase C: Same-regime transfer

Regime triads：

```text
dryland_sparse_vegetation
irrigated_managed_agriculture
humid_high_vegetation
semi_arid_transition
rainfed_agriculture
mountain_cold_terrain_stress
```

---

## 5. Completion status labels

```text
planned
implemented
smoke_tested
full_run
validated
paper_ready
blocked
```

每个 experiment card 必须写 status。
