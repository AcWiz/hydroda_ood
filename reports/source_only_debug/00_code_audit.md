# Source-Only Backbone 代码审计报告

## 审计日期
2026-05-15

## 审计范围
- `hydroda/data/dataset.py` — mask 构造逻辑
- `hydroda/training/trainer.py` — 训练目标、评估逻辑
- `hydroda/metrics/skill.py` — metric 公式
- `hydroda/baselines/source_only.py` — 预测器加载与推理
- `hydroda/models/resunet.py` — 模型架构
- `configs/model_resunet_main.yaml` — 训练配置

---

## 1. 训练目标

### 确认: 始终是 raw increment

**dataset.py:169-170**:
```python
increment_surface = (analysis_surface - forecast_surface).astype(np.float32)
increment_rootzone = (analysis_rootzone - forecast_rootzone).astype(np.float32)
```

**trainer.py:544**:
```python
target = torch.stack([inc_surface, inc_rootzone], dim=1)
```

训练目标是 `analysis - forecast` 的 raw increment (m³/m³ 单位)。✅ 一致。

### target_increment_normalization

当前 checkpoint 配置: `target_increment_normalization=False`

这意味着模型直接输出 raw increment，不需要反归一化。

**trainer.py:546-549**: 当 `target_increment_normalization=True` 时，target 会被归一化:
```python
target = (target - inc_mean_t) / inc_std_t
```

当前 `target_increment_normalization=False`，此分支不会执行。模型预测的是 raw increment。✅ 一致。

---

## 2. Mask 构造 — CRITICAL BUG

### loss_mask (dataset.py:189-196)

```python
loss_mask = (
    (self._active_region_mask > 0.5)
    & (base_valid_mask > 0.5)          # ← CHANNEL 11: SMAP观测可用性
    & np.isfinite(forecast_surface)
    & np.isfinite(forecast_rootzone)
    & np.isfinite(analysis_surface)
    & np.isfinite(analysis_rootzone)
).astype(np.float32)
```

### metric_mask (dataset.py:200)

```python
metric_mask = np.logical_and(region_mask, label_valid_mask).astype(np.float32)
```

**label_valid_mask** 包含: `forecast_surface`, `forecast_rootzone`, `analysis_surface`, `analysis_rootzone`, `increment_surface`, `increment_rootzone` 的 finiteness 检查。**不包含** `base_valid_mask`。

### 关键差异

| 条件 | loss_mask | metric_mask |
|------|-----------|-------------|
| active_region_mask | ✅ | ✅ |
| base_valid_mask (ch11) | ✅ | ❌ |
| 6 field finite | ✅ (4 fields) | ✅ (6 fields) |

### 数据集覆盖统计 (Check 1)

| Split | loss_mask 覆盖率 | metric_mask 覆盖率 | 比率 |
|-------|-----------------|-------------------|------|
| source_train | 6.35% | 21.01% | 30% |
| source_val | 1.37% | 21.01% | 6.5% |
| target_query | 0.32% | 2.74% | 12% |

**结论**: loss_mask 覆盖率比 metric_mask 少 70-93%。模型在训练时无法看到 ~70-93% 的评估像素。❌ **CRITICAL BUG**。

### 为什么 base_valid_mask 不应该在 loss_mask 中

1. Channel 11 是 SMAP 观测可用性标志，不表示输入特征质量
2. 其他 10 个输入通道 (vegetation opacity, brightness temperature, etc.) 在 SMAP 不可用时仍然有效
3. 模型从这些通道中仍可提取有用信息来预测 DA increment
4. 卷积神经网络依赖空间连续性，SMAP 像素聚集分布破坏了这种连续性
5. 模型在 SMAP 与非 SMAP 像素之间的空间分布差异导致泛化失败

---

## 3. 评估指标

### analysis_skill_vs_forecast (skill.py:43-61)

```python
def analysis_skill_vs_forecast(pred_analysis, true_analysis, forecast, mask):
    p, t, f = _valid_flat(pred_analysis, true_analysis, forecast, mask=mask)
    rmse_pred = _rmse(p, t)
    rmse_fcst = _rmse(f, t)
    return 1.0 - rmse_pred / rmse_fcst
```

公式: `skill = 1 - RMSE(pred_an, true_an) / RMSE(forecast, true_an)`

**Sanity check 验证**:
- Zero predictor (pred_an = forecast): skill = 0.0 ✅
- Perfect predictor (pred_an = true_an): skill = 1.0 ✅

Metric 公式完全正确。

### trainer._eval_source_val 中的 skill 计算 (trainer.py:450-455)

```python
rmse_s = sqrt(sum_sq_err_s / n_pixels_s)
rmse_fcst_s = sqrt(sum_sq_fcst_err_s / n_pixels_s)  # sum_sq_fcst_err = sum(target^2)
skill_s = 1.0 - rmse_s / rmse_fcst_s
```

其中 `sum_sq_fcst_err_s = sum(target_flat^2)` = sum of (true_increment)^2。

验证: RMSE(forecast, analysis) = RMSE(forecast, forecast + true_increment) = sqrt(mean(true_increment^2)) = sqrt(mean(target^2))。✅ 正确。

---

## 4. 模型初始化 — SECONDARY BUG

### zero_raw_increment_init=False

当前 checkpoint 配置: `zero_raw_increment_init=False`

**resunet.py:41-45**:
```python
if zero_raw_increment_init:
    nn.init.zeros_(self.head.weight)
    nn.init.zeros_(self.head.bias)
```

当 `zero_raw_increment_init=False` 时，`head` (Conv2d) 使用默认的 Kaiming 初始化。

**后果**:
- 随机初始化时，输出权重的 std ≈ 0.1
- 初始 pred_increment ≈ 0.05-0.08 (与 Check 5 数据一致)
- 初始 skill ≈ -6 到 -37 (远低于 forecast-only baseline)
- 即使经过 12 个 epoch 训练，模型仍未收敛到可接受范围

**修复**: `zero_raw_increment_init: true` 将输出头初始化为零，使初始预测为零增量，initial skill ≈ 0。

---

## 5. 模型架构

**SmallResUNet** (resunet.py:26-56):
- 3 层 encoder + bottleneck + 2 层 decoder
- ConvBlock: 2×Conv2d + GroupNorm + GELU + skip connection
- Head: 1×1 Conv2d (width → 2 channels)
- 参数量 (width=32): ~74,000

架构本身无问题。问题是输入通道 11 (base_valid_mask) 通过 loss_mask 间接影响了训练信号分布。

---

## 6. 预测器推理

**SourceOnlyBackbonePredictor** (source_only.py):
- 加载 checkpoint → 初始化 SmallResUNet → 加载权重
- predict(): normalize input → model.forward() → denormalize (if needed) → pred_an = forecast + pred_inc
- 当 `target_increment_normalization=False` 时，不进行增量反归一化

**确认**: 预测器正确使用 `_has_inc_norm` 标志判断是否需要反归一化。当前 checkpoint 中 `inc_mean=None, inc_std=None`，因此 `_has_inc_norm=False`，不进行反归一化。✅ 一致。

---

## 7. 数据流完整路径验证

```
DA.nc → HydroDADataset.__getitem__()
  → input[12, H, W] → x
  → target[4, H, W]  → analysis_surface, analysis_rootzone
  → forecast_surface = input[0], forecast_rootzone = input[1]
  → increment = analysis - forecast
  → loss_mask = region & ch11 & finite(4 fields)
  → metric_mask = region & finite(6 fields)  [NO ch11]

Trainer.train():
  → x_norm = (x - ch_mean) / ch_std
  → target = stack([inc_surface, inc_rootzone])
  → pred = model(x_norm)    # [B, 2, H, W]
  → loss = MaskedHuberLoss(pred, target, loss_mask)  # delta=0.01

SourceOnlyBackbonePredictor.predict():
  → x_norm = (x - ch_mean) / ch_std
  → pred = model(x_norm)
  → (no denorm since inc_mean/inc_std is None)
  → pred_analysis = forecast + pred_increment

Metric:
  → skill = 1 - RMSE(pred_an, true_an) / RMSE(forecast, true_an)
  → mask = metric_mask (wider than loss_mask)
```

**唯一不匹配**: loss_mask (训练) ≠ metric_mask (评估)。❌

---

## 8. 根因总结

| 优先级 | Bug | 位置 | 影响 |
|--------|-----|------|------|
| P0 | loss_mask 包含 base_valid_mask | dataset.py:189-196 | 训练只覆盖 2-7% 的空间像素，评估覆盖 21%，导致泛化崩溃 |
| P1 | zero_raw_increment_init=False | model config | 随机初始化导致初始 pred_inc 远离 0，initial skill 为大的负数 |

### 修复后的预期行为

1. **loss_mask 修复**: 训练覆盖 ~21% 的空间像素 (与 metric_mask 对齐)，模型能学习全空间映射
2. **zero_raw_increment_init 修复**: 初始预测为零增量，initial skill ≈ 0，模型从 forecast-only baseline 开始优化
3. **source_val skill**: 应从 -37 提升到 > 0 (至少不低于 forecast-only)
