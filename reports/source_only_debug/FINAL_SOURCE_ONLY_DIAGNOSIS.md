# HydroDA-OOD Source-Only 诊断报告（最终版）

**日期**: 2026-05-15
**状态**: BUG 已修复并验证

---

## 一句话结论

Pipeline 有两个 CRITICAL bug，均已修复并验证。修复后 smoke training (1 epoch) 在 source_val 上 skill_surface 从 **-37 → +0.45**，skill_rootzone 从 **-150 → +0.10**。

---

## 发现的 Bug 及修复

### Bug 1 (CRITICAL): loss_mask 包含 base_valid_mask

**位置**: `hydroda/data/dataset.py` 第 189-196 行

**原因**: `loss_mask` 要求 `base_valid_mask > 0.5`（channel 11, SMAP 观测可用性），导致训练只在 ~2-7% 的空间像素上进行（SMAP 观测覆盖区），但评估在 ~21% 的像素上进行（所有有效标签像素）。模型无法泛化到未见过的 non-SMAP 像素。

**修复**: 移除 `base_valid_mask > 0.5` 要求，使 `loss_mask` 与 `metric_mask` 对齐。

```python
# 修复前:
loss_mask = (region_mask & base_valid_mask & finite_4_fields)

# 修复后:
loss_mask = (region_mask & finite_6_fields)  # 包含 increment finiteness
```

**验证 (Check 1)**:
| Split | loss_mask 修复前 | loss_mask 修复后 | metric_mask | 对齐? |
|-------|-----------------|-----------------|-------------|-------|
| source_train | 6.35% | 21.01% | 21.01% | ✅ 1.000 |
| source_val | 1.37% | 21.01% | 21.01% | ✅ 1.000 |
| target_query | 0.32% | 2.74% | 2.74% | ✅ 1.000 |

### Bug 2 (HIGH): zero_raw_increment_init=False

**位置**: 训练时未传递 `--zero_raw_increment_init` 标志

**原因**: 模型输出头使用随机初始化，初始 pred_increment ~0.07 (m³/m³)，导致初始 skill ~-6。正确初始化为 0 使模型从 forecast-only baseline (skill=0) 开始优化。

**修复**: 训练时确保 `zero_raw_increment_init=True`（通过 `--zero_raw_increment_init` CLI 标志）。

### Bug 3 (MEDIUM): trainer._eval_source_val mask 索引错误

**位置**: `hydroda/training/trainer.py` 第 427-428 行

**原因**: `loss_mask` shape 为 [B, H, W]（无 channel 维度），但 eval 代码使用 `mask_bool[:, 0]` 错误地沿 H 维度切片，导致 IndexError。

**修复**: 使用 `valid_s = mask_bool`（全空间 mask）替代 `valid_s = mask_bool[:, 0]`。

### Bug 4 (MEDIUM): YAML boolean 参数加载

**位置**: `scripts/train/train_source_only_backbone.py` 第 55-56 行

**原因**: `action="store_true"` 的 argparse 参数默认值为 `False`（非 `None`），导致 YAML 配置中的 `True` 值无法通过 `set_defaults()` 覆盖（因为 `parser.get_default(arg_key) is None` 检查失败）。

**修复**: 将 `action="store_true"` 的 default 设为 `None`，并添加对应的 `--no_*` 标志以便显式禁用。

---

## Sanity Check 完整验证结果

### Check 1: Dataset Audit ✅

| Split | N samples | loss_mask | metric_mask | loss/metric ratio |
|-------|-----------|-----------|-------------|------------------|
| source_train | 16277 | 21.01% | 21.01% | **1.000** |
| source_val | 2892 | 21.01% | 21.01% | **1.000** |
| target_query | 3359 | 2.74% | 2.74% | **1.000** |

### Check 2: Zero Predictor ✅

| Split | skill_surface | skill_rootzone |
|-------|--------------|----------------|
| source_train | 0.000000 | 0.000000 |
| source_val | 0.000000 | 0.000000 |
| target_query | 0.000000 | 0.000000 |

### Check 3: Perfect Predictor ✅

| Split | skill_surface | skill_rootzone | an_rmse_s | inc_rmse_s |
|-------|--------------|----------------|-----------|------------|
| source_train | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| source_val | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| target_query | 1.000000 | 1.000000 | 0.000000 | 0.000000 |

### Check 4: Checkpoint Config (旧 checkpoint) ✅

确认旧 checkpoint 配置:
- `target_increment_normalization=False`
- `zero_raw_increment_init=False`
- `inc_mean=None`, `inc_std=None`
- width=32, lr=0.0003, max_epochs=30

### Check 5: Prediction Distribution (旧 checkpoint) ❌

旧 checkpoint 在修复后的数据集上仍然表现不佳（预期结果——此 checkpoint 用错误 loss_mask 训练）:

| Split | pred/true std ratio | skill_surface |
|-------|--------------------|---------------|
| source_train | 6.82x | -7.64 |
| source_val | 13.78x | -19.27 |
| target_query | 4.90x | -7.75 |

### Check 6: Train/Eval Consistency ✅

修复后代码审计确认:
- Training target: raw increment ✅
- Skill formula: consistent ✅
- Mask: loss_mask = metric_mask ✅
- Denormalization: consistent (disabled when no inc_norm) ✅

### Check 7: Tiny Overfit ✅

- Model: SmallResUNet (width=16), zero_raw_increment_init=True
- loss_mask coverage: 21.01%, metric_mask: 21.01%
- Initial pred mean: 0.00000000 (zero-init 验证通过)
- **Final skill (step 500): 0.6653** (4 batches, 500 steps)

```
step   0: loss=0.003645  skill_s=0.0028    (near forecast-only baseline)
step 250: loss=0.000671  skill_s=0.5827    (significant improvement)
step 500: loss=0.000324  skill_s=0.6653    (strong positive skill)
```

---

## Smoke Training 验证结果

**配置**: SmallResUNet (width=32), 1 epoch, batch=4, lr=1e-3, zero_raw_increment_init=True, GPU 1

**关键指标**:

| 指标 | 旧 checkpoint (buggy, 12 epochs) | Smoke training (fixed, 1 epoch) |
|------|----------------------------------|-------------------------------|
| source_val skill_surface | **-36.7** | **+0.45** |
| source_val skill_rootzone | **-150.1** | **+0.10** |
| source_val rmse_surface | 0.1009 | 0.0050 |
| pred_surface std | 0.065 (24x true) | ~0.007 (~1x true) |
| Initial pred mean | ~0.055 | 0.000 (zero-init) |
| Best alpha (shrinkage) | N/A | 0.01 |

**训练日志摘录**:
```
Step 0:   pred_s=0.000/0.000  (zero-init 验证)  loss=0.0025
Step 400: pred_s=-0.002/0.004  loss=0.0008
Source val: rmse_s=0.0050  rmse_r=0.0012  skill_s=0.4484  skill_r=0.1023
```

---

## 修改文件清单

| 文件 | 修改内容 | 优先级 |
|------|---------|--------|
| `hydroda/data/dataset.py` | 移除 loss_mask 中的 base_valid_mask 要求 | P0 |
| `hydroda/training/trainer.py` | 修复 _eval_source_val 的 mask 索引 ([:, 0] → 全 mask) | P0 |
| `scripts/train/train_source_only_backbone.py` | 修复 boolean argparse 参数的 YAML 加载 | P1 |
| `configs/model_resunet_debug.yaml` | zero_raw_increment_init: true | P1 |
| `scripts/debug_source_only_pipeline.py` | 增强 debug 脚本 (时间统计/seasonal stats/--checks/--skip-checks) | P2 |
| `reports/source_only_debug/00_code_audit.md` | 新增: 完整代码审计报告 | P2 |

---

## 下一步

1. **重新训练正式 source-only backbone**: 使用修复后的 pipeline 和 `configs/model_resunet_main.yaml`，30 epochs
2. **K-cycle 实验**: 用 K=4, K=12 重新运行 source-only（当前仅 K=0）
3. **Phase 4B**: 在验证后的 source-only backbone 上构建 prompt-conditioned shared backbone
4. **多区域扩展**: 扩展到 US-R2 到 US-R6

---

## 给共同一作的建议

### 旧 source-only checkpoint 的处理
**必须丢弃**。`phase4_source_only_source_only_US-R1_w32_e30_lr0.0003_nonorm_nozero_s0_20260512_142418` 是在错误 loss_mask 和无 zero-init 条件下训练的，结果完全不代表模型能力。

### 是否需要重跑 forecast-only
**不需要**。Forecast-only 不依赖模型，其 baseline 始终正确。

### 论文主表中 source-only 的定位
修复后的 source-only 在仅 1 epoch 训练后就达到 skill_surface=+0.45。完整的 30 epoch 训练预期可以获得更高的 skill。这为所有后续方法（prompt-conditioned backbone, adapter, LoRA, HyperDA）提供了正确且有意义的 baseline。
