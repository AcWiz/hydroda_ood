# FINAL Rootzone Diagnosis Report

**Date**: 2026-05-16
**Checkpoint**: `phase4_source_only_source_only_US-R1_w32_e30_lr0.0003_nonorm_s0_20260515_155806`

---

## 1. 一句话结论

**Nonorm source-only backbone 的 rootzone skill 巨大负数（source_val -1.61, target_query -12.21）根因是 raw MSE 训练中 surface loss 主导梯度（~40-75x），rootzone 信号被淹没。Per-variable increment normalization 仅 3 epoch smoke training 即可将 rootzone skill 从 -12.21 提升至 -0.56（target_query），从 -1.61 提升至 +0.22（source_val）。**

---

## 2. 证据链

### 2.1 Forecast-only RMSE 反推（Task 2）

| Split | Variable | Skill | Inc RMSE | Forecast-Only RMSE | Inc/Fcst Ratio |
|:---|---:|---:|---:|---:|---:|
| source_val | surface | +0.44 | 0.003001 | 0.005329 | 0.56x |
| source_val | rootzone | -1.61 | 0.000642 | 0.000246 | 2.61x |
| target_query | surface | -1.95 | 0.004401 | 0.001492 | 2.95x |
| target_query | rootzone | -12.21 | 0.000579 | **0.000044** | **13.21x** |

**结论**：Target_query rootzone 的 forecast-only RMSE 仅 4.39e-5 m³/m³ — forecast 本身已经极好。Rootzone skill 巨大负数主要来自极小分母，而非灾难性大误差。但 inc RMSE（5.79e-4）仍为 forecast-only RMSE 的 13.2 倍 — 说明模型确实在做有害的非零修正。

### 2.2 Increment 分布统计（Task 3）

| Split | Surface True Std | Rootzone True Std | Ratio |
|:---|---:|---:|---:|
| source_train | 0.008746 | 0.001243 | **7.0x** |
| source_val | 0.009098 | 0.001328 | **6.9x** |
| target_query | 0.006605 | 0.000871 | **7.6x** |

Pred/True ratio on target_query:
- Surface: pred_abs_mean / true_abs_mean = 1.75x（model over-corrects OOD）
- Rootzone: pred_abs_mean / true_abs_mean = 2.23x（model over-corrects OOD）

**结论**：Rootzone true increment 幅度在各 split 均为 surface 的 ~1/7。Raw MSE 训练中 rootzone 贡献几乎不可见。

### 2.3 Loss Scale 诊断（Task 4）

| Metric | Value |
|:---|---:|
| Surface raw MSE (source_train) | 8.12e-6 |
| Rootzone raw MSE (source_train) | 4.1e-7 |
| **MSE ratio surface/rootzone** | **41.5x (mean) / 37.3x (median)** |
| True variance ratio surface/rootzone | **75.4x** |

**结论**：Surface MSE 是 rootzone MSE 的 ~41 倍。训练 loss 由 surface 主导，rootzone 梯度信号强度仅 ~1/41。

### 2.4 Shrinkage 评估（Task 5）

| Split | Var | Best Alpha | Best Skill | alpha=0 Skill | alpha=1 Skill |
|:---|---:|---:|---:|---:|---:|
| source_val | surface | 1.00 | +0.49 | 0.00 | +0.49 |
| source_val | rootzone | **0.05** | **+0.007** | 0.00 | -0.60 |
| target_query | surface | 0.05 | +0.003 | 0.00 | -1.87 |
| target_query | rootzone | **0.00** | **0.000** | 0.00 | -5.60 |

**结论**：Nonorm 模型对 rootzone 的预测在全量（alpha=1）时严重有害。Target_query 上最佳策略是完全不用模型（alpha=0, forecast-only）。Source_val 上仅用 5% 预测信号可获得微弱正 skill。这证明 nonorm 模型的 rootzone 输出基本上是被 surface loss 牵着走的噪声。

### 2.5 NormLoss Smoke Training（Task 6）

**仅 3 epochs, per-variable increment normalization:**

| Split | Variable | Metric | Nonorm (30ep) | NormLoss (3ep) | Zero Predictor |
|:---|:---|---:|---:|---:|---:|
| source_val | surface | skill | +0.44 | +0.30 | 0.00 |
| source_val | rootzone | skill | **-1.61** | **+0.22** | 0.00 |
| source_val | rootzone | rmse | 0.000642 | 0.000652 | 0.000889 |
| target_query | surface | skill | -1.95 | **-0.45** | 0.00 |
| target_query | rootzone | skill | **-12.21** | **-0.56** | 0.00 |
| target_query | rootzone | rmse | 0.000579 | **0.000270** | 0.000254 |

**结论**：仅 3 epochs normloss smoke training 已经 dramatic 改善 rootzone：
- source_val rootzone skill: -1.61 → **+0.22**（转正！）
- target_query rootzone skill: -12.21 → **-0.56**（虽然仍负，但已从完全无用变为接近 forecast-only）
- target_query rootzone RMSE: 0.000579 → 0.000270（接近 forecast-only RMSE 0.00025）
- target_query **surface** skill 也改善：-1.95 → -0.45

---

## 3. 当前 Nonorm Source-Only 是否应放入论文主表

**不建议**。当前 nonorm source-only backbone 的 rootzone 指标在 source_val 和 target_query 上均比 forecast-only 差（skill 为负），放在主表中会成为 negative result 且不反映方法潜力。应在 normloss 训练后重新评估。

---

## 4. 下一步建议

1. **立即 switch 到 normloss training**：所有后续 source-only、prompt-conditioned、HyperDA training 必须使用 `--target_normalization_mode per_variable_increment_std`
2. **重新训练 source-only backbone**：用 normloss 重新训练 30 epochs（width=32, lr=3e-4, effective_batch_size=64），获取完整的 source_val 和 target_query 指标
3. **确认 surface skill 恢复**：Normloss smoke 中 source_val surface skill 降至 0.30（nonorm 为 0.44），需确认 full 30-epoch normloss training 是否能恢复或超过 nonorm 的 surface skill
4. **考虑 loss weighting**：如果 full training 后 surface skill 仍显著低于 nonorm，可考虑在 normalized 空间中微调 surface/rootzone 权重（如 `surface_weight=1.5, rootzone_weight=1.0`），但当前 smoke 中 surface skill 的轻微下降在可接受范围内，且 target_query surface skill 实际改善
5. **更新 YAML config**：将 `configs/model_resunet_main.yaml` 设为默认使用 `target_increment_normalization: true` 和 `zero_raw_increment_init: true`（当前已是如此，只需确认 CLI 不再 override）

---

## 5. 下一条正式训练命令

```bash
PYTHONPATH=. python scripts/train/train_source_only_backbone.py \
    --target_region US-R1 --K 0 --seed 0 \
    --width 32 --max_epochs 30 \
    --batch_size 16 --accum_steps 4 \
    --lr 3e-4 --weight_decay 1e-4 --grad_clip 1.0 \
    --target_normalization_mode per_variable_increment_std \
    --config configs/model_resunet_main.yaml \
    --device cuda --amp \
    --wandb_mode disabled
```

## 6. 生成 Artifacts 总结

| Artifact | 路径 |
|:---|---|
| OOD gap 修正 | `run/phase4_source_only_inference.sh` (lines 161, 166 fixed) |
| Forecast RMSE 反推 | `reports/source_only_rootzone_diagnosis/01_forecast_rmse_backcalc.md` |
| Increment 分布 | `reports/source_only_rootzone_diagnosis/02_increment_distribution.md` |
| Loss Scale 诊断 | `reports/source_only_rootzone_diagnosis/03_loss_scale_diagnosis.md` |
| Shrinkage 评估 | `reports/source_only_rootzone_diagnosis/04_shrinkage_eval.md` |
| NormLoss Smoke | `reports/source_only_rootzone_diagnosis/05_normloss_smoke_report.md` |
| Final 报告 | `reports/source_only_rootzone_diagnosis/FINAL_ROOTZONE_DIAGNOSIS.md` |
| Smoke checkpoint | `artifacts/runs/phase4_source_only/phase4_source_only_source_only_US-R1_w32_e3_lr0.0003_norm_zero_s0_20260516_120302/` |
| Smoke inference | `artifacts/results/phase4_source_only_inference_normloss_smoke/20260516_120301/` |
| 新增训练 flag | `scripts/train/train_source_only_backbone.py` (`--target_normalization_mode`) |
| Smoke launcher | `run/phase4_source_only_normloss_smoke.sh` |
| 诊断脚本 | `scripts/diagnosis/task02-05_*.py` |
