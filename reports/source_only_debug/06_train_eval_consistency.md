# Check 6: Train/Eval Target Consistency

## Trainer (`hydroda/training/trainer.py`)

### Training target
```python
target = torch.stack([inc_surface, inc_rootzone], dim=1)  # line ~544
```
- Training target: **raw increment** (analysis - forecast)

### target_increment_normalization
- Appears 12 times in trainer.py
- When True: `target = (target - inc_mean_t) / inc_std_t` (normalizes target)
- When False: target stays as raw increment

### Source val evaluation (`_eval_source_val`, line ~365)
- Computes pooled RMSE then skill = 1 - rmse_pred / rmse_fcst
- `sum_sq_fcst_err = sum(target^2)` = sum of (true_increment)^2
- This is correct: RMSE(forecast, analysis) = sqrt(mean(increment^2))
- Denormalizes pred and target before computing physical-space RMSE

### Training log stats (pred vs target mean/std)
- Logs both pred and target mean/std at each log step
- If target_increment_normalization=True, logged stats are **normalized**
- If target_increment_normalization=False, logged stats are **raw** (m³/m³)

## Dataset (`hydroda/data/dataset.py`)

### loss_mask construction (line ~189)
```python
loss_mask = (
    (self._active_region_mask > 0.5)
    # base_valid_mask NOT in loss_mask ✅
```

### metric_mask construction (line ~200)
```python
metric_mask = np.logical_and(region_mask, label_valid_mask)
```
- NO base_valid_mask requirement
- Evaluates ALL region pixels with valid labels

## Metrics (`hydroda/metrics/skill.py`)

### analysis_skill_vs_forecast
- `skill = 1 - RMSE(pred_an, true_an) / RMSE(forecast, true_an)`
- Zero predictor (pred=forecast) → skill=0 ✅
- Perfect predictor (pred=true_an) → skill=1 ✅

## Consistency Verdict

| Aspect | Training | Evaluation | Consistent? |
|--------|----------|------------|-------------|
| Target variable | raw increment | raw increment | ✅ |
| Skill formula | 1 - rmse_pred/rmse_fcst | 1 - rmse_pred/rmse_fcst | ✅ |
| Mask | loss_mask | metric_mask | ❌ (if loss_mask has ch11) |
| Denormalization | in _eval_source_val | in skill.py (metric_mask) | ✅ |