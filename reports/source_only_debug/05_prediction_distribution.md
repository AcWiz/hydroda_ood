# Check 5: Prediction Distribution from Checkpoint


## Split: source_train
- N valid pixels (metric_mask): 344,250
- N valid pixels (loss_mask): 344,250

### Surface Increment
- PRED: mean=-0.0562014  std=0.0711677  p50=-0.0394297  p99=0.0334753
- TRUE: mean=-0.000535174  std=0.0104368  p50=0  p99=0.0338028
- std_ratio (pred/true): 6.8189
- abs_ratio (pred/true): 14.9257
- loss_mask std_ratio: 6.8189

### Rootzone Increment
- PRED: mean=-0.0121637  std=0.0560369
- TRUE: mean=-6.81155e-06  std=0.00107473
- std_ratio (pred/true): 52.1402

### Skill
- forecast RMSE surface: 0.010450
- pred RMSE surface: 0.090306
- skill surface: -7.641353
- forecast RMSE rootzone: 0.001075
- skill rootzone: -52.357378

## Split: source_val
- N valid pixels (metric_mask): 344,250
- N valid pixels (loss_mask): 344,250

### Surface Increment
- PRED: mean=-0.0726653  std=0.0670534  p50=-0.0575473  p99=0.00964933
- TRUE: mean=0.000369804  std=0.00486493  p50=0  p99=0.0167346
- std_ratio (pred/true): 13.7830
- abs_ratio (pred/true): 75.0230
- loss_mask std_ratio: 13.7830

### Rootzone Increment
- PRED: mean=-0.0109203  std=0.054507
- TRUE: mean=3.91836e-05  std=0.000448757
- std_ratio (pred/true): 121.4620

### Skill
- forecast RMSE surface: 0.004879
- pred RMSE surface: 0.098919
- skill surface: -19.274524
- forecast RMSE rootzone: 0.000450
- skill rootzone: -122.398776

⚠️ **WARNING**: pred_inc std is >10x true_inc std — possible normalization bug!

**CRITICAL**: pred/true std ratio = 13.78 — scale mismatch!

## Split: target_query
- N valid pixels (metric_mask): 44,850
- N valid pixels (loss_mask): 44,850

### Surface Increment
- PRED: mean=-0.0423612  std=0.0280212  p50=-0.0428918  p99=0.0462501
- TRUE: mean=0.00100546  std=0.00571341  p50=0  p99=0.0298633
- std_ratio (pred/true): 4.9045
- abs_ratio (pred/true): 29.5739
- loss_mask std_ratio: 4.9045

### Rootzone Increment
- PRED: mean=0.0106795  std=0.006583
- TRUE: mean=0.000197864  std=0.00108142
- std_ratio (pred/true): 6.0874

### Skill
- forecast RMSE surface: 0.005801
- pred RMSE surface: 0.050739
- skill surface: -7.746374
- forecast RMSE rootzone: 0.001099
- skill rootzone: -10.257156