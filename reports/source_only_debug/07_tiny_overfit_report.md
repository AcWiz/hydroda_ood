# Check 7: Tiny Overfit Test

## Setup
- Fixed batch of 4 samples (16277 in dataset)
- SmallResUNet (width=16)
- LR=1e-3, Adam, 500 steps
- loss_mask coverage: 21.01%
- metric_mask coverage: 21.01%

## Model Initialization
- zero_raw_increment_init=True ✅
- Initial pred mean: 0.00000000 (should be 0.0)

## Zero-Predictor Baseline (on loss_mask)
- surface RMSE: 0.024586
- rootzone RMSE: 0.003050

## Zero-Predictor Baseline (on metric_mask)
- surface RMSE: 0.024586
- rootzone RMSE: 0.003050

## Final Model (step 500)
- RMSE surface (metric_mask): 0.004114
- Skill surface (metric_mask): 0.665310

✅ Model learns on loss_mask — final RMSE < 50% of zero-predictor

✅ Model achieves positive skill on metric_mask (skill=0.6653)