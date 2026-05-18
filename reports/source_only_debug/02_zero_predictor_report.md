# Check 2: Zero Increment Predictor Sanity

## Expected: skill ≈ 0, an_rmse = forecast-only RMSE, inc_rmse = true inc RMSE


## Split: source_train
- skill_surface: 0.000000 ± 0.000000
- skill_surface range: [0.000000, 0.000000]
- skill_rootzone: 0.000000 ± 0.000000
- analysis RMSE surface: 0.009943
- analysis RMSE rootzone: 0.000909
- increment RMSE surface: 0.009943
- increment RMSE rootzone: 0.000909

## Split: source_val
- skill_surface: 0.000000 ± 0.000000
- skill_surface range: [0.000000, 0.000000]
- skill_rootzone: 0.000000 ± 0.000000
- analysis RMSE surface: 0.004219
- analysis RMSE rootzone: 0.000421
- increment RMSE surface: 0.004219
- increment RMSE rootzone: 0.000421

## Split: target_query
- skill_surface: 0.000000 ± 0.000000
- skill_surface range: [0.000000, 0.000000]
- skill_rootzone: 0.000000 ± 0.000000
- analysis RMSE surface: 0.002980
- analysis RMSE rootzone: 0.000506
- increment RMSE surface: 0.002980
- increment RMSE rootzone: 0.000506