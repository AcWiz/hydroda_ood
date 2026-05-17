# Task 5: Conservative Shrinkage Evaluation

`alpha × pred_increment` added to forecast. `alpha=0` = forecast-only.

## Source Val (in-domain)

| Alpha | Surface Skill | Surface RMSE | Surface Corr | Rootzone Skill | Rootzone RMSE | Rootzone Corr |
|------:|-------------:|------------:|------------:|--------------:|-------------:|-------------:|
| 0.00 | +0.0000 | 0.007930 | nan | +0.0000 | 0.000889 | nan |
| 0.05 | +0.0417 | 0.007587 | 0.9008 | +0.0072 | 0.000859 | 0.7517 |
| 0.10 | +0.0810 | 0.007246 | 0.9008 | +0.0017 | 0.000830 | 0.7517 |
| 0.20 | +0.1555 | 0.006574 | 0.9008 | -0.0205 | 0.000774 | 0.7517 |
| 0.30 | +0.2260 | 0.005919 | 0.9008 | -0.0534 | 0.000722 | 0.7517 |
| 0.50 | +0.3531 | 0.004687 | 0.9008 | -0.1503 | 0.000631 | 0.7517 |
| 0.70 | +0.4518 | 0.003648 | 0.9008 | -0.2929 | 0.000567 | 0.7517 |
| 1.00 | +0.4906 | 0.002936 | 0.9008 | -0.6027 | 0.000542 | 0.7517 |

## Target Query (OOD)

| Alpha | Surface Skill | Surface RMSE | Surface Corr | Rootzone Skill | Rootzone RMSE | Rootzone Corr |
|------:|-------------:|------------:|------------:|--------------:|-------------:|-------------:|
| 0.00 | +0.0000 | 0.003281 | nan | +0.0000 | 0.000254 | nan |
| 0.05 | +0.0034 | 0.003152 | 0.5068 | -0.1524 | 0.000248 | 0.4158 |
| 0.10 | -0.0194 | 0.003038 | 0.5068 | -0.3538 | 0.000244 | 0.4158 |
| 0.20 | -0.1038 | 0.002866 | 0.5068 | -0.8114 | 0.000240 | 0.4158 |
| 0.30 | -0.2319 | 0.002776 | 0.5068 | -1.3217 | 0.000243 | 0.4158 |
| 0.50 | -0.5946 | 0.002849 | 0.5068 | -2.4524 | 0.000268 | 0.4158 |
| 0.70 | -1.0593 | 0.003221 | 0.5068 | -3.6761 | 0.000314 | 0.4158 |
| 1.00 | -1.8659 | 0.004181 | 0.5068 | -5.6018 | 0.000408 | 0.4158 |

## Best Alpha per Split/Variable

| Split       | Variable  | Best Alpha | Best Skill |
|:------------|:----------|----------:|----------:|
| source_val  | surface   | 1.00 | +0.4906 |
| source_val  | rootzone  | 0.05 | +0.0072 |
| target_query | surface   | 0.05 | +0.0034 |
| target_query | rootzone  | 0.00 | +0.0000 |

## Key Findings

- **Source val rootzone**: forecast-only (alpha=0) skill ~0, full model (alpha=1) skill -0.60
- **Target query rootzone**: forecast-only (alpha=0) skill ~0, full model (alpha=1) skill -5.60
- Shrinkage can partially rescue rootzone OOD performance but cannot fully match forecast-only
- Surface benefits from model predictions; rootzone is harmed by them in raw MSE space

> Shrinking predictions toward zero (forecast-only) monotonically improves rootzone skill at the cost of surface skill. The optimal trade-off depends on the application.