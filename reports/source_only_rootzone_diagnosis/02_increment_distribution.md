# Task 3: Increment Distribution Statistics

## Per-Split Distribution Summary

| Split        | Variable      | N Pixels | Mean     | Std      | Abs Mean | Abs P95  | Abs P99  |
|:-------------|:--------------|---------:|---------:|---------:|---------:|---------:|---------:|
| source_train_true_surface |               | 17,212,500 | -0.000315 | 0.008746 | 0.002817 | 0.017256 | 0.040483 |
| source_train_true_rootzone |               | 17,212,500 | +0.000029 | 0.001243 | 0.000282 | 0.001457 | 0.004362 |
| source_train_pred_surface |               | 17,212,500 | -0.000178 | 0.008354 | 0.002880 | 0.016801 | 0.038448 |
| source_train_pred_rootzone |               | 17,212,500 | +0.000032 | 0.000958 | 0.000339 | 0.001368 | 0.003608 |
| source_val_true_surface |               | 99,557,100 | -0.000540 | 0.009098 | 0.003015 | 0.018070 | 0.040639 |
| source_val_true_rootzone |               | 99,557,100 | +0.000010 | 0.001328 | 0.000287 | 0.001438 | 0.004081 |
| source_val_pred_surface |               | 99,557,100 | -0.000394 | 0.008483 | 0.003043 | 0.017442 | 0.038020 |
| source_val_pred_rootzone |               | 99,557,100 | +0.000007 | 0.000978 | 0.000347 | 0.001386 | 0.003491 |
| target_query_true_surface |               | 4,485,000 | -0.000258 | 0.006605 | 0.001846 | 0.011369 | 0.027582 |
| target_query_true_rootzone |               | 4,485,000 | +0.000016 | 0.000871 | 0.000134 | 0.000663 | 0.001926 |
| target_query_pred_surface |               | 4,485,000 | -0.000602 | 0.008944 | 0.003226 | 0.019512 | 0.040508 |
| target_query_pred_rootzone |               | 4,485,000 | +0.000037 | 0.000861 | 0.000299 | 0.001248 | 0.002930 |

## Pred/True Ratios

| Split        | Variable  | Pred Std/True Std | Pred AbsMean/True AbsMean | Pred AbsP95/True AbsP95 |
|:-------------|:----------|------------------:|--------------------------:|------------------------:|
| source_train | surface   | 0.955 | 1.022 | 0.974 |
| source_train | rootzone  | 0.771 | 1.201 | 0.939 |
| source_val   | surface   | 0.932 | 1.009 | 0.965 |
| source_val   | rootzone  | 0.736 | 1.209 | 0.964 |
| target_query | surface   | 1.354 | 1.748 | 1.716 |
| target_query | rootzone  | 0.989 | 2.234 | 1.882 |

## Key Findings

- **source_train**: surface true_inc std = 0.008746, rootzone true_inc std = 0.001243, ratio = 7.0x
- **source_val**: surface true_inc std = 0.009098, rootzone true_inc std = 0.001328, ratio = 6.9x
- **target_query**: surface true_inc std = 0.006605, rootzone true_inc std = 0.000871, ratio = 7.6x

> The rootzone true increment is dramatically smaller than surface in magnitude, confirming that raw MSE training is dominated by surface loss.