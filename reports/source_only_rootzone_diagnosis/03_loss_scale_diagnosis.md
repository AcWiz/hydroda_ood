# Task 4: Loss Scale Diagnosis

Evaluated on 800 source_train samples, 800 requested.

## Model MSE on Source-Train (Nonorm Checkpoint)

| Metric | Value |
|:---|---:|
| Surface MSE (raw) | 0.00000812 |
| Rootzone MSE (raw) | 0.00000041 |
| MSE Ratio (surface/rootzone) mean | 41.5x |
| MSE Ratio (surface/rootzone) median | 37.3x |

## True Increment Scale (Source-Train)

| Metric | Value |
|:---|---:|
| True surface increment std | 0.007787 |
| True rootzone increment std | 0.00089690 |
| Std ratio (surface / rootzone) | 8.7x |
| Variance ratio (surface / rootzone) | 75.4x |
| AbsMean ratio (surface / rootzone) | 10.3x |

## Key Findings

- **Raw MSE surface/rootzone ratio**: ~41x (surface dominates)
- **True increment std ratio**: ~9x (surface ≈ 9x larger)
- **True increment variance ratio**: ~75x
- **Implication**: the training loss is dominated by surface MSE. Rootzone gets ~1/41th the gradient signal.

> Raw MSE training is fundamentally surface-dominated. The rootzone signal is ~100x weaker in variance, making it nearly invisible to the optimizer. Per-variable normalization is required to give rootzone equal footing.