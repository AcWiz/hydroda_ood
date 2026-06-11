# HyperDA Target Adaptation Design

> Superseded by Protocol V4.4 zero/few-shot generalization. This design is
> retained for legacy/internal full-target reproduction context only. The active
> paper-facing protocol freezes the source prior after source training, uses
> `target_context=2015-2021` input-side prompt context, optional K-shot
> `target_support` labels for K in {0,4,12}, no main `target_val`, and final
> `target_eval=2023-2025`.

## Historical Goal

This document captured the V4.3 historical target adaptation protocol before
the V4.4 zero/few-shot migration:

```text
source_fit:    2015-2021 source domains
source_val:    2022 source domains for source checkpoint / architecture selection
target_train:  2015-2021 held-out target domain for target-specific adaptation
target_val:    2022 held-out target domain for preregistered adaptation selection
target_eval:   2023-2025 held-out target domain for final evaluation only
```

This is no longer the paper-facing claim. The active claim is zero/few-shot
target generalization with K in {0,4,12}.

## Method Position

HyperDA remains a parameter-space transfer method:

```text
z_R = E_phi(P_R)
zeta_gen = H_psi(z_R)
DeltaSM_hat = f_theta0(x; zeta_R)
```

The target adaptation stage freezes the shared backbone prior and the source-trained hypernetwork:

```text
frozen:    theta0, H_psi, adapter basis banks
trainable: target latent, adapter coefficient residuals, lightweight operator residuals, output-head residual, residual gain
```

This kept the method distinct from ordinary full fine-tuning while using the
complete labeled target training period in the legacy/internal reproduction
path.

## Network Design

The main implementation should build on the current `HyperAdapterConditionalResUNet`:

- keep the existing ResUNet encoder/decoder topology;
- keep basis-factorized adapters at bottleneck, decoder-2, and decoder-1;
- add an optional target latent vector that shifts the prompt representation before frozen generated-operator modules consume it;
- add optional trainable coefficient residuals for the adapter mixture logits;
- add a residual output calibration layer that applies per-output-channel monthly gain and bias.

The residual gain is:

```text
DeltaSM_final[c] = gain[month, c] * DeltaSM_operator[c] + bias[month, c]
```

with gains initialized to one and biases initialized to zero.

## Target Adaptation Protocol

The target adaptation script should:

- load a source-trained prompt-conditioned or HyperDA checkpoint;
- freeze `theta0`, `H_psi`, and source-trained adapter basis parameters;
- train only the registered target adaptation parameters on `target_train` dates;
- optionally select the adaptation checkpoint or residual-gain step using `target_val=2022`;
- save metadata recording trainable parameter names, trainable parameter count, adaptation dates hash, target validation usage, and split manifest hash;
- never read `target_eval=2023-2025` labels during adaptation.

## Loss

The target adaptation objective is:

```text
L = L_increment
  + 0.25 * L_analysis
  + lambda_prior * ||coefficient_residual||^2
  + lambda_latent * ||target_latent||^2
  + lambda_gain * ||gain - 1||^2
  + lambda_gain_smooth * monthly_smoothness(gain, bias)
```

The first executable implementation can expose the regularization terms and use only the modules needed by tests and the follow-up adaptation script. The data-path implementation must preserve the existing `source_fit_only` normalization discipline.

## Baselines and Ablations

Legacy V4.3 tables compared:

```text
Forecast-only
Source-only backbone
Prompt-conditioned shared backbone
Adapter tuning on target_train
LoRA tuning on target_train
HyperDA generated operator
HyperDA-Adapt: target latent + coefficient residual + residual gain
HyperDA-Refine: HyperDA-Adapt plus lightweight operator/head residual
```

Under V4.4, the main table is forecast-only, source-only, prompt-conditioned,
and HyperDA K=0/4/12. Full-target variants from this document are
secondary/internal reproduction results.

## Verification

Required tests:

- target adaptation modules expose exactly the intended trainable parameters when freeze is enabled;
- target latent changes the prompt-conditioned model output while frozen backbone parameters remain frozen;
- monthly residual gain initializes as identity and applies month-specific gain/bias;
- protocol tests assert that HyperDA target adaptation uses target 2015-2021 for adaptation, target 2022 for selection only, and target 2023-2025 for final evaluation only.
