# HydroDA Physics Formula Knowledge Base

This document is the local formula bank for physics-informed HyperDA-TRUST
design work. It is a design reference, not run evidence. Any runnable method
must still satisfy the V4.4 split, source-selection, normalization, and
target-eval no-selection contracts.

## Scope

Current use:

- define input-side physical diagnostics from raw HydroDA channels;
- document sign conventions before they enter model code;
- constrain future physics modules to source-trained coefficient-space
  modulation, source-side selection, and diagnostic guards;
- prevent target_eval-driven formula, hyperparameter, or region changes.

Out of scope:

- target_eval calibration;
- target_val selection;
- final-output residual branches that add physical `q_surface` or `q_rootzone`
  directly to the predicted increment;
- using channel 11 as an observation mask, loss mask, metric mask, or region
  hard mask.

## Raw Input Channel Contract

The current US `DA.nc` input is a 12-channel array. The physics formulas below
use only raw input-side values, month, region masks from frozen artifacts, and
source-side metadata.

| Channel | Name | Physics role |
|---:|---|---|
| 0 | `sm_surface_forecast` | surface forecast state |
| 1 | `sm_rootzone_forecast` | root-zone forecast state |
| 2 | `soil_temp_layer1_forecast` | soil temperature proxy |
| 3 | `surface_temp_forecast` | surface temperature proxy |
| 4 | `mwrtm_vegopacity` | vegetation optical-depth proxy |
| 5 | `tb_h_obs` | H-polarized observed brightness temperature |
| 6 | `tb_v_obs` | V-polarized observed brightness temperature |
| 7 | `tb_h_obs_errstd` | H-polarized observation error scale |
| 8 | `tb_v_obs_errstd` | V-polarized observation error scale |
| 9 | `tb_h_obs_assim` | H-polarized assimilated/simulated TB proxy |
| 10 | `tb_v_obs_assim` | V-polarized assimilated/simulated TB proxy |
| 11 | `base_valid_mask` | diagnostic coverage only |

Channel 11 may be summarized as bounded diagnostic coverage. It must not be
used as a hard gate for observations, loss, metrics, or regions.

## DA Increment And EnKF Anchor

HydroDA predicts the analysis increment:

```text
analysis_increment = analysis_soil_moisture - forecast_soil_moisture
pred_analysis      = forecast_soil_moisture + pred_increment
```

The data-assimilation update anchor is the standard Kalman/EnKF form:

```text
Delta x_t = K_t (y_t - H_t x_t^-)
x_t^+     = x_t^- + Delta x_t
K_t       = P_t H_t^T (H_t P_t H_t^T + R_t)^-1
```

Project mapping:

- `analysis_increment` corresponds to `x_t^+ - x_t^-`;
- `TB_obs - TB_assim` is an input-side observation-space innovation proxy;
- source-side covariance summaries can estimate how observed innovations map
  to source increments, but target-side labels cannot be used for that bank.

Source basis: the SMAP L4 soil moisture ATBD describes an EnKF assimilation
system for merging land-model estimates and SMAP brightness-temperature
information. See the SMAP L4 ATBD PDF:
https://smap.jpl.nasa.gov/files/smap2/L4_SM_InitRel_v1.pdf

## Brightness-Temperature Innovation Sign

For polarization `p in {H, V}`:

```text
d_H = (tb_h_obs - tb_h_obs_assim) / (tb_h_obs_errstd + eps)
d_V = (tb_v_obs - tb_v_obs_assim) / (tb_v_obs_errstd + eps)
m_p = -tanh(d_p)
```

Sign convention:

- `d_p > 0` means the observation is brighter than the assimilated/simulated TB
  proxy.
- At L-band, higher soil moisture generally lowers emissivity and brightness
  temperature, all else equal. Therefore `d_p > 0` is a dry-direction signal.
- `m_p = -tanh(d_p)` is a wet-support score: `m_p > 0` supports wetter
  increments, while `m_p < 0` supports drier increments.

Default confidence terms:

```text
rho_H = 1 / (1 + tb_h_obs_errstd^2)
rho_V = 1 / (1 + tb_v_obs_errstd^2)
```

The confidence terms are bounded summaries, not target-label-calibrated
weights.

## Tau-Omega / RTM Anchor

The L-band tau-omega radiative-transfer model motivates vegetation attenuation,
polarization mismatch, and weak-observation confidence diagnostics:

```text
gamma = exp(-tau * sec(theta))
TB_p approx T_s e_p gamma
          + T_c (1 - omega) (1 - gamma) (1 + r_p gamma)
```

HydroDA diagnostics:

```text
vegetation_attenuation_risk = 1 - gamma
polarization_mismatch_risk  = bounded((TB_V_obs - TB_H_obs)/(TB_V_obs + TB_H_obs + eps)
                                      - same_assim)
weak_observation_confidence_risk = bounded_low_confidence(rho_H, rho_V)
```

For the current input set, `tau` is proxied by `mwrtm_vegopacity`; `theta` is
held fixed to the implementation convention used by the existing physical
diagnostics, unless a future protocol explicitly changes it before freezing a
method.

Source basis: the SMAP L2/L3 passive soil moisture ATBD documents the
tau-omega radiative-transfer basis for passive microwave soil moisture
retrieval. See:
https://nsidc.org/sites/default/files/l2_sm_p_atbd_rev_g_final_oct2021_0.pdf

## Surface To Root-Zone Coupling Anchor

The exponential-filter soil-water-index recursion supports a memory/coupling
interpretation from surface soil moisture to deeper states:

```text
SWI_n = SWI_{n-1} + K_n (SM_surface(t_n) - SWI_{n-1})
```

HydroDA uses only current input-side states and source-side metadata:

```text
B_vert = tanh((sm_surface_forecast - sm_rootzone_forecast) / s_vert)
C_rz   = Cov_source(DeltaSM_rootzone, DeltaSM_surface)
         / (Var_source(DeltaSM_surface) + lambda)
```

`B_vert` is an instantaneous vertical contrast proxy. `C_rz` is a source-fit
gain-prior summary. Neither may be fit from `target_context`, `target_val`,
`target_eval`, or legacy `target_full_train` labels.

Source basis: Albergel et al. describe the exponential-filter approach for
near-surface to root-zone soil moisture transfer:
https://hess.copernicus.org/articles/12/1323/2008/hess-12-1323-2008.pdf

## Source-Fit Gain Priors

Physics-gain summaries are allowed only from `source_fit` labels:

```text
G_v,p = Cov_source(DeltaSM_v, d_p) / (Var_source(d_p) + lambda)
G0_v,b = Cov_source(DeltaSM_v, B_b) / (Var_source(B_b) + lambda)
C_rz = Cov_source(DeltaSM_rootzone, DeltaSM_surface)
       / (Var_source(DeltaSM_surface) + lambda)
```

Allowed roles:

```text
bank construction: source_fit only
checkpoint/hyperparameter selection: source_val only
target_eval: final offline evaluation only, no selection
```

Forbidden roles for bank construction or eta/threshold selection:

```text
target_context
target_support
target_val
target_eval
target_full_train
```

## M3_14 Physics Feature Set

The M3_14 design uses a raw input-side physical token `z_phys`. The minimum
schema is:

```text
d_H, d_V
m_H, m_V
gamma
rho_H, rho_V
B_pol
B_temp
B_vert
source_fit_gain_prior_summaries
finite_input_coverage
base_valid_mask_fraction_diagnostic_only
```

Suggested basis maps:

```text
B_H    = gamma * rho_H * m_H
B_V    = gamma * rho_V * m_V
B_pol  = bounded(observed_polarization_contrast - assimilated_polarization_contrast)
B_temp = tanh(abs(soil_temp_layer1_forecast - surface_temp_forecast) / s_temp)
B_vert = tanh((sm_surface_forecast - sm_rootzone_forecast) / s_vert)
```

These features can condition coefficient logits and weak source-fit
regularization. They cannot create a direct final-output increment residual.

## Physics-AI Design Rule

Physics enters HyperDA-TRUST as:

- input-side features;
- coefficient-space structure;
- weak source-fit regularization;
- source-side metadata;
- diagnostics and optional shrink-only guards.

Physics must not enter as a free target-tuned output residual. The default
operator injection is bounded coefficient-logit modulation:

```text
logits_l = logits_l_M3design
           + sigmoid(g_l) * 0.05 * DeltaLogits_l(z_prompt, z_phys)
```

Source basis: Willard et al. survey physics-guided and hybrid physics-ML
method classes, including ways to inject scientific knowledge into losses,
features, and architectures:
https://arxiv.org/abs/2003.04919

## Implemented M3_14 Diagnostics

Current code/tests for M3_14 verify:

- `d_p > 0` is recorded as dry-direction support;
- `m_p = -tanh(d_p)` gives wet support when `d_p < 0`;
- channel 11 is diagnostic coverage only;
- physics features read only `x_raw` or raw `x`, `month`, `region_mask`, and
  source-side metadata;
- source-fit gain banks reject target-side records;
- coefficient modulation is bounded and does not bypass the HyperDA-TRUST
  coefficient path;
- diagnostic guards can only shrink variable gates and cannot amplify a final
  residual.

Implemented source name:

```text
raw_input_side_formula_gain
```

Implemented schema:

```text
m3_14_raw_input_side_formula_gain_v1
```
