# Baseline Expansion Literature Plan

This note organizes candidate literature baselines for HydroDA-OOD /
HyperDA-TRUST. It is a paper-planning artifact, not an implementation spec.

Status: literature metadata checked on 2026-06-23. Any method listed here must
still be promoted through the active protocol/spec review before it appears in a
paper-main result table.

## Protocol Guardrails

HydroDA-OOD is neural land DA analysis-increment emulation, not direct soil
moisture prediction. Every runnable baseline in this document must predict
analysis increments:

```text
analysis_increment = analysis_soil_moisture - forecast_soil_moisture
pred_analysis      = forecast_soil_moisture + pred_increment
```

V4.4 target-side information rules:

- K=0 uses no target labels.
- K=4 and K=12 use only the K labeled `target_support` DA cycles.
- `target_context` may be used only as 2015-2021 input-side target context.
- `target_val` is unused in the main protocol.
- `target_eval` 2023-2025 labels are evaluation-only.
- No target evaluation data may be used for training, prompt construction,
  adaptation sample selection, normalization, early stopping, model selection,
  threshold calibration, hyperparameter tuning, or region definition.
- Source-side model selection and method promotion use source validation or a
  preregistered source-side episodic policy, never target validation/evaluation.

Retired or internal methods remain excluded from the paper-main baseline ladder:
ridge calibration, BORA-style residual adapters, APO, full target training,
target-region supervised experts, target-val selection, target-support means,
target-monthly means, and other target-history oracles.

## Recommended DG Main-Table Baselines

These are candidate additions if the paper expands beyond the currently frozen
core ladder. The main baseline ladder should prioritize source-only DG and
multi-source DG methods. Target-context TTA/alignment methods are not the main
paper family.

| Title | Venue / year | Link | Category | HydroDA-OOD mapping |
| --- | --- | --- | --- | --- |
| SWAD: Domain Generalization by Seeking Flat Minima | NeurIPS 2021 | <https://openreview.net/forum?id=zkHlu_3sJYU> | Paper-main runnable, source-only DG | Train the increment model on labeled source_fit domains only; use source_val for checkpoint/model selection; evaluate once on target_eval. Uses no target_context, target_support, target_val, or target_eval labels. |
| Domain Generalization with MixStyle | ICLR 2021 | <https://openreview.net/forum?id=6xHJ37MVxxp> | Paper-main runnable, source-only DG | Add feature-statistics mixing across source domains during increment-model training. Selection remains source_val only. Uses no target-side labels or target_eval adaptation. |
| Domain-Inspired Sharpness-Aware Minimization Under Domain Shifts | ICLR 2024 | <https://openreview.net/forum?id=I4wB3HA3dJ> | Future source-only DG candidate | Recent sharpness-aware DG method. If implemented, train only on source_fit labels and select by source_val. |
| Unknown Domain Inconsistency Minimization for Domain Generalization | ICLR 2024 | <https://openreview.net/forum?id=eNoiRal5xi> | Paper-main runnable, source-only DG | Implemented as `udim_unknown_domain_inconsistency_minimization`: a SAM-style source-only unknown-domain perturbation with region-loss inconsistency regularization. Selection remains source_val only; no target_context or target labels are used. |
| Moment Alignment: Unifying Gradient and Hessian Matching for Domain Generalization | UAI 2025 | <https://openreview.net/forum?id=EzwlQDs5Ck> | Future source-only invariance candidate | Modern gradient/Hessian moment matching view of DG. Higher implementation cost than SWAD/MixStyle. |
| LoRA: Low-Rank Adaptation of Large Language Models | ICLR 2022 | <https://openreview.net/forum?id=nZeVKeeFYf9> | K-shot PEFT candidate | Use low-rank trainable updates on a frozen source-trained increment backbone. K=4/12 may use only K labeled target_support cycles; K=0 is not allowed to tune LoRA parameters. Steps, rank, learning rate, and selection must be source-side preregistered. |
| Parameter-Efficient Transfer Learning for NLP | ICML 2019 | <https://proceedings.mlr.press/v97/houlsby19a.html> | K-shot adapter candidate | Use adapter modules as a lightweight target adaptation baseline against SAFE diagnostics / future few-shot extension. K=4/12 use only support cycles; all backbone/source-stage selection remains source-side. |
| BitFit: Simple Parameter-efficient Fine-tuning | ACL 2022 | <https://aclanthology.org/2022.acl-short.1/> | K-shot minimal PEFT candidate | Bias-only K-shot update on the source-trained increment model. This is the lowest-capacity PEFT control; K=4/12 use only target_support labels and no target_val/eval signal. |

### Main-Table Selection Recommendation

If the baseline ladder is expanded, add only a compact representative set first:

- SWAD.
- MixStyle.
- One recent source-only DG method beyond SWAD/MixStyle, selected before
  target_eval: Domain-Inspired SAM, Unknown Domain Inconsistency Minimization,
  or Moment Alignment.
- One best K-shot PEFT baseline selected by a source-side preregistered rule:
  Adapter, LoRA, or BitFit.

Important protocol note: the current machine-readable baseline spec still treats
Adapter/LoRA K-shot work as appendix or follow-up until explicitly promoted.
This document records the literature case for promotion; it does not override
`specs/baselines.yaml` or `specs/hyperda_v4.yaml`.

## Appendix / Diagnostic Baselines

These methods are scientifically useful but should not be first-line paper-main
baselines unless the implementation and protocol burden is justified.

| Title | Venue / year | Link | Category | HydroDA-OOD mapping |
| --- | --- | --- | --- | --- |
| Deep CORAL: Correlation Alignment for Deep Domain Adaptation | ECCV Workshops 2016 | <https://arxiv.org/abs/1607.01719> | Internal diagnostic only, old unsupervised alignment baseline | Too old for the refreshed main-table defense. If retained in code, it must be marked `internal_diagnostic_old_baseline_not_paper_main`, never a default wrapper method, and never allowed by the paper-main registry. |
| Test-time Adaptation for Regression by Subspace Alignment | ICLR 2025 | <https://openreview.net/forum?id=SXtl7NRyE5> | Runnable diagnostic, not DG paper-main | Strong regression TTA method, but it is still target-context adaptation rather than source-only DG. Do not place in the DG main table unless the paper explicitly adds a TTA appendix table. |
| Test-time Correlation Alignment | ICML 2025 | <https://openreview.net/forum?id=0dualJz9OI> | Runnable diagnostic, not DG paper-main | A target-context-only LinearTCA variant can align source and target_context feature correlations without target labels. Do not adapt on target_eval batches; run target_eval only after the target_context-fitted transform is frozen. |
| Self-Bootstrapping for Versatile Test-Time Adaptation | ICML 2025 | <https://openreview.net/forum?id=Li4rieeClO> | Runnable diagnostic, not DG paper-main | Relevant as a TTA diagnostic, but not part of the main DG baseline ladder. |
| Leveraging Vision-Language Models for Improving Domain Generalization in Image Classification | CVPR 2024 | <https://openaccess.thecvf.com/content/CVPR2024/html/Addepalli_Leveraging_Vision-Language_Models_for_Improving_Domain_Generalization_in_Image_Classification_CVPR_2024_paper.html> | Related work only for current backbone | Depends on CLIP/VLM image-classification distillation. It is not a fair current SmallResUNet DA-increment baseline unless HydroDA adopts a comparable vision/weather foundation backbone. |
| A Simple Framework for Learning from Multiple Experts in Domain Generalization | NeurIPS 2024 | <https://openreview.net/forum?id=SYjxhKcXoN> | Future source-expert diagnostic | Multi-expert classification regularization is relevant to source-regime specialist routing, but a HydroDA regression version would be a separate expert-distillation diagnostic rather than the current low-cost DG ladder. |
| Fishr: Invariant Gradient Variances for OOD Generalization | ICML 2022 | <https://arxiv.org/abs/2109.02934> | Appendix diagnostic, source-domain invariance | Train on labeled source_fit domains and regularize cross-domain gradient variance. No target data is needed. Heavier implementation than SWAD/MixStyle, so best as a scientific comparison or later appendix baseline. |
| Tent: Fully Test-Time Adaptation by Entropy Minimization | ICLR 2021 | <https://openreview.net/forum?id=uXl3bZLkr3c> | Appendix diagnostic, classification-oriented TTA | Include mainly to explain why entropy TTA is a poor default for increment regression. If implemented, it cannot use target_eval for adaptation under the V4.4 main protocol; any target_context-only variant must be clearly marked diagnostic. |
| CoTTA: Continual Test-Time Domain Adaptation | CVPR 2022 | <https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html> | Appendix diagnostic, continual TTA | Classification/segmentation-oriented and source-free. A HydroDA-OOD variant would need strict target_context-only adaptation; continual updates over target_eval would violate the main evaluation contract. |
| Test-Time Training with Self-Supervision | ICML 2020 | <https://proceedings.mlr.press/v119/sun20b.html> | Appendix diagnostic, auxiliary self-supervision | Possible only if HydroDA-OOD adds an auxiliary self-supervised or reconstruction loss using input-side data. Target_eval labels remain forbidden, and target_eval input-driven updates would need to be excluded from main V4.4. |
| MAML: Model-Agnostic Meta-Learning | ICML 2017 | <https://proceedings.mlr.press/v70/finn17a.html> | Appendix few-shot meta-learning | Train meta-episodes over source regions only, then adapt on K target_support cycles for K=4/12. No target_val/eval selection. Useful if source-region episodic training infrastructure is added. |
| Visual Prompt Tuning | ECCV 2022 | <https://arxiv.org/abs/2203.12119> | Appendix PEFT/prompting diagnostic | Useful only if a Transformer or ViT-style HydroDA backbone is introduced. For the current convolutional/adapter HyperDA path, it is related PEFT context rather than a runnable fair baseline. |

## Weather / Climate Related Work

These models should be cited for positioning the weather/climate foundation-model
context. They should not be implemented as fair HydroDA-OOD baselines unless the
project later adopts a comparable weather foundation backbone and an increment
emulation head under the same V4.4 leakage controls.

| Title | Venue / year | Link | Category | HydroDA-OOD mapping |
| --- | --- | --- | --- | --- |
| ClimaX: A Foundation Model for Weather and Climate | ICML 2023 | <https://openreview.net/forum?id=TowCaiz7Ui> | Related-work-only weather foundation model | Useful for framing heterogeneous-variable weather/climate foundation models. Not a fair baseline for DA increment emulation without a comparable backbone, inputs, target-context policy, and source-only selection rule. |
| FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators | arXiv 2022 / weather ML line | <https://arxiv.org/abs/2202.11214> | Related-work-only global weather model | Forecasting model for atmospheric variables, not land DA increment emulation. Cite for ML weather forecasting context, not as a runnable baseline. |
| GraphCast: Learning Skillful Medium-Range Global Weather Forecasting | Science 2023 | <https://www.science.org/doi/10.1126/science.adi2336> | Related-work-only global weather model | Medium-range global weather forecasting. It does not predict land DA increments and cannot be compared fairly without an increment-specific adaptation layer and V4.4-safe training/evaluation split. |
| Pangu-Weather: 3D High-Resolution Global Weather Forecasting | Nature 2023 | <https://www.nature.com/articles/s41586-023-06185-3> | Related-work-only global weather model | Global atmospheric forecasting model. Use for positioning, not as a direct baseline for soil-moisture analysis increments. |
| GenCast: Probabilistic Weather Forecasting with Machine Learning | Nature 2024 | <https://www.nature.com/articles/s41586-024-08252-9> | Related-work-only probabilistic weather model | Probabilistic medium-range weather forecasting. Relevant to ML weather progress, but not directly comparable to target-region DA increment emulation. |
| NeuralGCM: Neural General Circulation Models for Weather and Climate | Nature 2024 | <https://www.nature.com/articles/s41586-024-07744-y> | Related-work-only hybrid weather/climate model | Hybrid physics/ML GCM context. Not a runnable HydroDA-OOD baseline unless recast to output DA increments under the same land variables, masks, and split contract. |
| Aurora: A Foundation Model for the Earth System | Nature 2025 | <https://www.nature.com/articles/s41586-025-09005-y> | Related-work-only Earth-system foundation model | Cite for broader Earth-system foundation modeling and task adaptation. Fair comparison would require an Aurora-based increment emulator with no target_val/eval leakage. |
| VAE-Var: Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology | ICLR 2025 | <https://openreview.net/forum?id=utz99dx2RN> | DA-specific future/diagnostic baseline | Highly relevant to reviewer framing because it is a 2025 neural DA method. It estimates analysis states through a learned variational objective; a HydroDA version would require a new observation/prior-cost implementation, not just the current ResUNet increment trainer. |
| FuXi-DA: a generalized deep learning data assimilation framework for assimilating satellite observations | npj Climate and Atmospheric Science 2025 | <https://www.nature.com/articles/s41612-025-01039-3> | DA-specific related/future baseline | Directly relevant because it maps background fields plus satellite observations to analyses and reports analysis increments. Current HydroDA-OOD can cite it as modern neural DA context; a fair runnable baseline would require adapting the fusion architecture to SMAP land variables and V4.4 splits. |
| A data-to-forecast machine learning system for global weather | Nature Communications 2025 | <https://www.nature.com/articles/s41467-025-62024-1> | End-to-end weather DA/forecast related work | FuXi Weather demonstrates cycling ML-based DA plus forecasting. It is too large and atmospheric for the current baseline ladder, but useful for showing why HydroDA-OOD focuses on leakage-controlled DA increment emulation rather than full operational replacement. |
| WeatherPEFT: Task-Adaptive Parameter-Efficient Fine-Tuning for Weather Foundation Models | ICLR 2026 Poster | <https://openreview.net/forum?id=eFExhM3tKr> | Related-work-only weather PEFT / possible future baseline | Closest weather-specific PEFT reference. It should remain related work unless HydroDA-OOD adopts a weather foundation model backbone and applies its adaptation only through target_context or K target_support cycles under V4.4. |
| UniCA: Unified Covariate Adaptation for Time Series Foundation Model | ICLR 2026 Poster | <https://openreview.net/forum?id=I8q4MZb4OP> | Future TSFM/covariate-adaptation baseline | Relevant if HydroDA-OOD adds a time-series foundation-model backbone or a covariate-aware temporal encoder. Not a fair current ResUNet baseline because it assumes TSFM adaptation rather than gridded DA increment emulation. |
| When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency | ICLR 2026 Poster | <https://openreview.net/forum?id=05PqjBzN6S> | Diagnostic drift/adaptation gate | Useful as a source-side/target_context-only diagnostic for deciding whether target_context is informative enough to justify adaptation. It is not itself an increment predictor. |
| PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting | ICLR 2026 Poster | <https://openreview.net/forum?id=Lk9SqMQzhX> | Future time-series forecasting backbone/control | Modern efficient nonstationary time-series baseline. It should remain future/related unless the HydroDA input representation is recast as temporal sequences rather than dense gridded DA cycles. |
| Generalized Spherical Neural Operators: Green's Function Formulation | ICLR 2026 Poster | <https://openreview.net/forum?id=XkGjzSDTnm> | Future neural-operator/weather backbone | Relevant to global weather neural operators and spherical geometry. It is not a current US/CN/AU regional land-grid baseline, but it can support future cross-continent/global-grid discussion. |

### Expanded Candidate Prioritization

Recommended order after the current SWAD/MixStyle DG set:

1. The current recent source-only DG ladder now includes Domain-Inspired SAM,
   UDIM, Moment Alignment, and IU; add QT-DoG only if reviewer pressure
   justifies another flat-minima baseline.
2. Keep `ssa_reg_target_context_subspace_alignment`,
   `tca_target_context_correlation_alignment`, and
   `self_bootstrap_target_context_consistency_tta` as target-context-only
   diagnostics, not DG main-table methods.
3. Keep `vae_var_neural_da` and `fuxi_da_style_fusion` as DA-specific future
   baselines; cite them in related work now, implement only if the paper needs
   a neural-DA-method comparison rather than a DG comparison.
4. Keep WeatherPEFT, UniCA, PhaseFormer, and GSNO as 2026/future-method
   candidates; do not run them under the current SmallResUNet protocol.

## Verification Checklist

Before promoting any baseline from this plan into a paper table or run ladder:

- Confirm the entry has title, venue/year, link, category, and HydroDA-OOD
  mapping.
- Confirm every runnable baseline predicts DA increments, not direct soil
  moisture.
- Confirm K=0 uses no target labels.
- Confirm K=4/12 uses only K labeled target_support cycles.
- Confirm unlabeled target_context use is input-side only and limited to the
  2015-2021 context period.
- Confirm target_val and target_eval are not used for training, prompt
  construction, adaptation, model selection, hyperparameter tuning,
  normalization, threshold calibration, or sample selection.
- Confirm source-side selection or preregistered source-side episodic policy is
  recorded for each promoted baseline.
- Confirm retired/internal methods such as ridge, BORA, APO, full target
  training, target-region supervised experts, and target-val selection remain
  excluded from the paper-main baseline ladder.
