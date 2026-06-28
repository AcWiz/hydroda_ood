# Recent Domain Generalization Literature Scouting, 2024-2026

Last updated: 2026-06-24

This note re-centers the HydroDA-OOD literature scan on domain generalization
(DG), not test-time adaptation (TTA). TTA papers remain useful as appendix
diagnostics or protocol contrasts, but the paper's main claim is cross-domain
generalization / domain transfer for neural land DA analysis-increment
emulation.

This document is a scouting artifact, not an implementation spec. Any method
promoted into a runnable baseline must still pass the HydroDA-OOD protocol
review in `specs/baselines.yaml` and the no-leakage checklist.

## HydroDA-OOD Fit Criteria

HydroDA-OOD is neural land DA analysis-increment emulation:

```text
analysis_increment = analysis_soil_moisture - forecast_soil_moisture
pred_analysis      = forecast_soil_moisture + pred_increment
```

For a paper-facing DG baseline:

- training must use source labels only;
- model selection must use source validation or a preregistered source-side
  rule;
- no target labels are allowed for K=0;
- `target_eval` 2023-2025 is final offline evaluation only;
- source-only DG methods should not require target-context optimization.

## DG-First Shortlist

These are the most relevant recent DG papers to consider after the current
source-only DG baseline ladder.

Runnable HydroDA-OOD shortlist:

```text
swad
mixstyle
disam
udim
moment_align
iu
```

| Priority | Paper | Venue / year | Link | Why it matters | HydroDA-OOD status |
| --- | --- | --- | --- | --- | --- |
| High | Continuous Temporal Domain Generalization | NeurIPS 2024 | <https://openreview.net/forum?id=G24fOpC3JE> | Directly targets temporally evolving domains; relevant to 2015-2025 hydroclimatic shift. | Strong related work; future temporal DG baseline if we add temporal-domain state modeling. |
| High | Non-stationary Domain Generalization: Theory and Algorithm | UAI 2024 | <https://openreview.net/forum?id=AMxdbjUvWg> | Treats domains evolving along time or space rather than fixed stationary domains. | Strong conceptual anchor for land-climate shifts; not a drop-in ResUNet baseline. |
| High | Rethinking Multi-domain Generalization with A General Learning Objective | CVPR 2024 | <https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_Multi-domain_Generalization_with_A_General_Learning_Objective_CVPR_2024_paper.html> | Modern multi-source DG objective. | Candidate source-only diagnostic if implementation cost is acceptable. |
| High | Unknown Domain Inconsistency Minimization for Domain Generalization | ICLR 2024 | <https://openreview.net/forum?id=eNoiRal5xi> | Source-domain inconsistency minimization with sharpness-aware training flavor. | Runnable as `udim`; source-only paper-main candidate using the HydroDA regression adaptation. |
| High | Domain-Inspired Sharpness-Aware Minimization Under Domain Shifts | ICLR 2024 | <https://openreview.net/forum?id=I4wB3HA3dJ> | Recent sharpness/DG method; extends the flat-minima story beyond SWAD. | Runnable as `disam`; source-only paper-main candidate. |
| High | Moment Alignment: Unifying Gradient and Hessian Matching for Domain Generalization | UAI 2025 | <https://openreview.net/forum?id=EzwlQDs5Ck> | Modern theory connecting gradient/Hessian matching and invariant learning. | Runnable as `moment_align`; source-only paper-main candidate. |
| Medium | Learning Robust Spectral Dynamics for Temporal Domain Generalization | NeurIPS 2025 | <https://openreview.net/forum?id=efrFbKYobs> | Spectral view of temporal drift; useful for seasonal/multi-year shift framing. | Related work / future temporal baseline. |
| Medium | Continuous Domain Generalization | NeurIPS 2025 | <https://openreview.net/forum?id=KxcysQw6Ma> | Models domains as continuous latent variations rather than discrete IDs. | Future extension pool; not first-round runnable. |
| Medium | QT-DoG: Quantization-Aware Training for Domain Generalization | ICML 2025 | <https://openreview.net/forum?id=OS2ZVeHI4U> | Recent flat-minima/quantization DG line. | Future extension pool; overlaps with SWAD/DISAM. |
| Medium | One-Step Generalization Ratio Guided Optimization for Domain Generalization | ICML 2025 | <https://openreview.net/forum?id=Tv2JDGw920> | Optimization objective aimed at domain generalization. | Candidate only after closer algorithm review. |
| Medium | Seeking Consistent Flat Minima for Better Domain Generalization via Refining Trainable Parameters | CVPR 2025 | <https://openaccess.thecvf.com/content/CVPR2025/html/Li_Seeking_Consistent_Flat_Minima_for_Better_Domain_Generalization_via_Refining_CVPR_2025_paper.html> | Recent flat-minima DG extension. | Candidate source-only appendix baseline; lower priority because SWAD already covers this family. |
| Medium | LFME: A Simple Framework for Learning from Multiple Experts in Domain Generalization | NeurIPS 2024 | <https://openreview.net/forum?id=SYjxhKcXoN> | Multi-expert learning for DG. | Relevant to source-regime specialist bank and source-expert routing. |
| Medium | Weight Diffusion for Future: Learn to Generalize in Non-Stationary Environments | NeurIPS 2024 | <https://openreview.net/forum?id=2cFUYnNL1m> | Learns future-generalizing weights under nonstationarity. | Conceptually adjacent to HyperDA operator generation; likely related work. |

## Broader 2024-2026 DG Papers

### Source-Only / Training-Time DG

| Paper | Venue / year | Link | Notes for HydroDA-OOD |
| --- | --- | --- | --- |
| HYPO: Hyperspherical Out-Of-Distribution Generalization | ICLR 2024 | <https://openreview.net/forum?id=VXak3CZZGC> | Representation geometry; likely classification-oriented but useful related work. |
| Understanding Domain Generalization: A Noise Robustness Perspective | ICLR 2024 | <https://openreview.net/forum?id=I2mIxuXA72> | Useful for discussing label/analysis-noise robustness across source regions. |
| Context is Environment | ICLR 2024 | <https://openreview.net/forum?id=8VPWfqtQMX> | Conceptual in-context DG framing; related work rather than immediate baseline. |
| StyDeSty: Min-Max Stylization and Destylization for Single Domain Generalization | ICML 2024 | <https://proceedings.mlr.press/v235/liu24ad.html> | Single-domain augmentation method; less aligned because HydroDA has multiple source regions. |
| Not Just Pretty Pictures: Toward Interventional Data Augmentation Using Text-to-Image Generators | ICML 2024 | <https://openreview.net/forum?id=b89JtZj9gm> | Generative augmentation for image DG; not suitable for gridded DA tensors. |
| Disentangled Prompt Representation for Domain Generalization | CVPR 2024 | <https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Disentangled_Prompt_Representation_for_Domain_Generalization_CVPR_2024_paper.html> | VFM/CLIP prompt DG; related work unless HydroDA adopts a vision foundation backbone. |
| A2XP: Towards Private Domain Generalization | CVPR 2024 | <https://openaccess.thecvf.com/content/CVPR2024/html/Yu_A2XP_Towards_Private_Domain_Generalization_CVPR_2024_paper.html> | Privacy-DG; not a current HydroDA baseline fit. |
| Efficiently Assemble Normalization Layers and Regularization for Federated Domain Generalization | CVPR 2024 | <https://openaccess.thecvf.com/content/CVPR2024/html/Le_Efficiently_Assemble_Normalization_Layers_and_Regularization_for_Federated_Domain_Generalization_CVPR_2024_paper.html> | Federated DG; source-region analogy possible, but federated assumptions add unnecessary burden. |
| Towards Combating Frequency Simplicity-biased Learning for Domain Generalization | NeurIPS 2024 | <https://openreview.net/forum?id=VMiLdBkCJM> | Frequency shortcut mitigation; possible diagnostic for spatial-frequency shortcuts. |
| Partial Transportability for Domain Generalization | NeurIPS 2024 | <https://openreview.net/forum?id=2V5LTfhcfd> | Causal DG theory; useful for paper discussion, less for implementation. |
| Cross-modal Representation Flattening for Multi-modal Domain Generalization | NeurIPS 2024 | <https://openreview.net/forum?id=UixTytSVOl> | Multimodal DG; current HydroDA input is not framed as cross-modal. |
| CLIPCEIL: Domain Generalization through CLIP via Channel Refinement and Image-text Alignment | NeurIPS 2024 | <https://openreview.net/forum?id=MqeCU0tXAY> | CLIP-specific DG; related work only for current ResUNet/HyperDA. |
| Leveraging Vision-Language Models for Improving Domain Generalization in Image Classification | CVPR 2024 | <https://openaccess.thecvf.com/content/CVPR2024/html/Addepalli_Leveraging_Vision-Language_Models_for_Improving_Domain_Generalization_in_Image_Classification_CVPR_2024_paper.html> | VLM/CLIP distillation for RGB image classification; not a fair current HydroDA ResUNet increment-regression baseline. |
| Is Large-scale Pretraining the Secret to Good Domain Generalization? | ICLR 2025 | <https://openreview.net/forum?id=wCOJpXm0Me> | Important caution when comparing DG algorithms with pretrained foundation models. |
| Decoupled Finetuning for Domain Generalizable Semantic Segmentation | ICLR 2025 | <https://openreview.net/forum?id=qZEdmyqCHF> | Dense prediction DG; relevant but segmentation/pretraining-specific. |
| Regularizing Energy among Training Samples for Out-of-Distribution Generalization | ICLR 2025 | <https://openreview.net/forum?id=Lbx9zdURxe> | Classification energy regularization; probably not direct for regression. |
| Out-of-distribution Generalization for Total Variation based Invariant Risk Minimization | ICLR 2025 | <https://openreview.net/forum?id=c4wEKJOjY3> | Modern invariant-risk training objective; possible source-only diagnostic. |
| Federated Domain Generalization with Data-free On-server Matching Gradient | ICLR 2025 | <https://openreview.net/forum?id=8TERgu1Lb2> | Federated DG; too far from centralized source training. |
| Set Valued Predictions For Robust Domain Generalization | ICML 2025 | <https://openreview.net/forum?id=QxZfMpsFn3> | Robust set prediction; useful for uncertainty discussion, not point-RMSE baseline. |
| OOD-Chameleon: Is Algorithm Selection for OOD Generalization Learnable? | ICML 2025 | <https://openreview.net/forum?id=0rDn6BDNiF> | Meta-selection of OOD algorithms; relevant to baseline-selection discussion. |
| LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation | ICML 2025 | <https://openreview.net/forum?id=LB5F02kwAv> | Dense medical segmentation DG; conceptually useful but not DA tensor-safe. |
| Better to Teach than to Give: Domain Generalized Semantic Segmentation via Diffusion as Hierarchical Visual Catalyst | ICML 2025 | <https://openreview.net/forum?id=jvP1wbD0xh> | Dense prediction DG with diffusion teacher; too heavy for current baseline ladder. |
| Enhancing Foundation Models with Federated Domain Knowledge Alignment | ICML 2025 | <https://openreview.net/forum?id=6SIVFmjIm4> | Federated/foundation-model DG; related to future adapter/foundation work. |
| Domain Generalization in CLIP via Learning with Diverse Text Prompts | CVPR 2025 | <https://openaccess.thecvf.com/content/CVPR2025/html/Wen_Domain_Generalization_in_CLIP_via_Learning_with_Diverse_Text_Prompts_CVPR_2025_paper.html> | CLIP prompt DG; not a fair current ResUNet baseline. |
| Balanced Direction from Multifarious Choices: Arithmetic Meta-Learning for Domain Generalization | CVPR 2025 | <https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Balanced_Direction_from_Multifarious_Choices_Arithmetic_Meta-Learning_for_Domain_Generalization_CVPR_2025_paper.html> | Meta-learning DG; would require source-episode infrastructure. |
| Adversarial Domain Prompt Tuning and Generation for Single Domain Generalization | CVPR 2025 | <https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Adversarial_Domain_Prompt_Tuning_and_Generation_for_Single_Domain_Generalization_CVPR_2025_paper.html> | Single-domain prompt generation; less aligned than multi-source DG. |
| Gradient-Guided Annealing for Domain Generalization | CVPR 2025 | <https://openaccess.thecvf.com/content/CVPR2025/html/Ballas_Gradient-Guided_Annealing_for_Domain_Generalization_CVPR_2025_paper.html> | Source training schedule method; possible source-only diagnostic. |
| PEER Pressure: Model-to-Model Regularization for Single Source Domain Generalization | CVPR 2025 | <https://openaccess.thecvf.com/content/CVPR2025/html/Cho_PEER_Pressure_Model-to-Model_Regularization_for_Single_Source_Domain_Generalization_CVPR_2025_paper.html> | Single-source model regularization; less aligned than multi-source regions. |
| TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction | CVPR 2025 | <https://openaccess.thecvf.com/content/CVPR2025/html/Agarwal_TIDE_Training_Locally_Interpretable_Domain_Generalization_Models_Enables_Test-time_Correction_CVPR_2025_paper.html> | Vision-specific DG with correction; useful as related work, not direct baseline. |
| Unlearning during Training: Domain-Specific Gradient Ascent for Out-of-Distribution Generalization | ICLR 2026 | <https://openreview.net/forum?id=9ufS5Jl0O0> | Domain-specific feature unlearning; runnable as `iu`, a source-only paper-main candidate. |
| Exploring Mode Connectivity in Krylov Subspace for Domain Generalization | ICLR 2026 | <https://openreview.net/forum?id=fpH2GYXJwD> | Loss-landscape DG; overlaps with SWAD/flat-minima story. |
| Robustness to In-Domain Noise and Out-of-Domain Generalization | ICLR 2026 | <https://openreview.net/forum?id=wb83wO41QT> | Noise-aware DG; useful if source analysis increments contain region-dependent noise. |

### Temporal / Continuous / Dynamic DG

| Paper | Venue / year | Link | Notes for HydroDA-OOD |
| --- | --- | --- | --- |
| Continuous Temporal Domain Generalization | NeurIPS 2024 | <https://proceedings.neurips.cc/paper_files/paper/2024/hash/e6f32e64b9c27d153b46c94f0fe22b56-Abstract-Conference.html> | Strong 2024 anchor for evolving time distributions. |
| Continuous Domain Generalization | NeurIPS 2025 | <https://openreview.net/forum?id=KxcysQw6Ma> | Matches continuous hydroclimatic gradients better than discrete domain IDs. |
| Learning Robust Spectral Dynamics for Temporal Domain Generalization | NeurIPS 2025 | <https://openreview.net/forum?id=efrFbKYobs> | Drift/periodicity angle is relevant to seasonal and multi-year shifts. |
| Adaptive Mixture of Disentangled Experts for Dynamic Graph Out-of-Distribution Generalization | ICLR 2026 | <https://openreview.net/forum?id=q0O5LO7X4I> | Dynamic graph OOD; conceptually close to routed experts, but architecture mismatch. |

## Earth Observation / Spatial Generalization Anchors

These are useful for positioning HydroDA-OOD in geoscience/remote-sensing
robustness, even when they are not direct DA-increment baselines.

| Paper / resource | Venue / year | Link | Notes for HydroDA-OOD |
| --- | --- | --- | --- |
| Combining Domain Expertise for Spatial Generalization in Satellite Images | CVPRW EarthVision 2025 | <https://arxiv.org/html/2504.19737v1> | Multi-expert spatial DG for satellite images; relevant to source-regime specialist ideas. |
| Benchmarking Robustness of Earth Observation Foundation Models | NeurIPS 2025 | <https://openreview.net/forum?id=NfeA0waFaE> | EO robustness benchmark; cite for domain-shift importance in Earth observation. |
| Galileo: Learning Global & Local Features of Many Remote Sensing Modalities | ICML 2025 | <https://proceedings.mlr.press/v267/> | Remote-sensing foundation representation; useful future backbone context. |

## TTA Methods: Appendix / Diagnostic Only

TTA papers are not the main literature family for this paper. They may be used
to justify diagnostic baselines under a strict rule: target-context-only updates
must finish before final target evaluation, and `target_eval` may not drive any
adaptation or selection. The current diagnostic methods in this bucket are:

| Method | Venue / year | Link | HydroDA-OOD status |
| --- | --- | --- | --- |
| Test-time Adaptation for Regression by Subspace Alignment | ICLR 2025 | <https://openreview.net/forum?id=SXtl7NRyE5> | Runnable diagnostic as `ssa_reg_target_context_subspace_alignment`; not DG paper-main. |
| Test-time Correlation Alignment | ICML 2025 | <https://openreview.net/forum?id=0dualJz9OI> | Runnable diagnostic as `tca_target_context_correlation_alignment`; not DG paper-main. |
| Self-Bootstrapping for Versatile Test-Time Adaptation | ICML 2025 | <https://openreview.net/forum?id=Li4rieeClO> | Runnable diagnostic as `self_bootstrap_target_context_consistency_tta`; not DG paper-main. |

## Protocol / Evaluation Papers Worth Citing

| Paper | Venue / year | Link | Why cite |
| --- | --- | --- | --- |
| Rethinking the Evaluation Protocol of Domain Generalization | CVPR 2024 | <https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Rethinking_the_Evaluation_Protocol_of_Domain_Generalization_CVPR_2024_paper.html> | Direct support for strict no-test-domain-information evaluation. |
| Is Large-scale Pretraining the Secret to Good Domain Generalization? | ICLR 2025 | <https://openreview.net/forum?id=wCOJpXm0Me> | Useful caution when comparing modern foundation-model baselines against task-specific models. |
| OOD-Chameleon: Is Algorithm Selection for OOD Generalization Learnable? | ICML 2025 | <https://openreview.net/forum?id=0rDn6BDNiF> | Supports reporting algorithm-selection limitations and source-side selection rules. |

## Practical Recommendation

1. Keep the paper-facing DG baseline ladder source-only: `swad`, `mixstyle`,
   `disam`, `udim`, `moment_align`, and `iu` are the current runnable DG methods.
2. Keep QT-DoG, Continuous Temporal DG, and Continuous DG in the future
   extension pool unless a later protocol update promotes them.
3. Use temporal/continuous DG papers as related-work anchors for the actual
   hydroclimatic shift claim; implement them only if we add temporal-domain
   state modeling.
4. Keep SSA-Reg, TCA, and Self-Bootstrapping as explicit diagnostics, not
   paper-main DG baselines.
5. Keep CLIP/VLM prompt DG papers as related work only unless the model
   backbone changes.

## Search Scope

The scan covered official or primary sources where possible: OpenReview, CVF
Open Access, PMLR, NeurIPS proceedings, UAI accepted-paper pages, arXiv, and
official project pages. It intentionally favors 2024, 2025, and early 2026
papers over older classical DG baselines.
