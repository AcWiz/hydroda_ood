# HydroDA Refine Spatial Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight target-domain spatial residual head to Hydro_Adapt so target adaptation can correct high-frequency Surface errors while keeping the source Hydro prior frozen.

**Architecture:** Extend `HyperAdapterConditionalResUNet` with an optional `TargetSpatialResidualHead` that consumes normalized inputs and the current adapted increment prediction, predicts a small residual initialized to zero, and is trained only in Phase 5. The training/inference checkpoint config records whether the refine head is enabled so evaluation can reconstruct the same model.

**Tech Stack:** PyTorch, existing HydroDA model/training scripts, pytest.

---

### Task 1: Add Target Spatial Residual Module

**Files:**
- Modify: `hydroda/models/target_adaptation.py`
- Modify: `hydroda/models/hyper_conditional_unet.py`
- Test: `tests/test_hyperda_target_adaptation.py`

- [x] Write tests for zero initialization, shape preservation, and frozen-source trainability.
- [x] Run `PYTHONPATH=. pytest tests/test_hyperda_target_adaptation.py -q` and verify the new tests fail because the module/config does not exist.
- [x] Implement `TargetSpatialResidualHead` and wire it into `HyperAdapterConditionalResUNet`.
- [x] Run `PYTHONPATH=. pytest tests/test_hyperda_target_adaptation.py -q` and verify it passes.

### Task 2: Carry Refine Config Through Phase 5 Training And Predictor

**Files:**
- Modify: `scripts/train/train_hyperda_target_adapt.py`
- Modify: `hydroda/baselines/prompt_conditioned.py`
- Modify: `run/phase5_hyperda_target_adapt.sh`
- Test: `tests/test_hyperda_target_adapt_runner.py`

- [x] Write tests that Phase 5 checkpoint loading and predictor reconstruction preserve `enable_target_spatial_refine`.
- [x] Run the focused tests and verify they fail before implementation.
- [x] Add CLI/config fields and checkpoint reconstruction support.
- [x] Run the focused tests and verify they pass.

### Task 3: Verify Existing Protocol Tests

**Files:**
- Test: `tests/test_hyperda_target_adapt_runner.py`
- Test: `tests/test_hyperda_target_adaptation.py`
- Test: `tests/test_phase4_prompt_conditioned_protocol_text.py`

- [x] Run `PYTHONPATH=. pytest tests/test_hyperda_target_adapt_runner.py tests/test_hyperda_target_adaptation.py tests/test_phase4_prompt_conditioned_protocol_text.py -q`.
- [x] Run `python -m py_compile scripts/train/train_hyperda_target_adapt.py hydroda/models/hyper_conditional_unet.py hydroda/models/target_adaptation.py hydroda/baselines/prompt_conditioned.py`.
- [x] Run `bash -n run/phase5_hyperda_target_adapt.sh run/phase5_hyperda_target_adapt_inference.sh`.
