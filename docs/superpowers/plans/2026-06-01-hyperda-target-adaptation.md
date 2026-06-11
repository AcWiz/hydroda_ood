# HyperDA Target Adaptation Implementation Plan

> Superseded by Protocol V4.4 zero/few-shot generalization. This plan is
> retained for legacy/internal full-target reproduction context only. The active
> paper-facing HyperDA path is `run/phase5_hyperda_zero_few_shot.sh`, where
> target labels are limited to `target_support` with K in {0,4,12}, target_val is
> unused in the main protocol, and final evaluation is `target_eval=2023-2025`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Historical goal:** Implement the executable foundation for HyperDA target
historical adaptation with frozen source-trained hypernetwork/operator priors
and trainable target-specific latent, adapter coefficient residuals, and monthly
residual gain. In V4.4 this is legacy/internal reproduction context; the active
main path is zero/few-shot K=0/4/12.

**Architecture:** Extend the current HyperDA ResUNet with small target-adaptation
modules instead of full-backbone fine-tuning. For this legacy path, keep all
temporal protocol metadata explicit in run scripts and tests: target 2015-2021
trains adaptation, target 2022 selects adaptation, target 2023-2025 is final
evaluation only.

**Tech Stack:** Python, PyTorch, pytest, existing HydroDA model/training/protocol modules.

---

## File Structure

- Create `hydroda/models/target_adaptation.py`: reusable modules for target latent prompt shifts, trainable adapter coefficient residuals, monthly residual gain, and freeze/trainable parameter helpers.
- Modify `hydroda/models/hyper_adapters.py`: allow coefficient logit residuals without changing default behavior.
- Modify `hydroda/models/hyper_conditional_unet.py`: add optional target adaptation path and helper methods for freezing source priors.
- Modify `run/phase4_hyperda.sh`: rename the run output text from zero-shot-style HyperDA to target historical adaptation and expose adaptation flags.
- Create `run/phase5_hyperda_target_adapt.sh`: preregistered target adaptation entrypoint skeleton using target train/val/eval protocol text.
- Modify `specs/hyperda_v4.yaml`: clarify frozen `H_psi` and trainable target operator variables in the main variant.
- Test `tests/test_hyperda_target_adaptation.py`: unit coverage for target latent, coefficient residuals, freeze helpers, and residual gain.
- Modify `tests/test_phase4_prompt_conditioned_protocol_text.py`: assert run scripts use target historical adaptation wording and no zero-shot main-protocol claim.

### Task 1: Target Adaptation Modules

**Files:**
- Create: `hydroda/models/target_adaptation.py`
- Test: `tests/test_hyperda_target_adaptation.py`

- [ ] **Step 1: Write the failing tests**

```python
import torch

from hydroda.models.target_adaptation import MonthlyResidualGain, TargetLatentPrompt


def test_target_latent_prompt_adds_trainable_shift():
    module = TargetLatentPrompt(prompt_dim=4, latent_dim=2)
    z = torch.zeros(3, 4)
    out = module(z)
    assert out.shape == z.shape
    assert module.latent.requires_grad
    assert module.proj.weight.requires_grad


def test_monthly_residual_gain_is_identity_at_initialization():
    gain = MonthlyResidualGain(out_channels=2, n_months=12)
    y = torch.randn(3, 2, 4, 5)
    months = torch.tensor([1, 6, 12])
    out = gain(y, months)
    assert torch.allclose(out, y)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_hyperda_target_adaptation.py -q`

Expected: FAIL with `ModuleNotFoundError` for `hydroda.models.target_adaptation`.

- [ ] **Step 3: Implement minimal modules**

Add `TargetLatentPrompt` and `MonthlyResidualGain` with identity initialization and shape validation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_hyperda_target_adaptation.py -q`

Expected: PASS.

### Task 2: Adapter Coefficient Residuals

**Files:**
- Modify: `hydroda/models/hyper_adapters.py`
- Modify: `hydroda/models/target_adaptation.py`
- Test: `tests/test_hyperda_target_adaptation.py`

- [ ] **Step 1: Write the failing test**

```python
import torch

from hydroda.models.hyper_adapters import BasisHyperAdapter
from hydroda.models.target_adaptation import AdapterCoefficientResidual


def test_adapter_coefficient_residual_changes_coefficients():
    adapter = BasisHyperAdapter(channels=4, prompt_dim=6, n_basis=3, adapter_bottleneck=2)
    residual = AdapterCoefficientResidual(n_basis=3)
    z = torch.randn(2, 6)
    base = adapter.coefficients(z)
    residual.logit_delta.data[0] = 2.0
    shifted = adapter.coefficients(z, logit_residual=residual())
    assert shifted.shape == base.shape
    assert not torch.allclose(shifted, base)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_hyperda_target_adaptation.py::test_adapter_coefficient_residual_changes_coefficients -q`

Expected: FAIL because `AdapterCoefficientResidual` or `logit_residual` is missing.

- [ ] **Step 3: Implement coefficient residual support**

Add `AdapterCoefficientResidual` and update `BasisHyperAdapter.coefficients(z, logit_residual=None)` / `forward(..., logit_residual=None)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_hyperda_target_adaptation.py::test_adapter_coefficient_residual_changes_coefficients -q`

Expected: PASS.

### Task 3: HyperDA Model Target Adaptation Path

**Files:**
- Modify: `hydroda/models/hyper_conditional_unet.py`
- Test: `tests/test_hyperda_target_adaptation.py`

- [ ] **Step 1: Write failing tests**

```python
import torch

from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet


def test_hyperda_freeze_source_prior_leaves_only_target_adaptation_trainable():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
    )
    model.freeze_source_prior_for_target_adaptation()
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all(
        name.startswith("target_")
        or name.startswith("residual_gain")
        or "coefficient_residual" in name
        for name in trainable
    )


def test_hyperda_target_adaptation_forward_accepts_month():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
    )
    x = torch.randn(2, 12, 16, 16)
    z = torch.randn(2, 8)
    month = torch.tensor([1, 12])
    y = model(x, z, month=month)
    assert y.shape == (2, 2, 16, 16)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_hyperda_target_adaptation.py -q`

Expected: FAIL because constructor arguments and `freeze_source_prior_for_target_adaptation` are missing.

- [ ] **Step 3: Implement model integration**

Add optional target adaptation modules, pass coefficient residuals into each adapter, apply target latent before adapter conditioning, apply monthly residual gain at output, and add freeze/trainable helper methods.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_hyperda_target_adaptation.py tests/test_hyperda_model.py -q`

Expected: PASS.

### Task 4: Protocol Text and Script Entrypoints

**Files:**
- Modify: `run/phase4_hyperda.sh`
- Create: `run/phase5_hyperda_target_adapt.sh`
- Modify: `specs/hyperda_v4.yaml`
- Modify: `tests/test_phase4_prompt_conditioned_protocol_text.py`

- [ ] **Step 1: Write failing protocol tests**

Add assertions that HyperDA scripts contain:

```text
target_train=2015-2021
target_val=2022
target_eval=2023-2025
freeze_hypernetwork=true
trainable=target_latent,adapter_coefficient_residuals,residual_gain
```

and do not describe the main protocol as zero-shot.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_phase4_prompt_conditioned_protocol_text.py -q`

Expected: FAIL because the new target adaptation script and text are missing.

- [ ] **Step 3: Update scripts and spec text**

Add explicit protocol lines and the target adaptation entrypoint skeleton. The skeleton may call the current training script only for source-stage training and must not pretend target adaptation training is implemented if no full dataset runner exists yet.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_phase4_prompt_conditioned_protocol_text.py -q`

Expected: PASS.

### Task 5: Final Verification

**Files:**
- All changed files.

- [ ] **Step 1: Run focused tests**

Run: `PYTHONPATH=. pytest tests/test_hyperda_target_adaptation.py tests/test_hyperda_model.py tests/test_full_target_train_protocol.py tests/test_phase4_prompt_conditioned_protocol_text.py -q`

Expected: PASS.

- [ ] **Step 2: Compile changed Python files**

Run: `python -m py_compile hydroda/models/target_adaptation.py hydroda/models/hyper_adapters.py hydroda/models/hyper_conditional_unet.py`

Expected: exit code 0.

- [ ] **Step 3: Inspect diff**

Run: `git diff --stat && git diff --check`

Expected: no whitespace errors.
