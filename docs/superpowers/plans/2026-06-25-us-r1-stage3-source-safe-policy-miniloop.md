# US-R1 Stage 3 Source-Safe Policy Miniloop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the US-R1 seed0 K12 support-gain diagnostic into a source-side SAFE policy miniloop and evaluate whether the existing paper-safe Stage 3 machinery can produce nonzero K-shot target_eval rows without using target_eval for calibration.

**Architecture:** Reuse the existing `AUTO_GENERATE_SAFE_POLICY` path in `run/phase5_hyperda_zero_few_shot_eval.sh`. The wrapper generates source pseudo-episode rows via `scripts/eval/run_stage3_source_safe_policy_calibration.py`, exports `safe_policy.json` through `scripts/eval/calibrate_source_safe_guard.py`, then runs target_eval K0/K4/K12 with `STAGE3_KSHOT_MODE=paper_safe`.

**Tech Stack:** Bash wrappers, Python calibration scripts, PyTorch checkpoints, JSON/CSV run artifacts, pytest protocol checks.

---

### Task 1: Freeze The Diagnostic Evidence

**Files:**
- Create: `reports/experiments/US_R1_seed0_stage3_support_gain_tta_decision_20260625.md`

- [ ] **Step 1: Write an experiment card from completed artifacts**

Use:

```bash
artifacts/runs/phase5_hyperda_zero_few_shot_eval/US-R1_s0_support_gain_diag_20260624T162851Z/overview.json
artifacts/runs/phase5_hyperda_zero_few_shot_eval/US-R1_s0_tta_prompt_align_diag_20260624T171201Z/overview.json
```

Record:

```text
K0 fixed baseline: surface_WRMSE=0.0027795482 rootzone_WRMSE=0.0002266361
K4 diagnostic: gain_s=1.0 gain_r=1.0 no WRMSE movement
K12 diagnostic: gain_s=0.5 gain_r=0.5 surface_WRMSE=0.0027052343 rootzone_WRMSE=0.0002145653
TTA diagnostic: context_tta_effective=false source_fit_source_val_only_dimension_mismatch_identity_fallback pred_delta=0.0
```

- [ ] **Step 2: Verify the card does not make a paper-facing K-shot claim**

Run:

```bash
rg -n "paper-facing improvement|paper claim|TTA improvement" reports/experiments/US_R1_seed0_stage3_support_gain_tta_decision_20260625.md
```

Expected: no unsupported positive paper-facing claim.

### Task 2: Run Focused Protocol Checks

**Files:**
- Read-only verification of existing tests and scripts.

- [ ] **Step 1: Run SAFE policy unit checks**

Run:

```bash
PYTHONPATH=. pytest tests/test_p2_8_source_safe_guard_calibration.py -q
```

Expected: pass. If unrelated pre-existing failures appear, record exact failures and continue only if the failing tests are not in the SAFE policy path.

- [ ] **Step 2: Run few-shot runner policy parsing checks**

Run:

```bash
PYTHONPATH=. pytest tests/test_hyperda_few_shot_runner.py -q
```

Expected: pass or record pre-existing failures. The critical checks are policy source validation, nonzero K-shot policy enforcement, and metadata propagation.

### Task 3: Generate Minimal Source-Side SAFE Policy

**Files:**
- Create artifacts under `artifacts/runs/stage3_source_safe_policy_cache/`.

- [ ] **Step 1: Run wrapper-driven source-side calibration**

Use a minimal source-side evidence screen, not a paper-main calibration:

```bash
AUTO_GENERATE_SAFE_POLICY=1 \
STAGE3_KSHOT_MODE=paper_safe \
STAGE3_CONTEXT_TTA=none \
K_LIST="0 4 12" \
EVAL_MAX_SAMPLES=0 \
TARGET_CONTEXT_MAX_SAMPLES=0 \
SAFE_POLICY_CANDIDATE_SET=stage3_conservative_v1 \
SAFE_POLICY_CALIBRATION_STAGE=coarse \
SAFE_POLICY_SOURCE_QUERY_MAX_SAMPLES=128 \
SAFE_POLICY_PSEUDO_TARGET_REGIONS="US-R2,US-R3" \
SAFE_POLICY_EVIDENCE_LEVEL=weaker \
SAFE_POLICY_ALLOW_IN_CHECKPOINT_SOURCE_EPISODES=1 \
SAFE_POLICY_KSHOT_UPDATE_REQUIREMENT=nonzero_update \
EVAL_OUTPUT_LEVEL=compact \
bash run/phase5_hyperda_zero_few_shot_eval.sh "" US-R1 0 1 \
  artifacts/runs/phase5_hyperda_zero_few_shot_eval/US-R1_s0_source_safe_miniloop_20260625
```

Expected:
- source checkpoint sha starts with `799d58bd`;
- `safe_policy.json` is generated or reused under the cache root;
- K-shot rows include `policy_source=source_side_episode_calibration`;
- if policy export fails because no nonzero K4/K12 source-side candidate exists, stop and record that as the result.

### Task 4: Audit Target Eval Artifacts

**Files:**
- Read:
  - `artifacts/runs/phase5_hyperda_zero_few_shot_eval/US-R1_s0_source_safe_miniloop_20260625/overview.json`
  - `K*/adapt/metadata.json`
  - `K*/eval/US-R1/summary.json`

- [ ] **Step 1: Parse metrics and safety metadata**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("artifacts/runs/phase5_hyperda_zero_few_shot_eval/US-R1_s0_source_safe_miniloop_20260625")
rows = json.loads((root / "overview.json").read_text())
for row in sorted(rows, key=lambda r: int(r["K"])):
    print(row["K"], row.get("status"), row.get("paper_facing_run"), row.get("policy_source"), row.get("stage3_posterior_decision"), row.get("source_policy_candidate_id"), row.get("safe_policy_json_sha256"), row.get("surface_rmse_latw"), row.get("rootzone_rmse_latw"))
PY
```

Expected:
- K0 is paper-facing zero-shot.
- K4/K12 are paper-facing only if `stage3_posterior_decision=accepted`; otherwise report them as source-policy fallback/rejected diagnostics.
- target_eval is not used for selection or policy calibration.

### Task 5: Close With A Decision

**Files:**
- Optionally create: `reports/experiments/US_R1_seed0_stage3_source_safe_miniloop_20260625.md`

- [ ] **Step 1: Summarize one of three outcomes**

Use one of these decisions:

```text
Outcome A: source-side policy accepted K12 and improves vs K0 -> expand source-side calibration to all source regions and more seeds before any paper claim.
Outcome B: source-side policy accepted but target_eval worsens -> keep as negative SAFE policy diagnostic, do not tune target_eval.
Outcome C: source-side policy cannot export or rejects K-shot -> K12 target-support diagnostic remains an upper-bound signal only; shift next effort to Stage 2/source-side methods or redesign source-side policy features.
```
