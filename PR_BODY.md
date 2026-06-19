## Summary

Five-commit addition to `zero-few-shot-protocol` ahead of the Phase 6A BORA-HyperDA work.

- **Robust prompt encoder** — new `RobustInputSideDAPromptEncoder` (median/IQR) as a source-stage context-encoder option, plus numerical-stability fix in the base `RegionPromptEncoder`. Backwards-compatible factory in baselines + train scripts; old checkpoints default to `current_mean_std`. Channel 11 (`base_valid_mask`) is deliberately neutralised so it cannot leak as observation/region semantics.
- **Dataset fix** — `_get_date_records` now preserves the `_DATES_KEY_FALLBACKS` chain when the primary key resolves to an empty list, fixing target_eval fallback in zero/few-shot protocols.
- **Locked-eval preflight** — `review_hyperda_zero_few_shot_run` now validates source-safe selection artifacts (recommendation, leakage audit, selected guard config, calibration summary, source checkpoint, split manifest) with frozen-recipe guard, hash pinning, and a full metadata audit payload before any target_eval summary is written.
- **HyperDA+ source-prior matrix scaffolding** — H0–H4 candidates declared in `configs/experiments/hyperda_plus_source_prior_matrix.yaml`, source-only protocol wrapper, aggregator report, and protocol-text tests. H0–H3 runnable; H4 scaffolded with a TODO.
- **P2 audit + US-R1 WRMSE ablation utilities** — read-only auditors for the P2.8 locked-eval workflow and the US-R1 WRMSE ablation table.

## Test plan

- [x] `pytest tests/test_hyperda_few_shot_runner.py tests/test_hyperda_model.py tests/test_hyperda_run_review.py tests/test_phase4_prompt_conditioned_protocol_text.py tests/test_prompt_conditioned_smoke.py tests/test_zero_few_shot_protocol.py tests/test_audit_hyperda_p2_suite.py tests/test_hyperda_plus_source_prior_matrix_report.py tests/test_us_r1_wrmse_ablation_table.py` — 112 passed
- [x] `pytest tests/ --ignore=tests/test_der_router.py` — 607 passed, 12 pre-existing failures (`test_der_router`, `test_ridge_baseline`, `test_protocol_dataset_hardening`, `test_hyperda_target_adapt_runner::test_phase5_*_script_forwards_*`) verified to fail on clean `HEAD` before this branch's commits and unrelated to its changes.
- [ ] CI

## Commits

1. `feat(prompt-encoder): add robust input-side DA diagnostics + numerical-stability guard`
2. `fix(data): dataset date-records fallback when primary split key is empty`
3. `feat(analysis): hyperda zero/few-shot locked-eval preflight + metadata audit`
4. `feat(phase4+): hyperda+ source-prior matrix scaffolding (H0-H4)`
5. `feat(analysis): p2 suite audit + us-r1 wrmse ablation table utilities`

Also drops `docs/HYPERDA_STABILITY_PLAN.md` (content folded into the hyperda+ matrix scaffolding notes).

## Notes

- Working tree still has 14 untracked files outside this PR's scope (12 Phase 6A BORA-HyperDA artifacts + 2 BORA Codex plans). They are intentionally not included.
- `.claude/scheduled_tasks.lock` is a local session lock and is gitignored in practice.

## Open PR

Title: `feat: robust prompt encoder + locked-eval preflight + hyperda+ matrix`
Base: `main` · Head: `zero-few-shot-protocol`
