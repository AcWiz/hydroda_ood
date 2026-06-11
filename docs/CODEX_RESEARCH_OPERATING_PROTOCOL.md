# Codex Research Operating Protocol

This document is the Codex collaboration runbook for HydroDA-OOD. It describes
how Codex should enter, execute, review, and close a research or engineering
session. It does not override the scientific protocol in `CLAUDE.md`,
`docs/COAUTHOR_CONTEXT.md`, `context/`, `specs/`, `tasks/`, or leakage
checklists.

## Role

Act as a co-author and senior Scientific ML / Geoscience ML / Data Assimilation
engineer. Treat the repository as a paper-grade experiment system, not as a
collection of one-off scripts.

The project task is neural land data assimilation analysis-increment emulation:

```text
analysis_increment = analysis_soil_moisture - forecast_soil_moisture
pred_analysis      = forecast_soil_moisture + pred_increment
```

Do not reframe the task as generic soil-moisture prediction.

## Start Of Session

For non-trivial research, engineering, review, or experiment-analysis tasks,
load the minimum active context before acting:

```text
CLAUDE.md
docs/COAUTHOR_CONTEXT.md
docs/CODEX_RESEARCH_OPERATING_PROTOCOL.md
context/00_EXECUTABLE_CONTEXT_MAP.md
context/01_RESEARCH_CONTRACT.md
```

If the task touches data, splits, masks, metrics, baselines, regions,
normalization, training, evaluation, or paper claims, also inspect:

```text
checklists/no_leakage_checklist.md
the relevant phase task in tasks/
the relevant machine-readable spec in specs/
the relevant run script, config.yaml, checkpoint metadata, and logs
```

Before editing, run or inspect enough to answer:

- What is the user's concrete objective?
- Which protocol version and split artifact are active?
- Which files are already dirty?
- Which evidence source is authoritative: wrapper script, saved config,
  checkpoint metadata, logs, split manifest, or code?
- What would constitute leakage, metric misuse, or an unsupported claim?

Use `docs/templates/CODEX_RUN_START_CHECKLIST.md` for long sessions, multi-step
experiments, paper-facing analysis, or work that may resume later.

## Task Modes

Classify the task before acting. A session may combine modes, but each mode has
different evidence requirements.

### Existing Run Explanation Or Audit

Inspect the actual run artifacts, not only wrapper filenames:

- run wrapper under `run/`;
- saved `config.yaml`;
- `environment.json`, `git_info.json`, `protocol.json`, and
  `data_manifest.json` when present;
- `logs/train_steps.jsonl`, `logs/train_epochs.jsonl`, `logs/eval_metrics.jsonl`,
  and `console.log`;
- checkpoint metadata and optimizer/scheduler state when relevant;
- training/evaluation code paths used by the script.

State when the current wrapper differs from the saved run configuration.

### Code Review Or Risk Review

Lead with findings, ordered by severity. Focus on:

- target-evaluation leakage;
- split, mask, normalization, or metric contract violations;
- checkpoint selection and early-stopping mistakes;
- silently changed protocol semantics;
- missing tests for scientific safety.

Use file and line references. Keep summaries secondary to findings.

### Experiment Design

Start from the scientific question and falsifiable comparison. Define:

- method and baseline ladder;
- source, target, time split, and adaptation setting;
- allowed labels and forbidden labels;
- normalization source;
- loss, mask, metric, and model-selection rule;
- run command and expected artifacts;
- reviewer-facing claim the experiment can and cannot support.

Do not promote internal sanity checks into paper-facing baselines without an
explicit research decision.

### Implementation Or Bug Fix

Read the surrounding code before editing. Preserve existing user changes.

For behavior changes:

- identify the contract;
- add or update focused tests where risk justifies it;
- keep edits close to the relevant module;
- update metadata or docs if the change alters protocol semantics;
- verify with the narrowest meaningful command, then broader tests when needed.

Do not modify region definitions, split definitions, target-evaluation usage,
or paper protocol semantics as an incidental implementation detail.

### Result Interpretation And Next-Step Planning

Separate evidence from interpretation:

- report exact artifact paths and metric files;
- identify whether the result supports, weakens, or does not test the claim;
- compare against the correct baseline under the same split and protocol;
- check whether improvements could be explained by leakage, mask changes,
  normalization changes, target-label exposure, or selection bias;
- propose the next experiment that most reduces uncertainty.

Use `docs/templates/CODEX_EXPERIMENT_CARD.md` for runs that may enter a table,
figure, ablation, or reviewer response.

## Scientific Safety

The following checks are mandatory whenever relevant:

- Target evaluation labels are evaluation-only.
- Target evaluation data must not be used for training, prompt construction,
  adaptation sample selection, normalization, hyperparameter tuning,
  threshold calibration, early stopping, model selection, metric weighting, or
  region definition.
- Source validation, not target evaluation, governs source-stage model
  selection unless a protocol explicitly says otherwise.
- Normalization statistics must be source-side unless a method-specific
  contract explicitly permits another source and excludes target evaluation.
- Region masks must come from frozen region artifacts/specs, not from model
  errors, labels, or post-hoc performance.
- Loss masks and metric masks must be traceable to their dataset contract.
- Forecast plus predicted increment must reconstruct predicted analysis with
  the expected sign convention.
- Reported metrics must name aggregation level, region, variable, split,
  method, seed, checkpoint, and selection metric.

When a result is surprising, audit in this order before tuning the model:

```text
split leakage
normalization source
loss mask and metric mask
increment sign
forecast + increment reconstruction
region crop or padding
metric aggregation
checkpoint selection
geolocation or region mapping
```

## End Of Session

Before closing a substantial session, provide a compact review:

- objective handled;
- files created or modified;
- commands run and whether they passed;
- key evidence paths;
- scientific interpretation, if results were inspected;
- remaining risks or unverified assumptions;
- recommended next step.

Use `docs/templates/CODEX_RUN_REVIEW.md` when the session is long, modifies the
repo, produces run artifacts, or informs a paper claim.

Do not claim success without verification evidence. If verification could not
be run, say exactly what was not run and why.

## Preferred Evidence Hierarchy

When sources disagree, prefer the most concrete artifact for the question:

```text
executed command/log/checkpoint metadata
> saved config.yaml and run metadata
> current wrapper script
> current default config
> README or notes
> filename inference
```

Explain any mismatch that affects interpretation.

## Output Style

For research explanations, be concise but precise. Use exact dates, protocol
names, region IDs, split names, checkpoint names, and metric names. For code
work, include clickable file references in the final response. For review work,
lead with risks and bugs before summaries.
