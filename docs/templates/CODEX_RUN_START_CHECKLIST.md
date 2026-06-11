# Codex Run Start Checklist

Use this checklist at the start of long, resumed, paper-facing, or risky
HydroDA-OOD sessions.

## Session Objective

- User request:
- Task mode: explanation/audit | review | experiment design | implementation | result interpretation
- Expected deliverable:
- Paper-facing claim affected: yes | no

## Worktree And Context

- Current branch:
- Dirty files relevant to this task:
- Dirty files to avoid touching:
- Recent commits inspected:
- Active protocol version:
- Active split artifact:
- Active region artifact:

## Context Loaded

- `CLAUDE.md`
- `docs/COAUTHOR_CONTEXT.md`
- `docs/CODEX_RESEARCH_OPERATING_PROTOCOL.md`
- `context/00_EXECUTABLE_CONTEXT_MAP.md`
- `context/01_RESEARCH_CONTRACT.md`
- `checklists/no_leakage_checklist.md`
- Relevant `tasks/*.md`:
- Relevant `specs/*.yaml`:

## Evidence To Inspect

- Run script:
- Saved config:
- Checkpoint:
- Training logs:
- Evaluation logs:
- Split manifest:
- Data manifest:
- Training code path:
- Evaluation code path:
- Tests:

## Scientific Safety Scan

- Target evaluation labels are eval-only:
- Normalization source:
- Model-selection source:
- Early-stopping source:
- Loss mask contract:
- Metric mask contract:
- Increment sign convention:
- Region mask provenance:
- Reviewer risk most likely to matter:

## Planned Commands

```bash
git status --short
```

Additional commands:

```bash

```

## Stop Conditions

- Ask the user before changing scientific protocol semantics.
- Ask the user before deleting artifacts that are the only evidence for a run.
- Stop and report if target-evaluation leakage is detected.
