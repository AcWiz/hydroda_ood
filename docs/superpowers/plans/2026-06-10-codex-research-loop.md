# Codex Research Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable Codex research operating protocol, session templates, and minimal entry-point pointers for HydroDA-OOD.

**Architecture:** This is a documentation-only change. The core protocol lives in one canonical document under `docs/`, templates live under `docs/templates/`, and existing entry documents get short pointers without absorbing the full protocol.

**Tech Stack:** Markdown documentation, repository conventions, `rg` for verification.

---

### Task 1: Add The Protocol Document

**Files:**
- Create: `docs/CODEX_RESEARCH_OPERATING_PROTOCOL.md`

- [ ] **Step 1: Create the protocol**

Write a Markdown document with these sections:

```markdown
# Codex Research Operating Protocol

## Role
Define Codex as a co-authoring research agent for HydroDA-OOD, not a generic script editor.

## Start Of Session
List the minimum context files and the start checklist.

## Task Modes
Define evidence requirements for explanation/audit, review, experiment design, implementation, and result interpretation.

## Scientific Safety
List leakage, split, normalization, mask, and metric checks.

## End Of Session
Define required summary, verification, risks, and next-step outputs.
```

- [ ] **Step 2: Verify the protocol has no incomplete markers**

Run:

```bash
rg -n "T""BD|TO""DO|FI""XME|IMPLEMENT""_LATER|REPLACE""_ME" docs/CODEX_RESEARCH_OPERATING_PROTOCOL.md
```

Expected: no matches.

### Task 2: Add Session Templates

**Files:**
- Create: `docs/templates/CODEX_RUN_START_CHECKLIST.md`
- Create: `docs/templates/CODEX_RUN_REVIEW.md`
- Create: `docs/templates/CODEX_EXPERIMENT_CARD.md`

- [ ] **Step 1: Create the start checklist**

The checklist should capture worktree state, task mode, loaded context, relevant artifacts, leakage-risk scan, planned commands, and expected deliverable.

- [ ] **Step 2: Create the run review template**

The review template should capture completed actions, files changed, commands run, verification evidence, interpretation, residual risks, and next steps.

- [ ] **Step 3: Create the experiment card**

The experiment card should capture the scientific question, method, split,
mask/loss/normalization/scheduler settings, command, checkpoint, metrics,
interpretation, and reviewer risks.

- [ ] **Step 4: Verify templates have no incomplete markers**

Run:

```bash
rg -n "T""BD|TO""DO|FI""XME|IMPLEMENT""_LATER|REPLACE""_ME" docs/templates/CODEX_*.md
```

Expected: no matches.

### Task 3: Add Minimal Entry Pointers

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/COAUTHOR_CONTEXT.md`
- Modify: `context/00_EXECUTABLE_CONTEXT_MAP.md`

- [ ] **Step 1: Add the Codex pointer to `AGENTS.md`**

Add a short paragraph telling Codex to read `docs/CODEX_RESEARCH_OPERATING_PROTOCOL.md` before complex research, engineering, experiment, or review tasks.

- [ ] **Step 2: Add the collaboration pointer to `docs/COAUTHOR_CONTEXT.md`**

Add one short section that names the Codex operating protocol as the collaboration runbook.

- [ ] **Step 3: Register the protocol in `context/00_EXECUTABLE_CONTEXT_MAP.md`**

Add one short bullet under the global context section stating that Codex sessions should also load the operating protocol.

### Task 4: Verify Cross-References

**Files:**
- Inspect: `AGENTS.md`
- Inspect: `docs/COAUTHOR_CONTEXT.md`
- Inspect: `context/00_EXECUTABLE_CONTEXT_MAP.md`
- Inspect: `docs/CODEX_RESEARCH_OPERATING_PROTOCOL.md`

- [ ] **Step 1: Check all references exist**

Run:

```bash
rg -n "CODEX_RESEARCH_OPERATING_PROTOCOL|CODEX_RUN_START_CHECKLIST|CODEX_RUN_REVIEW|CODEX_EXPERIMENT_CARD" AGENTS.md docs context/00_EXECUTABLE_CONTEXT_MAP.md
```

Expected: each new document is referenced at least once.

- [ ] **Step 2: Review the diff**

Run:

```bash
git diff -- AGENTS.md docs/CODEX_RESEARCH_OPERATING_PROTOCOL.md docs/templates docs/COAUTHOR_CONTEXT.md context/00_EXECUTABLE_CONTEXT_MAP.md docs/superpowers/specs/2026-06-10-codex-research-loop-design.md docs/superpowers/plans/2026-06-10-codex-research-loop.md
```

Expected: only documentation additions and short entry-point updates.
