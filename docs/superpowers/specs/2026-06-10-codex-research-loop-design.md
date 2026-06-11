# Codex Research Loop Design

## Purpose

This spec defines a lightweight but enforceable operating protocol for Codex as
a co-authoring research agent in HydroDA-OOD. The goal is to make every Codex
session start from the right scientific context, inspect the correct evidence,
avoid leakage-prone shortcuts, and end with a useful research review plus next
steps.

## Scope

The design adds a documentation package, not a new training or evaluation
system. It does not replace `CLAUDE.md`, `docs/COAUTHOR_CONTEXT.md`, phase
tasks, specs, or checklists. Those documents continue to define the scientific
protocol. The new protocol defines how Codex should operate around them.

## Files

Create:

- `docs/CODEX_RESEARCH_OPERATING_PROTOCOL.md`
- `docs/templates/CODEX_RUN_START_CHECKLIST.md`
- `docs/templates/CODEX_RUN_REVIEW.md`
- `docs/templates/CODEX_EXPERIMENT_CARD.md`
- `docs/superpowers/plans/2026-06-10-codex-research-loop.md`

Modify with short pointers only:

- `AGENTS.md`
- `docs/COAUTHOR_CONTEXT.md`
- `context/00_EXECUTABLE_CONTEXT_MAP.md`

## Operating Model

Codex should treat the repository as a paper-grade scientific ML and data
assimilation system. At the beginning of non-trivial work, Codex loads the
minimal active context, classifies the task, checks the worktree, identifies the
relevant scripts/configs/checkpoints/logs, and states the evidence it will use.

The protocol distinguishes five common task modes:

1. Explanation or audit of an existing run.
2. Code review or risk review.
3. Experiment design.
4. Implementation or bug fix.
5. Result interpretation and next-step planning.

Each mode has a small required evidence checklist. For example, an experiment
audit should inspect the wrapper script, saved `config.yaml`, logs, checkpoint
metadata, split manifest, and relevant training/evaluation code rather than
inferring settings from filenames alone.

## Leakage and Reviewer Discipline

The protocol emphasizes reviewer-facing checks:

- target evaluation labels are evaluation-only;
- normalization and model selection must be source-side unless explicitly
  contracted otherwise;
- split artifacts and region masks must be named and traceable;
- reported claims require metric evidence and the exact run artifacts;
- anomalous results should trigger audits of mask, normalization, increment
  sign, reconstruction, metric aggregation, and split leakage before model
  tuning.

## Templates

The start checklist captures the session premise before action. The run review
captures completed actions, verification, residual risks, and the next suggested
research step. The experiment card records a single run in a compact form that
can later be used for paper tables, audit trails, or reviewer responses.

## Entry Points

`AGENTS.md` should tell Codex to read the operating protocol before complex
research or engineering tasks. `docs/COAUTHOR_CONTEXT.md` should mention the
protocol as the Codex collaboration entry point. The executable context map
should register the protocol without changing phase-specific scientific rules.

## Non-Goals

- No changes to model code, training behavior, or evaluation metrics.
- No replacement of existing V4.4 research protocol documents.
- No new mandatory automation scripts.
- No rewriting of existing dirty worktree changes.
