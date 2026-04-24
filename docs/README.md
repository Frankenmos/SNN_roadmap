# Docs Index

Start here when you want repo context without spelunking through old plans.

## Quick Reference

- [SPATIAL_HEADS.md](SPATIAL_HEADS.md) - Spatial target head comparison and quick setup
- [CLAUDE_INSTRUCTIONS.md](CLAUDE_INSTRUCTIONS.md) - Rules and patterns for AI agents working on this repo

## Current Planning

- [current/observation_expansion_plan.md](current/observation_expansion_plan.md) - Plan for adding action feedback (action_result, alerts, score_cumulative)
- [current/observation_feedback_investigation.md](current/observation_feedback_investigation.md) - Investigation findings about missing observation fields
- [current/feedback_diagnostics_note.md](current/feedback_diagnostics_note.md) - Observed feedback JSONL shape, timing semantics, and candidate 24-dim meta shape

## Current

- Active verification state: `Zero` is the live run (analysis not yet complete on final checkpoints).

- [PHASE2_WORKING_LOG.md](PHASE2_WORKING_LOG.md) - Phase 2 (CoarseToFineTargetHead) implementation complete
- [current/REPO_STATE.md](current/REPO_STATE.md)
  primary source of truth for what the code does today, what is stable,
  and what is still open; now reflects the semantic click action space,
  token-pointer target head, masked-critic semantics, and
  the update-before-eval trainer ordering
- [current/action_refactor.md](current/action_refactor.md)
  status note for the landed action-space work; now covers both the old
  `MOVE/ATTACK` stage and the later simplification to `SMART`, plus what
  broader action-space work is still deferred
- [current/THE_BPTT.md](current/THE_BPTT.md)
  current staged BPTT/TBPTT reasoning; still relevant because the repo
  now runs Stage-1 TBPTT and we still need to evaluate the verdict; use
  this for the historical "why TBPTT was needed" context
- [current/working_log.md](current/working_log.md)
  compressed implementation log; now includes the rename-boundary
  cleanup, semantic-action migration, token-pointer target head,
  masked-critic decision, and the latest future-branch notes
- [current/help-needed.md](current/help-needed.md)
  practical handoff note for what still needs user input, env-backed
  verification, or tooling support after the semantic-action /
  token-pointer migration
- [current/BPTT_test_review_report (1).md](<current/BPTT_test_review_report%20(1).md>)
  independent review focused on the live branch after TBPTT and SMART;
  especially useful for understanding why the structured spatial click
  head and positional encoding were worth landing
- [current/Claude_rapport.md](current/Claude_rapport.md)
  independent 2026-04-20 review snapshot; still useful for reward/eval
  reasoning, but no longer the primary source of truth after the
  2026-04-21 repo-state refresh
- [current/urgent.md](current/urgent.md)
  contains the active BPTT review baseline and the naming/continuity repair
  checklist; still useful for comparing recommendations against current
  implementation status

## Tooling

- [tooling/INSPECTOR_SNIPPETS.md](tooling/INSPECTOR_SNIPPETS.md)
  one-shot probes for observation dumps
- [tooling/TEST_SNIPPETS.md](tooling/TEST_SNIPPETS.md)
  reusable command snippets around tests and analysis

## Archive

These are worth reading for historical reasoning, but they are no
longer the current execution plan.

- [archive/spatial_target_migration_spec_BPTT_test.md](archive/spatial_target_migration_spec_BPTT_test.md)
  full spatial head migration spec (Phase 0-3); Phases 0-2 complete,
  Phase 3 (heatmap) pending
- [archive/NEXT_FIXES_PLAN.md](archive/NEXT_FIXES_PLAN.md)
- [archive/NEXT_FIXES_PLAN_3.md](archive/NEXT_FIXES_PLAN_3.md)
- [archive/planned_fixes.md](archive/planned_fixes.md)
- [archive/plan.md](archive/plan.md)
- [archive/working_log_2026-04-20_pre_compress.md](archive/working_log_2026-04-20_pre_compress.md)
  verbose implementation history before the current log was compressed
- [archive/observations_2026-04-20/](archive/observations_2026-04-20)
  archived external action-space discussion and draft files that
  informed the Stage-1 refactor

## Ideas

Speculative or future-branch documents.

- [ideas/WHEN_SHIT_GETS_DONE.md](ideas/WHEN_SHIT_GETS_DONE.md)
- [ideas/RAYPLAN.md](ideas/RAYPLAN.md)
- [ideas/WHY_YOLO_BAD.md](ideas/WHY_YOLO_BAD.md)

## Notes

Side notes and external takes. Potentially insightful, but not a source
of truth by themselves.

- [notes/Ai_compatible_testing.md](notes/Ai_compatible_testing.md)
- [notes/Jules_think.md](notes/Jules_think.md)
