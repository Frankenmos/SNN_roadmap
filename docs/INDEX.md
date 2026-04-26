# Documentation Index

Start here when you want repo context without spelunking through old plans.

---

## Quick Reference (Start Here)

| Document | Lines | Purpose |
|----------|-------|---------|
| [REPO_STATE.md](current/REPO_STATE.md) | 258 | Primary source of truth for what the code does today |
| [SPATIAL_HEADS.md](SPATIAL_HEADS.md) | 51 | Spatial target head comparison and quick setup |
| [CLAUDE_INSTRUCTIONS.md](CLAUDE_INSTRUCTIONS.md) | 105 | Rules and patterns for AI agents working on this repo |

---

## Current (Active Planning & State)

### Repository State & Planning
- [REPO_STATE.md](current/REPO_STATE.md) - What the repo does today; includes semantic action space, token-pointer target head, masked-critic semantics
- [working_log.md](current/working_log.md) - Compressed implementation history (297 lines)
- [open_questions.md](current/open_questions.md) - Open questions and items needing user input

### Action History & Observation Protocol
- [ACTION_FEEDBACK_PLAN.md](current/ACTION_FEEDBACK_PLAN.md) - Current stream action-feedback token protocol
- [action_history_bridge_plan.md](current/action_history_bridge_plan.md) - Historical 24-dim action-history bridge protocol
- [feedback_diagnostics_note.md](current/feedback_diagnostics_note.md) - Observed feedback JSONL shape and timing semantics from the bridge investigation

### BPTT & Training Architecture
- [THE_BPTT.md](current/THE_BPTT.md) - BPTT/TBPTT design notes and historical analysis (699 lines)

### Reviews & Analysis
- [review_architectural_fixes_2026-04-23.md](archive/review_architectural_fixes_2026-04-23.md) - Review of what's landed vs. still live (857 lines)
- [deep-research-report.md](deep-research-report.md) - Current-state review of local BPTT_test code (372 lines)
- [feedback_diagnostics_note.md](current/feedback_diagnostics_note.md) - Observed feedback diagnostics

---

## Archive (Historical)

### Specifications & Migration Plans
- [spatial_target_migration_spec_BPTT_test.md](archive/spatial_target_migration_spec_BPTT_test.md) - Full spatial head migration spec (Phase 0-3); Phases 0-2 complete, Phase 3 pending (898 lines)
- [phase2_spec_2026-04-23.md](archive/phase2_spec_2026-04-23.md) - Coarse-to-fine target head specification (Phase 2 complete) (536 lines)
- [action_refactor.md](archive/action_refactor.md) - Status note for the landed action-space work
- [action_history_bridge_verification_2026-04-24.md](archive/action_history_bridge_verification_2026-04-24.md) - Verification report for action-history bridge

### Implementation Records
- [phase2_3_implementation_record_2026-04-23.md](archive/phase2_3_implementation_record_2026-04-23.md) - Phase 2/3 implementation details with lessons learned (518 lines)
- [observation_expansion_record_2026-04-24.md](archive/observation_expansion_record_2026-04-24.md) - 24-dim action-history bridge implementation record
- [observation_feedback_investigation_2026-04-23.md](archive/observation_feedback_investigation_2026-04-23.md) - Investigation findings about missing observation fields

### Fix Plans (Historical)
- [planned_fixes.md](archive/planned_fixes.md) - Earlier fix planning
- [fix_plan_archive_2026-04-18.md](archive/fix_plan_archive_2026-04-18.md) - Previous fixes plan (formerly NEXT_FIXES_PLAN.md)
- [fix_plan_archive_2026-04-19.md](archive/fix_plan_archive_2026-04-19.md) - Another iteration of fixes plan (formerly NEXT_FIXES_PLAN_3.md)
- [plan.md](archive/plan.md) - General planning document

### Working Logs & Reviews
- [working_log_2026-04-20_pre_compress.md](archive/working_log_2026-04-20_pre_compress.md) - Verbose implementation history before compression (164 lines)
- [independent_review_2026-04-20.md](archive/independent_review_2026-04-20.md) - Independent 2026-04-20 review snapshot (493 lines)
- [observations_2026-04-20/external_discussion_log.md](archive/observations_2026-04-20/external_discussion_log.md) - **Raw conversation transcript** (4824 lines - archived for reference)

### Verification Reports
- [action_history_bridge_verification_2026-04-24.md](archive/action_history_bridge_verification_2026-04-24.md) - Bridge verification report

---

## Ideas (Speculative & Future)

### Future Branches
- [WHEN_SHIT_GETS_DONE.md](ideas/WHEN_SHIT_GETS_DONE.md) - Non-neuromorphic branch plan (587 lines)
- [RAYPLAN.md](ideas/RAYPLAN.md) - Distributed training plan with Ray (824 lines)
- [THROUGHPUT_PLAN.md](ideas/THROUGHPUT_PLAN.md) - Transport and measurement companion to the Ray plan (470 lines)
- [distributed_roadmap.md](ideas/distributed_roadmap.md) - Distributed training roadmap (75 lines)

### Analysis & Critique
- [WHY_YOLO_BAD.md](ideas/WHY_YOLO_BAD.md) - Analysis of YOLO approach issues (460 lines)

---

## Tooling

- [TEST_SNIPPETS.md](tooling/TEST_SNIPPETS.md) - Reusable command snippets around tests and analysis (367 lines)
- [INSPECTOR_SNIPPETS.md](tooling/INSPECTOR_SNIPPETS.md) - One-shot probes for observation dumps (200 lines)

---

## Notes (External Takes & Side Notes)

- [Jules_think.md](notes/Jules_think.md) - External thoughts (42 lines)
- [Ai_compatible_testing.md](notes/Ai_compatible_testing.md) - AI testing notes (28 lines)

---

## By Topic

### Training & PPO
- [THE_BPTT.md](current/THE_BPTT.md) - BPTT/TBPTT design
- [review_architectural_fixes_2026-04-23.md](archive/review_architectural_fixes_2026-04-23.md) - Training fixes analysis
- [deep-research-report.md](deep-research-report.md) - Current-state review
- [bptt_review_checklist.md](archive/bptt_review_checklist.md) - BPTT review baseline checklist

### Architecture (SNN, Policy, Attention)
- [THE_BPTT.md](current/THE_BPTT.md) - Temporal SNN architecture
- [SPATIAL_HEADS.md](SPATIAL_HEADS.md) - Spatial head comparison
- [spatial_target_migration_spec_BPTT_test.md](archive/spatial_target_migration_spec_BPTT_test.md) - Migration spec

### Action Spaces & Target Heads
- [ACTION_FEEDBACK_PLAN.md](current/ACTION_FEEDBACK_PLAN.md) - Current stream-token action feedback protocol
- [action_history_bridge_plan.md](current/action_history_bridge_plan.md) - Historical 24-dim bridge protocol
- [SPATIAL_HEADS.md](SPATIAL_HEADS.md) - Spatial target head reference
- [action_refactor.md](archive/action_refactor.md) - Action refactor status
- [phase2_spec_2026-04-23.md](archive/phase2_spec_2026-04-23.md) - Coarse-to-fine spec (Phase 2 complete)

### Reward Functions
- [REPO_STATE.md](current/REPO_STATE.md) - Current reward function status
- [deep-research-report.md](deep-research-report.md) - Reward analysis

### Diagnostics & Analysis
- [feedback_diagnostics_note.md](current/feedback_diagnostics_note.md) - Feedback diagnostics
- [review_architectural_fixes_2026-04-23.md](archive/review_architectural_fixes_2026-04-23.md) - What's landed vs. live
- [observation_feedback_investigation_2026-04-23.md](archive/observation_feedback_investigation_2026-04-23.md) - Investigation findings

### Distributed Training (Future)
- [RAYPLAN.md](ideas/RAYPLAN.md) - Ray distributed plan
- [THROUGHPUT_PLAN.md](ideas/THROUGHPUT_PLAN.md) - Ray transport and throughput plan
- [distributed_roadmap.md](ideas/distributed_roadmap.md) - Roadmap

---

## File Size Reference

| File | Lines | Location |
|------|-------|----------|
| external_discussion_log.md | 4824 | archive/observations_2026-04-20/ |
| spatial_target_migration_spec_BPTT_test.md | 898 | archive/ |
| review_architectural_fixes_2026-04-23.md | 857 | archive/ |
| THE_BPTT.md | 699 | current/ |
| RAYPLAN.md | 824 | ideas/ |
| WHEN_SHIT_GETS_DONE.md | 587 | ideas/ |
| phase2_spec_2026-04-23.md | 536 | archive/ |
| phase2_3_implementation_record_2026-04-23.md | 518 | archive/ |
| independent_review_2026-04-20.md | 493 | archive/ |
| planned_fixes.md | 473 | archive/ |
| THROUGHPUT_PLAN.md | 470 | ideas/ |
| WHY_YOLO_BAD.md | 460 | ideas/ |
| deep-research-report.md | 372 | docs/ |
| TEST_SNIPPETS.md | 367 | tooling/ |
| plan.md | 363 | archive/ |
| working_log.md | 297 | current/ |
| REPO_STATE.md | 258 | current/ |
| feedback_diagnostics_note.md | 237 | current/ |
| INSPECTOR_SNIPPETS.md | 200 | tooling/ |

---

**Last Updated:** 2026-04-26
