# Docs Index

**Navigation:** See [INDEX.md](INDEX.md) for a comprehensive, topic-organized index of all documentation.

---

## Quick Reference (Start Here)

| Document | Purpose |
| --- | --- |
| [INDEX.md](INDEX.md) | **Comprehensive navigation index** - all docs organized by topic |
| [REPO_STATE.md](current/REPO_STATE.md) | Primary source of truth for what the code does today |
| [SPATIAL_HEADS.md](SPATIAL_HEADS.md) | Spatial target head comparison and quick setup |
| [CLAUDE_INSTRUCTIONS.md](CLAUDE_INSTRUCTIONS.md) | Rules and patterns for AI agents working on this repo |

## Current Planning

- [ACTION_FEEDBACK_PLAN.md](current/ACTION_FEEDBACK_PLAN.md) - Current stream-token action feedback protocol
- [FRAGMENT_PPO.md](current/FRAGMENT_PPO.md) - Fragment PPO memory contract and Ray boundary
- [RAY_STATUS.md](current/RAY_STATUS.md) - Current Ray implementation status
- [feedback_diagnostics_note.md](current/feedback_diagnostics_note.md) - Feedback JSONL shape and meta layout

## Current State

- Active verification state: `banana_b2048_e4_a10` is the latest live run
  (around 5,000 episodes so far; behavior review pending before final conclusions)

- [REPO_STATE.md](current/REPO_STATE.md) - What the repo does today (semantic click action space, configurable spatial target head, masked-critic, update-before-eval)
- [working_log.md](current/working_log.md) - Compressed implementation history
- [open_questions.md](current/open_questions.md) - Open questions and items needing user input or tooling support

## Architecture & Training

- [THE_BPTT.md](current/THE_BPTT.md) - BPTT/TBPTT design notes and historical analysis
- [ARCHITECTURE.md](current/ARCHITECTURE.md) - Current architecture reference
- [review_architectural_fixes_2026-04-23.md](archive/review_architectural_fixes_2026-04-23.md) - Review of what's landed vs. still live (archived from current/)

## Reviews & Analysis

- [deep-research-report.md](deep-research-report.md) - Current-state review of local BPTT_test code
- [feedback_diagnostics_note.md](current/feedback_diagnostics_note.md) - Feedback diagnostics

## Tooling

- [TEST_SNIPPETS.md](tooling/TEST_SNIPPETS.md) - Reusable command snippets around tests and analysis
- [INSPECTOR_SNIPPETS.md](tooling/INSPECTOR_SNIPPETS.md) - One-shot probes for observation dumps

## Archive

Historical reasoning and specifications. No longer the current execution plan.

### Key Archived Specs
- [spatial_target_migration_spec_BPTT_test.md](archive/spatial_target_migration_spec_BPTT_test.md) - Full spatial head migration spec (Phase 0-3)
- [phase2_spec_2026-04-23.md](archive/phase2_spec_2026-04-23.md) - Coarse-to-fine target head spec (Phase 2 complete)
- [action_history_bridge_verification_2026-04-24.md](archive/action_history_bridge_verification_2026-04-24.md) - Bridge verification report
- [action_refactor.md](archive/action_refactor.md) - Historical MOVE/ATTACK action-refactor status, superseded by the current semantic click stack

### Implementation Records (Previously "Plans")
- [phase2_3_implementation_record_2026-04-23.md](archive/phase2_3_implementation_record_2026-04-23.md) - Phase 2/3 implementation with lessons learned
- [observation_expansion_record_2026-04-24.md](archive/observation_expansion_record_2026-04-24.md) - 24-dim action-history bridge implementation
- [observation_feedback_investigation_2026-04-23.md](archive/observation_feedback_investigation_2026-04-23.md) - Observation feedback investigation
- [action_history_bridge_plan.md](archive/action_history_bridge_plan.md) - Historical 24-dim action-history bridge protocol

### Fix Plans (Historical)
- [planned_fixes.md](archive/planned_fixes.md)
- [fix_plan_archive_2026-04-18.md](archive/fix_plan_archive_2026-04-18.md) (formerly NEXT_FIXES_PLAN.md)
- [fix_plan_archive_2026-04-19.md](archive/fix_plan_archive_2026-04-19.md) (formerly NEXT_FIXES_PLAN_3.md)
- [plan.md](archive/plan.md)
- [bptt_review_checklist.md](archive/bptt_review_checklist.md) (formerly urgent.md) - BPTT review checklist

### Working Logs & Reviews
- [working_log_2026-04-20_pre_compress.md](archive/working_log_2026-04-20_pre_compress.md) - Verbose implementation history before compression
- [independent_review_2026-04-20.md](archive/independent_review_2026-04-20.md) (formerly Claude_rapport.md) - Independent 2026-04-20 review snapshot
- [observations_2026-04-20/](archive/observations_2026-04-20) - Raw conversation transcripts (see README in that directory)

## Ideas (Speculative)

- [WHEN_SHIT_GETS_DONE.md](ideas/WHEN_SHIT_GETS_DONE.md) - Non-neuromorphic branch plan
- [GPU_THROUGHPUT_OPTIMIZATION_PLAN.md](ideas/GPU_THROUGHPUT_OPTIMIZATION_PLAN.md) - GPU, payload, and async optimization ladder
- [RAYPLAN.md](ideas/RAYPLAN.md) - Distributed training plan with Ray
- [THROUGHPUT_PLAN.md](ideas/THROUGHPUT_PLAN.md) - Transport and measurement companion to the Ray plan
- [WHY_YOLO_BAD.md](ideas/WHY_YOLO_BAD.md) - Analysis of YOLO approach issues
- [distributed_roadmap.md](ideas/distributed_roadmap.md) - Distributed training roadmap

## Notes

Side notes and external takes. Potentially insightful, but not a source of truth.

- [Jules_think.md](notes/Jules_think.md)
- [Ai_compatible_testing.md](notes/Ai_compatible_testing.md)
