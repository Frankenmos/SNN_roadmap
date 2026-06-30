# Observations Archive (2026-04-20)

This directory contains raw conversation transcripts and discussion logs from external action-space reviews that informed the Stage-1 refactor.

## Contents

- **external_discussion_log.md** (4824 lines) - Raw conversation transcript covering:
  - Action token design and hybrid head discussion
  - BPTT vs TBPTT for SNN architectures
  - EventProp vs surrogate gradients analysis
  - Masking techniques (attention heads, KV cache, neurons, observation data)
  - Action space design discussions

## Purpose

These are **archived reference materials**, not current specifications. They preserve the reasoning and discussion that led to:
- The 24-dim action-history bridge protocol
- Semantic action space (`NO_OP`, `LEFT_CLICK`, `RIGHT_CLICK`)
- Token-pointer target head architecture

## Navigation

For the current state of these implementations, see:
- [action_history_bridge_plan.md](../../current/action_history_bridge_plan.md)
- [REPO_STATE.md](../../current/REPO_STATE.md)
- [spatial_target_migration_spec_BPTT_test.md](../spatial_target_migration_spec_BPTT_test.md)

## Date Notes

Archived on 2026-04-20. Some recommendations in these discussions have been implemented; others remain speculative or were superseded by later architectural decisions.
