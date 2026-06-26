# Plan: Observation Expansion - Adding Action Feedback

**Created:** 2026-04-23
**Status:** 24-dim action-history bridge implemented

## Problem Statement

The agent currently has limited awareness of its own actions' effects:

- Knows it executed SMART at `(x, y)` via `last_action_token`
- Does not know directly whether SC2 accepted/executed that action
- Does not see game score progression in policy input
- Does not currently get useful `action_result` or `alerts` data

## Discovery: PySC2 Has Useful Feedback

From PySC2 docs and the `theta-1` diagnostic dump:

| Field | What It Provides | Current Read |
|-------|------------------|--------------|
| `last_actions` | PySC2 actions executed since previous observation | Useful: `451` tracks `Smart_screen` execution |
| `score_cumulative` | Score/progress vector | Useful: indices `0`, `3`, and `5` moved in `theta-1` |
| `action_result` | Error codes for action results | Empty in inspected DefeatRoaches runs |
| `alerts` | Alert notifications | Empty in inspected DefeatRoaches runs |

See [feedback_diagnostics_note_2026-04-24.md](feedback_diagnostics_note_2026-04-24.md) for the exact
diagnostic JSONL shape and timing semantics.

## Phase 1: Inspection

Done:

- Added `LastActionDiagnosticsWrapper`
- Added `ScoreDiagnosticsWrapper`
- Wired both into `eval.py`, `train.py`, `envs/setup_env.py`, and `config.yaml`
- Inspected `theta-1` diagnostics
- Confirmed clean timing examples where:
  - dispatched `Smart_screen` returns `current_frame.last_action_ids = [451]`
  - dispatched `no_op` returns `current_frame.last_action_ids = []`

Important caveat:

- Diagnostic JSONL files append by default.
- Reusing the same output path across eval invocations can repeat `episode` and
  `step` labels.
- Future tooling should add a `run_id` or truncate option.

## Phase 2: Design Expanded Meta Vector

Previous `meta_vec[19]`:

| Slice | Dim | Meaning |
|-------|-----|---------|
| `0:11` | 11 | player features |
| `11:14` | 3 | semantic available-action mask |
| `14:15` | 1 | PySC2 last-action index |
| `15:19` | 4 | agent bridge token `[type, x_norm, y_norm, extra]` |

Implemented evidence-backed expansion:

| Offset | Field | Dim | Encoding |
|--------|-------|-----|----------|
| `19` | `last_any_action_executed` | 1 | `1.0` if `last_actions` is non-empty |
| `20` | `last_smart_executed` | 1 | `1.0` if `451 in last_actions` |
| `21` | `score_total_delta` | 1 | clipped/normalized delta of `score_cumulative[0]` |
| `22` | `killed_value_delta` | 1 | clipped/normalized delta of `score_cumulative[5]` |
| `23` | `score_penalty_bit` | 1 | `1.0` if `score_total_delta < 0` |

Target landed: `META_VECTOR_DIM` expanded from `19` to `24`.

The source-of-truth protocol note is now
[action_history_bridge_plan.md](action_history_bridge_plan.md).

Deferred until it earns space:

- `action_result`: observed empty so far
- `alerts`: observed empty so far
- broader score vector: useful for diagnostics, probably too wide for the first
  policy-input expansion

## Phase 3: Implementation Plan

Files to modify:

- `agent_core/policy_protocol.py` - add offsets/dims and update `META_VECTOR_DIM`
- `obs_space/obs_space_2.py` - extract `last_actions` and score deltas
- `agent_core/spiking_policy.py` - expand meta encoder input if needed by config
- `config.yaml` - update `model.vector_input_dim`
- tests under `tests/` and `tests/MockedEnv/`

Implementation notes:

1. Track previous `score_cumulative` inside `ObservationExtractor` or a small
   reset-aware state helper.
2. Reset score-delta state on episode reset.
3. Keep `last_actions` extraction stateless.
4. First-pass score transforms:
   - `score_total_delta = clip(delta(score_cumulative[0]), -10, 10) / 10`
   - `killed_value_delta = clip(delta(score_cumulative[5]), 0, 100) / 100`
   - `score_penalty_bit = 1.0 if delta(score_cumulative[0]) < 0 else 0.0`
5. Treat checkpoint compatibility as broken after the dimension change.

## Phase 4: Testing

Needed:

- Unit tests for empty/non-empty `last_actions`
- Unit tests for score delta reset and score delta clipping
- `PolicyInputBatch` shape tests for `META_VECTOR_DIM = 24`
- Policy forward-pass smoke test
- Short eval/training smoke test with diagnostics enabled

## Status

- [x] Phase 1a: Add focused eval diagnostics for last-action feedback and score deltas
- [x] Phase 1b: Inspect `theta-1` JSONL and identify useful feedback fields
- [x] Phase 1c: Document observed diagnostic shape and timing semantics
- [x] Phase 2a: Document first-pass score-delta normalization/clipping
- [x] Phase 2b: Confirm or revise score normalization before code
- [x] Phase 3: Implement 24-dim `meta_vec`
- [x] Phase 4: Test and smoke-run

---

**Next Step:** Run evals with the current checkpoint shape and inspect the
policy-input diagnostics to confirm the bridge fields are populated in live
PySC2 observations.
