# Action-History Bridge Encoding

Created: 2026-04-24
Status: Implemented as the current 24-dim meta-vector protocol

## Why This Exists

I agree with the direction here: the agent should not only remember what it
tried to do, it should also see the next observation's compact feedback about
whether the game accepted the command and whether score moved afterward.

This keeps the bridge small and evidence-backed while giving the recurrent
policy a much cleaner temporal clue than raw pixels alone.

## Meta Layout

`META_VECTOR_DIM = 24`.

| Slice | Dim | Meaning |
|-------|-----|---------|
| `0:11` | 11 | player features |
| `11:14` | 3 | semantic available-action mask |
| `14:15` | 1 | PySC2 last-action index |
| `15:19` | 4 | attempted action bridge `[type, x_norm, y_norm, extra]` |
| `19` | 1 | `last_any_action_executed` |
| `20` | 1 | `last_smart_executed` |
| `21` | 1 | `score_total_delta` |
| `22` | 1 | `killed_value_delta` |
| `23` | 1 | `score_penalty_bit` |

The bridge slice is now `meta_vec[15:24]`.

## Encoding

- `last_any_action_executed`: `1.0` if `obs.observation.last_actions` is non-empty.
- `last_smart_executed`: `1.0` if `451 in obs.observation.last_actions`.
- `score_total_delta`: `clip(delta(score_cumulative[0]), -10, 10) / 10`.
- `killed_value_delta`: `clip(delta(score_cumulative[5]), 0, 100) / 100`.
- `score_penalty_bit`: `1.0` if raw `delta(score_cumulative[0]) < 0`.

The score delta is reset-aware. The first observation after reset emits zero
deltas and seeds the previous score for the next decision.

`peek_observation()` computes the same delta preview without updating the
extractor's stored previous score, so PPO bootstrapping does not consume the
signal before the real next policy step.

## Compatibility

This protocol is checkpoint-incompatible with 19-dim `meta_vec` checkpoints.
Use fresh checkpoints or add explicit adapter logic before attempting to load
older policy weights.

Deferred fields remain out of policy input for now:

- `action_result`, because diagnostics observed it empty.
- `alerts`, because diagnostics observed it empty.
- broader score vectors, because the first pass only needs compact progress
  and kill-value signals.
