# Feedback Diagnostics Note

> Archived 2026-06-26.
>
> This describes the first feedback-diagnostics investigation before the
> current protocol-3 action-effect token settled. Current protocol details live
> in `docs/current/ACTION_FEEDBACK_PLAN.md`.

Created: 2026-04-24

## Why This Exists

I agree with the repo direction here: before widening `meta_vec`, we should make the environment tell us what it actually emits after real actions. The important distinction is:

- `AvailableActionsDiagnosticsWrapper`: what was available and what action we dispatched
- `LastActionDiagnosticsWrapper`: what the next observation reports through `last_actions`, `action_result`, and `alerts`
- `ScoreDiagnosticsWrapper`: how `score_cumulative` and reward move over time

That gives us a clean evidence trail before we teach the policy to see these fields.

## New Eval Flags

```powershell
python eval.py --run_name <run_name> --best --episodes 1 --inspect_actions --inspect_last_action --inspect_score
```

Default outputs when `--run_name` is known:

- `analysis_results/<run_name>/available_actions_diagnostics.jsonl`
- `analysis_results/<run_name>/last_action_diagnostics.jsonl`
- `analysis_results/<run_name>/score_diagnostics.jsonl`

For a fuller alignment dump:

```powershell
python eval.py --run_name <run_name> --best --episodes 1 --inspect --inspect_policy_input --inspect_actions --inspect_last_action --inspect_score
```

## Observed JSONL Shape

`LastActionDiagnosticsWrapper` writes one record per logged env return:

```text
{
  event: "reset" | "step",
  episode: int,
  step: int,
  agent_index: int,
  previous_frame: {
    available_action_ids: list[int],
    last_action_ids: list[int],
    last_action_names: list[str],
    action_result: list[int],
    alerts: list[int],
    game_loop: list[int]
  },
  current_frame: {
    available_action_ids: list[int],
    last_action_ids: list[int],
    last_action_names: list[str],
    action_result: list[int],
    alerts: list[int],
    game_loop: list[int]
  },
  dispatched_action: {
    function_id: int | null,
    function_name: str | null,
    arguments: json
  },
  dispatched_action_in_previous_available: bool | null,
  feedback_summary: {
    has_last_actions: bool,
    has_action_result: bool,
    has_alerts: bool,
    dispatched_function_seen_in_current_last_actions: bool | null
  }
}
```

`ScoreDiagnosticsWrapper` writes the reward/score half:

```text
{
  event: "reset" | "step",
  episode: int,
  step: int,
  agent_index: int,
  reward: float,
  episode_reward: float,
  last: bool | null,
  game_loop: list[int],
  alerts: list[int],
  score_cumulative: list[float],
  score_cumulative_named: dict[str, float],
  score_delta: list[float] | null,
  score_delta_named: dict[str, float] | null,
  score_total: float | null,
  score_total_delta: float | null,
  score_nonzero_delta_indices: list[int]
}
```

## Timing Semantics

The wrapper records the action passed into `env.step([action_func])` and the
timestep returned by that same call.

- `dispatched_action` is what our agent sent from the previous observation.
- `current_frame.last_action_ids` is what PySC2 reports as executed since the
  previous observation.
- `previous_frame.last_action_ids` is only the previous record's feedback,
  included for alignment/debugging. It is not the new feedback for the current
  action.

Observed clean sequence:

```text
step 29: dispatched Smart_screen -> current_frame.last_action_ids = [451]
step 30: dispatched no_op        -> current_frame.last_action_ids = []
step 31: dispatched Smart_screen -> current_frame.last_action_ids = [451]
step 32: dispatched Smart_screen -> current_frame.last_action_ids = [451]
step 33: dispatched no_op        -> current_frame.last_action_ids = []
```

So `451 in current_frame.last_action_ids` is a useful "SC2 accepted/executed a
Smart command since the previous observation" bit.

## Score Timing Semantics

`score_cumulative` is part of the returned observation, so score feedback has
the same observation-boundary timing as `last_actions`.

For a transition:

```text
obs_t -> agent dispatches action_t -> env.step(...) -> obs_t+1
```

the useful score feedback is:

```text
score_delta_t+1 = obs_t+1.score_cumulative - obs_t.score_cumulative
```

That means:

- The action chosen from `obs_t` cannot see its own score effect yet.
- The next decision, chosen from `obs_t+1`, can see the previous transition's
  score delta if the extractor carries a reset-aware previous score.
- Reset records should use zero or `None` deltas; they must not inherit the last
  score from a previous episode.

Observed clean kill/progress examples from `theta-1`:

```text
step 114: score_delta[0] = +9,  score_delta[5] = +100
step 115: score_delta[0] = -1,  score_delta[5] = 0
step 126: score_delta[0] = +10, score_delta[5] = +100
step 166: score_delta[0] = +10, score_delta[3] = +450, score_delta[5] = +100
```

The `score_total_delta` field is therefore a compact "did the game score move
on the previous transition?" signal, while `killed_value_delta` is a cleaner
"did we kill something valuable?" signal in this minigame.

## Current Empirical Read

From the `theta-1` diagnostic dump on 2026-04-24:

- wrapped dispatched actions included `Smart_screen` 129 times
- PySC2 `current_frame.last_action_ids` included `Smart_screen` 133 times
- `action_result` was empty in all inspected records
- `alerts` was empty in all inspected records
- `score_cumulative[0]` moved with reward-like score changes
- `score_cumulative[5]` moved by `+100` on killed-unit-value events
- `score_cumulative[3]` moved by `+450` on total-value-unit changes

Observed useful score indices:

| Index | Name from wrapper | Observed behavior |
|-------|-------------------|-------------------|
| `0` | `score` | moved by `-1`, `+9`, or `+10`; best compact progress delta |
| `3` | `total_value_units` | moved by `+450` on some unit-value changes |
| `5` | `killed_value_units` | moved by `+100` on kill-value events |

Suggested first-pass score encoding:

| Field | Raw source | Suggested transform |
|-------|------------|---------------------|
| `score_total_delta` | `delta(score_cumulative[0])` | clip to `[-10, 10]`, divide by `10` |
| `killed_value_delta` | `delta(score_cumulative[5])` | clip to `[0, 100]`, divide by `100` |
| `score_penalty_bit` | `score_total_delta < 0` | binary `0.0` or `1.0` |

Important caveat: diagnostic JSONL files are append-only today. If an eval is
run multiple times into the same path, `episode`/`step` pairs can repeat across
invocations. Prefer a fresh output path, delete the old diagnostic file before a
new run, or add a future `run_id` / truncate mode before relying on
episode-step uniqueness.

## Historical Action-History Meta Shape

Previous `meta_vec[19]` before the 24-dim bridge expansion:

| Slice | Dim | Meaning |
|-------|-----|---------|
| `0:11` | 11 | player features |
| `11:14` | 3 | semantic available-action mask |
| `14:15` | 1 | PySC2 last-action index |
| `15:19` | 4 | agent bridge token `[type, x_norm, y_norm, extra]` |

The evidence-backed feedback extension then produced a 24-dim `meta_vec`:

| New field | Dim | Encoding |
|-----------|-----|----------|
| `last_any_action_executed` | 1 | `1.0` if `last_actions` is non-empty |
| `last_smart_executed` | 1 | `1.0` if `451 in last_actions` |
| `score_total_delta` | 1 | clipped/normalized delta of `score_cumulative[0]` |
| `killed_value_delta` | 1 | clipped/normalized delta of `score_cumulative[5]` |
| `score_penalty_bit` | 1 | `1.0` if `score_total_delta < 0` |

That 24-dim layout is now historical. The current protocol is
`POLICY_INPUT_SCHEMA = "stream_action_effect_feedback_v2"`:

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `meta_vec` | `[B, 15]` | player features, semantic action availability, PySC2 last-action index |
| `action_feedback_tokens` | `[B, 1, 12]` | attempted action, click coordinates, execution bits, score-delta feedback, and post-action effect feedback |

For the source of truth, use `agent_core/policy_protocol.py` and
`docs/current/ACTION_FEEDBACK_PLAN.md`. The old 24-dim bridge plan lives at
`docs/archive/action_history_bridge_plan.md`.

Deferred until it earns space:

- `action_result`: observed empty so far
- `alerts`: observed empty so far
- broader score vector: useful for diagnostics, probably too wide for first
  policy-input expansion

## Follow-Up Agent Prompt

Use this if a future agent needs to analyze the generated JSONL files without changing code:

```text
Inspect analysis_results/<run_name>/last_action_diagnostics.jsonl and score_diagnostics.jsonl. Summarize:
1. frequency and values of action_result
2. frequency and values of alerts
3. how often dispatched actions appear in current_frame.last_action_ids
4. which score_cumulative indices change, with example deltas
5. candidate compact encodings for future action-feedback tokens, without implementing them yet
```
