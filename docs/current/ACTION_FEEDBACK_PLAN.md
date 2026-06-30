# Action Feedback Protocol

Updated: 2026-06-26

This is the current action-feedback source of truth. It supersedes older
action-history bridge plans.

## Current Contract

- Protocol: `POLICY_PROTOCOL_VERSION = 3`
- Schema: `POLICY_INPUT_SCHEMA = "stream_action_effect_feedback_v2"`
- Policy input carries `action_feedback_tokens [B, 1, 12]`
- Stable `meta_vec [B, 15]` carries only player features, semantic action
  availability, and the PySC2 last-action index

The current 95-token stream is:

```text
49 spatial + 24 entity + 20 selection + 1 action_feedback + 1 meta
```

## Token Fields

| Offset | Field | Meaning |
| --- | --- | --- |
| 0 | `bridge_action_type` | Previous semantic action from `ActionSpace` |
| 1 | `x_norm` | Previous target x normalized to screen |
| 2 | `y_norm` | Previous target y normalized to screen |
| 3 | `executed_smart` | `451 in obs.observation.last_actions` |
| 4 | `any_executed` | Any PySC2 last action was reported |
| 5 | `score_delta_norm` | Score delta, clipped/normalized |
| 6 | `killed_value_delta_norm` | Killed-unit-value delta |
| 7 | `score_penalty_bit` | Score delta was negative |
| 8 | `target_near_enemy` | Previous Smart target near a previous-frame enemy |
| 9 | `friendly_moved_toward_target` | Friendly motion toward the previous target |
| 10 | `enemy_health_drop_norm` | Enemy health drop after the action |
| 11 | `friendly_health_drop_norm` | Friendly health drop after the action |

`peek_observation()` must not advance feedback tracker state. `reset()` clears
the tracker.

## What Is Historical

- The 24-dim `meta_vec` bridge is obsolete.
- The 9-field feedback bridge inside `meta_vec` is obsolete.
- V5/V6 are run-family names, not reward-function versions.

## Files

- Protocol constants: `agent_core/policy_protocol.py`
- Extraction and feedback assembly: `obs_space/obs_space_2.py`
- Action-effect helpers: `obs_space/action_effects.py`
- Smart outcome detector used by reward v4: `obs_space/smart_outcome_detector.py`
- Reward consumer: `agent_core/rewards/defeat_roaches_v4.py`
