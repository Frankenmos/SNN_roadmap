# Action Detection Problem

Updated: 2026-05-10

## Problem Statement

The current DefeatRoaches action path uses semantic policy action
`RIGHT_CLICK`, which dispatches PySC2 `Smart_screen(x, y)`.

That creates a semantic ambiguity:

```text
Smart_screen accepted != attack succeeded
```

`obs.observation.last_actions` can tell us that PySC2 accepted function `451`
(`Smart_screen`), but `Smart_screen` remains the same function whether the
click means:

- attack a visible roach
- walk toward empty ground
- sidestep
- run to a corner
- do nothing useful while still producing valid actions

The `banana_smart_v5_b2048_e4_a10` stochastic eval exposed this failure mode:
the policy repeatedly clicked the left-bottom region, kept emitting valid
`Smart_screen` actions, and produced no score or kill progress. The action
transport worked; the learned behavior exploited a non-combat basin.

## Current Signals

Current observability already contains useful pieces, but they have different
meanings:

| Signal | Meaning | Limitation |
|--------|---------|------------|
| `451 in last_actions` | `Smart_screen` was accepted/executed | Does not distinguish attack vs move |
| Smart target near enemy | Attack intent proxy | Does not prove damage |
| Enemy health drop | Attack effect proxy | One-step timing can miss delayed effects |
| `score_cumulative[5]` increase | Kill-value progress | Sparse and delayed |
| Enemy unit count decrease | Kill success | Sparse terminal-ish signal |
| Friendly moved toward target | Move-like effect proxy | Can be good kiting or bad corner drift |
| Edge/corner target repetition | Stall/exploit proxy | Can punish legitimate kiting if used alone |

The clean conceptual split is:

```text
Smart executed = command delivery
target near enemy = attack intent
enemy health drop = attack success
kill value / enemy count drop = kill success
edge/corner target with no damage = stall exploit
```

## Option 1: One-Step Explicit Attack Effect Detector

This is closest to the current `ActionEffectTracker`.

For each transition:

```text
obs_t -> Smart_screen(x, y) -> obs_t+1
```

mark the action as attack-like when:

- previous action was `RIGHT_CLICK`
- target was near at least one previous-frame enemy
- enemy alliance-summed health decreased in `obs_t+1`

Suggested labels:

```text
smart_accepted
smart_target_near_enemy
smart_enemy_health_drop
smart_damage_like = smart_target_near_enemy && smart_enemy_health_drop
smart_null = smart_accepted && !smart_enemy_health_drop
```

Pros:

- Simple.
- Uses fields already present in `feature_units`.
- Easy to expose in `action_feedback_tokens`, diagnostics, and reward logs.
- Directly fixes the misconception that `Smart_screen` means attack.

Cons:

- One-step health drops may miss delayed attacks.
- Splash/pathing/tick timing can blur attribution.
- If multiple Smart commands are spammed, the previous click may get credit for
  damage caused by earlier positioning.

Good use:

- Immediate diagnostics.
- Low-weight reward shaping.
- PPO logging columns such as `attack_like_smart_count`,
  `null_smart_count`, and `corner_null_smart_count`.

## Option 2: Pending Smart Window Attribution

Keep a short queue of recent Smart commands and assign later damage/kill events
to the most plausible pending target.

Example state:

```text
pending_smart = [
  {step, x, y, near_enemy, nearest_enemy_distance, edge_score}
]
```

When enemy health drops or killed-value increases within `N` env steps, assign
credit to the closest recent pending Smart target near an enemy.

Possible labels:

```text
smart_damage_within_3
smart_kill_within_10
smart_null_after_5
smart_corner_null_after_5
```

Pros:

- More faithful to SC2 timing.
- Can detect attack success even when damage appears a few observations later.
- Reduces false negatives from one-step detector.

Cons:

- More stateful.
- Attribution is heuristic when commands are spammed every step.
- Needs reset-safe handling and tests around episode boundaries.

Good use:

- Main explicit attack detector if we want robust labels.
- Better reward shaping than one-step only.
- Eval trace summaries that answer "did this click eventually matter?"

## Option 3: Intent-Effect Classifier Over Smart Commands

Classify each Smart command into behavioral buckets:

```text
attack_like      = target near enemy and damage/kill soon
engage_like      = target near enemy but no damage yet
move_like        = friendlies moved toward target, target not near enemy
kite_like        = moved away from enemy while maintaining enemy proximity
stall_like       = edge/corner target, no damage, no kill progress
null_or_unclear  = no clear effect
```

This extends the current `classify_action_effect()` idea beyond
`move_and_damage`, `move_like`, `damage_like`, and `null_or_unclear`.

Pros:

- Produces interpretable diagnostics.
- Lets reward code avoid crude "corner bad" logic.
- Can separate useful kiting from empty corner running.

Cons:

- More design choices: radius, window, edge thresholds, kiting criteria.
- Labels are still heuristics, not ground truth.

Good use:

- Dashboard/eval diagnosis.
- Reward shaping with different weights per behavior class.
- Training regression checks after action-space changes.

## Option 4: Direct Reward Bolt-On For Smart Target Quality

This is the current `RewardFunctionV4` style, but stricter.

Reward/penalize the action immediately from the chosen target:

```text
if Smart target near visible enemy:
    +small_reward
else if Smart target far from all visible enemies:
    -small_penalty
```

Potential additions:

- Scale reward by inverse distance to nearest enemy.
- Penalize when target is near screen edge and far from enemies.
- Increase penalty if repeated far Smart targets occur while enemies are alive.

Pros:

- Fast to implement.
- Dense signal.
- Helps the spatial head learn "click near enemies" even before kills happen.

Cons:

- This is intent shaping, not success detection.
- Can teach clicking near roaches without learning good fighting.
- Bad thresholds can prevent legitimate repositioning.

Good use:

- Short stabilization branch.
- Early curriculum / warmup for spatial target quality.

## Option 5: Null-Smart Penalty With Damage Gate

Penalize Smart commands that produce no useful effect after a short window.

Example:

```text
if enemies_alive
and Smart_screen accepted
and no enemy_health_drop within N steps
and no killed_value increase within N steps:
    penalty
```

Optional stronger condition:

```text
if target far from enemies:
    larger penalty
```

Pros:

- Attacks the actual exploit: valid Smart spam without combat progress.
- Less brittle than hard-coding bottom-left.
- Does not punish edge movement if it produces damage soon after.

Cons:

- Delayed credit/penalty is more complex to wire into per-step rewards.
- Needs careful magnitude so exploration is not crushed.
- Can punish setup movement before a later attack if the window is too short.

Good use:

- Reward stabilization once explicit window detector exists.
- Anti-stall shaping that generalizes beyond one corner.

## Option 6: Edge/Corner Guardrail

Add a small penalty for repeated nonproductive Smart targets near screen edges or
corners.

Example:

```text
edge = x <= margin or x >= 83 - margin or y <= margin or y >= 83 - margin
corner = x near edge_x and y near edge_y

if enemies_alive
and Smart_screen accepted
and edge_or_corner
and no recent enemy damage
and repeated similar target region:
    penalty
```

Pros:

- Directly targets the observed left-bottom strategy.
- Easy to log and understand.
- Useful as a guardrail after the explicit detector says "nonproductive."

Cons:

- Brittle if used alone.
- Could punish legitimate kiting against map boundaries.
- The policy may find a different non-corner stall pattern.

Good use:

- Low-weight secondary guardrail.
- Diagnostics counter, not primary success definition.

Bad use:

- Do not make `if bottom_left: punish` the main fix. That treats the symptom,
  not the semantic problem.

## Option 7: Curriculum Or Environment Split

Teach Smart semantics in easier stages before full DefeatRoaches pressure:

- stationary enemy click task
- harmless roaches
- constrained enemy movement
- explicit "click near enemy" shaping phase
- then full DefeatRoaches

Pros:

- Avoids asking PPO to discover target semantics under full survival pressure.
- Makes spatial-head debugging cleaner.
- Can compare `coarse_to_fine` vs `token_pointer` without combat chaos.

Cons:

- More environment work.
- Does not replace real full-task reward correctness.
- Needs transfer checks so the policy does not overfit curriculum maps.

Good use:

- If reward bolt-ons still produce edge/stall strategies.
- If target-head quality remains ambiguous under full task pressure.

## Recommended Path

The best next branch should combine one explicit detector with one conservative
reward guardrail.

### Stage 1: Instrument Explicit Labels

Add diagnostic and PPO counters for:

```text
smart_accepted_count
smart_target_near_enemy_count
smart_enemy_health_drop_count
smart_damage_like_count
smart_null_count
smart_edge_target_count
smart_corner_null_count
```

This should be mostly read-only instrumentation first. The goal is to see
whether future runs are actually attacking or merely issuing valid Smart
commands.

### Stage 2: Reward Attack Effects

Add small positive reward for:

```text
Smart target near enemy + enemy health drop soon
```

Keep or tune existing direct target-near-enemy shaping, but do not let it stand
alone as "attack success."

### Stage 3: Penalize Nonproductive Smart

Add a low-to-moderate penalty for:

```text
enemies alive
Smart accepted
target far from enemies
no enemy health drop / kill progress within short window
```

### Stage 4: Add Corner Guardrail Only Behind Nonproductivity

Use edge/corner penalties only when the detector already says the behavior is
nonproductive:

```text
edge/corner + repeated + no damage = extra penalty
```

Do not punish all edge movement. Real kiting can touch edges.

## Implementation Notes

The clean implementation boundary is probably:

- `obs_space/action_effects.py`
  - extend effect tracking and classification
  - optionally add pending Smart attribution
- `obs_space/action_feedback_encoder.py`
  - expose stable compact feedback bits only if protocol changes are intended
- `agent_core/rewards/defeat_roaches_v4.py` or a new `v5`
  - use detector outputs for shaping
  - keep reward magnitudes small until verified
- `Utility/*diagnostics_wrapper.py`
  - log explicit labels and counters
- `tools/analysis/results.py` / dashboard
  - summarize null Smart, attack-like Smart, and corner-null Smart rates

Protocol caution:

- Adding new `action_feedback_tokens` dimensions changes checkpoint
  compatibility.
- If we only need reward shaping and diagnostics, prefer adding detector state
  inside reward/diagnostics first, without changing `POLICY_INPUT_SCHEMA`.

## Evaluation Criteria

A healthier run should show:

- high `Smart_screen` acceptance remains
- target distribution is not dominated by one corner
- `smart_target_near_enemy_rate` rises
- `smart_enemy_health_drop_rate` rises
- `smart_null_rate` falls
- killed-value or enemy-count progress appears in eval
- deterministic and stochastic eval do not collapse to the same edge strategy

For the specific left-bottom exploit, the important test is not "does it stop
clicking left-bottom?" The important test is:

```text
When it clicks left-bottom, does that produce damage or tactical value?
```

If not, it should become expensive.
