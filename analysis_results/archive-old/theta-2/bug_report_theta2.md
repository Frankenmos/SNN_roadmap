# Bug Report: Theta-2 Training Analysis

**Date:** 2026-04-25
**Training Run:** theta-2
**Status:** Critical Issues Found

---

## Summary

Theta-2 training shows severe learning failure with 88.1% no-op domination and negative average rewards (-87.62). After analyzing the codebase, I've identified **2 critical bugs** and **3 design issues** that explain the learning failure.

---

## Critical Bug #1: Action History From Wrong Source

### Location
- `agent.py:186,232,277,279`
- `obs_space/obs_space_2.py:369-380`

### Problem
The action history is tracked from `action_space.get_last_token()` instead of from the observation stream `obs.observation.last_actions`.

**Code Flow:**
```python
# agent.py line 186
self.last_action_token = self.action_space.get_last_token()

# agent.py line 232, 277, 279  
self.last_action_token = self.action_space.get_last_token()

# obs_space_2.py line 369-380
agent_last = self._normalize_last_action_token(last_action_token)
# This gets added to meta_vec as "attempted action"
```

### Why This Is Wrong
1. **What gets stored:** The last action the agent *attempted* to dispatch
2. **What should be stored:** The last action from `obs.observation.last_actions` (what the game *actually executed*)

According to `action_history_bridge_plan.md`:
- The bridge should contain: `attempted action [type, x_norm, y_norm, extra]` plus feedback fields
- Feedback should include: `last_any_action_executed`, `last_smart_executed`

### Current Behavior
- `agent_last` always contains what the policy tried to do
- It doesn't contain whether the game actually executed it
- The observation has `obs.observation.last_actions` which has the *actual* executed action
- But we're not using it for the main action token

### Impact
- The policy sees its own attempted action as history, not what actually happened
- This breaks the feedback loop where the policy should learn "I tried X but the game did Y"
- Makes it harder to learn from execution failures

### Fix Required
The action history token should come from the observation stream, not from action_space:
1. Read `obs.observation.last_actions` in the extractor
2. Build the token from what actually executed
3. Keep the attempted action separate for debugging

---

## Critical Bug #2: Reward Function Coefficients Causing No-Op Dominance

### Location
- `config.yaml:13` - `reward_scale: 0.1`
- `config.yaml:17-22` - Reward coefficients

### Problem
The reward coefficients are misconfigured, making no-op the optimal policy.

**Analysis of Reward Coefficients:**
```yaml
reward_scale: 0.1              # All rewards scaled down by 10x
damage_dealt_coef: 0.10        # Very low reward for dealing damage
damage_taken_coef: 0.15        # HIGHER penalty for taking damage
kill_reward_coef: 30.0         # But scaled down to 3.0 effective
step_penalty: 0.02             # Small per-step penalty
```

### Why This Causes No-Op Dominance
1. **Step penalty accumulates:** -0.02 per step * 134 avg steps = -2.68 per episode
2. **Damage dealt is weak:** 0.10 * damage, then scaled by 0.1 = 0.01 * damage
3. **Damage taken hurts more:** 0.15 * damage, then scaled by 0.1 = 0.015 * damage
4. **Kill reward is rare:** Only on kills, and scaled down to 3.0 effective

**Net effect:** Any action that risks taking damage is heavily punished. The safest policy is to do nothing.

### Evidence from Theta-2 Results
```
avg_health_reward:        -0.434  # Agent taking damage
avg_engagement_reward:     0.010  # Almost no damage dealt
avg_score_reward:          0.0002 # Almost no kills
avg_end_of_episode_reward: -0.223 # More losses than wins
no-op domination:          88.1%
```

### Fix Required
1. Remove or reduce `reward_scale` (currently 0.1 is too aggressive)
2. Balance `damage_dealt_coef` vs `damage_taken_coef` (currently favors passivity)
3. Consider if step_penalty is needed at all

---

## Design Issue #1: Reward Function Has Wrong Terminal Detection

### Location
- `agent_core/rewards/defeat_roaches_v3.py:195-201`

### Problem
Terminal rewards use `current_enemy_count == 0` for win detection instead of `obs.reward > 0`.

```python
if obs.last():
    if current_enemy_count == 0:  # This check
        total_reward += self.win_reward
```

### Why This Matters
PySC2's `obs.reward` is the ground truth for episode outcome. Using unit count as a proxy:
- May miss true win conditions
- May incorrectly reward partial clears
- Doesn't match the environment's actual win signal

---

## Design Issue #2: TBPTT Window vs Action History Mismatch

### Location
- `config.yaml:11` - `tbptt_window: 128`

### Observation
The TBPTT window is 128 steps, but action history is only 1 step (the last action).

For DefeatRoaches with ~134 steps per episode:
- TBPTT can carry state across ~1 episode
- But action history only sees the last attempted action
- This temporal mismatch limits learning of action sequences

### Note
This is not a bug per se, but a design limitation documented in `action_history_bridge_plan.md`:
> "Deferred fields remain out of policy input for now: broader action-history token groups"

---

## Design Issue #3: Missing Action History Wrapper

### Location
- `Utility/feedback_diagnostics_wrapper.py` - Has `LastActionDiagnosticsWrapper`

### Observation
There's a `LastActionDiagnosticsWrapper` that logs `obs.observation.last_actions`, but it's only for diagnostics.

The codebase lacks a proper action-history wrapper that:
1. Tracks the stream of executed actions from observations
2. Provides this to the policy as temporal context
3. Matches the action_history_bridge_plan specification

Currently, action history is a "best effort" reconstruction from:
- `action_space.get_last_token()` (attempted action)
- `obs.observation.last_actions` (only in diagnostics)

---

## TBPTT Logic Review

### Status: IMPLEMENTED CORRECTLY

The TBPTT implementation in `agent_core/ppo_trainer.py` is correct:

1. **Chunk building** (`_build_tbptt_chunks`): Properly splits rollouts at episode boundaries
2. **State carry** (`_replay_packed_chunk_group`): Carries `next_state -> state_in` inside training graph
3. **Done boundaries** (`_reset_replay_state_rows`): Resets state on terminal
4. **Helper steps** (`sample_mask`): Masks non-learnable steps

The TBPTT is **not** the cause of learning failure.

---

## Recommendations

### Immediate Fixes (Critical)
1. **Fix reward scaling:** Set `reward_scale: 1.0` or remove it entirely
2. **Balance damage coefficients:** Set `damage_dealt_coef > damage_taken_coef`
3. **Consider removing step_penalty** or making it much smaller

### Action History Fix (Important)
1. Create an `action_history_wrapper.py` that tracks `obs.observation.last_actions`
2. Modify `agent.py` to read action history from the wrapper, not `action_space`
3. Store both "attempted" and "executed" action history

### Reward Function Cleanup
1. Use `obs.reward > 0` for win detection
2. Consider simpler reward structure before adding complexity

---

## Conclusion

The primary cause of theta-2's learning failure is **the reward function configuration**, not TBPTT or architecture issues. The 88.1% no-op domination is a rational response to a reward structure that heavily penalizes risk-taking.

The action history tracking issue is a secondary problem that limits learning efficiency but is not the primary cause of failure.

**Priority:** Fix reward coefficients first, then address action history tracking.
