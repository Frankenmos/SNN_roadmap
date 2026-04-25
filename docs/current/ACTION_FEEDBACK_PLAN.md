# Action Feedback Architecture Plan

**Date:** 2026-04-25
**Status:** Proposed
**Related:** `action_history_bridge_plan.md`, `theta-2/bug_report_theta2.md`

---

## Executive Summary

After verifying bridge documentation and analyzing theta-1/theta-2 diagnostic outputs, **Gipity's assessment is correct**: the current bridge has two separate sources, but the architecture is "smelly" — action feedback lives as a compact meta side-channel while the policy is token/stream-based.

**Recommendation:** Fix reward coefficients first (urgent), then replace the meta bridge with proper stream tokens.

---

## Current Bridge Layout (Verified)

From `policy_protocol.py` and `action_history_bridge_plan.md`:

| Slice | Dim | Meaning | Source |
|-------|-----|---------|--------|
| `meta_vec[0:11]` | 11 | Player features | `obs.observation.player` |
| `meta_vec[11:14]` | 3 | Semantic available-action mask | Computed from `obs.observation.available_actions` |
| `meta_vec[14]` | 1 | PySC2 last_action index | `obs.observation.last_actions[0]` |
| `meta_vec[15:19]` | 4 | Attempted action `[type, x_norm, y_norm, extra]` | `action_space.get_last_token()` ❌ |
| `meta_vec[19]` | 1 | `last_any_action_executed` | Derived from `obs.observation.last_actions` |
| `meta_vec[20]` | 1 | `last_smart_executed` | Check if `451 in obs.observation.last_actions` |
| `meta_vec[21]` | 1 | `score_total_delta` | Delta from `obs.observation.score_cumulative[0]` |
| `meta_vec[22]` | 1 | `killed_value_delta` | Delta from `obs.observation.score_cumulative[5]` |
| `meta_vec[23]` | 1 | `score_penalty_bit` | Binary flag if score decreased |

**Total:** 24 dimensions

---

## Verification from Theta-1 Diagnostics

### last_action_diagnostics.jsonl (last 20 lines)

The wrapper **already tracks everything we need**:

```json
{
  "dispatched_action": {
    "function_id": 451,
    "function_name": "Smart_screen",
    "arguments": [[0], [45, 78]]  // ← Coordinates are here!
  },
  "current_frame": {
    "last_action_ids": [451],     // ← PySC2 feedback
    "last_action_names": ["Smart_screen"]
  },
  "feedback_summary": {
    "dispatched_function_seen_in_current_last_actions": true
  }
}
```

**Key finding:**
- `dispatched_action.arguments` contains the **actual coordinates we sent**
- `obs.observation.last_actions` only contains **function IDs** (no coordinates)
- The wrapper has both, but they're only used for diagnostics

### score_diagnostics.jsonl

```json
{
  "score_cumulative": [-7.0, 0.0, 0.0, 450.0, 0.0, 0.0, ...],
  "score_delta": [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...],
  "score_delta_named": {
    "killed_value_units": 0.0,
    "score": -1.0
  }
}
```

**Score deltas are working correctly** in the current bridge.

---

## The "Smelly" Architecture Problem

### Current State (Side-Channel Bridge)

```
┌─────────────────────────────────────────────────────────────┐
│                    Policy Input                            │
├─────────────────────────────────────────────────────────────┤
│ Spatial Tokens (49) │ Entity Tokens (24) │ Selection (20)  │
├─────────────────────────────────────────────────────────────┤
│                    meta_vec [24]                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ player[11] │ avail[3] │ last_idx[1] │ bridge[9]     │   │
│  │              │          │             │ ┌───────────┐ │   │
│  │              │          │             │ │ attempted │ │   │
│  │              │          │             │ │ +feedback │ │   │
│  │              │          │             │ └───────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

The action feedback is **glued onto the tail of meta_vec**, not part of the token stream.

### Desired State (Stream Tokens)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Policy Input                            │
├─────────────────────────────────────────────────────────────────┤
│ Spatial (49) │ Entity (24) │ Selection (20) │ ACTION_EVENTS (N)│
├─────────────────────────────────────────────────────────────────┤
│                    meta_vec [15]                               │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ player[11] │ avail[3] │ last_idx[1]                       │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

Action feedback becomes **first-class tokens** in the stream.

---

## Why Coordinates Matter (The Gipity Point)

### If we only use `obs.observation.last_actions`:
```
last_action_ids = [451]  # Smart_screen executed
```
We lose: **WHERE we clicked** (the most important credit-assignment signal)

### If we use wrapper-tracked data:
```
{
  "dispatched": { "function_id": 451, "arguments": [[0], [45, 78]] },
  "executed": [451],
  "match": true
}
```
We keep: **Both the intent and the outcome**

---

## Priority Plan

### Phase 1: Fix Reward Coefficients (URGENT)

**Why first:** The 88.1% no-op domination makes it impossible to evaluate any other changes.

**Changes to `config.yaml`:**
```yaml
hyperparameters:
  reward_scale: 1.0        # Was 0.1 — remove aggressive scaling

reward:
  damage_dealt_coef: 0.15  # Was 0.10 — increase
  damage_taken_coef: 0.10  # Was 0.15 — decrease (was higher!)
  step_penalty: 0.005      # Was 0.02 — reduce by 4x
```

**Rationale:**
- Current: `damage_taken_coef (0.15) > damage_dealt_coef (0.10)` → passivity is rational
- Fixed: `damage_dealt_coef (0.15) > damage_taken_coef (0.10)` → aggression is rewarded
- `reward_scale: 0.1` was crushing all rewards to noise level

**Keep current bridge temporarily** so we can compare before/after.

### Phase 2: Create Action Feedback Encoder

**New file:** `obs_space/action_feedback_encoder.py`

```python
class ActionFeedbackEncoder:
    """
    Encodes action feedback as event tokens.

    One token per observation containing:
    - Attempted action type + coordinates
    - Execution success (from PySC2 last_actions)
    - Score delta feedback
    """
    def encode_feedback(
        self,
        dispatched_action,      # from wrapper: {function_id, arguments}
        obs_last_actions,        # from observation: [ids]
        score_delta,             # from score_cumulative
    ) -> torch.Tensor:
        # Returns single event token with shape [feedback_dim]
        pass
```

**Protocol changes:**
1. `agent.py` stores dispatched action in context
2. Extractor reads dispatched action + observation feedback
3. Encoder creates feedback token
4. Feedback tokens added to token stream (not meta_vec)

### Phase 3: Replace Meta Bridge with Stream Tokens

**File changes:**
- `obs_space/obs_space_2.py`: Remove agent_last from meta_vec
- `policy_protocol.py`: Reduce META_VECTOR_DIM from 24 → 15
- `obs_space/obs_space_2.py`: Add `action_feedback_tokens` to PolicyInputBatch

**Meta vector becomes:**
```python
META_VECTOR_DIM = 15  # Was 24
# player[11] + available[3] + last_idx[1] = 15
# agent_last[4] + action_history[5] moved to token stream
```

**New PolicyInputBatch:**
```python
@dataclass
class PolicyInputBatch:
    spatial_obs: torch.Tensor        # [B, 27, 84, 84]
    entity_features: torch.Tensor    # [B, 24, F_unit]
    entity_mask: torch.Tensor        # [B, 24]
    selection_features: torch.Tensor # [B, 20, 7]
    selection_mask: torch.Tensor     # [B, 20]
    action_feedback_tokens: torch.Tensor  # [B, N, feedback_dim] ← NEW
    meta_vec: torch.Tensor           # [B, 15] ← reduced
    state_in: SNNState | None = None
```

### Phase 4: Update Policy Network

**Changes to `agent_core/spiking_policy.py`:**
1. Add `action_feedback_embedding` to token embeddings
2. Include feedback tokens in attention sequence
3. Update token-type embeddings for new token type

**Token sequence becomes:**
```
[spatial_tokens, entity_tokens, selection_tokens, feedback_tokens, meta_token]
```

---

## Implementation Order

| Step | Task | Files | Risk |
|------|------|-------|------|
| 1 | Fix reward coefficients | `config.yaml` | Low |
| 2 | Run short training (theta-3) | - | - |
| 3 | Create ActionFeedbackEncoder | `obs_space/action_feedback_encoder.py` | Low |
| 4 | Update PolicyInputBatch protocol | `policy_protocol.py`, `obs_space_2.py` | Medium (checkpoint incompat) |
| 5 | Update policy network | `spiking_policy.py` | Medium |
| 6 | Run comparison training (theta-4) | - | - |

---

## Open Questions

1. **Feedback token dimensionality:** How large should each feedback token be?
   - Suggested: 16-32 dims (enough to encode action_type + 2D coords + outcome)

2. **History length:** How many past actions to keep as tokens?
   - Current: 1 (in meta bridge)
   - Suggested: Start with 1, expand to 3-5 if needed

3. **Checkpoint migration:** Old checkpoints will be incompatible with META_VECTOR_DIM change
   - Need adapter or fresh training

---

## References

- `docs/current/action_history_bridge_plan.md` — Current 24-dim protocol
- `docs/current/THE_BPTT.md` — TBPTT architecture
- `analysis_results/theta-1/last_action_diagnostics.jsonl` — Wrapper output format
- `analysis_results/theta-2/bug_report_theta2.md` — Bug analysis
