# SNN-PPO Architecture

> Archived 2026-06-26.
>
> This was the long architecture reference from before the V6 fine skip,
> bf16/Ray-eval/reward repairs, and docs cleanup. The live concise reference is
> now `docs/current/ARCHITECTURE.md`.

**Last Updated:** 2026-05-06
**Status:** Active Development

> **Stream Token Migration Status**
>
> ✅ Token stream docs show 95 tokens, with action feedback at index 93 and meta at index 94.
> ✅ Token type docs show ACTION_FEEDBACK=3, META=4, and TOKEN_TYPE_GROUPS=5.
> ✅ PolicyInputBatch docs include `action_feedback_tokens [B, 1, 12]`, `meta_vec [B, 15]`, and `state_in` as `syn/mem [B, 2, 95, 64]`.
> ✅ PolicyNetwork docs include the action-feedback encoder, the reduced meta encoder, 95-token attention/SNN flow, and config-matching SNN alpha/beta values.

This document ties together the complete architecture of the SNN+PPO DefeatRoaches agent.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Flow](#data-flow)
3. [Observation Pipeline](#observation-pipeline)
4. [Token Streams](#token-streams)
5. [Policy Network](#policy-network)
6. [Action Space & Dispatch](#action-space--dispatch)
7. [Action Feedback Tokens](#action-feedback-tokens)
8. [Reward Function](#reward-function)
9. [Training Loop (TBPTT-PPO)](#training-loop-tbptt-ppo)
10. [Fragment-Based Rollouts](#fragment-based-rollouts)
11. [Configuration](#configuration)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING LOOP                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐  │
│  │ PySC2   │────▶│ Observation  │────▶│ Policy      │────▶│ Action       │  │
│  │ Env     │     │ Extractor    │     │ Network     │     │ Space        │  │
│  └─────────┘     └──────────────┘     └─────────────┘     └─────────────┘  │
│       │                  │                   │                   │         │
│       │                  │                   │                   ▼         │
│       │                  │                   │            ┌─────────────┐  │
│       │                  │                   │            │ PySC2       │  │
│       │                  │                   │            │ Function    │  │
│       │                  │                   │            └─────────────┘  │
│       │                  │                   │                   │         │
│       │                  │                   │                   ▼         │
│       │                  │                   │            ┌─────────────┐  │
│       │                  │                   │            │ Reward      │  │
│       │                  │                   │            │ Function    │  │
│       │                  │                   │            └─────────────┘  │
│       │                  │                   │                   │         │
│       ▼                  ▼                   ▼                   ▼         │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                          PPO Rollout Memory                          │  │
│  │  [obs, action, log_prob, value, reward, done, state_in, masks]      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                      │                                     │
│                                      ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    TBPTT-PPO Update                                 │  │
│  │  - Build chunks (tbptt_window=128)                                  │  │
│  │  - Replay with state carry                                          │  │
│  │  - Compute GAE advantages                                            │  │
│  │  - PPO clipped objective                                             │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Single Environment Step

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Step t → t+1                                                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  PySC2 Observation                                                           │
│  │                                                                           │
│  ├── feature_screen[27, 84, 84]                                             │
│  ├── feature_units[N, 23]                                                   │
│  ├── single_select / multi_select                                           │
│  ├── player[11]                                                              │
│  ├── available_actions                                                       │
│  ├── last_actions                                                            │
│  └── score_cumulative[13]                                                    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              ObservationExtractor (obs_space/obs_space_2.py)          │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  • Spatial: CNN features → pooled to 49 tokens                       │    │
│  │  • Entity: feature_units → up to 24 tokens                          │    │
│  │  • Selection: selected units → up to 20 tokens                      │    │
│  │  • Action feedback: previous action + outcome → 1 stream token      │    │
│  │  • Meta: player + available + PySC2 last-action → 15-dim vector     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                 PolicyInputBatch (policy_protocol.py)                 │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  spatial_obs:       [B, 27, 84, 84]                                  │    │
│  │  entity_features:   [B, 24, 21]                                      │    │
│  │  entity_mask:       [B, 24]                                           │    │
│  │  selection_features:[B, 20, 7]                                       │    │
│  │  selection_mask:    [B, 20]                                           │    │
│  │  action_feedback_tokens: [B, 1, 12]                                  │    │
│  │  meta_vec:          [B, 15]                                           │    │
│  │  state_in:          (syn[B, 2, 95, 64], mem[B, 2, 95, 64])           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              PolicyNetwork (spiking_policy.py)                        │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  1. Token encoders (spatial/entity/selection/action_feedback/meta)     │    │
│  │  2. Token type embeddings                                             │    │
│  │  3. SDPA self-attention (all tokens)                                  │    │
│  │  4. Fast + slow token-temporal SNN pathways                           │    │
│  │  5. Action head (3-way: NO_OP, LEFT_CLICK, RIGHT_CLICK)              │    │
│  │  6. Target head (token-pointer or coarse-to-fine)                    │    │
│  │  7. Value head (scalar critic)                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ActionSample {action_id, x, y, log_prob, value, next_state}               │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              ActionSpace (action_space/action_space.py)               │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  RIGHT_CLICK → Smart_screen(x, y)                                    │    │
│  │  LEFT_CLICK → (masked unavailable)                                   │    │
│  │  NO_OP → no_op()                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  PySC2 Function Call                                                       │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   RewardFunctionV4                                   │    │
│  │  V3 combat shaping + Smart/no-op action guidance                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Observation Pipeline

### Input: PySC2 Observation

| Field | Shape | Meaning |
|-------|-------|---------|
| `feature_screen` | [27, 84, 84] | Screen features (height map, visibility, etc.) |
| `feature_units` | [N, 23] | All visible units with attributes |
| `single_select` | [1, 7] or empty | Currently selected single unit |
| `multi_select` | [K, 7] or empty | Currently selected multiple units |
| `player` | [11+] | Player statistics (minerals, supply, etc.) |
| `available_actions` | List[int] | Function IDs currently available |
| `last_actions` | List[int] | Function IDs executed last step |
| `score_cumulative` | [13] | Cumulative score breakdown |
| `game_loop` | [1] | Game time in loops |

### Output: PolicyInputBatch

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PolicyInputBatch                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ spatial_obs    │  │ entity_features│  │selection_feat  │                 │
│  │ [B, 27, 84, 84]│  │   [B, 24, 21]  │  │   [B, 20, 7]   │                 │
│  │                │  │                │  │                │                 │
│  │ CNN features   │  │ feature_units  │  │ selected units │                 │
│  │ from screen    │  │ (padded to 24) │  │ (padded to 20) │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ entity_mask    │  │ selection_mask │  │   meta_vec     │                 │
│  │    [B, 24]     │  │    [B, 20]     │  │    [B, 15]     │                 │
│  │                │  │                │  │                │                 │
│  │ which valid    │  │ which valid    │  │ player[11]     │                 │
│  │                │  │                │  │ avail[3]      │                 │
│  │                │  │                │  │ last_idx[1]   │                 │
│  │                │  │                │  │ (bridge moved  │                 │
│  │                │  │                │  │  to stream)    │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│                                                                              │
│  ┌────────────────┐                                                           │
│  │    state_in    │  SNN recurrent state (optional)                          │
│  │ (syn, mem)     │  syn/mem: [B, 2, 95, 64] (pathways, tokens, dims)      │
│  └────────────────┘                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Normalization

Entity and selection features are normalized using `RunningFeatureNormalizer`:
- Tracks running mean/std online
- Normalizes: health, shields, energy, position, cooldowns, etc.
- Only normalizes after `min_count_for_normalize` samples (default: 32)

---

## Token Streams

### Token Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              TOKEN STREAM                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Index:  0─48 │ 49─72 │ 73─92 │ 93    │ 94                                  │
│          │     │      │      │       │                                     │
│          ▼     ▼      ▼      ▼       ▼                                     │
│  ┌────────────┬───────────┬──────────┬───────┬─────────────────────────────┐ │
│  │  Spatial   │  Entity  │Selection │Action │          Meta                │ │
│  │  (49)      │   (24)   │  (20)    │ (1)   │           (1)               │ │
│  │            │          │          │       │                             │ │
│  │ CNN pooled │ feature_ │ selected │feedback│ meta_vec (15-dim)          │ │
│  │ 7×7 grid  │  units   │  units   │ token │                             │ │
│  └────────────┴───────────┴──────────┴───────┴─────────────────────────────┘ │
│                                                                              │
│  Total: 95 tokens per observation                                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Token Type Embeddings

Each token gets a type embedding:
- `TOKEN_TYPE_SPATIAL = 0`
- `TOKEN_TYPE_ENTITY = 1`
- `TOKEN_TYPE_SELECTION = 2`
- `TOKEN_TYPE_ACTION_FEEDBACK = 3`
- `TOKEN_TYPE_META = 4`

`TOKEN_TYPE_GROUPS = 5` (for embedding table size)

### Temporal Processing

Each token type flows through **dual-pathway token-temporal SNN**:

```
Token → Linear → [Fast SNN] ──┐
                         ├──► Combine → Readout
Token → Linear → [Slow SNN] ─┘

Fast: α=0.55, β=0.65
Slow: α=0.92, β=0.97
Combine: mean(mode) or concatenation
```

---

## Policy Network

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          PolicyNetwork Architecture                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: PolicyInputBatch                                                     │
│    ├─ spatial_obs [B, 27, 84, 84]                                           │
│    ├─ entity_features [B, 24, 21]                                           │
│    ├─ selection_features [B, 20, 7]                                         │
│    ├─ entity/selection masks                                                 │
│    ├─ action_feedback_tokens [B, 1, 12]                                     │
│    ├─ meta_vec [B, 15]                                                      │
│    └─ state_in syn/mem [B, 2, 95, 64]                                      │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         TOKEN ENCODERS                                 │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │  Spatial: CNN(27→128→64) → flatten to 49 tokens                         │  │
│  │  Entity: Linear(21→64) per token                                        │  │
│  │  Selection: Linear(7→64) per token                                      │  │
│  │  Action Feedback: Linear(9→64) per token                               │  │
│  │  Meta: Linear(15→64)                                                    │  │
│  │  Type Embeddings: learned embeddings for 5 token types                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                 │                                            │
│                                 ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      SELF-ATTENTION (SDPA)                              │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │  Input: 95 tokens × 64 dims (49+24+20+1+1)                             │  │
│  │  QK projection: attention_embed_dim=64                                 │  │
│  │  Attention β: 0.5 (soft clamping)                                       │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                 │                                            │
│                                 ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │              DUAL PATHWAY TOKEN-TEMPORAL SNN                            │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │  Each of 95 tokens → separate SNN state                                │  │
│  │  Fast pathway: α=0.55, β=0.65                                          │  │
│  │  Slow pathway: α=0.92, β=0.97                                          │  │
│  │  Combine: temporal_combine_mode="mean"                                 │  │
│  │  Output: 95 tokens × 64 dims                                            │  │
│  │  State: syn/mem [B, 2, 95, 64]                                         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                 │                                            │
│                                 ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        READOUT HEADS                                   │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │  Pool all tokens → latent [B, 64]                                      │  │
│  │                                                                        │  │
│  │  Action Head: latent → logits [B, 3]                                  │  │
│  │    (NO_OP, LEFT_CLICK, RIGHT_CLICK)                                    │  │
│  │    Masked by available_actions                                         │  │
│  │                                                                        │  │
│  │  Target Head: conditioned on action_id                                │  │
│  │    - token_pointer: categorical over 49 spatial tokens                │  │
│  │    - coarse_to_fine: coarse grid + local fine patch                   │  │
│  │    - factorized_xy_legacy: independent x/y logits                     │  │
│  │                                                                        │  │
│  │  Value Head: latent → scalar [B, 1]                                   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                 │                                            │
│                                 ▼                                            │
│  OUTPUT: action_logits, target, value, next_state                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `num_steps` | 1 | Spike accumulation steps per forward |
| `fast_token_snn_alpha` | 0.55 | Fast SNN decay |
| `fast_token_snn_beta` | 0.65 | Fast SNN reset |
| `slow_token_snn_alpha` | 0.92 | Slow SNN decay |
| `slow_token_snn_beta` | 0.97 | Slow SNN reset |
| `attention_embed_dim` | 64 | Attention QK dimension |
| `attention_pool_size` | 7 | Spatial token grid side (7×7 = 49 tokens) |
| `spatial_head_type` | "coarse_to_fine" | Current config default; `token_pointer` and `factorized_xy` remain available |

---

## Action Space & Dispatch

### Semantic Action Vocabulary

```python
POLICY_ACTION_NO_OP = 0      # Do nothing
POLICY_ACTION_LEFT_CLICK = 1 # Currently masked unavailable
POLICY_ACTION_RIGHT_CLICK = 2 # Maps to Smart_screen
```

### Dispatch Mapping

| Policy Action | PySC2 Function | Target From | Notes |
|---------------|----------------|-------------|-------|
| NO_OP | `no_op()` | - | Always available |
| LEFT_CLICK | (masked) | - | Disabled for now |
| RIGHT_CLICK | `Smart_screen("now", [x, y])` | Target head | Main spatial action |

### Bootstrap

On episode start, `select_army` is called automatically (outside PPO memory) to ensure the marine is selected before policy-controlled actions begin.

---

## Action Feedback Tokens

### Current Layout

Action feedback is a first-class stream token. The stable meta vector remains
small and carries only player features, semantic action availability, and the
PySC2 last-action id.

`meta_vec [15]`:

| Slice | Dim | Field | Source |
|-------|-----|-------|--------|
| 0:11 | 11 | Player features | `obs.observation.player` |
| 11:14 | 3 | Available action mask | Computed |
| 14:15 | 1 | PySC2 last-action index | `obs.observation.last_actions[0]` |

`action_feedback_tokens [B, 1, 12]`:

| Offset | Field | Source |
|--------|-------|--------|
| 0 | Bridge action type | `action_space.get_last_token()` |
| 1 | Normalized x | `action_space.get_last_token()` |
| 2 | Normalized y | `action_space.get_last_token()` |
| 3 | Smart_screen executed? | `451 in last_actions` |
| 4 | Any action executed? | `len(last_actions) > 0` |
| 5 | Score delta | `score_cumulative[0]` delta |
| 6 | Killed value delta | `score_cumulative[5]` delta |
| 7 | Score penalty bit | `score_delta < 0` |
| 8 | Target near enemy | Previous Smart target within 6 screen pixels of a previous-frame enemy |
| 9 | Friendly moved toward target | Tag-pinned survivors, or tagless unchanged-count median fallback |
| 10 | Enemy health drop | Clipped alliance-summed enemy health drop, normalized by 100 |
| 11 | Friendly health drop | Clipped alliance-summed friendly health drop, normalized by 100 |

The token has its own type embedding and attends alongside spatial, entity,
selection, and meta tokens.

---

## Reward Function

### RewardFunctionV4 Components

`RewardFunctionV4` extends `RewardFunctionV3`; it keeps the V3 combat and
positioning terms, then adds small action-aware guidance for the current
`RIGHT_CLICK -> Smart_screen(x, y)` policy.

| Component | Formula | Purpose |
|-----------|---------|---------|
| `damage_dealt` | `0.15 × (prev_enemy_health - curr_enemy_health)` | Reward aggression |
| `damage_taken` | `-0.10 × (prev_agent_health - curr_agent_health)` | Penalize damage (but less than dealt reward) |
| `kill_reward` | `30.0 × enemies_killed` | Sparse kill bonus |
| `positioning` | Distance-based reward | Encourage optimal kiting range |
| `step_penalty` | `-0.005` per step (while enemies exist) | Time pressure |
| `win_reward` | `+60.0` on win | Terminal reward |
| `loss_penalty` | `-30.0` on loss | Terminal penalty |
| `smart_near_enemy_reward` | up to `+0.08` | Reward Smart clicks near visible enemies |
| `smart_far_enemy_penalty` | `-0.03` | Discourage Smart clicks far from visible enemies |
| `noop_visible_enemy_penalty` | `-0.02` | Discourage no-op while enemies and Smart are available |

### Key Design Decisions

1. **Asymmetric damage coefficients:** `dealt_coef (0.15) > taken_coef (0.10)`
   - Previously was reversed, causing passivity
   - Now incentivizes engagement over hiding

2. **Sparse kill rewards:** Large bonus per kill, encourages completing kills

3. **Positioning reward:** Band reward for being in optimal range (target_distance=9.0)

---

## Training Loop (TBPTT-PPO)

### Rollout Collection

```
For each episode:
    For each environment step:
        obs → extractor → policy_input
        policy_input + state → policy → action_sample
        action_sample → action_space → pysc2_function
        pysc2_step → next_obs + reward
        reward_function.calculate_reward(next_obs)
        Store in PPO memory:
            - observation_batch (with state_in)
            - action, move_x, move_y
            - log_prob, value
            - reward, done
            - sample_mask (0 for helper steps)
        Update agent.snn_state = action_sample.next_state

    After rollout_steps (2048) or episode end:
        Set final_next for bootstrapping
        Run PPO update
```

### TBPTT Chunk Building

```
Rollout memory (ordered steps):
    [step_0, step_1, step_2, ..., step_N-1]

Split into chunks at:
    - tbptt_window boundary (128 steps)
    - Episode boundaries (done=True)

Chunk example:
    {
        "observations": stacked obs[0:127],
        "initial_state": state_in[0],
        "actions": actions[0:127],
        "length": 127,  # may be less at episode end
        "dones": dones[0:127],
        "sample_mask": masks[0:127]
    }
```

### Replay with State Carry

```
For each chunk:
    state = chunk.initial_state
    For t in range(chunk.length):
        policy_output = policy(obs[t], state_in=state)
        state = policy_output.next_state
        if done[t]:
            state = reset_state_rows(state)
    Collect outputs over time, compute PPO losses
```

### PPO Losses

| Loss | Formula | Notes |
|------|---------|-------|
| Policy | `-mean(min(ratio×A, clip(ratio, 0.8, 1.2)×A))` | Masked by sample_mask |
| Value | `0.5 × mean((return - value)²)` | Masked by sample_mask |
| Entropy | `-0.01 × mean(entropy)` | Normalized per head |

---

## Fragment-Based Rollouts

### Why Fragments?

The current implementation uses a **fragment-based rollout memory** design to prepare for distributed training with Ray:

```
Single-process path (current):
  store_transition() → memory (list)
  finalize_fragment() → RolloutFragment
  consume_pending_fragments() → list[RolloutFragment]
  update_policy(fragments) → per-fragment GAE → PPO loss

Distributed path (future):
  RolloutActor.collect_fragment() → RolloutFragment
  Learner aggregates fragments from N actors
  update_policy(fragments) → same method as local
```

### RolloutFragment Structure

[`RolloutFragment`](../../distributed/protocol.py) carries all rollout data with its own bootstrap tail:

| Field | Shape | Purpose |
|-------|-------|---------|
| `actor_id` | scalar | Which actor produced this fragment |
| `fragment_id` | scalar | Monotonic fragment index per actor |
| `policy_version` | scalar | Learner update count when collected |
| `spatial_obs` | `[T, 27, 84, 84]` | Screen features |
| `entity_features` | `[T, 24, 21]` | Unit features |
| `entity_mask` | `[T, 24]` | Valid entity slots |
| `selection_features` | `[T, 20, 7]` | Selected units |
| `selection_mask` | `[T, 20]` | Valid selection slots |
| `action_feedback_tokens` | `[T, 1, 12]` | Previous action + outcome |
| `meta_vec` | `[T, 15]` | Player + available + last-action |
| `actions`, `rewards`, `values`, ... | `[T]` | PPO data |
| `tail_next_policy_input` | `PolicyInputBatch` | Bootstrap for GAE |
| `tail_next_snn_state` | `(syn, mem)` | SNN state after tail |
| `policy_protocol_version` | scalar | Must equal 3 |
| `policy_input_schema` | scalar | Must equal "stream_action_effect_feedback_v2" |

### Per-Fragment GAE

With multi-actor collection, each fragment needs its own bootstrap value:

```python
for fragment in fragments:
    tail_value = _bootstrap_fragment_tail_value(fragment.tail_next_policy_input)
    advantages = _compute_advantages(
        fragment.rewards,
        fragment.values,
        fragment.dones,
        tail_value,  # Fragment-specific bootstrap
    )
    returns = (advantages + fragment.values).detach()
```

### Protocol Validation

Every fragment validates protocol compatibility before training:

```python
validate_policy_protocol(
    policy_protocol_version=3,
    policy_input_schema="stream_action_effect_feedback_v2",
)
```

This prevents silent corruption from stale actors or incompatible checkpoints.

### Migration Path to Ray

**What stays the same:**
- `RolloutFragment` structure
- `PPO.update_policy(fragments)` method
- Per-fragment GAE computation
- TBPTT chunking and PPO loss

**What changes:**
- Fragment production: Actor emits directly vs `finalize_fragment()` from memory
- Fragment transport: Ray object ref vs in-memory list
- Weight distribution: Learner → actors via Ray

See [`FRAGMENT_PPO.md`](FRAGMENT_PPO.md) for complete fragment documentation.

---

## Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `lr` | 5e-5 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `clip_eps` | 0.10 | PPO clipping |
| `batch_size` | 2048 | Recurrent chunk group size / PPO minibatch size in current `config.yaml` |
| `epochs` | 4 | PPO epochs per update |
| `rollout_steps` | 2048 | Steps between updates |
| `tbptt_window` | 128 | Truncated BPTT horizon |
| `target_kl` | 0.03 | Early stop if KL exceeded |
| `reward_scale` | 1.0 | Reward scaling (was 0.1, caused issues) |

---

## Configuration

### Config Structure (`config.yaml`)

```yaml
hyperparameters:     # Training hyperparameters
  lr, gamma, clip_eps, batch_size, epochs,
  rollout_steps, tbptt_window, entropy_coef, etc.

reward:              # Reward function config
  name: "defeat_roaches_v3"
  damage_dealt_coef, damage_taken_coef, kill_reward_coef, etc.

environment:         # Environment settings
  map_name, steps_per_episode, total_episodes, etc.

model:               # Policy network config
  spatial_input_shape, vector_input_dim, action_dim,
  num_steps, token_snn_alpha/beta, attention settings, etc.
```

### Output Files (per run)

| File | Location | Content |
|------|----------|---------|
| `config.yaml` | `models/{run_name}/` | Exact config used for run |
| `effective_config.json` | `models/{run_name}/` | Resolved model/ppo configs |
| `checkpoint.pth` | `models/{run_name}/` | Model + optimizer + scheduler |
| `best_checkpoint.pth` | `models/{run_name}/` | Best eval checkpoint |
| `training_logs.db` | `models/{run_name}/` | SQLite logs (steps, updates, evals) |

---

## File Reference

| File | Purpose |
|------|---------|
| `train.py` | Training entry point |
| `eval.py` | Evaluation entry point |
| `agent.py` | DefeatRoaches agent wrapper |
| `agent_core/spiking_policy.py` | Policy network architecture |
| `agent_core/ppo_trainer.py` | PPO trainer with TBPTT |
| `agent_core/policy_protocol.py` | Protocol definitions and constants |
| `obs_space/obs_space_2.py` | Observation extraction |
| `action_space/action_space.py` | Action dispatch |
| `agent_core/rewards/defeat_roaches_v3.py` | Reward function |

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [`REPO_STATE.md`](REPO_STATE.md) | Current repo status and what's done/not done |
| [`THE_BPTT.md`](THE_BPTT.md) | TBPTT design and historical analysis |
| [`FRAGMENT_PPO.md`](FRAGMENT_PPO.md) | Fragment-based memory management |
| [`RAY_STATUS.md`](RAY_STATUS.md) | Ray implementation phases and status |
| [`ACTION_FEEDBACK_PLAN.md`](ACTION_FEEDBACK_PLAN.md) | Stream token migration plan |
| [`SPATIAL_HEADS.md`](../SPATIAL_HEADS.md) | Spatial target head comparison |

---

## Version History

| Date | Changes |
|------|---------|
| 2026-04-25 | Initial architecture documentation |
| 2026-04-24 | Action feedback bridge expanded to 24 dims |
| 2026-04-22 | Multi-timescale SNN pathways added |
| 2026-04-21 | Stage-1 TBPTT implemented |
