# SNN-PPO Architecture

**Last Updated:** 2026-04-25
**Status:** Active Development

This document ties together the complete architecture of the SNN+PPO DefeatRoaches agent.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Flow](#data-flow)
3. [Observation Pipeline](#observation-pipeline)
4. [Token Streams](#token-streams)
5. [Policy Network](#policy-network)
6. [Action Space & Dispatch](#action-space--dispatch)
7. [Action Feedback Bridge](#action-feedback-bridge)
8. [Reward Function](#reward-function)
9. [Training Loop (TBPTT-PPO)](#training-loop-tbptt-ppo)
10. [Configuration](#configuration)

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
│  │  • Meta: player + available + bridge[9] → 24-dim vector             │    │
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
│  │  meta_vec:          [B, 24] ← contains action feedback bridge         │    │
│  │  state_in:          (syn[B, 94, 2, 64], mem[B, 94, 2, 64])           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              PolicyNetwork (spiking_policy.py)                        │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  1. Token encoders (spatial/entity/selection/meta)                    │    │
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
│  │                   RewardFunctionV3                                   │    │
│  │  damage_dealt, damage_taken, kills, positioning, terminal            │    │
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
│  │    [B, 24]     │  │    [B, 20]     │  │    [B, 24]     │                 │
│  │                │  │                │  │                │                 │
│  │ which valid    │  │ which valid    │  │ player[11]     │                 │
│  │                │  │                │  │ avail[3]      │                 │
│  │                │  │                │  │ last_idx[1]   │                 │
│  │                │  │                │  │ bridge[9]     │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│                                                                              │
│  ┌────────────────┐                                                           │
│  │    state_in    │  SNN recurrent state (optional)                          │
│  │ (syn, mem)     │  syn: [B, 94, 2, 64], mem: [B, 94, 2, 64]               │
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
│  Index:  0─48 │ 49─72 │ 73─92 │ 93                                           │
│          │     │      │      │                                              │
│          ▼     ▼      ▼      ▼                                              │
│  ┌────────────┬───────────┬──────────┬─────────────────────────────────────┐ │
│  │  Spatial   │  Entity  │Selection │              Meta                    │ │
│  │  (49)      │   (24)   │  (20)    │              (1)                     │ │
│  │            │          │          │                                      │ │
│  │ CNN pooled │ feature_ │ selected │ meta_vec (24-dim)                    │ │
│  │ 7×7 grid  │  units   │  units   │                                      │ │
│  └────────────┴───────────┴──────────┴─────────────────────────────────────┘ │
│                                                                              │
│  Total: 94 tokens per observation                                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Token Type Embeddings

Each token gets a type embedding:
- `SPATIAL_TOKEN = 0`
- `ENTITY_TOKEN = 1`
- `SELECTION_TOKEN = 2`
- `META_TOKEN = 3`

Future: `ACTION_FEEDBACK_TOKEN = 4` (planned)

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
│    ├─ meta_vec [B, 24]                                                      │
│    └─ state_in (syn, mem)                                                   │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         TOKEN ENCODERS                                 │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │  Spatial: CNN(27→128→64) → flatten to 49 tokens                         │  │
│  │  Entity: Linear(21→64) per token                                        │  │
│  │  Selection: Linear(7→64) per token                                      │  │
│  │  Meta: Linear(24→64)                                                    │  │
│  │  Type Embeddings: learned embeddings for token types                   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                 │                                            │
│                                 ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      SELF-ATTENTION (SDPA)                              │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │  Input: 94 tokens × 64 dims                                             │  │
│  │  QK projection: attention_embed_dim=64                                 │  │
│  │  Pooling: attention_pool_size=7 (outputs 7 pooled tokens)               │  │
│  │  Attention β: 0.5 (soft clamping)                                       │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                 │                                            │
│                                 ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │              DUAL PATHWAY TOKEN-TEMPORAL SNN                            │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │  Each of 94 pooled tokens → separate SNN state                         │  │
│  │  Fast pathway: α=0.55, β=0.65                                          │  │
│  │  Slow pathway: α=0.92, β=0.97                                          │  │
│  │  Combine: temporal_combine_mode="mean"                                 │  │
│  │  Output: 94 tokens × 64 dims                                            │  │
│  │  State: syn[94, 2, 64], mem[94, 2, 64] per pathway                     │  │
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
| `attention_pool_size` | 7 | Output pooled tokens |
| `spatial_head_type` | "coarse_to_fine" \| "token_pointer" | Target head architecture |

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

## Action Feedback Bridge

### Current Layout (24-dim meta_vec)

| Slice | Dim | Field | Source |
|-------|-----|-------|--------|
| 0:11 | 11 | Player features | `obs.observation.player` |
| 11:14 | 3 | Available action mask | Computed |
| 14:15 | 1 | PySC2 last_action index | `obs.observation.last_actions[0]` |
| 15:19 | 4 | **Attempted action** | `action_space.get_last_token()` |
| 19:20 | 1 | Any action executed? | `len(last_actions) > 0` |
| 20:21 | 1 | Smart_screen executed? | `451 in last_actions` |
| 21:22 | 1 | Score delta | `score_cumulative[0]` delta |
| 22:23 | 1 | Killed value delta | `score_cumulative[5]` delta |
| 23:24 | 1 | Score penalty bit | `score_delta < 0` |

### The "Smell" Problem

The action feedback lives in meta_vec as a **side-channel**, not as proper stream tokens.

```
Current:  meta_vec[15:24] = 9 dims of feedback glued onto meta
Desired:  action_feedback_tokens[N, ~9] in the token stream
```

### Planned Stream Token Architecture

See [`ACTION_FEEDBACK_PLAN.md`](ACTION_FEEDBACK_PLAN.md) for details.

Proposed action feedback token:

```
[ACTION_EVENT, bridge_type, x_norm, y_norm, executed_smart,
 any_executed, score_delta, kill_delta, penalty_bit]
```

This would become a first-class token type with its own embedding, attendable alongside spatial/entity/selection tokens.

---

## Reward Function

### RewardFunctionV3 Components

| Component | Formula | Purpose |
|-----------|---------|---------|
| `damage_dealt` | `0.15 × (prev_enemy_health - curr_enemy_health)` | Reward aggression |
| `damage_taken` | `-0.10 × (prev_agent_health - curr_agent_health)` | Penalize damage (but less than dealt reward) |
| `kill_reward` | `30.0 × enemies_killed` | Sparse kill bonus |
| `positioning` | Distance-based reward | Encourage optimal kiting range |
| `step_penalty` | `-0.005` per step (while enemies exist) | Time pressure |
| `win_reward` | `+60.0` on win | Terminal reward |
| `loss_penalty` | `-30.0` on loss | Terminal penalty |

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

### Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `lr` | 5e-5 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `clip_eps` | 0.10 | PPO clipping |
| `batch_size` | 128 | Minibatch size |
| `epochs` | 8 | PPO epochs per update |
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
| [`action_history_bridge_plan.md`](action_history_bridge_plan.md) | 24-dim bridge protocol |
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
