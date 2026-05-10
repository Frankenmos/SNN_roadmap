# Claude Instructions for SNN+SNN-roadmap

**Last Updated:** 2026-05-10

## Quick Rules

1. **Decompose complex tasks** - Break into smaller steps
2. **Document everything** - Update working logs as you go
3. **Ask when stuck** - Don't spin wheels on deep ambiguity
4. **No crazy instrumental solutions** - Keep it practical
5. **Tests pass before moving on** - `pytest tests -q`
6. **Read run names carefully** - V5 is a run family, not a reward class

## Repo Structure

```
agent_core/          # Core ML components
  ├─ spiking_policy.py      # SNN + attention policy
  ├─ ppo_trainer.py         # PPO with TBPTT
  ├─ target_heads.py        # Spatial target heads
  └─ policy_protocol.py     # Protocol & constants

agent.py            # Main agent orchestrator
train.py            # Training loop
config.yaml         # Hyperparameters

tests/              # Test suite
  ├─ test_PPO.py            # PPO tests
  ├─ test_agent.py          # Agent/policy tests
  └─ MockedEnv/             # Fake PySC2 for isolation

tools/analysis/     # Analysis & plotting
```

## Key Constants (from policy_protocol.py)

```python
SPATIAL_OBS_SHAPE = (27, 84, 84)      # Screen input
SPATIAL_TOKEN_COUNT = 49               # 7x7 grid
MAX_ENTITY_TOKENS = 24
MAX_SELECTION_TOKENS = 20
ACTION_FEEDBACK_TOKEN_DIM = 12         # Stream feedback token size
META_VECTOR_DIM = 15                   # Current stable meta vec size
POLICY_ACTION_DIM = 3                  # NO_OP, LEFT_CLICK, RIGHT_CLICK
```

## Current Reality Snapshot

- Current protocol: `POLICY_PROTOCOL_VERSION = 3`
- Current schema: `stream_action_effect_feedback_v2`
- Current policy input: `action_feedback_tokens [B, 1, 12]` and `meta_vec [B, 15]`
- Current reward implementation in code/config: `defeat_roaches_v4`
- Latest local run artifact: `banana_smart_v5_b2048_e4_a10`
- Important naming trap: there is no `RewardFunctionV5` in `agent_core/rewards/`.
  The V5 run uses `RewardFunctionV4` plus protocol-3 action-effect feedback.

## Action Semantics

- **0**: `NO_OP`
- **1**: `LEFT_CLICK` (masked unavailable on current wrapper)
- **2**: `RIGHT_CLICK` → `Smart_screen(x, y)`

## Spatial Target Heads

| Head Type | Positions | Status |
|-----------|-----------|--------|
| `factorized_xy` | 84×84 | Legacy |
| `token_pointer` | 49 (7×7) | ✅ Available fallback |
| `coarse_to_fine` | 7056 (7×7 × 12×12) | ✅ Current config default |
| `heatmap` | 7056 (84×84) | Future |

## Coarse-To-Fine Status

`CoarseToFineTargetHead` is implemented in `agent_core/target_heads.py`:

1. **Stage 1 (Coarse)**: 7×7 tokens → categorical → 49 cells
2. **Stage 2 (Fine)**: Selected cell → 12×12 local → 144 offsets
3. **Total**: 49 × 144 = 7056 positions

## Testing Pattern

```python
# Create mock batch
from MockedEnv.policy_batch import make_policy_batch
batch = make_policy_batch(batch_size=2, meta_dim=META_VECTOR_DIM, with_state=True)

# Create small policy for testing
net = PolicyNetwork(SPATIAL_OBS_SHAPE, META_VECTOR_DIM, POLICY_ACTION_DIM, ...)
net.device = torch.device("cpu")
net.to("cpu")
```

## Common Gotchas

1. **`self._action_dim`** - Not set until AFTER `_config` is built. Use parameter `action_dim` directly.
2. **Replay teacher-forcing** - Must use RECORDED `coarse_index` for fine-head evaluation, NOT resampled.
3. **State rank** - Must be rank-3 (legacy) or rank-4 (multi-timescale), not rank-2.
4. **Mask types** - `entity_mask`/`selection_mask` must be `torch.bool`, NOT float.
5. **V5 naming** - Do not invent `defeat_roaches_v5`. Check the run sidecar
   `effective_config.json`; V5 artifacts still report `reward.name = "defeat_roaches_v4"`.

## Run Timeline

| Run / version | Dates in local DB | Protocol | Reward | Read |
|---------------|-------------------|----------|--------|------|
| `banana_b2048_e4_a10` | 2026-04-27 -> 2026-04-28 | v2 / `stream_action_feedback_v1` | pre-action-aware V3 path | Older baseline; disappointing action behavior. |
| `banana_smart_v4_b2048_e4_a10` | 2026-05-06 | v2 / `stream_action_feedback_v1` | `defeat_roaches_v4` | Action-aware reward/curriculum run; high max reward but unstable/plateaued. |
| `banana_smart_v5_b2048_e4_a10` | 2026-05-06 -> 2026-05-10 | v3 / `stream_action_effect_feedback_v2` | `defeat_roaches_v4` | Latest artifact; worse reward read, no eval rows, late non-finite gradient warnings. |

## Update cadence

- After each significant change: Run `pytest tests -q`
- After implementation phase: Update working log
- If stuck > 15 min: Ask user for help

## Current Config Defaults

```yaml
hyperparameters:
  batch_size: 2048
  epochs: 4

model:
  spatial_head_type: "coarse_to_fine"
  tbptt_window: 128
  action_dim: 3
  vector_input_dim: 15

reward:
  name: "defeat_roaches_v4"

environment:
  run_name: "banana_smart_v4_b2048_e4_a10"  # config default; V5 was run by override
  steps_per_episode: 3600

distributed:
  num_rollout_actors: 10
  fragment_steps: 256
  global_rollout_steps: 2560
  required_policy_protocol_version: 3
  required_policy_input_schema: "stream_action_effect_feedback_v2"
```
