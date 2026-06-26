# Agent Instructions For This Repo

Updated: 2026-06-26

## Read First

1. `docs/current/REPO_STATE.md`
2. `docs/current/V5_COLLAPSE_AUDIT.md`
3. `docs/current/ARCHITECTURE.md`
4. `docs/SPATIAL_HEADS.md`

## Current Reality

- Protocol: `POLICY_PROTOCOL_VERSION = 3`
- Schema: `stream_action_effect_feedback_v2`
- Policy input: `action_feedback_tokens [B, 1, 12]` and `meta_vec [B, 15]`
- Reward: `defeat_roaches_v4`
- Current config run family: `banana_glasses_v6_b2048_e4_a10`
- V5 run family: `banana_smart_v5_b2048_e4_a10`, collapse artifact
- There is no `RewardFunctionV5`
- Old CNN/PPO kiting notes are historical, not current controls

## Key Constants

From `agent_core/policy_protocol.py`:

```python
SPATIAL_TOKEN_COUNT = 49
MAX_ENTITY_TOKENS = 24
MAX_SELECTION_TOKENS = 20
ACTION_FEEDBACK_TOKEN_DIM = 12
META_VECTOR_DIM = 15
POLICY_ACTION_DIM = 3
```

## Action Semantics

- `0`: `NO_OP`
- `1`: `LEFT_CLICK`, scaffolded but masked unavailable in DefeatRoaches
- `2`: `RIGHT_CLICK -> Smart_screen(x, y)`

## Spatial Head

Current default:

```yaml
model:
  spatial_head_type: "coarse_to_fine"
  fine_skip_connection: true
```

V5 failed with a no-skip fine stage. Deterministic V5 eval used one constant
fine sub-index for all 1,099 Smart clicks. Do not diagnose current code as
"missing positional embeddings"; V5 already had learned 2D positional encoding.

## Common Gotchas

- V5/V6 are run names, not reward versions.
- Pre-V6 checkpoints cannot load while `fine_skip_connection: true`.
- `policy_input_diagnostics*.jsonl` from old V5 analysis cannot be trusted for
  action-effect attribution when it was produced by a local re-extractor with
  no real `last_action_token`; prefer eval traces or direct action diagnostics.
- Entity recurrent carry is intentionally off until entity identity is pinned.
- `num_steps` is SNN micro-steps inside a policy forward, not environment
  rollout steps.
- `PPO_CNN/` and root `PPO_CNN_*` runtime surfaces are gone.

## Run Ledger

| Run | Status |
| --- | --- |
| `banana_b2048_e4_a10` | Historical pre-action-aware/protocol-v2 run |
| `banana_smart_v4_b2048_e4_a10` | Historical protocol-v2 reward-v4 run |
| `banana_smart_v5_b2048_e4_a10` | Collapse artifact: max reward 0, constant fine index |
| `banana_glasses_v6_b2048_e4_a10` | Post-fine-skip/glasses family; positive training reward, deterministic behavior still under review |

## Preferred Commands

```powershell
python train.py
python eval.py --run_name <run> --best --episodes 5
python -m distributed.ray_train --num-actors 10 --run-name <run>
python results.py --run-name <run> --report --aismart
pytest tests -q
```
