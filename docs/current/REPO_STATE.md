# Repo State

Updated: 2026-06-26

This is the primary source of truth for the live code. Older architecture
reviews and CNN/PPO-era run narratives live in `docs/archive/`.

## Current Stack

- Task: PySC2 `DefeatRoaches`
- Policy: hybrid CNN + token stream + spiking attention + dual-timescale token
  SNN
- PPO: fragment-based PPO with per-fragment GAE and ordered TBPTT replay
- Protocol: `POLICY_PROTOCOL_VERSION = 3`
- Schema: `stream_action_effect_feedback_v2`
- Action vocab: `NO_OP`, `LEFT_CLICK`, `RIGHT_CLICK`
- Live dispatch: `RIGHT_CLICK -> Smart_screen(x, y)`
- `LEFT_CLICK`: scaffolded but masked unavailable in the DefeatRoaches wrapper
- Spatial target head: `coarse_to_fine`
- Fine-stage repair: `fine_skip_connection: true`
- AMP: `bf16` by default on CUDA
- Reward: `defeat_roaches_v4`
- Distributed path: synchronous Ray rollout actors plus one learner

## Policy Input

The live policy input is:

```text
spatial_obs [B, 27, 84, 84]
entity_features [B, 24, F] + entity_mask
selection_features [B, 20, 7] + selection_mask
action_feedback_tokens [B, 1, 12]
meta_vec [B, 15]
state_in = (syn, mem)
```

The token stream has 95 tokens:

```text
49 spatial + 24 entity + 20 selection + 1 action_feedback + 1 meta
```

Spatial tokens have explicit learned 2D positional encoding. The target head
also keeps structured spatial context, so the old pooled-latent-only click
head is historical.

## Current Spatial Head

`coarse_to_fine` predicts one 7x7 coarse cell and then one 12x12 local offset.

V5 proved that the original fine stage was spatially blind: deterministic eval
used 7 coarse cells but the fine sub-index was the constant `10` for all 1,099
Smart clicks. Current code fixes that with a fine skip connection from pre-pool
84x84 conv features into the fine logits.

See:

- `docs/current/V5_COLLAPSE_AUDIT.md`
- `docs/SPATIAL_HEADS.md`

## Current Run Ledger

| Run | Read |
| --- | --- |
| `banana_b2048_e4_a10` | Historical pre-action-aware baseline. Do not use as a current architecture control. |
| `banana_smart_v4_b2048_e4_a10` | Historical action-aware reward/protocol-v2 run; better headline reward than V5 but unstable. |
| `banana_smart_v5_b2048_e4_a10` | Collapse artifact: 11,447 episodes, max reward `0.00`, no eval rows, fp16, no fine skip. |
| `banana_glasses_v6_b2048_e4_a10` | Post-fine-skip/glasses family; training reward became positive, max reward `555.85`, deterministic eval still needs scrutiny. |

Old CNN/PPO-era kiting narratives should not be treated as clean current
controls for V5/SNN architecture decisions.

## What Is Done

- Hybrid observation tokenization.
- Stream action-effect feedback token.
- Semantic action vocabulary with `RIGHT_CLICK -> Smart_screen`.
- Reset bootstrap outside PPO memory.
- Fragment-based rollout protocol.
- Ordered TBPTT replay with stored recurrent states.
- Masked critic semantics.
- Time cap stored as truncation rather than terminal `done`.
- Coarse-to-fine target head with teacher-forced replay evaluation.
- Fine skip connection for observation-dependent fine logits.
- Ray rollout/learner path.
- Ray deterministic eval and best-checkpoint plumbing.
- Extractor normalizer merge before Ray best-checkpoint save.
- Reward v4 with score-delta kill credit, corrected kiting-distance defaults,
  and Smart outcome shaping.
- bf16 AMP default.
- Repo cleanup removed old `PPO_CNN/` runtime surfaces.

## What Is Still Open

- Deterministic behavior after V6/V7 fixes still needs trace-level validation.
- Entity identity is not pinned; entity recurrent carry remains intentionally
  disabled.
- Selection actions and a broader action vocabulary are not implemented.
- `LEFT_CLICK` remains masked until there is a real no-alias purpose for it.
- Dedicated Ray eval actors are not implemented; eval borrows rollout actors.
- Step-level Ray logging is still thinner than the single-process logger.
- Full-game StarCraft is out of scope for this branch.

## Current Entrypoints

```powershell
python train.py
python eval.py --run_name <run> --best --episodes 5
python -m distributed.ray_train --num-actors 10 --run-name <run>
python results.py --run-name <run> --report --aismart
python dashboard.py
```

## Current Docs

- `docs/current/ARCHITECTURE.md`: concise live architecture
- `docs/current/V5_COLLAPSE_AUDIT.md`: V5 diagnosis and stale-claim cleanup
- `docs/current/ACTION_FEEDBACK_PLAN.md`: protocol-3 feedback token contract
- `docs/current/RAY_STATUS.md`: current distributed status
- `docs/current/THE_BPTT.md`: TBPTT reasoning note
- `docs/SPATIAL_HEADS.md`: target-head reference
