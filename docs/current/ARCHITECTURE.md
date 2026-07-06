# Architecture

Updated: 2026-07-02

This is the concise current architecture reference. It supersedes pre-V6
long-form architecture notes.

Interactive companion: `tools/viz/arch_explorer/` renders this pipeline as
an explorable 3D scene with per-zone shapes, math, and source excerpts
(verified against the code on 2026-07-06; no code-vs-doc discrepancies
found during that pass). `python -m tools.registry export <run>` feeds it
a live bundle so zones show a real run's learned time constants, action
mix, and snapshot lineage.

## Runtime Flow

```text
PySC2 obs
  -> ObservationExtractor
  -> PolicyInputBatch
  -> PolicyNetwork
  -> PPO / ActionSpace
  -> Smart_screen or no_op
```

## Policy Input

`PolicyInputBatch` contains:

- `spatial_obs [B, 27, 84, 84]`
- `entity_features [B, 24, F]` plus `entity_mask`
- `selection_features [B, 20, 7]` plus `selection_mask`
- `action_feedback_tokens [B, 1, 12]`
- `meta_vec [B, 15]`
- optional recurrent state `(syn, mem)`

The token stream has 95 tokens:

```text
49 spatial + 24 entity + 20 selection + 1 action_feedback + 1 meta
```

## Policy Network

- CNN encodes `feature_screen`.
- Pooled 7x7 spatial tokens receive explicit learned 2D position.
- Entity, selection, action-feedback, and meta groups get their own encoders
  and token-type embeddings.
- Spiking attention runs over the token stream.
- Fast and slow token-temporal SNN pathways carry state across environment
  steps.
- Entity and selection recurrent carry is intentionally disabled until slots
  are identity-pinned.

## Spatial Target Head

Current default:

```yaml
model:
  spatial_head_type: "coarse_to_fine"
  fine_skip_connection: true
```

`coarse_to_fine` predicts:

- coarse cell: 49-way categorical over the 7x7 grid
- fine offset: 144-way categorical inside the selected 12x12 cell

The fine skip connection feeds pre-pool 84x84 conv features into the fine
stage. This was added after V5 diagnostics showed a constant fine sub-index.

## Actions

Policy action IDs:

- `0`: `NO_OP`
- `1`: `LEFT_CLICK`, scaffolded but masked unavailable in DefeatRoaches
- `2`: `RIGHT_CLICK`, dispatched as `Smart_screen(x, y)`

One reset bootstrap `select_army` happens outside PPO memory.

## PPO / TBPTT

- Rollouts are stored as fragments.
- GAE is computed per fragment.
- Replay is ordered and chunked by `tbptt_window`.
- Helper/reset transitions are masked.
- Time caps are stored as truncations, not terminal `done`, so value bootstrap
  can continue through caps.
- The same fragment protocol supports local training and Ray training.

## Self-Imitation Learning (SIL, added 2026-06-30)

- Config-gated (`sil_enabled`); live in the V7 run.
- Trophy buffer: FIFO deque (`sil_buffer_size: 5000`) of single-step
  verified-good clicks with their pre-step recurrent state.
- Admission: a `RIGHT_CLICK` at step j is admitted only if step j+1's
  action-feedback tokens confirm engagement (`TARGET_NEAR_ENEMY` or
  `ENEMY_HEALTH_DROP`). Return-gating alone was rejected: marine auto-attack
  inflates returns for idle steps.
- Replay: `_run_sil_pass` after the PPO epochs, with its own optimizer step.
  Loss: `-sil_coef * (R - V(s))+.detach() * log pi(a|s)` (Oh et al. 2018,
  Eq. 2 shape).
- Known open concerns: the separate optimizer step moves the policy outside
  PPO's trust-region accounting, and stored recurrent states go stale as the
  network evolves (see REPO_STATE "What Is Still Open").

## Current Numerical Defaults

- `batch_size: 2048`
- `epochs: 4`
- `tbptt_window: 128`
- `lr: 5e-5`
- `amp_dtype: "bf16"`
- `reward.name: "defeat_roaches_v4"`

## Current Known Limits

- Entity identity is not pinned by `raw_units.tag`, so entity recurrent carry
  remains off.
- Deterministic behavior still needs trace-level validation after V6/V7 fixes.
- `LEFT_CLICK` is still a scaffold, not a learned live action.
- This is still a DefeatRoaches-specific research branch, not a full-SC2 agent.
