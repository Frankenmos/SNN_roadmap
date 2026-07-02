# Repo State

Updated: 2026-07-02

This is the primary source of truth for the live code. Older architecture
reviews and CNN/PPO-era run narratives are historical, not tracked current docs.

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
- SIL: self-imitation replay of feedback-verified good clicks
  (`sil_enabled: true` since V7; see `ARCHITECTURE.md` SIL section)
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
| `banana_glasses_v6_b2048_e4_a10` | Post-fine-skip/glasses family; training reward became positive, max reward `555.85`, deterministic eval still needs scrutiny. `best_checkpoint` deterministic eval is ALL no-op — best-by-native-score selects the idle auto-attack exploit; do not treat it as behaviorally best. |
| `banana_glasses_v7_sil_b2048_e4_a10` | Live (SIL enabled). ~ep 1730: first deterministic eval that actually attacks (precise targeting, sparse engagement); shaped reward still fully negative (avg −68, best −12); HIGH flags: clip fraction ~56%, approximate KL 0.066 vs target 0.03; pre-clip grad_norm ~160 vs clip 0.5. |

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
- SIL (2026-06-30): feedback-gated trophy buffer + `(R−V)+` imitation pass in
  `ppo_trainer.py`; tests in `tests/test_sil.py`. Uncommitted as of 2026-07-02.
- Smart-outcome diagnostics wrapper wired into `envs/setup_env.py` behind
  `use_smart_outcome_diagnostics`; eval flags in `eval.py`.
- `tools/analysis/probe_action_logits.py`: fidelity-exact replay of eval-trace
  inputs → per-step action logits/probs/value (built to diagnose det-eval idling).

## What Is Still Open

Measurement questions flagged 2026-07-02 (see `learning/TUTOR_INSTRUCTIONS.md`
§7 for the learning-session framing):

- Gradient scale: logged `grad_norm` (pre-clip) averages ~160 against a 0.5
  clip — every update is direction-only. Suspect: unnormalized value-loss on
  the shared trunk. Measure policy-loss vs value-loss grad norms separately.
- SIL trophy staleness: stored pre-step recurrent states age in the FIFO
  buffer across many policy versions. Log trophy age at replay; check V(s)
  sanity on old trophies.
- SIL vs trust region: is SIL's separate optimizer step the cause of the
  sustained 56% clip fraction / KL 0.066?
- Ground-click semantics: does a missed `Smart_screen` (move order) cancel
  in-progress auto-attacks, making exploration actively negative vs NO_OP?
  Verify in an eval trace before changing anything.
- Shaped reward has never been positive (best episode −12); `win_reward +60`
  remains unreachable. Raw-native-score ablation still queued.
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
