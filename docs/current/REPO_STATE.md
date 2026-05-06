# Repo State

Updated: 2026-05-06

## What The Repo Does Today

This is currently an SNN + PPO DefeatRoaches project with:

- hybrid observation tokenization from Fix 3
- fragment-based PPO with per-fragment GAE and an initial synchronous Ray rollout path
- `POLICY_PROTOCOL_VERSION = 2` with `policy_input_schema = "stream_action_feedback_v1"`
- Stage-1 TBPTT with ordered chunk replay and packed replay
- SDPA-backed attention as the current low-risk attention fast path
- SQLite logging plus analysis plots under `analysis_results/`
- semantic policy actions:
  `NO_OP`, `LEFT_CLICK`, `RIGHT_CLICK`
- eval-side trace capture plus per-trace analysis bundles for real-step inspection
- distributed smoke entrypoint:
  `python -m distributed.ray_train --num-actors 4 --max-updates N`
- dual-timescale token memory:
  fast + slow token-temporal SNN pathways feeding one shared control latent
- explicit 2D positional encoding on the spatial token grid
- a generic conditional target-head interface:
  `coarse_to_fine` is the current `config.yaml` default, while `token_pointer`
  remains an efficient lower-precision fallback/comparison head

The current policy input path is:

- spatial `feature_screen` -> CNN -> pooled spatial tokens
- `feature_units` -> entity tokens
- `multi_select` / `single_select` -> selection tokens
- `action_feedback_tokens[1, 9] = previous bridge action + execution/score feedback`
- `meta_vec[15] = player[11] + semantic_available_actions[3] + pysc2_last_action[1]`
- token-type embeddings
- spiking self-attention
- fast + slow token-temporal SNN pathways, combined into one latent readout
- PPO action / target / value readout, with the target head reading a retained
  spatial context map and replay teacher-forcing recorded target payloads

The current action path is:

- policy action vocab: `NO_OP`, `LEFT_CLICK`, `RIGHT_CLICK`
- current DefeatRoaches dispatch:
  `RIGHT_CLICK -> Smart_screen(x, y)`
  `LEFT_CLICK -> scaffolded but masked unavailable on this wrapper`
- one reset bootstrap `select_army` step outside PPO memory so policy-controlled steps start from a selected-army state

Why the repo moved there:

- `Attack_screen` was too semantically permissive in practice:
  on empty ground it could still behave like an aggressive movement primitive
- that gave `ATTACK` an unfair action-space advantage over `MOVE`
- collapsing to one contextual screen click is cleaner than pretending
  `MOVE` and `ATTACK` were already honest, disentangled commands

## Current Training Read

The active/latest training run is now `banana_b2048_e4_a10`, currently around
5,000 episodes. It is not ready for a final post-run conclusion yet, but the
behavior read is disappointing enough that reward/action-behavior diagnosis is
the next discussion point before treating this as a healthy training direction.

- live training checkpoint:
  `models/banana_b2048_e4_a10/checkpoint.pth`
- current analysis bundle:
  `analysis_results/banana_b2048_e4_a10/`
- active comparison artifact:
  `analysis_results/BPTT-1` is historical only (older `MOVE/ATTACK` semantics)
- track `banana_b2048_e4_a10` on its own trajectory (action mix, terminals,
  rewards, episode-phase behavior) before declaring any global trend conclusions

Current interpretation:

- reward refactor / rebalance is still urgent
- the current run's behavior should be reviewed before adding more action-space
  complexity
- the new semantic-action + `coarse_to_fine` stack now needs env-backed
  judgment before pulling the `heatmap` head or larger action-space work forward

## What Is Considered Current

- `docs/current/REPO_STATE.md`
  short repo truth and open questions
- `docs/current/ACTION_FEEDBACK_PLAN.md`
  current stream-token action-feedback protocol and bridge history
- `docs/current/THE_BPTT.md`
  still current because Stage-1 TBPTT is implemented and its long-term verdict is still open
- `docs/current/working_log.md`
  compressed recent-memory log
- `config.yaml`
  current default knobs
- live runtime code paths point at:
  `agent.py`, `train.py`, `eval.py`, and `agent_core/`

## What Is Considered Legacy

- `PPO_CNN/`
  restored older architecture snapshot kept for historical reference,
  archaeology, and old-code comparison rather than live runtime imports
- root `PPO_CNN_*` files
  compatibility wrappers for old commands; useful for transition, but not
  the canonical code path going forward

## What Is Done

- Fix 3 hybrid observation tokenization
- extractor hardening for normalizer stability and eval-stat isolation
- Stage-1 TBPTT
- packed replay speed pass
- SDPA attention swap
- action-space availability diagnostics wrapper
- Stage-1 action refactor:
  conditioned spatial action head, executed-action bridge token, and reset bootstrap outside PPO memory
- action-space simplification pass:
  the learned action vocab is now `NO_OP + SMART`, with `Smart_screen` as the only learned spatial primitive
- semantic-action migration:
  the learned action vocab is now `NO_OP / LEFT_CLICK / RIGHT_CLICK`, with
  `RIGHT_CLICK` mapped to `Smart_screen` and `LEFT_CLICK` masked unavailable so
  the current wrapper does not learn an alias
- generic target-head migration:
  the policy exposes build/sample/evaluate/decode target hooks and PPO now
  stores richer target payload fields without breaking external `(x, y)` logs
- token-pointer target head:
  available lower-precision fallback/comparison head that predicts one
  categorical distribution over pooled spatial tokens
- coarse-to-fine target head:
  implemented and selected by the current `config.yaml`; live env validation
  is still pending
- eval-side trace tooling:
  `--trace_episodes`, `analyze_eval_trace.py`, and checkpoint/activation inspection on saved eval episodes
- multi-timescale token memory:
  a faster token-temporal SNN path plus a slower path, combined before the PPO readout
- spatial localization repair:
  explicit positional encoding plus a structured click head for `SMART`
- training-mask semantics cleanup:
  masked transitions now mask the critic too, not only actor and entropy terms
- update-before-eval ordering:
  pending PPO updates now run before deterministic eval and best-checkpoint selection
- rollout-cadence semantics cleanup:
  pending PPO updates now flush as soon as `rollout_steps` is reached, even
  inside longer episodes
- stricter recurrent-state protocol:
  `PolicyInputBatch` now rejects malformed state ranks before replay reaches the
  policy

## What Is Not Done

- reward refactor / rebalance based on the newer wrapper-driven env read
- env-backed verification / tuning of `RewardFunctionV3` terminal and outcome
  semantics
- final refreshed current-run analysis/readout after `banana_b2048_e4_a10`
  finishes or reaches the next stable checkpoint block
- env-backed validation that timeout-as-truncation behaves as intended in the
  current single-process and Ray rollout paths
- env-backed validation that keeping `LEFT_CLICK` masked is still the correct
  no-alias choice on the current wrapper
- broader action-history token groups beyond the current one-step 9-field bridge
- env-backed validation of the current `coarse_to_fine` spatial head - see [../SPATIAL_HEADS.md](../SPATIAL_HEADS.md)
- selection actions and broader learnable action vocabulary beyond the current
  semantic click scaffold
- tag-pinned entity identity via `raw_units.tag`
- final verdict on whether the SNN + TBPTT branch is worth keeping as the long-term game-learning backbone
- broader multi-minigame / full-game branch
- reward-as-neuromodulator experiment inside the policy recurrent state
- ALIF / adaptive-threshold neuron swap in the policy
- `heatmap` spatial head (Phase 3)

## Known Compatibility Note

Checkpoints from before the 2026-04-20 action refactor should be treated as incompatible:

- `meta_vec` width changed from `28` to `32`
- the meta encoder input changed
- the policy readout now conditions spatial logits on stored action IDs

Checkpoints from before the 2026-04-21 Smart-screen action simplification should also be treated as incompatible:

- `cfg.model.action_dim` changed from `3` to `2`
- the learned policy action meaning changed from `NO_OP/MOVE/ATTACK` to `NO_OP/SMART`
- analysis helpers now infer action semantics from each run config so old runs stay readable

Checkpoints from before the 2026-04-21 multi-timescale temporal patch should also be treated as incompatible:

- the policy now includes both fast and slow token-temporal SNN pathways
- recurrent state now carries an extra temporal-pathway dimension internally

Checkpoints from before the 2026-04-22 spatial-head repair should also be treated as incompatible:

- the click head now has explicit spatial positional encoding
- the `SMART` coordinate head now uses a structured spatial branch instead of
  the older pooled-latent MLP head
- the policy state dict gained new spatial-click parameters

Checkpoints from before the 2026-04-22 semantic-action / token-pointer migration
should also be treated as incompatible:

- `cfg.model.action_dim` changed from `2` to `3`
- `meta_vec` width changed from `32` to `19`
- the bridge-token action vocabulary changed
- the live target head changed from factorized `x/y` logits to token-pointer
  logits over pooled spatial tokens
- PPO rollout memory and replay now carry richer target payload fields

Checkpoints from before the 2026-04-24 action-history bridge expansion should
also be treated as incompatible:

- `meta_vec` width changed from `19` to `24`
- the bridge slice changed from 4 attempted-action fields to 9 fields:
  attempted action plus last-action and score-delta feedback

Checkpoints from before the 2026-04-26 protocol version migration should also
be treated as incompatible:

- `POLICY_PROTOCOL_VERSION` introduced (current = 2)
- `policy_input_schema` introduced (current = "stream_action_feedback_v1")
- `meta_vec` width changed from `24` to `15`
- action-feedback fields moved from meta_vec to separate `action_feedback_tokens`
- fragment-based PPO with per-fragment GAE replaces single-bootstrap approach
- initial Ray trainer added: rollout actors collect fragments, one learner owns
  optimizer/scheduler/checkpoints, stale `policy_version` fragments are rejected

**Note (2026-04-27)**: The `coarse_to_fine` spatial head is implemented and
selected in `config.yaml`. `token_pointer` remains available as the lower-cost
fallback/comparison head. See [../SPATIAL_HEADS.md](../SPATIAL_HEADS.md) for
details.

## Explicit Deferrals

These were discussed, but intentionally not landed with the multi-timescale patch:

- reward-driven neuromodulation of membrane / synaptic state
- ALIF neuron swaps inside attention or token memory
- temporal state inside the attention block itself

Reason:

- they are higher-risk objective / state-semantics changes than the dual-pathway patch
- they would make PPO/TBPTT replay semantics harder to trust without a narrower experiment branch

## Immediate Priorities

1. Finalize reward semantics for `banana_b2048_e4_a10` using wrapper-derived terminal and outcome signals.
2. Env-verify the new semantic action mask and `coarse_to_fine` clicks on the live wrapper.
3. Verify and tune `RewardFunctionV3` terminal/outcome semantics against live
   traces.
4. Regenerate/review the main `banana_b2048_e4_a10` analysis bundle once the run
   reaches a stable checkpoint block or final stop point.
5. Compare `coarse_to_fine` against `token_pointer` only after a short live training/eval pass establishes current behavior.
6. Re-evaluate deterministic vs stochastic behavior before pulling Stage-2 action work forward.

## Future Branch Candidates

- offline `SMART` pretraining:
  take stronger old runs, relabel `MOVE` / `ATTACK` clicks into `SMART(x, y)`, and use them as a behavior-cloning warm start before online PPO
- click-outcome labeling:
  build a small detector over short-horizon observation deltas so future `SMART` datasets can distinguish attack-like, move-like, and null-effect clicks more honestly
- curriculum maps:
  start with easier environments that teach the two meanings of `SMART` separately before returning to full DefeatRoaches
- concrete curriculum sketches already worth remembering:
  `move-to-beacon` for purposeful locomotion, and a custom DefeatRoaches-like map where enemies are effectively harmless or constrained so the agent can learn enemy-directed clicking without immediate full micro pressure
- important boundary:
  these are branch ideas, not "just tweak the reward function a bit more" ideas

## Open Questions

1. Once the reward path is updated, does deterministic behavior recover, or is there still a deeper action-space or optimization bottleneck?
2. Is the 9-field one-step bridge enough for now, or should Stage 2 move longer action history into its own token group soon?
3. How much remaining weakness is reward shaping versus entity identity versus optimization instability?
4. If `SMART` remains hard to learn from pure PPO, is the better next move offline pretraining, curriculum maps, or both?
5. When we leave "make the SNN work" mode, do we keep this branch as the research branch and build a denser recurrent branch for the actual game-learning push?

## Archive Notes

- detailed pre-compression implementation history:
  `docs/archive/working_log_2026-04-20_pre_compress.md`
- external action-space ideation and scratch files:
  `docs/archive/observations_2026-04-20/`

## Practical Entry Points

- `README.md`
  general project entry
- `train.py`
  training entrypoint
- `eval.py`
  evaluation entrypoint
- `docs/README.md`
  doc map
