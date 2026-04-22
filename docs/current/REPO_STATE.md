# Repo State

Updated: 2026-04-22

## What The Repo Does Today

This is currently an SNN + PPO DefeatRoaches project with:

- hybrid observation tokenization from Fix 3
- Stage-1 TBPTT with ordered chunk replay and packed replay
- SDPA-backed attention as the current low-risk attention fast path
- SQLite logging plus analysis plots under `analysis_results/`
- explicit learned action semantics for `NO_OP` and `SMART`
- eval-side trace capture plus per-trace analysis bundles for real-step inspection
- dual-timescale token memory:
  fast + slow token-temporal SNN pathways feeding one shared control latent

The current policy input path is:

- spatial `feature_screen` -> CNN -> pooled spatial tokens
- `feature_units` -> entity tokens
- `multi_select` / `single_select` -> selection tokens
- `meta_vec[32] = player[11] + available_actions[16] + pysc2_last_action[1] + bridge_token[4]`
- token-type embeddings
- spiking self-attention
- fast + slow token-temporal SNN pathways, combined into one latent readout
- PPO action / spatial / value readout

The current action path is:

- policy action vocab: `NO_OP`, `SMART`
- `SMART -> Smart_screen(x, y)`
- one reset bootstrap `select_army` step outside PPO memory so policy-controlled steps start from a selected-army state

Why the repo moved there:

- `Attack_screen` was too semantically permissive in practice:
  on empty ground it could still behave like an aggressive movement primitive
- that gave `ATTACK` an unfair action-space advantage over `MOVE`
- collapsing to one contextual screen click is cleaner than pretending
  `MOVE` and `ATTACK` were already honest, disentangled commands

## Current Training Read

`BPTT-1` is the main post-refactor evidence run so far.

- current training checkpoint:
  `models/BPTT-1/checkpoint.pth`
  trained to episode ~5260
- `best_checkpoint.pth` is stale at episode 200 because best-model promotion is still tied to deterministic eval reward, and recent deterministic eval has stayed flat at `0.0`
- current DB trend is still positive on the shaped training reward:
  last-100 episode average reward is ~223, above the previous 100-episode window
- deterministic behavior is still weak:
  old late action mix was dominated by `NO_OP` plus `ATTACK`, while `MOVE` had almost vanished
- that now reads more like an action-semantics problem than a simple reward-only problem:
  `Attack_screen` was acting too much like a privileged attack-move token
- the action space has now been simplified to `NO_OP + SMART`, so the next evidence run should be interpreted under the new click semantics, not the old `MOVE/ATTACK` split

Current interpretation:

- the old Stage-1 action-space plumbing was internally correct but semantically crooked
- reward refactor / rebalance is still urgent
- the new Smart-screen action simplification should be judged before branching into wilder architecture alternatives

## What Is Considered Current

- `docs/current/REPO_STATE.md`
  short repo truth and open questions
- `docs/current/action_refactor.md`
  what the Stage-1 action refactor landed and what remains
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
- eval-side trace tooling:
  `--trace_episodes`, `analyze_eval_trace.py`, and checkpoint/activation inspection on saved eval episodes
- multi-timescale token memory:
  a faster token-temporal SNN path plus a slower path, combined before the PPO readout

## What Is Not Done

- reward refactor / rebalance based on the newer wrapper-driven env read
- terminal win/loss detection cleanup in `RewardFunctionV2`
- refreshed current-run analysis bundle for the **current** checkpoint instead of relying on the old best-checkpoint snapshots
- dedicated action-history token group replacing the 4-float bridge token in `meta_vec`
- selection actions and broader learnable action vocabulary beyond the current `SMART` click primitive
- tag-pinned entity identity via `raw_units.tag`
- final verdict on whether the SNN + TBPTT branch is worth keeping as the long-term game-learning backbone
- broader multi-minigame / full-game branch
- reward-as-neuromodulator experiment inside the policy recurrent state
- ALIF / adaptive-threshold neuron swap in the policy

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

## Explicit Deferrals

These were discussed, but intentionally not landed with the multi-timescale patch:

- reward-driven neuromodulation of membrane / synaptic state
- ALIF neuron swaps inside attention or token memory
- temporal state inside the attention block itself

Reason:

- they are higher-risk objective / state-semantics changes than the dual-pathway patch
- they would make PPO/TBPTT replay semantics harder to trust without a narrower experiment branch

## Immediate Priorities

1. Refactor the reward function so it reflects the newer env understanding from the wrappers instead of the older V2 proxy.
2. Fix the terminal win/loss check in `agent_core/rewards/defeat_roaches_v2.py`.
3. Regenerate the main `BPTT-1` analysis bundle against the live checkpoint/DB state so the static report stops lagging the run.
4. Re-evaluate deterministic vs stochastic behavior under the new `SMART` action space before pulling Stage-2 action work forward.

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
2. Is the 4-float bridge token enough for now, or should Stage 2 move action history into its own token group soon?
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
