# Repo State

Updated: 2026-04-21

## What The Repo Does Today

This is currently an SNN + PPO DefeatRoaches project with:

- hybrid observation tokenization from Fix 3
- Stage-1 TBPTT with ordered chunk replay and packed replay
- SDPA-backed attention as the current low-risk attention fast path
- SQLite logging plus analysis plots under `analysis_results/`
- explicit conditioned action semantics for `NO_OP`, `MOVE`, and `ATTACK`
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

- policy action vocab: `NO_OP`, `MOVE`, `ATTACK`
- `MOVE -> Move_screen(x, y)`
- `ATTACK -> Attack_screen(x, y)`
- one reset bootstrap `select_army` step outside PPO memory so policy-controlled steps start from a selected-army state

## Current Training Read

`BPTT-1` is the main post-refactor evidence run so far.

- current training checkpoint:
  `models/BPTT-1/checkpoint.pth`
  trained to episode ~5260
- `best_checkpoint.pth` is stale at episode 200 because best-model promotion is still tied to deterministic eval reward, and recent deterministic eval has stayed flat at `0.0`
- current DB trend is still positive on the shaped training reward:
  last-100 episode average reward is ~223, above the previous 100-episode window
- deterministic behavior is still weak:
  late action mix is dominated by `NO_OP` plus `ATTACK`, while `MOVE` has almost vanished
- eval diagnostics show this is **not** mainly an availability problem:
  after the reset bootstrap, `Move_screen` and `Attack_screen` are available on >99% of logged eval steps

Current interpretation:

- Stage-1 action-space plumbing appears to be working
- the policy is learning against an older reward landscape more than it is learning the micro we actually want
- reward refactor / rebalance is now a more urgent next step than Stage-2 action-space expansion

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

## What Is Done

- Fix 3 hybrid observation tokenization
- extractor hardening for normalizer stability and eval-stat isolation
- Stage-1 TBPTT
- packed replay speed pass
- SDPA attention swap
- action-space availability diagnostics wrapper
- Stage-1 action refactor:
  conditioned spatial `MOVE` / `ATTACK`, availability masking, executed-action bridge token, and reset bootstrap outside PPO memory
- eval-side trace tooling:
  `--trace_episodes`, `analyze_eval_trace.py`, and checkpoint/activation inspection on saved eval episodes
- multi-timescale token memory:
  a faster token-temporal SNN path plus a slower path, combined before the PPO readout

## What Is Not Done

- reward refactor / rebalance based on the newer wrapper-driven env read
- terminal win/loss detection cleanup in `RewardFunctionV2`
- refreshed current-run analysis bundle for the **current** checkpoint instead of relying on the old best-checkpoint snapshots
- dedicated action-history token group replacing the 4-float bridge token in `meta_vec`
- selection actions and broader learnable action vocabulary
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
4. Re-evaluate deterministic vs stochastic behavior after the reward pass before pulling Stage-2 action-space work forward.

## Open Questions

1. Once the reward path is updated, does deterministic behavior recover, or is there still a deeper action-space or optimization bottleneck?
2. Is the 4-float bridge token enough for now, or should Stage 2 move action history into its own token group soon?
3. How much remaining weakness is reward shaping versus entity identity versus optimization instability?
4. When we leave "make the SNN work" mode, do we keep this branch as the research branch and build a denser recurrent branch for the actual game-learning push?

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
