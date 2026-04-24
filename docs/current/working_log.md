# Working Log

Compressed current-memory version.

Verbose pre-compression snapshot:
`docs/archive/working_log_2026-04-20_pre_compress.md`

## 2026-04-24

- expanded the live meta protocol to `META_VECTOR_DIM = 24`
- replaced the old 4-field bridge-only slice with a 9-field one-step
  action-history bridge:
  attempted action `[type, x_norm, y_norm, extra]` plus executed-action and
  score-delta feedback
- kept `peek_observation()` non-mutating for score-history state so PPO
  bootstrapping can preview next observations without consuming the delta

## 2026-04-19

- froze the `PolicyInputBatch` protocol and moved the repo onto the hybrid observation path:
  spatial screen tensor, entity tokens, selection tokens, and `meta_vec`
- rewired `PolicyNetwork` and PPO around tokenized hybrid inputs instead of the old flat vector path
- hardened the observation extractor:
  safer running-stat updates, eval-stat isolation, schema validation, and better tests
- landed Stage-1 TBPTT with ordered chunk replay and helper-step masking semantics
- added packed replay and replay-side fast paths to make TBPTT affordable enough to iterate on
- swapped attention onto SDPA and refreshed logging / dashboard support for the current branch
- verification at the end of that pass:
  `pytest tests -q`
  `40 passed`

## 2026-04-20

- landed Stage-1 action refactor:
  conditioned spatial `MOVE` / `ATTACK`, explicit `Move_screen` / `Attack_screen`, availability masking, and executed-action bridge token plumbing
- moved the reset-only `select_army` step outside PPO memory and removed the old mid-episode helper fallback
- removed the scripted nearest-enemy attack targeting from the learned path
- kept rollout storage stable while making replay condition spatial logits on stored action IDs
- fixed a real PySC2 runtime mismatch where `FunctionCall` does not reliably expose `.name`
- aligned fake test action IDs with the real DefeatRoaches IDs
- fixed analysis-side action decoding for post-refactor runs:
  the analyzer now detects current action semantics from sibling run config and no longer mislabels `action=2` as no-op for `BPTT-1`
- regenerated `analysis_results/BPTT-1/instability_report.txt` after the analysis fix
- added optional eval-side episode trace artifacts:
  `eval.py` can now save per-step `.pt` traces with extracted `PolicyInputBatch` tensors and dispatched action metadata via `--trace_episodes` / `--trace_output_dir`, without touching the training DB
- added a separate eval-trace analysis entrypoint:
  `analyze_eval_trace.py` now turns those `.pt` sidecars into a compact image/report bundle without bloating `dashboard.py`, and it can optionally export `conv1` / `conv2` / `conv3` activation maps for a selected step
- verification for the eval trace path:
  `pytest tests\test_eval_trace.py tests\test_eval_trace_analysis.py tests\test_analysis_tools.py -q`
  `7 passed`
- verification after the refactor:
  `pytest tests -q`
  `47 passed`

## 2026-04-21

- renamed the live entrypoints and package surface to match what the repo actually is now:
  `train.py`, `eval.py`, `agent.py`, and `agent_core/`
- kept the older `PPO_CNN*` files and package modules as thin compatibility shims so old commands/imports still resolve during the transition
- inspected `BPTT-1` against the live DB, eval JSONLs, and saved eval traces instead of relying only on the older static report
- confirmed the current run is ahead of the saved report:
  `checkpoint.pth` is at episode ~5260, while `best_checkpoint.pth` is stale at episode 200 because deterministic eval has stayed flat
- confirmed the post-bootstrap action mask is not the main culprit:
  `Move_screen` and `Attack_screen` are available on >99% of logged eval steps
- confirmed the learned late-run action mix is the real concern:
  mostly `NO_OP` plus `ATTACK`, with `MOVE` nearly absent
- found that the current reward path is still the older proxy:
  `positioning_reward` is dead, and terminal win/loss still keys off `obs.reward > 0`
- re-ranked the immediate backlog:
  reward refactor / rebalance now sits ahead of Stage-2 action-history and selection-action work
- refreshed the current docs so the active source-of-truth files reflect the live run state instead of the older post-implementation snapshot
- landed a bounded architecture experiment:
  the policy now has dual-timescale token memory via fast + slow token-temporal SNN pathways combined before the shared latent readout
- kept the dual-timescale patch repo-native:
  the recurrent state stayed a plain `(syn, mem)` tuple, but each tensor now carries an internal pathway axis
- fixed an early post-landing bug in the dual-timescale patch:
  one recurrent-state mask multiply still used the old `[B, tokens, 1]` broadcast pattern and broke when `batch_size != 2`
- added a regression test for non-2 batch sizes so the pathway axis cannot silently alias the batch axis again
- fixed a real runtime bug in `defeat_roaches_v3.py` where `feature_units` could be a NumPy-like array and crash on truthiness checks
- explicitly deferred the riskier "grok" follow-ups for a later branch:
  reward neuromodulation, ALIF swaps, and attention-side temporal state were discussed but not implemented
- verification for the temporal-pathway patch:
  `pytest tests/test_agent.py tests/test_PPO.py tests/test_training_loop.py -q`
  `36 passed`
  `python -m compileall agent.py agent_core tests`
- simplified the learned action space from `NO_OP / MOVE / ATTACK` to `NO_OP / SMART`
- reason for that simplification:
  `Attack_screen` was semantically cheating by behaving too much like attack-move on empty ground, which made the old split cleaner on paper than in-game
- rewired the protocol, PPO masking, agent dispatch, bridge token semantics, and config around `Smart_screen`
- kept analysis/backfill compatibility for older runs:
  analysis tools now infer action semantics from each run config so the older 3-way runs remain readable
- updated eval-trace analysis and dashboard paths so they can understand both old `MOVE/ATTACK` runs and new `SMART` runs
- verification after the Smart-screen redesign:
  `pytest tests -q`
  `55 passed`

## 2026-04-22

- cleaned up the rename boundary so the live plumbing no longer depends on
  placeholder modules inside `PPO_CNN/`
- restored `PPO_CNN/policy_input.py`, `PPO_CNN/policy_network.py`,
  `PPO_CNN/PPO.py`, `PPO_CNN/reward_function.py`, and
  `PPO_CNN/reward_function_2.py` from the archived `old_scritps/`
  snapshot
- current repo contract after that cleanup:
  `agent_core/` is the canonical runtime package, while `PPO_CNN/` is now
  honest legacy code instead of a disguised alias layer
- repaired the main architecture bottleneck called out by the BPTT review:
  the click head no longer has to recover screen coordinates only from a pooled
  global latent
- added explicit 2D positional encoding to the pooled spatial token grid before
  attention
- kept a structured spatial branch alive through the policy and rewired the
  `SMART` click head to consume that retained spatial context plus latent/action
  conditioning
- kept the value and high-level action path global, but made localization
  genuinely spatial
- resolved the `policy_mask` ambiguity by treating it as a true training mask:
  critic loss is now masked too, not only actor and entropy terms
- moved PPO updates ahead of deterministic eval / best-checkpoint selection so
  eval reflects the freshest trained parameters
- added explicit recurrent-state contract assertions in PPO rollout storage and
  bootstrap-tail handling
- verification after the spatial-head repair:
  `pytest tests -q`
  `61 passed`
- captured several post-`SMART` action-space training ideas as future branch candidates rather than immediate reward-function edits
- offline pretraining idea:
  relabel stronger old `MOVE` / `ATTACK` trajectories into the new `SMART` action space and use them as a behavior-cloning warm start before PPO
- dataset-cleanup idea for that branch:
  classify old or future `SMART` clicks by effect using short-horizon observation deltas
  such as enemy health drop, friendly displacement, weapon-cooldown change, or null-effect clicks
- curriculum idea:
  split the meaning of `SMART` across easier tasks before returning to full DefeatRoaches
- concrete curriculum sketches worth remembering:
  `move-to-beacon` style map for purposeful locomotion
  a custom DefeatRoaches-like map with enemies placed so they cannot threaten much, to teach clicking near enemies / attack intent more directly
- explicit caution logged:
  these curriculum / offline-pretrain ideas should stay separate branch work and not be silently folded into the reward refactor just because they are tempting

## 2026-04-22 (compatibility and continuity pass)

- reconciled the `policy_mask` vs `sample_mask` transition:
  - kept `sample_mask` as the canonical name in runtime code paths
  - added compatibility aliases so legacy callers using `policy_mask` still read/write correctly
  - made `_build_tbptt_chunks()` accept `policy_masks=...` as a backward-compatible kwarg
  - made packed chunk output emit both `sample_mask` and `policy_mask` keys
- added rollout-continuity protection in `train.py` checkpointing:
  `save_checkpoint()` now skips persistence while PPO in-flight rollout memory is non-empty
  so resume behavior does not silently break exact on-policy continuity
- verified reward version entrypoint still keeps legacy path:
  `RewardFunctionV2` remains available through `agent_core.rewards.__init__`, while `v3` remains current default
- updated docs index text to mark `docs/current/urgent.md` as an active historical review source rather than an empty scratchpad

## 2026-04-22 (deep research follow-up)

- reviewed `docs/deep-research-report.md` against the live code and limited the
  patch set to contract-level fixes that did not require environment-side intent
- tightened the frozen observation protocol:
  `PolicyInputBatch` now rejects recurrent state tensors unless they are rank-3
  legacy or rank-4 multi-timescale, so malformed state now fails at the
  protocol boundary instead of later inside `spiking_policy.py`
- fixed rollout cadence for long episodes:
  `train.py` now flushes PPO updates immediately once `rollout_steps` is reached
  inside the episode loop, rather than waiting only for the episode boundary
- explicitly did **not** change time-cap semantics yet:
  the repo still needs a task-definition decision on whether
  `steps_per_episode` is a real horizon or only a training truncation before we
  touch PPO bootstrap behavior there
- added regression coverage for the new contract/cadence behavior:
  - `tests/test_policy_input.py`
  - `tests/test_training_loop.py`
- verification after the follow-up pass:
  `pytest tests/test_policy_input.py tests/test_training_loop.py tests/test_PPO.py tests/test_agent.py -q`
  `47 passed`
  `pytest tests -q`
  `63 passed`

## 2026-04-23 (semantic fixes before Phase 2)

- fixed action-dimension reporting in `spiking_policy.py`:
  added `action_dim` to `resolved_config()` output so analysis tools
  can correctly detect action semantics
- fixed action-semantics inference in `results.py`:
  added fallback logic to detect `semantic_pointer_v1` when
  `spatial_head_type == "token_pointer"` and `meta_input_dim == 19`,
  even when `action_dim` is missing from config
- regenerated Zero-3 instability report with corrected action labels
- **fixed time-cap semantics in `train.py`**:
  - changed `done = env_done or time_cap` to `done = env_only`
  - timeout is now treated as truncation, not terminal
  - value bootstrapping continues through time_cap
  - loop still exits at time_cap but doesn't treat it as episode end
- **verified rollout cadence is correct**:
  - the `rollout_steps` check already happens inside the step loop
  - PPO updates fire immediately when target is reached, not just at episode boundaries
- **increased TBPTT window from 32 to 128**:
  - allows gradients to flow 4x further back in time
  - better credit assignment for long-horizon events
  - enabled by available VRAM/compute
- **increased `steps_per_episode` from 600 to 3600**:
  - the 600 limit was legacy and doesn't match DefeatRoaches survival dynamics
  - games depend on survival + long time limit, not artificial short caps

## 2026-04-22 (semantic action + token-pointer migration)

- implemented Phase 1 + Phase 2 of the spatial-target migration plan
  instead of mixing it with any new TBPTT or reward rewrite
- replaced the live policy semantics with:
  `NO_OP`, `LEFT_CLICK`, `RIGHT_CLICK`
- current DefeatRoaches mapping after that migration:
  `RIGHT_CLICK -> Smart_screen(x, y)`
  `LEFT_CLICK` is scaffolded in code and protocol, but masked unavailable on
  the current wrapper so it cannot become a live policy alias
- shrank `meta_vec` from the older raw-action shape to the semantic one:
  `meta_vec[19] = player[11] + semantic_available_actions[3] + pysc2_last_action[1] + bridge_token[4]`
- expanded bridge-token semantics so the recurrent path can distinguish
  `LEFT_CLICK`, `RIGHT_CLICK`, bootstrap select, and no-op cleanly
- added `ActionSample` as the acting payload and threaded optional
  `target_index / coarse_index / fine_index` storage through PPO rollout
  memory without breaking external `act / move_x / move_y` logging
- replaced the hardcoded factorized `x + y` replay contract with a generic
  policy-side target-head interface:
  `build_target_head`, `sample_target`, `evaluate_target`,
  `encode_xy_to_target`, and `decode_target_to_xy`
- kept `factorized_xy` alive as a legacy target-head mode behind config
- made `token_pointer` the new default head:
  the policy now predicts one categorical distribution over the pooled spatial
  token grid and decodes token centers back to logged `(x, y)` screen clicks
- generalized PPO from "SMART is the only spatial action" to a semantic
  spatial-action set and moved target entropy / target log-prob evaluation out
  of PPO math and into the policy head
- preserved the core seam that was already correct:
  action selection still happens first, target selection second, and replay
  still teacher-forces the recorded action id
- updated analysis helpers and eval-trace helpers so new runs with
  `action_dim=3` plus `spatial_head_type=token_pointer` do not get mislabeled
  as legacy runs
- refreshed diagnostics / tests to the new contract:
  semantic availability, bridge token shape, token-pointer head shape,
  ActionSample returns, generic PPO losses, packed replay, and current
  policy-input diagnostics now all assert against the migrated behavior
- verification after the migration:
  `pytest tests -q`
  `63 passed`

## Next Checks

- refactor / rebalance the reward function using the newer wrapper-driven env understanding
- decide and document time-cap semantics:
  task horizon vs training truncation for PPO bootstrap
- env-verify the new semantic action mask:
  confirm `RIGHT_CLICK` behaves as intended and keeping `LEFT_CLICK` masked is
  still the right no-alias choice on the live wrapper
- decide whether the next spatial-head step is actually ready:
  `coarse_to_fine` should come next only after a short live training/eval pass
  confirms the token-pointer version is stable
- fix terminal win/loss detection in `RewardFunctionV2`
- regenerate the main `Zero` report bundle against the live DB/checkpoint state
- keep `analysis_results/BPTT-1` as historical context only (same obs function shape, but older action-space version)
- re-run deterministic and stochastic eval after the reward pass
- only then decide whether the next action-space step is:
  action-history token group,
  learnable selection actions,
  or both
- keep the future branch ideas visible but separate:
  offline `SMART` pretraining,
  effect-labeled click datasets,
  and curriculum maps for locomotion / attack semantics
- keep the larger branch questions open:
  entity identity, long-term SNN/TBPTT verdict, and whether this remains the mainline or the research branch
