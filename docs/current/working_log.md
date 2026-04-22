# Working Log

Compressed current-memory version.

Verbose pre-compression snapshot:
`docs/archive/working_log_2026-04-20_pre_compress.md`

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

## Next Checks

- refactor / rebalance the reward function using the newer wrapper-driven env understanding
- fix terminal win/loss detection in `RewardFunctionV2`
- regenerate the main `BPTT-1` report bundle against the live DB/checkpoint state
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
