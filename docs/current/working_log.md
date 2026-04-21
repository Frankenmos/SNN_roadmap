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

## Next Checks

- refactor / rebalance the reward function using the newer wrapper-driven env understanding
- fix terminal win/loss detection in `RewardFunctionV2`
- regenerate the main `BPTT-1` report bundle against the live DB/checkpoint state
- re-run deterministic and stochastic eval after the reward pass
- only then decide whether the next action-space step is:
  action-history token group,
  learnable selection actions,
  or both
- keep the larger branch questions open:
  entity identity, long-term SNN/TBPTT verdict, and whether this remains the mainline or the research branch
