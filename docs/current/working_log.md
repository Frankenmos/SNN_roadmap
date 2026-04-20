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
- verification after the refactor:
  `pytest tests -q`
  `47 passed`

## Next Checks

- let the current training run long enough to inspect deterministic and stochastic eval behavior
- use the action / policy-input diagnostics wrappers during eval to confirm the new semantics stay honest in live SC2 traces
- decide whether the next action-space step is:
  action-history token group,
  learnable selection actions,
  or both
- keep the larger branch questions open:
  entity identity, long-term SNN/TBPTT verdict, and whether this remains the mainline or the research branch
