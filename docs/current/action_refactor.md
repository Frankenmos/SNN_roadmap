# Action Refactor Status

Updated: 2026-04-21

This file is now a status note, not a speculative plan. Stage 1 of the
action refactor has landed.

## What Landed

### Action semantics

- policy action IDs stay bounded to `NO_OP=0`, `MOVE=1`, `ATTACK=2`
- dispatch is explicit and interpretable:
  - `MOVE -> Move_screen(x, y)`
  - `ATTACK -> Attack_screen(x, y)`
  - `NO_OP -> no_op()`
- `Smart_screen` is no longer used in the learned path
- the old scripted nearest-enemy attack heuristic is gone from normal policy control
- the old mid-episode `select_army` fallback is gone from normal control flow

### Reset bootstrap

- one non-learned bootstrap selection step remains at episode start
- that step happens outside PPO memory
- its only purpose is to reach the normal selected-army state where `MOVE` and `ATTACK` become available

### Bridge token / observation protocol

- `meta_vec` grew from `28` to `32`
- the extra 4 floats are an agent-owned executed-action bridge token:
  `[type_id, x_norm, y_norm, extra]`
- named offsets now define the bridge-token slice in `agent_core/policy_protocol.py`
- the bridge token records the executed action, not the policy's intent
- a reserved helper token type is used for the reset bootstrap selection
- the original PySC2 `last_actions` function ID is still kept separately in `meta_vec`

### Policy / PPO / TBPTT semantics

- the recurrent trunk now runs once per step
- the action head samples `action_type` first
- the spatial head then reuses the same latent and conditions on the chosen action type
- availability masking now gates the 3 policy actions from the `available_actions` slice in `meta_vec`
- both `MOVE` and `ATTACK` are treated as spatial for log-probability and entropy accounting
- rollout storage shape did not change; replay just conditions spatial logits on stored action IDs

### Runtime / fidelity cleanup

- fake PySC2 test IDs were aligned with real DefeatRoaches IDs so availability masking is truthful in tests
- the real PySC2 `FunctionCall` runtime shape is handled robustly:
  matcher logic now checks `.function`, then `.id`, then `.name` for fake/test objects

## Validation

- unit test suite after the refactor:
  `pytest tests -q`
- result:
  `47 passed`
- live runtime training now starts successfully after the `FunctionCall` compatibility fix

## Post-Run Read

The first real training pass after landing Stage 1 suggests the refactor
itself is mostly wired correctly, but it did **not** by itself solve
deterministic behavior.

What the diagnostics say:

- after the reset bootstrap, `Move_screen` and `Attack_screen` are
  available on >99% of logged eval steps
- deterministic traces are heavily `NO_OP` + `ATTACK` and almost never
  `MOVE`
- stochastic traces do use all three policy actions, including
  `MOVE`

Current interpretation:

- the Stage-1 action path is not mainly failing because the env is
  starving the policy of `MOVE` / `ATTACK`
- the bigger immediate problem looks like reward shaping and policy
  preference, not missing Stage-2 action vocabulary
- action-space expansion should stay deferred until reward refactor has
  had a chance to change the learned behavior

## Current Invariants

- learned policy vocab stays exactly 3-way:
  `NO_OP`, `MOVE`, `ATTACK`
- bridge-token type space is slightly wider than policy vocab only to represent the reset bootstrap helper honestly
- reset bootstrap is allowed only at episode start
- `select_rect` and other broader action-space tokens are still deferred
- old checkpoints from before this patch are incompatible

## What Remains For Stage 2+

- immediate next work is **not** automatically Stage 2:
  reward refactor now looks more urgent than broader action-space work
- replace the 4-float bridge token with a dedicated action-history token group
- add learnable selection actions such as `SELECT_POINT` / `SELECT_RECT`
- decide whether broader action vocab items should stay explicit or move toward richer token-conditioned interaction semantics
- revisit whether the bridge token still earns its keep once longer action history is available in the attention stream

## Archive Pointers

- archived external ideation and draft code:
  `docs/archive/observations_2026-04-20/`
- archived verbose implementation memory before working-log compression:
  `docs/archive/working_log_2026-04-20_pre_compress.md`
