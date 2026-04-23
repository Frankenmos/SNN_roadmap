
# Spatial Target Head Migration Spec for `BPTT_test`

Audience: GPT-5.4 Pro / Spark 5.3 / any coding agent implementing changes in `Frankenmos/SNN_roadmap`
Branch target: start from `BPTT_test`
Primary goal: replace the current factorized `x` / `y` spatial target with a generic semantic-action + conditional-target interface that can support:
1. token-pointer
2. coarse-to-fine
3. full heatmap

Secondary goal: do this without breaking recurrent replay / PPO bookkeeping / logging.

Status note (2026-04-23):
- Phase 0 is effectively done on the current branch.
- Phase 1 token-pointer is also done and is the current default head.
- Semantic action migration is landed using `NO_OP / LEFT_CLICK / RIGHT_CLICK`.
- `LEFT_CLICK` is scaffolded but intentionally masked unavailable on the current DefeatRoaches wrapper.
- Phase 2 coarse-to-fine and Phase 3 heatmap are still pending.

---

## Ground truth from the current branch

These facts are already true in the inspected branch and should be treated as invariants unless a phase explicitly changes them:

- `train.py` stores recurrent pre-step observation batches in PPO memory and also stores a recurrent bootstrap tail via `set_final_next(...)`.
- `agent.py` uses `DefeatRoaches.step()` to choose a high-level action and, when spatial, dispatches a spatial click action.
- `agent_core/ppo_trainer.py` already replays the policy under the **recorded high-level action IDs**, which is the correct seam to preserve.
- `agent_core/spiking_policy.py` already computes:
  - `latent`
  - `spatial_context`
  - `action_logits`
  - a conditional spatial head
- The current weakness is that the spatial head is still **factorized into `move_x_logits` and `move_y_logits`**.
- `config.yaml` currently sets:
  - `action_dim: 2`
  - `screen_size: 84`
  - `attention_pool_size: 7`
  - `tbptt_window: 32`

Implication: the easiest safe migration path is **not** to rewrite recurrence first. It is to preserve the current replay seam and swap the spatial target parameterization behind that seam.

Comment (2026-04-23):
- this section is now partly historical relative to the live branch
- the branch has already moved to `action_dim: 3`, semantic click actions,
  generic target-head plumbing, and a token-pointer default head
- keep this section as the original baseline the plan was written against

---

# Non-negotiable implementation rules

1. **Do not mix the target-head migration with TBPTT logic changes.**
   - Leave chunk construction, packed replay, and recurrent state resets alone unless a test proves they must change.

2. **Do not explode the action space into `(action_type, x, y)` tuples.**
   - Keep action semantics separate from target geometry.

3. **Do not break existing logs.**
   - Continue logging `act`, `move_x`, `move_y`, even if internal storage becomes index-based.

4. **Do not reuse existing checkpoints once the head changes.**
   - This is a new policy family.

5. **Do not expose two semantic click actions if they still map to the same environment call.**
   - If only one click action is actually distinct right now, mask the second one off until it is real.

6. **Teacher-force recorded targets during PPO replay.**
   - Replay must evaluate the recorded target under the recorded action ID. It must not resample a different target during loss computation.

---

# Recommended branch / commit strategy

Use this exact sequence:

- Phase 0: compatibility refactor with no behavior change
- Phase 1: token-pointer head
- Phase 2: coarse-to-fine head
- Phase 3: full heatmap head

Do **not** skip Phase 0. That phase is what makes the later heads cheap to add.

---

# Phase 0 — compatibility refactor (no behavior change)

> Status (2026-04-23): done in spirit.
> The branch now has the generic target-head seam, legacy factorized support,
> and generalized PPO target plumbing, but it has already moved beyond a pure
> no-behavior-change state because token-pointer is live.

## Objective

Create a generic spatial-target interface while keeping the current behavior as a compatibility head.

At the end of Phase 0:
- policy behavior should still be equivalent to current factorized `x` / `y`
- PPO should no longer be hardcoded to `move_x` + `move_y`
- later heads should only require adding a new target-head implementation, not rewriting PPO again

## Deliverables

### A. Add a generic target-head protocol

> Done.

Create a new file:

- `agent_core/target_heads.py`

Define a small protocol that every head must implement.

Recommended structure:

```python
from dataclasses import dataclass
import torch

@dataclass
class TargetSample:
    primary: torch.Tensor        # [B] long
    secondary: torch.Tensor      # [B] long, use -1 when unused
    x: torch.Tensor              # [B] long
    y: torch.Tensor              # [B] long
    log_prob: torch.Tensor       # [B] float

@dataclass
class TargetEval:
    log_prob: torch.Tensor       # [B] float
    entropy: torch.Tensor        # [B] float

class BaseSpatialTargetHead(nn.Module):
    def forward_head(self, latent, spatial_context, action_ids):
        ...
    def sample_target(self, head_state, action_ids, deterministic: bool) -> TargetSample:
        ...
    def evaluate_target(self, head_state, action_ids, primary, secondary) -> TargetEval:
        ...
    def encode_xy(self, x, y) -> tuple[torch.Tensor, torch.Tensor]:
        ...
    def decode_xy(self, primary, secondary) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

### B. Implement a compatibility head

> Done, with slightly different naming in the live code.

Inside `agent_core/target_heads.py`, implement:

- `FactorizedXYCompatHead(BaseSpatialTargetHead)`

This head should reproduce current behavior as closely as possible:
- it should accept `latent`, `spatial_context`, `action_ids`
- it should produce `move_x_logits` and `move_y_logits`
- `primary = x`, `secondary = y`
- `decode_xy(primary, secondary)` returns `(primary, secondary)`
- `encode_xy(x, y)` returns `(x, y)`

This keeps behavior unchanged while the surrounding code is generalized.

### C. Make `PolicyNetwork` use a pluggable target head

> Done for `factorized_xy` and `token_pointer`.

Edit:

- `agent_core/spiking_policy.py`

#### Add config keys
Extend `self._config` and config parsing with:

- `spatial_head_type`
- `coarse_grid_size`
- `local_grid_size`

For Phase 0, default:

- `spatial_head_type = "factorized_xy"`

#### Instantiate the head
In `PolicyNetwork.__init__`, create:

```python
self.target_head = build_spatial_target_head(
    head_type=...,
    embed_dim=self._embed_dim,
    screen_size=self.screen_size,
    pool_size=self._pool_size,
    action_dim=self._action_dim,
)
```

`build_spatial_target_head(...)` should live in `agent_core/target_heads.py`.

#### Replace `conditioned_spatial_head(...)`
Do not delete it immediately if existing tests expect it.
Instead:
- either keep it as a compatibility wrapper,
- or replace call sites to use the new interface.

New preferred seam:

```python
head_state = self.target_head.forward_head(latent, spatial_context, action_ids)
sample = self.target_head.sample_target(head_state, action_ids, deterministic)
eval_out = self.target_head.evaluate_target(head_state, action_ids, primary, secondary)
```

### D. Generalize PPO storage format

> Done, but with a slightly more concrete payload:
> the live branch stores `target_index / coarse_index / fine_index` plus
> logged `move_x / move_y`.

Edit:

- `agent_core/ppo_trainer.py`

Current transition payload stores:
- `action`
- `move_x`
- `move_y`
- `log_prob`
- `reward`
- `value`
- `done`

Replace this with:
- `action`
- `target_primary`
- `target_secondary`
- `target_x`
- `target_y`
- `log_prob`
- `reward`
- `value`
- `done`

Notes:
- `target_x` and `target_y` are for logs / DB continuity
- `target_primary` and `target_secondary` are the true training payload
- for heads that only need one index, use `secondary = -1`

### E. Generalize PPO action/target math

> Done for the current factorized + token-pointer heads.

Still in `agent_core/ppo_trainer.py`:

#### Replace `_spatial_action_mask(...)`
Current code assumes one specific spatial action ID.
Replace with a set-based check.

Recommended helper:

```python
def _spatial_action_mask(self, actions: torch.Tensor) -> torch.Tensor:
    spatial_ids = torch.tensor(self.policy_net.spatial_action_ids, device=actions.device)
    return (actions[..., None] == spatial_ids).any(dim=-1).float()
```

The policy should expose `self.spatial_action_ids`.

#### Update `select_action(...)`
After sampling `action`:
- if action is non-spatial:
  - set `primary = -1`
  - set `secondary = -1`
  - set `x = 0`, `y = 0`
  - target log-prob contribution = `0`
- if action is spatial:
  - build target head state
  - sample target
  - combine log-prob:
    - `action_log_prob + target_log_prob`

Return:
- `action_id`
- `target_primary`
- `target_secondary`
- `target_x`
- `target_y`
- `log_prob`
- `value`
- `next_state`

#### Update `_calculate_losses(...)`
Replace direct `move_x_dist` / `move_y_dist` logic with:

- action distribution (always)
- target evaluation only for spatial actions

Loss contract:
- non-spatial:
  - PPO ratio is based on action log-prob only
- spatial:
  - PPO ratio is based on action + target log-prob

Entropy contract:
- always include action entropy
- only add target entropy for spatial samples

### F. Update `agent.py`

> Done.

Edit `DefeatRoaches.step()` and `update_policy()` call plumbing.

Replace the returned tuple from `ppo.select_action(...)` so `agent.step()` now receives:
- `action`
- `target_primary`
- `target_secondary`
- `x`
- `y`
- `log_prob`
- `value`
- `next_state`

Use `x` and `y` only for dispatch into `ActionSpace`.

### G. Update `train.py`

> Done.

In `train_agent(...)`, change the stored transition call to:

```python
agent.ppo.store_transition(
    policy_input,
    action_tensor,
    target_primary_tensor,
    target_secondary_tensor,
    log_prob_tensor,
    reward_tensor,
    value_tensor,
    done_tensor,
    target_x_tensor,
    target_y_tensor,
    sample_mask=...
)
```

If you want to avoid changing `store_transition(...)` too much, `target_x` and `target_y` can be computed again later from `primary/secondary`; but keeping them explicit is better for logs.

### H. Phase 0 tests

> Done for the current branch shape.

Add or update tests in:
- `tests/test_PPO.py`
- `tests/test_agent.py`

Required tests:
1. factorized compatibility head reproduces the old encode/decode contract
2. non-spatial action has zero target log-prob contribution
3. spatial action combines action and target log-prob
4. replay evaluates recorded `primary/secondary`, does not resample them
5. `target_x` / `target_y` still flow through logs

### I. Phase 0 acceptance criteria

Do not start Phase 1 until all of these are true:
- all existing PPO tests pass
- old behavior is still reachable with `spatial_head_type="factorized_xy"`
- logs still show `move_x` / `move_y`
- TBPTT replay tests still pass without modification to chunk logic

---

# Phase 1 — token-pointer head

## Objective

> Status (2026-04-23): done.
> This is the current default spatial head on the branch.

Replace factorized `x` / `y` with a **joint coarse 2D target** over the existing `7 x 7` spatial token grid.

This is the first real new head and the safest one to implement because the current architecture already computes a `7 x 7` spatial context.

## Deliverables

### A. Implement `TokenPointerHead`

> Done, with current code naming based on `TokenPointerTargetHead`.

File:
- `agent_core/target_heads.py`

New class:
- `TokenPointerHead(BaseSpatialTargetHead)`

#### Input
- `latent`: `[B, 64]`
- `spatial_context`: `[B, D, 7, 7]`
- `action_ids`: `[B]`

#### Internal representation
Flatten `spatial_context` to `[B, 49, D]`.

Build an action-conditioned query using:
- latent embedding
- action embedding

Recommended scoring:
- project query to `[B, D]`
- score each token by dot product or small MLP against each spatial token

Return:
- `token_logits: [B, 49]`

### B. Encoding / decoding contract

> Done.
> Current branch uses center decoding on the `7 x 7` pooled grid.

Assuming:
- `screen_size = 84`
- `pool_size = 7`
- `cell_size = 12`

For a sampled `token_index`:
- `row = token_index // 7`
- `col = token_index % 7`
- `x = col * 12 + 6`
- `y = row * 12 + 6`

Use cell centers for the first implementation.

Encoding from `x, y`:
- `col = clamp(x // 12, 0, 6)`
- `row = clamp(y // 12, 0, 6)`
- `primary = row * 7 + col`
- `secondary = -1`

### C. PPO integration

> Done.

No further PPO redesign should be needed after Phase 0.

For this head:
- `target_primary = token_index`
- `target_secondary = -1`

### D. Agent / train integration

> Done.

No new interface changes if Phase 0 was done correctly.
`agent.step()` should receive:
- sampled absolute `x, y`
- and dispatch them exactly like before

### E. Tests

> Done for the token-pointer path currently in the repo.

Add:
1. exact encode/decode roundtrip from token index to center coordinate
2. `encode_xy(...)` maps any pixel into the expected coarse cell
3. target log-prob and entropy shape checks
4. replay teacher-forcing test for recorded token index

### F. Acceptance criteria

The head is ready to compare when:
- token-pointer head runs end-to-end without touching PPO core again
- logged `move_x`, `move_y` match token centers
- training is stable for at least a smoke test rollout
- deterministic evaluation produces visibly more coherent click regions than the old factorized head

---

# Phase 2 — coarse-to-fine head (recommended long-term default)

## Objective

> Status (2026-04-23): not started yet.

Restore full 84 x 84 click precision while keeping the computational structure aligned with the current 7 x 7 spatial token grid.

This should be the mainline target head once Phase 1 is stable.

## Deliverables

### A. Implement `CoarseToFineHead`

File:
- `agent_core/target_heads.py`

New class:
- `CoarseToFineHead(BaseSpatialTargetHead)`

#### Stage 1: coarse cell
Exactly like `TokenPointerHead`:
- produce `coarse_logits: [B, 49]`

#### Stage 2: local refinement
After a coarse cell is selected:
- gather the chosen coarse token feature `[B, D]`
- fuse it with:
  - latent
  - action embedding
- produce `fine_logits: [B, 144]` for a `12 x 12` local offset

Recommended first implementation:
- simple MLP refinement head, not a new local conv tower

### B. Encoding / decoding contract

For absolute `(x, y)`:
- `coarse_col = x // 12`
- `coarse_row = y // 12`
- `coarse_index = coarse_row * 7 + coarse_col`

- `fine_col = x % 12`
- `fine_row = y % 12`
- `fine_index = fine_row * 12 + fine_col`

Decode:
- `x = coarse_col * 12 + fine_col`
- `y = coarse_row * 12 + fine_row`

This is exact and invertible.

### C. Replay rule (critical)

During acting:
- sample coarse cell
- sample fine offset conditioned on the sampled coarse cell

During replay:
- compute coarse logits
- evaluate the **recorded** coarse index
- compute fine logits conditioned on the **recorded** coarse index
- evaluate the **recorded** fine index

Do **not** resample the coarse cell during replay.

### D. PPO integration

For this head:
- `target_primary = coarse_index`
- `target_secondary = fine_index`

Target log-prob:
- `coarse_log_prob + fine_log_prob`

Target entropy:
- `coarse_entropy + fine_entropy`

Total PPO log-prob for spatial actions:
- `action_log_prob + coarse_log_prob + fine_log_prob`

### E. Tests

Required:
1. exact roundtrip for all representative `(x, y)` points
2. replay teacher-forcing uses recorded coarse cell for fine-head evaluation
3. wrong coarse cell changes fine logits as expected
4. non-spatial actions bypass both coarse and fine losses

### F. Acceptance criteria

This should become the default head when:
- encode/decode is exact
- replay math is stable
- smoke training shows sharper click placement than token-pointer
- performance is at least as stable as token-pointer

---

# Phase 3 — full heatmap head

## Objective

> Status (2026-04-23): not started yet.

Provide the cleanest pure joint-2D target distribution:
- one categorical distribution over all `84 x 84 = 7056` positions

This is the highest-capacity version, but not the first one to ship.

## Deliverables

### A. Implement `HeatmapHead`

File:
- `agent_core/target_heads.py`

New class:
- `HeatmapHead(BaseSpatialTargetHead)`

#### Suggested architecture
Start from:
- `spatial_context: [B, D, 7, 7]`
- add latent bias + action bias as current head already does conceptually
- run a small conv tower
- upsample to `[B, D, 84, 84]`
- apply final `1 x 1` conv to get `[B, 1, 84, 84]`
- flatten to `[B, 7056]`

### B. Encoding / decoding contract

Encode:
- `primary = y * 84 + x`
- `secondary = -1`

Decode:
- `x = primary % 84`
- `y = primary // 84`

### C. PPO integration

For this head:
- `target_primary = flat_index`
- `target_secondary = -1`

Target log-prob:
- `heatmap_log_prob`

Target entropy:
- `heatmap_entropy`

Total PPO log-prob for spatial actions:
- `action_log_prob + heatmap_log_prob`

### D. Numerical precautions

Because entropy is now over 7056 classes:
- log entropy separately from action entropy
- consider normalizing or monitoring it before tuning coefficients
- verify AMP stability before large runs

### E. Tests

Required:
1. flat-index encode/decode roundtrip
2. sampled heatmap target decodes correctly
3. replay evaluates recorded flat index
4. entropy magnitude remains finite under AMP

### F. Acceptance criteria

Only promote this head if:
- training remains numerically stable
- compute cost is acceptable
- it materially improves targeting over coarse-to-fine

---

# Semantic action migration plan

> Status (2026-04-23): done, with slightly different final naming.
> The current branch uses `NO_OP / LEFT_CLICK / RIGHT_CLICK` instead of
> `PRIMARY / SECONDARY`, and only `RIGHT_CLICK` is currently live on the
> DefeatRoaches wrapper.

This is separate from the head change but should be staged early so future action tokens fit naturally.

## Recommendation

Do **not** hardcode `SMART` forever.

Instead, create semantic policy action IDs such as:
- `POLICY_ACTION_NO_OP`
- `POLICY_ACTION_PRIMARY_CLICK`
- `POLICY_ACTION_SECONDARY_CLICK`

If the environment is not yet ready to distinguish two real click types:
- expose only one as available
- keep the other masked off

This avoids action aliasing while keeping the code future-proof.

## Files to edit

### `agent_core/policy_protocol.py`
Add:
- semantic action IDs
- `SPATIAL_ACTION_IDS`
- availability helpers

### observation / meta construction
Wherever `meta_vec` packs available actions:
- expose semantic action availability, not raw PySC2 availability only

### `action_space/action_space.py`
Add or generalize:
- `dispatch(action_id, x, y, obs)`
- `primary_click(...)`
- `secondary_click(...)`
- leave `bootstrap_select_army(...)` alone

### `agent.py`
Replace:
- `if action == POLICY_ACTION_SMART: ...`
with:
- semantic dispatch using `dispatch(...)`

---

# Concrete file-by-file edit list

This is the surgical checklist a coding agent should follow.

## 1. `config.yaml`
Add:
- `model.spatial_head_type`
- `model.coarse_grid_size`
- `model.local_grid_size`

Status:
- done for Phase 0 + Phase 1
- current live defaults are `action_dim: 3`, `vector_input_dim: 19`,
  `spatial_head_type: token_pointer`, `coarse_grid_size: 7`,
  `local_grid_size: 12`, `target_decode_mode: center`

Phase defaults:
- Phase 0: `factorized_xy`
- Phase 1: `token_pointer`
- Phase 2: `coarse_to_fine`
- Phase 3: `heatmap`

## 2. `agent_core/target_heads.py`
New file.
Implement:
- `TargetSample`
- `TargetEval`
- `BaseSpatialTargetHead`
- `FactorizedXYCompatHead`
- `TokenPointerHead`
- `CoarseToFineHead`
- `HeatmapHead`
- `build_spatial_target_head(...)`

Status:
- partially done
- shared target-head dataclasses / base protocol, legacy factorized support,
  and token-pointer are landed
- coarse-to-fine and heatmap are still pending

## 3. `agent_core/policy_protocol.py`
Refactor constants:
- semantic action IDs
- spatial action set
- helper predicates if helpful

Status:
- done

## 4. `agent_core/spiking_policy.py`
Changes:
- parse new config
- instantiate `self.target_head`
- expose `self.spatial_action_ids`
- replace direct x/y head usage with generic head calls

If keeping backward compatibility temporarily:
- keep old `conditioned_spatial_head(...)` only as an internal helper for `FactorizedXYCompatHead`

Status:
- mostly done
- the generic target-head path is live
- compatibility behavior is still available, but the branch no longer centers
  its main path around the old direct `conditioned_spatial_head(...)` contract

## 5. `agent_core/ppo_trainer.py`
Changes:
- generalize spatial-action mask
- generalize action selection return payload
- store `target_primary/secondary`
- evaluate target via target head
- keep recurrent replay logic otherwise unchanged

Status:
- done for the current heads
- payload naming differs slightly from this draft in the live code

## 6. `agent.py`
Changes:
- consume new action-selection payload
- use `x, y` only for environment dispatch
- separate bootstrap helper path from learnable policy path as today

Status:
- done

## 7. `train.py`
Changes:
- store new target payload in PPO memory
- keep DB/log outputs backward-compatible with `move_x`, `move_y`

Status:
- done

## 8. `action_space/action_space.py`
Changes:
- semantic dispatch layer
- action token export reflects semantic click types

Status:
- done

## 9. observation / wrappers
Likely files:
- `obs_space/obs_space_2.py`
- wrappers that populate available action metadata

Changes:
- semantic availability bits for policy actions
- do not leak two identical click semantics if they are not truly distinct

Status:
- done for the current wrapper
- semantic availability is live and the second click action is intentionally
  masked off to avoid aliasing

## 10. tests
Edit:
- `tests/test_PPO.py`
- `tests/test_agent.py`
- optionally `tests/test_training_loop.py`

Add new focused tests before any long training run.

Status:
- done for the landed Phase 0 + Phase 1 work
- the full current suite passes on the branch

---

# Test matrix by phase

## Phase 0
- compatibility head encode/decode
- target log-prob plumbing
- non-spatial bypass
- recurrent replay still passes

## Phase 1
- token index ↔ center coordinate
- teacher-forced token replay
- smoke train with token pointer

## Phase 2
- exact coarse/fine ↔ pixel roundtrip
- fine-head conditioned on recorded coarse cell
- smoke train with coarse-to-fine

## Phase 3
- flat index ↔ pixel roundtrip
- heatmap replay
- AMP / entropy sanity

---

# Run-order instructions for a coding model

Use this exact order.

1. Create `agent_core/target_heads.py`
2. Wire in `FactorizedXYCompatHead`
3. Switch PPO to generic target payload
4. Make existing tests pass
5. Add token-pointer head
6. Add token-pointer tests
7. Add coarse-to-fine head
8. Add coarse-to-fine tests
9. Add heatmap head
10. Add heatmap tests
11. Only then consider changing `action_dim` or exposing a second click action if the environment actually supports it cleanly

---

# What should NOT change in this migration

- reward function logic
- recurrent state structure `(syn, mem)`
- TBPTT chunk segmentation
- rollout bootstrapping contract
- logging schema names (`act`, `move_x`, `move_y`) unless there is a compelling reason

---

# Final recommendation

If only one version is likely to become the keeper, it is:

- Phase 0 refactor
- then Phase 2 coarse-to-fine

But do not jump straight to coarse-to-fine. Use token-pointer first to validate that the interface and replay math are correct.

That will save debugging time.
