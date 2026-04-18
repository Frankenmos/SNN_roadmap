# Plan: Post-Baseline Execution Order

This file is the short tactical roadmap for the next coding sessions.
It intentionally stays shorter than [RAYPLAN.md](RAYPLAN.md) and
[WHY_YOLO_BAD.md](WHY_YOLO_BAD.md), which hold the long-form reasoning.

If there is any conflict between "what feels exciting" and this file,
follow this file.

---

## Current Status

- A fresh clean baseline run is in progress in WSL:
  `models/run_20260418_164634`
- That run is the first serious reference run on the newer stabilized
  code path.
- Do **not** reshape the current experiment narrative around
  `run_20260417_024205` anymore; use it as a mixed transitional run,
  not as the primary benchmark.

### What the fresh run is for

- confirm whether the current stabilized PPO stack still collapses to
  attack-only behavior
- measure whether kiting survives longer than in the mixed old/new run
- give us one clean baseline before larger architecture changes

### What we should not do while it runs

- do not keep mutating the meaning of reward/logging metrics and then
  compare them directly to the live run
- do not start Ray implementation against the old flat-buffer contract
- do not redesign reward, action space, obs space, and distributed
  runtime all at once

---

## The Order That Matters

The correct order is:

1. finish observing the clean baseline run
2. refactor the **protocol / rollout buffer contract**
3. redesign action space + history handling locally
4. redesign observation stream into tokenized inputs locally
5. retune reward only after the new policy contract exists
6. distribute with Ray
7. extend DB/logging for distributed mode

The key principle:

**Protocol first, tokens second, Ray third.**

Not:

- "DB first"
- "Ray first"
- "ultimate token dream first"

---

## Phase 0: Baseline Readout

### Goal

Use `run_20260418_164634` as the clean answer to:

- does the stabilized trainer still collapse into attack-only?
- do nonfinite gradients still appear?
- does deterministic eval track training reward meaningfully?
- does kiting survive into late training?

### When the run is done, inspect

- action mix over time
- deterministic eval curve
- `ppo_updates.mean_kl`
- `ppo_updates.clip_fraction`
- `ppo_updates.explained_variance`
- `ppo_updates.nonfinite_grad_steps`
- latest and best checkpoint metadata

### Decision gate

- If the clean run is still attack-dominant and basin-collapsing:
  proceed with the policy/input redesign.
- If it is suddenly healthy and stable:
  still proceed carefully, but use it as the true boring baseline.

---

## Phase 1: Protocol / Buffer Overhaul

### This is the next coding phase after the baseline

This is the most important structural change.

### Goal

Replace the current implicit local rollout contract:

- `memory`
- `final_next`
- hardcoded `(spatial_obs, vector_obs, action, move_x, move_y, ...)`

with explicit objects that can survive policy changes.

### New core objects

- `PolicyInputBatch`
- `PolicyOutputBatch`
- `ActionSample`
- `TransitionRecord`
- `RolloutFragment`
- `EpisodeSummary`
- `UpdateSummary`

### Files likely touched

- `PPO_CNN/PPO.py`
- `PPO_CNN_agent.py`
- `PPO_CNN_run.py`
- new module: `distributed/protocol.py`
- maybe `Utility/logger_utils.py`

### Concrete target

- PPO should consume **fragments**, not one flat rollout with one
  global bootstrap tail.
- GAE should be computed per fragment.
- the protocol should allow new action heads and token inputs without
  rewriting the whole trainer.

### Why this comes before tokens

Because once obs/action/history become tokenized, the current
flat-buffer assumptions become even more brittle.

### Bug notes to keep in mind

- `PPO.final_next` is structurally wrong for multi-fragment /
  multi-actor training.
- `ppo_updates.episode_id` logging is already shaky because update
  events are not tied cleanly to episode rows.

---

## Phase 2: Action Space + History Redesign

### Goal

Make the action interface closer to what we actually want to learn,
and make recent action history part of the policy input.

### What to do first

- redesign the action space around explicit primitives / arguments
- add **past sampled actions** as history, not past raw logits

### What not to do first

- do not feed back raw old policy output vectors yet
- do not make the history contract autoregressive and recursive on day 1

### Recommended first history payload

- previous `action_type`
- previous action arguments
- previous `done`
- maybe previous reward bucket / sign

### Files likely touched

- `action_space/action_space.py`
- `PPO_CNN_agent.py`
- `PPO_CNN/PPO.py`
- `tests/test_agent.py`
- `tests/test_PPO.py`

### Success criterion

- new action semantics work in single-process training first
- history is stored and replayed consistently
- no hidden helper behavior steals credit from the policy

---

## Phase 3: Tokenized Observation Stream

### Goal

Move from dense ad-hoc obs handling toward a tokenized policy input
that can carry:

- observation tokens
- history tokens
- masks
- optional available-action context

### Recommended rule

Use **fixed padded token tensors + masks** at the protocol boundary.

Do not start with fully ragged variable-length transport.

### Recommended first token groups

- observation tokens derived from screen/entity summaries
- history tokens from past actions
- optional global/context token

### Files likely touched

- `obs_space/obs_space_2.py` or a replacement module
- `PPO_CNN/policy_network.py`
- protocol dataclasses
- tests for batching / masking

### Success criterion

- token inputs train locally in single-process mode
- replay shape is stable
- logging can still tell us what the agent is doing

---

## Phase 4: Reward Retune

### Goal

Only after the new action/input contract exists, revisit reward shaping.

### Why this is later

If we change reward before action/input redesign, we will not know
whether the gain came from:

- better credit assignment
- better observability
- better action semantics
- or just reward hacking

### Allowed before this phase

- tiny safety fixes
- consistency fixes
- logging cleanups

### Not allowed before this phase

- big reward redesign as the explanation for every behavior failure

---

## Phase 5: Ray / Distributed PPO

### Goal

Distribute the now-stable protocol, not the old local assumptions.

### Start shape

- 1 learner
- N rollout actors
- optional 1 eval actor
- synchronous PPO update boundary

### Use

- fragment-based rollout transport
- learner-owned optimizer
- policy versioning
- single writer for logs

### Companion doc

Read [RAYPLAN.md](RAYPLAN.md) before touching this phase.

---

## Phase 6: Distributed Logging + DB Schema

### Important

This is **not** the first overhaul.
This is a downstream phase.

### Goal

Once the protocol and Ray path exist, extend schema/logging to track:

- `actor_id`
- `policy_version`
- `fragment_id`
- `global_update_index`
- distributed eval rows

### Why this is later

If we change schema first, we still do not know what the distributed
payload even is.

### Companion note

The logger should eventually be organized around events emitted from
actors / learner, not around the current fragile episode-map pattern.

---

## Immediate Coding Priority After The Baseline

When `run_20260418_164634` gives us enough signal, the next actual code
branch should be:

### `protocol-fragment-refactor`

That branch should do only:

1. define protocol objects
2. refactor PPO to fragment-based replay
3. adapt the local trainer to the new protocol
4. keep the current model/action/obs behavior as intact as possible

It should **not** also try to do:

- tokenized obs
- new reward logic
- Ray
- massive DB rewrite

---

## Questions To Re-ask After The Baseline

Once the current run gives us data, answer these before Phase 1 starts:

1. Is attack-only collapse still the dominant failure mode?
2. Are nonfinite gradients still present in the clean run?
3. Is the action redesign urgent enough to do before tokenized obs?
4. Does the current reward still actively favor the wrong local optimum?
5. Is the current checkpoint/eval path trustworthy enough to compare
   future architecture variants?

---

## Companion Docs

- [WHY_YOLO_BAD.md](WHY_YOLO_BAD.md):
  why we should not change everything at once
- [RAYPLAN.md](RAYPLAN.md):
  distributed actor-learner roadmap
- [NEXT_FIXES_PLAN.md](NEXT_FIXES_PLAN.md):
  historical reasoning log for earlier PPO/SNN fixes
- [action_space/action_plan.md](action_space/action_plan.md):
  action-space redesign sketch

---

## One-Line Rule

**Get the clean baseline, then stabilize the protocol, then change what
the policy sees and does, then scale it out.**
