# Ray Implementation Status

> Archived 2026-06-26.
>
> This status file predates the Ray eval/best-checkpoint and extractor
> normalizer repairs. The current concise Ray status lives at
> `docs/current/RAY_STATUS.md`.

**Last Updated:** 2026-05-06
**Current Phase:** Initial synchronous configurable-actor Ray path implemented;
live SC2 smoke/throughput validation still needed for the tuned config

This document tracks the implementation status of the distributed training
migration to Ray, following the plan in [`RAYPLAN.md`](../ideas/RAYPLAN.md).

---

## Phase Overview

| Phase | Name | Status | Completion Date |
|-------|------|--------|------------------|
| 0 | Fragment-based single-process | ✅ Complete | 2026-04-26 |
| 1 | RolloutActor | ✅ Initial | 2026-04-26 |
| 2 | LearnerCoordinator | ✅ Initial | 2026-04-26 |
| 3 | Checkpoint/Resume learner-owned | ⚠️ Partial | 2026-04-26 |
| 4 | LoggerActor | Pending | - |
| 5 | EvalActor | Pending | - |
| 6 | Distributed config | ✅ Initial | 2026-04-26 |
| 7 | Test ladder | ⚠️ Partial | 2026-04-26 |

---

## Phase 0: Fragment-Based Single-Process ✅

**Goal:** Refactor the learner to consume fragments locally, establishing the data contract for Ray.

**Status:** Complete

### What Was Implemented

1. **Transport Objects** ([`distributed/protocol.py`](../../distributed/protocol.py))
   - ✅ `TransitionRecord` - Single transition data
   - ✅ `RolloutFragment` - Dense rollout fragment with bootstrap tail
   - ✅ `EpisodeSummary` - Episode-level metadata
   - ✅ `WeightSnapshot` - Learner weights with protocol validation
   - ✅ `UpdateSummary` - Update metadata for logging
   - ✅ `validate_policy_protocol()` - Protocol compatibility checker

2. **Fragment Protocol** ([`policy_protocol.py`](../../agent_core/policy_protocol.py))
   - ✅ `POLICY_PROTOCOL_VERSION = 3`
   - ✅ `POLICY_INPUT_SCHEMA = "stream_action_effect_feedback_v2"`
   - ✅ Fragment validation in `__post_init__`

3. **PPO Refactor** ([`ppo_trainer.py`](../../agent_core/ppo_trainer.py))
   - ✅ `finalize_fragment()` - Converts memory list to `RolloutFragment`
   - ✅ `consume_pending_fragments()` - Retrieves fragments for update
   - ✅ `update_policy(fragments)` - Accepts `list[RolloutFragment]`
   - ✅ Per-fragment GAE with `_bootstrap_fragment_tail_value()`
   - ✅ `update_count` / `global_update_index` tracking

4. **Logging Extensions** ([`logger_utils.py`](../../Utility/logger_utils.py))
   - ✅ SQLite schema: `steps.actor_id`, `steps.policy_version`, `steps.fragment_id`
   - ✅ SQLite schema: `ppo_updates.global_update_index`
   - ✅ SQLite schema: `episodes.actor_id`, `episodes.policy_version`
   - ✅ SQLite schema: `eval_runs.policy_version` + protocol columns
   - ✅ Fixed `_safe_add_column()` to re-raise real schema errors

5. **Checkpoint Extensions** ([`train.py`](../../train.py), [`eval.py`](../../eval.py))
   - ✅ Checkpoints save `policy_input_schema`
   - ✅ Checkpoints save `policy_version` / `global_update_index`
   - ✅ Eval validates checkpoint protocol before loading

6. **Local Path Preservation**
   - ✅ Single-process training loop still works
   - ✅ Collection uses `memory` list, finalizes to fragments
   - ✅ Update consumes fragments directly
    - ✅ Ray dependencies isolated to the distributed entrypoint/actor layer

### Acceptance Criteria (from RAYPLAN.md §7)

- ✅ Current single-process training still runs
- ✅ PPO update path works from fragments
- ✅ Existing smoke tests still pass (local subset verified after Ray changes)
- ✅ Fragment schema preserves `PolicyInputBatch` contract
- ✅ Fragment schema preserves target-head indices
- ✅ `policy_protocol_version` and `policy_input_schema` validated
- ✅ Learner computes GAE per fragment
- ✅ Time-limit resets separated from terminal `done`

### Files Modified

- `distributed/protocol.py` (new)
- `distributed/__init__.py` (new)
- `agent_core/policy_protocol.py`
- `agent_core/ppo_trainer.py`
- `Utility/logger_utils.py`
- `train.py`
- `eval.py`
- `tests/test_distributed_protocol.py` (new)

---

## Phase 1: RolloutActor ✅ Initial

**Goal:** Build one real rollout actor that owns one environment and returns fragments.

**Status:** Initial implementation complete

### Deliverables

- ✅ `distributed/rollout.py` - Ray-free `LocalRolloutWorker`
- ✅ `distributed/ray_actor.py` - Ray actor wrapper
- ✅ Actor API: `set_weights()`, `collect_fragment()`, `get_stats()`, `close()`
- ⏳ `distributed/runtime.py` - Not needed yet; entrypoint owns runtime setup

### Actor Rules (from RAYPLAN.md §8)

- ✅ Env lives forever inside actor (unless crash)
- ✅ Policy replica is inference-only
- ✅ Policy weights copied from learner snapshots
- ✅ Actor keeps local `snn_state`, extractor history, reward state
- ✅ Actor keeps local action-feedback state, resets with episode lifecycle
- ✅ Records terminal `done`, timeout `truncated`, episode-reset boundaries separately
- ✅ Returns CPU tensors inside `RolloutFragment`
- ⚠️ Extractor normalizers are actor-local after initial sync; aggregation is future work

### Acceptance Criteria

- ⚠️ One actor can collect fragment through the Ray API: implemented, live smoke pending
- ✅ Fragment construction contains valid bootstrap tail
- ✅ Fragment can finish early at terminal/time-cap boundary
- ✅ Actor reset path is implemented after terminal/time-cap episodes
- ⏳ Needs live SC2 smoke with 1 actor, 4 actors, and the current 10-actor config

---

## Phase 2: LearnerCoordinator ✅ Initial

**Goal:** Build the learner that owns optimizer, aggregates fragments, and publishes weights.

**Status:** Initial implementation complete

### Deliverables

- ✅ `distributed/learner.py` - Learner implementation
- ✅ `distributed/ray_train.py` - Synchronous Ray training entrypoint
- ✅ CLI: `python -m distributed.ray_train --num-actors 4 --max-updates N`

### Learner Responsibilities

- ✅ Master `PolicyNetwork`
- ✅ `PPO` optimizer and scheduler
- ✅ Checkpoint save/load through existing `train.py` helpers
- ✅ Policy versioning
- ✅ Rollout aggregation
- ✅ GAE / return computation via fragment PPO
- ✅ Minibatch PPO updates
- ⏳ Best-model selection waits for EvalActor

### Acceptance Criteria (from RAYPLAN.md §9)

- ✅ Actor count and fragment size are config-driven
- ✅ Learner update count advances once per global batch
- ✅ Actor fragments are rejected if `policy_version` is stale
- ✅ Scheduler keys off learner update count
- ⏳ Needs live SC2 smoke for 4-actor and current 10-actor configs

---

## Phase 3: Checkpoint/Resume Learner-Owned

**Goal:** Make checkpointing learner-owned and actor-light.

**Status:** Partial

### Checkpoint Should Include

- ✅ Learner policy weights
- ✅ Optimizer state
- ✅ Scheduler state
- ✅ Global update count / policy version
- ⚠️ Global environment step count not yet checkpointed
- ✅ Best eval reward field preserved
- ✅ `policy_protocol_version`
- ✅ `policy_input_schema`
- ✅ Run config snapshot uses the active config path

### Checkpoint Should NOT Depend On

- Live actor object IDs
- Actor-local episode counters
- Actor-local action-feedback buffers
- Actor-local env instances

### Acceptance Criteria (from RAYPLAN.md §10)

- Stop/resume preserves optimizer/scheduler
- Actor count can change between resumes
- Best checkpoint logic still works

---

## Phase 4: LoggerActor

**Goal:** Replace local queue/process logging with distributed writer.

**Status**: Pending

### Deliverables

- `distributed/logger_actor.py`
- Event schema definitions

### Why Not Direct SQLite?

- Concurrent multi-process writes from many actors are fragile
- Single-writer via actor is safer
- Everyone emits plain event dicts
- Logger actor owns SQLite connection

### Acceptance Criteria (from RAYPLAN.md §11)

- Many actors can emit logs without DB corruption
- Analysis can slice by actor and policy version
- Update rows not tied to fragile episode map

---

## Phase 5: EvalActor

**Goal:** Dedicated evaluation actor.

**Status**: Pending

### Why Separate?

- Deterministic eval should not perturb rollout timing
- Eval resets not mixed into rollout actors
- Eval cadence tied to learner updates, not actor episode count

### Eval Behavior

- Receives latest weights and `policy_version`
- Runs deterministic episodes only
- Logs mean/std/min/max reward
- Never contributes to training

### Acceptance Criteria (from RAYPLAN.md §12)

- Eval can lag training without harming correctness
- Best-checkpoint promotion tied to eval results

---

## Phase 6: Distributed Config

**Goal:** Make distributed config explicit.

**Status**: Initial implementation complete

### Current Config Section

```yaml
distributed:
  enabled: false
  num_rollout_actors: 10
  num_eval_actors: 0
  fragment_steps: 256
  global_rollout_steps: 2560
  learner_device: "cuda"
  actor_device: "cpu"
  sc2_runtime_profile: "linux_headless"
  serialize_env_resets: true
  ray_local_mode: false
  object_store_memory_gb: 8
  actor_cpus: 1
  learner_gpus: 1
  eval_every_updates: 50
  max_updates: 0
  required_policy_protocol_version: 3
  required_policy_input_schema: "stream_action_effect_feedback_v2"
```

### Acceptance Criteria (from RAYPLAN.md §13)

- ✅ Ray run launches with repo-root/config path propagation
- ✅ Actor count and fragment size config-driven
- ✅ Actors fail fast on protocol mismatch
- ✅ Local single-process mode still works from same config
- ✅ Env resets can be serialized to avoid Windows SC2 temp-map races

---

## Phase 7: Test Ladder

**Goal:** Add tests from pure data logic to Ray smoke.

**Status**: Partial

### Test Layers

**Layer A: Pure unit tests**
- Fragment schema round-trip
- Fragment preserves `PolicyInputBatch` tensors
- Multi-fragment GAE correctness
- Bootstrap handling (terminal vs truncated)
- Reset-mask handling across episode boundaries
- Policy-version tagging

**Layer B: Local integration (no Ray)**
- Learner consumes two locally fabricated fragments
- Checkpoint/resume with fragment-based PPO
- Logger ingests fragment/event payloads

**Layer C: Ray smoke tests**
- One learner + one actor
- One learner + two actors
- Actor respawn path
- Deterministic eval actor path

**Layer D: Performance sanity**
- Steps/sec vs single-process baseline
- Learner idle time
- Object payload size

### Acceptance Criteria (from RAYPLAN.md §15)

- All Layer A tests pass
- Layer B integration passes
- Layer C smoke tests pass
- Throughput meaningfully higher than baseline

---

## Progress Summary

**Completed:**
- Fragment protocol defined and validated
- PPO refactored for fragment-based updates
- Per-fragment GAE computation
- Protocol versioning and schema validation
- Logging extended for actor/version metadata
- Checkpoint metadata extended
- Ray-free rollout collector added
- Ray rollout actor wrapper added
- Learner coordinator added
- Synchronous configurable-actor Ray entrypoint added

**In Progress:**
- Live SC2 smoke for `python -m distributed.ray_train --num-actors 4`, followed
  by the current 10-actor config

**Next Steps:**
1. Run 1-update live smoke: `python -m distributed.ray_train --num-actors 4 --max-updates 1`
2. Measure rollout/update wall time and object store memory pressure
3. Add extractor normalizer aggregation or make the actor-local choice explicit
4. Add EvalActor / best-checkpoint promotion

---

## Related Documentation

- [`RAYPLAN.md`](../ideas/RAYPLAN.md) - Full distributed training plan
- [`THROUGHPUT_PLAN.md`](../ideas/THROUGHPUT_PLAN.md) - Transport and throughput strategy
- [`FRAGMENT_PPO.md`](FRAGMENT_PPO.md) - Fragment-based memory management
- [`ARCHITECTURE.md`](ARCHITECTURE.md) - Current system architecture
