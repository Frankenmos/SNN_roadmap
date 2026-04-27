# Fragment-Based PPO and Memory Management

**Last Updated:** 2026-04-26
**Status:** Implemented locally; initial synchronous Ray path added

This document explains the fragment-based rollout memory management that enables
distributed training with Ray.

---

## Table of Contents

1. [Why Fragments?](#why-fragments)
2. [The RolloutFragment Contract](#the-rolloutfragment-contract)
3. [Data Flow: From Collection to Update](#data-flow-from-collection-to-update)
4. [Per-Fragment GAE Computation](#per-fragment-gae-computation)
5. [Protocol Validation](#protocol-validation)
6. [Migration to Ray](#migration-to-ray)

---

## Why Fragments?

### The Single-Bootstrap Problem

The original PPO implementation stored transitions in a flat list and maintained
**one global bootstrap tail** (`PPO.final_next`):

```python
# Old approach
for step in rollout:
    memory.append(transition)
final_next = last_observation  # One tail for entire rollout
update_policy(memory, final_next)  # GAE computed once
```

This works for single-process collection but breaks with multiple actors:
- Actor A finishes a fragment with a non-terminal tail
- Actor B finishes a fragment with a non-terminal tail
- Each fragment needs its **own bootstrap value** for correct GAE

### The Fragment Solution

`RolloutFragment` carries its own bootstrap tail:

```python
class RolloutFragment:
    ...
    tail_next_policy_input: PolicyInputBatch | None  # Bootstrap for this fragment
    tail_next_snn_state: SNNState | None             # SNN state after tail
```

Now the learner can:
1. Receive N fragments from N actors
2. Compute GAE **per fragment** using each fragment's tail
3. Concatenate after advantages/returns are valid

---

## The RolloutFragment Contract

Located in [`distributed/protocol.py`](../../distributed/protocol.py).

### Core Fields

| Field | Shape | Purpose |
|-------|-------|---------|
| `actor_id` | scalar | Which actor produced this fragment |
| `fragment_id` | scalar | Monotonic fragment index per actor |
| `policy_version` | scalar | Learner update count when fragment was collected |
| `spatial_obs` | `[T, 27, 84, 84]` | Screen features |
| `entity_features` | `[T, 24, 21]` | Unit features |
| `entity_mask` | `[T, 24]` | Valid entity slots |
| `selection_features` | `[T, 20, 7]` | Selected units |
| `selection_mask` | `[T, 20]` | Valid selection slots |
| `action_feedback_tokens` | `[T, 1, 9]` | Previous action + outcome |
| `meta_vec` | `[T, 15]` | Player + available + last-action |
| `actions` | `[T]` | Selected action IDs |
| `move_x`, `move_y` | `[T]` | Decoded click coordinates |
| `target_index` | `[T]` | Token-pointer target index |
| `coarse_index` | `[T]` | Coarse-to-fine coarse index |
| `fine_index` | `[T]` | Coarse-to-fine fine index |
| `old_log_probs` | `[T]` | Policy log-probs at collection time |
| `values` | `[T]` | Critic values at collection time |
| `rewards` | `[T]` | Environment rewards |
| `dones` | `[T]` | Terminal episode flags |
| `truncateds` | `[T]` | Time-limit/timeout flags |
| `episode_reset_mask` | `[T]` | State reset boundaries |
| `sample_mask` | `[T]` | Learnable (non-helper) steps |
| `pre_step_snn_state` | `(syn, mem)` | SNN state at fragment start |
| `tail_next_policy_input` | `PolicyInputBatch` | Bootstrap tail for GAE |
| `tail_next_snn_state` | `(syn, mem)` | SNN state after tail |

### Metadata Fields

| Field | Purpose |
|-------|---------|
| `episode_summaries` | Per-episode reward/step counts |
| `reward_component_summaries` | Per-episode reward breakdowns |
| `step_counters` | Fragment-level step counts |
| `policy_protocol_version` | Must equal `POLICY_PROTOCOL_VERSION = 2` |
| `policy_input_schema` | Must equal `"stream_action_feedback_v1"` |

### Properties

```python
fragment.num_steps           # Total steps in fragment
fragment.num_learnable_steps # Steps with sample_mask > 0
fragment.terminated          # Any done == 1?
fragment.truncated           # Any truncated == 1?
```

### Methods

```python
fragment.as_policy_input_batch(state_in=None)  # Reconstruct PolicyInputBatch
fragment.immutable_step_counters()              # Read-only step counts
```

---

## Data Flow: From Collection to Update

### Current Hybrid Path (Single-Process)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Collection Phase                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Agent.step()                                                                │
│    │                                                                         │
│    ▼                                                                         │
│  PPO.store_transition(obs, action, ...)                                     │
│    │                                                                         │
│    ▼                                                                         │
│  PPO.memory (list[dict])                                                    │
│    │                                                                         │
│    ▼ (episode end or budget)                                                │
│  PPO.set_final_next(next_obs)                                               │
│    │                                                                         │
│    ▼                                                                         │
│  PPO.finalize_fragment()                                                    │
│    │                                                                         │
│    ▼                                                                         │
│  PPO.pending_fragments (list[RolloutFragment])                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          Update Phase                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  train.py calls agent.update_policy(fragments)                              │
│    │                                                                         │
│    ▼                                                                         │
│  PPO.consume_pending_fragments()                                            │
│    │                                                                         │
│    ▼                                                                         │
│  PPO.update_policy(fragments: list[RolloutFragment])                         │
│    │                                                                         │
│    ├─► For each fragment:                                                    │
│    │    _fragment_tensors(fragment)                                         │
│    │    _bootstrap_fragment_tail_value(fragment.tail_next_policy_input)     │
│    │    _compute_advantages(rewards, values, dones, tail_value)             │
│    │                                                                         │
│    ├─► Concatenate all advantages/returns                                    │
│    │                                                                         │
│    └─► TBPTT chunks → PPO loss → optimizer step                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Ray Path (Distributed)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RolloutActor                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Actor runs env loop locally                                               │
│  Accumulates transition data in local buffers                              │
│  When fragment_steps reached or episode boundary occurs:                   │
│    finalize_fragment(actor_id, fragment_id, policy_version)                │
│    return RolloutFragment to learner via Ray object ref                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          Learner                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Publish policy_version V                                                   │
│  Wait for N fragments from actors                                           │
│  Aggregate fragments                                                         │
│  Reject stale fragments where policy_version != V                          │
│  update_policy(fragments)  ← Same method as local                           │
│  Publish policy_version V+1                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key insight:** The **update path is identical**. Ray only changes how fragments are produced.

---

## Per-Fragment GAE Computation

Located in [`agent_core/ppo_trainer.py`](../../agent_core/ppo_trainer.py) `update_policy()`.

### The Algorithm

```python
for fragment in fragments:
    # Get bootstrap value from fragment's tail
    tail_value = _bootstrap_fragment_tail_value(fragment.tail_next_policy_input)

    # Compute GAE over this fragment only
    advantages = _compute_advantages(
        fragment.rewards,
        fragment.values,
        fragment.dones,
        tail_value,  # Fragment-specific bootstrap
    )

    # Returns = advantages + values
    returns = (advantages + fragment.values).detach()

    # Concatenate with other fragments later
```

### Why This Matters

With multi-actor collection:
- Fragment A ends at step 512, episode ongoing
- Fragment B ends at step 512, episode ongoing
- Fragment C ends at step 512, episode ongoing

Each fragment needs **its own tail value** for correct bootstrapping. The old
"one global final_next" cannot represent this.

---

## Protocol Validation

### Version Checking

Every `RolloutFragment` and `WeightSnapshot` validates protocol compatibility:

```python
def __post_init__(self):
    validate_policy_protocol(
        policy_protocol_version=self.policy_protocol_version,
        policy_input_schema=self.policy_input_schema,
    )
```

### Constants (from [`policy_protocol.py`](../../agent_core/policy_protocol.py))

```python
POLICY_PROTOCOL_VERSION: Final[int] = 2
POLICY_INPUT_SCHEMA: Final[str] = "stream_action_feedback_v1"
```

### What Gets Validated

| Check | Purpose |
|-------|---------|
| `policy_protocol_version == 2` | Rejects v1 checkpoints (24-dim meta_vec) |
| `policy_input_schema == "stream_action_feedback_v1"` | Rejects incompatible token layouts |

### Failure Mode

If an actor with stale protocol tries to send data:

```python
ValueError: policy input schema mismatch: 'old_schema' != 'stream_action_feedback_v1'
```

This fails **before** any training happens, preventing silent corruption.

---

## Migration to Ray

### What Stays The Same

| Component | Ray vs Local |
|-----------|--------------|
| `RolloutFragment` structure | Identical |
| `PPO.update_policy(fragments)` | Identical |
| Per-fragment GAE | Identical |
| TBPTT chunking | Identical |
| PPO loss computation | Identical |
| Checkpoint format | Identical |

### What Changes

| Component | Local | Ray |
|-----------|-------|-----|
| Fragment production | `finalize_fragment()` from memory | Actor emits directly |
| Fragment transport | In-memory list | Ray object ref |
| Actor lifecycle | Part of main process | Separate Ray worker |
| Weight distribution | N/A (single process) | Learner → actors via Ray |

### Implemented Ray Pieces

- `distributed/rollout.py` - Ray-free `LocalRolloutWorker`
- `distributed/ray_actor.py` - Ray actor wrapper with `set_weights()` / `collect_fragment()`
- `distributed/learner.py` - Learner coordinator and weight payloads
- `distributed/ray_train.py` - Synchronous 4-actor entrypoint
- `config.yaml distributed:` - default 4 rollout actors, 512-step fragments, 2048-step global batch

### Current Caveats

- Live SC2 smoke is still needed for the 4-actor path.
- Step-level logs are still local-trainer only; Ray logs episode summaries and PPO updates first.
- Extractor normalizers are synced to actors initially, then actor-local until we add normalizer aggregation.
- Windows SC2 can race while creating temporary maps; `distributed.serialize_env_resets`
  serializes `env.reset()`/`create_game` while rollout stepping remains parallel.
- EvalActor / best-checkpoint promotion is not implemented yet.

---

## Related Documentation

- [`RAYPLAN.md`](../ideas/RAYPLAN.md) - Full distributed training plan
- [`THROUGHPUT_PLAN.md`](../ideas/THROUGHPUT_PLAN.md) - Transport and optimization strategy
- [`ARCHITECTURE.md`](ARCHITECTURE.md) - Current system architecture
- [`distributed/protocol.py`](../../distributed/protocol.py) - Fragment implementation
