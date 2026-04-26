# THROUGHPUT_PLAN.md

Less dramatic name for the "need for speed" doc.

This is the performance companion to `RAYPLAN.md` and `ARCHITECTURE.md`.
The goal is not to optimize before the distributed trainer exists. The
goal is to keep the first Ray version from accidentally turning every
SC2 step into a cloud of tiny Python objects and huge surprise copies.

## Status (2026-04-26)

The first transport-prep slice is in place: typed fragment dataclasses,
policy protocol/schema validation, actor/version/protocol log fields, and
checkpoint schema validation. The learner still uses the local `PPO.memory`
list, so the next correctness step is fragment-based PPO/GAE in the
single-process trainer before adding Ray workers.

---

## 1. The Funny Part

The current data structures are both good and extremely honest:

- `PolicyInputBatch` is a proper `dataclass(slots=True)` with tensor
  validation, stacking, slicing, device transfer, and recurrent state.
- `ActionSample` is a clean scalar acting result with action id, click
  target, target-head indices, log-prob, value, and next SNN state.
- `PPO.memory` is a `list[dict]` where each transition stores a
  `PolicyInputBatch` plus many small CPU tensors.
- `PPO.update_policy()` later stacks those small tensors, builds another
  `list[dict]` of TBPTT chunks, then packs chunk groups into dense
  `[T, group, ...]` tensors for replay.

Locally, this is fine. It is easy to inspect and hard to be confused by.

Distributed, it is the kind of structure that politely says:

```text
  "I hope you like serialization overhead."
```

So the speed plan is not "rewrite the model." It is:

```text
  keep the semantics
  pack earlier
  ship fewer objects
  measure the big copies
```

---

## 2. Payload Reality

Current model/effective config:

```yaml
model:
  spatial_input_shape: [27, 84, 84]
  vector_input_dim: 15
  attention_pool_size: 7
  attention_embed_dim: 64
  spatial_head_type: "coarse_to_fine"
  coarse_grid_size: 7
  local_grid_size: 12

derived_policy_protocol:
  action_feedback_tokens: [1, 9]
  total_tokens: 95

hyperparameters:
  rollout_steps: 2048
  tbptt_window: 128
  batch_size: 128
  epochs: 8
```

The spatial tensor dominates transport cost:

```text
  one spatial_obs step:
    27 * 84 * 84 float32
    190,512 floats
    762,048 bytes
    ~= 0.73 MiB

  one 512-step actor fragment:
    512 * 0.73 MiB
    ~= 372 MiB just for spatial_obs

  one 2048-step learner batch:
    ~= 1.45 GiB just for spatial_obs
```

The other inputs are smaller:

```text
  entity_features:
    24 * 21 float32 ~= 2.0 KiB / step

  selection_features:
    20 * 7 float32 ~= 0.55 KiB / step

  meta_vec:
    15 float32 ~= 60 bytes / step

  action_feedback_tokens:
    1 * 9 float32 ~= 36 bytes / step

  one full SNN state snapshot:
    syn + mem, 2 pathways, 95 tokens, 64 dims
    ~= 95 KiB float32
```

Conclusion: spatial transport and recurrent-state snapshots are the
first places to measure. Action ids and rewards are not the problem.

---

## 3. Golden Rule

Do not start with clever compression if the trainer is not correct yet.

The first distributed trainer should prefer:

- dense fragment tensors
- one Ray object per fragment
- exact same PPO math
- exact same stored action targets
- no asynchronous policy lag

Then measure. Then optimize.

---

## 4. Target Transport Shape

Current local shape:

```text
PPO.memory:
  list[
    {
      "observation_batch": PolicyInputBatch(...),
      "action": tiny tensor,
      "move_x": tiny tensor,
      "reward": tiny tensor,
      ...
    },
    ...
  ]
```

Distributed shape should be columnar:

```text
RolloutFragment:
  actor_id: int
  fragment_id: int
  policy_version: int
  policy_protocol_version: int
  policy_input_schema: str

  spatial_obs: Tensor[T, 27, 84, 84]
  entity_features: Tensor[T, 24, 21]
  entity_mask: Tensor[T, 24]
  selection_features: Tensor[T, 20, 7]
  selection_mask: Tensor[T, 20]
  action_feedback_tokens: Tensor[T, 1, 9]
  meta_vec: Tensor[T, 15]

  actions: Tensor[T]
  move_x: Tensor[T]
  move_y: Tensor[T]
  target_index: Tensor[T]
  coarse_index: Tensor[T]
  fine_index: Tensor[T]
  old_log_probs: Tensor[T]
  values: Tensor[T]
  rewards: Tensor[T]
  dones: Tensor[T]
  truncateds: Tensor[T]
  episode_reset_mask: Tensor[T]
  sample_mask: Tensor[T]

  recurrent_state: packed state plan
  tail_next_policy_input: PolicyInputBatch-shaped tensors
```

One fragment should be one payload, not 512 Python dictionaries.

---

## 5. Measurement Before Tuning

Add cheap timing and size counters before changing representation:

```text
actor side:
  env_step_ms
  observation_extract_ms
  policy_inference_ms
  fragment_pack_ms
  fragment_bytes
  steps_per_second_per_actor

learner side:
  ray_get_ms
  aggregate_ms
  gae_ms
  cpu_to_gpu_ms
  ppo_update_ms
  learner_idle_ms
  update_steps_per_second

object store:
  bytes_per_fragment
  refs_per_update
  spilled_bytes, if Ray exposes it
```

The first success metric is not raw max throughput. It is knowing where
the time went.

---

## 6. Phase A: Correct Dense Fragments

Build the first distributed transport as dense tensors:

- actor accumulates lists internally while stepping
- actor packs into one `RolloutFragment` before returning
- learner receives one object ref per fragment
- learner computes GAE per fragment
- learner flattens only after returns/advantages are valid
- learner feeds the existing PPO replay path

Acceptance criteria:

- 1 actor and 2 actor smoke tests pass
- fragment payload has no raw PySC2 objects
- all target-head indices survive transport
- `policy_protocol_version` and `policy_input_schema` are validated
- throughput counters exist

This phase can still be float32. Correct first.

---

## 6.1. Correctness Traps While Chasing Throughput

The dangerous optimizations are the ones that look like transport tweaks but
change the learning problem:

- Do not drop `action_feedback_tokens`; they are no longer embedded in
  `meta_vec`.
- Do not collapse `done`, `truncated`, and episode reset into one flag. Terminal
  `done` controls GAE bootstrapping; reset boundaries control recurrent and
  reward/extractor/action-feedback state.
- Do not ship only decoded `move_x/move_y` for spatial actions. The PPO update
  also needs the sampled target-head indices (`target_index`, `coarse_index`,
  `fine_index`) to recompute the correct old action log-prob path.
- Do not send actor-side CNN/spatial-token outputs in v1. That removes or
  changes learner-side gradients through the encoder unless the model is
  deliberately redesigned.
- Do not let actors continue collecting under stale weights while the learner
  performs multiple PPO epochs unless we intentionally move to an off-policy
  correction algorithm.

---

## 7. Phase B: Remove Object Confetti

Once correctness is stable, remove per-step Python object overhead:

- replace `list[dict]` transition transport with typed dataclasses
- keep metadata small and plain
- avoid one tensor object per scalar field per step
- pack masks, actions, rewards, and indices into single contiguous tensors
- avoid repeated `PolicyInputBatch.stack()` on hundreds of batch-size-1 objects

The desired learner-side path:

```text
RolloutFragment
    -> fragment-local GAE
    -> TBPTTChunkBatch
    -> packed replay tensor group
    -> PPO losses
```

The local single-process path can keep an adapter so debugging remains
pleasant.

---

## 8. Phase C: Spatial Payload Options

Spatial frames are the big cost. Options, in increasing risk:

### Option 1: float32 dense

- simplest
- exact current behavior
- biggest payload

Use for first correctness pass.

### Option 2: float16 normalized spatial tensors

- halves spatial payload
- easy to implement
- likely acceptable because actor inference already uses AMP on CUDA paths
- still needs loss/parity checks

Good first speed experiment.

### Option 3: uint8 feature-layer transport

- closest to original PySC2 feature-layer storage
- quarter of float32 payload
- learner converts to float and divides by `255.0`
- requires preserving or reconstructing pre-normalized feature layers cleanly

Potentially best medium-term transport format, but do not mix it into
the first Ray correctness branch.

### Option 4: actor-side spatial encoding

- actor sends post-CNN or post-token features instead of raw spatial input
- much smaller payload
- changes where gradients can flow
- not compatible with normal learner-side PPO replay unless carefully
  redesigned

Do not do this for v1.

---

## 9. Phase D: Recurrent-State Packing

Current PPO storage keeps pre-step recurrent state in each stored
`PolicyInputBatch`. With the current model:

```text
state shape per row:
  [2 pathways, 95 tokens, 64 dims]

state tensors:
  syn and mem
```

For transport, there are three levels:

### Level 1: store every pre-step state

- simplest
- mirrors current code
- expensive but correct

### Level 2: store TBPTT chunk initial states

- store only the state needed at each replay chunk boundary
- current `tbptt_window: 128` means far fewer state snapshots
- requires fragment finalization to know chunk boundaries

This is the best likely speed/size win after dense fragments work.

### Level 3: recompute states from fragment start

- smallest state payload
- more learner compute
- fragile around chunk shuffling and episode resets

Not first.

---

## 10. Phase E: Learner Throughput

The learner will do heavy replay:

```text
2048 transitions
8 PPO epochs
TBPTT chunks of 128
coarse_to_fine spatial target evaluation
```

Tune after measurement:

- keep `batch_size` as maximum active TBPTT steps per chunk group
- avoid CPU/GPU transfers inside inner loops
- preallocate pack buffers where practical
- keep masks boolean and contiguous
- move complete tensor blocks to GPU, not tiny pieces
- track `target_kl` early stopping effects on real update time

Do not let actors collect under stale weights while the learner updates
unless we intentionally move away from synchronous PPO.

---

## 11. Phase F: Logging Speed

Logging should not become the silent bottleneck:

- actor emits compact episode and reward summaries
- learner emits update summaries
- logger actor owns SQLite
- diagnostics wrappers stay opt-in
- eval traces can store float16 policy inputs like `Utility/eval_trace.py`

Never stream full per-step policy inputs into SQLite.

---

## 12. Suggested Config Additions

```yaml
distributed:
  enabled: false
  num_rollout_actors: 4
  fragment_steps: 512
  global_rollout_steps: 2048
  sc2_runtime_profile: "linux_headless"
  required_policy_protocol_version: 2
  required_policy_input_schema: "stream_action_feedback_v1"

throughput:
  fragment_float_dtype: "float32"      # later: "float16"
  spatial_transport: "normalized_float" # later: "uint8_feature_layer"
  recurrent_state_transport: "per_step" # later: "tbptt_initial"
  episode_boundary_transport: "done_truncated_reset_masks"
  emit_timing_metrics: true
  emit_fragment_size_metrics: true
```

These are not all implementation requirements for the first PR. They are
names for the knobs we will probably want once the first Ray smoke works.

---

## 13. Acceptance Criteria

The speed work is real when:

- one update reports bytes per fragment and learner idle time
- `4 x 512` collection reaches a full 2048-step update cleanly
- object refs per update are O(num actors), not O(num steps)
- dense fragment transport is faster than single-process wall-clock
- any float16 or uint8 transport mode has a parity check against float32
- checkpoint/resume remains learner-owned
- visual Windows inspection remains separate from headless Ray collection

---

## Final Position

Build correctness with dense fragments. Measure. Then attack the big
objects in order:

1. spatial payload dtype/format
2. recurrent-state snapshots
3. Python object count
4. CPU/GPU transfer placement
5. logging volume

That order is the least glamorous and probably the fastest path to an
actually faster trainer.
