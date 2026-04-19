# The BPTT Question

What follows is a code-grounded note on what Backpropagation Through
Time would mean for this project, what PPO is doing today, and which
changes are actually required.

This document treats the uploaded tutorial as the conceptual anchor and
the current codebase as the implementation anchor.

---

## 1. What BPTT means in plain English

The tutorial's core point is simple:

- a recurrent state at step `t+1` depends on the recurrent state at step
  `t`
- therefore the loss at step `t+1` can create gradients on parameters
  used at step `t`
- that only happens if the computation graph keeps the temporal link
  alive across the unrolled sequence

In our project, the recurrent object is not a vanilla `h_t`, but the
token-temporal SNN state:

- `state_in = (syn, mem)`
- each tensor is `[B, N, D]`
- `PolicyNetwork.forward(...)` consumes `state_in` and returns
  `next_state`

So the same principle applies:

- if `next_state_t` is fed as `state_in_(t+1)` **inside the training
  graph**, we get BPTT
- if `state_in_(t+1)` is replayed as detached cached data, we do **not**
  get BPTT through env steps

---

## 2. What PPO is doing today

### Collection path

Current rollout collection:

1. `agent.step(obs)` runs the policy once and updates `self.snn_state`
2. the returned `policy_input` already carries the **pre-step** state
3. `PPO.store_transition(...)` saves that batch after
   `observation_batch.detach().to(device="cpu")`

Relevant code:

- `PPO_CNN_agent.py`
- `PPO_CNN/PPO.py`

That means the rollout stores:

- observation at step `t`
- detached pre-step recurrent state for step `t`
- action / log-prob / value / reward / done

### Update path

Current PPO update:

1. stack detached step-batches into one flat rollout batch
2. compute GAE over the flat time axis
3. random-shuffle timestep indices
4. run `policy_net(batch_observation)` independently on each sampled
   minibatch
5. ignore the returned `next_state`

That is the critical detail:

- the update path uses stored `state_in` as an input feature
- it does **not** carry `next_state_t` into step `t+1` during training
- it does **not** preserve temporal order inside minibatches

So the current trainer is best described as:

- **state replay PPO**
- not **BPTT PPO**

---

## 3. Why this is not BPTT

The BPTT tutorial says the backward pass must walk from later timestep
losses back into earlier recurrent computations.

Today that temporal link is severed in three different ways:

### 3.1 Stored recurrent state is detached

`PolicyInputBatch.detach()` detaches `state_in`, and rollout storage uses
it.

So during PPO update, `state_in_t` is a fixed tensor, not a graph edge
coming from the previous forward step.

Consequence:

- the loss at `t+1` cannot backprop through the state transition that
  produced `state_in_(t+1)`

### 3.2 Timesteps are shuffled IID

`update_policy()` does `perm = torch.randperm(rollout_size)` and trains
on random timestep minibatches.

Consequence:

- temporal adjacency is destroyed during optimization
- even if the stored states were not detached, the update code still
  would not be replaying the recurrent chain in order

### 3.3 Returned `next_state` is ignored during training

`policy_net(batch_observation)` returns `next_state`, but PPO discards
it in the update pass.

Consequence:

- the recurrent transition is used during acting
- but not used as a differentiable transition during training replay

---

## 4. One subtle but important extra problem: helper steps

This is the part that becomes impossible to ignore once we ask for true
BPTT.

Today we only store transitions when `learnable == True`.

But `agent.step(obs)` always advances `self.snn_state`, even on
non-learnable helper steps such as fallback `select_army`.

That means:

- recurrent state evolves on **every** env step
- rollout memory currently keeps only **some** of those steps

So if we tried to do BPTT only across stored learnable steps, we would
not reconstruct the same recurrent trajectory that happened during data
collection whenever helper steps existed between them.

This also means current PPO is dropping:

- helper-step state transitions
- helper-step rewards
- helper-step dones

That was already a rollout abstraction compromise before BPTT; BPTT just
makes the mismatch much more obvious.

---

## 5. What we should aim for: TBPTT, not full-rollout BPTT

For this project, the right target is **Truncated BPTT (TBPTT)**.

Why not full BPTT over the whole rollout?

- rollouts are long
- PPO reuses data for multiple epochs
- full unroll graphs would be expensive and unstable
- we only need bounded temporal credit assignment, not an unbroken graph
  across an entire update

So the practical target should be:

- collect ordered rollout sequences
- split them into windows of length `K`
- detach state only at window boundaries
- do BPTT **within** each window

Recommended new hyperparameter:

- `tbptt_window` or `bptt_steps`

This is separate from the current policy `num_steps`, which is only the
internal spike-accumulation loop inside one env step.

---

## 6. Needed changes

### 1. Stop treating rollout samples as IID timesteps

Current PPO update shuffles flat timesteps.

Needed change:

- build minibatches from contiguous time windows, not random individual
  steps
- shuffle windows/chunks if desired, but preserve order inside each
  chunk

Why:

- BPTT requires ordered unroll

---

### 2. Store every env step that mutates recurrent state

Current memory stores only learnable steps.

Needed change:

- store **all** env steps that pass through the policy and mutate
  `self.snn_state`
- add explicit masks:
  - `policy_loss_mask`
  - optionally `entropy_loss_mask`
  - maybe `value_loss_mask` if we decide to mask critic too

Recommended default:

- policy loss masked off on helper steps
- entropy masked off on helper steps
- value loss kept on, because value is state-based and helper steps are
  still real environment states with real returns

Why:

- recurrent replay must match the acted trajectory
- reward/done flow should remain faithful across helper steps

---

### 3. Replay recurrent state as a sequence, not as detached per-step data

Current memory stores detached `state_in` on every step and PPO reuses
that detached state directly for each replayed step.

Needed change:

- keep enough information to seed a chunk with its initial recurrent
  state
- then unroll the policy forward across the chunk using the returned
  `next_state`

Two workable options:

Option A, minimal:

- keep storing `state_in` per step
- when building a chunk, use only the first step's `state_in` as the
  chunk initial state
- ignore stored states for later steps in that chunk

Option B, cleaner:

- store observations stateless
- store chunk-start recurrent states separately

Recommendation:

- start with **Option A** to minimize code churn

---

### 4. Unroll the policy inside PPO update

Needed change in PPO update:

Pseudo:

```python
state = chunk_init_state.detach()  # boundary detach only
for t in range(chunk_len):
    step_batch = obs_t.with_state(state)
    action_logits, move_x_logits, move_y_logits, values_t, state = policy(step_batch)
    if done_t:
        state = policy.init_concrete_state(batch_size=...)
```

Then stack outputs over time and compute PPO losses on the stacked
results.

Why:

- this is the actual BPTT/TBPTT link the current code is missing

---

### 5. Reset recurrent state on episode boundaries during replay

During acting, episode reset zeroes the SNN state.

Needed change during training replay:

- if `done[t] == 1`, zero recurrent state before the next replayed step
- do this inside the sequence unroll, not only at collection time

Why:

- otherwise replay would leak state across episodes even if acting did
  not

---

### 6. Keep GAE/returns on the true rollout order

Advantage computation already runs over rollout time order.

Needed change:

- if we start storing helper steps too, GAE must run over the full
  ordered trajectory including those steps
- PPO actor loss can still be masked on helper steps

Why:

- otherwise returns no longer match the actual sequence that generated
  the recurrent state

---

### 7. Add a sequence-aware rollout structure

Current memory is a flat Python list of step dicts.

Needed change:

- either add a chunk builder on top of the existing list
- or introduce a dedicated sequence batch container

The container needs at least:

- ordered observations
- initial recurrent state per chunk
- actions / move heads / old log-probs
- rewards / values / dones
- loss masks

Recommendation:

- do not change `PolicyInputBatch` into a time-major structure
- add a separate sequence/chunk abstraction in PPO code

Why:

- `PolicyInputBatch` is currently a clean single-step protocol
- overloading it with time semantics would blur responsibilities

---

### 8. Change minibatch sampling policy

Current policy:

- random timestep sampling

Needed policy:

- random **chunk** sampling
- ordered replay inside chunk

Possible shapes:

- `[T, B, ...]` time-major chunks
- or simple Python loop over time for each chunk if we keep the first
  implementation small

---

### 9. Decide what stays detached

For TBPTT we still want truncation boundaries.

So:

- detach at chunk start
- do **not** detach between steps inside the chunk
- detach again when moving to the next chunk

This is the recurrent equivalent of the tutorial's backward pass ending
at a chosen truncation horizon.

---

### 10. Add tests that prove we really have BPTT

This part matters a lot because "we replay state" and "we do BPTT" are
easy to confuse.

Needed tests:

1. **Temporal-gradient test**
   A loss on step `t+1` should create gradients on parameters used to
   produce the recurrent transition at step `t`.

2. **Sequence-order test**
   Chunked ordered replay should differ from shuffled timestep replay in
   the intended way.

3. **Done-reset test**
   Replay must zero recurrent state after terminal steps.

4. **Helper-step fidelity test**
   Inserting a helper step between learnable steps should preserve the
   acted recurrent trajectory during replay.

5. **Window truncation test**
   Gradients should stop at TBPTT boundaries by design.

---

## 7. File-by-file impact

### `PPO_CNN/PPO.py`

This is the main rewrite.

Needed work:

- rollout memory schema grows masks / maybe sequence metadata
- add chunk builder
- replace flat shuffled minibatching with chunked replay
- unroll policy through time during update
- reset state on done inside replay
- mask actor/entropy losses where needed

### `PPO_CNN_run.py`

Needed work:

- stop dropping helper steps from rollout storage
- store loss masks that distinguish learnable vs helper steps
- ensure `final_next` / bootstrap logic matches the new all-step
  rollout semantics

### `PPO_CNN_agent.py`

Probably small changes only.

Needed work:

- return enough information for storage masks
- possibly rename or clarify `learnable` into something more explicit
  like `policy_trainable_step`

### `PPO_CNN/policy_network.py`

Probably no architecture rewrite needed, but replay helpers would help.

Potential additions:

- small helper to reset state with a done mask
- optional `forward_sequence(...)` helper, though a PPO-side loop is
  enough for the first implementation

### `PPO_CNN/policy_input.py`

Likely no fundamental protocol change needed.

Possible addition:

- helper utilities for sequence chunk assembly if that becomes awkward in
  PPO

### Tests

Likely new dedicated file(s), because this is not just a small extension
of the current unit tests.

---

## 8. What we do **not** need

To avoid overbuilding:

- we do **not** need to manually implement gradients like the tutorial
  does; PyTorch autograd is enough once sequence replay preserves the
  graph
- we do **not** need to checkpoint live recurrent state
- we do **not** need to change the observation-tokenization design for
  BPTT
- we do **not** need to change the action space

The missing piece is training-time sequence replay, not a new model
family.

---

## 9. Bottom line

Current code has:

- recurrent state during acting
- recurrent-state replay during PPO update

Current code does **not** have:

- BPTT through env steps

To get there, the minimum conceptual shift is:

- from **flat detached timestep replay**
- to **ordered chunk replay with live state carry inside the training
  graph**

And because helper steps currently mutate recurrent state while being
dropped from memory, the rollout format itself must become more faithful
before BPTT can be correct.
