# WHY_YOLO_BAD.md

## Thesis

The fastest way to lose months in RL is to change:

- the observation space
- the action space
- the policy architecture
- the recurrent/state story
- the logging schema
- and the training runtime

all at once, then call whatever happens "research".

That is not research. That is turning the codebase into a slot machine.

For this repo specifically, the dangerous part is not just "too many ideas". The dangerous part is that several of those ideas touch the **same boundary**: the thing the actor sees, the thing the learner replays, and the thing the logger knows how to describe.

So the real rule is:

**Do not scale or distribute an unstable data contract.**

---

## 1. Why YOLO is especially bad in RL

In normal software, a broken change often fails loudly.

In RL, a broken change often still trains.

It just trains:

- slower
- noisier
- into the wrong behavior
- with logs that do not tell you why

That is why YOLO is so seductive in RL. The system still moves. There is still a reward curve. There are still checkpoints. It *feels* alive. But you may have simultaneously changed:

- what the policy can represent
- what credit assignment means
- what the agent is actually conditioned on
- how PPO computes old-vs-new policy ratios
- how recurrent state is replayed
- and how the logs summarize all of that

Once that happens, you do not know which knob helped, which knob hurt, and which knob quietly invalidated the experiment.

---

## 2. Why YOLO is especially bad in *this* repo

This repo already has several places where policy design and training infrastructure are coupled:

- `obs_space/obs_space_2.py` still hardcodes a dense `spatial_obs + vector_obs` contract, with a fixed 100-d vector and a hardcoded `(27, 84, 84)` dimension story.
- `PPO_CNN/PPO.py` is still structurally tied to the current action heads and to one linear rollout with one bootstrap tail.
- `PPO_CNN_agent.py` is the place where "policy choice" becomes "game action", so action-space redesign directly affects rollout semantics.
- `Utility/logger_utils.py` still assumes a fixed schema for step/action logging.
- the current checkpoint and eval flow assume the learner, the actor, and the logger all agree on the same head layout and replay payload.

That means if you change obs/action/policy shape carelessly, you are not just changing the model. You are changing the **sample protocol**.

That is the real monster.

---

## 3. The ordering question: tokens first or Ray first?

Short answer:

**Neither. Protocol first. Then local token/action redesign. Then Ray.**

More honest answer:

If you already know the current observation/action contract is wrong for where you want to go, then **Ray should not come first**. If you distribute the wrong contract, you just make the wrong assumptions harder to remove.

But that does **not** mean:

"Spend weeks inventing the perfect transformer input before touching infra."

That is another trap.

The right order is:

1. define a stable model/training boundary
2. do the smallest local token/action redesign that exercises that boundary
3. get one or two honest single-process baselines
4. then distribute that boundary with Ray

So the answer is:

- **do not Ray the current hardcoded `(spatial, vector, action, move_x, move_y, final_next)` contract if you already plan to throw it away**
- **do not fully YOLO the ultimate tokenized architecture before you have a stable rollout/replay protocol either**

The bridge between the two is the protocol.

---

## 4. The protocol you actually want

Before you do the big obs/action redesign, create a policy-facing protocol that is generic enough to survive model changes.

Something like:

### `PolicyInputBatch`

- `obs_tokens`
- `obs_mask`
- `history_tokens`
- `history_mask`
- `state_in`
- `available_action_mask`
- `meta`

### `PolicyOutputBatch`

- `action_type_logits`
- `action_arg_logits`
- `value`
- `state_out`
- `aux`

### `ActionSample`

- `action_type`
- `action_args`
- `log_prob`
- `entropy_terms`

### `RolloutFragment`

- `policy_version`
- `actor_id`
- `inputs`
- `actions`
- `rewards`
- `values`
- `dones`
- `bootstrap_tail`
- `episode_summaries`
- `debug`

If you build **that** boundary first, then the distributed trainer does not need to care whether the model inside is:

- CNN + vector
- pooled spatial tokens
- entity tokens
- history tokens
- transformer only
- spiking transformer

It only needs to know how to move fragments around.

---

## 5. Can the distributed pipeline be made reusable while you do crazy model stuff?

Yes, but only up to a point.

The reusable part is:

- rollout actor lifecycle
- learner update loop
- checkpointing
- policy versioning
- logging transport
- eval orchestration

The non-reusable part is:

- exact model input format
- exact action head structure
- exact replay payload needed for PPO/recurrent replay
- exact debug metrics you want to log

So the honest answer is:

**Yes, the distributed pipeline can be made mostly reusable, but not magically immune to policy-interface changes.**

The goal is not "never touch infra again".

The goal is:

**touch only the protocol layer when model I/O changes, not the whole trainer.**

That is a huge difference.

---

## 6. What should be stable, and what should be free to change?

### Make these stable

- actor API: `set_weights()`, `collect_fragment()`
- learner API: `ingest_fragments()`, `update()`, `save_checkpoint()`
- eval API: `evaluate(weights_version)`
- transport object names and meanings
- checkpoint top-level structure
- logger event categories

### Allow these to change

- token encoder internals
- number and type of observation tokens
- history token construction
- action vocabulary
- transformer depth/width
- spiking memory layout
- aux losses

That is how you get a flexible research loop without rebuilding the world every time.

---

## 7. Past actions as tokens: good idea or pain?

It is a good idea.

It is also absolutely a pain.

But it is a *useful* pain.

Why it helps:

- it gives the model a notion of short-term intent
- it makes action sequences easier to model
- it helps when the environment is partially observable
- it reduces the burden on hidden state to remember "what I just tried"

In this repo specifically, it also directly attacks a real weakness: action execution and selection history are still only weakly observable through the current vector features.

So yes, action history as tokens makes sense.

---

## 8. Past **actions** versus past **policy outputs**

This distinction matters a lot.

### Safer v1: past sampled actions and arguments

Examples:

- previous `action_type`
- previous `(x, y)`
- previous `hotkey_id`
- maybe previous `reward`
- maybe previous `done`

This is straightforward to store and replay.

### Riskier v1: past raw policy outputs

Examples:

- previous logits
- previous probabilities
- previous hidden auxiliary predictions

This is much trickier, because now your future input depends on the **behavior policy's exact outputs** at rollout time.

That creates awkward questions:

- do we store the exact old outputs and feed them back during PPO update?
- do we recompute them recursively during replay?
- if we recompute them, with which weights?
- if they came from an older policy version, are we now conditioning the current policy on stale behavior-policy artifacts?

My recommendation:

**First version should use past action tokens, not past raw logits/probability vectors.**

Past actions are already very informative.
Past logits are research-y, but they are a nasty place to start.

---

## 9. Tokenized observation: good idea or pain?

Also both.

Why it is promising:

- the current dense `(27, 84, 84)` input is generic but not very intentional
- the policy is already halfway token-based internally after pooling
- entity- or region-style tokens fit a transformer much more naturally
- history tokens and action tokens fit elegantly once everything is tokenized

Why it is dangerous:

- variable token count means masks/padding become real infrastructure
- PPO replay now needs to preserve tokenization semantics
- recurrent state shape may stop being fixed if token count changes dynamically
- logging becomes harder because the input is no longer human-readable by default

So again, not a bad idea. Just a bad idea to mix with five other moving parts.

---

## 10. The hidden trap: variable token count

If you go tokenized observations, decide early whether the model sees:

- a fixed padded token grid with masks
- or a truly variable-length token list

For training systems and for Ray, **fixed padded tensors + masks** are much easier to live with.

Why:

- batching is easy
- object transport is easy
- checkpoint/replay shapes are predictable
- PPO update code stays vectorized

If you go fully ragged early, everything gets harder:

- learner collation
- replay
- logging
- testing
- distributed transport

So the mature choice is:

**fixed tensor shapes at the protocol boundary, even if the semantic token count is variable inside the mask**

---

## 11. What baselines should exist before the big redesign?

Because this is your first RL repo, here is the blunt advice:

You need boring baselines.

Not because boring is fun. Because boring is what lets you tell whether a cool idea helped.

Minimum useful ladder:

### Baseline A: current stabilized trainer

- current obs path
- current PPO
- current logging
- current distributed = off

This is your "the repo still works" anchor.

### Baseline B: action-space redesign only

- keep current obs
- keep current policy backbone
- change only the action semantics / heads

This tells you whether the action redesign is helping on its own.

### Baseline C: history features only

- keep current obs mostly intact
- add past action summary in the simplest possible form
- not full token history yet

This tells you whether history helps before the full token rewrite.

### Baseline D: tokenized obs + action history, still single-process

- new protocol
- new model input
- still local trainer

This tells you whether the modeling idea itself is promising.

### Baseline E: same as D, but Ray

- same protocol
- same model
- only runtime changes

This tells you whether distribution improved throughput without changing learning behavior too much.

That is how you stop lying to yourself.

---

## 12. What should be logged no matter what?

No matter how crazy the model gets, keep these stable:

- reward totals
- episode length
- action-type counts
- invalid / unavailable action rate
- clip fraction
- KL
- explained variance
- grad norm
- return stats
- eval mean/std
- policy version
- actor id

If you go tokenized:

- token count before padding
- token mask utilization
- history length actually used

If you go new action heads:

- head usage frequency
- invalid action substitutions
- argument entropy per head

If those stay stable, you can still compare runs when the model changes.

---

## 13. The real recommendation for *this* repo

If I were steering this project, I would do:

1. create a stable protocol layer
2. locally refactor PPO to consume structured fragments instead of the one-flat-buffer assumption
3. redesign action space and minimal history conditioning locally
4. only then introduce tokenized observations in a padded+mask format
5. get one real local baseline
6. then Ray the protocol

Notice what does **not** happen:

- not "Ray first on the old contract"
- not "ultimate token-transformer dream first with no baseline"

That is the middle path between paralysis and YOLO.

---

## 14. What will definitely require touching infra later?

Even with a good protocol, these future changes will still require some plumbing work:

- changing from fixed-size tokens to ragged tokens
- changing from simple PPO replay to sequence-unrolled autoregressive replay
- adding raw past-policy-output tokens instead of sampled-action tokens
- adding new action argument types that are not just categorical heads
- adding distributed replay buffers or off-policy correction

That is okay.

The point is not to avoid future work forever.
The point is to avoid **rewriting actor, learner, logger, checkpoint, and eval every single time.**

---

## 15. One sentence version

Scale **after** you stabilize the contract, and innovate **one layer at a time**.

That is how you stay ambitious without turning the repo into archaeology.
