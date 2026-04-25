# Next Fixes — Planning & Reasoning

Post-recovery-run plan for the two HIGH-impact structural suspects from
the session log. This document intentionally goes Socratic — problems
first, solutions second — so the *why* lands before the *how*.

---

## Fix 1 — Stateful/Stateless SNN Mismatch

### 1.1 What we know from the code

- Rollout:
  [PPO_CNN_agent.py:83-85](PPO_CNN_agent.py#L83-L85). The agent carries
  `self.snn_state` across env steps. At env step 500 of an episode, the
  SNN has integrated 500 steps of membrane + synaptic history.
- Training:
  [PPO.py:179-181](PPO_CNN/PPO.py#L179-L181). `self.policy_net(..., state=None)`
  — every minibatch forward re-initializes the SNN from zeros.

### 1.2 Socratic walkthrough

Before we decide *how* to fix this, let's be precise about *what* is
wrong. Three questions:

> **Q1.** At env step 500, the agent samples action `a` with
> `log_prob = old_logp`. What inputs did the network see to produce
> `old_logp`?

The observation at step 500, **plus** accumulated neuron state from
steps 1..499. The LIFs have non-zero membrane potentials, the attention
block's LIFs have non-zero Q/K/V membranes, the slow integrator `snn3`
has effectively integrated evidence for 500 steps.

> **Q2.** During the PPO update, we pull that transition into a random
> minibatch and recompute `new_logp` for the same obs with `state=None`.
> What inputs does the network see this time?

The observation at step 500, **plus** all-zero neuron state. Same
weights — but the *network's effective computation* is different,
because the LIFs start cold.

> **Q3.** What is PPO's importance ratio `r = exp(new_logp − old_logp)`
> measuring in this setup?

It's comparing the probability of `a` under
**policy π_stateful(obs at t=500 | state from t=1..499)** against the
probability under
**policy π_stateless(obs at t=500 | state=0)**.
These are *not the same policy*. The PPO clipping bound, the advantage
weighting, the KL — all are built on the assumption that the only
difference between "old" and "new" is weight updates. Here we're
comparing **two different stateful/stateless regimes** on top of any
weight difference.

> **Q4.** Why does this match the observed failure pattern — learn
> quickly early, drift after peak?

Early episodes are short, so at env step 20 the SNN's accumulated state
is close to zero anyway — stateful ≈ stateless. As the policy improves
and survives longer, state accumulates more meaning, so the mismatch
grows. The "old_logp vs new_logp" ratio becomes progressively more
biased, PPO's updates become progressively more wrong, and the policy
drifts.

> **Q5.** Can surrogate grads hide this? Can advantage normalization fix
> it? Can clipping it out via `clip_eps` compensate?

No. Surrogate grads are orthogonal (they fix the spike-vs-smooth issue
in forward-backward, not the state-consistency issue). Advantage
normalization is a zero-mean transform that can't detect systematic
per-sample ratio bias. Clipping masks the symptoms but the policy is
still being pulled in a biased direction on every sample.

### 1.3 Options on the table

| # | Option | Keeps temporal memory? | Complexity | Memory cost |
|---|---|---|---|---|
| i | Store `snn_state` with each transition, replay it during training | Yes (exact) | Medium | ~3 MB/transition |
| ii | Rollout with `state=None` every env step | No — kills multi-timescale integration across env steps | Trivial | Zero |
| iii | Truncated BPTT over K-step sub-trajectories | Yes within window | High (breaks random minibatching) | Moderate |
| iv | Frame-stack the last K obs; always state=None | Only within the stack | Low | Low (K× input channels) |

### 1.4 Recommendation: Option (i) with CPU-side storage

The whole point of this architecture is multi-timescale temporal
integration — fast/medium/slow time constants. Options (ii) and (iv)
abandon the thesis. Option (iii) breaks PPO's random minibatching,
which interacts badly with everything we just fixed. Option (i) is the
correct-by-construction fix.

**Memory budget** (per transition, fp32):

- `(syn1, mem1)`: `[16, 84, 84]` × 2 ≈ 0.9 MB
- `(syn2, mem2)`: `[32, 84, 84]` × 2 ≈ 1.8 MB
- `mem3`: `[64, 42, 42]` ≈ 0.45 MB
- `(mem_q, mem_k, mem_v)`: `[49, 64]` × 3 ≈ 37 KB
- **Total**: ~3.2 MB per transition → ~9.6 GB for 3000 transitions
  in fp32.

Mitigations if that's too much:

- Store on CPU; move to GPU per-minibatch (same pattern as
  `spatial_obs` / `vector_obs` today).
- Store in `float16` (half the cost) since state precision is not
  critical for gradient correctness.
- Only store every N-th step's state and recompute intermediate states
  from there (a cheap form of gradient checkpointing for the replay).

### 1.5 Concrete implementation plan

Files to touch:

- [PPO_CNN_agent.py](PPO_CNN_agent.py): return the **pre-step**
  `snn_state` from `step()` alongside everything else.
- [PPO_CNN_run.py](PPO_CNN_run.py): pass it into `store_transition`.
- [PPO_CNN/PPO.py](PPO_CNN/PPO.py):
  - `store_transition`: accept and store the nested state tuple,
    detached + on CPU in fp16.
  - `update_policy`:
    - Stack the per-transition states into batched tensors. The state
      is a tuple of tuples — stack element-wise.
    - In each minibatch, index into the stacked state, move to GPU,
      cast to `amp_dtype`, pass into `policy_net(..., state=batch_state)`.
  - The bootstrap forward for `last_next_value` should also use the
    **last transition's post-step state**, not `state=None`.
- [PPO_CNN/policy_network.py](PPO_CNN/policy_network.py): no change —
  `forward(..., state=...)` already accepts the exact tuple shape.

Subtle points:

- **Pre-step vs post-step state**: the state stored must be the state
  *entering* the forward pass that produced `old_logp`. Not the one
  coming out. That's the input, not the output, of the policy call.
- **Batch dim**: the rollout is `B=1`. When we stack `T` transitions,
  each state tensor gains a leading batch dim of `T`. During minibatch
  training we index the batch dim with `idx`.
- **`num_steps=2` internal loop**: the policy's inner time-loop
  overwrites local neuron state `num_steps` times per forward, but the
  state we pass in is the **initial condition** for that loop.
  Consistent with rollout behavior, so correctness is preserved.
- **Memory budget probe**: add a one-line print of
  `sizeof(stacked state) / 1e9` at the start of `update_policy` so we
  can see what we're allocating. Kill-switch in case it blows up.

### 1.6 Verification signals

With the new `ppo_updates` instrumentation in place:

- `mean_kl` should drop substantially — the policies are now
  comparable apples-to-apples, so most of the ratio noise goes away.
- `clip_fraction` should drop — fewer samples hit the clipping bound
  because ratios are no longer biased.
- `explained_variance` should improve — the critic is no longer
  fitting a moving target induced by the state mismatch.
- Empirically: the late-episode drift should weaken or disappear.

Independent sanity check: in a short debug run, assert that when
training runs with the recorded state, the recomputed `new_logp`
at epoch 0 is numerically close to `old_logp` (they won't be equal
because of fp16 + autocast, but the delta should be small).

---

## Fix 2 — Move-Head Entropy Asymmetry

### 2.1 What we know from the code

- [PPO.py:`_calculate_losses`](PPO_CNN/PPO.py): entropy bonus is

    ```python
    entropy = action_dist.entropy() + is_move * (move_x.entropy() + move_y.entropy())
    entropy_loss = self.entropy_coef * entropy.mean()
    ```

  where `is_move = (action == MOVE_ACTION_ID).float()`.

### 2.2 Socratic walkthrough

> **Q1.** For a rollout step where the sampled high-level action was
> `attack`, what is the entropy-bonus contribution?

Only `H(action) ≤ log(3) ≈ 1.1`. The move heads exist but contribute
zero thanks to the `is_move` mask.

> **Q2.** For a rollout step where the sampled high-level action was
> `move`, what is the entropy-bonus contribution?

`H(action) + H(move_x) + H(move_y) ≤ log(3) + 2·log(84) ≈ 9.9`.

> **Q3.** Multiply both by `entropy_coef = 0.01`. What does the gradient
> signal effectively say?

Per-sample bonus: attack ≈ 0.011, move ≈ 0.099. **The policy is being
paid ~10× more, per gradient step, to commit to moving than to
attacking** — not because moving is better for the environment reward,
but purely as an entropy-bonus accounting artifact.

> **Q4.** Is this the correct regularizer for "explore the joint
> action space"?

No. The *goal* of the entropy bonus is to prevent premature collapse of
whatever distribution the policy is currently using. The asymmetry
arises because we're summing entropies of two different dimensionalities
with the *same coefficient*. The "extra exploration pressure" on move
coords isn't wrong per se — moving *is* in a higher-dimensional action
space — but coupling it to the high-level action choice means the
policy learns to *pick the move action* to unlock extra bonus, not to
*explore better within the chosen mode*.

> **Q5.** Does this match the observed collapse pattern?

Yes. The action-mix plot shows kiting emerge around episode 800-1000
(attack-heavy) then drift toward move+no-op by episode 3500. The
asymmetric entropy bonus is a continuous gradient toward "pick move
more often", which is exactly what the data shows.

### 2.3 Options on the table

| # | Option | What it fixes | Downsides |
|---|---|---|---|
| a | **Normalize each head's entropy by its `log(n)`** so all heads contribute to `[0, 1]` | Attack and move samples contribute comparable entropy bonuses | Has to be done consistently in both training and rollout (though rollout doesn't compute entropy; only training needs this) |
| b | Split `entropy_coef` into per-head coefficients (e.g. `action_entropy_coef`, `move_entropy_coef`) | Maximum flexibility | Two more hyperparameters; hard to tune blindly |
| c | Remove `is_move` mask, always include move-head entropy | Simplifies code | Makes the bias **worse** — move-head entropy is always added regardless of sampled action |

### 2.4 Recommendation: Option (a) — normalized per-head entropy

The entropy of a uniform categorical over `n` outcomes is `log(n)`. So:

```python
H_action_norm = action_dist.entropy() / math.log(action_dim)
H_x_norm      = move_x_dist.entropy() / math.log(screen_size)
H_y_norm      = move_y_dist.entropy() / math.log(screen_size)
entropy = H_action_norm + is_move * (H_x_norm + H_y_norm)
```

Each normalized term lives in `[0, 1]`, so:

- Attack sample contribution: up to `1 · entropy_coef`
- Move sample contribution: up to `3 · entropy_coef`

Still not *perfectly* symmetric (move samples now get 3× not 10×), but
the three move-related terms now correspond to three *distinct*
exploration axes (which action, which x, which y), so a 3× total is
defensible. We can also tweak to `H_action_norm + is_move · 0.5 · (H_x_norm + H_y_norm)`
if we want to explicitly down-weight the move coords.

### 2.5 Concrete implementation plan

Files to touch:

- [PPO.py:`_calculate_losses`](PPO_CNN/PPO.py): divide each head's
  entropy by its `log(n)` and combine. Same change in the diagnostics
  `entropy_mean` calculation so the logged entropy reflects the
  normalized value.
- Optionally, add `move_entropy_scale` to `config.yaml` so we can tune
  the relative weight of move-coord entropy without touching code.

### 2.6 Verification signals

- `mean_entropy` in `ppo_updates` should become more comparable across
  rollouts regardless of action mix.
- Empirical action entropy from the diagnostic should stabilize higher
  for longer (policy won't collapse toward "move" for bonus-hunting
  reasons).
- Action-mix plot should show a more balanced attack/move distribution
  after convergence instead of drifting to all-move.

---

## Ordering

1. **Run the D+E training first.** We already queued that. If it alone
   fixes the drift, Fix 1 may be less urgent. (It won't be moot —
   stateful/stateless mismatch is still structurally wrong — but the
   priority order will be clearer.)
2. **Fix 2 before Fix 1.** (IMPLEMENTED) The entropy asymmetry was a 10-line change
   with clean verification, where we normalized by dividing by math.log(n).
   Doing Fix 2 first also means when we evaluate Fix 1, the entropy signal in the
   logs is already trustworthy.
3. **Fix 1** last among the structural pair.
4. Then resume the deferred backlog (reward redesign, test + env-setup
   fixes), and finally the EventProp Socratic.

## Anchor for Fix-1 success

When `run_with_state_replay` finishes, compare its late-stage
`mean_kl` and `clip_fraction` (from `ppo_updates`) against the
recovery-knob run. If state replay is doing what we expect:

- `mean_kl` should be meaningfully lower (e.g. < 0.5× the recovery
  run's value)
- `clip_fraction` should be meaningfully lower
- `explained_variance` should be higher

If all three are unchanged, the state mismatch was not the dominant
issue and we look elsewhere (reward shaping, `num_steps`, etc.).
