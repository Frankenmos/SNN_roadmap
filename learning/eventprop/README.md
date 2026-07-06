# EventProp from scratch — a verified core, and notes toward actor-critic

This folder is **Stage A** of the EventProp learning ladder (see
[`../../docs/EVENTPROP_MIGRATION_PLAN.md`](../../docs/EVENTPROP_MIGRATION_PLAN.md)):
a small, from-scratch, **verified** EventProp implementation in pure numpy, with
nothing RL in it yet. The point is to own every gradient before wiring it into an
actor-critic. Everything here runs with numpy only (no torch, no GeNN).

```
eventprop_snn.py    forward LIF sim + exact-gradient backward pass
gradient_check.py   proves the backward matches finite differences (the deliverable)
yinyang.py          the standard tiny spike-timing dataset
train_yinyang.py    trains the net end-to-end; accuracy is the ultimate gradient check
```

Run, in order:

```
python gradient_check.py     # -> ALL CHECKS PASSED -- the adjoint is exact.
python train_yinyang.py      # -> best test accuracy ~73% (linear baseline ~64%)
```

Both are quick (seconds / ~a couple of minutes on CPU). `gradient_check.py` prints
`cos=1.000000  relL2=0.0000%` per layer; `train_yinyang.py` climbs from chance to
~73%, beating the linear baseline, which is only possible if the hidden spiking
layer is being trained by a correct gradient.

---

## 1. What EventProp is (one page)

A spiking net is a hybrid dynamical system: smooth leaky-integrator ODEs between
spikes, and instantaneous jumps *at* spikes whose *timing* depends on the
weights. Surrogate-gradient methods pretend the threshold is smooth.
**EventProp instead computes the exact gradient** by treating each spike as a
parameter-dependent event and running the adjoint (reverse) pass through it.

Our network (in `eventprop_snn.py`):

```
input spikes (latency coded) -> hidden LIF layer (spiking) -> output LI (non-spiking)
```

Each neuron is a leaky integrate-and-fire unit with an exponential-current
synapse (two state variables, current `I` and voltage `V`):

```
tau_syn * I' = -I        (I gets a +w kick at each presynaptic spike)
tau_mem * V' = -V + I
```

Hidden neurons spike when `V >= v_th` and reset `V -> 0`. Output neurons never
spike; we read out the **time-average of their voltage** and put softmax
cross-entropy on it. (That is ml_genn's `avg_var` readout — and, crucially, it is
the same shape you use for an actor-critic: readout voltages become policy logits
and a value estimate.)

**The one idea that makes the backward pass "EventProp" and not a surrogate:**
because a hidden spike resets `V` to a constant `0`, there is *no* gradient path
through the membrane value. The only way a hidden weight changes the loss is by
**moving the spike's crossing time**. We make the crossing time differentiable by
linear interpolation within the step,
`frac = (v_th - V_prev)/(V_new - V_prev)`, and its derivative carries a
`1/(V_new - V_prev) ~ 1/(dt * Vdot)` factor. That `1/Vdot` **is** the
implicit-function-theorem term from Wunderlich & Pehle 2021. Everything else is
ordinary reverse-mode through linear dynamics.

The weight gradient ends up being what the paper says it is: a sum of the
postsynaptic adjoint current sampled at presynaptic spike times.

### Is it actually exact? Yes — verified.

`gradient_check.py` compares the analytic gradient to finite differences across
several random seeds. Output-layer weights match to **machine precision**
(`cos=1.000000, relL2=0.0000%`). Hidden-layer weights match to machine precision
too **in smooth regions** — see the honest caveat in §3.

---

## 2. Toward actor-critic: the plan and the pitfalls

The extension plan (Stages B–D) is in the migration doc. The key structural idea:
**treat each environment step as one independent EventProp trial**, and let PPO's
advantages carry credit across steps. EventProp only does within-trial
spike-timing credit; GAE does across-step credit. They compose cleanly, so the
SNN can stay feedforward and you avoid all the cross-step recurrence pain.

The hook that makes RL possible: EventProp needs `dL/d(readout)`. For supervised
learning that comes from cross-entropy (`loss_and_output_error`). For actor-critic
you compute the **policy-gradient / PPO** loss w.r.t. the readout voltages
(logits and value) with ordinary autograd-style math, and pass *that* vector in as
`d_readout`. `backward()` propagates it through the spikes to the weights. Nothing
else in the core changes. That is why the readout adjoint being machine-exact is
the important result here.

### The problems EventProp brings to actor-critic — and the fixes

**(a) Zero gradient from silent neurons.** EventProp's gradient is a sum *over
spikes*. A neuron that never fires contributes nothing, and — because exact
gradients are blind to spike *creation* — nothing pulls it back to life. In RL
this is worse than in supervised learning: a policy that starts near-silent, or
an exploration phase that quiets neurons, can zero out whole regions of the
gradient.

*How ml_genn solves it (you asked specifically):* the loss-shaping paper
(Nowotny, Turner, Knight 2022) adds a **spike-count regularisation** that pins
each hidden neuron near a target firing rate:

```
L_reg = (1/2) * k_reg * sum_l ( mean_batch(spike_count_l) - nu_hidden )^2
```

Its gradient is injected as a small **jump on the adjoint** `lam_V` at each of the
neuron's spikes:

```
lam_V,l  +=  -(k_reg / N_batch) * ( mean_batch(spike_count_l) - nu_hidden )
```

In ml_genn's `EventPropCompiler` these are the parameters **`reg_nu_upper`**
(= `nu_hidden`, the target spike count per trial) and
**`reg_lambda_upper` / `reg_lambda_lower`** (= `k_reg`, the strength).

The honest catch: this jump happens *at spike times*, so it can only push a neuron
that **already spikes at least once** toward the target rate — it cannot
resurrect a truly silent neuron. So it is always paired with **weight
initialisation hot enough that neurons fire from the start** (the paper tabulates
init means/stds). Both matter. `train_yinyang.py` uses exactly this combination:
`reg_k`, `reg_nu`, and a positive-mean `W_hid` init — and prints the mean spike
count per neuron so you can watch it stay off zero.

**(b) Blindness to spike creation/deletion (critical points).** Where a neuron
*grazes* threshold, an infinitesimal weight change creates or destroys a spike and
the loss jumps discontinuously. The exact gradient "sees" a finite slope right up
to the cliff and nothing about the cliff itself, so gradient descent can walk off
it. Mitigation: **loss shaping** — designing the loss / regulariser to keep
neurons comfortably above or below threshold rather than grazing — which is what
(a) also buys you. You can literally see these critical points in our
`gradient_check.py`: they are the weights it reports as "skipped".

**(c) The `1/Vdot` divergence.** At a near-tangential crossing `Vdot^- -> 0`, so
the spike-time sensitivity `1/Vdot` blows up and one gradient entry explodes.
Standard fix: **gradient-norm clipping** (`train_yinyang.py` clips the global
grad norm to 1.0).

**(d) Non-stationary targets and the trial structure.** PPO's target changes
every epoch (advantages depend on the current value function), and episodes have
variable length while an EventProp trial is fixed-length. The per-step-trial
framing in the plan sidesteps the length mismatch; the changing target is fine
because we recompute `d_readout` from the current PPO loss each update. There is
**no published EventProp-RL yet**, so expect to iterate here — this is the
research part.

---

## 3. What building this taught me (the honest lab notes)

These are real lessons that fell out of getting the gradient check to pass. They
are the actual content of "understanding EventProp".

1. **A factor-of-`dt` scaling bug in the readout drive.** The readout is a *time
   average*, so `dL/dV_out(t)` is `d_readout / n_steps`, not `d_readout`. Getting
   this wrong left every gradient a constant multiple off — directions perfect
   (cosine `0.9988`), magnitudes `~200x` wrong. The finite-difference check is the
   only thing that catches a pure scale error; a network would still "train",
   just with a silently wrong effective learning rate.

2. **Discretised spiking loss is a staircase.** With hard thresholds on a time
   grid, moving a weight shifts a spike by whole steps, so the loss jumps in tiny
   O(dt) steps rather than varying smoothly. Naive central finite differences with
   a small `eps` then read either `~0` (inside a tread) or a huge spurious slope
   (straddling a step). Two fixes together made verification trustworthy:
   sub-step **interpolation of the crossing time** (smooths the loss to O(dt)),
   and a **consistency filter** in the check (trust finite differences only where
   two `eps` values agree; skip the rest).

3. **The skipped weights are not a bug — they are the physics.** The weights the
   check skips are exactly the near-tangential crossings of §2(b). Their loss is
   *genuinely* non-differentiable. This is why the most faithful EventProp
   implementations (jaxsnn) simulate in **continuous time with analytic spike
   times** instead of a grid — it removes the staircase entirely and matches
   finite differences to `1e-7`. Our discretised version is the readable teaching
   version; the event-based version is the next rung if you want it.

4. **`num_steps` / spike-count regime matters.** EventProp only has something to
   be "exact" about if there is real spike *timing* structure. A one-tick-per-step
   analog regime (like the main SNN_roadmap agent) gives it almost nothing to
   work with — which is half of why EventProp is a poor fit for that architecture
   and a fine fit for this latency-coded toy.

---

## 4. Next steps (the ladder)

- **Stage B — REINFORCE on CartPole.** Reuse `forward`/`backward` unchanged;
  replace `loss_and_output_error` with a policy-gradient `d_readout`
  (`-advantage * dlogpi/dlogits`), one env step = one trial. Simplest possible RL;
  if the pole stays up, EventProp + policy gradient works.
- **Stage C — add a critic** (a second readout voltage) → A2C.
- **Stage D — add PPO clipping + GAE** → the actor-critic you wanted.

Reference implementations worth reading: `lolemacs/pytorch-eventprop` (single
layer, the manual backward), `electronicvisions/jaxsnn` (event-based, most
faithful), `genn-team/ml_genn` (`EventPropCompiler`, production, has the reg
parameters above).
