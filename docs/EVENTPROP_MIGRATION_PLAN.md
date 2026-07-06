# EventProp vs. TBPTT: feasibility & migration plan

Author: planning session, 2026-07-06
Companion to: [`E-prop.md`](E-prop.md) (the deep-research derivation)
Status: **plan only** — no code changed. This is a decision document.

---

## 0. TL;DR (read this first)

**You asked: "can we ditch TBPTT for EventProp?"** After reading the model
(`agent_core/spiking_policy.py`), the trainer (`agent_core/ppo_trainer.py`),
and the ml_genn / GeNN docs, the honest answer is:

- **As a drop-in via ml_genn's `EventPropCompiler`: no.** Three independent
  blockers, each sufficient on its own (pure-LIF requirement, supervised-only
  loss API, dataset-only training loop). See §4.
- **As a hand-written PyTorch EventProp adjoint on the spiking core, hybrid
  with autograd for everything else: possible, but it is a research build,
  not a swap** — and it would replace *how the backward is computed on the
  temporal SNN pathways*, not "TBPTT" as a whole. See §5, Route B.
- **The framing needs one correction up front:** EventProp is **not the
  opposite of BPTT**. ml_genn's own docs call it *"a form of event-based
  Backpropagation Through Time using exact gradients."* In this repo it would
  sit on the *same* temporal axis your TBPTT already handles. So "ditch TBPTT
  for EventProp" really means "replace surrogate-gradient backward with an
  exact adjoint backward" — narrower in concept, far more invasive in code.
- **There is a name trap that may change your whole target:** the file is
  called `E-prop.md` but its content is **EventProp** (Wunderlich & Pehle
  2021, exact adjoint). **e-prop** (Bellec et al. 2020, eligibility
  propagation) is a *different* algorithm — approximate, online, forward-in-
  time, and **already demonstrated on reinforcement learning**. ml_genn ships
  **both** (`EventPropCompiler` *and* `EPropCompiler`). For an RL agent that
  carries state across env steps, **e-prop is arguably the better fit than
  EventProp.** See §6.

**Recommendation:** do **not** rip out TBPTT. Do a **scoped, isolated
experiment** first (§7, Route D) that validates an exact-gradient / online
spiking learner on a supervised proxy *before* it touches the PPO trunk. That
de-risks the hard parts and answers the real question — *does an exact-gradient
SNN core even help this hybrid?* — for a few days of work instead of a rewrite.

---

## 1. Terminology: EventProp ≠ e-prop (and ml_genn has both)

| | **EventProp** | **e-prop** |
|---|---|---|
| Paper | Wunderlich & Pehle 2021 (Sci. Rep.) | Bellec et al. 2020 (Nature Comms) |
| Gradient | **Exact** (adjoint / Pontryagin) | **Approximate** (drops long-range terms) |
| Direction | **Backward** adjoint over the trial | **Forward**, online eligibility traces |
| Memory | O(#spikes) | O(1) per synapse (running trace) |
| Native task | **Supervised** (spike-time / voltage loss) | Supervised **and reward-based / RL** |
| ml_genn class | `EventPropCompiler` | `EPropCompiler` |
| Your doc | describes **this** | (the filename says this) |

The two are easy to conflate because your doc is filed under `E-prop.md`.
Keep them separate: **your derivation is EventProp**; the RL-friendly cousin
that shares the filename is **e-prop**. This distinction drives §6.

---

## 2. What "TBPTT" actually is in *this* repo

This is the crux, and it is subtle. The codebase has **two temporal axes**,
and they are not the same thing:

### Axis A — inside one env step (the "spike loop")
`config.yaml: model.num_steps = 1`. The inner loop
`for _ in range(self.num_steps)` (`spiking_policy.py` ~L952) runs **once**.
So each env step is **exactly one SNN tick**. There is essentially **no
intra-trial spike-timing structure** today.

### Axis B — across env steps (what your TBPTT trains)
The recurrent object is the second-order `snn.Synaptic` state
`(syn, mem)` = `(I, V)`, shape `[B, 2 pathways, num_tokens, 64]`, carried
between env steps via `agent.snn_state` and reset per episode. Training
replays each fragment step-by-step, carrying `(syn, mem)` **without detach**
inside a window of `tbptt_window = 128` env steps, detaching only at window
boundaries (`ppo_trainer.py` `_pack_chunk_group` ~L1766, `_replay_packed_chunk_group`
~L1840), then backprops the PPO scalar with `fast_sigmoid` surrogate gradients
(`spiking_policy.py:490`, backward at `ppo_trainer.py:1298`).

**Why this matters for EventProp:** EventProp's entire value proposition —
event-based backward, memory scaling with spike count, *exact spike-timing*
gradients — is about **when neurons spike within a trial of many fine
timesteps** (Axis A). But in this repo the temporal signal lives on **Axis B**,
and Axis A is a single tick. Inputs are injected as **analog current** (conv
features → LIF input), with **no Poisson/latency spike encoding**. So today's
network is closer to an **analog recurrent net made of leaky units** than to a
temporally spike-coded SNN.

To make EventProp *meaningful* here you would have to reconceive a 128-env-step
fragment as **one continuous LIF trajectory** (observations injected as input
currents at each step boundary, episode `done` = trial boundary), and derive
the adjoint backward over that whole trajectory. That is coherent — and in that
framing EventProp genuinely *is* an exact-gradient replacement for the
surrogate-gradient TBPTT. But it changes the modelling regime, not just the
optimizer. Hold that thought for §5.

---

## 3. What the network actually is (the mismatch)

EventProp's adjoint is derived for a **pure network of LIF neurons with
exponential synapses whose only trainable parameters are synaptic weights**.
This network is a **hybrid**, most of which is *not* that:

| Component | File / loc | LIF? | EventProp adjoint exists? |
|---|---|---|---|
| Conv backbone (conv1/2/3 + ReLU) | `spiking_policy.py` ~L493–498, 851–945 | No | No (standard autograd) |
| Entity/Selection/Meta encoders (ReLU MLP) | `spiking_policy.py` | No | No |
| Learned 2D positional tokens, type embeddings | `spiking_policy.py` | No | No |
| **Spiking self-attention** (`scaled_dot_product_attention` over Q/K/V spikes) | `spiking_policy.py` ~L263–292 | `snn.Leaky` (stateless/step) + **softmax attention** | **No** — attention has no EventProp rule |
| **Fast/slow token-temporal SNN** (`snn.Synaptic`, learnable α,β) | `spiking_policy.py` ~L316–345, 550–559 | **Yes** (2nd-order LIF) | Yes, *with extra terms* for learnable τ |
| Shared FC trunk, `actor_fc`, `critic_fc` | `spiking_policy.py` ~L567–568, 983–985 | No | No |
| Coarse-to-fine spatial head (49-way + 144-way categorical) | `target_heads.py` `CoarseToFineTargetHead` | No | No (structured readout, not voltage/spike-time) |

**The exact adjoint can, at most, cover the `snn.Synaptic` pathways** (and
maybe the `snn.Leaky` attention neurons). Everything else — which is the
**majority of the parameters** and most of what learns the DefeatRoaches policy
— must stay on ordinary autograd. Any real integration is therefore a
**hybrid**: autograd for conv/attention/heads, EventProp adjoint for the LIF
core, chained at the boundary. The single hardest theoretical gap is the
**softmax attention applied to spike trains** — EventProp says nothing about
backpropagating exact gradients through it.

---

## 4. Route A — ml_genn `EventPropCompiler` wholesale: **not viable**

Three blockers, each fatal on its own:

1. **Pure-LIF requirement.** `EventPropCompiler` supports only
   `LeakyIntegrateFire` hidden neurons + `LeakyIntegrate` readouts. It has no
   conv backbone, no softmax attention, no coarse-to-fine categorical head, no
   token embeddings. Adopting it = **rebuild the model as a vanilla recurrent
   LIF net** — i.e. throw away the V6/V7 architecture, not migrate it.

2. **Supervised-only loss.** The compiler accepts only
   `SparseCategoricalCrossentropy` (classification) and `MeanSquareError`
   (regression). There *is* a `Loss` base class (ABC: `add_to_neuron`,
   `set_target`) so a custom loss is technically possible, and you could smuggle
   a PPO readout-gradient in as an MSE pseudo-target (set `target = readout −
   upstream_grad` so the injected error equals the gradient). But that is a
   trick layered on a framework built for fixed labels.

3. **Dataset-only training loop.** The entry point is
   `compiled_net.train({input: spikes}, {output: labels}, num_epochs)`. There
   is no first-class online / RL loop, no exposed per-iteration manual
   backward. PPO needs rollout → GAE → clipped surrogate with advantages that
   change every epoch → SIL. You'd be driving GeNN's compiled GPU model through
   a bespoke loop it wasn't designed for, **marshalling data between PySC2 /
   PyTorch (host) and GeNN (device) every step**, and losing PyTorch autograd
   for all the non-LIF heads (which EventProp can't train anyway).

Plus the toolchain tax (least fundamental, but real): **GeNN on Windows + a
5090**. Installable now via conda-forge `pygenn-cuda` (needs Visual Studio
2019+ and a matching CUDA toolkit), but the 5090 is Blackwell (sm_120) → needs
CUDA 12.8+, and whether the prebuilt `pygenn-cuda` targets sm_120 on Python
3.12 today **needs verification** before you'd trust it.

And the empty-set signal: a web sweep for **EventProp + reinforcement learning**
turns up **nothing**. Spiking RL that exists (PopSAN and friends) uses
**surrogate gradients**, not EventProp. You would be in genuinely unexplored
territory — fine for research, bad for a "just swap the optimizer" expectation.

**Verdict:** Route A is a different project (greenfield LIF agent), not a
migration of this one. Not recommended as a path off TBPTT.

---

## 5. Route B — hand-written PyTorch EventProp adjoint (hybrid): possible, but a research build

Keep PySC2 / PPO / GAE / SIL / the conv+attention+heads exactly as they are on
autograd. Replace **only** the `snn.Synaptic` pathways' backward with an
`autograd.Function` that:

- **forward:** simulates the 2nd-order LIF over the fragment, records state at
  spike times (and enough to reconstruct V̇⁻ at threshold crossings);
- **backward:** given `grad_output` from the heads (delivered by ordinary
  autograd), integrates the adjoint ODE backward with the jump conditions from
  `E-prop.md` §"Jump conditions", returns grad w.r.t. input current **and**
  w.r.t. weights (and α, β).

This is exactly how jaxsnn / lolemacs-style implementations embed EventProp
inside a larger differentiable graph (`custom_vjp` / `autograd.Function`). It is
implementable. But be clear-eyed about scope and payoff:

- **You are re-deriving and coding the adjoint for the *second-order,
  learnable-τ* neuron** (`snn.Synaptic` with `learn_alpha`, `learn_beta`). The
  doc's math is the standard LIF+exp-synapse; the learnable time constants add
  gradient terms not written there.
- **Exact gradients only reach a minority of the network.** Conv, attention,
  and heads — most parameters — still get ordinary gradients. The headline
  EventProp wins (3× faster, 4× less memory) are quoted for *pure-SNN
  classification*; they **will not materialize** for a hybrid whose backward
  cost is dominated by conv + attention BPTT.
- **`num_steps = 1` undercuts the point.** With one tick per env step and
  analog input, there's little intra-trial spike-*timing* for EventProp to be
  "exact" about. To exploit it you'd move to a multi-timestep, spike-coded
  regime — a modelling change with its own retuning cost that may hurt the
  DefeatRoaches performance you already have.
- **You inherit EventProp's known pathologies** (both are in `E-prop.md`):
  exact gradients are **blind to spike creation/deletion** → needs *loss
  shaping* / near-threshold regularization (worse and less studied under RL
  exploration); and the `1/V̇⁻` factor **diverges** at tangential threshold
  crossings → needs clipping.
- **Two backward paths.** SIL (`_run_sil_pass`, `ppo_trainer.py` ~L2420) does a
  second `.backward()` through the SNN; it needs the same treatment.

**Verdict:** viable as a *research effort* with uncertain payoff, not a
weekend swap. If pursued, it must be staged behind Route D.

---

## 6. Route C — reconsider the target: **e-prop**, not EventProp

If the underlying goal is *"a more principled / online / memory-lean gradient
for my spiking RL agent,"* then **EventProp may be the wrong tool and e-prop
the right one**:

- **e-prop is online and forward** — eligibility traces × a learning signal, no
  backward unroll and no fixed-length trial. That maps naturally onto the
  step-by-step PPO acting loop and the cross-env-step state carry (Axis B),
  where EventProp's backward-adjoint-over-a-fixed-trial fights the RL structure.
- **e-prop was demonstrated on reward-based / RL tasks** in the original paper
  (reward-modulated learning signals). EventProp has **no** RL precedent.
- **ml_genn ships `EPropCompiler`** — but the same three Route-A blockers apply
  to it (pure-LIF, supervised API, dataset loop), so ml_genn-e-prop is not a
  drop-in either. The value here is **e-prop as an algorithm to implement
  natively in PyTorch** as an *approximate but cheap* alternative to TBPTT: an
  eligibility-trace rule can replace the 128-step unroll with an O(1)-memory
  running trace and still route a policy-gradient learning signal.
- **Cost:** e-prop's gradient is **approximate** (it drops the long-range
  temporal terms EventProp keeps exactly). Whether that approximation is good
  enough for kiting behaviour is an empirical question — again, exactly what
  Route D is for.

**This is the most important steer in the document:** before committing to
EventProp because the file is named after e-prop, decide *which property you
actually want* — **exactness** (EventProp) or **online/cheap/RL-native**
(e-prop). They pull in opposite directions.

---

## 7. Route D (recommended) — scoped proxy experiment before touching PPO

Don't gamble the working DefeatRoaches pipeline on an unproven backward pass.
Validate the learner in isolation first. Concrete staging:

**Stage 0 — decide the target (½ day, no code).**
Answer §6's question: exactness vs. online. Pick EventProp *or* e-prop as the
thing to prototype. Write the choice and the success metric into this doc.

**Stage 1 — standalone learner on a supervised toy (2–4 days).**
Implement the chosen rule as a PyTorch `autograd.Function` on a **single
recurrent LIF/Synaptic layer**, trained on a spike-latency or
`tiny_skirmish`-derived supervised task (the `envs/tiny_skirmish/` lab already
exists and is decoupled from PySC2). Gate on:
  - **gradient check** vs. finite differences (EventProp should hit <1e-4
    relative; e-prop will *not* — measure how far off, that's the point);
  - it actually learns the toy at all;
  - measure real memory/speed vs. a TBPTT baseline on the *same* toy.
This answers "does the math/code work" with **zero risk** to the main agent.

**Stage 2 — swap the neuron only, keep TBPTT scaffolding (behind a flag).**
Add a `gradient_mode: {tbptt | eventprop | eprop}` config switch. In the
non-TBPTT modes, replace the `snn.Synaptic` backward inside
`_replay_packed_chunk_group` with the Stage-1 `autograd.Function`, leaving the
conv/attention/heads on autograd. Keep the rollout/GAE/SIL machinery unchanged.
Run DefeatRoaches head-to-head against the TBPTT baseline. **Decision gate:** if
the exact/online SNN core doesn't beat (or at least match) TBPTT on
sample-efficiency or wall-clock, stop — the hybrid's bottleneck was never the
SNN backward.

**Stage 3 (only if Stage 2 wins) — lean into it.** Move to multi-timestep spike
coding (`num_steps > 1`), add loss shaping / gradient clipping, extend to the
attention neurons, consider learnable delays (the `deventprop` extension).

This ordering means the first real go/no-go costs days, not a rewrite, and
never breaks `master`.

---

## 8. Pain points, ranked

1. **Category mismatch (Axis A vs B) + `num_steps=1`.** EventProp optimizes
   intra-trial spike timing; your temporal signal is cross-env-step and your
   inner loop is one tick. Making EventProp meaningful changes the modelling
   regime. *(Conceptual — the deepest one.)*
2. **Non-LIF majority of the network** (conv, softmax attention, embeddings,
   categorical heads). Exact adjoint covers only the `Synaptic` core; attention
   over spikes has no EventProp rule at all. *(Architectural — hard limit.)*
3. **RL objective, not supervised.** PPO's clipped surrogate + value + entropy +
   SIL, with advantages changing every epoch, vs. EventProp/ml_genn's
   fixed-label supervised design. No EventProp-RL precedent exists.
4. **ml_genn's closed training loop** (`train(X, y)`) + host/device marshalling
   with PySC2. *(Integration.)*
5. **EventProp's own pathologies:** blindness to spike creation/deletion (needs
   loss shaping), `1/V̇⁻` divergence (needs clipping). Likely worse under RL
   exploration. *(Numerical stability.)*
6. **Second-order, learnable-τ neuron** (`snn.Synaptic`, learn_alpha/beta) — the
   adjoint has extra terms beyond the doc's standard LIF. *(Derivation effort.)*
7. **Two backward paths** (PPO + SIL) both go through the SNN.
8. **Toolchain** (only if ml_genn): GeNN on Windows + Blackwell 5090 + Py 3.12,
   sm_120/CUDA 12.8 support unverified. *(Ops — least fundamental.)*
9. **Test surface.** `tests/test_PPO.py` (tbptt replay), `test_sil.py`,
   `test_agent.py`/`test_analysis_tools.py` (assert on `token_snn.snn.alpha/beta`)
   all encode the current SNN/TBPTT internals and would need updates.

---

## 9. File touch-list (if Route B/D Stage 2 proceeds)

Core:
- `agent_core/spiking_policy.py` — the `snn.Synaptic`/`snn.Leaky` neurons,
  `surrogate.fast_sigmoid()` (L490), state init/reset. Primary edit site.
- `agent_core/ppo_trainer.py` — `_replay_packed_chunk_group` (~L1840),
  `_pack_chunk_group` detach (~L1766), `.backward()` (L1298), SIL
  `_run_sil_pass` (~L2420).
- `config.yaml` — add `gradient_mode`; `tbptt_window` (L11), `num_steps` (L123),
  `fast/slow_token_snn_alpha/beta` (L125–128).

State / plumbing (only if the state representation changes):
- `agent.py` (snn_state carry, L283/292/341), `agent_core/policy_protocol.py`
  (`PolicyInputBatch.state_in`), `distributed/protocol.py`
  (`pre_step_snn_state`), `distributed/rollout.py`, `train.py`.

Readout:
- `agent_core/target_heads.py` — if the loss/readout definition changes.

Tests:
- `tests/test_PPO.py`, `tests/test_sil.py`, `tests/test_agent.py`,
  `tests/test_analysis_tools.py`.

Deps (only if ml_genn): `requirements*.txt` (+ `pygenn`); note `spikingjelly`
is pinned but unused (dead dep) and `snntorch==0.9.4` is the real one.

---

## 10. Open questions for you (decision points)

1. **Exactness or online?** (§6) — do you want EventProp's exact gradient, or
   e-prop's online/cheap approximation? This picks the algorithm.
2. **Goal behind the swap?** Sample-efficiency? Wall-clock/memory? Neuromorphic
   deployment (Loihi 2 / SpiNNaker2)? Or curiosity/learning? The answer changes
   whether *any* of this is worth it — for pure curiosity, Route D Stage 1 is a
   great exercise; for a production win on DefeatRoaches, the payoff is doubtful
   because the SNN core isn't the bottleneck.
3. **Willing to change the modelling regime** (`num_steps > 1`, spike-coded
   inputs)? Without that, EventProp has little to be exact about.
4. **Keep TBPTT as the baseline** behind a `gradient_mode` flag (recommended),
   or hard cutover (not recommended)?

---

## 11. Bottom line

Ditching TBPTT for EventProp is **not a swap and not a drop-in**. Via ml_genn it
is a rewrite into a different architecture plus an unproven RL-on-supervised
hack. As a hand-written hybrid it is a real research build whose exact gradients
would touch only a minority of the network, with uncertain payoff because the
backward bottleneck here is conv+attention, not the spiking core. And the whole
question deserves a step back: **the RL-native cousin, e-prop, may be the better
target than the exact EventProp your doc actually describes.**

The recommended next action is **not** to branch the trainer, but to run
**Route D Stage 1** — a standalone, zero-risk validation on a supervised toy —
and let the gradient check and a memory/speed measurement tell you whether to go
further. That keeps `master` safe and turns a months-long "maybe" into a
few-days "yes/no."
