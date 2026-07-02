# Credit Assignment, Socratically — What Each Problem Is, How to Solve It, and Whether to Add It

Updated: 2026-06-29

> **STATUS 2026-07-02:** Problem 1's verdict ("SIL — not yet") has been
> overtaken: **SIL was implemented on 2026-06-30** (feedback-gated admission,
> not pure return-gating — the marine auto-attack confound in Problem 3 is
> exactly why) and is live in V7. The *teaching content* of every section
> remains current; only that one "should you add it?" answer is historical.
> This doc is the reference register and mastery test for Unit 10 in
> `learning/TUTOR_INSTRUCTIONS.md`.

Companion to `AUDIT_RECONCILIATION_2026-06-29.md`. That document says *what the
code does*. This one teaches *why* — each problem the research audit pointed at,
broken down so you can reason about it yourself instead of taking the method
names on faith.

The thread that runs through all five: **the click that wins is rare, late, and
ambiguous.** Every method below is a different answer to one of those three
words — *rare*, *late*, *ambiguous*.

A note on register: each section ends with a question. The point of the question
is not that you answer me — it's that if you can answer it, you understand the
method well enough to decide whether to build it.

---

## Problem 1 — "Rare success washes out" → Self-Imitation Learning (SIL)

### What is the problem, concretely?

Picture one update. Your 10 actors just played ~2560 steps. Out of those, maybe
**9** were a Smart click that genuinely hit a roach and contributed to a kill.
The other ~2551 were movement, no-ops, missed clicks, kiting.

PPO's policy gradient is an *average* over all those steps. Your 9 great clicks
are 0.35% of the batch. Their gradient signal — "do *that* again" — is drowned.

There is a second, sneakier half to this. PPO is **on-policy**: once an update
consumes a rollout, that data is thrown away. So that one brilliant click from
three updates ago? Gone. You can't keep nudging the policy toward it. You got
*one* weak vote for it, averaged into noise, and then you deleted the evidence.

> *Question to sit with:* if a great click only ever gets one diluted vote and is
> then destroyed, how is the policy supposed to *lock on* to it?

### How SIL solves it, in theory

SIL keeps a **trophy case**: a replay buffer of your past high-return
transitions. During every PPO update, you replay a batch from the trophy case
and add an extra loss:

```
L_sil = −log π(a|s) · max(0, R − V(s))
```

Read that term by term:

- `−log π(a|s)` is "make this good past action more likely" (standard policy
  gradient direction).
- `R − V(s)` is "how much better was this old return than I *currently* think
  this state is worth?"
- `max(0, ·)` is the gate: **only imitate if the old action was genuinely
  better than your current estimate.** If your policy has already caught up and
  `V(s)` now matches `R`, the gate is zero and SIL does nothing — it doesn't
  drag you backward.

So instead of one diluted vote that gets deleted, your best moments get
rehearsed *every update*, for as long as they're still ahead of your current
policy. That is the whole paper (Oh et al. 2018), and its tagline — *"exploiting
past good experiences indirectly drives deep exploration"* — is exactly the
mechanism: by rehearsing a known-good click, you eventually make it likely
enough to *sample* again, which re-explores the region around it.

### Should you add it?

**Not yet — and your smart-outcome shaping is already half of it.**

SIL's trophy case is empty if you have no successes. The audit itself says so:
*"SIL can't bootstrap from zero successes."* In V5 you had zero. Now (V6+) you
have clicks landing and a `smart_outcome_detector` emitting `attack_likely`
rewards *immediately* on the click step.

That shaping is doing a job SIL would also do — it stops the good click from
being invisible — but it does it the cheap way: reward *now*, while the memory
of which command caused it is one step old, instead of replaying it *later*.

So the real question is empirical: **does shaping alone plateau?** Run V6-style
long enough to see a learning curve. If reward climbs and then sticks while
click-quality is still occasionally good-but-rare, *then* SIL is the natural
next move, because you'll finally have a steady stream of trophies to fill the
case.

> *Question to sit with:* what would the learning curve *look like* if the
> problem were "rare successes washing out" specifically? (Hint: not a flat
> line at zero — that's "clicks don't land." Something else.)

---

## Problem 2 — "The critic might be hurting" → fix it, or go critic-free (GRPO / RLOO)

### What is the problem, concretely?

PPO computes an **advantage** for each step:

```
advantage = return − V(s)
```

`V(s)` is your critic's guess at "how good is this state." Subtracting it is
supposed to *remove* the obvious part of the return (the part any action would
have gotten) and leave only the part *your action* caused. Less variance →
cleaner gradient. That is the entire reason the critic exists.

But if `V(s)` is **garbage** — if explained variance is near 0 — then you are
subtracting *noise*. The baseline that was supposed to *remove* variance is now
*adding* it. Your updates get a worse signal than if you'd used no critic at all.

Your critic is in exactly that suspicious zone: EV roughly `0.003–0.22`. Low but
positive — so it's not clearly net-negative yet, but it's not clearly helping
either.

> *Question to sit with:* if `V(s)` is just wrong, in what sense is
> `return − V(s)` worse than `return − (a constant average)`?

### Why is the critic hard *here*?

Two reasons, and they're the LLM-RL people's reasons too:

1. **Rewards are sparse and late.** A roach death is a big reward that arrives
   many steps after the click that earned it. The critic has to *predict* that
   future reward from the current 84×84×27 screen. That is hard.
2. **The state is rich.** A small value head on a spiking latent is being asked
   to summarize a lot. Undercapacity → low EV.

This is precisely why GRPO (DeepSeekMath) and RLOO **throw the critic away**.

### How critic-free methods solve it, in theory

Replace the learned `V(s)` with the **mean return of a group** of rollouts:

```
advantage_i = return_i − mean(returns in the group)
```

No learned model. No wrong baseline. The group mean is just arithmetic — it
cannot be "inaccurate," it's a fact about your samples.

GRPO groups by **the same starting state** (the same prompt, in LLM land):
sample K answers to one prompt, z-score their rewards, the bad ones get
negative advantage, the good ones positive. RLOO does the same with a
leave-one-out mean.

### The catch — and it's a big one for SC2

GRPO needs **K rollouts from the *same* state.** For an LLM that's free: pass
the same prompt through K times. For StarCraft it is not free at all — you
cannot cheaply clone a mid-game battlefield K times. Your 10 actors each get a
*different* randomized start. So you cannot group by identical state. At best
you group by *episode* or by *similar start config*, which is a far weaker
baseline — it loses the thing that makes GRPO clean.

And there's a published counterweight: *"Learning Without Critics?"*
(2511.03527) finds that on **long-horizon** control tasks, learned critics
*still* beat critic-free baselines — because a group mean is a single number
with no *state-dependent* credit. It can say "this batch was above average" but
not "this *specific* state is valuable." DefeatRoaches episodes are 3600 steps.
That's long-horizon. Going critic-free here is genuinely risky.

### Should you add it?

**Measure first — you already have the instrument.** `explained_variance` is
logged every update. The decision tree is:

- EV is low-but-positive **and reward is climbing** → critic is fine. Leave it.
- EV ≈ 0 **and reward plateaus** → the critic is plausibly net-negative. *Now*
  it's worth an experiment.

And the cheapest experiment is **not** full GRPO. It's an **episode-level return
baseline** (RLOO-flavored): subtract the mean return of the episode (or batch)
instead of `V(s)`. One line change in the advantage computation, no new
networks, no identical-state requirement. If that beats the critic on the
plateau, you've confirmed the diagnosis cheaply. Only escalate to GRPO-style
grouping if you can solve the "same state" problem (e.g. deterministic resets to
a saved scenario — which is its own project).

> *Question to sit with:* your episodes are 3600 steps. A group-mean baseline
> ignores *where in the episode* you are. At step 50 (nothing happened yet) and
> step 3500 (a roach is about to die), is "subtract the episode mean" giving
> you the same quality of information? Which one does it mislead you on?

---

## Problem 3 — "Did *my* click cause that, or did a marine auto-attack?" → CCA / HCA

### What is the problem, concretely?

A roach's health drops to zero. Reward fires. GAE spreads that reward backward
over recent steps and credits whatever actions happened near the death — *even
if those actions were a no-op and the damage came from a unit auto-attacking on
its own while you idled.*

The agent literally cannot tell **skill from luck**. It might learn "no-op near
a fight is good" because no-ops kept coinciding with roach deaths that your
*earlier* click actually caused.

> *Question to sit with:* if damage happens whether or not your click was the
> cause, how does the gradient tell the difference?

### How CCA / HCA solve it, in theory

**Counterfactual Credit Assignment (CCA)** trains a *second* model — a baseline
conditioned on **action-independent** features of the future. Because that
baseline is constructed so it *cannot see your action*, subtracting it leaves
behind exactly the part of the return your action *caused*. "Separating skill
from luck" is the paper's own phrase.

**Hindsight Credit Assignment (HCA)** asks a different question: *"given the
future that actually happened, what's the probability my policy would have taken
this action?"* Credit goes inversely to that probability — rare-but-correct
clicks get amplified, because they were *surprising* given the good outcome.
That is the formal version of "credit the unlikely good move."

### The catch

- Both need **an extra trained model** (a future-conditional baseline, or a
  hindsight action-probability model). More machinery, more things to train
  stably.
- HCA's ratio is **high-variance** and breaks when the action had near-zero
  probability: if a correct click almost never gets sampled, you can't estimate
  "how surprising was it," so it gets ~zero credit. (This is *another* reason
> SIL pairs well — SIL *keeps* those rare clicks alive in replay so HCA could
> eventually see them. But you're not building HCA, so file this as trivia.)

### Should you add it?

**No — and you've already built the cheap version.**

Your `smart_outcome_detector` is a **hand-built attribution heuristic**: it looks
back within a 5-step window from a Smart click and classifies the outcome
(`attack_likely` if an enemy's health dropped after a click near it, etc.). That
*is* credit assignment — crude, heuristic, but it directly attributes a health
drop to the click that plausibly caused it, and rewards that click *immediately*.

CCA/HCA would automate and sharpen this. But you only reach for the expensive,
principled version when you have **evidence the heuristic is wrong and it's
costing you** — e.g. the detector rewards clicks that didn't actually cause
damage, and you can see the policy exploiting that. Until then, the detector is
doing 80% of the job for 5% of the complexity.

> *Question to sit with:* can you think of a situation where your detector would
> give `attack_likely` to a click that did *not* cause the damage? (If you can
> name one, that's the seam where CCA would eventually earn its keep.)

---

## Problem 4 — "The agent camps in a safe corner forever" → RND

### What is the problem, concretely?

There is a comfortable local optimum: sit in the corner, take no damage, accrue
only the small step penalty. From *inside* the policy, sitting still has
decent-ish value (you're not dying), so the gradient has no strong reason to
leave. The agent has found a "safe" basin that is nowhere near the global
optimum (which requires attacking), and it cannot tell, because everything
outside the basin is *unknown* and therefore has no estimated value to lure it
out.

This is the classic exploration problem: **you can't be greedy toward a reward
you haven't found, and you won't find it by being greedy.**

> *Question to sit with:* if every unseen state has, by default, an *unknown*
> value, what should the agent *assume* that value is to be willing to go look?

### How RND solves it, in theory

Random Network Distillation adds an **intrinsic reward for novelty**:

1. Take a fixed random network (frozen, never trained) — call its output the
   "target."
2. Train a tiny *predictor* network to imitate that target on the states the
   agent visits.
3. The predictor's **error** is the intrinsic reward.

On states the agent has visited a thousand times (the corner), the predictor has
learned to copy the target well → low error → **low novelty reward.** On states
it has never seen, the predictor is wrong → high error → **high novelty reward.**
The corner stops being comfortable because it pays nothing; the unexplored map
pays a constant trickle of novelty. The agent is pulled out of the basin by
curiosity, not by the (still-unknown) task reward.

It is cheap (two small networks, one frozen), modular (it's just a reward term
added on top), and composes with everything — including the SNN, because it's a
side network, not a change to the policy.

### Should you add it?

**Only if a current run still shows corner-stalling.**

V5 had this symptom badly (all clicks bottom-left, never engaging). But V6 has
the skip connection *and* the smart-outcome shaping *and* the right-click
curriculum — three things that may have already broken the basin. The shaping
rewards engaging the enemy; the curriculum penalizes no-op early. The corner may
already be uncomfortable.

**Measure first.** Watch a V6 eval trace: does the agent still retreat to a
corner and sit? If yes → RND is the right, cheap fix. If no → you'd be adding
machinery to solve a problem you no longer have.

> *Question to sit with:* novelty reward rewards *any* unseen state, including
> useless ones (a weird patch of empty map). What stops RND from making the
> agent wander forever instead of fighting? (Think about what happens to novelty
> reward once everywhere has been seen.)

---

## Problem 5 — "The reward arrives too late for BPTT to reach back to the click" → e-prop

### What is the problem, concretely?

This is the **SNN-native** version of problems 1 and 3 combined, and it is the
one item on the whole list that is *more natural on a spiking net than on a
normal ANN*.

Your SNN learns with **BPTT** — backprop-through-time, in chunks of
`tbptt_window = 128`. When a reward arrives, the error signal has to flow
*backward through the SNN's spiking state*, step by step, to reach the synapses
that fired when the click happened. Two consequences:

1. **It's expensive** — you carry hidden state and gradients across 128 steps.
2. **It's horizon-limited** — if a click's reward arrives *beyond* the 128-step
   window, the gradient literally cannot reach the click. It's invisible to
   learning.

For DefeatRoaches, a click-to-damage gap is usually well within 128 steps, so
BPTT is adequate *today*. But the *shape* of the problem — a delayed sparse
reward that must be credited to an earlier spike — is exactly what spiking
brains solve natively.

> *Question to sit with:* if you can't afford to remember 1000 steps of state
> for backprop, what could a synapse *store locally* so that, when a reward
> finally shows up, it already "knows" whether it helped?

### How e-prop solves it, in theory

**Eligibility traces.** Each synapse keeps a *fading local memory* of its own
recent spiking — an "eligibility trace" that decays over time. It requires no
backward pass; it's just a running average the synapse maintains itself.

When a delayed reward finally arrives (a "modulatory signal"), the update is:

```
Δw ∝ eligibility_trace × reward_signal
```

Synapses that fired just before the reward have a high trace → they get
strengthened. Synapses that fired long ago have a trace near zero → they're
left alone. **The trace already remembers "I was active recently," so the reward
doesn't need to backpropagate through time to find the responsible synapses —
they've already tagged themselves.**

This is the biologically-plausible alternative to BPTT, and it is the strongest
argument on the entire list for *staying on Path A with the SNN*: it turns the
spiking substrate from a handicap (spikes are awkward for BPTT) into a feature
(eligibility traces are natural for spikes). It is the one method here that
would make "we use an SNN" a *research contribution* rather than a constraint
you're working around.

### The catch

It is a **change to the learning rule**, not a bolt-on loss. It replaces BPTT.
That is the biggest commitment on this list — you'd be re-plumbing how the whole
network learns, not adding a term to the loss.

### Should you add it?

**Only if you decide the SNN learning rule itself is the point of the project.**

Right now BPTT-with-window-128 covers your click-to-damage horizons fine. e-prop
earns its keep in two situations:

1. When rewards routinely arrive **beyond your BPTT horizon** (they don't, yet).
2. When **biological plausibility / SNN-native learning is itself the research
   question** you want to answer.

If neither is true today, this is the "interesting later" tier — the thing you
build when "make an SNN agent that learns" is solved and the next question is
"make the SNN learn the *spiking* way."

> *Question to sit with:* if you replaced BPTT with e-prop, what would you lose?
> (BPTT is *exact* within its window; e-prop is an *approximation*. What does
> that approximation cost you on the clicks that *are* within the window?)

---

## The question underneath all five — are you measuring the right thing?

Before adding any of the methods above, there is a cheaper and more important
check: **is your reward the actual objective?**

Every term in `defeat_roaches_v4` is *shaped* — `damage_dealt`, `kill_reward`,
distance bands, smart-outcome, step penalty. They are all proxies pointing *at*
the goal. Dense proxies can be hacked: the agent maximizes the proxy and the
true goal — the native SC2 score — stalls. Think of the failure: an agent that
learns to nudge into the `distance_reward` band and hover, or to fish for
`attack_likely` bonuses, *without* actually winning. The shaped reward goes up;
the game score does not.

The good news: **you already log the native score** (`score_diagnostics_*`). You
just don't train on it. So the cheapest experiment on this whole list isn't SIL
or RND — it's an ablation where the reward **is** the raw SC2 score (or score +
a terminal win/loss term), to confirm the shaping tracks the real objective and
isn't being gamed.

Verify the scaffold is aligned *before* you build more scaffolding on top of it.

> *Question to sit with:* if your shaped reward is climbing but the native game
> score is flat, what has the agent actually learned? And — how would you even
> notice, if you only ever look at the shaped reward?

---

## So — should *all* of them be added?

**No.** Three reasons, and they matter more than any single method:

### 1. They overlap. Stacking them stacks complexity on the *same* problems.

| Underlying problem | Methods that address it |
|---|---|
| Rare success washes out / wrong credit | smart-outcome **shaping** (have it), **SIL**, **CCA/HCA**, **e-prop** |
| Bad baseline | **fix critic**, **RLOO baseline**, (GRPO — doesn't fit SC2) |
| No exploration | **right-click curriculum** (have it), **RND** |

You already own two cheap answers (shaping, curriculum). Adding SIL *and* CCA
*and* e-prop would be three overlapping fixes for "credit the rare good click."
Pick the *one* that matches your *measured* failure, not all three.

### 2. Order is not optional. Most of these are gated on a foundation.

You cannot fill SIL's trophy case with zero successes. You cannot tell if the
critic is the bottleneck without a learning curve. You cannot justify CCA until
you've shown the detector mis-attributes. **Every one of these is a fix for a
*specific, measurable failure mode* — and right now your only measured failure
was "clicks don't land," which you already fixed.**

The right workflow is: run, watch the metrics (`explained_variance`, reward
curve, click-quality eval traces), let the *next* failure announce itself, and
reach for exactly the method that failure points at. The failure tells you which
door to open. Don't open all five at once — you won't know which one helped.

### 3. Each adds machinery. Add machinery only on evidence.

A trophy-case buffer, a predictor network, a hindsight model, a brand-new
learning rule — each is something to train, tune, debug, and keep stable. Every
piece of machinery you add without evidence is a piece that can quietly break
and that you'll have to maintain. The cheapest code is the code you didn't write.

> *The one-line version:* **measure, then fix.** The audit was a map of where
> you *might* go next. The code you've written is the road that gets you to the
> point where the map becomes useful.
