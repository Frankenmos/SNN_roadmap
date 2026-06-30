# Audit Reconciliation — the Credit-Assignment Research Audit vs the Code

Updated: 2026-06-29

Question answered: the credit-assignment research audit (SIL / GRPO / RLOO /
CCA / HCA / RND / e-prop, "stay on Path A with the SNN") — what did it ask for,
what does the code actually do, and where (if anywhere) are there real mistakes?

This is the companion to `V5_COLLAPSE_AUDIT.md`. That document explains the
*architectural* collapse (fine-stage spatial blindness). This one reconciles the
*algorithmic* research menu against the code as it stands today.

## Short answer

- **The audit is solid.** Every citation is real and the load-bearing technical
  claims hold up (SIL's `(R−V)+` is verbatim Eq. 2 of Oh et al. 2018; GRPO's
  "K rollouts from the same state" limit is real and is the correct reason it
  does not port literally to SC2). Two minor footnotes only — see the appendix.
- **None of the audit's five ranked recommendations are in the code.** This is
  **not a mistake.** The goal was "make it work and learn first," and the audit
  itself assumed a foundation that the code was still being built on when the
  audit landed.
- **What was built instead is the correct prerequisite**, not a detour. The
  audit's own caveat — *"SIL can't bootstrap from zero successes"* — is exactly
  why click-landing had to come first.
- **One genuinely worth-a-careful-eye item** (the right-click curriculum's
  ratio bias) and **one to keep watching** (the low-EV critic still in the GAE
  loop). Neither is a bug; both are noted below so the next person (you) sees
  them coming.
- **One forward-looking critique worth acting on eventually**: every reward term
  in the code is *shaped*, not the game's true objective — so at some point
  validate against the **raw SC2 score** (which you already log but don't train
  on). See the dedicated section below.

## What the audit recommended vs what the code does

| # | Audit recommendation | What the code does today | Verdict |
|---|---|---|---|
| 1 | **SIL** — replay buffer of past high-return transitions + a `(R−V)₊` auxiliary loss so rare good clicks aren't washed out. | Standard PPO + GAE in `agent_core/ppo_trainer.py`. No imitation buffer, no `(R−V)₊` term. | **Correctly deferred.** In V5, `0 / 1090` clicks were near an enemy — there were no successes to imitate. SIL postdates clicks-landing. |
| 2 | **Confront the critic** — measure whether the EV≈0 critic is net-negative; if so, fix the value head *or* swap the advantage baseline for a robust RLOO-style batch-return mean. | EV **is** measured and logged (`explained_variance` in `update_policy`). The critic is still the GAE baseline. Advantages are batch-normalized — that is *standard PPO advantage standardization*, not a critic-replacement. | **Diagnostic half done; action half deferred.** Appropriate: you can watch EV now and only act if it stalls. |
| 3 | **RND** — a novelty-predictor bonus to pull the agent out of the corner-stall. Conditional ("only if corner-stall persists"). | No predictor network, no novelty term. | **Correctly deferred.** The corner-stall is a V5 symptom; the skip-connection + reward shaping may have removed it. Re-check before building RND. |
| 4 | **e-prop / eligibility traces** — SNN-native delayed credit; an alternative learning rule to BPTT. Highest effort. | SNN trains with **BPTT** (TBPTT window 128). No eligibility traces. | **Correctly deferred.** The audit ranked this last and called it a research-grade commitment. Right to leave it. |
| 5 | **CCA / HCA** — principled "was it *my* click?" attribution. | None. | **Correctly parked.** The audit explicitly said park it. |

## What was built instead — and why that was the right layer to be on

The audit's recommendations all assume the same thing: *the agent can already
produce an occasional correct, damaging click, and the problem is that the
gradient for that click gets diluted or mis-attributed.* That assumption is the
whole premise. V5 violated it:

- `1,094 / 1,099` Smart clicks landed on two bottom-left coordinates.
- `fine` sub-index was the constant `10` for **all** `1,099` clicks.
- `0 / 1090` clicks were near an enemy.

There is no credit to assign when no click ever connects. So the work that
actually shipped is the foundation the audit was standing on:

1. **Fine skip connection** (`fine_skip_connection: true`, `CoarseToFineTargetHead`)
   — feeds pre-pool 84×84 conv features into the fine stage so a sub-cell click
   can depend on what is actually on screen. This is what lets clicks localize.
2. **Smart-outcome detector + reward shaping** (`defeat_roaches_v4`,
   `smart_outcome_detector`) — classifies each click's outcome
   (`attack_likely` / `fired_likely` / `null_unclear`) within a 5-step window and
   emits an *immediate* per-click reward. **This is, in spirit, a cheap answer
   to the audit's #1.** Where SIL would replay a rare good click later, shaping
   rewards the click *now*, while the memory of which command caused it is
   fresh. It is the "make it work and learn first" version of the same idea.
3. **Right-click curriculum** (`right_click_curriculum_*`) — a self-rolled
   exploration nudge (decaying no-op logit penalty) to discourage passive
   no-op early in training. Related to the exploration theme the audit's RND
   addresses, but hand-built and task-specific rather than a novelty model.
4. **Plumbing**: Ray eval / best-checkpoint, extractor-normalizer sync, kill /
   terminal / timeout reward semantics, bf16 AMP default.

V6 (`banana_glasses_v6`) reached max reward `555.85` after this stack. That is
the "it works" milestone the audit's menu is meant to *follow*, not precede.

> The ordering was right. The audit was a menu for "clicks are landing but rare
> successes wash out." You were still on "make clicks land." Fixing the latter
> first is exactly what the audit's own SIL caveat demanded.

## Things genuinely worth a careful eye (not bugs — reading-aids)

### 1. The right-click curriculum biases the PPO ratio early in training

`_apply_right_click_curriculum` subtracts a decaying penalty from the no-op
logit. It runs in **both** sampling (`select_action`) and replay
(`_forward_replay_step_tensors`). The penalty size depends on `self.update_count`.

Consequence: the *old* log-prob stored at rollout time used the penalty from the
*previous* update; the *new* log-prob computed during the update uses the
current (slightly smaller) penalty. So `ratio = exp(new − old)` carries a small
systematic drift away from 1 that is **not** caused by the network weights
changing — it is caused by the curriculum decaying.

This is almost certainly **intentional** (it is a curriculum: it is *supposed*
to push the policy off no-op), and it decays to zero over
`right_click_curriculum_updates` (120 updates). It is not a bug. It is worth
knowing because, during those first ~120 updates, the KL/clip diagnostics are
measuring curriculum drift as well as real policy movement. If you ever see
weird early-training KL spikes you can't explain, this is the first place to
look.

### 2. The low-EV critic is still the GAE baseline

The audit's concern ("at EV≈0 the baseline is adding variance, not removing
it") is not yet acted on — the critic is still inside `_compute_advantages`.
That is fine *for now*: EV is logged every update, so you can watch it. The
trigger to revisit audit recommendation #2 is concrete and observable:

> If `explained_variance` stays near 0 **while learning plateaus** (reward
> flattens but EV does not climb), the critic is plausibly net-negative and the
> RLOO-style return-baseline swap becomes worth trying.

Until that trigger fires, leave the critic alone. Don't optimize a thing you
haven't measured failing.

### 3. `clip_eps: 0.10` (tighter than the textbook 0.2)

Not a mistake — a deliberate stability choice, paired with `target_kl: 0.03`
early-stopping. Just flagging it so nobody "fixes" it back to 0.2 thinking it's
a typo. With a spiking net and bf16, conservative clipping is the safer default.

## Critique (added 2026-06-29): validate against the raw SC2 score

Every reward term in `defeat_roaches_v4` is *shaped* — `damage_dealt_coef`,
`kill_reward_coef`, distance bands, `step_penalty`, smart-outcome bonuses. None
of it **is** the game's actual objective. These are proxies meant to *point at*
the objective. The classic RL failure mode for dense shaping is **reward
hacking**: the agent maximizes the proxy and leaves the true goal behind — e.g.
farming `attack_likely` signals, or hovering in the distance band for
`distance_reward` without ever committing to a kill.

The defense is to periodically check the proxy against the **real thing**: the
native SC2 score (PySC2's `score_cumulative`), which is what the DeepMind
mini-game benchmarks report and what "doing well at DefeatRoaches" actually
means. Two things make this cheap here:

1. **You already log it.** The harness writes `score_diagnostics_{det,stoch}.jsonl`
   every run. The native score is already sitting in your results — it's just
   not the training signal.
2. **It's the literature's metric.** Comparison to published DefeatRoaches
   numbers only works on the native score, never on your shaped reward.

The critique, concretely: at some point run an **ablation / eval where the
reward *is* the raw native score** (or native score + a terminal win/loss term)
and confirm one of two outcomes:

- The shaped-reward policy's **native score** climbs in step with the shaped
  reward → shaping is aligned with the true goal, keep it. *(This is the
  expected healthy outcome — and the one worth verifying rather than assuming.)*
- The shaped reward climbs while the **native score** flattens or drops → the
  shaping is being gamed; find and trim the offending term.

This belongs to the same "measure, then fix" family as everything else above:
the native score is the ground-truth ruler. The shaping is a *scaffold*; the
critique is simply "remember to measure the building against the ruler, not
against the scaffold." It is also the single cheapest experiment on the entire
audit menu, because the data already exists — you're only changing what you
optimize, not what you observe.

## Where the audit's recommendations stand now

All five are **correctly parked**, and the parking order matches the audit's
own ranking:

1. SIL — revisit *after* a run shows real, repeatable damaging clicks but slow
   policy lock-on. The smart-outcome shaping may already cover the same need;
   check whether shaping alone plateaus before building SIL.
2. Critic — watch EV. Act only on the trigger above.
3. RND — only if a fresh run still shows corner-stalling behavior.
4. e-prop — only if the SNN learning rule itself becomes the research question.
5. CCA/HCA — keep parked.

## Appendix: citation verification (2026-06-29)

All twelve citations independently verified. Real, correctly attributed,
mechanism descriptions accurate. Two precision footnotes:

- **GRPO (2402.03300)** — the mechanism and the "K rollouts per same state"
  applicability limit are correct. The framed quote *"hard to train to be
  accurate when reward is sparse"* is a **paraphrase**, not verbatim; the
  paper's *primary* stated reason for dropping the critic is memory/compute
  cost, with token-level reward sparsity as a secondary remark. Does not change
  the recommendation.
- **RLOO (2402.14740)** — the paper is real and does use a leave-one-out
  group baseline, but it is an RLHF-adoption paper (Ahmadian et al., Cohere),
  **not** the RLOO originator. RLOO originates with Kool et al.,
  arXiv:1905.03193 (AAAI 2020). Cite the originator for the concept.

Everything else (SIL, SPEAR, Dr.GRPO, "Learning Without Critics?", CCA, HCA,
COMA, HER, RND, e-prop) checked out clean — correct ID, authors, venue, year,
and mechanism.

## TL;DR

The audit was a good, well-sourced menu for the *next* phase. You correctly
spent this phase building the foundation it assumed. Nothing here was "done
wrong." The two reading-aids (curriculum ratio drift; watch the critic's EV)
are the only things to keep in your peripheral vision.
