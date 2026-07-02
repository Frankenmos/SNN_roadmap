# Learning Materials — curated for this project

Companion to `TUTOR_INSTRUCTIONS.md`. Organized by curriculum unit. Priority
markers: ★ = core, do it; ◆ = strong second pass; ○ = only if the itch
strikes. Everything here was chosen for *this* project — a general ML
curriculum would look different.

A note on how to use this list: **materials are for between sessions, not
instead of them.** The pattern that works: tutor session introduces the idea
→ material consolidates it → next session opens by testing it. Reading first
and discussing second tends to produce the "makes sense" illusion.

---

## Unit M — Math reactivation

- ★ **3Blue1Brown — Essence of Calculus** (YouTube playlist, ~3h total).
  Chain rule and derivatives as pictures. You were prépa-strong; this is
  reactivation, watch at 1.5×.
- ★ **3Blue1Brown — Essence of Linear Algebra** (~3h). Matrix-as-
  transformation intuition; needed for attention (QKᵀ) and for reading
  network code without fear.
- ◆ **Deisenroth, Faisal, Ong — "Mathematics for Machine Learning"** (free
  PDF at mml-book.github.io). Chapters 5 (vector calculus) and 6
  (probability). Reference, not cover-to-cover.
- ○ **Blitzstein & Hwang — Introduction to Probability** (free at
  probabilitybook.net) if expectation/variance manipulation feels shaky
  during Unit 2.

## Units 0–2 — RL foundations & policy gradients

- ★ **Sutton & Barto — "Reinforcement Learning: An Introduction"** (2nd ed.,
  free PDF at incompleteideas.net/book). THE book. For this project read:
  Ch. 3 (MDPs), Ch. 9 intro (function approximation, skim), **Ch. 13 (policy
  gradient) carefully** — 13.1–13.4 is exactly Unit 2 and Unit 3.
- ★ **OpenAI Spinning Up — "Intro to Policy Optimization"**
  (spinningup.openai.com). The single best written derivation of the policy
  gradient + the baseline lemma ("Extra material: proof that the gradient of
  the expected return..."). Short, rigorous, code-adjacent. Read the three
  "Intro to RL" essays plus the Vanilla Policy Gradient doc page.
- ◆ **David Silver's RL course** (DeepMind/UCL, YouTube). Lectures 1–2
  (MDPs), 7 (policy gradient). Slower but deep; good if Sutton reads dry.
- ○ **Lilian Weng — "Policy Gradient Algorithms"** (lilianweng.github.io).
  Dense survey; excellent as a *map* after Units 2–5, overwhelming before.

## Units 3–5 — Advantage, GAE, PPO

- ★ **Schulman et al. 2017 — "Proximal Policy Optimization Algorithms"**
  (arXiv:1707.06347). Short and readable as papers go. Read after the tutor
  session, not before. Your config deviates from it on purpose (clip 0.10 vs
  0.2) — find where and ask why.
- ★ **Schulman et al. 2015 — "High-Dimensional Continuous Control Using
  Generalized Advantage Estimation"** (arXiv:1506.02438). Sections 1–3 only.
  This is where the λ dial lives.
- ★ **"The 37 Implementation Details of Proximal Policy Optimization"**
  (ICLR Blog Track 2022, by Huang et al.). Gold for YOU specifically: it is
  the gap between the paper and a working implementation — which is exactly
  the gap between your intuition and your code. Read with `ppo_trainer.py`
  open and check off which details your code has.
- ◆ **Spinning Up — PPO and "Kinds of RL Algorithms"** doc pages.
- ○ **Achiam's Spinning Up exercises** — implement REINFORCE on CartPole
  yourself in <150 lines. One evening; converts Unit 2 from "understood" to
  "owned". Honestly the highest-value ○ on this list.

## Unit 6 — BPTT / recurrence

- ◆ **Karpathy — "The Unreasonable Effectiveness of Recurrent Neural
  Networks"** (blog, 2015). Old but the cleanest intuition for unrolling.
- ◆ Your own `docs/current/THE_BPTT.md` and `FRAGMENT_PPO.md` — read them as
  a student now, not as the author. Where you can't reconstruct the
  reasoning, that's a session topic.

## Unit 7 — Spiking neural networks

- ★ **snntorch tutorials 1–5** (snntorch.readthedocs.io) — Eshraghian's
  official series: LIF neuron → surrogate gradients → training. Your network
  is built from exactly these parts (`snn.Leaky`, `snn.Synaptic`,
  `surrogate.fast_sigmoid`). Do them as notebooks, not reading.
- ★ **Neftci, Mostafa, Zenke 2019 — "Surrogate Gradient Learning in Spiking
  Neural Networks"** (arXiv:1901.09948). THE review of why/how gradients
  flow through spikes. Directly answers "what approximation am I making?"
- ◆ **Eshraghian et al. 2023 — "Training Spiking Neural Networks Using
  Lessons From Deep Learning"** (arXiv:2109.12894) — the snntorch paper;
  broader context, rate vs temporal coding.
- ◆ **Bellec et al. 2020 — "A solution to the learning dilemma for recurrent
  networks of spiking neurons"** (e-prop, Nature Communications). Read the
  main text only, for Unit 10's eligibility-traces thread and the "SNN-native
  credit assignment" dream. Skip the supplement.

## Unit 8 — Attention / transformers

- ★ **Jay Alammar — "The Illustrated Transformer"** (jalammar.github.io).
  You already started a transformer roadmap once — this is the resurrection
  point.
- ★ **3Blue1Brown — "Attention in transformers, visually explained"**
  (2024 video). QKV as geometry; pairs with computing a 2-token example by
  hand in the session.
- ◆ **Vaswani et al. 2017 — "Attention Is All You Need"** (arXiv:1706.03762).
  After the two above, read §3 only (the architecture).
- ○ **Karpathy — "Let's build GPT"** (YouTube, 2h). If you ever want
  attention in your fingers rather than your head.

## Units 9–11 — Reward design, credit assignment, evaluation

- ★ **Oh et al. 2018 — "Self-Imitation Learning"** (arXiv:1806.05635). You
  implemented Eq. 2 of this paper. Read it AFTER Unit 10's SIL session and
  check: where does your implementation deviate (admission rule! single-step
  trophies!) and can you defend each deviation?
- ★ **Amodei et al. 2016 — "Concrete Problems in AI Safety"** §"Reward
  hacking" (arXiv:1606.06565) — the canonical framing of Unit 9. Short
  section; alternatively DeepMind's "Specification gaming: the flip side of
  AI ingenuity" blog post (2020) — 20 minutes, and its examples list is
  delightfully horrifying.
- ◆ **Burda et al. 2018 — "Exploration by Random Network Distillation"**
  (arXiv:1810.12894). Parked in your roadmap, but the mechanism is 2 pages.
- ◆ **Kool et al. 2019 — "Buy 4 REINFORCE Samples, Get a Baseline for
  Free!"** (arXiv:1905.03193 / AAAI-2020 version) — RLOO's origin;
  DeepSeekMath (arXiv:2402.03300) §4 for GRPO. Read to understand why
  neither fits SC2 cleanly (the same-state grouping problem).
- ◆ Your own `docs/current/CREDIT_ASSIGNMENT_SOCRATIC.md` — the mastery test
  for Unit 10 is literally answering its embedded questions.

## The inspiration shelf (why you're here — read for fuel, not curriculum)

- ○ **Silver et al. 2017 — "Mastering the game of Go without human
  knowledge"** (AlphaGo Zero, Nature) — readable, and after Unit 5 you will
  understand ~70% of it, which is a wonderful feeling.
- ○ **Vinyals et al. 2019 — AlphaStar** (Nature, "Grandmaster level in
  StarCraft II...") — your project's giant cousin; recognize the pieces
  (entity tokens! spatial features! auto-regressive actions!).
- ○ **Vinyals et al. 2017 — SC2LE** (arXiv:1708.04782) — the paper that
  defines your mini-game and reports the baseline DefeatRoaches scores your
  native-score eval should eventually be compared against.
- ○ **Jumper et al. 2021 — AlphaFold 2** (Nature) — not RL, but attention
  everywhere; after Unit 8 the Evoformer stops being magic.

## Course-shaped alternatives (if self-pacing fails)

- ◆ **HuggingFace Deep RL Course** (free, hands-on, PPO unit included) —
  structured, gamified, good if the paper-and-book path stalls.
- ○ **CS285 (Berkeley, Levine)** lecture videos — graduate-level policy
  gradient lectures (5–9); heavier math, matches prépa taste.

---

## Anti-procrastination clause

You have written roadmaps before and abandoned them — your words. So, rules:

1. This file is a *menu*, not a queue. You never owe it completion.
2. The unit order in TUTOR_INSTRUCTIONS.md wins over curiosity-driven
   wandering *during sessions*; wandering is free the rest of the time.
3. One ★ item per unit is enough to pass the unit. ◆ and ○ are bonuses.
4. If two weeks pass with no PROGRESS.md entry, the next session starts with
   15 minutes of Unit 2's mastery question, whatever else was planned. (The
   policy gradient derivation is the load-bearing wall; everything else can
   decay and be rebuilt from it.)
