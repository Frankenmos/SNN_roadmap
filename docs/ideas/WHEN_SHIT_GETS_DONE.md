# When Shit Gets Done

The third-branch plan. After the SNN monstrosity has a baseline and the
BPTT attempt has a verdict, this is where we stop fighting biology and
just try to win the game.

This document treats the current repo as the starting point and
intentionally ignores the neuromorphic commitment. The idea space
(tokenized proprioception, entity-aware observation, CNN+tokens hybrid,
action history as input) stays. The spiking substrate doesn't.

---

## 1. Why this branch exists

We will have, by the time this branch starts:

- **Branch A** — vanilla PPO + SNN Spikformer + surrogate gradients.
  The "does the monstrosity work at all?" baseline.
- **Branch B (current)** — same as A plus GPT's truncated BPTT attempt
  from `THE_BPTT.md`. The "does neuromorphic recurrence trainably carry
  information?" answer.
- **Branch C (this doc)** — drop the SNN. Use a pragmatic dense model
  that keeps the obs/action ideas but swaps the substrate for whatever
  the literature says learns fastest and debugs easiest.

The neuromorphic angle is preserved on A and B. This branch exists to
give us a shot at actually beating DefeatRoaches (and later maps)
without the surrogate-gradient drag, and a clean vehicle to steal
whatever modern arch papers give us.

---

## 2. What we keep

Everything that is architecture-agnostic in Fix 3 stays:

- `PolicyInputBatch` protocol — shape-locked, does not care what
  consumes it. Good.
- Spatial CNN branch on `feature_screen` (27 channels → 49 tokens).
- Entity tokens: `feature_units` → per-unit MLP with `unit_type`
  embedding → pad to 24 + mask.
- Selection tokens: `multi_select` → 20 slots + mask.
- Meta token: `player[11]` + available-actions binary mask + last-action
  embedding with no-action sentinel.
- Token-type embedding (4 groups: spatial / entity / selection / meta).
- Reward function V2 (after the rebalance noted in the instability
  correlation pass — score >> engagement coefficient flip).
- All `analysis_results/` tooling. The DB schema
  (`ppo_updates`, `eval_runs`, `reward_components`, including the Fix 3
  mask-utilization columns) already speaks everything this branch will
  emit.

---

## 3. What we drop

- `TokenTemporalSNN` and its `snn.Synaptic(syn, mem)` state.
- `SpikingSelfAttention` and all spike-driven Q/K/V.
- Surrogate-gradient plumbing (`spike_grad`, `learn_alpha/beta`,
  `nonfinite_grad_steps` tracking — the last one stops being relevant
  when gradients are dense).
- The `num_steps` inner spike-accumulation loop. One policy call per env
  step.

---

## 4. Candidate architectures

Ranked by my bet on "will learn DefeatRoaches to >200 deterministic eval
reward fastest and with fewest surprises."

### Option 1 — Modern AlphaStar-lite (recommended)

- **Attention block**: 2 layers × 2 heads × 128 embed dim, standard
  dense multi-head attention over the full 94-token sequence. Same
  shape AlphaStar used for its entity transformer.
- **Recurrent core**: `nn.GRU` on the pooled token representation.
  State shape `[num_layers, B, D]` — simpler than SNN's
  `[B, N, D]` per-token state, and GRU beats LSTM on param count for
  identical behaviour at this scale.
- **Policy head**: autoregressive. Action type → then conditional on
  type, emit move_x/move_y or pointer-over-entity-tokens. AlphaStar's
  pointer-network trick is the missing piece for "click this specific
  roach" action grounding — we don't have it yet, and adding it is the
  single biggest expected action-side lift.
- **Action history**: ring buffer of last K=8 actions tokenized and
  added as a fifth token group. K=8 because kiting dependency is ~12-15
  steps but action history only needs to cover recency, not full
  horizon.

**Pros**:
- Closest thing to a proven SC2 baseline — AlphaStar reached
  grandmaster level on this exact spine.
- TBPTT integration is trivial once Branch B lands it — GRU hidden
  state is a single tensor, no mask-per-slot subtlety.
- Reference implementations exist for every component
  ([MarcoMeter/recurrent-ppo-truncated-bptt][tbptt],
  [CleanRL LSTM PPO][cleanrl-ppo]).
- Every failure mode has documented debugging lore.

**Cons**:
- Not architecturally novel. The 5090 would yawn.
- GRU saturates on long-horizon maps. Fine for DefeatRoaches, will
  bottleneck later.

### Option 2 — Mamba-core

Same encoding and attention as Option 1. Replace the GRU with a
**Mamba-2 block** as the recurrent core.

- Selective state-space model, O(L) time+memory during training
  (parallel scan), O(1) per step during acting.
- `mamba-ssm` PyPI package exists, under active development.
- Drama paper (2024) showed Mamba as world model in model-based RL on
  Atari100k with only 7M params ([arxiv][drama]).
- Mamba-3 (Princeton, 2026) improves state tracking further
  ([blog][mamba3]).

**Pros vs Option 1**:
- Better long-range memory without BPTT depth blowing up compute.
- Parallel-scan training is actually faster than sequential GRU unroll.
- State is still a single fixed-size vector — debuggable like GRU.

**Cons vs Option 1**:
- Fewer RL reference implementations — Drama is model-based, Decision
  Mamba is offline. On-policy Mamba-PPO is still thin literature.
- The `mamba-ssm` CUDA kernel can be fragile on Windows + new PyTorch.
  Worth a compile-and-run check before committing.
- Every "why did learning stall" question has less existing lore.

### Option 3 — Decision Transformer-style in-context memory

No recurrent state. Feed last K timesteps as interleaved
`(state_tokens, action_tokens, reward_or_RTG_tokens)` with a causal
mask. Standard transformer with KV cache during acting.

**Pros**:
- Fully attention-based. No recurrent-state debugging ever.
- Attention weights directly visualize what the policy is "looking at"
  in its own history.
- 2025 UTR paper ([arxiv][utr]) cuts sequence length 3× via merged
  tokens. GPG-HT ([arxiv][gpght]) extends DT to on-policy.

**Cons**:
- Quadratic in K. K~16 to match GRU kiting range means 3K=48 tokens
  per step of history on top of the per-step tokens. Manageable, not
  free.
- On-policy DT is active research. PPO clip objective interacting
  with per-token loss is subtle — not a solved problem in 2026.
- More divergence from the existing codebase than Option 1.

### Option 4 — Transformer-XL-style segment recurrence

Dense attention per segment, pass hidden state across segments.
Technically the right architecture for very long episodes, but
DefeatRoaches episodes are 76 steps — segment recurrence is overkill.
Defer until we leave DefeatRoaches.

---

## 5. Rankings

**Learning performance (DefeatRoaches scale):**
1. Option 1 (Modern AlphaStar-lite)
2. Option 2 (Mamba-core) — probably equal, modest edge on longer maps
3. Option 3 (Decision Transformer) — could surprise us up or down
4. SNN+Spikformer+TBPTT (Branches A/B) — will learn, paying
   surrogate-gradient tax every step

**Debuggability (easiest first):**
1. Option 1 — GRU is in every textbook
2. Option 3 — attention heatmaps are self-documenting
3. Option 2 — Mamba is mostly clean; kernel/install friction is the risk
4. SNN+Spikformer — surrogate-gradient + sparse binary activations =
   silent failures are routine

**If only one branch gets built: Option 1.** It is what a serious
engineer would write to win, it gives every future option a clean
baseline to beat, and its protocol-compatibility with the other branches
means we can A/B the exact same env/reward/logging setup.

Option 2 is the obvious follow-up once Option 1 is learning — it's a
drop-in replacement for the GRU, same `state_in/state_out` contract,
same `PolicyInputBatch`.

---

## 6. Steal list (from what modern papers give us)

Things to copy with minimal shame:

- **AlphaStar**: 2×2-head entity transformer at 128 dim, LSTM/GRU core,
  auto-regressive policy head with pointer network over entity tokens.
  Pointer network is the piece we don't have yet and it matters.
- **Decision Transformer (2021)**: action history as explicit token
  stream. Drops into Option 1 as the fifth token group.
- **Drama / Decision Mamba (2024-2026)**: Mamba as recurrent core.
  Option 2 upgrade path.
- **Mamba-3 (2026)**: relevant for longer-horizon maps.
- **UTR (2025)**: token merging to cut sequence length — relevant if
  Option 3 happens.
- **Perceiver IO pattern**: cross-attention from variable entity set
  into fixed latent. Natural fit if entity counts explode on later maps
  (`N_max=24` will not hold for bigger minigames).

---

## 7. Reference implementations worth cribbing

- [MarcoMeter/recurrent-ppo-truncated-bptt][tbptt] — the canonical
  PyTorch TBPTT+PPO reference. Same TBPTT plumbing works for GRU,
  LSTM, and with minor adaptation, Mamba. Use it as the skeleton once
  Branch B's TBPTT is verified.
- [CleanRL PPO docs][cleanrl-ppo] — LSTM variant is a single-file
  recurrent PPO, easy to diff against.
- [Drama paper][drama] — Mamba integration patterns.
- [AlphaStar Nature paper][alphastar-nature] +
  [architecture decipher][alphastar-decipher] — architecture reference
  for the entity transformer sizing, pointer network, AR policy head.
- [Decision Transformer][dt-arxiv] — the sequence-modeling formulation
  if Option 3 happens.

---

## 8. Migration sketch (Option 1)

1. New branch off `master` after Branch A/B converge. Keep
   `PolicyInputBatch` unchanged — it's already architecture-agnostic.
2. Replace `SpikingSelfAttention` with stacked `nn.MultiheadAttention`
   (2 layers, 2 heads, 128 dim) + feed-forward + LayerNorm. Keep the
   token-type embedding. Keep the mask.
3. Replace `TokenTemporalSNN` with `nn.GRU(input=D, hidden=D,
   num_layers=1)`. State becomes `[1, B, D]`. `state_in/state_out`
   contract on `PolicyInputBatch` stays identical.
4. Add `ActionHistoryEncoder`: last K=8 actions → 3 parallel
   embeddings (action_id, move_x_bucket, move_y_bucket) → sum → token
   group 5 of length 8.
5. Add pointer-network policy head: action type logits, then
   conditional on type emit either move_{x,y} logits OR pointer
   attention over entity tokens. Skip this in phase 1 if time-pressed
   — current discrete action space still works.
6. Delete `num_steps` inner spike loop. One forward per env step.
7. Port Branch B's TBPTT plumbing once it's verified. GRU is strictly
   easier than per-token SNN state so this port is downhill.
8. Run against Branch A baseline on the same DB. Same
   `analysis_results/` tooling, same reward, same env.

**Expected first-week signal**: >200 deterministic eval reward within
500 episodes on DefeatRoaches. If it doesn't hit that, it's a
structural bug not a research problem — Option 1 is fighting
hyperparameters, not biology.

---

## 9. What this branch gets for free

- `analysis_results/` tooling works as-is. DB schema already carries
  everything this branch emits (the Fix 3 mask-util columns stay
  meaningful — entity/selection tokens are unchanged).
- Eval harness works as-is.
- Reward function V2 (post-rebalance) works as-is.
- Env wrapper works as-is.
- Every Fix-3-era `test_policy_input.py` / mock-env test carries over
  — the protocol is the same.

---

## 10. The anti-discipline clause

This branch is explicitly for experimentation. It's allowed to:

- YOLO hyperparameters. No Socratic walkthrough per knob.
- Try two or three arches in parallel (Option 1 and Option 2 from the
  same `PolicyInputBatch` contract is cheap).
- Borrow code from reference implementations with minimal adaptation.
- Skip the "protocol-lock-before-code" discipline that Fix 1-3 got.

It is NOT allowed to:

- Delete the reward-function V2 rebalance.
- Mutate `PolicyInputBatch`. Keep protocol stable across all three
  branches so we can A/B cleanly.
- Skip writing down what worked and what didn't — cheap notes in a
  `branch_C_log.md`, not thesis-level write-ups.
- Merge back to `master` without beating Branch A's best eval reward
  on the same seed.

---

## 11. Success criteria

This branch wins if:

- Deterministic eval reward on DefeatRoaches exceeds Branch A's best
  by a clear margin (≥30%) at equal wall-clock.
- Training curve has smaller CoV than Branch A at the same point.
- Det/stoch gap is ≤2× (same bar Fix 3 set for the SNN branches).

Failure mode to watch: if Option 1 lifts to ~200 reward and then
plateaus without beating Branch A's ceiling, the bottleneck moved to
the reward shaping or action space, not the architecture. Sanity-check
before blaming the model.

---

## References

[tbptt]: https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt
[cleanrl-ppo]: https://docs.cleanrl.dev/rl-algorithms/ppo/
[drama]: https://arxiv.org/abs/2410.08893
[mamba3]: https://pli.princeton.edu/blog/2026/mamba-3-improved-sequence-modeling-using-state-space-principles
[alphastar-nature]: https://www.nature.com/articles/s41586-019-1724-z
[alphastar-decipher]: https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/
[dt-arxiv]: https://arxiv.org/abs/2106.01345
[utr]: https://arxiv.org/abs/2510.21448
[gpght]: https://arxiv.org/html/2508.17218

- [MarcoMeter recurrent-ppo-truncated-bptt][tbptt] — canonical PyTorch
  TBPTT+PPO reference.
- [CleanRL PPO][cleanrl-ppo] — LSTM recurrent PPO single-file.
- [Drama][drama] — Mamba-based model-based RL (2024).
- [Mamba-3][mamba3] — improved state-space modeling (Princeton, 2026).
- [AlphaStar Nature][alphastar-nature] + [decipher][alphastar-decipher]
  — entity transformer + LSTM + AR policy head at scale.
- [Decision Transformer][dt-arxiv] — sequence-modeling RL formulation.
- [UTR][utr] — Unified Token Representations for DT (2025).
- [GPG-HT][gpght] — history-aware on-policy Decision Transformer (2025).

---

# Part 2 — Cross-Domain Steal List from LLM / Transformer Land

You asked: does it make sense to steal LLM techniques for this RL
project, given that we're not training an LLM? Yes, and here's the
honest ranking. Transformers and sequence modeling are the same
mathematical substrate whether the target is a token or an action.
LLM research has thrown orders of magnitude more GPU-hours at the
building blocks we're using — their tricks translate directly when the
information flow is similar, and cleanly don't translate when it isn't.

Each technique below is graded ★ to ★★★★★ on expected payoff for this
specific project, with a "when to steal" note.

## 12. The steal shortlist

### 12.1 LoRA / Low-Rank Adapters — ★★★★★

**What it is**: train a low-rank delta `ΔW = BA` (rank `r ≪ D`) on top
of frozen base weights. Param count drops from `D²` to `2rD` — ~1% of
original at r=8, matching full fine-tune performance on many tasks.

**Why it matters for us**: the single most surprising 2025 finding —
[Thinking Machines Lab's "LoRA Without Regret"][lora-without-regret]
showed **rank-1 LoRA fully matches full fine-tuning on policy gradient
RL**. The reason: PPO's reward signal is so sparse (one scalar per
trajectory) that even a rank-1 adapter absorbs the gradient. Rule of
thumb: LoRA learning rate = 10× the full fine-tune LR.

**Direct use cases**:
- **Per-map specialization**: train base policy on DefeatRoaches,
  LoRA-finetune per-map without catastrophic forgetting. Stable model
  zoo, not a single fragile checkpoint.
- **Multi-head exploration**: swap LoRA deltas for different strategies
  (aggressive / conservative / focus-fire) at eval time.
- **Ablation velocity**: test architectural variants as LoRA deltas
  instead of full retrains. Cheap A/Bs.

**Not useful for**: single-task from-scratch training. LoRA shines on
fine-tune, not initial learning. This is a Stage 2+ tool.

**Cost**: ~50 LOC to wrap `nn.Linear`. [verl has a PPO+LoRA reference
implementation][verl-lora].

### 12.2 Mixture of Experts (MoE) — ★★★★

**What it is**: `K` expert MLPs, a router picks top-`k` per input
(usually k=2). Only active experts compute — more capacity at constant
FLOPs.

**Why it matters for us**: [PA-MoE (Feb 2026)][pa-moe] explicitly
diagnosed that "most RL agents use a single policy network throughout
an episode, causing simplicity bias." PA-MoE + GiGPO achieved +7.7pp
over baseline by letting different experts activate for different
phases.

For DefeatRoaches the phase structure is real and clean:
- Open (no enemies in screen)
- Engagement (pick target)
- Kite (cooldown winding up, retreat)
- Cleanup (focus-fire low-HP)

4 experts, top-2 gating, routing on pooled token vector. The router
would learn to activate different combinations per phase. This is
exactly what our current policy *cannot* do — a single dense MLP is
forced to be competent-at-everything-mediocrely.

**Stability concern**: RL + MoE training is [actively being
stabilized][rspo] — RSPO (Oct 2025) introduced sequence-level
importance sampling to handle router shifts during off-policy updates.
Known footgun. On-policy PPO + MoE is cleaner than off-policy but still
not a solved problem.

**Cost**: ~200 LOC for a small MoE FFN block. Slots in after the last
attention layer, before the GRU.

### 12.3 Chain-of-Thought / Latent Scratchpad — ★★★

**What it is**: before emitting the final output, spend N "thinking"
tokens in latent space the model can attend to. In LLMs these are
observable text; in RL they're latent with no direct supervision.

**Why it matters for us**: high-entropy "forking tokens" were shown to
drive effective RL in LLM reasoning — only ~20% of tokens actually
steer the decision ([NeurIPS 2025 Poster][forking-tokens]). For our
policy: add 4-8 learnable scratchpad tokens at the start of the token
sequence, include them in attention layers, strip them before the
action head. PPO discovers what's useful to compute there. No auxiliary
loss needed. [Satori (Feb 2025)][satori] (chain-of-action-thought RL)
and [Latent CoT for Visual Reasoning (Oct 2025)][latent-cot] both
verified the pattern generalizes beyond text.

**Direct use**: concat `N=4` learnable scratchpad embeddings to the
94-token sequence. Attention block gets `[B, 98, D]`. Strip before
policy head. Zero extra parameters (just `nn.Parameter(4, D)`). Adds
~4% attention compute.

**Concern**: interpretability is zero. Unlike LLM CoT where you can
*read* the scratchpad, ours is a black box. If the policy plateaus with
scratchpad tokens active, you can't distinguish "scratchpad is helping
subtly" from "scratchpad is doing nothing." Debug-hostile.

### 12.4 FlashAttention / SDPA — ★★★★★ (free win)

**What it is**: fused-kernel attention, 2-4× faster than naive, memory
linear in seq-len instead of quadratic.

**Why it matters for us**: literally free. Use
`torch.nn.functional.scaled_dot_product_attention` instead of manual
`Q @ K.T / sqrt(D)`. PyTorch 2.1+ picks FlashAttention 2 automatically
on supported hardware. RTX 5090 supports it. Zero arch change, zero
debugging cost, 2-3× speedup.

**Cost**: one-line swap per attention block. This should be in *every*
branch (including Branches A and B, not just C).

### 12.5 RoPE on the action-history token group — ★★★

**What it is**: rotary positional encoding. Rotates Q/K in 2D feature
subspaces by angles that depend on position; the resulting attention
scores depend on *relative* position.

**Why it matters for us**: exactly one of our token groups has
meaningful ordering — action history. Entity tokens are slot-indexed
(arbitrary order). Selection tokens same. Meta is single-token. Spatial
tokens have 2D position but the CNN already encoded that. So apply
RoPE selectively, only on action-history tokens, not globally.

**Cost**: ~30 LOC. Mask which token groups get rotation in the
attention block.

### 12.6 Retrieval-Augmented memory — ★★ (deferred)

**What it is**: external memory of past `(obs, action, outcome)`
triples. At act time, retrieve top-k similar past experiences and
cross-attend.

**Why it matters**: [RA-DT (Oct 2024)][ra-dt] showed episodic retrieval
helps on long-episode sparse-reward tasks. [MemRL (2026)][memrl] learns
*which* past experiences to retrieve via Q-values instead of cosine
similarity.

**Why not now**: DefeatRoaches has ~76-step episodes with dense reward.
Retrieval shines in the opposite regime — long horizons, sparse signal.
Correct tool, wrong problem. Revisit when we leave DefeatRoaches.

## 13. The "not yet" list

| Technique | Why skip for us | When to revisit |
|---|---|---|
| **Speculative decoding** | One action per env step, nothing to speculate | Never for this project |
| **ROME / MEMIT (weight-level fact editing)** | Translates thin to RL; "inject expert demo as weight edit" is research, not engineering | Deep side-quest |
| **Ring attention / long-context tricks** | Our sequence is 94 tokens + 8 history. Not long | Only if Option 3 (DT) with K ≫ 32 |
| **GQA / MLA (KV-cache compression)** | Our model is tiny; KV cache is already small | Only at scale >100M params |
| **RLHF / Constitutional AI** | We have a real numeric reward, not preferences | Never for this game |
| **Quantization-aware training** | 5090 has 32GB; not memory-bound | Only if deploying to edge hardware |
| **Parallel / tree sampling** | LLM-inference-specific | Never for policy acting |

## 14. Integration order for Branch C

**Stage 1 — first week, minimum to validate arch choice:**
- Option 1 from Part 1 (AlphaStar-lite: dense transformer + GRU + AR
  policy head)
- **§12.4 FlashAttention / SDPA** — no reason not to
- **§12.5 RoPE on action-history tokens** — if action history is in
  Stage 1

**Stage 2 — after Stage 1 beats Branch A on eval reward:**
- **§12.2 MoE FFN** — 4 experts, top-2 gating, routing on pooled
  vector. Addresses phase-bias.
- **Pointer network policy head** (from Part 1) — not an LLM-steal but
  the single biggest action-side lift we haven't done.

**Stage 3 — cross-map generalization (deferred, maybe months out):**
- **§12.1 LoRA adapters** per map, base policy frozen.
- **§12.3 Latent scratchpad tokens** for maps where reaction chains
  matter more than DefeatRoaches.

**Stage 4 — long-horizon maps, if we ever leave DefeatRoaches:**
- **§12.6 RA-DT style external memory**.
- **Mamba-core upgrade** from Part 1 Option 2.

## 15. The diminishing returns reality check

Honest take: the difference between Option 1 (GRU + dense transformer)
and a hypothetical "everything from Part 2 enabled" arch on
DefeatRoaches is probably close to 0. The minigame doesn't require
that much capacity. Most of these LLM-steals earn their keep on:

- Harder minigames (multi-base, full SC2)
- Multi-task generalization across maps
- Longer episodes with delayed reward
- Transfer learning from pretraining

For **beating DefeatRoaches specifically**, I expect the impact
ranking to be:

- ★★★★★ Fix 3 (observation tokenization) — already done
- ★★★★★ Reward rebalance (kill reward >> engagement coefficient, per
  the reward-shaping correlation audit)
- ★★★★ Pointer network action head — the single biggest action-side
  lift
- ★★★ GRU + dense transformer swap (Part 1 Option 1)
- ★★ Everything else in Part 2 of this doc

So the correct framing for Branch C:
- **Part 1 is the "beat the game" spine.**
- **Part 2 is the "what to reach for when the game doesn't stop us but
  the next map does" upgrade menu.**

Build Part 1, verify it wins, *then* cherry-pick from Part 2 based on
where the failure mode actually lands.

## 16. Part 2 references

- [LoRA Without Regret — Thinking Machines Lab][lora-without-regret]
  — rank-1 LoRA matches full fine-tune on policy gradient RL.
- [verl: PPO+LoRA reference implementation][verl-lora]
- [PA-MoE / Phase-Aware MoE for Agentic RL (Feb 2026)][pa-moe]
- [RSPO: Stable RL for MoE (Oct 2025)][rspo]
- [Satori: RL with Chain-of-Action-Thought (Feb 2025)][satori]
- [Latent CoT for Visual Reasoning (Oct 2025)][latent-cot]
- [RA-DT: Retrieval-Augmented Decision Transformer (Oct 2024)][ra-dt]
- [MemRL: Self-Evolving Agents (2026)][memrl]
- [Forking Tokens — NeurIPS 2025 Poster][forking-tokens]

[lora-without-regret]: https://thinkingmachines.ai/blog/lora/
[verl-lora]: https://verl.readthedocs.io/en/latest/advance/ppo_lora.html
[pa-moe]: https://arxiv.org/html/2602.17038
[rspo]: https://arxiv.org/abs/2510.23027
[satori]: https://arxiv.org/pdf/2502.02508
[latent-cot]: https://arxiv.org/abs/2510.23925
[ra-dt]: https://arxiv.org/abs/2410.07071
[memrl]: https://arxiv.org/html/2601.03192v1
[forking-tokens]: https://neurips.cc/virtual/2025/loc/san-diego/poster/115123

---

## 17. Note on your specific ask

You mentioned "ROME ALE, Mamba, CoT, MoE." I covered:
- **Mamba** — Part 1 §4 Option 2, fully addressed
- **CoT** — Part 2 §12.3
- **MoE** — Part 2 §12.2
- **ROME** — Part 2 §13 not-yet table; translation to RL is thin and
  research-heavy, not engineering-ready
- **ALE** — interpreted as possibly **LoRA** (phonetic mangling) or
  **RoPE** (typo). Both covered (§12.1 and §12.5). If you meant something
  else, ping me with the spelled-out name and I'll add it.

Other things you might have meant but didn't name explicitly, which I
included because they're obvious wins or obvious skips:
- **FlashAttention** — free speed win (§12.4)
- **RAG / retrieval-augmented** — deferred but explained (§12.6)
- **Everything in the not-yet table** (§13) — explicitly excluded with
  reasoning so you don't have to re-ask

