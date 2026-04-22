# Claude Rapport — 2026-04-20

My independent read of the repo after the Stage-1 action refactor landed
and the BPTT-1 run started. This is the state as I see it, with things
I think are right, things I think are concerning, and a ranked
diagnosis of the deterministic-eval-0 signal.

Files I read to produce this:
- `PPO_CNN/policy_network.py`
- `PPO_CNN/policy_input.py`
- `PPO_CNN/PPO.py`
- `PPO_CNN_agent.py`
- `PPO_CNN_eval.py`
- `action_space/action_space.py`
- `obs_space/obs_space_2.py`
- `PPO_CNN/reward_function_2.py`
- `config.yaml`
- `analysis_results/BPTT-1/instability_report.txt`, `training_metrics.csv`, `training_progress.png`
- `docs/current/{REPO_STATE.md, action_refactor.md, working_log.md, THE_BPTT.md}`

---

## 1. Executive summary

**What's working**: the Stage-1 refactor landed cleanly. Conditional
action factorization is correct end-to-end (select_action, PPO loss,
TBPTT replay). Availability masking is applied in both rollout and
replay paths. TBPTT implementation is solid — chunked sequential
replay with per-row state resets on `done` and alive-masking for
variable-length chunks. Bridge token is normalized to [0,1] for x/y
and embedded by type. Training reward is **2× the pre-refactor run**
at comparable episode counts (150 vs 73 rolling 100-ep avg).

**What's concerning**: deterministic eval is still locked at 0.00 ±
0.00 across 5 episodes at ep 950. I've ruled out the bootstrap-in-eval
hypothesis — the bootstrap fires correctly in both modes. The evidence
now points to **argmax-locked-on-NO_OP**: the action distribution's
mode is NO_OP at ~40% (visible in the action-mix stackplot), and pure
argmax picks it every step → zero reward. Stochastic sampling distributes
across all three actions and scores 150 on average.

**Root cause**: a combination of NO_OP being locally safe (net reward
~0) and ATTACK being learnable-coords (§Q11 option B of the refactor
plan) — ATTACK starts with random spatial targeting, so ATTACK is
worse than random early in training, pushing the policy toward NO_OP
as the "don't be wrong" default. Reward shaping exacerbates this:
engagement coef (0.6 × damage dealt) is continuous and easy to
accumulate, kill coef (+10 × kills) is sparse, and health penalty
(-0.4 × damage taken) makes ATTACK-while-bad expensive.

**What to do next**: three cheap diagnostics before changing any
hyperparameter. Details in §6.

---

## 2. Architecture state — what's correct

### 2.1 Conditional action factorization

Verified end-to-end:

- [PPO.py:112-192](PPO_CNN/PPO.py#L112-L192) `select_action`: two-pass
  forward. `encode_step_tensors` → latent → `action_head` → mask logits
  → sample `a` → `conditioned_spatial_head(latent, a)` → sample `x, y`
  if spatial.
- [PPO.py:165-183](PPO_CNN/PPO.py#L165-L183) `is_spatial` gates the
  spatial log-prob contribution. `move_x, move_y` are zeroed for
  non-spatial actions.
- [PPO.py:1117-1202](PPO_CNN/PPO.py#L1117-L1202) `_calculate_losses`:
  same factorization in the update pass, using the stored `action` to
  drive `is_spatial`. PPO ratio is consistent between rollout and
  update.
- Entropy is normalized per-head via `1/log(dim)` — action head,
  spatial heads. Spatial entropy gated by `is_spatial`. Solves the
  pre-refactor entropy-imbalance concern cleanly.

### 2.2 Availability masking

- [PPO.py:86-110](PPO_CNN/PPO.py#L86-L110) `_policy_action_availability`
  reads the 16-slot PySC2 availability from `meta_vec`, maps to the
  3-slot policy vocab via `MOVE_AVAILABLE_ACTION_INDEX` and
  `ATTACK_AVAILABLE_ACTION_INDEX` constants from `policy_input.py`.
  NO_OP is always available.
- Mask applied to `action_logits` in both `select_action`
  ([PPO.py:135](PPO_CNN/PPO.py#L135)) and `_forward_replay_step_tensors`
  ([PPO.py:617](PPO_CNN/PPO.py#L617)). Symmetric. Good.

### 2.3 Bridge token plumbing

- [policy_input.py:76-85](PPO_CNN/policy_input.py#L76-L85) named offsets:
  `META_PLAYER_FEATURE_OFFSET`, `META_AVAILABLE_ACTION_OFFSET`,
  `META_LAST_ACTION_INDEX_OFFSET`, `AGENT_LAST_ACTION_OFFSET`. No
  negative indexing anywhere.
- [policy_input.py:94-98](PPO_CNN/policy_input.py#L94-L98) bridge
  vocabulary is 4-wide (NO_OP, MOVE, ATTACK, BOOTSTRAP_SELECT), one
  slot wider than the 3-way policy vocab. Clean separation between
  what the policy emits vs. what it observes.
- [obs_space_2.py:362-385](obs_space/obs_space_2.py#L362-L385)
  `_normalize_last_action_token`: x and y divided by `(SPATIAL_OBS_SHAPE[-1] - 1) = 83`,
  type ID kept raw as non-negative float, extra passed through. Matches
  the design from the external discussion.
- [policy_network.py:170-232](PPO_CNN/policy_network.py#L170-L232)
  MetaEncoder embeds the type ID via `bridge_action_embedding(4, D/8)`
  and concatenates with `(x, y, extra)` passthrough. Clean.

### 2.4 Reset bootstrap

- [PPO_CNN_agent.py:155-170](PPO_CNN_agent.py#L155-L170): fires when
  `bootstrap_pending AND can_select_army AND NOT (can_move OR can_attack)`.
  **Not gated on `deterministic`** — runs identically in training and
  eval. I verified this is the case by reading `PPO_CNN_eval.py` which
  calls `agent.step(obs, deterministic=...)` without any bootstrap
  override.
- `bootstrap_pending = True` reset in [agent.reset()](PPO_CNN_agent.py#L221)
  — fires once per episode.
- Bootstrap step returns `(action_func, None, 0, 0, state, 0.0, 0.0,
  None, False)` with `learnable=False` → not stored in PPO memory. Good.
- The bridge token for the bootstrap is `BRIDGE_ACTION_BOOTSTRAP_SELECT`
  (id 3), so the policy can observe "my previous action was a
  bootstrap select" via the bridge-action embedding.

### 2.5 TBPTT

- [PPO.py:532-575](PPO_CNN/PPO.py#L532-L575) `_build_tbptt_chunks`:
  splits rollout into windows of `tbptt_window=32` env steps, breaking
  early at `done`.
- [PPO.py:644-784](PPO_CNN/PPO.py#L644-L784) `_pack_chunk_group`: packs
  multiple chunks into a `[T, B_chunks, ...]` tensor with `alive_mask`
  for variable lengths.
- [PPO.py:786-931](PPO_CNN/PPO.py#L786-L931) `_replay_packed_chunk_group`:
  sequential forward per timestep with state carry. Per-row done reset
  via `_reset_replay_state_rows`. Alive-mask index selection so only
  active rows compute. This is a real TBPTT implementation, not just
  per-step replay.

### 2.6 Group masked-mean readout + unstable-slot zeroing

Confirmed still present from our earlier fixes:
- [policy_network.py:474-491](PPO_CNN/policy_network.py#L474-L491)
  `_zero_entity_state` zeros both entity AND selection slots between
  steps.
- [policy_network.py:538-560](PPO_CNN/policy_network.py#L538-L560)
  `_group_masked_mean` replaces the flatten+linear readout with
  4-group masked-mean concat → `shared_fc1(256→128)`.

---

## 3. Empirical findings from `BPTT-1`

At ep 996:

| Metric | Value | Baseline (fix3_hybrid_obs @ 96 eps) | Verdict |
|---|---|---|---|
| Rolling 100-ep reward | 150.34 | 73.45 | 2× improvement |
| Max episode reward | 605.60 | 385.20 | 1.6× |
| Avg episode length | 102.0 | 76.5 | Agent survives longer |
| Full-run reward std | 91.95 | 87.02 | Similar volatility |
| `entity_mask_util` | 0.404 | (not logged pre-refactor) | ✓ In §Anchor band [0.4, 0.6] |
| `selection_mask_util` | 0.317 | (not logged) | ✓ Reasonable |
| `entity_count_p50/p99` | 11.27 / 13.09 | (not logged) | ✓ Well under N_max=24 |
| Late-stage `mean_entropy` | 1.849 | — | Non-collapsed |
| Late-stage `clip_fraction` | 0.206 | — | Moderate, not extreme |
| Late-stage `grad_norm` (pre-clip) | 22.834 | — | **Aggressive pre-clip, 45× over clip=0.5** |
| `nonfinite_grad_steps` late | 1 | — | Rare but real |
| `explained_variance` | 0.773 | — | Critic is healthy |
| Late-stage `mean_kl` | 0.0040 | — | Under `target_kl=0.03` |
| Deterministic eval @ ep 950 | **0.00 ± 0.00 (n=5)** | 78.80 (ep 1700, n=1) | **Regression** |

Plateau detector flags plateau at ep 661 with late-stage CoV 0.72.
Reward curve is visibly climbing (training_progress.png shows 40 → 150
trend), so the plateau flag is a false positive from the detector's
variance-inflated threshold — not a real concern yet.

**Empirical action entropy** (from the stackplot at the bottom of
training_progress.png): starts ~1.05 (near log(3)=1.1), decays to ~0.6
by ep 1000. Stackplot action mix at end of training is roughly
`40% NO_OP / 25% MOVE / 35% ATTACK`. **NO_OP is the plurality mode,
so pure argmax picks it.**

---

## 4. Concerns flagged

### 4.1 Deterministic eval = 0 is a real issue

Highest-severity concern. The gap between stochastic training (150) and
deterministic eval (0) is **effectively infinite**, worse than the
pre-refactor run (661 vs 78.8 ≈ 8×). Diagnosis in §5.

### 4.2 Grad-norm pre-clip = 22.8 with clip at 0.5

[PPO.py:442](PPO_CNN/PPO.py#L442) clips to 0.5. Pre-clip norm averaging
22.8 over the last 11 updates means gradients are being scaled down by
~45× every step. That's aggressive. Possible causes:

- `reward_scale=0.1` in config + large advantages = PPO ratio surges.
- Surrogate-gradient noise at spike thresholds compounding through
  attention + temporal SNN.
- Clip set too tight relative to the actual gradient distribution.

Not a bug per se, but worth tracking. Drop clip to 1.0 if you want
less aggressive throttling; measure effect on stability.

### 4.3 Reward shaping imbalance still unfixed

Per-step averages from [training_metrics.csv](analysis_results/BPTT-1/training_metrics.csv):

| Component | Per step | Per episode (×102 steps) |
|---|---|---|
| health_reward (damage taken × -0.4) | -1.68 | -171 |
| engagement_reward (damage dealt × +0.6) | +2.56 | +261 |
| score_reward (kills × +10) | +0.16 | +16 |
| end_of_episode_reward | -0.19 | -0.2 (terminal only) |

**Engagement dominates score by 16×** over the episode. The policy's
dominant positive signal is "do continuous damage," not "kill roaches
fast." NO_OP is safer than ATTACK-while-targeting-poorly because it
avoids the health penalty while score bonus is rare enough to ignore.

**Relevant: `reward_scale=0.1` in [config.yaml:13](config.yaml#L13)**
scales everything down by 10× before GAE. This is standard PPO
stability practice but means the critic is learning against targets
~10× smaller than the raw reward components.

### 4.4 Terminal win detection bug

[reward_function_2.py:102-108](PPO_CNN/reward_function_2.py#L102-L108):

```python
if obs.last():
    if obs.reward > 0:  # "Win" condition
        ...
    else:  # "Lose" condition
        ...
```

`obs.reward` on the terminal step is the **last step's score delta**,
not a win/loss indicator. A roach kill on the terminal step gives
positive step-reward without actually winning. Misclassifies some
episodes as wins when they're not. Small effect (avg EoE is -0.19,
meaning most terminal steps yield the loss branch) but still wrong.

Proper win signal in DefeatRoaches: `obs.observation.score_cumulative`
or checking remaining enemy count == 0 at terminal.

### 4.5 Dead code in `action_space.py`

[action_space.py:66-90](action_space/action_space.py#L66-L90)
`nearest_enemy_unit_center` is defined but **never called** from the
agent. The agent uses raw `(move_x, move_y)` from the policy for
ATTACK. Keeping it for reference is fine, but it's misleading when
skimming the file — looks like it's in the loop.

### 4.6 §Q11 decision (ATTACK learnable-coords) has learning debt

The Stage-1 implementation chose option B of my §Q11 (ATTACK
learnable-spatial). This was a judgment call I'd recommended against
for Stage 1, precisely because it forces the move heads to learn
target selection from scratch. Pre-refactor, ATTACK got free scripted
targeting via `nearest_enemy_unit_center`.

The cost is visible in the BPTT-1 run: early ATTACK clicks at random
coordinates are worse than NO_OP, so the policy under-weights ATTACK.
Combined with §4.3 (kill reward being dominated by engagement reward),
the policy settles on NO_OP as the least-bad default.

Fix options if we want to revisit:
- Roll back to scripted ATTACK coords for Stage 1.5.
- Warm-start move heads by supervised pre-training on
  `nearest_enemy_unit_center` outputs.
- Scale kill_reward up (e.g., 10 → 30) so successful ATTACKs dominate
  engagement noise.

---

## 5. Diagnosis: why is deterministic eval = 0?

Ranked by what the evidence supports.

### Hypothesis #1 — Argmax locked on NO_OP ⭐⭐⭐⭐⭐

Evidence:
- Action stackplot shows ~40% NO_OP mode at ep 1000.
- Empirical entropy ~0.6 — committed but not collapsed.
- Argmax deterministically picks the mode → NO_OP every step → 0 reward.
- Stochastic sampling still hits MOVE/ATTACK ~60% of the time →
  training reward of 150.
- No prior eval regression of this exact kind before Stage-1 refactor.
  The old run's deterministic eval (78.8 @ ep 1700) had scripted
  ATTACK coords always aimed at enemies.

**Confidence: high.** This is the single cleanest explanation for the
stochastic/deterministic divergence. The bootstrap works in eval, the
architecture is correct, the factorization is correct. The policy
just learned that NO_OP is locally safer than ATTACK/MOVE with bad
coords.

### Hypothesis #2 — ATTACK coord-learning debt ⭐⭐⭐⭐

Evidence:
- Pre-refactor, ATTACK had scripted targeting → always aimed at an
  enemy. This reliably earned engagement + score rewards.
- Post-refactor, ATTACK uses `(move_x, move_y)` from the policy.
  Those heads start from random init and must learn to target from
  reward signal alone.
- Until the move heads learn targeting, ATTACK clicks mostly miss →
  ATTACK's expected reward is lower than the pre-refactor baseline.
- This feeds hypothesis #1: NO_OP becomes relatively more attractive
  than ATTACK-with-random-targeting.

**Confidence: high**, but this is a compounding factor, not an
independent cause.

### Hypothesis #3 — Reward shaping pushing toward safe defaults ⭐⭐⭐

Evidence: §4.3 breakdown — engagement × 0.6 dominates kill × 10
across an episode. Dying-while-attacking can still net negative from
the health penalty. NO_OP accrues neither reward nor penalty directly.

**Confidence: medium.** Contributes to #1 by making NO_OP locally
safer than exploratory action.

### Hypothesis #4 — Selection lost mid-episode, no recovery ⭐⭐

Evidence: [agent.py:155](PPO_CNN_agent.py#L155) bootstrap is
one-shot-per-episode. If selection is lost mid-episode, no
re-selection happens. Then MOVE/ATTACK become unavailable →
availability mask forces NO_OP for the rest.

In DefeatRoaches this is unlikely because marines don't lose
selection until all are dead (at which point the episode is ending
anyway). But possible edge case.

**Confidence: low for DefeatRoaches specifically**, higher for future
multi-base maps.

### Hypothesis #5 — TBPTT or replay bug ⭐

Evidence: none. Replay logic in PPO.py is internally consistent with
rollout logic. If it were buggy, training reward would also suffer.
Training reward doubled vs pre-refactor, so TBPTT is a net positive.

**Confidence: very low.** Not the issue.

---

## 6. Recommended next moves, ordered by cost/value

### 6.1 Diagnostic: log per-episode action mix in eval ⭐⭐⭐⭐⭐

~10 LOC in [PPO_CNN_eval.py:188-194](PPO_CNN_eval.py#L188-L194). Count
action IDs per eval episode, log at end.

Expected signal:
- **All NO_OP every step** → hypothesis #1 confirmed. Move to §6.2.
- **MOVE/ATTACK early, then NO_OP** → selection lost (#4). Look at
  bootstrap logic.
- **MOVE/ATTACK throughout but reward still 0** → coords never hit
  anything. Hypothesis #2 / different analysis.

This is the single cheapest signal. Don't touch anything else first.

### 6.2 Diagnostic: ε-greedy deterministic eval ⭐⭐⭐⭐

Add a `--eval_epsilon` flag to `PPO_CNN_eval.py` that, when > 0,
randomizes each argmax choice with probability ε. Run eval with
ε=0.05 and ε=0.10.

Expected signal:
- **Score lifts from 0 to non-trivial with small ε** → argmax-trap
  confirmed; policy is capable of scoring, just not deterministically.
- **Score stays 0** → deeper issue, hypothesis #1 is wrong.

Independent of #6.1. Both together give a clean diagnosis.

### 6.3 Cheap fix: entropy floor ⭐⭐⭐

If #6.1 + #6.2 confirm argmax-trap, raise `entropy_coef` from 0.01 to
0.03 and retrain. Makes the policy less committed to the mode. Also
consider entropy annealing (high early, decay late) — more effort.

### 6.4 Cheap fix: NO_OP step penalty ⭐⭐⭐

Add `-0.1` per step to the reward when the action is NO_OP. Discourages
the "do nothing because it's safe" solution. 3 LOC in
`reward_function_2.py`. Restart required.

### 6.5 Medium fix: reward rebalance ⭐⭐⭐

Bump `kill_reward` coefficient from 10 to 25-30. Or halve engagement
coefficient from 0.6 to 0.3. Goal: score-based reward dominates
engagement across an episode. Restart required.

### 6.6 Architectural fix: warm-start ATTACK coords ⭐⭐

If #6.1-#6.5 don't fully close the gap, consider this: during the
first N episodes (e.g. 500), the agent overrides the learned
(move_x, move_y) for ATTACK with `nearest_enemy_unit_center()`. Learn
the action-type head on scripted ATTACK targeting. After N episodes,
stop the override and let move heads take over. This gives the move
heads a target distribution to regress toward.

Alternative: supervised warm-start the move heads on
`nearest_enemy_unit_center` outputs before PPO training starts.

### 6.7 Fix terminal win detection ⭐

Low priority — small effect on reward signal — but should be corrected
when convenient. Use `obs.observation.score_cumulative` or check
remaining enemy count.

### 6.8 Delete dead `nearest_enemy_unit_center` or use it ⭐

House-keeping. Either remove from action_space.py, or bring it back
as §6.6's scripted warm-start.

---

## 7. Things I'd flag as "don't fix yet"

- **Don't widen `embed_dim` or scale the model up yet.** The failure
  mode is a policy-shape issue, not a capacity issue. More params
  won't fix argmax-trap.
- **Don't switch off spiking attention / replace with dense
  `nn.MultiheadAttention`.** That's the Branch-C direction from
  `WHEN_SHIT_GETS_DONE.md`, and it's the right call eventually, but
  not as a fix for this specific failure.
- **Don't rewrite TBPTT.** It's working.
- **Don't touch `tbptt_window=32`** yet. The current window corresponds
  to ~0.32 episodes at 102 step-avg-length — reasonable local credit
  assignment window.
- **Don't drop conditional factorization back to independent heads.**
  That was a bug; rolling it back would re-introduce spurious gradient.
- **Don't touch the reset bootstrap** unless #6.1 shows it's not
  firing in eval.

---

## 8. Open questions (what I can't answer from code alone)

1. **Is the eval `reward` cumulative game score or shaped reward?**
   [eval.py:192](PPO_CNN_eval.py#L192) uses `next_obs.reward` which is
   PySC2's game-side reward (score delta), not `RewardFunctionV2`'s
   shaped reward. So deterministic eval measuring 0.00 means zero
   game-score gain, which is worse than "shaped reward is negative"
   — the agent is literally not killing any roaches at all. This is
   consistent with hypothesis #1 (NO_OP every step) but makes it
   stark.

2. **How long has deterministic eval been flat at 0?** The eval_runs
   table has 19 rows logged. Running the analyzer to plot eval
   mean_reward over training would tell us whether det eval is
   gradually improving or stuck since ep ~50. If flat at 0 throughout,
   argmax-trap was there from the start. If trending up, just needs
   more training.

3. **What are the move_x/move_y heads actually outputting for
   ATTACK?** Could be confirmed by dumping a few eval trajectories
   with `--inspect_policy_input` and `--inspect_actions`. Are ATTACK
   coords clustered near enemies or uniform?

4. **Is `reward_scale=0.1` still the right choice?** Was likely set
   when total episode rewards were much bigger. With current shaping
   magnitudes, might be unnecessary or even harmful.

---

## 9. Summary: my opinion

**The architecture work landed well.** The conditional factorization,
bridge token, named offsets, TBPTT, availability masking are all
correct and well-structured. The repo is in its cleanest state since
Fix 3.

**The symptom (det eval = 0) is policy-shape, not infrastructure.**
The most likely cause is argmax-locked-on-NO_OP, driven by
§Q11-option-B (learnable ATTACK coords without warm-start) compounded
by reward shaping favoring "safe" over "lethal." The plumbing works;
the policy just hasn't learned to commit to spatial actions.

**Cheapest path to diagnosis: log eval action mix + try ε-greedy
eval.** Two short commits, one ~10 LOC each. Either confirms
hypothesis #1 and points at fixes, or rules it out and points
somewhere else.

**If confirmed, cheapest fix: raise `entropy_coef` or add NO_OP step
penalty.** Either breaks the argmax-trap without architecture changes.
Reward rebalance (kill coef ↑) and warm-start ATTACK (§6.6) are the
medium-term follow-ups if entropy + step penalty aren't enough.

**Don't scale the model, don't touch TBPTT, don't rewrite anything.**
The current code is solid. The learning outcome needs nudging, not
rebuilding.
