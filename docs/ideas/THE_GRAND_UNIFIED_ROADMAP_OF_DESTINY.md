# The Grand Unified Roadmap of Destiny (and Also Some Bugs)

**Date:** 2026-05-10
**Status:** Proposed
**Mood:** Cautiously optimistic with mild espresso jitters

---

## Executive Summary

You've got an SNN+PPO DefeatRoaches agent that:
- ✅ Has all the fancy architectural bells and whistles
- ✅ Passes 108 tests
- ✅ Successfully trained for ~5,000 episodes
- ❌ Tragically collapses into no-op/passive behavior

This document proposes a structured path forward that prioritizes **reward sanity**, **env validation**, and **training stability** before adding more complexity.

---

## The Current Situation

### What Works (Architecture)
- SNN spiking policy with dual-timescale token memory
- Semantic action space: `NO_OP`, `LEFT_CLICK`, `RIGHT_CLICK` → `Smart_screen(x, y)`
- Coarse-to-fine spatial head (7056 positions)
- Stage-1 TBPTT with 128-step window
- Fragment-based PPO with per-fragment GAE
- Action Effect Feedback v2 (12-dim stream tokens)
- Initial Ray distributed rollout path

### What Doesn't (Behavior)
- The agent learns to passively watch its marines die
- No-op dominates the action mix despite available actions
- Reward signal is not effectively shaping the desired behavior

### The Root Cause Hypothesis
The reward function (`defeat_roaches_v4` is ready but unvalidated) and action-space semantics are fighting each other. The agent finds a local optimum: "do nothing and get tiny step penalties forever" rather than "click aggressively and get big uncertain rewards."

---

## The Roadmap

### Phase 1: Validation & Diagnostics (Week 1) - UPDATED 2026-05-10

**CRITICAL FINDING:** V5 regressed from V4 despite same reward function.

| Run | Protocol | Max Reward | Final-100 Avg | Issue |
|-----|----------|------------|---------------|-------|
| V4 | 9-dim feedback | 211.93 | -2.51 | Plateau/instability |
| V5 | 12-dim action-effect | **0.00** | **-49.76** | **Complete failure + non-finite gradients** |

**Root Cause (user observation):** Agent hugging bottom-left corner because:
- Smart_screen turns into attack when enemies in range (but API doesn't expose this clearly)
- Agent learns "click away from enemies" → corner hugging
- Action-effect feedback (bits 8-11) either adds noise or doesn't help

**New Direction:**

| Task | Method | Success Criterion |
|------|--------|-------------------|
| Revert to V4 protocol | Test 9-dim vs 12-dim feedback side-by-side | 9-dim should match V4 results |
| Attack detection | Build explicit "did this click become an attack?" detector | Binary flag in feedback |
| Click quality metrics | Track distance to nearest enemy per click | Mean distance < 10 pixels |

**Deliverable:** `docs/current/validation_report_v4.md` with:
- Action mix breakdown (early/mid/late episode)
- Click quality heatmap
- Reward component analysis
- "Why it's still broken" section

### Phase 2: Reward Triage (Week 1-2)

**Goal:** Make the reward function actually incentivize winning.

| Issue | Proposed Fix | Priority |
|-------|--------------|----------|
| No-op dominates | Increase `noop_visible_enemy_penalty` to 0.1 | HIGH |
| Smart doesn't click enemies | Increase `smart_near_enemy_reward` to 0.2 | HIGH |
| Terminal outcome weak | Double `win_reward` and `loss_penalty` | HIGH |
| Step penalty too high | Reduce `step_penalty` from 0.005 to 0.001 | MEDIUM |
| Distance band ambiguous | Remove `distance_reward_clip` | MEDIUM |

**Alternative Approach if V4 Fails:**
- Implement sparse reward-only: +100 for win, -50 for loss, nothing else
- Use curriculum: start with 1 roach, scale up to full game

### Phase 3: Training Stability (Week 2-3)

**Goal:** Make training not collapse after episode 2000.

| Technique | Status | Next Step |
|-----------|--------|-----------|
| TBPTT | Implemented | Tune window: try 64, 128, 256 |
| Learning rate schedule | Config only | Implement warmup + cosine decay |
| Gradient clipping | Not implemented | Add max_norm=0.5 |
| Value function clipping | Not implemented | Clip returns during GAE |
| Entropy schedule | Config only | Start high (0.02), decay to 0.005 |

**Critical Experiment:**
Run an A/B test with:
- A: Current settings
- B: Higher LR (1e-4), more aggressive entropy, clipped values

### Phase 4: Distributed Scaling (Week 3-4)

**Goal:** Get the Ray path actually working at scale.

| Task | Command | Success Criterion |
|------|---------|-------------------|
| 1-actor smoke | `python -m distributed.ray_train --num-actors 1 --max-updates 5` | Completes without crash |
| 4-actor smoke | `python -m distributed.ray_train --num-actors 4 --max-updates 5` | 4x throughput of 1-actor |
| 10-actor full run | `python -m distributed.ray_train --num-actors 10 --max-updates 100` | Stable training |
| Throughput measurement | Check `tbptt_forward_calls` in logs | >500 steps/sec |

**Known Issues:**
- Extractor normalizers are actor-local after initial sync
- Windows SC2 temp-map races (partially mitigated)
- EvalActor not implemented

### Phase 5: The Fork (Week 4+)

**Decision Point:** Based on results from Phases 1-4, choose a path:

#### Option A: Double Down on SNN
If training shows promise:
- Implement ALIF neurons for adaptive thresholds
- Add reward-driven neuromodulation
- Extend action history to K=8 token group
- Add entity identity via `raw_units.tag`

#### Option B: Pragmatic Dense Branch
If SNN continues to struggle:
- Follow `WHEN_SHIT_GETS_DONE.md` plan
- Swap SNN for GRU/Mamba core
- Keep observation tokenization (it's good!)
- Add AlphaStar-style pointer network for entity selection

#### Option C: Curriculum & Offline Pretraining
If reward engineering keeps failing:
- Build move-to-beacon curriculum
- Relabel old trajectories with SMART semantics
- Behavior cloning warm start → PPO fine-tune

---

## Immediate Next Steps (In Order)

1. **TONIGHT:** Start `banana_smart_v4_b2048_e4_a10` training run
   - Checkpoint every 100 episodes
   - Log action mix every episode
   - Stop if no Smart clicks by episode 200

2. **TOMORROW:** Run full diagnostic eval on `banana_b2048_e4_a10` best checkpoint
   ```bash
   python eval.py --run_name banana_b2048_e4_a10 --best --episodes 10 \
       --inspect --inspect_policy_input --inspect_actions \
       --inspect_last_action --inspect_score
   ```

3. **THIS WEEK:** Read through diagnostic JSONLs and write validation report
   - Are clicks actually landing near enemies?
   - Is the coarse-to-fine head working as intended?
   - What's the actual episode-phase action mix?

4. **NEXT WEEK:** Based on findings, either:
   - Tweak V4 reward coefficients and re-run
   - Pivot to sparse reward + curriculum
   - Start implementing training stability improvements

---

## Known Technical Debt

| Area | Issue | Impact | Fix Priority |
|------|-------|--------|--------------|
| Reward | V4 unvalidated in live env | Training may be mis-specified | HIGH |
| Spatial head | Coarse-to-fine never env-verified | Clicks may be nonsense | HIGH |
| Distributed | Ray smoke only done at small scale | Production rollout risky | MEDIUM |
| Testing | No integration tests with real env | Regressions possible | MEDIUM |
| Logging | Step-level logs Ray-incompatible | Can't debug distributed runs | LOW |
| Checkpointing | No explicit global step count | Resume may be off by N steps | LOW |

---

## Success Metrics

### Minimum Viable Victory (MVV)
- Agent wins >30% of episodes deterministically
- Smart clicks constitute >40% of actions
- Mean reward >5.0
- Training stable for 5000+ episodes without collapse

### Stretch Goals
- Win rate >50% deterministically
- Agent kites (retreats while firing)
- Training stable for 20000+ episodes
- Throughput >1000 steps/sec with Ray

---

## Appendix: Open Questions for User

1. **Reward semantics:** Do you trust `defeat_roaches_v4`'s action-aware bonuses, or should we go sparse-only?

2. **Spatial head:** Is `coarse_to_fine` worth keeping, or should we fall back to `token_pointer` for simplicity?

3. **Training horizon:** Should `steps_per_episode` remain at 3600, or is that incentivizing passivity?

4. **Branch point:** At what point do we abandon the SNN branch and switch to `WHEN_SHIT_GETS_DONE`?

5. **Compute budget:** How many overnight runs are you willing to burn on reward tuning before pivoting?

---

## Resources

- Current run: `models/banana_b2048_e4_a10/checkpoint.pth`
- V4 config: Ready in `config.yaml` (set `environment.run_name`)
- Analysis tools: `results.py`, `dashboard.py`, `analyze_eval_trace.py`
- Smoke commands: See `docs/tooling/TEST_SNIPPETS.md`

---

**Remember:** The goal is to beat DefeatRoaches, not to build the prettiest SNN. Sometimes the ugliest GRU that actually learns is better than the most elegant spiking architecture that doesn't.

*Now go forth and make those marines click things.*
