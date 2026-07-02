# Parameter Drift Analysis: Zero Run

**Date:** 2026-04-26
**Checkpoint:** Episode 450, best_eval_reward = 0.4
**Source:** Dashboard analysis + checkpoint inspection

---

## Alpha/Beta Drift Patterns

### Fast Pathway (reactive memory)

| Parameter | Init | Final | Δ | Interpretation |
|-----------|------|-------|---|----------------|
| α (decay) | 0.55 | 0.5984 | +0.05 | Decay slowed → longer memory |
| β (reset) | 0.65 | 0.7043 | +0.05 | Reset weakened → less forgetting |

**Effect:** Fast pathway became ~10% "slower" — it retains information longer than initially configured.

### Slow Pathway (stable memory)

| Parameter | Init | Final | Δ | Interpretation |
|-----------|------|-------|---|----------------|
| α (decay) | 0.92 | 0.9472 | +0.03 | Slightly longer memory |
| β (reset) | 0.97 | 0.96 | -0.01 | Slightly more forgetting |

**Effect:** Slow pathway barely moved — staying at its "very slow" configuration.

### Key Finding: Convergence, Not Divergence

```
Fast:  α=0.55 → 0.60    β=0.65 → 0.70
Slow:  α=0.92 → 0.95    β=0.97 → 0.96

              Init                  Trained
Fast α  ●─────────────●                        0.55  →  0.60
Slow α  ●─────────────●                        0.92  →  0.95
        ─────────────────────────────────────────────────
         0.5          0.7          0.9          1.0
```

The pathways **drift toward each other** rather than specializing:
- Fast gained memory (moves toward slow's territory)
- Slow stayed roughly put (didn't become even slower)

### Possible Explanations

1. **Task doesn't need extreme timescale separation**
   - DefeatRoaches episodes are short (~134 steps on average)
   - Combat is tight — most relevant history is last few seconds
   - Dual timescales may be overkill for this temporal horizon

2. **Architecture handles temporal mixing elsewhere**
   - TBPTT with 128-step windows provides temporal credit assignment
   - Attention itself does cross-token mixing
   - SNN pathways may be absorbing what's "left over"

3. **Training is still early**
   - Episode 450 of 10,000 planned
   - Patterns may change with more data
   - Worth tracking trajectory, not just endpoint

### What to Watch

If α/β continue converging, we end up with a **single effective timescale** — the dual-pathway architecture would be doing extra work for no gain. This is a simplification opportunity:

```
If fast α → 0.85 and slow α → 0.90:
  → Single pathway at α≈0.88 might work just as well
  → 2× fewer parameters, 2× fewer state tensors
  → Faster throughput, less memory
```

---

## Dead Parameters: Attention LIF β

### Finding

```
attention.lif_q.beta:  0.5000 (unchanged from init)
attention.lif_k.beta:  0.5000 (unchanged from init)
attention.lif_v.beta:  0.5000 (unchanged from init)
```

### Why They're Dead

From `SpikingSelfAttention` (stateless per call):

```python
def forward(self, ...):
    # QKV LIF neurons are RE-INITIALIZED every call
    q = self.lif_q(self.q_proj(x))  # fresh state each time
    k = self.lif_k(self.k_proj(x))
    v = self.lif_v(self.v_proj(x))
    # No temporal carry across env steps
```

**Result:** β never influences computation — there's no temporal dimension for it to act on. The gradients are zero (or near-zero), so parameters stay at initialization.

### Cost

- **Compute:** We're paying the spike-sparsity cost for no temporal benefit
- **Optimizer:** 3 parameters consuming gradient state uselessly
- **Complexity:** LIF neurons when a simple Heaviside would do the same job

### Fixes

| Option | Description | Complexity |
|--------|-------------|------------|
| Pin β | Set `learn_beta=False` for attention LIFs | Trivial |
| Remove LIF | Replace with direct `spike = (x > 0).float()` | Low |
| Make stateful | Add QKV state carry across env steps | High, architectural |

**Recommendation:** Pin β with `learn_beta=False` for now. Making attention stateful is interesting but requires rethinking the whole attention block.

---

## Action Distribution Shift

### Late-Episode Change (episodes 800-1400+)

| Phase | No-Op | Move | Attack | Interpretation |
|-------|-------|------|--------|----------------|
| Early | 30% | 5% | 65% | Cautious exploration |
| Late | ~15% | ~10% | ~75% | More decisive, aggressive |

**Correlation with rewards:** As no-op dropped and attack increased:
- Rolling mean climbed from -65 → -40
- Episode length increased
- Variance increased (more wins AND more catastrophic losses)

**Interpretation:** Agent learned that clicking more often = better survival, but is still figuring out WHERE to click.

---

## Eval vs Stochastic Gap

| Metric | Value |
|--------|-------|
| Best eval reward | 0.4 |
| Avg training reward (at save) | -65.09 |
| Avg training reward (final) | ~-40 |

**Gap:** ~65 points between deterministic eval and stochastic training

### Possible Causes

1. **Entropy is too high** — exploration is actively hurting performance
   - `entropy_coef: 0.01` may be too high for this sparse-reward task
   - Stochastic policy makes many "bad" exploratory clicks

2. **Deterministic policy is actually good**
   - The learned policy knows WHERE to click
   - But exploration noise corrupts the targeting

### Suggested Experiment

Run eval with varying stochasticity:
```
Determistic:  reward ≈ 0.4
Low temperature:  reward ≈ ?
Training temperature:  reward ≈ -40
```

If low-temperature eval gets close to deterministic, we know exploration is the problem.

---

## Conv1 Interpretability Opportunity

Gipity's suggestion: **visualize conv1 filters** to see if the network is specializing on the 27 PySC2 feature layers.

Example questions:
- Does any filter specialize in `player_relative` (friend/foe separation)?
- Does any filter ignore `height_map` (irrelevant for flat terrain)?
- Do filters group by unit type vs. spatial features?

This could reveal whether the CNN is learning sensible PySC2 features or just generic edge detectors.

---

## Action Items

| Priority | Task | Why |
|----------|------|-----|
| Low | Pin attention LIF β | Remove dead parameters |
| Low | Log α/β trajectory over training | Thesis-worthy plot |
| Medium | Conv1 filter visualization | Interpretability |
| Medium | Eval with temperature sweep | Diagnose stochastic gap |
| High | Train to convergence (10k eps) | See if α/β keep converging |

---

## Related Files

- Dashboard: `tools/analysis/dashboard.py`
- SpikingSelfAttention: `agent_core/spiking_policy.py`
- Config: `models/Zero/effective_config.json`
