# Next Fixes — Planning & Reasoning (Part 3)

Follow-up to `NEXT_FIXES_PLAN.md`, written after inspecting
`analysis_results/run_20260418_224920/`. Fixes 1 and 2 landed and the
run showed a specific failure mode that Fix 3 targets directly.

---

## Fix 3 — Hybrid Observation Tokenization

### 3.1 What we know from the code + inspector

- [obs_space/obs_space_2.py:58-77](obs_space/obs_space_2.py#L58-L77):
  `extract_observation` returns `(spatial [27, 84, 84], vector [100])`.
  The 100-d vector comes from `_extract_vector_features` distilling
  `feature_units` to ~11 hand-crafted scalars × 3 history frames.
- Policy concat at
  [PPO_CNN/policy_network.py](PPO_CNN/policy_network.py):
  `[3136 spatial after pool + 100 vector]`. The vector branch is ~3% of
  the total signal.
- Deterministic vs stochastic eval on `run_20260418_224920` @ ep 1700:
  training rolling mean = 661, deterministic eval mean = 78.80. ~8×
  gap. Visual inspection during real-time eval shows unreliable kiting
  under sampling; argmax brawls at close range and dies.
- The failure mode: the policy's `(attack, move, no_op)` logits at
  close range are near-tied because the obs does not carry the
  information a kiting decision actually hinges on (weapon cooldown,
  enemy range, selection state).

**Inspector data from the eval dump (166 records across 10 episodes):**

| Field | Shape | Populated | Notes |
|---|---|---|---|
| `feature_units` | `[N ≤ 23, 46]` | 166/166 | 46 attrs per unit. Currently only 11 aggregated. |
| `raw_units` | `[N ≤ 23, 46]` | 166/166 | Full range incl. unit `tag` (up to 4.3e9) for identity. |
| `multi_select` | `[K ≤ 19, 7]` | 149/166 (~90%) | Explicit "what's selected". Currently unused. |
| `single_select` | `[K ≤ 1, 7]` | 6/166 | Rare in DefeatRoaches. |
| `player` | `[11]` | 166/166 | Resources/supply/army counts (range 0..19 here). |
| `available_actions` | `[≤ 17]` | 166/166 | Action IDs available this step. |
| `last_actions` | `[1]` | 52/166 (~31%) | Needs "no action" sentinel slot. |
| `feature_minimap` | `[11, 64, 64]` | 166/166 | Coarse. Ignore for now. |
| `control_groups` | `[10, 2]` | 166/166 | All zeros in DefeatRoaches. Drop. |
| `build_queue`, `cargo`, `production_queue`, `alerts`, `upgrades`, `feature_effects`, `radar`, `raw_effects` | — | 0/166 | Empty for this minigame. Skip. |

Unit count distribution: min=2, max=23, mean=10.1, p50=10, p90=14,
**p99=22**. → `N_max = 24` covers p99 with margin.

`feature_units` values range includes `-5`, so at least one attribute
is signed. Naive normalization would distort it.

### 3.2 Socratic walkthrough

> **Q1.** The deterministic policy picks "attack" at close range to a
> roach with its cooldown still winding up. What obs feature tells it
> "your weapon is on cooldown, wait"?

Nothing reaches the policy. `weapon_cooldown` lives in `FeatureUnit`
and sits in the `[N, 46]` table; the current extractor never reads it.
The policy is genuinely blind to its own firing readiness.

> **Q2.** Why does stochastic sampling score 8× deterministic then?

Because the logits at close range are near-tied — "attack vs move" is
ambiguous without cooldown info. Sampling occasionally lands on move,
which is the correct call mid-cooldown. Argmax picks whichever head was
fractionally more confident (attack) and never escapes.

> **Q3.** Why tokenize instead of rasterizing the missing attributes
> onto new `feature_screen` channels?

Rasterization works for spatial-continuous attributes (health, shield).
It fails for:

- **Discrete** fields: `unit_type`, `order_id` — one channel per
  category blows up `C`.
- **Identity-dependent** fields: "this specific marine has cooldown
  0.3" — rasterization loses identity when units overlap.
- **Non-spatial** fields: `build_progress`, `cargo`, `weapon_cooldown`
  are unit properties, not fields.

Tokens preserve entity identity and carry discrete types natively via
embedding.

> **Q4.** How many unit slots do we need?

From the inspector: `p99 = 22`. `N_max = 24` covers p99 with margin,
is cheap (`24 × 46 × fp32 × 2048 ≈ 8.6 MB` per rollout), and keeps
attention token count manageable (~75 total vs current 49).

> **Q5.** Can the CNN branch shrink now that tokens carry semantic
> unit data?

Yes, but **not on day 1**. `feature_screen` channels like `unit_type`,
`player_relative`, `selected` duplicate what entity tokens will carry
— eventually we drop them. CNN surgery is a separate change. One
thing at a time.

> **Q6.** Where does "what's selected" live?

Its own token group, not buried in a meta scalar. `multi_select` is
populated in ~90% of steps and carries up to 19 units × 7 attrs — it
is qualitatively different from "what's on the field" and deserves
its own attention-visible identity.

> **Q7.** What stays in the meta token then?

Everything global and non-selection:

- `player[11]` (resources, supply, army counts)
- `available_actions` → fixed-length binary mask over the primitives
  we care about
- `last_actions[1]` → embedding with a "no action this step" sentinel
  (since only ~31% of steps carry one)
- (later) reward-bucket, episode-step fraction

One token, carries the "what game phase am I in, what did I just do"
context.

### 3.3 Options on the table

| # | Option | Coverage of info gap | Complexity |
|---|---|---|---|
| i | Rasterize missing attrs as new `feature_screen` channels | Partial (spatial-continuous only) | Low |
| ii | Hybrid: keep CNN on `feature_screen`, add `feature_units` tokens + `multi_select` tokens + meta token | Full (entity + selection + meta + spatial) | Medium |
| iii | Full tokenization, drop CNN entirely | Full, but big rewrite | High |
| iv | Hybrid + shrink CNN channels + past-action history tokens | Full + short-term memory | High (multi-change) |

### 3.4 Recommendation: Option (ii)

Smallest change that covers the full information gap. Leaves CNN
untouched, leaves action space untouched. One change, one signal —
same discipline as Fixes 1 and 2.

### 3.5 Concrete implementation plan

**Step A — Protocol lock-in (no model code yet).**

Define `PolicyInputBatch` as a dataclass. Freeze the shape before
writing encoder code; downstream buffer, checkpoint format, and tests
all consume this.

```
PolicyInputBatch:
    spatial_obs:            [B, 27, 84, 84]    # unchanged
    entity_features:        [B, 24, F_unit]    # feature_units, padded
    entity_mask:            [B, 24]
    selection_features:     [B, 20, 7]         # multi_select, padded
    selection_mask:         [B, 20]
    meta_vec:               [B, F_meta]        # player + avail_mask + last_act_embed
    state_in:               (snn_state tuple)  # unchanged
```

`F_unit` = curated subset of ~20 attrs from `pysc2.lib.features.FeatureUnit`
(see §3.7). `F_meta` ≈ 30-40 after the avail-actions mask + last-action
embedding + player scalars.

**Step B — Encoder modules.**

- `EntityEncoder`: per-unit MLP `F_unit → D`, mask-zeroed output.
  `unit_type` handled via `nn.Embedding`, concatenated with
  continuous attrs before the MLP.
- `SelectionEncoder`: similar, for multi_select `7 → D`.
- `MetaEncoder`: MLP `F_meta → D` → single token `[B, 1, D]`.
- Existing CNN: unchanged on day 1, emits `[B, 49, D]`.
- **Token-type embedding** `[4, D]` added to each group (spatial /
  entity / selection / meta) so attention distinguishes them.
- Concat → `[B, 49 + 24 + 20 + 1, D] = [B, 94, D]`.

**Step C — Rewire `PolicyNetwork.forward`.**

Forward takes the batch object. `SpikingSelfAttention` receives the
wider token sequence. Mask support added in attention (see §3.7 for
AMP-mask numerics).

**Step D — PPO buffer / replay.**

Every transition stores the full batch slice. VRAM budget at 2048
transitions:

- entity: `24 × F_unit × fp32 × 2048` ≈ 8 MB
- selection: `20 × 7 × fp32 × 2048` ≈ 1 MB
- meta: `≈ 40 × fp32 × 2048` ≈ 0.3 MB
- spatial (already shipping): ≈ 1.5 GB

New obs pieces are rounding error compared to what we already carry.

**Step E — `obs_space_2.py` replacement.**

`ObservationExtractor.extract_observation` returns the batch fields,
not `(spatial, vector)`. `feature_units` extraction becomes **identity
+ pad + mask**, not aggregation. No more hand-crafted scalars.

**Step F — Logging.**

Add to `ppo_updates`:

- `entity_mask_utilization` (mean fraction of real tokens)
- `entity_count_p50 / p99` (per-update)
- `selection_mask_utilization`

If utilization stays >90% consistently, `N_max` was too tight. Low
utilization is fine — it just means padding is doing its job.

### 3.6 Verification signals

- **Deterministic eval reward lifts toward stochastic.** Headline
  signal. If cooldown/range info fixes argmax brittleness, the 8×
  train/eval gap from run_20260418_224920 should narrow substantially.
- **`mean_entropy` drops slightly.** A better-informed policy is more
  confident. A sharpening, not a collapse.
- **Action mix becomes situation-dependent** instead of drifting to
  flat 75% attack. Measurable by slicing action mix by per-step unit
  count or health ratio.
- **Late-run reward CoV drops.** If the current noisiness comes from
  stochastic-only kiting, reliable kiting under argmax should stabilize
  episodes.

### 3.7 Subtle points

- **F_unit: all 46 vs curated subset.** Pre-select ~20 informative
  attrs: `unit_type`, `alliance`, `health`, `health_max`, `shield`,
  `shield_max`, `energy`, `weapon_cooldown`, `x`, `y`, `radius`,
  `build_progress`, `order_id_0`, `order_id_1`, `is_selected`,
  `is_in_cargo`, `assigned_harvesters`, `ideal_harvesters`, `active`,
  `hallucinated`. The encoder MLP would learn to ignore noise, but
  sample efficiency is better with a focused input.
- **Signed values in `feature_units` (observed min `-5`).** Standardize
  per-attribute using running mean/std, or keep the field as-is if it
  is actually categorical. Do not assume `[0, 255]`-style normalization.
- **`feature_units` vs `raw_units`.** `feature_units` is byte-clipped
  (max 255); `raw_units` has full range incl. unit `tag` for identity
  tracking. For Fix 3, use `feature_units` — we are not doing per-unit
  history yet. Switch to `raw_units.tag` when history comes.
- **Unit-type embedding.** `unit_type` is a categorical ID in ~[0,
  2000]. Treat it as an integer scalar and the MLP sees it as a magnitude
  — wrong. Use `nn.Embedding(max_unit_type, D_embed)` and concatenate
  with continuous attrs.
- **Mask handling in attention.** `SpikingSelfAttention` currently does
  not take a mask. Add `attn_weights.masked_fill(~mask, large_neg)`
  before softmax. **AMP + `-inf` is a known footgun** — use a finite
  large negative (e.g. `-1e4`) or force masked softmax to fp32.
- **`multi_select` overlap with `feature_units`.** Some units appear
  in both tables. That is fine — they carry different semantic roles
  (on-screen vs. currently-selected) and the token-type embedding
  keeps them distinct for attention.
- **`last_actions` sparsity.** Non-empty in only ~31% of steps. The
  last-action embedding table needs an explicit "no-action" sentinel
  index so the encoder does not have to guess.
- **`control_groups` and empty tables in DefeatRoaches.** Drop
  `control_groups`, `build_queue`, `cargo`, `production_queue`,
  `alerts`, `upgrades`, `feature_effects`, `radar`, `raw_effects` from
  the encoder. They are always empty in this minigame. Re-enable when
  we leave DefeatRoaches.
- **`feature_minimap [11, 64, 64]` is unused.** Always populated but
  coarse and not critical for DefeatRoaches (one screen). Potential
  future second CNN branch for strategic-map tasks; out of scope here.

---

## Ordering (relative to earlier plans)

1. **Fix 3 (this doc) comes before action-space redesign.** The visible
   failure mode (unreliable kiting under argmax) is an observation
   problem, not an action problem — the policy does not need new
   primitives to kite, it needs the information to know when to kite.
2. **Protocol-first discipline still applies.** Step A of Fix 3 is
   itself a partial `PolicyInputBatch` introduction — it is the first
   real payoff for the `plan.md` Phase 1 "protocol refactor" ordering.
   Doing Fix 3 forces the protocol to exist in concrete form.
3. **Action-space redesign** follows once Fix 3 has either (a) closed
   the train/eval gap (so we can measure the action-space change
   against a real baseline) or (b) visibly not closed it (so we know
   the bottleneck moved somewhere else).
4. **CNN channel shrink and past-action history tokens** are deferred
   to a later fix. Fix 3 deliberately does not touch them.

## Anchor for Fix 3 success

Re-run the analyzer on the Fix-3 run. If the fix is doing what we
expect:

- `deterministic eval reward mean` at comparable episode count should
  be **within 2× of training rolling mean** (currently 8×).
- `mean_entropy` should be lower but still > 0.3 (sharper, not collapsed).
- Action-mix slice by "any enemy health < 50" should show visibly more
  move% when close to low-HP enemies (focus-fire + reposition), not
  the flat 75% attack we see now.
- `entity_mask_utilization` median around 0.4-0.6 (10 units / 24 slots).
  Spikes to ~0.9 during peak fights tell us `N_max` is right-sized.

If `deterministic eval reward` does not lift, the bottleneck is not
cooldown/selection observability and we re-examine before touching
action space.
