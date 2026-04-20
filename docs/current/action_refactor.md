# Action Refactor — Token-First, Conditionally-Sampled Policy

Plan to integrate a tokenized action vocabulary, feed the agent's own
last action back as observation, and resolve the "policy always outputs
continuous activations even for no-op" problem.

**Scope locked:** action space, observation plumbing of the agent's
own token, and the minimum policy/PPO changes to factor the joint
action distribution correctly. Everything else — TBPTT, masking
regularization (DropHead, neuron dropout, observation dropout, latent
gating), bigger `embed_dim`, dense-transformer fork — lives in
`TBPTT_and_training_enhancements.md` (next plan, written after this
one lands).

Style follows `docs/archive/NEXT_FIXES_PLAN_3.md` — Socratic
walkthrough first, then concrete steps.

---

## 1. What we know

### 1.1 Code state (verified today)

- `action_space/action_space.py` — 3-way minimal API (`attack()`,
  `move()`, `find_units()`, `nearest_enemy_unit_center()`). No token
  output, no history capture.
- `obs_space/obs_space_2.py` — emits `PolicyInputBatch` with
  `meta_vec = [player(11), available_actions(16), last_action_id(1)]`
  → `META_VECTOR_DIM = 28`. The last-action ID comes from
  `obs.observation.last_actions`, which PySC2 has **already stripped
  of x/y arguments**.
- `PPO_CNN/policy_network.py` — `PolicyNetwork.forward` always emits
  `(action_logits[action_dim=3], move_x_logits[84], move_y_logits[84],
  state_value)` in parallel. Three independent heads per step. No
  conditioning on sampled action.
- `PPO_CNN/PPO.py` — `select_action` samples action, move_x, move_y
  independently. `log_prob = log π(a) + log π(x) + log π(y)`.
  Sum is over three independent categoricals.
- Draft files in `docs/ideas/observations/` — `action_space_v2.py`,
  `policy_input_v2.py`, `obs_space_2_v2.py` — sketches from the
  external discussion. Read but **not merged**.

### 1.2 Observed behaviors that motivate this refactor

- Policy computes `move_x/move_y` logits every step including for
  `no_op`. Those logits contribute to the loss via log-probability
  accumulation regardless of whether the sampled action was spatial.
- Gradient path from no-op rollouts flows into the move heads → move
  heads are trained on noise whenever a non-spatial action was
  executed.
- Entropy bonus currently normalizes per head (session memory from
  2026-04-15 fix). That helps, but doesn't address the gradient-on-
  irrelevant-samples issue.
- The network has a **1-dim categorical ID of last action** in
  meta_vec. It does NOT have the coordinates of its own last click.
  PySC2 strips those before we see them.

### 1.3 The non-obvious discovery

PySC2 does **not** encode the agent's own last click in the
observation:
- `feature_screen` has no cursor layer, no click trail, no ghost of
  where you clicked last frame.
- `obs.observation.last_actions` contains only function IDs (e.g. 12
  for `Attack_screen`, 13 for `Move_screen`) — the x/y arguments
  are stripped.

So if we want the policy to know "I just attack-clicked at (42,31)",
we must feed it ourselves. This is the core justification for adding
an agent-side action token to the observation. Confirmed in the
external discussion.

---

## 2. Socratic walkthrough

### Q1. Why is "always outputs continuous activations" a problem if we ignore the outputs anyway?

We don't actually ignore them — the loss does not. The joint
distribution is currently factored as three independent heads:

```
π(a, x, y | s) = π(a | s) · π(x | s) · π(y | s)
log_prob     = log π(a) + log π(x) + log π(y)
```

When the sampled action is `no_op`, the environment never uses the
`(x, y)` that the policy emitted — but PPO still adds
`log π(x) + log π(y)` to the trajectory's log-probability. The ratio
`log_prob_new - log_prob_old` therefore depends on `x, y` terms for
trajectories where `x, y` had zero effect on the reward. That is
spurious credit assignment: the move heads get gradient signal from
no-op samples.

Consequences:
- Move heads learn to predict x/y that correlate with the reward of
  *non-spatial* trajectories.
- Entropy of x/y heads is bounded below by the spurious signal.
- Policy sharpening is slower than it should be.

### Q2. What's the minimal fix that doesn't require a new architecture?

Factor the joint correctly using conditional independence:

```
π(a, x, y | s) = π(a | s) · [a is spatial] · π(x | s, a) · π(y | s, a)
log_prob_traj = log π(a) + is_spatial(a) · (log π(x | a) + log π(y | a))
```

Sampling:
1. Sample `a ~ π(· | s)`.
2. If `a` is spatial: sample `x, y ~ π(· | s, a)`. Otherwise `x, y`
   are undefined (don't sample, don't store).

Loss:
- PPO ratio uses the same conditional factorization in old_log_prob
  and new_log_prob. For non-spatial trajectories, both reduce to
  `log π(a)` alone.
- Entropy bonus: action-head entropy always counted; spatial-head
  entropy counted only over batch samples where spatial was
  sampled, averaged with the correct denominator.

This is sometimes called **action-argument masking** (the external
AI's framing) but the cleaner name is **conditional action
factorization**. It is not a regularization trick — it is the correct
probability model. The previous factoring was a bug.

### Q3. How does this relate to "autoregressive" heads?

They are cousins. Conditional factorization says `π(x,y|s,a)` depends
on the sampled `a`. The clean implementation is:

1. Shared trunk produces `h`.
2. Action logits `= actor_fc(h)`.
3. Sample `a`.
4. `h_cond = h + action_embedding(a)` (or concat + Linear).
5. Move_x, move_y logits from `h_cond`.

This is **latent autoregression**: the action embedding feeds back
into the representation before the spatial heads run — but it's all
one forward pass, no sequential transformer decoding. It gives you
AlphaStar's factorization benefit without AlphaStar's per-step
transformer-decode cost.

The external AI phrased this well: "reasoning stays in the recurrent
state, not as extra tokens you sample." Your existing
`TokenTemporalSNN` already carries latent state across env steps —
the action embedding is just a small addition to that stream.

### Q4. Why not just mask `move_x_logits` to `-inf` for non-spatial actions and call it done?

Two issues:
1. Masking the **logits** doesn't change the contribution to
   `log_prob` correctly — a uniform `-inf` gives `log π(x) = -log(84)`
   or similar depending on implementation, not zero.
2. Masking at the **sampled-action** level is the correct semantic.
   For non-spatial `a`, there is no random variable `x, y` to
   condition a distribution on — we simply don't sample them and
   don't include their log-probability.

Logit masking is a shortcut that may or may not give the same
gradient; conditional factorization is the right model.

### Q5. We want to feed the last action back into the observation. How should that be shaped?

Three options, ranked:

| # | Option | Cost | Fit |
|---|---|---|---|
| 1 | 4 extra floats in `meta_vec`: `[type, x/83, y/83, extra]` | 5 LOC, shape bump 28→32 | Bridge: good for Stage 1 |
| 2 | Dedicated action-history token group (last K=8 actions) in attention stream | New token group, ~150 LOC, RoPE on the K-dim | Destination: Stage 2 |
| 3 | Both: bridge now, migrate later | Re-plumb in Stage 2 | This is the plan |

Stage 1 uses Option 1. The bridge is explicit — once Stage 2 lands
a proper token group, we remove the 4 floats from `meta_vec` and
restore `META_VECTOR_DIM` to 28. The bridge lets us verify the
*information* is useful before paying for the attention-visible
representation.

### Q6. The external AI's patch writes `meta_vec[..., -4:]` in MetaEncoder. Is that safe?

No — brittle. If any future field is appended after the agent token,
the `-4:` indexing silently grabs the wrong bytes and nothing throws.

Fix: named offset constants in `PPO_CNN/policy_input.py`:

```python
AGENT_LAST_ACTION_DIM: Final[int] = 4
AGENT_LAST_ACTION_OFFSET: Final[int] = (
    META_PLAYER_FEATURE_DIM
    + META_AVAILABLE_ACTION_DIM
    + META_LAST_ACTION_INDEX_DIM
)
META_VECTOR_DIM: Final[int] = AGENT_LAST_ACTION_OFFSET + AGENT_LAST_ACTION_DIM
```

MetaEncoder and ObservationExtractor both reference the named
offset. Adding another meta field in the future requires updating
the offset constants, not remembering a slice index.

### Q7. Should the action `type` go in as a scalar or an embedding?

Embedding. `type` is categorical over 10 classes. Passing it as a
scalar causes the MLP to see `type=4` as "close to" `type=5` in
magnitude terms, which is not what categorical identity means. Use
`nn.Embedding(10, embed_dim_small)` and concat with the `(x/83,
y/83, extra)` floats.

The external AI's patch missed this — it passed the type as a raw
float. That's a small but real mistake; we correct it.

### Q8. The draft `action_space_v2.py` has a 10-token vocabulary. Do we use all of it in Stage 1?

No. The vocabulary **shape** is locked but only a subset is
**learnable** in Stage 1:

| Token ID | Name | Stage 1 learnable? | Notes |
|---|---|---|---|
| 0 | NO_OP | ✅ | existing |
| 1 | SELECT_POINT | ❌ (stub) | defer to Stage 2 |
| 2 | SELECT_RECT | ❌ (stub) | defer to Stage 2 |
| 3 | SMART | ⚠️ | alias of MOVE for DefeatRoaches; pick one |
| 4 | ATTACK | ✅ | existing |
| 5 | MOVE | ✅ | existing |
| 6 | HARVEST | ❌ (stub) | not used in DefeatRoaches |
| 7 | BUILD | ❌ (stub) | needs building_id — Stage 3+ |
| 8 | TRAIN | ❌ (stub) | needs unit_id — Stage 3+ |
| 9 | ABILITY | ❌ (stub) | needs ability_id — Stage 3+ |

The overlap between SMART (3) and MOVE (5) is real — v1 used
`Smart_screen` for the `.move()` helper. For Stage 1, pick **MOVE**
as the canonical token for the v1 `.move()` behavior and leave
SMART unused. Reconsider when we add proper game-context
disambiguation.

**Concretely for Stage 1**: the policy's `action_dim` stays at 3
(NO_OP, ATTACK, MOVE). The token format uses the 10-slot
vocabulary so Stage 2 expansion doesn't reshape the token.

### Q9. Does this refactor invalidate any active training run?

Yes. Any checkpoint from before `META_VECTOR_DIM` changed has a
different `MetaEncoder.fused_input_dim` → `nn.Linear` shape
mismatch on load. Given the current 3500-ep run is already dead
from the Fix-3.5 pool refactor, the incremental cost is zero.

### Q10. What does the conditional-sampling change mean for the PPO buffer and GAE?

Nothing changes structurally. The buffer already stores `action`,
`move_x`, `move_y`, `log_prob`. After this refactor:
- `log_prob` for non-spatial transitions stores only `log π(a)`.
- `move_x`, `move_y` for non-spatial transitions can be zero or
  "undefined" — we never read them for log-prob computation in the
  update pass (the `is_spatial(a)` gate handles it).

GAE is computed over rewards/values and is independent of the
action factorization. No change.

### Q11. What if the sampled action is ATTACK (spatial) but at eval time we use `nearest_enemy_unit_center` to pick the coords, ignoring the move heads?

Good catch. Current v1 behavior: action 0 = "attack nearest enemy"
(coords come from `nearest_enemy_unit_center`, not from the move
heads). Action 1 = "move to (move_x, move_y)". Action 2 = no_op.

So the move heads are **only** used for action 1. ATTACK already
has its coords chosen by a scripted heuristic.

Implication: Stage 1's conditional factorization treats ATTACK as
**non-spatial from the network's perspective** — `is_spatial(a)`
is True only for MOVE. ATTACK's coords are scripted, so no
log-probability contribution from the move heads. NO_OP: same.
Only MOVE gets the spatial log-prob contribution.

This tracks the current actual behavior and cleanly removes the
spurious gradient. Later, when ATTACK becomes learnable-coords
(Stage 2), we flip its `is_spatial` to True.

### Q12. The question the user explicitly asked: "we can do autoregressive critique following — reason based on information but not output reasoning as tokens"?

This is latent autoregression (Q3) plus an auxiliary value head
that conditions on action. Stage 1 scope does **not** include the
auxiliary critique head — it's architectural, belongs in the next
doc. But the action-embedding-into-trunk (Q3) is Stage 1, and that
gives you the core conditional-information benefit without any
token output.

---

## 3. Options on the table

| # | Option | Scope | My read |
|---|---|---|---|
| A | Minimal: token into `meta_vec`, no conditional factorization | 1 file, ~10 LOC | Doesn't fix continuous-activation — not worth |
| B | Stage 1: token into `meta_vec` + conditional factorization + action embedding in trunk | Action, obs, policy, PPO — ~200 LOC | **Recommended** |
| C | Full stage: B + dedicated action-history token group (K=8) with RoPE | Adds attention plumbing — ~350 LOC | Destination, defer to Stage 2 |
| D | B + available-action-mask-logits-to-(-inf) regularization | B + tiny | Composes with B. Include. |

**Recommendation: B + D.** Option C is the destination but we land
it in Stage 2 after Stage 1 proves the information is useful.
Option D is a free correctness win: the available-actions mask from
`meta_vec` is already computed; use it to gate `action_logits` so
the policy can't sample unavailable actions. This is a correctness
fix, not regularization.

---

## 4. Concrete implementation plan

### Step A — Vocabulary protocol lock-in (no code yet)

Freeze the 10-slot token vocabulary in `action_space/action_space.py`
as module-level constants. Document which slots are learnable in
Stage 1 per Q8. No Python code change until Step B.

### Step B — ActionSpace extension

`action_space/action_space.py`:
- Add `NO_OP=0, SELECT_POINT=1, ..., ABILITY=9` constants.
- Add `self.last_token = np.zeros(4, dtype=np.int32)` in `__init__`.
- Add `_token(type, x, y, extra)` helper that writes `last_token`.
- `attack()`, `move()` write their token on every call.
- Add `get_last_token()` returning a copy.
- **Keep the existing API of `attack()`/`move()` unchanged** (no API
  break — agent loop continues to work).
- Stubs for `select_point`, `select_rect`, `harvest`, `build`,
  `train`, `ability` — write token + return `no_op()`. They are
  not learnable yet.

### Step C — PolicyInput constants

`PPO_CNN/policy_input.py`:
- Add `AGENT_LAST_ACTION_DIM = 4`
- Add `AGENT_LAST_ACTION_OFFSET` per Q6.
- Update `META_VECTOR_DIM` to include it (28 → 32).
- Do **not** add a new token group. Bridge via meta_vec in Stage 1.
- Add optional `action_token_embed_dim` or similar if we want the
  type embedding sized independently — defer until Step E says we
  need it.

### Step D — ObservationExtractor

`obs_space/obs_space_2.py`:
- `extract_observation(obs, update_stats=True, last_action_token=None)`.
- Default `last_action_token = np.zeros(4, dtype=np.int32)` (no-op
  sentinel) so callers that don't pass it still work.
- Normalize coords: `x_norm = x / 83.0`, same for y.
- Append `[type, x_norm, y_norm, extra_norm]` to meta_vec at the
  named offset.
- Add a small sanity test that `extract_observation(obs).meta_vec.shape[-1] == META_VECTOR_DIM`.

### Step E — MetaEncoder (policy_network.py)

- Add `self.action_type_embedding = nn.Embedding(10, 8)` (8-dim
  embedding for the 10 action types).
- In `forward()`, split the meta_vec at the named offset:
  - `player = meta_vec[..., :player_dim]`
  - `available = meta_vec[..., player_dim : player_dim+available_dim]`
  - `last_id = meta_vec[..., player_dim+available_dim]` (existing PySC2 last-action ID)
  - `agent_token = meta_vec[..., AGENT_LAST_ACTION_OFFSET : AGENT_LAST_ACTION_OFFSET + AGENT_LAST_ACTION_DIM]`
  - `agent_type_emb = action_type_embedding(agent_token[..., 0].long().clamp(0, 9))`
  - `agent_coords = agent_token[..., 1:]` (x_norm, y_norm, extra_norm)
  - `fused = cat(player, available, last_action_emb, agent_type_emb, agent_coords)`
- Bump `fused_input_dim` accordingly.

### Step F — PolicyNetwork conditional forward

- Add `self.action_embedding_into_trunk = nn.Embedding(action_dim, embed_dim_trunk)`. (`action_dim=3` in Stage 1.)
- Modify `forward_step_tensors` to return **action logits only
  first**. After sampling (in `select_action`) the caller passes the
  sampled action back into a **second small forward step** to get
  conditional `(move_x_logits, move_y_logits)`.
- Concretely: refactor the readout into two callables:
  - `action_head(h) -> (action_logits, value)`
  - `spatial_head(h, sampled_action) -> (move_x_logits, move_y_logits)`
    where internally `h_cond = h + action_embedding_into_trunk(sampled_action)` and the move heads operate on `h_cond`.
- In training (when we already have the sampled `a` from the rollout),
  we run `action_head` and `spatial_head(h, a)` in sequence as one
  forward pass — no extra network call.

**Alternative low-cost variant**: keep `forward` emitting both in
parallel, but compute `h_cond` internally from the
argmax-of-logits during rollout and from the stored sampled action
during training. Cleaner to the caller but tangles the forward
signature. I prefer the explicit two-head split.

### Step G — PPO.select_action conditional sampling

- Compute `h, action_logits, value`.
- Apply **available-actions mask** (from Option D) on
  `action_logits`: `action_logits.masked_fill(~available_mask, -1e4)`.
- Sample `a` (or argmax for eval).
- If `a == MOVE` (spatial): compute `move_x_logits, move_y_logits`
  conditional on `a`, sample `x, y`, accumulate
  `log_prob_xy = log π(x|a) + log π(y|a)`.
- Else: `log_prob_xy = 0.0`, `x = y = 0` (sentinel).
- Return `(a, x, y, log_prob = log π(a) + log_prob_xy, value)`.

### Step H — PPO.update_policy conditional log-prob

- For every transition in the rollout batch:
  - Recompute `log π(a)` from current network.
  - If `a == MOVE`: also recompute `log π(x|a) + log π(y|a)` using
    the stored `x, y`.
  - `log_prob_new = log π(a) + is_spatial(a) · (log π(x|a) + log π(y|a))`.
- Ratio `r = exp(log_prob_new - log_prob_old)` applied to clipped
  surrogate — same formula as before.
- Entropy bonus: see §5 subtle point 5.2.

### Step I — Agent wiring (DefeatRoaches)

- `__init__`: `self.last_action_token = np.zeros(4, dtype=np.int32)`.
- `step`:
  - Call `self.extractor.extract_observation(obs, last_action_token=self.last_action_token)`.
  - After executing action: `self.last_action_token = self.action_space.get_last_token()`.
- `reset`: zero the token.

### Step J — Tests

- `tests/test_action_space.py`:
  - Every primitive writes the expected 4-tuple token.
  - `get_last_token()` returns a copy (not the internal buffer).
- `tests/test_policy_input.py`:
  - `meta_vec.shape[-1] == 32`.
  - `meta_vec[..., AGENT_LAST_ACTION_OFFSET:]` equals the token passed in.
- `tests/test_policy_forward.py` (new):
  - Conditional sample roundtrip: given a sampled `a`, recomputing
    `log_prob` with the same `(a, x, y)` gives the same scalar.
  - `a = no_op` gives `log_prob == log π(a)` (no move contribution).
- `tests/test_PPO.py`:
  - Loss computation over a mixed-action batch: no-op samples do
    not contribute to move-head gradient (mock gradient check).

---

## 5. Subtle points

### 5.1 The "always outputs continuous activations" resolution

The network still **computes** move_x, move_y logits every forward
pass — that's unavoidable without a major architectural split. What
changes:
1. We don't feed them into the joint log-probability for non-spatial
   samples.
2. We don't accumulate gradient from those samples into the move
   heads.
3. The action embedding feeds back into `h_cond` so when the move
   heads *are* used, they see what action was chosen.

This is the correct probability model. Further regularization of
the move heads (DropHead, neuron dropout) is a separate lever in
the next plan doc.

### 5.2 Entropy bonus rebalancing

PPO entropy bonus is `-Σ_h entropy(h)` summed over heads, currently
normalized per head (session memory 2026-04-15). After Stage 1:

- Action head: always contributes.
- Move heads: contribute only for spatial samples.

The natural formula:

```python
entropy_bonus = H(a) + spatial_frac * (H(x) + H(y))
# where spatial_frac = fraction of batch with a == MOVE
```

If `spatial_frac ≈ 0.3` (typical), the move-head contribution is
~30% of its pre-Stage-1 weight. This will make total entropy bonus
smaller. May need to adjust `entropy_coef` to compensate — empirical.

### 5.3 PPO clip ratio consistency

The ratio must be computed with the same factorization in rollout
and update. Don't mix old "independent log-prob sum" from an old
checkpoint with new "conditional log-prob" from current network.
The current 3500-ep run was checkpoint-killed anyway, so no
contamination risk.

### 5.4 Gradient path through the sampled action

`action_embedding_into_trunk(sampled_action)` feeds into the move
heads via `h_cond`. This is a differentiable path **conditional on
the sample**. Because the sample is from a categorical (not
reparameterized), gradient does not flow from x,y back to action
logits through the embedding — only through the value function and
entropy regularizer. This is standard PPO behavior. If you want
gradient to flow action→move-head through the sample you need
Gumbel-Softmax or straight-through — **out of scope** for Stage 1.

### 5.5 The bridge→destination migration story

Stage 1 puts the agent token in meta_vec. Stage 2 will:
1. Remove the 4 dims from `meta_vec` (restore 28).
2. Add a new token group `ActionHistoryTokens[B, 8, D]` with RoPE
   on the time dimension.
3. Update `TOKEN_TYPE_GROUPS = 5`, `TOTAL_TOKEN_COUNT = 102`.
4. The `action_type_embedding(10, 8)` from MetaEncoder gets
   repurposed into the per-token embedding within the new group.

Stage 2 is explicitly **not** in this plan. Stage 1's purpose is to
validate that the information is useful (measurable by: policy
commits more cleanly to spatial vs non-spatial actions under
deterministic eval, and move-head gradient norm on non-spatial
batches drops to near-zero).

### 5.6 Available-actions mask (Option D)

We already compute `available_actions` binary mask in meta_vec.
Apply it to `action_logits` before softmax:

```python
# action_logits shape [B, action_dim]
# available_mask shape [B, action_dim] — we need to MAP the
# DEFEAT_ROACHES_ACTION_IDS 16-slot mask to the 3-slot action_dim.
mapped_mask = torch.stack([
    available_mask[:, DEFEAT_ROACHES_ACTION_IDS.index(Attack_screen_id)],
    available_mask[:, DEFEAT_ROACHES_ACTION_IDS.index(Move_screen_id)],
    torch.ones_like(...),  # no_op always available
], dim=-1)
action_logits = action_logits.masked_fill(~mapped_mask.bool(), -1e4)
```

This is pure correctness — we can't sample an unavailable action.
Currently the agent handles unavailability by scripting (the
`can_attack`/`can_move` gates in `DefeatRoaches.step`). Moving that
gate into the policy's action sampling removes the need for the
scripted fallback and gives the policy *gradient signal* about
availability — it learns "when available_actions doesn't include
Attack_screen, don't even try."

### 5.7 PySC2 argument strip-quirk preservation

`obs.observation.last_actions[0]` (the existing PySC2 last-action
ID in meta_vec) and the agent token's `type` field are redundant
but not identical:
- PySC2's ID uses the full action-function namespace
  (function_id 12 for Attack_screen, 13 for Move_screen, etc.).
- Agent token's `type` uses the compact 10-slot vocabulary.

Both are carried in meta_vec for now (cheap, 1 dim + 4 dims). We
can drop PySC2's last-action ID in Stage 2 when the dedicated
action-history token group takes over the agent-perspective role
entirely.

---

## 6. Verification signals

After Stage 1 implementation:

- **Move-head gradient norm on non-spatial batches → ≈ 0**. This is
  the headline signal. Run a controlled batch of only no-op
  transitions through `update_policy`; inspect
  `policy.move_x_fc.weight.grad.norm()`. Should be near zero. If
  not, the conditional factorization is wrong.
- **Action mix sharpens under deterministic eval**. Before: roughly
  uniform 33/33/33 attack/move/noop. After: committed action mix
  that reflects strategy. Measurable by shannon entropy of the
  empirical action distribution at eval.
- **Entropy bonus magnitude drops**. Expected because x,y-entropy
  contributes only for spatial samples now. Adjust `entropy_coef`
  upward if exploration visibly collapses.
- **Training reward should not regress**. We are removing spurious
  gradient, not information. If reward regresses, something in the
  refactor is wrong (most likely entropy imbalance or sampling bug).
- **Inspector dump of meta_vec last 4 dims** at step t+1 equals the
  executed action token at step t. Pure plumbing check.

---

## 7. Ordering

1. **This plan** — action refactor Stage 1. Scope: action vocabulary
   lockdown, token capture, meta_vec bridge, conditional action
   factorization, available-actions mask, agent wiring, tests. **One
   plan, one signal.**
2. **Next plan** — `TBPTT_and_training_enhancements.md`. Scope:
   TBPTT implementation (the "trace" from `THE_BPTT.md`), masking
   regularization (DropHead / neuron dropout / observation dropout /
   latent gate between attention and SNN), `embed_dim` scaling
   decisions, possibly dense-transformer fork.
3. **Stage 2 action refactor** — dedicated action-history token
   group with RoPE, removal of the meta_vec bridge. After Stage 1
   verification signals confirm the bridge is earning its keep.
4. **Stage 3+** — expanded learnable action vocabulary
   (SELECT_POINT, SELECT_RECT, then harvest/build/train/ability for
   non-DefeatRoaches maps).

---

## 8. Skeptical notes on the external discussion

User flagged: the other AI "can sound confident but has little
experience actually coding." Things I'd push back on from the
`observations.md` dialogue:

- **`meta_vec[..., -4:]` indexing** — brittle, silently breaks on
  any future meta field addition. Fixed in this plan via named
  offsets (Q6).
- **Passing `type` as a scalar float** — loses categorical identity
  semantics. Fixed via `nn.Embedding(10, 8)` (Q7).
- **"Action masking is one if statement"** — undersold.
  Conditional factorization affects the probability model, the
  entropy bonus, the PPO ratio, the rollout buffer semantics, and
  the gradient flow. It's one *concept* but touches four files
  (Q2, §5.2, §5.3).
- **MaskedLinear / MoE replacement as a one-liner** — premature.
  That's an architectural regularization change that composes
  nastily with the conditional-factorization change if done
  together. Deferred to next plan explicitly (§7).
- **"Three lines of code and you get 80% of autoregressive
  benefit"** — the *idea* is right (latent autoregression), but
  the implementation needs more care than three lines. Proper
  `action_embedding_into_trunk` + two-head forward split + PPO
  update consistency are not one-liner changes. §4 steps E-H
  spell out what's actually required.
- **DropHead, neuron dropout, observation dropout, latent gate** —
  all proposed in the same conversation turn. Each is a separate
  decision with its own interaction risks. Bundled together they
  are hard to ablate. **Deferred explicitly to the next plan.**
- **The 10-token vocabulary vs only-3-learnable reality** — the
  external AI was enthusiastic about the full vocabulary. In Stage
  1 only 3 slots are learnable; the other 7 are token-format
  reservations. Q8 tracks this.
- **`SMART` vs `MOVE` overlap** — the external AI included both in
  the vocabulary without disambiguating. Q8 picks MOVE and parks
  SMART unused until there's a concrete game-context difference.

Nothing in the conversation is *wrong* at an idea level; the gaps
are implementation-surface-area and scope discipline. This plan
tightens both.

---

## 9. Open questions for you

Before coding:

1. **Attack coords**: currently scripted via
   `nearest_enemy_unit_center`. Keep scripted in Stage 1 (no
   spatial log-prob from ATTACK) per Q11? Or make ATTACK
   learnable-coords now (then `is_spatial(ATTACK) = True`)? My
   recommendation: **keep scripted** in Stage 1 — one fix at a
   time, verify conditional factorization works with current
   coord-sourcing, then make ATTACK learnable in Stage 2 with
   pointer-network-style target selection (discussed in
   `WHEN_SHIT_GETS_DONE.md` §2).
2. **Entropy_coef retune**: leave it alone in Stage 1, adjust only
   if exploration visibly collapses? Or preemptively bump by ~1/
   expected-spatial-fraction?
3. **Available-actions mask scope**: full 3-slot mask per Stage 1
   (§5.6), or defer the masking to Stage 2 to keep the diff
   smaller? My recommendation: **include in Stage 1** — it's pure
   correctness, no tuning risk.
4. **Two-head forward vs internal h_cond**: §4 Step F spells out
   two choices. My recommendation: **explicit two-head split**
   (cleaner), but the internal-h_cond variant is a valid
   low-churn alternative if you want a smaller diff.

Ping when you've read this. No code yet. We align on the plan,
then execute step by step per the session discipline.
