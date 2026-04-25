# Implementation Plan: Phase 2 (Coarse-to-Fine) and Phase 3 (Heatmap)

**Date:** 2026-04-23
**Branch:** `BPTT_test`
**Current Status:** Phase 0+1 complete, semantic actions landed, token-pointer head is default

---

## Part 1: Lessons Learned from Failures

### 1.1 Gradient Instability (from `instability_report.txt`)

**Symptoms:**
- 2 updates with non-finite `grad_norm`
- 2 skipped optimizer steps in late training
- Suggested issue: surrogate gradient stability and reward/value scale mismatch

**Root Cause Analysis:**
The factorized x/y head created an implicit entropy bonus imbalance:
- Move actions got 2× entropy bonus (x + y) compared to no-op
- This created policy overconfidence in spatial actions
- Combined with large action spaces, this led to gradient explosion

**Mitigation for Phase 2/3:**
1. **Per-head entropy normalization** — already in `BaseSpatialTargetHead._normalized_entropy()`
2. **Target log-prob scaling** — consider scaling coarse+fine log-prob contribution
3. **Gradient clipping** — add explicit clipping in target heads
4. **Separate logging** — monitor coarse vs fine log-prob scales

### 1.2 Late-Stage Instability

**Symptoms:**
- High CoV (23.81) after plateau at episode 2354
- Clip fraction 0.095 suggests policy still changing significantly
- Plateau detection indicates model stopped improving

**Interpretation:**
The 7×7 token-pointer head is **too coarse** to express precise targeting:
- Can only click on 49 discrete locations (cell centers)
- Limits the policy's ability to learn fine-grained targeting
- Explains the plateau — policy has exhausted the representational capacity

**Why Phase 2 Should Help:**
- Coarse-to-fine allows 7×7×12×12 = 7056 effective positions
- Keeps computational efficiency of coarse grid
- Adds refinement head for precision

### 1.3 Replay Math is Critical

**The spec emphasizes:** teacher-force recorded targets during replay. Do NOT resample.

**Why this matters:**
- If you resample coarse during replay, the fine head sees the wrong distribution
- This causes policy collapse where coarse and fine predictions diverge
- The PPO contract requires evaluating the **recorded** target, not a resampled one

**Implementation implication:**
In `evaluate()`, always use the recorded `coarse_index` to build the fine-head state:

```python
# WRONG: resample coarse
coarse_logits = ...  # recompute coarse
fine_logits = self._build_fine(coarse_logits)  # uses resampled coarse

# RIGHT: use recorded coarse
fine_logits = self._build_fine_from_recorded(recorded_coarse_index)
```

### 1.4 Entropy Bonus Balance

**Historical bug (fixed):** Entropy was computed per-dimension for factorized x/y,
giving move actions 2× the entropy bonus of no-op actions.

**Lesson for Phase 2:**
When summing `coarse_entropy + fine_entropy`:
- Normalize each separately (already done)
- Consider logging them separately to detect imbalances
- Watch for spatial actions being over-weighted in the loss

---

## Part 2: Phase 2 — Coarse-to-Fine Implementation

### 2.1 Architecture Overview

```
Input: latent [B, latent_dim], spatial_context [B, embed_dim, 7, 7], action_ids [B]

Stage 1 (Coarse):
- Query MLP on latent + action embedding
- Dot-product scoring against 7×7 spatial tokens
- Output: coarse_logits [B, 49]

Stage 2 (Fine):
- Gather selected coarse token feature
- Fuse with latent + action embedding
- MLP refinement head
- Output: fine_logits [B, 144] for 12×12 local grid

Encode/Decode:
- coarse_index = (y // 12) * 7 + (x // 12)
- fine_index = (y % 12) * 12 + (x % 12)
- x = (coarse_index % 7) * 12 + (fine_index % 12)
- y = (coarse_index // 7) * 12 + (fine_index // 12)
```

### 2.2 Implementation: `CoarseToFineTargetHead`

Add to `agent_core/target_heads.py`:

```python
class CoarseToFineTargetHead(BaseSpatialTargetHead):
    head_type = "coarse_to_fine"

    def __init__(
        self,
        *,
        embed_dim: int,
        latent_dim: int,
        action_dim: int,
        coarse_grid_size: int = 7,
        local_grid_size: int = 12,
        screen_size: int = 84,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.latent_dim = int(latent_dim)
        self.coarse_grid_size = int(coarse_grid_size)
        self.local_grid_size = int(local_grid_size)
        self.screen_size = int(screen_size)
        self.coarse_token_count = coarse_grid_size * coarse_grid_size
        self.fine_token_count = local_grid_size * local_grid_size

        # Coarse stage (reuse token-pointer logic)
        self.action_condition_embedding = nn.Embedding(int(action_dim), self.latent_dim)
        self.coarse_query_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.token_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Fine stage (local refinement)
        self.fine_fusion_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 2 + self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.fine_readout = nn.Linear(self.embed_dim, self.fine_token_count)

    def build(
        self,
        latent: torch.Tensor,
        spatial_context: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> TargetHeadState:
        """Build both coarse and fine logits."""
        batch_size = latent.size(0)

        # Coarse stage (same as token-pointer)
        action_ids = action_ids.clamp(0, self.action_condition_embedding.num_embeddings - 1)
        action_emb = self.action_condition_embedding(action_ids)
        coarse_query = self.coarse_query_mlp(torch.cat((latent, action_emb), dim=-1))
        tokens = spatial_context.flatten(2).transpose(1, 2)  # [B, 49, D]
        coarse_scores = torch.einsum("bd,bnd->bn", coarse_query, self.token_proj(tokens))

        # Fine stage: build logits for each possible coarse cell
        # This is expensive: we need [B, 49, 144] for all combinations
        # For efficiency, compute fine features per token and project
        token_features = self.token_proj(tokens)  # [B, 49, D]

        # Broadcast for fine readout: [B, 49, D] -> [B, 49, 1, D]
        # Then concat with latent+action -> [B, 49, 1, D+2*latent_dim]
        # Then project to [B, 49, 144]

        # More efficient: pre-compute fine logits per token, gather during sample/eval
        fine_features = torch.cat([
            token_features,  # [B, 49, D]
            latent.unsqueeze(1).expand(-1, self.coarse_token_count, -1),  # [B, 49, latent_dim]
            action_emb.unsqueeze(1).expand(-1, self.coarse_token_count, -1),  # [B, 49, latent_dim]
        ], dim=-1)  # [B, 49, D + 2*latent_dim]

        fine_logits_per_token = self.fine_readout(
            self.fine_fusion_mlp(fine_features)
        )  # [B, 49, 144]

        return TargetHeadState(
            head_type=self.head_type,
            primary_logits=coarse_scores,  # [B, 49]
            secondary_logits=fine_logits_per_token,  # [B, 49, 144]
        )

    def sample(
        self,
        head_state: TargetHeadState,
        deterministic: bool = False,
    ) -> TargetSample:
        """Sample coarse cell, then fine offset within that cell."""
        # Stage 1: sample coarse
        coarse_dist = torch.distributions.Categorical(logits=head_state.primary_logits.float())
        if deterministic:
            coarse_index = head_state.primary_logits.float().argmax(dim=-1)
        else:
            coarse_index = coarse_dist.sample()

        # Stage 2: sample fine conditioned on the sampled coarse cell
        batch_indices = torch.arange(coarse_index.size(0), device=coarse_index.device)
        fine_logits_for_coarse = head_state.secondary_logits[
            batch_indices, coarse_index
        ]  # [B, 144]
        fine_dist = torch.distributions.Categorical(logits=fine_logits_for_coarse.float())

        if deterministic:
            fine_index = fine_logits_for_coarse.float().argmax(dim=-1)
        else:
            fine_index = fine_dist.sample()

        x, y = self.decode_target_to_xy(coarse_index=coarse_index, fine_index=fine_index)

        # Combined log-prob and entropy
        coarse_lp = coarse_dist.log_prob(coarse_index)
        fine_lp = fine_dist.log_prob(fine_index)
        coarse_ent = self._normalized_entropy(coarse_dist.entropy(), self.coarse_token_count)
        fine_ent = self._normalized_entropy(fine_dist.entropy(), self.fine_token_count)

        return TargetSample(
            x=x,
            y=y,
            coarse_index=coarse_index,
            fine_index=fine_index,
            log_prob=coarse_lp + fine_lp,
            entropy=coarse_ent + fine_ent,
        )

    def evaluate(
        self,
        head_state: TargetHeadState,
        *,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        target_index: torch.Tensor | None = None,
        coarse_index: torch.Tensor | None = None,
        fine_index: torch.Tensor | None = None,
    ) -> TargetEval:
        """Evaluate recorded coarse and fine indices (teacher-forcing)."""
        del target_index

        if coarse_index is None or fine_index is None:
            if x is None or y is None:
                raise ValueError("coarse_to_fine evaluate requires either coarse_index+fine_index or x+y")
            encoded = self.encode_xy_to_target(x.long(), y.long())
            coarse_index = encoded["coarse_index"]
            fine_index = encoded["fine_index"]
        else:
            coarse_index = coarse_index.long()
            fine_index = fine_index.long()
            # Handle masked indices (-1 for non-spatial actions)
            valid_mask = coarse_index >= 0

        # Evaluate coarse
        coarse_dist = torch.distributions.Categorical(logits=head_state.primary_logits.float())
        coarse_lp = coarse_dist.log_prob(coarse_index)
        coarse_lp = torch.where(valid_mask, coarse_lp, torch.zeros_like(coarse_lp))

        # Evaluate fine conditioned on RECORDED coarse index
        batch_indices = torch.arange(coarse_index.size(0), device=coarse_index.device)
        # Clamp coarse_index for gathering (handle -1 gracefully)
        safe_coarse = torch.clamp(coarse_index, 0, self.coarse_token_count - 1)
        fine_logits_for_coarse = head_state.secondary_logits[
            batch_indices, safe_coarse
        ]  # [B, 144]
        fine_dist = torch.distributions.Categorical(logits=fine_logits_for_coarse.float())
        fine_lp = fine_dist.log_prob(fine_index)
        fine_lp = torch.where(valid_mask, fine_lp, torch.zeros_like(fine_lp))

        # Entropy
        coarse_ent = self._normalized_entropy(coarse_dist.entropy(), self.coarse_token_count)
        fine_ent = self._normalized_entropy(fine_dist.entropy(), self.fine_token_count)

        return TargetEval(
            log_prob=coarse_lp + fine_lp,
            entropy=coarse_ent + fine_ent,
        )

    def encode_xy_to_target(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """Encode (x, y) to (coarse_index, fine_index)."""
        x = x.long().clamp(0, self.screen_size - 1)
        y = y.long().clamp(0, self.screen_size - 1)

        coarse_col = torch.div(x, self.local_grid_size, rounding_mode="floor")
        coarse_row = torch.div(y, self.local_grid_size, rounding_mode="floor")
        coarse_index = coarse_row * self.coarse_grid_size + coarse_col

        fine_col = x % self.local_grid_size
        fine_row = y % self.local_grid_size
        fine_index = fine_row * self.local_grid_size + fine_col

        return {
            "x": x,
            "y": y,
            "target_index": None,
            "coarse_index": coarse_index.long(),
            "fine_index": fine_index.long(),
        }

    def decode_target_to_xy(
        self,
        *,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        target_index: torch.Tensor | None = None,
        coarse_index: torch.Tensor | None = None,
        fine_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode (coarse_index, fine_index) to (x, y)."""
        del x, y, target_index

        if coarse_index is None or fine_index is None:
            raise ValueError("coarse_to_fine decode requires coarse_index and fine_index")

        coarse_index = coarse_index.long().clamp(0, self.coarse_token_count - 1)
        fine_index = fine_index.long().clamp(0, self.fine_token_count - 1)

        coarse_col = coarse_index % self.coarse_grid_size
        coarse_row = torch.div(coarse_index, self.coarse_grid_size, rounding_mode="floor")

        fine_col = fine_index % self.local_grid_size
        fine_row = torch.div(fine_index, self.local_grid_size, rounding_mode="floor")

        x_out = coarse_col * self.local_grid_size + fine_col
        y_out = coarse_row * self.local_grid_size + fine_row

        return x_out.long().clamp(0, self.screen_size - 1), y_out.long().clamp(0, self.screen_size - 1)
```

### 2.3 PPO Integration

**No changes needed** if Phase 0 was done correctly. The generic interface handles:
- `coarse_index` → `target_primary`
- `fine_index` → `target_secondary`

### 2.4 Config Changes

In `config.yaml`:

```yaml
model:
  spatial_head_type: "coarse_to_fine"  # Change from "token_pointer"
  # coarse_grid_size: 7  # Already exists
  # local_grid_size: 12  # Already exists
```

### 2.5 Tests for Phase 2

Add to `tests/test_PPO.py` or `tests/test_target_heads.py`:

```python
def test_coarse_to_fine_encode_decode_roundtrip():
    """Exact roundtrip for all representative (x, y) points."""
    head = CoarseToFineTargetHead(...)
    # Test corners, centers, and random points
    test_points = [(0, 0), (83, 83), (42, 42), (6, 6), (66, 66)]
    for x, y in test_points:
        encoded = head.encode_xy_to_target(torch.tensor([x]), torch.tensor([y]))
        decoded = head.decode_target_to_xy(
            coarse_index=encoded["coarse_index"],
            fine_index=encoded["fine_index"]
        )
        assert decoded[0].item() == x
        assert decoded[1].item() == y

def test_coarse_to_fine_replay_teacher_forcing():
    """Replay uses recorded coarse cell for fine-head evaluation."""
    # This is critical: evaluate() must use recorded coarse_index,
    # not re-sample it
    pass

def test_coarse_to_fine_nonspatial_bypass():
    """Non-spatial actions should bypass both coarse and fine losses."""
    pass
```

---

## Part 3: Phase 3 — Heatmap Head (Optional)

### 3.1 When to Implement

**Only implement Phase 3 if:**
- Phase 2 is stable and shows improvement
- Compute budget allows for more expensive forward pass
- You need the highest possible targeting precision

### 3.2 Architecture Overview

```
Input: spatial_context [B, D, 7, 7] + latent bias + action bias

Conv tower:
- Conv2d(D, D, 3x3) + ReLU
- Conv2d(D, D, 3x3) + ReLU
- Upsample to [B, D, 84, 84]
- Conv2d(D, 1, 1x1) -> [B, 1, 84, 84]
- Flatten -> [B, 7056]

Encode: primary = y * 84 + x, secondary = -1
Decode: x = primary % 84, y = primary // 84
```

### 3.3 Numerical Precautions

**Entropy over 7056 classes is much larger than action entropy:**
- Log per-head entropy separately
- Consider normalizing or scaling before tuning coefficients
- Verify AMP (Automatic Mixed Precision) stability

---

## Part 4: Implementation Checklist

### Phase 2 Steps

1. [ ] Implement `CoarseToFineTargetHead` in `agent_core/target_heads.py`
2. [ ] Add factory function in `build_spatial_target_head()`
3. [ ] Add encode/decode roundtrip tests
4. [ ] Add teacher-forcing replay tests
5. [ ] Run unit tests without env
6. [ ] Change `config.yaml` to use `coarse_to_fine`
7. [ ] Run a short smoke test (100-200 episodes)
8. [ ] Compare against token-pointer baseline
9. [ ] If stable, run full training run

### Phase 3 Steps (Only if Phase 2 successful)

1. [ ] Implement `HeatmapHead` in `agent_core/target_heads.py`
2. [ ] Add flat-index encode/decode tests
3. [ ] Add AMP/entropy sanity checks
4. [ ] Run unit tests
5. [ ] Change `config.yaml` to use `heatmap`
6. [ ] Run smoke test with gradient monitoring
7. [ ] Compare compute cost vs coarse-to-fine

---

## Part 5: Monitoring and Success Criteria

### 5.1 Key Metrics to Watch

| Metric | What It Shows | Danger Threshold |
|--------|---------------|------------------|
| `grad_norm` | Gradient stability | > 50 or inf/nan |
| `coarse_entropy` | Coarse head exploration | < 0.1 (collapse) |
| `fine_entropy` | Fine head exploration | < 0.1 (collapse) |
| `clip_fraction` | PPO policy change | > 0.3 (unstable) |
| `explained_variance` | Value function quality | < 0.5 (poor) |
| `win_rate` | Task performance | Should increase |

### 5.2 Success Criteria

**Phase 2 is successful when:**
- [ ] No non-finite gradients in 500+ episodes
- [ ] Coarse and fine entropy both remain > 0.1
- [ ] Deterministic eval shows visibly sharper clicks than token-pointer
- [ ] Win rate is at least as stable as token-pointer baseline

**Phase 3 is successful when:**
- [ ] All Phase 2 criteria pass
- [ ] Compute cost is acceptable (check FPS)
- [ ] Materially improves targeting over coarse-to-fine

---

## Part 6: Rollback Strategy

If Phase 2 shows instability:
1. Revert `config.yaml` to `spatial_head_type: "token_pointer"`
2. Check logs for which head (coarse or fine) is causing issues
3. Consider:
   - Reducing learning rate
   - Adding gradient clipping
   - Adjusting entropy coefficient
   - Fixing replay teacher-forcing bug

If Phase 3 is numerically unstable:
1. Likely cause: entropy scale over 7056 classes
2. Fix: normalize heatmap entropy more aggressively
3. Or: skip Phase 3 — coarse-to-fine may be sufficient

---

## Part 7: Open Questions

1. **Time-cap semantics**: From `open_questions.md`, need to decide if `steps_per_episode` is a real horizon or just truncation. This affects PPO bootstrap logic.

2. **LEFT_CLICK masking**: Should it stay masked or gain a real env mapping? This affects action space design.

3. **Reward semantics**: What outcome signal do we trust most? Wrapper-derived, raw `obs.last()`, or raw reward?

4. **Next focus after Phase 2**: Should it be:
   - `coarse_to_fine` (this plan)
   - Reward-only stabilization
   - Action-history tokens
   - Time-cap semantics

---

## Appendix: File Edit Summary

| File | Change |
|------|--------|
| `agent_core/target_heads.py` | Add `CoarseToFineTargetHead` class |
| `config.yaml` | Change `spatial_head_type` to `coarse_to_fine` |
| `tests/test_PPO.py` | Add coarse-to-fine tests |
| `tests/test_target_heads.py` | (Create) Focused target-head tests |
