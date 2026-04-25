# Phase 2 Implementation Plan: Coarse-to-Fine Target Head

**Date:** 2026-04-23
**Branch:** `BPTT_test`
**Status:** Ready to implement

---

## Executive Summary

Based on Zero-3 analysis and prior documentation, Phase 1 (token-pointer) has hit a representational ceiling. The 7×7 = 49 position click space is too coarse for precise DefeatRoaches targeting, leading to:

- Plateau at ep 2354
- No-Op dominance (88.5% of late-training steps)
- Late-stage CoV of 23.81 (high instability)
- Only 49 discrete click positions (cell centers only)

**Phase 2 coarse-to-fine** restores full 84×84 precision while keeping the computational efficiency of the 7×7 spatial token grid.

---

## What Phase 2 Does

**Current (Phase 1):**
- 7×7 spatial tokens → 1 categorical → 49 positions
- Each position is a cell center (e.g., pixel 6, 18, 30, ...)
- Cannot express precise clicks within cells

**Phase 2:**
- Stage 1 (Coarse): 7×7 spatial tokens → categorical → 49 cells
- Stage 2 (Fine): Selected cell token → 12×12 local refinement → 144 offsets
- Total: 49 × 144 = 7056 effective positions
- Covers full 84×84 screen with natural precision hierarchy

---

## Why Phase 2 Should Help (from Zero-3 analysis)

### Problem: Representational Bottleneck
The token-pointer head exhausted its capacity by ep 2354. With only 49 positions, the policy:
- Cannot distinguish between clicking "near roach" vs "far but still in same cell"
- Over-generalizes targeting decisions
- Loses precision as episodes get longer

### Solution: Two-Stage Hierarchy
1. **Coarse stage** answers "which general region?" (7×7)
2. **Fine stage** answers "where exactly in that region?" (12×12)

This matches the natural structure of:
- The 7×7 spatial token grid (already computed)
- The 84×84 screen (12×12 = 84 pixels per coarse cell)

---

## Architecture

```
Input: latent [B, latent_dim], spatial_context [B, embed_dim, 7, 7], action_ids [B]

Stage 1 (Coarse) - same as token-pointer:
- Query MLP on latent + action embedding
- Dot-product scoring against 7×7 spatial tokens
- Output: coarse_logits [B, 49]

Stage 2 (Fine) - NEW:
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

---

## Implementation: CoarseToFineTargetHead

Add to `agent_core/target_heads.py` (after `TokenPointerTargetHead`):

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
        self.action_dim = int(action_dim)
        self.coarse_grid_size = int(coarse_grid_size)
        self.local_grid_size = int(local_grid_size)
        self.screen_size = int(screen_size)
        self.coarse_token_count = coarse_grid_size * coarse_grid_size
        self.fine_token_count = local_grid_size * local_grid_size

        # Coarse stage (reuse token-pointer logic)
        self.action_condition_embedding = nn.Embedding(self.action_dim, self.latent_dim)
        self.coarse_query_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.token_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Fine stage (local refinement per coarse token)
        # Pre-compute fine logits for all coarse tokens, gather during sample/eval
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

        # Coarse scores
        coarse_scores = torch.einsum("bd,bnd->bn", coarse_query, self.token_proj(tokens))

        # Fine stage: pre-compute fine logits for each coarse token
        # This is more efficient than building on-demand during sampling
        token_features = self.token_proj(tokens)  # [B, 49, D]

        # Broadcast and fuse: [B, 49, D] + [B, latent_dim] + [B, latent_dim]
        fine_features = torch.cat([
            token_features,
            latent.unsqueeze(1).expand(-1, self.coarse_token_count, -1),
            action_emb.unsqueeze(1).expand(-1, self.coarse_token_count, -1),
        ], dim=-1)  # [B, 49, D + 2*latent_dim]

        # Project to fine logits per token
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

        # Stage 2: sample fine conditioned on sampled coarse cell
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

        # Combined log-prob and entropy (normalized separately)
        coarse_lp = coarse_dist.log_prob(coarse_index)
        fine_lp = fine_dist.log_prob(fine_index)
        coarse_ent = self._normalized_entropy(coarse_dist.entropy(), self.coarse_token_count)
        fine_ent = self._normalized_entropy(fine_dist.entropy(), self.fine_token_count)

        return TargetSample(
            x=x,
            y=y,
            coarse_index=coarse_index,
            fine_index=fine_index,
            target_index=None,
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

        # Evaluate fine conditioned on RECORDED coarse index (critical!)
        batch_indices = torch.arange(coarse_index.size(0), device=coarse_index.device)
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

---

## Integration Points

### 1. Update `build_spatial_target_head()` factory

In `agent_core/target_heads.py`, add the factory case:

```python
def build_spatial_target_head(
    *,
    head_type: str,
    embed_dim: int,
    latent_dim: int,
    action_dim: int,
    screen_size: int,
    coarse_grid_size: int = 7,
    local_grid_size: int = 12,
    target_decode_mode: str = "center",
) -> BaseSpatialTargetHead:
    if head_type == "factorized_xy":
        return FactorizedXYTargetHead(
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            screen_size=screen_size,
        )
    if head_type == "token_pointer":
        return TokenPointerTargetHead(
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            coarse_grid_size=coarse_grid_size,
            local_grid_size=local_grid_size,
            screen_size=screen_size,
            target_decode_mode=target_decode_mode,
        )
    if head_type == "coarse_to_fine":
        return CoarseToFineTargetHead(
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            coarse_grid_size=coarse_grid_size,
            local_grid_size=local_grid_size,
            screen_size=screen_size,
        )
    raise ValueError(f"Unknown spatial_head_type: {head_type}")
```

### 2. Update `config.yaml`

```yaml
model:
  spatial_head_type: "coarse_to_fine"  # Change from "token_pointer"
  coarse_grid_size: 7
  local_grid_size: 12
  # (other config remains the same)
```

### 3. Update `results.py` action semantics detection

```python
def _infer_action_semantics(self) -> str:
    # ... existing code ...

    # Add coarse_to_fine detection
    spatial_head = str(model_cfg.get("spatial_head_type", "")).lower()
    if spatial_head == "coarse_to_fine":
        return "semantic_pointer_v1"  # Same action semantics, different head

    # ... rest of existing code ...
```

---

## Tests

### Encode/Decode Roundtrip Test

```python
def test_coarse_to_fine_encode_decode_roundtrip():
    head = CoarseToFineTargetHead(
        embed_dim=64,
        latent_dim=64,
        action_dim=3,
        coarse_grid_size=7,
        local_grid_size=12,
        screen_size=84,
    )

    # Test corners, centers, and random points
    test_points = [
        (0, 0), (83, 83),  # corners
        (42, 42),          # center
        (6, 6), (66, 66),  # near corners
        (30, 45), (57, 12),  # random
    ]

    for x, y in test_points:
        encoded = head.encode_xy_to_target(
            torch.tensor([x]),
            torch.tensor([y])
        )
        decoded_x, decoded_y = head.decode_target_to_xy(
            coarse_index=encoded["coarse_index"],
            fine_index=encoded["fine_index"]
        )
        assert decoded_x.item() == x, f"Failed for ({x}, {y})"
        assert decoded_y.item() == y, f"Failed for ({x}, {y})"
```

### Teacher-Forcing Replay Test

```python
def test_coarse_to_fine_replay_teacher_forcing():
    """Verify evaluate() uses recorded coarse, not resampled."""
    head = CoarseToFineTargetHead(...)
    latent = torch.randn(2, 64)
    spatial_context = torch.randn(2, 64, 7, 7)
    action_ids = torch.tensor([2, 2])  # RIGHT_CLICK

    # Build head state
    head_state = head.build(latent, spatial_context, action_ids)

    # Sample some targets
    sample = head.sample(head_state, deterministic=False)

    # Evaluate should use recorded coarse_index for fine-head
    eval_result = head.evaluate(
        head_state,
        coarse_index=sample.coarse_index,
        fine_index=sample.fine_index
    )

    # If we manually corrupt the recorded coarse_index, fine logits should change
    fake_coarse = (sample.coarse_index + 1) % 49
    eval_result_fake = head.evaluate(
        head_state,
        coarse_index=fake_coarse,
        fine_index=sample.fine_index
    )

    # Results should be different (proving fine-head uses recorded coarse)
    assert not torch.allclose(eval_result.log_prob, eval_result_fake.log_prob)
```

---

## Expected Improvements

Based on Zero-3 patterns, Phase 2 should:

1. **Increase click precision** - 7056 positions vs 49
2. **Reduce No-Op dominance** - agent can express precise actions
3. **Delay/reduce plateau** - more representational capacity
4. **Improve gradient flow** - fine head gets more direct signal

### Success Criteria

- [ ] No non-finite gradients in 500+ episodes
- [ ] Coarse and fine entropy both remain > 0.1
- [ ] Deterministic eval shows visibly sharper clicks than token-pointer
- [ ] Win rate is at least as stable as token-pointer baseline
- [ ] No-Op percentage drops below 50% (from 88.5%)

---

## Rollback Strategy

If Phase 2 shows instability:
1. Revert `config.yaml` to `spatial_head_type: "token_pointer"`
2. Check logs for which head (coarse or fine) is causing issues
3. Consider:
   - Reducing learning rate further
   - Adding gradient clipping
   - Adjusting entropy coefficient
   - Fixing replay teacher-forcing bug

---

## What Phase 2 Does NOT Fix

Phase 2 is purely a targeting precision upgrade. It does NOT address:
- Reward shaping issues (still need reward refactor)
- Time-limit semantics (already fixed separately)
- Action-history tokens (future feature branch)
- Entity identity via tags (future feature)

---

## Open Questions

1. **Should we pre-train the coarse stage** from Zero-3 checkpoint?
   - Option A: Initialize coarse from token-pointer weights
   - Option B: Train from scratch
   - Recommendation: Try A first, fallback to B if unstable

2. **Entropy balance** - should we weight coarse vs fine entropy differently?
   - Current: `coarse_ent + fine_ent`
   - Alternative: Scale fine entropy higher (more fine-grained exploration)
   - Recommendation: Keep equal initially, monitor separately

---

## Implementation Order

1. Add `CoarseToFineTargetHead` to `target_heads.py`
2. Update `build_spatial_target_head()` factory
3. Add encode/decode roundtrip tests
4. Add teacher-forcing replay tests
5. Update `config.yaml` to use `coarse_to_fine`
6. Run: `pytest tests -q` (verify no regressions)
7. Run a short smoke test (100-200 episodes) named `Zero-4-smoke`
8. If stable, run full training as `Zero-4`

---

## References

- Original spec: `docs/spatial_target_migration_spec_BPTT_test.md`
- Implementation details: `docs/current/phase2_3_implementation_plan.md`
- Zero-3 analysis: `analysis_results/Zero-3/instability_report.txt`
- Deep research review: `docs/deep-research-report.md`
