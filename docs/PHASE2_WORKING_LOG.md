# Phase 2: CoarseToFineTargetHead - Implementation Complete

**Status**: ✅ Complete (2026-04-23)

## What Was Done

Implemented `CoarseToFineTargetHead` for full 84×84 click precision while maintaining alignment with the 7×7 spatial token grid.

### Implementation Details

| Method | Purpose |
|--------|---------|
| `encode_xy_to_target()` | (x, y) → (coarse_index, fine_index) |
| `decode_target_to_xy()` | Exact roundtrip |
| `build()` | Returns `primary_logits[B, 49]` + `secondary_logits[B, 49, 144]` |
| `sample()` | Hierarchical: sample coarse, then fine conditioned on coarse |
| `evaluate()` | **Teacher-forcing**: uses recorded coarse_index for fine-head |

### Files Modified

- [`agent_core/target_heads.py`](../agent_core/target_heads.py) - Added `CoarseToFineTargetHead` (~170 lines)
- [`agent_core/spiking_policy.py`](../agent_core/spiking_policy.py) - Added import and wiring

### Files Created

- [`tests/test_coarse_to_fine_head.py`](../tests/test_coarse_to_fine_head.py) - 19 tests

### Test Results

- 19/19 new tests pass
- 82/82 total tests pass

### To Enable

```yaml
# config.yaml
model:
  spatial_head_type: "coarse_to_fine"
```

### Key Design Point

**Replay rule (critical)**: During PPO evaluation, the fine-head uses the **recorded** `coarse_index`, not a resampled one. This ensures correct teacher-forcing.

---

This log is retained only as a concise implementation record; superseded
design details are not part of the tracked docs.
