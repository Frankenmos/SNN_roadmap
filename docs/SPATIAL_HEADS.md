# Spatial Target Heads

Current state: **Phase 2 Complete** - `CoarseToFineTargetHead` implemented.

## Available Heads

| Head | Precision | Status | Config Value |
|------|-----------|--------|--------------|
| `FactorizedXYTargetHead` | 84×84 (factorized) | ✅ Legacy | `factorized_xy` |
| `TokenPointerTargetHead` | 7×7 (49 cells) | ✅ Current default | `token_pointer` |
| `CoarseToFineTargetHead` | 7×7×12×12 = 7056 positions | ✅ Implemented | `coarse_to_fine` |
| `HeatmapHead` | 84×84 = 7056 positions | 🚧 Future | `heatmap` |

## Quick Reference

### FactorizedXYTargetHead
- Separate `x_logits[B, 84]` and `y_logits[B, 84]`
- Legacy compatibility head
- Independent distributions (no joint modeling)

### TokenPointerTargetHead (current default)
- Single `token_logits[B, 49]` over 7×7 pooled grid
- Returns cell centers (e.g., pixel 6, 18, 30, ...)
- Lower precision but efficient

### CoarseToFineTargetHead (Phase 2)
- **Coarse**: `primary_logits[B, 49]` (7×7 grid)
- **Fine**: `secondary_logits[B, 49, 144]` (12×12 per cell)
- Full 84×84 precision = 7056 unique positions
- **Critical**: `evaluate()` uses **recorded** `coarse_index` for fine-head (teacher-forcing)

## To Switch Heads

Edit `config.yaml`:

```yaml
model:
  spatial_head_type: "coarse_to_fine"  # or "token_pointer", "factorized_xy"
```

## File Locations

- Implementation: [`agent_core/target_heads.py`](../agent_core/target_heads.py)
- Tests: [`tests/test_coarse_to_fine_head.py`](../tests/test_coarse_to_fine_head.py)
- Working log: [`docs/PHASE2_WORKING_LOG.md`](PHASE2_WORKING_LOG.md)
- Full spec (archived): [`docs/archive/spatial_target_migration_spec_BPTT_test.md`](archive/spatial_target_migration_spec_BPTT_test.md)

## Test Coverage

- `CoarseToFineTargetHead`: 19 tests (encode/decode, build, sample, evaluate, integration)
- Full test suite: 82 tests pass
