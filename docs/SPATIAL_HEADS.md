# Spatial Target Heads

Updated: 2026-06-26

Current state: `CoarseToFineTargetHead` is the config default, with
`fine_skip_connection: true`.

## Available Heads

| Head | Precision | Status | Config Value |
| --- | --- | --- | --- |
| `FactorizedXYTargetHead` | 84x84 factorized | Legacy compatibility | `factorized_xy` |
| `TokenPointerTargetHead` | 7x7 cells | Available fallback | `token_pointer` |
| `CoarseToFineTargetHead` | 7x7 x 12x12 = 7056 positions | Current default | `coarse_to_fine` |
| `HeatmapHead` | 84x84 = 7056 positions | Future | `heatmap` |

## Coarse-To-Fine

- Coarse logits: `primary_logits [B, 49]`
- Fine logits: `secondary_logits [B, 49, 144]`
- Replay/evaluate uses the recorded `coarse_index` for the fine head.
- With `fine_skip_connection: true`, pre-pool conv2 features
  `[B, 32, 84, 84]` feed the fine stage so sub-cell logits can depend on the
  actual screen content at each pixel.

The fine skip is important. V5 deterministic diagnostics showed the old
fine-stage path used one constant fine sub-index (`10`) for all 1,099 Smart
clicks. V5 is therefore a cautionary artifact, not the current architecture.

See `docs/current/V5_COLLAPSE_AUDIT.md`.

## Switching Heads

Edit `config.yaml`:

```yaml
model:
  spatial_head_type: "coarse_to_fine"  # token_pointer or factorized_xy also exist
```

If `fine_skip_connection` is on, old no-skip checkpoints cannot load because
the head has extra parameters. Flip it off only when inspecting pre-V6
checkpoints.

## Files

- Implementation: `agent_core/target_heads.py`
- Policy integration: `agent_core/spiking_policy.py`
- Tests: `tests/test_coarse_to_fine_head.py`
- Historical spec: `docs/archive/spatial_target_migration_spec_BPTT_test.md`
