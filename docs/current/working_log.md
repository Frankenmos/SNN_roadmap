# Working Log

Updated: 2026-06-26

This file is intentionally short and only tracks current-era repo state.

## 2026-06-26 Docs Cleanup

- Moved point-in-time external takes and superseded reviews out of
  `docs/current/`.
- Added `V5_COLLAPSE_AUDIT.md` so the V5 diagnosis is not scattered across
  chat, logs, and analysis artifacts.
- Replaced long plan-style docs with compact current-state references.
- Treated old CNN/PPO-era kiting notes as historical, not current V5/SNN
  controls.

## Current Technical State

- Live policy protocol is v3:
  `stream_action_effect_feedback_v2`.
- Live action vocab is:
  `NO_OP`, `LEFT_CLICK`, `RIGHT_CLICK`; `RIGHT_CLICK` maps to
  `Smart_screen(x, y)`, and `LEFT_CLICK` remains masked in DefeatRoaches.
- Live target head is `coarse_to_fine` with `fine_skip_connection: true`.
- CUDA AMP is configured as `bf16`; fp16 GradScaler instability is not the
  current default.
- Reward v4 includes corrected kiting-distance defaults, score-delta kill
  credit, and Smart outcome rewards.
- Ray training has deterministic eval/best-checkpoint plumbing and extractor
  normalizer sync before best-save.

## Latest Run Read

- V5 (`banana_smart_v5_b2048_e4_a10`) is the collapse artifact:
  max reward `0.00`, no eval rows, constant fine sub-index in deterministic
  stage-0 diagnostics.
- V6 (`banana_glasses_v6_b2048_e4_a10`) is the post-fine-skip/glasses family:
  training reward became positive and max reward reached `555.85`, but
  deterministic eval still needs scrutiny.

## Next Work

- Use V6/V7 style runs, not V5, as the current comparison surface.
- Keep old CNN/PPO and pre-protocol-v3 run narratives in archive only.
- Verify deterministic behavior with trace-level click quality before adding
  more action-space complexity.
