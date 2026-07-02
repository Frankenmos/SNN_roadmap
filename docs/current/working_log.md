# Working Log

Updated: 2026-07-02

This file is intentionally short and only tracks current-era repo state.

## 2026-07-02 V7 First Deterministic Engagement + Smell List + Learning Folder

- V7 (`banana_glasses_v7_sil_b2048_e4_a10`, SIL enabled) @ ~ep 1730: first
  deterministic eval that actually attacks the roaches (precise targeting,
  sparse engagement). Not yet rigorously attributed to SIL vs more training —
  probe P(RIGHT) and `sil_gate_open_fraction` logs are the attribution tools.
- Instability report HIGH flags: clip fraction ~56%, approximate KL 0.066
  (target 0.03). Shaped reward still fully negative (avg −68, best −12).
- Smell review produced five measurement questions — recorded in
  `REPO_STATE.md` "What Is Still Open" (gradient scale vs 0.5 clip, SIL trophy
  staleness, SIL vs trust region, ground-click auto-attack cancellation,
  never-positive reward / raw-score ablation).
- Created `learning/` (tutor instructions, materials, progress log): paused
  feature development in favor of owning the architecture; measurement-only
  code changes until the curriculum catches up.
- Docs reconciled with code: SIL added to `ARCHITECTURE.md` and
  `REPO_STATE.md`; status banners added to the two 2026-06-29 audit docs
  (their "SIL is not in the code" claims are now historical).
- Grad-norm decomposition logging added to `ppo_trainer.py`: per-module
  pre-clip norms (`grad_norm_trunk/actor_head/critic_head/target_head`) +
  `sil_grad_norm` for the SIL pass. FOUND+FIXED: SIL diagnostics
  (`sil_loss`, `sil_gate_open_fraction`, …) were returned by the trainer but
  silently dropped at persistence — not in `PPO_UPDATE_COLUMNS`
  (`Utility/logger_utils.py`). All new columns added; DB migrates itself via
  `_safe_add_column`. Full suite 176 pass.
- Scout findings: V7 has run only **61 PPO updates** (1730 episodes) — the
  right-click curriculum (120 updates) is STILL ACTIVE, so curriculum drift
  is a live confound in the 56% clip-fraction / KL 0.066 reading. Checkpoints
  are ~3.5 MiB; `policy_version = update_count` is stored inside. Ray path
  never emits STEP / REWARD_COMP records → `steps` and `reward_components`
  tables are empty in ALL runs (v5/v6/v7) → dashboard Policy + Reward tabs
  render empty (root cause, not a dashboard bug per se). eval.py verified
  read-only w.r.t. training artifacts (no torch.save, no models/ writes;
  outputs only under analysis_results/).

## 2026-06-30 SIL Implemented

- Feedback-gated SIL in `agent_core/ppo_trainer.py` (trophy deque, `(R−V)+`
  imitation pass with its own optimizer step), config flags in `config.yaml`,
  wired via `agent.py`. Tests: `tests/test_sil.py`. Trigger: V6 deterministic
  eval fully idle (P(NO_OP)≈0.97, +3.46 logit gap) while stochastic won.

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
