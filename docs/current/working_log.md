# Working Log

Updated: 2026-07-13

This file is intentionally short and only tracks current-era repo state.

## 2026-07-13 Checkpoint resume safety (branch fix/checkpoint-resume-safety)

- Context: verified that `attention.lif_q/k/v.beta` receive an exactly-zero
  gradient (membranes re-init to zero every forward + single LIF step, so
  beta multiplies zero; outputs bit-identical under beta 0.01<->0.99; smoke
  snapshots u5-u20 all 0.5 exact). Same finding + fix already existed in
  `analysis_results/archive-old/Zero/parameter_drift_analysis.md`
  (2026-04-26) and was forgotten. The planned `learn_beta=False` cleanup is
  checkpoint-safe for MODEL state (buffer keeps the `beta` key) but would
  have tripped a resume trap: optimizer `load_state_dict` raises on the
  changed param set, the old blanket `except` renamed checkpoint.pth to
  `.corrupted` and silently restarted at episode 0 with already-loaded
  weights + fresh optimizer. This branch removes that trap first.
- `train.load_checkpoint` rewritten (validate-then-apply): model keys/shapes
  and optimizer param-group sizes are checked BEFORE any state is applied;
  incompatibility raises `CheckpointResumeError` with an actionable message
  instead of mutating the agent. The checkpoint file is never renamed or
  deleted anymore. Fresh-run state only for: no file, or the deliberate
  policy-protocol mismatch skip (unchanged).
- Explicit weights-only resume added: `train.py --resume_weights_only` /
  `ray_train --resume-weights-only` restores policy weights, extractor
  state, and counters (episode, update_count, best_eval_reward) with a
  fresh optimizer/scheduler — the sanctioned path across a trainable-
  param-set change. Recorded in the run manifest via `resolved_launch`;
  Mission Control `build_launch_command` mirrors the new flag.
- Tests: `tests/test_checkpoint_resume.py` (8) — clean roundtrip, optimizer
  mismatch fails atomically (param->buffer scenario in miniature), weights-
  only resume, shape mismatch, unreadable file left in place, protocol
  mismatch, missing file. Full suite 239 passing. CLI parse smoke-tested on
  both entry points with real imports.
- NOTE: the attention-beta cleanup itself is NOT in this branch (one fix at
  a time); it should land at a run boundary and use weights-only resume if
  a lineage needs to continue across it.

## 2026-07-07 V7 @ ep 5500 read: SIL trust-region runaway CONFIRMED

- The 2026-07-02 smell-list question "SIL vs trust region" is now answered
  empirically from the run DB (`ppo_updates`, 316 rows): approx KL grows
  monotonically 0.004 (u1) -> 0.03 (u46) -> 0.31 (u76) -> ~1.0 (u180+),
  clip fraction saturates ~0.88 at clip_eps 0.10. Curriculum ended at u120,
  KL kept climbing after — curriculum-drift excuse is dead.
- SIL gate never closes: `sil_gate_open_fraction` flat at 0.52-0.66 for 240+
  updates, buffer pinned at 5000, `sil_grad_norm` (pre-clip) rising 3 -> 13.
  Cause: critic miscalibrated (EV 0.12-0.20; trace replay showed V~-25 vs
  episode landing +1) so (R-V)+ stays wide open -> permanent imitation push.
- Behavior: det eval declines monotonically -3.6 (ep1469) -> -5.4 (~ep3700)
  -> -9.0 (ep5227, n=5, std=0.00 = stereotyped). Det diagnostics: 3254/3274
  actions are Smart_screen. Aiming (WHERE) is good — trace replay shows
  clicks on roaches; dosage (WHEN/how often) is the failure. Training shaped
  reward climbs (-70 -> -43) while native eval score falls = shaped-vs-raw
  divergence; the queued raw-score validation is now urgent.
- `docs/notes/more-opinions-to-check.md` verified claim-by-claim against
  code: SIL separate-step (2a) CONFIRMED (`_run_sil_pass` own backward+step,
  ppo_trainer.py:2514, called after epoch loop :1341) — note it also shares
  the Adam optimizer (moment contamination). Stale stored SNN state in SIL
  entries (2b) CONFIRMED (:2392). "Clip SIL advantage to (R-V)+" ALREADY
  IMPLEMENTED (:2507). Window-boundary zeroing worry NOT APPLICABLE — chunks
  start from stored per-step state (:1516-1524). Delay-line burn-in claim
  NOT APPLICABLE (no delay lines in agent_core; state is (syn, mem) only).
  LEFT_CLICK entropy contamination HANDLED (masked_fill -1e4 :408 + per-head
  entropy normalization :550). Entity/selection carry OFF CONFIRMED
  (spiking_policy.py:418). V-trace/IMPALA: agree with notes' own conclusion —
  premature; no actor lag exists.
- Recommended order: (1) merge SIL into PPO loss (single backward, clipped)
  or at minimum separate optimizer + KL guard, lower sil_coef; (2) fix the
  critic so the gate can taper (return scale/normalization); (3) run the
  cheap ratio diagnostic (one epoch, no optimizer step, ratio vs 1) to size
  replay fidelity; (4) raw-score validation; (5) uint8 spatial storage +
  admission tightening (R>0) later.
- LATER SAME DAY — plan written: `docs/current/SIL_TRUST_REGION_FIX_PLAN.md`
  (Stage 0 ratio probe = measurement-only, Stage 1 merged SIL, Stage 2
  return normalization; one variable per run) + teaching doc
  `learning/MERGING_LOSSES.md`. Two NEW findings while grounding the plan:
  (a) `target_kl=0.03` early-stop (ppo_trainer.py:1331) has been firing at
  epoch 1 → the run trains at EFFECTIVE EPOCHS = 1; (b) grad decomposition
  shows `grad_norm_actor_head` decayed to 0.002–0.03 vs critic 11–160 →
  PPO barely trains the actor (88% clipped = zero policy grad); SIL's
  separate pass is the dominant actor-training signal. Collection verified
  SYNCHRONOUS (`_collect_sync_fragments`) → at epoch-1/group-1 the ratio
  MUST be ~1; whatever `kl_update_start` shows is replay fidelity, not
  policy movement.

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

- Mock-env audit (`tests/MockedEnv/`): fake pysc2 is installed globally for
  every test via `conftest.py` `pytest_plugins`; `policy_batch.py` factories
  are protocol-pinned (good). Gaps found: mocked obs feed an ALL-ZERO
  spatial stream (the `.player_relative` attribute on `MockFeatureScreen` is
  consumed by nothing — extractor reads the full 27-layer array); the
  production numeric entity-row path (`_project_numeric_rows` +
  `_FEATURE_UNIT_INDEX`) has zero test coverage (mocks always take the
  object/`getattr` path, which silently defaults 15 of 21 curated fields to
  0.0); fake `Move_screen` id 13 ≠ real 331 (only mislabels a last-action
  vocab slot; availability check uses Smart 451, which is correct);
  `attack_range` on mock units and root-level `ai_test_utils.py` are both
  orphaned/vestigial.
- Assessed `E:\SNN\codex-experiment` (TinySkirmish, Codex 2026-06-28): a
  clean numpy grid-skirmish digital twin of the TENSOR protocol
  (`PolicyInputBatch` v3 shapes exact, NO_OP/LEFT/RIGHT_CLICK vocab,
  itemized reward dict that must sum to total) — not of the pysc2 seam
  (27 spatial channels are hand-authored). `real_snn_bridge/rollout`
  demonstrably ran the real `PolicyNetwork`+`PPO` end-to-end (CPU+CUDA
  renders committed). Caveats: NOT under version control (`.git` empty),
  vendored 405-line copy of `policy_protocol.py` that can drift, LEFT_CLICK
  is a placeholder. Complementary to the mock: twin covers network+PPO
  learning dynamics; mock covers the pysc2→extractor seam; only real-env
  diagnostics cover feature_screen semantics.
- TinySkirmish IMPORTED into the repo as `envs/tiny_skirmish/` (same day,
  user-approved): vendored `policy_protocol` copy deleted — bridge modules now
  import the real `agent_core` directly (sys.path eviction machinery and
  `--snn-repo` flags removed); self-checks wrapped as
  `tests/test_tiny_skirmish.py` (6 tests, render/live skip without
  Pillow/pygame); CLI shim at `scripts/run_tiny_skirmish.py`. Left behind in
  `E:\SNN\codex-experiment`: renders/, empty `.git`, stray HTML. Verified:
  bridge forward pass reports `policy_class_module: agent_core.spiking_policy`;
  full suite 182 pass. Housekeeping/verification tooling — no agent or
  training-pipeline change (pause respected).

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
