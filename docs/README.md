# Docs

Updated: 2026-06-26

Start with the current docs below. Anything in `docs/archive/` is historical
unless a current doc links to it for background.

## Start Here

| File | Purpose |
| --- | --- |
| `current/REPO_STATE.md` | Live code and run-family state |
| `current/ARCHITECTURE.md` | Concise live architecture |
| `current/V5_COLLAPSE_AUDIT.md` | V5 collapse diagnosis and stale-claim cleanup |
| `SPATIAL_HEADS.md` | Target-head reference |
| `current/ACTION_FEEDBACK_PLAN.md` | Protocol-3 action feedback token |
| `current/RAY_STATUS.md` | Current Ray rollout/eval status |
| `current/THE_BPTT.md` | TBPTT reasoning and entity-memory caveat |
| `current/open_questions.md` | Questions that still need evidence |

## Important Naming

- V5 and V6 are run families, not reward-function versions.
- V5 used `defeat_roaches_v4`; there is no `RewardFunctionV5`.
- Old CNN/PPO-era run narratives are historical, not the current V5/SNN
  control surface.
- The old `PPO_CNN/` package and root `PPO_CNN_*` launchers were removed.
  Current entrypoints are `train.py`, `eval.py`, and
  `python -m distributed.ray_train`.

## Current Run Anchors

- V5: `banana_smart_v5_b2048_e4_a10`
  Collapse artifact: max reward `0.00`, no eval rows, constant fine sub-index.
- V6: `banana_glasses_v6_b2048_e4_a10`
  Post-fine-skip/glasses family: positive training reward, but deterministic
  eval still needs trace-level scrutiny.

## Archive

Archived docs preserve old plans, external takes, and superseded reviews. They
are useful for archaeology, but they should not be read as current state.

Common archived references:

- `archive/architecture_2026-05-06_pre_v6.md`
- `archive/deep_research_report_2026-06-17_superseded.md`
- `archive/kimi_v5_autopsy_external_take_2026-06-11.md`
- `archive/action_feedback_plan_2026-04-25_superseded.md`
- `archive/working_log_2026-06-17_pre_docs_cleanup.md`
- `archive/spatial_target_migration_spec_BPTT_test.md`

## Ideas

`docs/ideas/` is speculative. Use it for future branch concepts, not current
implementation truth.
