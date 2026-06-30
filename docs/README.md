# Docs

Updated: 2026-06-26

Start with the current docs below. This directory keeps tracked documentation
focused on current code and active references.

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
