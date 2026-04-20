# Repo State

Updated: 2026-04-19

## What The Repo Does Today

This is currently an SNN + PPO DefeatRoaches project with:

- hybrid observation tokenization from Fix 3
- Stage-1 TBPTT through the recurrent token state
- packed replay for faster TBPTT updates
- SDPA-backed attention as the current low-risk attention fast path
- SQLite logging plus analysis plots under `analysis_results/`

The current policy path is:

- spatial `feature_screen` -> CNN -> pooled spatial tokens
- `feature_units` -> entity tokens
- `multi_select` / `single_select` -> selection tokens
- `player + available_actions + last_action` -> meta token
- token-type embeddings
- spiking self-attention
- token-temporal SNN
- PPO actor / critic heads

## What Is Considered Current

- `docs/current/THE_BPTT.md`
  still current because Stage-1 TBPTT is implemented, but its long-term
  verdict is not settled yet
- `docs/current/working_log.md`
  live implementation memory
- `config.yaml`
  current default knobs

## What Is Done

- Fix 3 hybrid observation tokenization
- extractor hardening for normalizer stability and eval-stat isolation
- Stage-1 TBPTT
- packed replay speed pass
- SDPA attention swap
- action-space availability diagnostics wrapper

## What Is Not Done

- tag-pinned entity identity via `raw_units.tag`
- final verdict on whether the SNN+BPTT branch is worth keeping as the
  long-term game-learning backbone
- tokenized action-space redesign
- broader multi-minigame / full-game branch

## Open Questions

1. Does Stage-1 TBPTT materially improve deterministic behavior, not
   just sampled behavior?
2. Is the remaining deterministic weakness mostly action-space design,
   entity identity, or general optimization instability?
3. When we leave “make the SNN work” mode, do we keep this branch as the
   research branch and build a more tractable dense recurrent branch for
   the actual game-learning push?

## Practical Entry Points

- `README.md`
  general project entry
- `PPO_CNN_run.py`
  training entrypoint
- `PPO_CNN_eval.py`
  evaluation entrypoint
- `docs/README.md`
  doc map

## Diagnostics Worth Remembering

Current config/logging now supports:

- observation inspector
- policy-input diagnostics
- available-actions diagnostics
- TBPTT update timing/cost metrics in `ppo_updates`

Those knobs now live in `config.yaml` as well as CLI flags where
appropriate.
