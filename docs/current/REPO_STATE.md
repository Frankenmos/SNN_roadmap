# Repo State

Updated: 2026-04-20

## What The Repo Does Today

This is currently an SNN + PPO DefeatRoaches project with:

- hybrid observation tokenization from Fix 3
- Stage-1 TBPTT with ordered chunk replay and packed replay
- SDPA-backed attention as the current low-risk attention fast path
- SQLite logging plus analysis plots under `analysis_results/`
- explicit conditioned action semantics for `NO_OP`, `MOVE`, and `ATTACK`

The current policy input path is:

- spatial `feature_screen` -> CNN -> pooled spatial tokens
- `feature_units` -> entity tokens
- `multi_select` / `single_select` -> selection tokens
- `meta_vec[32] = player[11] + available_actions[16] + pysc2_last_action[1] + bridge_token[4]`
- token-type embeddings
- spiking self-attention
- token-temporal SNN
- PPO action / spatial / value readout

The current action path is:

- policy action vocab: `NO_OP`, `MOVE`, `ATTACK`
- `MOVE -> Move_screen(x, y)`
- `ATTACK -> Attack_screen(x, y)`
- one reset bootstrap `select_army` step outside PPO memory so policy-controlled steps start from a selected-army state

## What Is Considered Current

- `docs/current/REPO_STATE.md`
  short repo truth and open questions
- `docs/current/action_refactor.md`
  what the Stage-1 action refactor landed and what remains
- `docs/current/THE_BPTT.md`
  still current because Stage-1 TBPTT is implemented and its long-term verdict is still open
- `docs/current/working_log.md`
  compressed recent-memory log
- `config.yaml`
  current default knobs

## What Is Done

- Fix 3 hybrid observation tokenization
- extractor hardening for normalizer stability and eval-stat isolation
- Stage-1 TBPTT
- packed replay speed pass
- SDPA attention swap
- action-space availability diagnostics wrapper
- Stage-1 action refactor:
  conditioned spatial `MOVE` / `ATTACK`, availability masking, executed-action bridge token, and reset bootstrap outside PPO memory

## What Is Not Done

- dedicated action-history token group replacing the 4-float bridge token in `meta_vec`
- selection actions and broader learnable action vocabulary
- tag-pinned entity identity via `raw_units.tag`
- final verdict on whether the SNN + TBPTT branch is worth keeping as the long-term game-learning backbone
- broader multi-minigame / full-game branch

## Known Compatibility Note

Checkpoints from before the 2026-04-20 action refactor should be treated as incompatible:

- `meta_vec` width changed from `28` to `32`
- the meta encoder input changed
- the policy readout now conditions spatial logits on stored action IDs

## Open Questions

1. Does the new action refactor improve deterministic behavior and action commitment after real training, not just sampled behavior?
2. Is the 4-float bridge token enough for now, or should Stage 2 move action history into its own token group soon?
3. How much remaining weakness is action-space design versus entity identity versus optimization instability?
4. When we leave "make the SNN work" mode, do we keep this branch as the research branch and build a denser recurrent branch for the actual game-learning push?

## Archive Notes

- detailed pre-compression implementation history:
  `docs/archive/working_log_2026-04-20_pre_compress.md`
- external action-space ideation and scratch files:
  `docs/archive/observations_2026-04-20/`

## Practical Entry Points

- `README.md`
  general project entry
- `PPO_CNN_run.py`
  training entrypoint
- `PPO_CNN_eval.py`
  evaluation entrypoint
- `docs/README.md`
  doc map
