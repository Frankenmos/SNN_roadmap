# Open Questions

Updated: 2026-06-26

## Needs Live Evidence

- Does the current fine-skip `coarse_to_fine` head produce deterministic clicks
  that stay near useful targets, not just stochastic diversity?
- Does `RIGHT_CLICK -> Smart_screen(x, y)` remain the right minimal action
  abstraction for DefeatRoaches?
- Should `LEFT_CLICK` stay masked until a distinct selection/control use case
  exists?
- Are V6/V7 positive training rewards coming from real improved combat or from
  reward-shaping side effects?

## Architecture

- When should entity identity be pinned with `raw_units.tag` so entity-token
  recurrent carry can be enabled?
- Is the SNN/TBPTT branch worth keeping as the main research branch after the
  current behavior is stable?
- Should the next target-head comparison be against `token_pointer` or a true
  84x84 heatmap?

## Training And Evaluation

- Do deterministic eval traces match stochastic training improvements?
- Is borrowed-actor Ray eval good enough, or do we need dedicated EvalActors?
- Do we need richer Ray step logging before the next long run?

## Explicitly Historical

- Old CNN/PPO-era run narratives are not a current control for V5/SNN. They
  belong in archive unless we intentionally inspect those old experiments.
