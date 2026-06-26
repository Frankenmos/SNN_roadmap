# V5 Collapse Audit

Updated: 2026-06-26

Question answered: what was right, wrong, or stale in the V5 summary?

## Short Answer

The summary was right that V5 was a real collapse and that exact-zero max
reward is diagnostic. It was stale on the concrete architectural mechanism:
V5 already had explicit learned 2D positional encoding on the 49 spatial
tokens. The sharper V5 failure was that the `coarse_to_fine` fine stage could
not see sub-cell screen content, so it learned a static fine-offset prior.

In deterministic V5 stage-0 eval, the policy used 7 different coarse cells but
the fine index was exactly `10` for all 1,099 Smart clicks. That is much more
specific than "coarse-to-fine is hard" or "no positional embeddings."

## Hard Evidence

V5 artifact:

- Run: `banana_smart_v5_b2048_e4_a10`
- Protocol/schema: v3 / `stream_action_effect_feedback_v2`
- Reward implementation: `defeat_roaches_v4`
- Episodes: 11,447
- PPO updates: 672
- Max reward: `0.00`
- Final-100 average: `-49.76`
- Eval rows: none
- Late instability: 7 non-finite grad-norm updates and 16 skipped optimizer
  steps in the late tail

V5 effective config already had:

- `spatial_head_type: "coarse_to_fine"`
- `spatial_positional_encoding: "learned_xy_mlp"`
- `spatial_tokens: 49`
- `amp_dtype: "torch.float16"`
- no `fine_skip_connection`

Stage-0 deterministic diagnostic:

- 1,099 / 1,104 dispatched actions were `Smart_screen` (`99.5%`)
- 1,094 / 1,099 Smart clicks landed on two bottom-left coordinates:
  `(10,72)` and `(22,72)`
- coarse cells used: 7 / 49
- fine sub-indices used: 1 / 144
- fine sub-index: constant `10` for all 1,099 Smart clicks
- target-near-enemy rate: `0 / 1090`
- enemy-health-drop rate: `56 / 1090` (`5.1%`)

Source files:

- `analysis_results/banana_smart_v5_b2048_e4_a10/stage0_analysis.py`
- `analysis_results/banana_smart_v5_b2048_e4_a10/instability_report.txt`
- `models/banana_smart_v5_b2048_e4_a10/effective_config.json`

## What The Summary Got Right

- V5 collapsed hard. `0.00` max reward over 11k+ episodes is not a normal
  slow-learning curve.
- V5 is a regression relative to earlier headline reward runs.
- The issue is not just a hyperparameter nit; the action-target path was not
  producing useful target clicks.
- The dorsal/ventral language is a reasonable high-level metaphor: the policy
  had "what/context" tokens, but the visuomotor click pathway did not have
  enough spatially faithful sub-cell evidence.

## What Was Wrong Or Stale

- Missing positional embeddings were not the V5 concrete bug. V5 already used
  `learned_xy_mlp` positional encoding.
- The missing-dorsal-stream diagnosis was too broad. The concrete V5 symptom
  was fine-stage spatial blindness inside `CoarseToFineTargetHead`.
- The "branch from kiting baseline" line is historical CNN/PPO-era context, not
  a clean current SNN/V5 control. Keep it in archive, not in current planning.
- V5 is not `RewardFunctionV5`. It is a run-family name using
  `defeat_roaches_v4`.
- "No eval rows" was partly structural for that run family; Ray eval/best-save
  plumbing was repaired later.

## What Changed After V5

Commit `5f05381` (`Solved blindness`) added the stage-0 diagnostic and the fine
skip connection:

- `PolicyNetwork` taps pre-pool 84x84 conv2 features.
- The target head receives those as `fine_features`.
- `CoarseToFineTargetHead` adds per-pixel fine skip scores to the fine MLP
  logits.
- With the flag off, old checkpoints load as before.
- With the flag on, new parameters make old no-skip checkpoints incompatible.

Commit `657e2c9` then repaired surrounding confounds:

- Ray eval and best-checkpoint plumbing.
- Extractor normalizer sync before best-save.
- Reward kill credit and terminal/timeout semantics.
- Corrected kiting-distance scale.
- Smart outcome reward signals.
- bf16 AMP default instead of fp16 GradScaler.

V6 (`banana_glasses_v6_b2048_e4_a10`) is therefore not a single-variable proof,
but it is strong evidence that V5's zero-reward collapse was not simply "action
effect feedback is poison." V6 reached max reward `555.85` and positive
training reward, while deterministic eval still needs trace-level scrutiny.

## Current Interpretation

The V5 collapse was a stack of issues, but the most diagnostic one is:

```text
coarse head could choose cells, fine head could not localize inside them
```

That produced repeated bottom-left clicks, no target-near-enemy feedback, and
almost no real damage. The current code addresses that with the fine skip
connection, then stabilizes the surrounding training/eval/reward plumbing.

## Recommendation

- Do not continue debugging V5 as if it were the current architecture.
- Use V5 as a regression artifact and lesson.
- Use V6/V7-style runs for current behavior checks.
- Compare deterministic and stochastic click quality with eval traces before
  adding action-space complexity.
- Keep old CNN/PPO kiting narratives archived unless explicitly doing old-run
  archaeology.
