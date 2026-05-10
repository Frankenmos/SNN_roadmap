# Help Needed

Updated: 2026-05-10

This note is the practical follow-up to the semantic-action /
stream-action-feedback and `coarse_to_fine` migration.
It lists what I can keep doing alone, what I need from you, and what
would make the next round of fixes much easier or safer to land.

## What I Can Keep Doing Alone

- tighten protocol contracts and add unit / integration tests that do not
  depend on a live SC2 environment
- refactor trainer / policy / analysis code when the intended semantics are
  already clear from the repo
- tighten diagnostics around the current `coarse_to_fine` spatial head and
  compare against `token_pointer` when we need a lower-cost baseline
- improve docs, logging, and validation so future changes are easier to trust
- implement low-risk guardrails when the tests and current architecture make
  the intent obvious

## What I Need From You

- sanity-check the timeout-as-truncation behavior in a live run:
  `steps_per_episode` is currently treated as a rollout/episode cap, not a
  true terminal state for PPO bootstrap
- confirm whether keeping `LEFT_CLICK` masked off is still the right call for
  the current DefeatRoaches wrapper, or whether you want a real distinct env
  mapping before we grow the action space further
- run one short env-backed pass on the current semantic-action /
  `coarse_to_fine` branch so we can see whether clicks look sane in practice
- run env-backed verification after reward / terminal changes:
  the remaining important fixes now depend more on real game behavior than on
  pure code correctness
- confirm what outcome signal we trust most for DefeatRoaches:
  wrapper-derived enemy/friendly state, raw `obs.last()`, raw reward, or some
  combination
- sanity-check whether the current `RIGHT_CLICK` semantics are learning the
  behavior you actually want, especially under deterministic eval
- review whether the V5 action-effect feedback protocol helped or hurt; the
  latest local V5 artifact regressed badly despite using the same V4 reward
  implementation

## What I Need You To Test

- latest artifact to inspect:
  `banana_smart_v5_b2048_e4_a10`
- compare it directly against `banana_smart_v4_b2048_e4_a10`; V5 is protocol 3
  / action-effect feedback v2, while V4 used the older 9-dim feedback-token
  protocol
- for the comparison, prioritize action mix and click quality before reward:
  Smart frequency, no-op frequency while enemies are visible, and whether
  clicks land near roaches
- inspect the V5 late-update instability:
  non-finite gradients, skipped optimizer steps, high clip fraction, and any
  correlation with action/effect feedback counters
- a small live run on the current `coarse_to_fine` head
- an optional before/after comparison against `token_pointer` if the current
  run looks worse or ambiguous
- a smoke run that reaches `steps_per_episode` to confirm truncation/reset
  behavior still looks sane
- deterministic and stochastic eval on the same checkpoint after reward-path
  changes
- at least one trace-capture run with the current diagnostics if behavior still
  looks wrong:
  `python eval.py --run_name <run> --best --episodes 5 --inspect --inspect_policy_input --inspect_actions`
- if possible, one short training run where you watch whether:
  `coarse_to_fine` clicks look sane,
  mid-episode PPO flushes still behave normally,
  and masked-left-click semantics do not create surprising no-op patterns

## Good Joint Work For Us

- verify timeout-as-truncation behavior with real traces before changing PPO
  bootstrap logic again
- review reward semantics together after you inspect one or two real traces
- decide whether the next branch after current spatial-head verification is:
  reward-only stabilization,
  Ray throughput / smoke hardening,
  or action-history tokens
- decide what counts as "good enough" for this branch:
  deterministic win rate,
  mean reward,
  action mix,
  or a combination

## Tooling That Would Help Me

- any repeatable env-backed smoke command you trust
- a lightweight way for you to hand me env observations / traces when a run
  looks wrong but I cannot execute the environment directly
- diagnostic-output hygiene for eval:
  deterministic and stochastic eval should write distinct JSONL files by
  default, or at least warn before appending to the same
  `*_diagnostics.jsonl` paths, because mixed-mode traces are easy to misread
- a short note about which diagnostics you consider most trustworthy:
  DB metrics, eval traces, observation JSONL, or manual replay inspection
- if you want me to move faster on experimental branches:
  a preferred naming/versioning convention for new checkpoint families after
  intentionally incompatible policy migrations

## Highest-Value Next Inputs

1. one env-backed verification pass on the current semantic-action /
   `coarse_to_fine` branch
2. your preferred trusted outcome signal for reward / terminal cleanup
3. your preference for the next focus:
   reward semantics,
   Ray smoke / throughput,
   or action-history tokens

## Current Blockers I Intentionally Left Alone

- env-truth validation for reward and terminal outcome logic
- whether `LEFT_CLICK` should stay masked or gain a real env mapping
- live validation of the current `coarse_to_fine` spatial head
- any larger change that would alter checkpoint compatibility again without a
  clearly chosen new branch goal
