# Help Needed

Updated: 2026-04-22

This note is the practical follow-up to the semantic-action /
token-pointer migration.
It lists what I can keep doing alone, what I need from you, and what
would make the next round of fixes much easier or safer to land.

## What I Can Keep Doing Alone

- tighten protocol contracts and add unit / integration tests that do not
  depend on a live SC2 environment
- refactor trainer / policy / analysis code when the intended semantics are
  already clear from the repo
- implement the next spatial-head step once the current token-pointer version
  is confirmed stable enough to build on
- improve docs, logging, and validation so future changes are easier to trust
- implement low-risk guardrails when the tests and current architecture make
  the intent obvious

## What I Need From You

- decide the `steps_per_episode` meaning:
  is it a real task horizon, or only a training truncation?
  this directly decides whether PPO should treat the cap as terminal or keep
  bootstrapping through it
- confirm whether keeping `LEFT_CLICK` masked off is still the right call for
  the current DefeatRoaches wrapper, or whether you want a real distinct env
  mapping before we grow the action space further
- run one short env-backed pass on the new semantic-action branch so we can
  see whether token-pointer clicks are already saner than the old factorized
  head in practice
- run env-backed verification after reward / terminal changes:
  the remaining important fixes now depend more on real game behavior than on
  pure code correctness
- confirm what outcome signal we trust most for DefeatRoaches:
  wrapper-derived enemy/friendly state, raw `obs.last()`, raw reward, or some
  combination
- sanity-check whether the current `RIGHT_CLICK` semantics are learning the
  behavior you actually want, especially under deterministic eval

## What I Need You To Test

- a small before/after run on the new token-pointer head before we move to
  `coarse_to_fine`
- a small before/after run once we touch time-cap semantics
- deterministic and stochastic eval on the same checkpoint after reward-path
  changes
- at least one trace-capture run with the current diagnostics if behavior still
  looks wrong:
  `python eval.py --run_name <run> --best --episodes 5 --inspect --inspect_policy_input --inspect_actions`
- if possible, one short training run where you watch whether:
  token-pointer clicks look sane,
  mid-episode PPO flushes still behave normally,
  and masked-left-click semantics do not create surprising no-op patterns

## Good Joint Work For Us

- define the time-cap semantics explicitly before changing PPO bootstrap logic
- review reward semantics together after you inspect one or two real traces
- decide whether the next branch after token-pointer verification is:
  `coarse_to_fine`,
  reward-only stabilization,
  or action-history tokens
- decide what counts as "good enough" for this branch:
  deterministic win rate,
  mean reward,
  action mix,
  or a combination

## Tooling That Would Help Me

- a repo-safe Git setup for the sandbox user:
  right now `git status` is blocked by a `safe.directory` ownership warning,
  so basic Git inspection from this environment is awkward
- any repeatable env-backed smoke command you trust
- a lightweight way for you to hand me env observations / traces when a run
  looks wrong but I cannot execute the environment directly
- a short note about which diagnostics you consider most trustworthy:
  DB metrics, eval traces, observation JSONL, or manual replay inspection
- if you want me to move faster on experimental branches:
  a preferred naming/versioning convention for new checkpoint families after
  intentionally incompatible policy migrations

## Highest-Value Next Inputs

1. your decision on `steps_per_episode`:
   real horizon or truncation
2. one env-backed verification pass on the new semantic-action / token-pointer branch
3. your preference for the next focus:
   reward semantics,
   `coarse_to_fine`,
   or time-cap semantics

## Current Blockers I Intentionally Left Alone

- time-cap / timeout semantics in PPO bootstrap
- env-truth validation for reward and terminal outcome logic
- whether `LEFT_CLICK` should stay masked or gain a real env mapping
- the `coarse_to_fine` upgrade until token-pointer gets at least one live pass
- any larger change that would alter checkpoint compatibility again without a
  clearly chosen new branch goal
