# SmartOutcomeDetector Follow-Up

> Archived 2026-06-26.
>
> This was a short follow-up checklist. The current code now uses
> `SmartOutcomeDetector` inside `RewardFunctionV4`, fixes the weapon-cooldown
> column lookup, and has dedicated tests. Keep this file only as historical
> context.

Updated: 2026-05-10

## Status

The original SmartOutcomeDetector review blockers have been addressed in the
Windows/source workspace. The detector remains diagnostics-only; it is not fed
back into the policy input.

## Fixed

### PySC2 Smart_screen parsing

`Utility/smart_outcome_diagnostics_wrapper.py`

The wrapper now detects PySC2-style action calls by `function == 451` and
extracts the target from `arguments[1]`, matching `Smart_screen("now", [x, y])`.
It also keeps mapping-style fallback parsing for lightweight tests.

### Previous/current frame timing

`Utility/smart_outcome_diagnostics_wrapper.py`

The wrapper stores per-agent previous frame snapshots and previous raw
`feature_units` from reset/previous step. When a Smart call is dispatched, it
creates the pending click from the pre-action frame before calling `env.step()`,
then resolves against the post-action frame.

### One detector source of truth

`obs_space/smart_outcome_detector.py`

The duplicate wrapper-side classifier was removed. The wrapper now only parses
PySC2 calls, manages frame timing, and logs JSONL. `SmartOutcomeDetector` owns
classification and attribution.

### Feature-unit indices

`obs_space/smart_outcome_detector.py`

Raw feature-unit cooldown extraction now follows the same convention as
`obs_space/action_effects.py`: alliance=1, health=2, weapon_cooldown=8,
x=12, y=13, tag=29. Tests cover raw numeric rows and object-style feature units.

### Cooldown-based `fired_likely`

`obs_space/smart_outcome_detector.py`

`fired_likely` now uses real previous/current friendly weapon cooldown snapshots:
ready/low cooldown followed by a large cooldown increase near an enemy target.
The old placeholder path that always returned false is gone.

### Repeated Smart spam attribution

`obs_space/smart_outcome_detector.py`

A single enemy-health drop is attributed to one most plausible pending Smart
click instead of resolving every pending click as `attack_likely`. Other pending
clicks have their health baseline updated so the same damage event is not
counted again.

### Eval entrypoint

`eval.py`
`envs/setup_env.py`

Smart outcome logging is available behind opt-in eval flags:

```text
--inspect_smart_outcomes
--smart_outcome_output analysis_results/<run>/smart_outcomes.jsonl
--smart_outcome_every 1
--smart_outcome_window 5
```

## Remaining Caveat

The current machine can run lightweight unit tests, but not live SC2/PySC2 eval.
The wrapper parser test skips if PySC2 is unavailable. Final validation still
needs one real eval run in the proper game environment, then a manual read of the
JSONL distribution against replay intuition.

Suggested eval command:

```text
python eval.py --run_name banana_smart_v5_b2048_e4_a10 --best --episodes 5 \
    --inspect_smart_outcomes \
    --smart_outcome_output analysis_results/banana_v5/smart_outcomes.jsonl
```

## Verification

Focused tests were converted to `unittest` so they do not require pytest:

```text
.venv/bin/python -m unittest discover -s tests -p 'test_smart_outcome*.py'
```

Result:

```text
Ran 13 tests in 0.022s
OK (skipped=1)
```

The skipped test is the wrapper import/parser group when PySC2 is not installed
in the lightweight workspace. Syntax compilation of the changed files also
passes.
