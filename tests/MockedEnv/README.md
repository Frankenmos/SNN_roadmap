# MockedEnv

Shared test helpers and real-env diagnostics for the SC2 integration edge.

## What lives here

- `fake_pysc2.py`: lightweight fake `pysc2` install plus mock observation builder.
- `fixtures.py`: shared pytest fixtures (`fake_actions`, `make_obs`).
- `policy_batch.py`: reusable `PolicyInputBatch` test factories.

## Real-env diagnostic flow

The real SC2 checks still belong to the live environment, not the fake one.
Use the `Utility/policy_input_diagnostics_wrapper.py` wrapper via:

```powershell
python eval.py --run_name <run_name> --best --episodes 5 --inspect_policy_input --policy_input_output analysis_results\<run_name>\policy_input_diagnostics.jsonl
```

That log is meant to answer the questions the local sandbox cannot:

- Are `available_actions` IDs what we expect in-game?
- Are `last_actions` populated the way we modeled them?
- Do `multi_select` / `single_select` produce the expected selection tokens?
- Are entity/selection counts and truncation margins sane during real fights?
- Does the extractor emit the masks and meta fields we expect on live timesteps?
