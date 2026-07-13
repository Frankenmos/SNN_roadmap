# Experiment identity, truthful logging, and Model Git

This is deliberately not a second experiment database. The artifacts have
different jobs:

| Artifact | Job | Mutation rule |
|---|---|---|
| `run_manifest.json` | Birth certificate: first resolved config, source tree, runtime, launch | Created once, never rewritten |
| `resume_events.jsonl` | One launch/resume phase per row | Append only |
| `training_logs.db` | Measurements from episodes, updates, and evals | SQLite stream with additive migrations |
| `checkpoint.pth` | Exact operational resume point, including optimizer state | Mutable by design |
| `model_git/objects/<sha256>.pth` | Small policy/extractor history for scientific comparison | Immutable/content addressed |
| `model_git/index.jsonl` | Parent chain plus run/phase/config/source identity | Append only and hash chained |

The manifest answers “what experiment did I intend to run?” SQLite answers
“what happened?” Model Git answers “how did the learned representation move?”

## Truthful logging

Every launch gets a monotonically increasing `phase_id`; episodes, PPO updates,
aggregate evals, and individual eval episodes carry it. Existing columns remain
in place for old dashboards.

- `episodes.total_reward` remains the shaped reward compatibility field.
- `episodes.shaped_reward` says that explicitly.
- `episodes.native_reward` is the environment/SC2 reward.
- termination vs time-limit truncation, action counts, and an episode component
  JSON summary are retained.
- `eval_runs` remains the aggregate table; `eval_episodes` holds every raw eval
  episode so standard deviation is no longer reconstructed from actor means.
- PPO loss, KL, entropy, and clip fraction are sample-weighted across uneven
  TBPTT groups. `epochs_ran` is the actual count after KL early stopping.
- `*_update_start` ratio/KL fields capture the first pre-optimizer TBPTT group;
  `update_start_scope` and `update_start_sample_count` make that sampling scope
  explicit. They are an early trust-region diagnostic, not a claim that every
  later minibatch had that same distribution.
- SIL rows include admission reason counts, sampled entry age, gate-weight
  quantiles, and trunk/actor/target gradient decomposition.

The dashboard reads columns defensively with `PRAGMA table_info`, so legacy DBs
remain usable. New individual eval rows appear under “Individual evaluation
episodes.”

## Model Git

The default cadence is updates 5, 10, …, 200, then every 25 updates. A snapshot
contains policy weights, extractor state, and the resolved policy constructor
configuration, but no optimizer/scheduler state.
Training still resumes only from `checkpoint.pth`.

```powershell
python -m tools.registry list <run>
python -m tools.registry show <run>:u10
python -m tools.registry diff <run>:u5 <run>:u10
python -m tools.registry verify <run>
```

Refs can use `u<N>`, `sha/<unique-prefix>`, `tag/<name>`, `latest`,
`checkpoint`, or `best`.

```powershell
python -m tools.registry tag <run> promising-u10 <run>:u10
python -m tools.registry show <run>:tag/promising-u10
```

Tags are immutable. To declare that a new run begins from another object, set
its ancestry before its first snapshot:

```powershell
python -m tools.registry fork <child_run> <parent_run>:u10
```

`snapshots/policy_u<N>.pth` remains as a write-once compatibility ref. If a
resume/fork produces different content at the same update number, the old ref
is preserved and both objects remain in the append-only Model Git index.

## Representation migration

The checked-in `probes/tiny_skirmish_v1.json` is the deterministic, versioned
probe specification and expected tensor digest. Generate its ignored `.pt`
tensor artifact locally; this avoids putting a 24 MiB derived binary in Git:

```powershell
python -m tools.representation_migration probe-create
```

The resulting TinySkirmish trajectories permit offline comparisons without
launching SC2 and without retaining every activation.

```powershell
python -m tools.representation_migration compare <run>:u5 <run>:u10 `
  --probe probes/tiny_skirmish_v1.pt `
  --out analysis_results/<run>/representation_u5_u10.json

python -m tools.representation_migration timeline <run> `
  --probe probes/tiny_skirmish_v1.pt `
  --out analysis_results/<run>/representation_timeline.json
```

Reports combine weight deltas with latent linear CKA, latent subspace principal
angles, recurrent synaptic/membrane comparisons, action agreement and
Jensen-Shannon divergence, value distribution/error/correlation, and primary
and secondary spatial-target distribution movement. Raw TinySkirmish probe
features are normalized using each artifact's saved extractor state before
policy replay. Action distributions respect each probe's semantic availability
mask; spatial-target distributions are compared under a fixed `RIGHT_CLICK`
condition so a NO_OP argmax does not make the target comparison meaningless.
Replay keeps only those compact named tensors; it does not dump
the attention stack or every convolutional activation.

New snapshots embed `policy_config`, so their architecture can be rebuilt even
if the run's YAML later changes. Legacy snapshots without that field fall back
to the run's saved `config.yaml` and may fail honestly if the historical
constructor is no longer supported by the current checkout.

These reports measure response migration on a fixed probe distribution. They
do not prove that gameplay improved. A policy can preserve CKA while changing a
critical action boundary, or move representations substantially while keeping
behavior stable.

## Deferred real-SC2 acceptance check

The automated suite covers migrations, TinySkirmish, synthetic policies,
saved `.pth` artifacts, registry ancestry, and corruption detection. The game
installation is intentionally left to the user-facing acceptance pass:

```powershell
python -m distributed.ray_train --num-actors 1 --max-updates 10 --eval-every-updates 5 --eval-episodes 2 --run-name lineage_sc2_smoke
python -m tools.registry verify lineage_sc2_smoke
python -m tools.registry diff lineage_sc2_smoke:u5 lineage_sc2_smoke:u10
python -m tools.representation_migration compare lineage_sc2_smoke:u5 lineage_sc2_smoke:u10 --probe probes/tiny_skirmish_v1.pt --out analysis_results/lineage_sc2_smoke/representation_u5_u10.json
python eval.py --run_name lineage_sc2_smoke --episodes 3 --deterministic --visualize
python -c "import sqlite3; c=sqlite3.connect('models/lineage_sc2_smoke/training_logs.db'); print('episodes', c.execute('select phase_id, shaped_reward, native_reward, terminated, truncated from episodes order by episode_id desc limit 5').fetchall()); print('updates', c.execute('select phase_id, epochs_ran, update_start_scope, update_start_sample_count, kl_update_start, mean_kl, clip_fraction from ppo_updates order by update_id desc limit 5').fetchall()); print('eval groups', c.execute('select eval_group_id, num_episodes from eval_runs').fetchall()); print('eval rows', c.execute('select eval_group_id, count(*) from eval_episodes group by eval_group_id').fetchall())"
```

Expected terminal/artifact evidence:

- training prints `phase=0`, two-episode eval summaries at u5/u10, then
  `Policy snapshot <12 hex> saved` at u5 and u10;
- `verify` prints `OK: 2 objects, 2 index events`;
- `diff` prints a finite global L2 delta and identifies the embedded
  `snapshot.policy_config` as its config source;
- the representation command writes JSON containing `weights`, `latent`,
  `snn_syn`, `snn_mem`, `action`, `value`, `target_primary`, and
  `target_secondary` sections;
- the SQL probe prints non-null phase/shaped/native fields, explicit
  `first_tbptt_group_pre_optimizer` scope, and matching eval aggregate/raw-row
  counts of two for both eval groups.

Acceptance checklist:

1. `run_manifest.json` stays byte-identical across resume; a new phase row is
   appended instead.
2. DB episode rows contain both shaped and native reward; eval aggregate count
   equals the number of matching `eval_episodes` rows.
3. PPO rows contain the true `epochs_ran`, update-start ratio diagnostics, and
   finite weighted KL/clip metrics.
4. `verify` reports `OK`; u5 and u10 are different SHA objects with parent
   linkage and `checkpoint.pth` still resumes normally.
5. The representation JSON is finite and complete, then the visual eval is used
   to decide whether the measured migration was actually beneficial.

Fail the acceptance pass if the registry verifier is not `OK`, a scheduled
object is absent, `checkpoint.pth` cannot resume, required DB fields are null
for newly produced rows, aggregate eval counts disagree with raw episode rows,
the report contains `NaN`, or the visual eval contradicts the interpretation
you planned to draw from the offline metrics. Report back the five command
outputs plus whether the three rendered episodes attacked, idled, or oscillated.
