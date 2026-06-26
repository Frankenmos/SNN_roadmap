# Ray Status

Updated: 2026-06-26

## Current State

The Ray path is a synchronous distributed rollout setup with one learner and
configurable rollout actors:

```powershell
python -m distributed.ray_train --num-actors 10 --run-name <run>
```

Implemented:

- `RolloutActor` collects `RolloutFragment`s through the same protocol used by
  local PPO.
- `Learner` owns policy, optimizer, scheduler, checkpointing, and update count.
- Fragments carry protocol/schema metadata and are rejected if stale.
- Per-fragment GAE and TBPTT replay are shared with the local PPO path.
- Deterministic eval can borrow training actors.
- Before best-checkpoint save, actor extractor normalizer stats are folded back
  into the learner so checkpoints do not ship a count-0 normalizer.
- `best_checkpoint.pth` is now supported on the Ray path after eval.

## Current Config Defaults

```yaml
distributed:
  num_rollout_actors: 10
  fragment_steps: 256
  global_rollout_steps: 2560
  eval_every_updates: 50
  num_eval_actors: 0
```

`num_eval_actors` is reserved for a later dedicated eval pool. Today eval
borrows training actors.

## Remaining Work

- Dedicated EvalActor pool, if borrowed actors become too disruptive.
- Better step-level Ray logging; the Ray path currently prioritizes episode
  summaries and PPO update rows.
- Throughput tuning after reward/behavior stability is no longer the main
  bottleneck.

## Files

- `distributed/ray_train.py`
- `distributed/ray_actor.py`
- `distributed/learner.py`
- `distributed/rollout.py`
- `distributed/protocol.py`
- Tests: `tests/test_ray_eval.py`, `tests/test_normalizer_merge.py`
