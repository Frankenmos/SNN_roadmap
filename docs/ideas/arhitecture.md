# arhitecture.md

Intentional filename spelling: this is the map-room companion to
`RAYPLAN.md`.

This document is the architecture picture, not the implementation plan.
`RAYPLAN.md` says what to build and in what order. This file shows where
the pieces live, what moves between them, and where the risky boundaries are.

The target is still:

- custom Ray layer
- synchronous actor-learner PPO
- one learner, many rollout actors
- dedicated eval actor
- single distributed logger
- resolved bridge-token / action-history contract treated as frozen protocol

---

## 1. Whole-System Map

```text
                         SNN / PPO / SC2 DISTRIBUTED ARCHITECTURE

  +-----------------------------------------------------------------------------------+
  |                              HUMAN WATCH PATH                                     |
  |                                                                                   |
  |  Windows visual SC2 distribution                                                  |
  |  PySC2 / DeepMind feature-layer viewer                                            |
  |  nice wrapper for watching spatial segmentation, actions, traces                  |
  |                                                                                   |
  |        +------------------------+        +--------------------------------+        |
  |        |  visual SC2Env          | -----> |  inspection / diagnostics      |        |
  |        |  visualize=True         |        |  feature layers + action view  |        |
  |        +------------------------+        +--------------------------------+        |
  |                                                                                   |
  |  Purpose: understand behavior. Not the first Ray scaling runtime.                 |
  +-----------------------------------------------------------------------------------+


  +-----------------------------------------------------------------------------------+
  |                              RAY TRAINING PATH                                    |
  |                                                                                   |
  |  One runtime family for v1: same OS, same Python env, same filesystem root,       |
  |  same SC2 launch profile. Likely Linux/headless for rollout collection.           |
  |                                                                                   |
  |                                                                                   |
  |          weights V                                            metrics/events      |
  |     +----------------+                                      +----------------+     |
  |     | Ray object     | <----------------------------------- | Learner        |     |
  |     | store          |                                      | Coordinator    |     |
  |     +-------^--------+                                      +---+------+-----+     |
  |             |                                               ^   |      |           |
  |             | pull weights                                  |   |      | checkpoint|
  |             |                                               |   |      v           |
  |   +---------+------------------------------+                |   | +----------+     |
  |   |                                        |                |   | | models/  |     |
  |   |                                        | fragments      |   | | ckpts    |     |
  |   v                                        |                |   | +----------+     |
  | +------------------+  +------------------+ | +------------------+                 |
  | | RolloutActor 0   |  | RolloutActor 1   | | | RolloutActor N   |                 |
  | | SC2Env           |  | SC2Env           | | | SC2Env           |                 |
  | | Policy replica   |  | Policy replica   | | | Policy replica   |                 |
  | | local SNN state  |  | local SNN state  | | | local SNN state  |                 |
  | | reward state     |  | reward state     | | | reward state     |                 |
  | +--------+---------+  +--------+---------+ | +--------+---------+                 |
  |          |                     |           |          |                           |
  |          +----------+----------+-----------+----------+                           |
  |                     |                                                         logs|
  |                     v                                                             |
  |               +----------------+                         +-------------------+    |
  |               | RolloutFragment| ----------------------> | LoggerActor       |    |
  |               | batch payload  |     summaries/events    | single SQLite     |    |
  |               +----------------+                         | write path        |    |
  |                                                        +-+-------------------+    |
  |                                                        | training_logs.db         |
  |                                                        +--------------------------+
  |                                                                                   |
  |               +----------------+                                                  |
  | weights V --->| EvalActor      |---- eval summaries ----> LoggerActor             |
  |               | deterministic  |---- best score --------> LearnerCoordinator      |
  |               +----------------+                                                  |
  +-----------------------------------------------------------------------------------+
```

The main idea: Ray distributes environment stepping, not optimization.
Only the learner mutates optimizer state. Actors are fast, disposable
experience machines.

---

## 2. One Update Tick

The first distributed version should move in visible beats:

```text
                         ONE SYNCHRONOUS PPO UPDATE

       learner policy version V
                |
                v
      +---------------------+
      | publish weights V   |
      +----------+----------+
                 |
          +------+------+
          |             |
          v             v
   +-------------+   +-------------+          +-------------+   +-------------+
   | actor 0     |   | actor 1     |   ....   | actor 2     |   | actor 3     |
   | 512 steps   |   | 512 steps   |          | 512 steps   |   | 512 steps   |
   +------+------+   +------+------+          +------+------+   +------+------+
          |                 |                        |                 |
          +-----------------+-----------+------------+-----------------+
                                      |
                                      v
                         +-------------------------+
                         | 2048 fresh transitions |
                         | 4 RolloutFragments     |
                         +-----------+-------------+
                                     |
                                     v
                         +-------------------------+
                         | GAE per fragment        |
                         | terminal/truncated safe |
                         +-----------+-------------+
                                     |
                                     v
                         +-------------------------+
                         | flatten after returns   |
                         | PPO epochs = 8          |
                         | batch_size = 128        |
                         | TBPTT window = 128      |
                         +-----------+-------------+
                                     |
                                     v
                         +-------------------------+
                         | optimizer step(s)       |
                         | policy version V + 1    |
                         +-------------------------+
```

Current config gravity:

- `rollout_steps: 2048`
- `num_rollout_actors: 4`
- recommended `fragment_steps: 512`

That gives the clean first shape:

```text
        4 actors x 512 learnable steps = 2048-step PPO batch
```

Using `fragment_steps: 256` is valid, but it means two actor collection
waves per update. That may be useful later, but it is not the simplest
first diagram.

---

## 3. Learner Core

```text
                          LEARNER COORDINATOR

 +---------------------------------------------------------------------------------+
 | owns mutable training state                                                     |
 |                                                                                 |
 |  +---------------------+       +---------------------+       +----------------+ |
 |  | master PolicyNetwork|       | PPO optimizer       |       | LR scheduler   | |
 |  | SNN + attention     |       | clip, entropy, KL   |       | update-count   | |
 |  | coarse_to_fine head |       | critic loss         |       | based          | |
 |  +----------+----------+       +----------+----------+       +-------+--------+ |
 |             |                             |                          |          |
 |             +-----------------------------+--------------------------+          |
 |                                           |                                     |
 |                                           v                                     |
 |                              +-------------------------+                        |
 |                              | fragment aggregation    |                        |
 |                              | GAE per fragment        |                        |
 |                              | PPO replay with TBPTT   |                        |
 |                              +------------+------------+                        |
 |                                           |                                     |
 |         +--------------------+------------+-------------+----------------+      |
 |         |                    |                          |                |      |
 |         v                    v                          v                v      |
 | +---------------+    +---------------+          +---------------+ +-----------+ |
 | | WeightSnapshot|    | UpdateSummary |          | Checkpoint    | | Eval gate | |
 | | policy_version|    | loss/entropy  |          | best/current  | | cadence   | |
 | +---------------+    +---------------+          +---------------+ +-----------+ |
 +---------------------------------------------------------------------------------+
```

Learner owns:

- policy weights
- optimizer state
- scheduler state
- global update index
- global environment step count
- checkpoint save/load
- best eval promotion
- PPO update semantics
- GAE/returns

Learner does not own:

- live SC2 environments
- actor-local recurrent state
- actor-local bridge/action-history buffers
- direct SQLite writes from many workers

---

## 4. Rollout Actor Interior

```text
                              ROLLOUT ACTOR

 +---------------------------------------------------------------------------------+
 | one actor process                                                               |
 |                                                                                 |
 |    +-------------+       +--------------------+       +----------------------+  |
 |    | SC2Env      | ----> | ObservationExtractor| ---> | PolicyInputBatch     |  |
 |    | DefeatRoaches      | feature screen       |      | spatial_obs          |  |
 |    | step/reset  |      | entities/selection   |      | entity tokens        |  |
 |    +------+------+      | meta + bridge        |      | selection tokens     |  |
 |           ^             +----------+-----------+      | meta_vec             |  |
 |           |                        |                  +----------+-----------+  |
 |           |                        v                             |              |
 |           |             +----------------------+                 |              |
 |           |             | local protocol state |                 |              |
 |           |             | snn_state            |                 |              |
 |           |             | bridge/action history|                 |              |
 |           |             | reward state         |                 |              |
 |           |             +----------+-----------+                 |              |
 |           |                        |                             v              |
 |           |                        |                  +----------------------+  |
 |           |                        +----------------> | policy replica       |  |
 |           |                                           | inference only       |  |
 |           |                                           +----------+-----------+  |
 |           |                                                      |              |
 |           |                                                      v              |
 |           |                                           +----------------------+  |
 |           +------------------------------------------ | ActionSample         |  |
 |                                                       | action_id            |  |
 |                                                       | x/y                  |  |
 |                                                       | target/coarse/fine   |  |
 |                                                       | log_prob/value       |  |
 |                                                       | next_snn_state       |  |
 |                                                       +----------------------+  |
 |                                                                                 |
 +---------------------------------------------------------------------------------+
```

The actor is allowed to know how to act in SC2. It is not allowed to
optimize the policy.

Actor state is local and disposable:

- SC2 environment
- observation extractor
- reward function state
- current SNN state
- resolved bridge/action-history state
- episode lifecycle

If an actor dies, the learner should respawn it and send the latest
weights. No checkpoint should depend on that actor's internal buffers.

---

## 5. RolloutFragment Contract

The fragment is the bridge from local trajectory collection to learner
optimization.

```text
                         ROLLOUT FRAGMENT

 +---------------------------------------------------------------------------------+
 | identity                                                                        |
 |   actor_id                                                                      |
 |   fragment_id                                                                   |
 |   policy_version                                                                |
 |   policy_input_version                                                          |
 |   bridge_schema_version                                                         |
 |                                                                                 |
 | policy inputs, length T                                                         |
 |   spatial_obs             [T, 27, 84, 84]                                       |
 |   entity_features         [T, 24, F_entity]                                     |
 |   entity_mask             [T, 24]                                               |
 |   selection_features      [T, 20, 7]                                            |
 |   selection_mask          [T, 20]                                               |
 |   meta_vec                [T, 19] or resolved future width                      |
 |   pre_step_snn_state      state used during action selection                    |
 |                                                                                 |
 | action / PPO data                                                               |
 |   actions                 [T]                                                   |
 |   move_x, move_y          [T] decoded click logs                                |
 |   target_index            [T] token-pointer target or -1                        |
 |   coarse_index            [T] coarse_to_fine coarse cell or -1                  |
 |   fine_index              [T] coarse_to_fine local cell or -1                   |
 |   log_probs               [T] old policy log-prob                               |
 |   values                  [T] old value estimate                                |
 |   rewards                 [T] shaped reward                                     |
 |   dones                   [T] terminal/truncation marker                        |
 |                                                                                 |
 | bootstrap tail                                                                  |
 |   tail_next_policy_input                                                        |
 |   tail_next_snn_state                                                           |
 |   terminated / truncated                                                        |
 |                                                                                 |
 | summaries                                                                       |
 |   episode summaries                                                             |
 |   reward component summaries                                                    |
 |   step counters                                                                 |
 +---------------------------------------------------------------------------------+
```

Do not ship:

- raw PySC2 observation objects
- live function objects
- optimizer state
- mutable actor buffers
- one global `final_next` pretending to cover many fragments

---

## 6. Data Structure Reality Check

The current local code uses a healthy mix of nice protocol objects and
very Pythonic rollout storage.

```text
                     CURRENT LOCAL DATA STRUCTURES

  +-----------------------+      +---------------------------+
  | PolicyInputBatch      |      | ActionSample              |
  | dataclass, slots=True |      | dataclass, slots=True     |
  | tensors + validation  |      | scalar acting result      |
  +-----------+-----------+      +-------------+-------------+
              |                                |
              | stored each step               | unpacked each step
              v                                v
  +-------------------------------------------------------------------+
  | PPO.memory                                                        |
  |                                                                   |
  | list[dict]                                                        |
  |   {                                                               |
  |     "observation_batch": PolicyInputBatch with pre-step state,    |
  |     "action": tensor, "move_x": tensor, "move_y": tensor,         |
  |     "target_index": tensor, "coarse_index": tensor,               |
  |     "fine_index": tensor, "log_prob": tensor,                     |
  |     "reward": tensor, "value": tensor, "done": tensor,            |
  |     "sample_mask": tensor                                         |
  |   }                                                               |
  +-----------------------------+-------------------------------------+
                                |
                                | update_policy()
                                v
  +-------------------------------------------------------------------+
  | stack many tiny tensors                                            |
  | build list[dict] TBPTT chunks                                      |
  | pack chunk groups into dense [T, group, ...] tensors               |
  | replay through policy                                              |
  +-------------------------------------------------------------------+
```

This is charmingly honest and good for local iteration. For Ray, it is
also the part that will start sending invoices: too many small Python
objects, many tiny tensors, and late packing.

The distributed transport should therefore be more columnar:

```text
                         RAY-FRIENDLY FRAGMENT

      not this:                         prefer this:

  [dict, dict, dict, ...]       +----------------------------------+
  tiny tensors per step         | RolloutFragment dataclass        |
  Python object confetti        | one object ref per fragment      |
                                | dense tensors by field           |
                                | metadata as plain small values   |
                                +----------------------------------+
```

Rough payload pressure from the current config:

```text
  spatial_obs per step:
      27 * 84 * 84 float32 = 190,512 floats
                           = 762,048 bytes
                           ~= 0.73 MiB

  512-step fragment spatial payload:
      ~= 372 MiB before entity/meta/state/action tensors

  2048-step global PPO batch spatial payload:
      ~= 1.45 GiB before overhead

  one SNN state snapshot:
      syn + mem, 2 pathways, 94 tokens, 64 dims
      ~= 94 KiB float32 per transition if stored every step
```

So the first correct Ray version can be dense float32 and simple, but
the first fast Ray version needs an explicit throughput plan. See
`THROUGHPUT_PLAN.md`.

---

## 7. Policy Input Shape

This is the current policy-input beast from `config.yaml`:

```text
                             POLICY INPUT PIPELINE

    +-------------------+      +----------------------+      +------------------+
    | spatial screen    |      | entity features      |      | selected units   |
    | [27, 84, 84]      |      | [24, F_entity]       |      | [20, 7]          |
    +---------+---------+      +----------+-----------+      +--------+---------+
              |                           |                           |
              v                           v                           v
    +-------------------+      +----------------------+      +------------------+
    | CNN + pool        |      | entity encoder       |      | selection encoder|
    | 7 x 7 tokens      |      | masked tokens        |      | masked tokens    |
    +---------+---------+      +----------+-----------+      +--------+---------+
              |                           |                           |
              +---------------------------+---------------------------+
                                          |
                                          v
                              +-----------------------+
                              | meta_vec [19]         |
                              | player stats          |
                              | availability          |
                              | pysc2 last action     |
                              | bridge/action input   |
                              +-----------+-----------+
                                          |
                                          v
             +----------------------------------------------------------+
             | concat tokens + type/position encoding                   |
             +----------------------------+-----------------------------+
                                          |
                                          v
             +----------------------------------------------------------+
             | attention block                                            |
             +----------------------------+-----------------------------+
                                          |
                                          v
             +----------------------------------------------------------+
             | dual-timescale token SNN                                  |
             | fast alpha/beta = 0.55 / 0.65                             |
             | slow alpha/beta = 0.92 / 0.97                             |
             +----------------------------+-----------------------------+
                                          |
                                          v
             +----------------------------------------------------------+
             | shared latent                                             |
             | action head: action_dim = 3                               |
             | spatial head: coarse_to_fine, 7 coarse x 12 local         |
             | value head                                                |
             +----------------------------------------------------------+
```

Important point: the fragment schema should preserve the exact
`PolicyInputBatch` tensors used when the actor selected the action.
The learner should not rebuild bridge tokens or reinterpret action
history later.

---

## 8. Bridge / Action-History Boundary

The bridge-token work should be resolved before the Ray branch starts.
After that, Ray treats the result as a versioned input contract.

```text
       local actor time                                      learner replay time

 +-------------------------+                           +-------------------------+
 | action executed in SC2  |                           | stored tensor replay    |
 +-----------+-------------+                           +------------+------------+
             |                                                      ^
             v                                                      |
 +-------------------------+         serialized in fragment          |
 | bridge/action-history   | ---------------------------------------+
 | buffer updates locally  |
 +-----------+-------------+
             |
             v
 +-------------------------+
 | next PolicyInputBatch   |
 | includes resolved input |
 +-------------------------+
```

Rules:

- actors maintain the mutable bridge/action-history state
- fragments store the resulting tensors and schema version
- learner validates versions and trains on stored tensors
- checkpoints record policy-input and bridge schema versions
- distributed code does not invent new action semantics

---

## 9. Runtime Profiles

Visual and headless are not just vibes. They are SC2 distribution and
launch profiles.

```text
                    SC2 RUNTIME PROFILE SPLIT

 +--------------------------------------+     +----------------------------------+
 | Windows visual profile               |     | Linux/headless profile           |
 |                                      |     |                                  |
 | visualize=True                       |     | rollout collection               |
 | PySC2 feature-layer viewer           |     | Ray actors                       |
 | nice spatial segmentation wrapper    |     | no human watch requirement       |
 | manual inspection                    |     | throughput and stability         |
 +------------------+-------------------+     +----------------+-----------------+
                    |                                      |
                    v                                      v
          behavior debugging                      distributed training
```

For v1 Ray, pick one runtime profile and stay inside it. Mixed
Windows/Linux distributed execution is a later problem, not the first
milestone.

---

## 10. Logging And Checkpoints

```text
                         EVENT AND STATE OWNERSHIP

    rollout actors             eval actor                 learner
         |                         |                         |
         | step summaries          | eval summaries           | update summaries
         | reward components       | deterministic scores      | checkpoints
         v                         v                         v
  +--------------------------------------------------------------------+
  |                            LoggerActor                             |
  |                                                                    |
  |  one SQLite connection                                             |
  |  actor_id                                                          |
  |  policy_version                                                    |
  |  fragment_id                                                       |
  |  global_update_index                                               |
  |  bridge_schema_version                                             |
  +-------------------------------+------------------------------------+
                                  |
                                  v
                         +------------------+
                         | training_logs.db |
                         +------------------+


                         +------------------+
                         | checkpoint.pth   |
                         | best_checkpoint  |
                         +---------+--------+
                                   ^
                                   |
                            learner only
```

The logger serializes events. The learner serializes training state.
Actors should be easy to kill and recreate.

---

## 11. Ownership Table

```text
 +----------------------+-------------------------+-------------------------------+
 | component            | owns                    | must not own                  |
 +----------------------+-------------------------+-------------------------------+
 | LearnerCoordinator   | master policy           | live SC2 envs                 |
 |                      | optimizer/scheduler     | actor-local buffers           |
 |                      | GAE/returns             | many SQLite write handles     |
 |                      | checkpoints             |                               |
 +----------------------+-------------------------+-------------------------------+
 | RolloutActor         | one SC2 env             | optimizer state               |
 |                      | policy replica          | checkpoint authority          |
 |                      | local SNN state         | global update counter         |
 |                      | bridge/action history   |                               |
 +----------------------+-------------------------+-------------------------------+
 | EvalActor            | eval env                | training fragments            |
 |                      | deterministic policy    | optimizer state               |
 +----------------------+-------------------------+-------------------------------+
 | LoggerActor          | SQLite connection       | policy training state         |
 |                      | event ordering          | SC2 envs                      |
 +----------------------+-------------------------+-------------------------------+
```

---

## 12. File Map For The First Ray Branch

```text
 repo root
 |
 +-- distributed/
 |   |
 |   +-- protocol.py          RolloutFragment, TransitionRecord, summaries
 |   +-- local_actor.py       non-Ray actor shell for local tests
 |   +-- ray_actor.py         Ray RolloutActor / EvalActor wrappers
 |   +-- learner.py           LearnerCoordinator
 |   +-- logger_actor.py      single distributed writer
 |   +-- ray_train.py         entrypoint
 |
 +-- agent_core/
 |   |
 |   +-- ppo_trainer.py       consume fragments, GAE per fragment
 |   +-- policy_protocol.py   policy input and version constants
 |
 +-- Utility/
 |   |
 |   +-- config.py            explicit path-based config loading
 |   +-- log_schema.py        optional schema ownership split
 |
 +-- envs/
 |   |
 |   +-- setup_env.py         config-driven map/screen/runtime profile
 |
 +-- docs/ideas/
     |
     +-- RAYPLAN.md          staged build order
     +-- arhitecture.md      this map
     +-- THROUGHPUT_PLAN.md  payload, memory, and speed tuning
```

---

## 13. Non-Negotiables

```text
  NO:  N trainers each stepping their own optimizer
  YES: 1 learner, N rollout actors

  NO:  concatenate fragments then pretend one final_next exists
  YES: GAE per fragment, flatten after returns are valid

  NO:  rebuild bridge/action-history inputs on the learner
  YES: store exact tensors used while acting

  NO:  visual/headless mixed runtime as the first distributed milestone
  YES: one runtime profile for Ray, visual wrapper kept for inspection

  NO:  many workers writing SQLite
  YES: one LoggerActor

  NO:  async streaming PPO before off-policy correction exists
  YES: synchronous policy-version boundary first
```

If this picture stays true while the code changes, the distributed branch
should remain debuggable.
