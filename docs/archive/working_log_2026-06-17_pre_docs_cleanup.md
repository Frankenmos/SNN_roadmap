# Working Log

> Archived 2026-06-26.
>
> This was the compressed implementation history before the docs cleanup. It is
> useful archaeology, but `docs/current/working_log.md` is now intentionally
> short and only tracks the latest live state.

Compressed current-memory version.

Verbose pre-compression snapshot:
`docs/archive/working_log_2026-04-20_pre_compress.md`

## 2026-06-13 — V7 fix-batch adversarial review (19-agent audit)

Reviewed the user's V7 bug-fix batch (normalizer merge, Ray eval, reward
fixes, bf16, entity sort). 14 findings confirmed, 0 refuted; verifiers
downgraded several severities (calibrated, not rubber-stamp). Batch is
fundamentally sound. ONE real functional bug found + FIXED:

- FIXED: `SmartOutcomeDetector` read the wrong feature_unit column for
  weapon_cooldown — `_FEATURE_UNIT_WEAPON_COOLDOWN = 8` is actually
  `shield_ratio`; the real index is 25 (verified against
  `features.FeatureUnit`). On the numeric production path (feature_units is a
  NamedNumpyArray -> numeric branch) the `fired_likely` outcome read
  shield_ratio (~always 0 for marines) instead of weapon cooldown, so the
  `smart_fired_likely_reward` (0.06) was DEAD in the live V7 reward. The
  `attack_likely` (0.12, keys off enemy health drop) term was unaffected.
  Fix: derive all six column indices from `features.FeatureUnit` in
  smart_outcome_detector.py (mirrors obs_space_2.py), so they can't drift.
  The test helper `raw_unit` in tests/test_smart_outcome_detector.py
  ENCODED THE SAME BUG (wrote cooldown to row[8]), which is why CI was blind
  — fixed it to index by the enum too, so the existing
  `test_fired_likely_uses_real_cooldown_change` now actually guards it.
  Full suite still 170 pass.

Other 13 findings (NOT fixed — latent, known, or design choice):
- Resume-sync heterogeneous-baseline (3 findings, ray_train.py:246-268):
  `_merge_extractor_sync_states` subtracts each actor's OWN baseline but
  re-adds only the first actor's once -> drops/double-counts if baselines
  ever differ. LATENT: cannot fire (baseline broadcast identically once,
  max_restarts=0). User already flagged this as "optional hardening" in the
  V7 note. One-line fix available (subtract against the shared baseline +
  assert equality) if ever enabling fault tolerance / async eval actors.
- Eval std_reward (ray_train.py:311): understates variance when
  eval_episodes > num_actors; EXACT under shipped config (5 eps, 10 actors
  -> 1 ep/actor). Diagnostic-only, doesn't affect best-ckpt selection.
- Eval leaves SmartOutcomeDetector pending clicks undrained (run_eval_sweep
  never calls calculate_reward); bounded per-episode, reset between
  episodes. Really a symptom that ALL reward observe_action work is wasted
  during eval (eval scores native reward). Perf nit, not correctness.
- Per-step clamp on smart-outcome reward squashes simultaneous resolutions
  to [-0.02,0.12] (sum, not per-outcome); bounded, arguably intended.
- fp16 scaler.update() after clip-skip mis-tracks scale — DEAD on bf16
  (scaler disabled); only matters if amp_dtype reverted to fp16.
- Entity-sort NaN poisons determinism; merge_state_dicts lacks the shape
  guard subtract has; a few test-coverage gaps (round-trip, multi-outcome
  clamp, real-class reset pin). All low/nit, defensive.

## 2026-06-11 (later) — architecture deep review (10-agent audit), verified bugs

Full review of the whole experiment (curriculum-readiness / full-game
viability / SNN-vs-ANN). Bugs VERIFIED in code during the review:

1. CRITICAL (eval confound): Ray checkpoints ship a count=0 feature
   normalizer — the learner's extractor never sees observations, and
   `_broadcast_weights(include_extractor_state=False)` never syncs actor
   stats back. `normalize()` is a passthrough below `min_count_for_normalize`
   (obs_space_2.py:133-134), so EVERY deterministic eval ever run on a Ray
   checkpoint fed the policy raw unnormalized entity/selection features it
   never trained on. V5-vs-V6 eval comparisons remain internally consistent
   (same confound both sides) and training-time DB counters are unaffected
   (actor-side normalizers live), but absolute eval numbers are suspect.
2. Reward: the wave-clearing kill is systematically uncredited — when the
   last roach dies and the new wave spawns within one frame, enemy count
   jumps 1->4 and `current < previous` never fires (defeat_roaches_v3.py:184).
   The single most valuable event in the game gives zero reward.
3. Reward: `win_reward` is unreachable (enemy_count==0 at obs.last() never
   happens due to respawn); every episode ends with a flat -30 loss penalty
   (defeat_roaches_v3.py:195-201) regardless of kills.
4. Reward: distance band 7-11px is ~2-3 game units (84px ~ 24 game units)
   = standing inside roach melee; marine max range is ~17px. Probable
   pixels-vs-game-units confusion — the shaping rewards brawling, not kiting.
5. Entity cap: MAX_ENTITY_TOKENS=24 with first-N truncation drops freshly
   spawned roaches exactly when the agent clears a wave.
6. fp16 AMP everywhere on a Blackwell card — the GradScaler/nonfinite-skip
   machinery exists only because of fp16; bf16 would delete it.
7. Replay perf: TBPTT replay re-runs the FULL encoder sequentially per step
   at tiny effective batch; convs/encoders/attention are non-recurrent and
   could be precomputed in one batched pass (only the (syn,mem) token loop
   needs sequencing) — an order-of-magnitude update speedup left on the table.

Review verdicts (full reasoning with the agents): the PPO/TBPTT core and
protocol boundary are solid; the task periphery (action vocab, reward,
availability mask offsets) is frozen to DefeatRoaches by Final constants;
full SC2 is out of reach for ANY substrate on this rig (3 actions vs 573
functions, no minimap/camera path, 24-entity cap, ~34 env-steps/s measured);
at num_steps=1 with attention computed once outside the loop the network has
never actually run as an SNN (the loop is a quantizer — the 'fair SNN test'
of num_steps 8-16 with input inside the loop + rate readouts has never been
performed).

- Stage 0 verification of the fine-stage spatial-blindness diagnosis, using a
  deterministic eval of `banana_smart_v5_b2048_e4_a10` (`checkpoint.pth`;
  `best_checkpoint.pth` was never written for this run):
  - 99.5% of dispatched actions are `Smart_screen`; 1,094 of 1,099 clicks land
    on just two coords, `(10,72)` and `(22,72)` (bottom-left corner exploit)
  - **fine sub-index is the constant `10` for all 1,099 clicks**, across 7
    different coarse cells and 5 episodes — the fine head's argmax does not
    depend on the observation or the chosen cell
  - cross-check on the 2026-05-10 stochastic eval: fine indices cover 139/144
    nearly uniformly (top index `10` at only 2.2%), while coarse concentrates
    58% of mass in bottom-row cells 42/43 — coarse learned a (bad) preference,
    fine stayed at its uniform prior. Exactly the static-prior signature
    predicted by the architecture analysis of `CoarseToFineTargetHead`.
  - ground-truth click quality: joining clicks with sampled enemy positions,
    nearest enemy is >=27px away on every joinable step; 0% within 12px.
    Enemy health drops on only 5.1% of steps (incidental idle auto-attack);
    1 kill across 7 episodes.
- found a diagnostics-only bug while analyzing:
  `Utility/policy_input_diagnostics_wrapper.py` re-extracts observations with
  its own `ObservationExtractor` and never passes `last_action_token`, so
  `bridge_type` / `x_norm` / `y_norm` / `target_near_enemy` /
  `friendly_moved_toward_target` in `policy_input_diagnostics*.jsonl` are
  ALWAYS zero. These fields cannot be trusted in any past dump. The agent's
  real policy input is unaffected (`agent.py` passes the real token).
  `Utility/smart_outcome_diagnostics_wrapper.py` does NOT share the flaw — it
  parses dispatched actions from `env.step()` directly.
- analysis script kept at
  `analysis_results/banana_smart_v5_b2048_e4_a10/stage0_analysis.py`
- conclusion: Stage 0 passed; proceed with Stage 1 (skip connection feeding
  per-pixel conv features to the fine stage).
- implemented Stage 1: fine skip connection for `CoarseToFineTargetHead`,
  config-gated via `model.fine_skip_connection` (+ `fine_skip_dim: 32`):
  - `spiking_policy.encode_step_tensors` taps conv2 output (84×84×32, the
    last full-resolution point before MaxPool) and returns it inside a
    `SpatialContextBundle(tokens, fine_features)` in the spatial_context
    slot; `build_target_head` unwraps it, so the PPO trainer, replay path,
    and other heads are untouched
  - head side: per-pixel keys = `LayerNorm(Linear(32 -> fine_skip_dim))`,
    query from the existing `(latent, action_emb, coarse_token)` fine input,
    scores = `q·k/sqrt(d)` reshaped so cell/fine indices line up 1:1 with
    `encode_xy_to_target` (84 = 7×12); scores are ADDED to the existing
    `fine_mlp` logits (the old path becomes a learned prior)
  - flag OFF (default): byte-identical behavior, no new parameters, old
    checkpoints load. Flag ON: new head parameters, so checkpoints made
    without the flag cannot be loaded — flip `fine_skip_connection: true`
    only when launching the next training run, flip back to eval old runs
  - sampling/evaluate/teacher-forcing semantics unchanged (only the fine
    logit values changed); per-step replay cost adds keys
    `[B, 49, 144, 32]` (~29 MB fp32 at the typical replay group size of
    ~64 rows) — watch for OOM if replay groups degenerate to many tiny
    chunks
  - verification: `pytest tests/test_coarse_to_fine_head.py -q` -> 28 passed
    (new: obs-dependence of fine logits, cell-locality, single-pixel-to-
    fine-index alignment, bundle integration, sample/evaluate consistency);
    full `pytest tests -q` -> 144 passed, 1 failed
    (`test_eval_diagnostic_paths_split_by_mode`, pre-existing Windows
    path-separator issue, unrelated); gradient smoke test: fine-logit loss
    reaches 100% of fine_features elements, all finite
  - note: `tools/analysis/analyze_eval_trace.py` rebuilds the policy without
    passing `spatial_head_type` (pre-existing); its conv-activation probe
    still works for flag-off checkpoints
- prepared the Stage 1 verification run `banana_glasses_v6_b2048_e4_a10`
  ("the fine stage got glasses"):
  - config: flipped `fine_skip_connection: true` and set
    `environment.run_name`; ALL other hyperparameters identical to V5 for a
    clean single-variable A/B (deliberately did not touch `entropy_coef` or
    the right-click curriculum — if V6 still corner-locks with working fine
    vision, entropy is the next single knob)
  - launch: `python -m distributed.ray_train --num-actors 10 --run-name
    banana_glasses_v6_b2048_e4_a10`
  - reminder: to eval pre-V6 checkpoints, flip `fine_skip_connection` back
    to false first (strict state-dict load)
  - confirmed while preparing: the Ray training path never runs
    deterministic eval and never writes `best_checkpoint.pth`
    (`distributed/ray_train.py` only passes `best_eval_reward` through;
    `distributed.num_eval_actors` / `eval_every_updates` are dead config
    keys referenced nowhere) — V5's missing eval_runs rows were structural.
    V6 success checks therefore run through manual `eval.py` passes:
    clicks-near-enemy rate > 0, fine_index obs-dependent (not constant in
    deterministic mode), no bottom-left corner lock.

## 2026-05-10

- reviewed the V5 run family artifact `banana_smart_v5_b2048_e4_a10`:
  - important naming note: V5 is a run-family name, not `RewardFunctionV5`
  - the run sidecar reports `reward.name = "defeat_roaches_v4"`
  - the actual protocol jump was `POLICY_PROTOCOL_VERSION = 3` and
    `stream_action_effect_feedback_v2`
  - local DB spans 2026-05-06 16:18:24 to 2026-05-10 17:01:16 with
    11,447 episodes and 672 PPO updates
  - headline read is bad: max reward `0.00`, final-100 average `-49.76`,
    and late non-finite-gradient / skipped-optimizer warnings
- inspected V5 stochastic eval diagnostics after the live visual read that the
  policy runs to the left-bottom corner and sidesteps there:
  - confirmed dispatch itself is clean: `Smart_screen` is accepted/executed and
    `last_actions` reports function `451`
  - confirmed the semantic failure: accepted `Smart_screen` does not mean the
    click became an attack
  - added `docs/current/action-detection-problem.md` to capture Smart attack
    detection options, reward guardrails, and protocol cautions
- refreshed docs so Claude / future agents do not confuse the V5 run family
  with a nonexistent `defeat_roaches_v5` implementation

## 2026-05-06

- implemented Action Effect Feedback v2:
  - bumped `POLICY_PROTOCOL_VERSION` to 3 and schema to
    `stream_action_effect_feedback_v2`
  - widened `action_feedback_tokens` from `[B, 1, 9]` to `[B, 1, 12]`
  - added effect bits for target-near-enemy, friendly movement toward target,
    enemy health drop, and friendly health drop
  - added rollout counters for policy action mix and Smart effect outcomes
- documented `banana_b2048_e4_a10` as the then-latest pre-V4 baseline run,
  around 5,000 episodes
- updated current docs to match the live config drift:
  `batch_size: 2048`, `num_rollout_actors: 10`, `fragment_steps: 256`, and
  `global_rollout_steps: 2560`
- noted that behavior from the latest run is disappointing and should be
  analyzed before adding more action-space complexity
- prepared the next overnight experiment as `banana_smart_v4_b2048_e4_a10`:
  - added `defeat_roaches_v4`, an action-aware reward layer over V3
  - gives a small bonus for `Smart_screen` clicks near visible enemies
  - gives a small penalty for no-op while enemies and Smart are available
  - added a temporary PPO no-op logit penalty while Smart is available
    (`right_click_curriculum_updates: 120`, penalty starts at `2.0` and
    linearly decays to zero)
  - intent is to fight the observed no-op/passive-autoattack collapse without
    changing the minigame or adding offline pretraining yet
  - verification: `pytest -q` -> `108 passed`

## 2026-04-26

- moved the 9-dim action feedback bridge out of `meta_vec` and into
  `action_feedback_tokens [B, 1, 9]`
- reduced `META_VECTOR_DIM` from 24 to 15:
  `player[11] + semantic_available_actions[3] + pysc2_last_action[1]`
- added `ActionFeedbackEncoder` in `obs_space/action_feedback_encoder.py`
- added action feedback token type embedding and policy token-stream plumbing
- updated TBPTT packing, eval trace serialization, and policy input diagnostics
- checkpoint protocol bumped to `POLICY_PROTOCOL_VERSION = 2`; old checkpoints
  are intentionally incompatible with the stream-token policy

## 2026-04-24

- implemented action-history bridge expansion (commit e964d27)
- expanded `META_VECTOR_DIM` from 19 to 24
- added 5 new feedback fields to meta_vec:
  - `last_any_action_executed`: 1.0 if `obs.last_actions` is non-empty
  - `last_smart_executed`: 1.0 if `451 in obs.last_actions`
  - `score_total_delta`: clipped/normalized delta of `score_cumulative[0]`
  - `killed_value_delta`: clipped/normalized delta of `score_cumulative[5]`
  - `score_penalty_bit`: 1.0 if score delta is negative
- implementation in `obs_space/obs_space_2.py`:
  - `_extract_action_history_vector()` computes the 5 new fields
  - `_score_delta()` tracks previous score for delta computation
  - `update_feedback_state` parameter controls state mutation
  - `peek_observation()` sets `update_feedback_state=False`
  - `reset()` clears `_previous_score_cumulative`
- protocol constants in `agent_core/policy_protocol.py`:
  - `AGENT_ACTION_TOKEN_DIM = 4`
  - `ACTION_HISTORY_DIM = 5`
  - `AGENT_LAST_ACTION_DIM = 9`
  - New offset constants for each action-history field
- added 4 new tests in `tests/test_observation_extractor.py`:
  - empty and smart last-action marking
  - score delta clipping and penalty bits
  - score reset between episodes
  - peek doesn't consume score state
- verification:
  - all 9 observation extractor tests pass
  - all 21 policy/agent tests pass
  - edge cases reviewed (None scores, short arrays, empty actions, reset, peek)
  - full verification report saved to `docs/archive/action_history_bridge_verification_2026-04-24.md`
- documentation:
  - the 24-dim layout is documented historically at
    `docs/archive/action_history_bridge_plan.md`
  - `config.yaml` was updated to `vector_input_dim: 24` for that intermediate
    protocol, before the later stream-token migration reduced it to 15
- checkpoint compatibility: breaking change; old 19-dim checkpoints cannot load

## 2026-04-19

- froze the `PolicyInputBatch` protocol and moved the repo onto the hybrid observation path:
  spatial screen tensor, entity tokens, selection tokens, and `meta_vec`
- rewired `PolicyNetwork` and PPO around tokenized hybrid inputs instead of the old flat vector path
- hardened the observation extractor:
  safer running-stat updates, eval-stat isolation, schema validation, and better tests
- landed Stage-1 TBPTT with ordered chunk replay and helper-step masking semantics
- added packed replay and replay-side fast paths to make TBPTT affordable enough to iterate on
- swapped attention onto SDPA and refreshed logging / dashboard support for the current branch
- verification at the end of that pass:
  `pytest tests -q`
  `40 passed`

## 2026-04-20

- landed Stage-1 action refactor:
  conditioned spatial `MOVE` / `ATTACK`, explicit `Move_screen` / `Attack_screen`, availability masking, and executed-action bridge token plumbing
- moved the reset-only `select_army` step outside PPO memory and removed the old mid-episode helper fallback
- removed the scripted nearest-enemy attack targeting from the learned path
- kept rollout storage stable while making replay condition spatial logits on stored action IDs
- fixed a real PySC2 runtime mismatch where `FunctionCall` does not reliably expose `.name`
- aligned fake test action IDs with the real DefeatRoaches IDs
- fixed analysis-side action decoding for post-refactor runs:
  the analyzer now detects current action semantics from sibling run config and no longer mislabels `action=2` as no-op for `BPTT-1`
- regenerated `analysis_results/BPTT-1/instability_report.txt` after the analysis fix
- added optional eval-side episode trace artifacts:
  `eval.py` can now save per-step `.pt` traces with extracted `PolicyInputBatch` tensors and dispatched action metadata via `--trace_episodes` / `--trace_output_dir`, without touching the training DB
- added a separate eval-trace analysis entrypoint:
  `analyze_eval_trace.py` now turns those `.pt` sidecars into a compact image/report bundle without bloating `dashboard.py`, and it can optionally export `conv1` / `conv2` / `conv3` activation maps for a selected step
- verification for the eval trace path:
  `pytest tests\test_eval_trace.py tests\test_eval_trace_analysis.py tests\test_analysis_tools.py -q`
  `7 passed`
- verification after the refactor:
  `pytest tests -q`
  `47 passed`

## 2026-04-21

- renamed the live entrypoints and package surface to match what the repo actually is now:
  `train.py`, `eval.py`, `agent.py`, and `agent_core/`
- kept the older `PPO_CNN*` files and package modules as thin compatibility shims so old commands/imports still resolve during the transition
- inspected `BPTT-1` against the live DB, eval JSONLs, and saved eval traces instead of relying only on the older static report
- confirmed the current run is ahead of the saved report:
  `checkpoint.pth` is at episode ~5260, while `best_checkpoint.pth` is stale at episode 200 because deterministic eval has stayed flat
- confirmed the post-bootstrap action mask is not the main culprit:
  `Move_screen` and `Attack_screen` are available on >99% of logged eval steps
- confirmed the learned late-run action mix is the real concern:
  mostly `NO_OP` plus `ATTACK`, with `MOVE` nearly absent
- found that the current reward path is still the older proxy:
  `positioning_reward` is dead, and terminal win/loss still keys off `obs.reward > 0`
- re-ranked the immediate backlog:
  reward refactor / rebalance now sits ahead of Stage-2 action-history and selection-action work
- refreshed the current docs so the active source-of-truth files reflect the live run state instead of the older post-implementation snapshot
- landed a bounded architecture experiment:
  the policy now has dual-timescale token memory via fast + slow token-temporal SNN pathways combined before the shared latent readout
- kept the dual-timescale patch repo-native:
  the recurrent state stayed a plain `(syn, mem)` tuple, but each tensor now carries an internal pathway axis
- fixed an early post-landing bug in the dual-timescale patch:
  one recurrent-state mask multiply still used the old `[B, tokens, 1]` broadcast pattern and broke when `batch_size != 2`
- added a regression test for non-2 batch sizes so the pathway axis cannot silently alias the batch axis again
- fixed a real runtime bug in `defeat_roaches_v3.py` where `feature_units` could be a NumPy-like array and crash on truthiness checks
- explicitly deferred the riskier "grok" follow-ups for a later branch:
  reward neuromodulation, ALIF swaps, and attention-side temporal state were discussed but not implemented
- verification for the temporal-pathway patch:
  `pytest tests/test_agent.py tests/test_PPO.py tests/test_training_loop.py -q`
  `36 passed`
  `python -m compileall agent.py agent_core tests`
- simplified the learned action space from `NO_OP / MOVE / ATTACK` to `NO_OP / SMART`
- reason for that simplification:
  `Attack_screen` was semantically cheating by behaving too much like attack-move on empty ground, which made the old split cleaner on paper than in-game
- rewired the protocol, PPO masking, agent dispatch, bridge token semantics, and config around `Smart_screen`
- kept analysis/backfill compatibility for older runs:
  analysis tools now infer action semantics from each run config so the older 3-way runs remain readable
- updated eval-trace analysis and dashboard paths so they can understand both old `MOVE/ATTACK` runs and new `SMART` runs
- verification after the Smart-screen redesign:
  `pytest tests -q`
  `55 passed`

## 2026-04-22

- cleaned up the rename boundary so the live plumbing no longer depends on
  placeholder modules inside `PPO_CNN/`
- restored `PPO_CNN/policy_input.py`, `PPO_CNN/policy_network.py`,
  `PPO_CNN/PPO.py`, `PPO_CNN/reward_function.py`, and
  `PPO_CNN/reward_function_2.py` from the archived `old_scritps/`
  snapshot
- current repo contract after that cleanup:
  `agent_core/` is the canonical runtime package, while `PPO_CNN/` is now
  honest legacy code instead of a disguised alias layer
- repaired the main architecture bottleneck called out by the BPTT review:
  the click head no longer has to recover screen coordinates only from a pooled
  global latent
- added explicit 2D positional encoding to the pooled spatial token grid before
  attention
- kept a structured spatial branch alive through the policy and rewired the
  `SMART` click head to consume that retained spatial context plus latent/action
  conditioning
- kept the value and high-level action path global, but made localization
  genuinely spatial
- resolved the `policy_mask` ambiguity by treating it as a true training mask:
  critic loss is now masked too, not only actor and entropy terms
- moved PPO updates ahead of deterministic eval / best-checkpoint selection so
  eval reflects the freshest trained parameters
- added explicit recurrent-state contract assertions in PPO rollout storage and
  bootstrap-tail handling
- verification after the spatial-head repair:
  `pytest tests -q`
  `61 passed`
- captured several post-`SMART` action-space training ideas as future branch candidates rather than immediate reward-function edits
- offline pretraining idea:
  relabel stronger old `MOVE` / `ATTACK` trajectories into the new `SMART` action space and use them as a behavior-cloning warm start before PPO
- dataset-cleanup idea for that branch:
  classify old or future `SMART` clicks by effect using short-horizon observation deltas
  such as enemy health drop, friendly displacement, weapon-cooldown change, or null-effect clicks
- curriculum idea:
  split the meaning of `SMART` across easier tasks before returning to full DefeatRoaches
- concrete curriculum sketches worth remembering:
  `move-to-beacon` style map for purposeful locomotion
  a custom DefeatRoaches-like map with enemies placed so they cannot threaten much, to teach clicking near enemies / attack intent more directly
- explicit caution logged:
  these curriculum / offline-pretrain ideas should stay separate branch work and not be silently folded into the reward refactor just because they are tempting

## 2026-04-22 (compatibility and continuity pass)

- reconciled the `policy_mask` vs `sample_mask` transition:
  - kept `sample_mask` as the canonical name in runtime code paths
  - added compatibility aliases so legacy callers using `policy_mask` still read/write correctly
  - made `_build_tbptt_chunks()` accept `policy_masks=...` as a backward-compatible kwarg
  - made packed chunk output emit both `sample_mask` and `policy_mask` keys
- added rollout-continuity protection in `train.py` checkpointing:
  `save_checkpoint()` now skips persistence while PPO in-flight rollout memory is non-empty
  so resume behavior does not silently break exact on-policy continuity
- verified reward version entrypoint still keeps legacy path:
  `RewardFunctionV2` remains available through `agent_core.rewards.__init__`, while `v3` remains current default
- updated docs index text to mark `docs/archive/bptt_review_checklist.md` (formerly urgent.md) as an active historical review source rather than an empty scratchpad

## 2026-04-22 (deep research follow-up)

- reviewed `docs/archive/deep_research_report_2026-06-17_superseded.md` against the live code and limited the
  patch set to contract-level fixes that did not require environment-side intent
- tightened the frozen observation protocol:
  `PolicyInputBatch` now rejects recurrent state tensors unless they are rank-3
  legacy or rank-4 multi-timescale, so malformed state now fails at the
  protocol boundary instead of later inside `spiking_policy.py`
- fixed rollout cadence for long episodes:
  `train.py` now flushes PPO updates immediately once `rollout_steps` is reached
  inside the episode loop, rather than waiting only for the episode boundary
- explicitly did **not** change time-cap semantics yet:
  the repo still needs a task-definition decision on whether
  `steps_per_episode` is a real horizon or only a training truncation before we
  touch PPO bootstrap behavior there
- added regression coverage for the new contract/cadence behavior:
  - `tests/test_policy_input.py`
  - `tests/test_training_loop.py`
- verification after the follow-up pass:
  `pytest tests/test_policy_input.py tests/test_training_loop.py tests/test_PPO.py tests/test_agent.py -q`
  `47 passed`
  `pytest tests -q`
  `63 passed`

## 2026-04-23 (semantic fixes before Phase 2)

- fixed action-dimension reporting in `spiking_policy.py`:
  added `action_dim` to `resolved_config()` output so analysis tools
  can correctly detect action semantics
- fixed action-semantics inference in `results.py`:
  added fallback logic to detect `semantic_pointer_v1` when
  `spatial_head_type == "token_pointer"` and `meta_input_dim == 19`,
  even when `action_dim` is missing from config
- regenerated Zero-3 instability report with corrected action labels
- **fixed time-cap semantics in `train.py`**:
  - changed `done = env_done or time_cap` to `done = env_only`
  - timeout is now treated as truncation, not terminal
  - value bootstrapping continues through time_cap
  - loop still exits at time_cap but doesn't treat it as episode end
- **verified rollout cadence is correct**:
  - the `rollout_steps` check already happens inside the step loop
  - PPO updates fire immediately when target is reached, not just at episode boundaries
- **increased TBPTT window from 32 to 128**:
  - allows gradients to flow 4x further back in time
  - better credit assignment for long-horizon events
  - enabled by available VRAM/compute
- **increased `steps_per_episode` from 600 to 3600**:
  - the 600 limit was legacy and doesn't match DefeatRoaches survival dynamics
  - games depend on survival + long time limit, not artificial short caps

## 2026-04-22 (semantic action + token-pointer migration)

- implemented Phase 1 + Phase 2 of the spatial-target migration plan
  instead of mixing it with any new TBPTT or reward rewrite
- replaced the live policy semantics with:
  `NO_OP`, `LEFT_CLICK`, `RIGHT_CLICK`
- current DefeatRoaches mapping after that migration:
  `RIGHT_CLICK -> Smart_screen(x, y)`
  `LEFT_CLICK` is scaffolded in code and protocol, but masked unavailable on
  the current wrapper so it cannot become a live policy alias
- shrank `meta_vec` from the older raw-action shape to the semantic one:
  `meta_vec[19] = player[11] + semantic_available_actions[3] + pysc2_last_action[1] + bridge_token[4]`
- expanded bridge-token semantics so the recurrent path can distinguish
  `LEFT_CLICK`, `RIGHT_CLICK`, bootstrap select, and no-op cleanly
- added `ActionSample` as the acting payload and threaded optional
  `target_index / coarse_index / fine_index` storage through PPO rollout
  memory without breaking external `act / move_x / move_y` logging
- replaced the hardcoded factorized `x + y` replay contract with a generic
  policy-side target-head interface:
  `build_target_head`, `sample_target`, `evaluate_target`,
  `encode_xy_to_target`, and `decode_target_to_xy`
- kept `factorized_xy` alive as a legacy target-head mode behind config
- made `token_pointer` the new default head:
  the policy now predicts one categorical distribution over the pooled spatial
  token grid and decodes token centers back to logged `(x, y)` screen clicks
- generalized PPO from "SMART is the only spatial action" to a semantic
  spatial-action set and moved target entropy / target log-prob evaluation out
  of PPO math and into the policy head
- preserved the core seam that was already correct:
  action selection still happens first, target selection second, and replay
  still teacher-forces the recorded action id
- updated analysis helpers and eval-trace helpers so new runs with
  `action_dim=3` plus `spatial_head_type=token_pointer` do not get mislabeled
  as legacy runs
- refreshed diagnostics / tests to the new contract:
  semantic availability, bridge token shape, token-pointer head shape,
  ActionSample returns, generic PPO losses, packed replay, and current
  policy-input diagnostics now all assert against the migrated behavior
- verification after the migration:
  `pytest tests -q`
  `63 passed`

## 2026-06-17 Repo Cleanup

- removed the old `PPO_CNN/` package and root `PPO_CNN_*` launchers; the
  canonical entrypoints are now only `train.py`, `eval.py`, and the Ray module
- removed scratch/obsolete surfaces:
  `action_space/action_space_preview.py`, `agent_core/rewards/legacy_reward.py`,
  `Utility/obs_sapce.py`, `Utility/script.py`, and `Utility/valid_actions.py`
- removed legacy action aliases from `agent_core.policy_protocol`; live code now
  names the semantic action as `RIGHT_CLICK` rather than `SMART/MOVE/ATTACK`
- centralized small numeric coercion helpers in `obs_space/_numeric.py` and
  derived production `FeatureUnit` indices from the PySC2 enum
- fixed eval-trace analysis so `coarse_to_fine` checkpoints rebuild with the
  correct target-head config and label actions as `no-op / left_click / right_click`
- made `PolicyInputDiagnosticsWrapper` explicit that its local extractor does
  not have the agent's real `last_action_token`; action-effect attribution
  should come from eval traces
- retained `factorized_xy` and `_replay_chunk_group_reference` as explicit,
  tracked compatibility/reference debt rather than silent current architecture

## Next Checks

- refactor / rebalance the reward function using the newer wrapper-driven env understanding
- env-verify timeout-as-truncation behavior:
  `steps_per_episode` is treated as a cap/truncation for PPO bootstrap, not a
  true terminal task horizon
- env-verify the new semantic action mask:
  confirm `RIGHT_CLICK` behaves as intended and keeping `LEFT_CLICK` masked is
  still the right no-alias choice on the live wrapper
- validate the current spatial-head step:
  `coarse_to_fine` is implemented and selected in `config.yaml`; it still needs
  a short live training/eval pass before we draw conclusions or move toward
  `heatmap`
- verify and tune `RewardFunctionV4` terminal/outcome/action-shaping semantics against live
  traces
- review the `banana_smart_v5_b2048_e4_a10` diagnostics/traces against the live
  DB/checkpoint state, and compare directly to `banana_smart_v4_b2048_e4_a10`
- keep `analysis_results/BPTT-1` as historical context only (same obs function shape, but older action-space version)
- re-run deterministic and stochastic eval after the reward pass
- only then decide whether the next action-space step is:
  action-history token group,
  learnable selection actions,
  or both
- keep the future branch ideas visible but separate:
  offline `SMART` pretraining,
  effect-labeled click datasets,
  and curriculum maps for locomotion / attack semantics
- keep the larger branch questions open:
  entity identity, long-term SNN/TBPTT verdict, and whether this remains the mainline or the research branch
