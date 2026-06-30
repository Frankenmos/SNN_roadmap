# Working Log

## 2026-04-19
- Step A: added `PPO_CNN/policy_input.py` with the frozen `PolicyInputBatch` protocol, Fix 3 constants (`N_max = 24`, `K_max = 20`, selection width `7`), strict shape validation, and minimal batch helpers (`to`, `detach`, `index_select`).
- Why this shape lock matters:
  `PolicyInputBatch` is the contract that later steps will share between the obs extractor, PPO buffer, and policy forward pass. Freezing it first keeps us from doing silent tensor-shape drift while Step B/C/D/E land.
- Why the masks are `torch.bool` already:
  Step C needs attention masking, and bool masks make that intent explicit. It also prevents an easy bug where `0/1` float masks get treated like features instead of validity flags.
- Why `F_unit` and `F_meta` stay dynamic here:
  the plan fixes the slot counts (`24`, `20`, `7`) now, but the exact curated feature count and meta width get finalized in later steps. Locking the outer protocol first lets us keep momentum without guessing those widths too early.
- Why I added `to`, `detach`, and `index_select`:
  Step D will need to move full observation batches between CPU/GPU and slice mini-batches out of rollout storage. Adding those helpers now keeps the later PPO rewrite small and mechanical instead of repetitive.
- Step A: added `tests/test_policy_input.py` to lock the protocol in place before the encoder/model rewiring.
- Tests cover three things on purpose:
  valid Fix-3 shapes, consistent batch slicing across every field, and mask dtype enforcement. That gives us confidence the new protocol is safe even before any live code uses it.
- Runtime path intentionally unchanged after Step A: agent, PPO, and observation extraction still use the current `(spatial, vector)` flow until later steps.

- Step B: added `EntityEncoder`, `SelectionEncoder`, and `MetaEncoder` in `PPO_CNN/policy_network.py`, plus shared protocol constants for the curated unit fields, selection-field order, token counts, and DefeatRoaches action-ID mask space.
- Why the curated entity list differs slightly from the prose list in the plan:
  local `pysc2.lib.features.FeatureUnit` exposes `health_ratio` and `shield_ratio`, but not `health_max` / `shield_max`. I used the ratio fields as the closest feature-units-native substitute while keeping the rest of the §3.7 list intact.
- Why selection also embeds `unit_type`:
  `multi_select` is built from `UnitLayer`, whose first column is again `unit_type`. Reusing the same modeling rule here avoids feeding unit-type IDs as fake magnitudes in either token group.
- Why `MetaEncoder` expects structured raw meta fields instead of a pre-embedded tail:
  the protocol stays model-agnostic if the extractor ships `player + avail_mask + last_action_index` as numbers and the encoder owns the learned last-action embedding internally.
- Step B tests:
  added shape/masking tests that verify padded entity and selection slots stay exactly zero after encoding, and that the meta path emits a single `[B, 1, D]` token as intended.

- Step C: rewired `PolicyNetwork.forward` to accept `PolicyInputBatch`, build spatial/entity/selection/meta token groups, add token-type embeddings, concatenate them, and run the whole `[B, spatial + 24 + 20 + 1, D]` sequence through attention and the temporal SNN.
- Why token count is dynamic in code even though the plan says `[B, 94, D]`:
  the production config still lands at `49 + 24 + 20 + 1 = 94`, but I kept the spatial-token side derived from the configured pool size so the small unit tests can still run with a reduced pooling grid.
- Masking details that matter:
  I moved attention logits to `fp32`, used `-1e4` for masked keys, and softmaxed there before casting back. That follows the AMP warning in §3.7 and avoids `-inf`-driven NaNs.
- I also zero masked token outputs before and after the temporal SNN update:
  this prevents padded entity/selection slots from retaining stale recurrent state when the real unit count shrinks from one step to the next.
- Step C tests:
  direct policy tests now build real `PolicyInputBatch` objects, verify state continuity under the new API, and confirm the learnable SNN/attention time constants still receive finite gradients.

- Step D: rewired PPO storage/replay around `PolicyInputBatch`.
  Each transition now stores one detached CPU batch slice that already contains `state_in`, and rollout replay uses `PolicyInputBatch.stack(...)` / `index_select(...)` instead of carrying separate `spatial`, `vector`, and SNN-state arrays.
- Why I folded `state_in` into the batch instead of keeping a side-channel:
  the plan’s protocol already names `state_in` as part of the batch contract, and once PPO followed that contract the update path became much simpler and harder to mismatch.
- Step E: replaced the old 100-d aggregate vector extractor with a hybrid batch extractor in `obs_space/obs_space_2.py`.
  It now emits:
  `spatial_obs [1,27,84,84]`
  `entity_features [1,24,21] + mask`
  `selection_features [1,20,7] + mask`
  `meta_vec [1,28] = player[11] + avail_mask[16] + last_action_index[1]`
- Important inference I made in Step E:
  the plan’s prose list mentions `health_max` / `shield_max`, but local `FeatureUnit` does not expose them. I kept the protocol focused on `feature_units` as requested and used `health_ratio` / `shield_ratio` instead.
- Selection handling detail:
  the main path uses `multi_select`, but if it is empty and `single_select` is populated, I fall back to the single-select row so “what is selected” does not disappear on those rare frames.
- Running-stat note:
  the per-attribute running mean/std lives in the extractor and is now saved into checkpoints as `extractor_state`, then restored by both training resume and eval loading. That keeps observation normalization stable across sessions.
- Step F: added PPO stats for `entity_mask_utilization`, `entity_count_p50`, `entity_count_p99`, and `selection_mask_utilization`, and extended `Utility/logger_utils.py` so they persist into `ppo_updates`.
- Local verification after Steps D/E/F:
  `pytest tests\test_policy_input.py tests\test_PPO.py tests\test_agent.py tests\test_training_loop.py -q`
  `22 passed`
- Follow-up note to self:
  add optional SC2-env-backed smoke tests/wrappers after the core fix, so real observation-schema checks can run when the map client is available without making the default unit suite brittle.

- Follow-up tooling pass:
  split fake PySC2 setup and reusable batch factories into `tests/MockedEnv/` so the test harness has a clearer home:
  `fake_pysc2.py`, `fixtures.py`, `policy_batch.py`, and a short `README.md`.
- `tests/conftest.py` now only handles path setup and loads `MockedEnv.fixtures`, instead of carrying the whole fake-SC2 implementation inline.
- Added `Utility/policy_input_diagnostics_wrapper.py` for the live-env blind spots.
  It logs:
  raw `available_actions` IDs,
  raw `last_actions`,
  `feature_units` / selection counts and truncation,
  small raw row samples,
  emitted `PolicyInputBatch` counts/utilization,
  and the meta token’s decoded last-action / avail-mask summary.
- Wired the diagnostics wrapper into `envs/setup_env.py` and exposed it through `PPO_CNN_eval.py` with:
  `--inspect_policy_input`
  `--policy_input_output`
  `--policy_input_every`
- Added a local wrapper-format test:
  `tests/MockedEnv/test_policy_input_diagnostics_wrapper.py`
  so the JSONL schema is checked before asking for real SC2 logs.
- Local verification after the tooling pass:
  `pytest tests\test_policy_input.py tests\test_PPO.py tests\test_agent.py tests\test_training_loop.py tests\MockedEnv\test_policy_input_diagnostics_wrapper.py -q`
  `23 passed`
- Planning pass:
  reviewed the external critiques against the actual Fix-3 code and wrote `planned_fixes.md` as an adjudicated change order.
  Main outcome: the real bugs are in extractor normalization safety and eval-time stat drift; several other review points were valid concerns but not actual live bugs in this codebase.
- Follow-up planning clarification:
  tightened the plan around SNN-state compatibility. The token-temporal state is concretely `[B, N, D]`, so its shape did change with the new `94`-token budget. What stayed "unchanged" was only the presence of the `state_in` field in the batch protocol. Checkpoints do not serialize live `snn_state`, so there is no missing state-migration bug, but pre-Fix-3 checkpoints are still architecture-incompatible and should be treated as unsupported rather than conceptually "the same state shape."
- Extractor hardening pass:
  tightened `RunningFeatureNormalizer` in `obs_space/obs_space_2.py` so normalization now waits for a warm-up sample count, skips near-constant dimensions instead of dividing by a tiny std, and clips active normalized outputs to a sane range. This was the main numerical-risk fix from `planned_fixes.md`.
- Why I chose "skip low-variance dims" instead of "always divide by a safer floor":
  for fields that are effectively constant in DefeatRoaches, preserving the raw value is less surprising than manufacturing a pseudo-z-score from a tiny denominator. It keeps rare nonzero events informative without turning them into million-scale spikes.
- Eval-stat isolation pass:
  `PPO_CNN_agent.step()` now threads `update_stats=not deterministic` into the extractor, so deterministic evaluation stops mutating the running observation statistics used by later training or later evals.
- Schema hardening pass:
  extractor construction now validates curated entity and selection field names against the local PySC2 enums, and numeric-row projection raises if the table width is narrower than the required enum column. That converts silent all-zero feature failure into an immediate error.
- Regression coverage added:
  `tests/test_observation_extractor.py` now locks the normalizer behavior, fail-fast field validation, and last-action sentinel separation; `tests/test_agent.py` now checks that deterministic steps leave extractor stats untouched while stochastic steps still update them.
- Local verification after the hardening pass:
  `pytest tests\test_observation_extractor.py tests\test_agent.py tests\test_policy_input.py tests\test_PPO.py tests\test_training_loop.py tests\MockedEnv\test_policy_input_diagnostics_wrapper.py -q`
  `28 passed`
- BPTT analysis pass:
  read the uploaded tutorial against the current PPO rollout/update code and wrote `THE_BPTT.md`.
  Main conclusion: the project currently replays detached recurrent state during PPO updates, but it does not do BPTT through env steps because the update path shuffles flat timesteps, ignores returned `next_state`, and never re-unrolls the recurrent chain inside the training graph.
- Important extra finding from that pass:
  helper steps are currently dropped from PPO memory even though they still mutate `self.snn_state` during acting. That means true BPTT cannot be implemented correctly by "just chunking learnable steps"; rollout storage itself has to become faithful to every state-mutating env step, with masked actor loss where appropriate.
- Follow-up BPTT clarification:
  added the entity-slot identity caveat to `THE_BPTT.md`. Current entity tokens come from padded `feature_units` rows with no tag-based slot pinning, so clean cross-step entity memory is not guaranteed. Important nuance: that is already a live recurrent-design caveat today because token-temporal state is carried across entity slots during acting. My current recommendation is a staged approach: TBPTT first for spatial/selection/meta, with entity-token recurrent carry explicitly disabled until we introduce identity pinning.
- Stage-1 TBPTT implementation:
  rewired PPO update from flat shuffled timestep replay to ordered TBPTT chunk replay. Rollout memory now stores every policy-touched env step with a `policy_mask`, PPO re-unrolls `next_state -> state_in` inside update-time sequence replay, and helper steps contribute to recurrent/value learning without receiving actor loss.
- Recurrent-state staging choice:
  to keep the first BPTT pass honest, `policy_network.py` now zeroes the entity-token recurrent slice between env steps. Spatial, selection, and meta tokens keep temporal carry; entity tokens stay one-step-only until we introduce tag-based slot pinning.
- Verification added for the new path:
  `tests/test_PPO.py` now checks ordered chunk replay and done-boundary reset behavior, `tests/test_training_loop.py` now checks helper-step storage, and `tests/test_agent.py` now checks entity-token recurrent carry is zeroed. Full local suite after the rewrite:
  `pytest tests -q`
  `31 passed`
- TBPTT speed pass, policy-side groundwork:
  added `PolicyNetwork.forward_step_tensors(...)` as a replay-only fast path so PPO can feed pre-packed tensors directly without rebuilding a `PolicyInputBatch` every timestep. The public `forward(PolicyInputBatch)` contract stays unchanged and now just forwards into the tensor path.
- Added `PolicyNetwork.reset_state_rows(...)`:
  recurrent done resets now have a dedicated mask-based helper. This keeps the reset semantics explicit and avoids rebuilding one-row zero states in Python during replay.
- TBPTT speed pass, packed replay:
  `PPO.py` now packs each chunk group once into time-major tensors (`[T, B, ...]` for obs fields plus `done`, `policy_mask`, and `alive_mask`) and replays that packed group with one forward call per replay timestep. The slow per-step `PolicyInputBatch` assembly path is retained only as `_replay_chunk_group_reference(...)` for regression testing.
- Why I kept the old replay implementation around as a reference:
  this refactor changes the hot path, so having a known-good replay implementation still in-tree made it possible to add exact equivalence tests instead of trusting the optimization by inspection.
- New PPO update instrumentation:
  each `ppo_updates` row now records `update_wall_seconds`, `tbptt_chunks`, `tbptt_chunk_groups`, `tbptt_window`, `tbptt_group_max_steps`, `tbptt_group_mean_active_chunks`, and `tbptt_forward_calls`. That gives the next real run enough signal to separate "still slow because of true compute" from "still slow because batching is poor."
- Logging and regression coverage for the speed pass:
  `Utility/logger_utils.py` now persists the new TBPTT metrics, `tests/test_PPO.py` locks packed replay shapes/equivalence/reset behavior/forward-call counts, and `tests/test_logger_utils.py` runs the actual log-listener insert path against a temp SQLite DB.
- Local verification after the packed replay pass:
  `pytest tests -q`
  `36 passed`
- Attention kernel quick win:
  swapped `SpikingSelfAttention` from manual `QK^T -> masked softmax -> AV` to `torch.nn.functional.scaled_dot_product_attention(...)`. This keeps the same dense attention semantics but lets PyTorch dispatch to its fused SDPA backends automatically when the runtime/device supports them.
- Why SDPA and not FlexAttention here:
  our attention pattern is still plain dense attention with a simple padding mask. SDPA is the lowest-risk fast path for that case, while FlexAttention is more useful once we need custom score modifications or structured sparsity.
- Action-space diagnostics pass:
  the old `Utility/available_actions_wrapper.py` printer only showed newly seen available actions, which is fine for quick inspection but not enough for tokenized action-space design. I replaced it with a dual-purpose module: the original printer stays, and a new `AvailableActionsDiagnosticsWrapper` writes JSONL with previous/current `available_actions`, dispatched PySC2 call info, `last_actions`, and small selection/unit-count breadcrumbs.
- Eval wiring for action-space design:
  `PPO_CNN_eval.py` now supports `--inspect_actions`, `--actions_output`, and `--actions_every`, and `envs/setup_env.py` can wrap the env with the structured action diagnostics logger. That gives us real per-step availability/dispatch traces to design the next action tokenization pass against.
- Analysis/dashboard catch-up pass:
  `tools/analysis/results.py` now loads the current BPTT/TBPTT fields already present in `ppo_updates`, carries `move_x/move_y` through `steps`, and derives phase-of-episode action mix (`early/mid/late`) so the dashboard can reason about policy timing instead of only whole-run action shares.
- Dashboard alignment with the current branch:
  `tools/analysis/dashboard.py` now exposes a real TBPTT/speed section (update wall time, chunk/group metrics, forward calls, derived throughput), splits eval curves into deterministic vs stochastic with an explicit reward-gap plot, adds reward-efficiency and move-target views, and surfaces phase-specific action mix panels.
- Checkpoint introspection upgraded for this repo:
  `tools/analysis/analyze_pth.py` and the dashboard checkpoint tab now summarize top-level checkpoint metadata, saved extractor normalizer stats, and learned SNN/attention `alpha`/`beta` parameters. That replaces some of the old generic "weight cloud" emphasis with branch-specific signals that are actually useful while debugging the SNN+BPTT path.
- AI-friendly analysis export:
  `results.py` now supports `--aismart`, which writes a focused bundle of static PNG panels under `analysis_results/<run_name>/ai_friendly_results/`. The goal is not to duplicate the whole Streamlit dashboard, but to expose the high-signal panels we most often end up discussing back in text-only contexts: reward, episode length, efficiency, entropy, whole/phase action mix, move-target heatmap, TBPTT speed, eval split/gap, and reward-component trends when present.
- Local verification for the analysis pass:
  `pytest tests -q`
  `40 passed`

## 2026-04-20
- Stage-1 action refactor landed:
  policy action vocab stays `NO_OP / MOVE / ATTACK`, `Smart_screen` is gone from the learned path, and both `MOVE` and `ATTACK` now dispatch explicit screen coordinates.
- Bridge-token plumbing landed end to end.
  `meta_vec` is now `32` dims with named offsets for the agent-owned `[type, x_norm, y_norm, extra]` token, and the agent feeds back the executed action semantics instead of only relying on PySC2 `last_actions`.
- Reset bootstrap is now explicit and outside PPO memory.
  On reset frames where only `select_army` is available, the agent emits one bootstrap selection step, records a reserved helper bridge token, and only starts policy-controlled PPO steps after the marines are selected.
- Policy/PPO refactor landed without changing rollout storage shape.
  The recurrent trunk runs once, action logits are masked from `available_actions`, and the spatial head is conditioned on the chosen action type from the same latent. PPO replay uses the same conditioning path and now treats both `MOVE` and `ATTACK` as spatial for log-prob and entropy accounting.
- Runtime/fidelity cleanup:
  removed the old scripted nearest-enemy attack targeting from the main control loop, removed the mid-episode `select_army` fallback path, and aligned the fake PySC2 test IDs with the real DefeatRoaches action IDs so availability masking is truthful in tests.
- Verification after the action refactor:
  `pytest tests -q`
  `47 passed`
