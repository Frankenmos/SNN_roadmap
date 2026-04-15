# SNN Roadmap: Next Fixes and Refactors

## What was fixed now
- Removed unresolved merge conflict markers in `PPO_CNN/policy_network.py`.
- Removed duplicated second copy of `dashboard.py`.
- Fixed `best_avg_reward` tracking in `PPO_CNN_run.py` so checkpoint metadata is meaningful.
- Made logger shutdown flush buffered writes in `Utility/logger_utils.py`.
- Added missing runtime dependencies in `requirements.txt` (`torch`, `torchvision`, `dotmap`, `ray`).

## What I would change next

## 1) Reward function redesign
Current issues:
- Agent health source is inconsistent with observation extractor.
- Reward is mostly dense damage +/- health and may incentivize low-quality skirmish loops.
- Terminal win/loss signal is relatively small compared to cumulative shaping.

Plan:
1. Define one authoritative health and combat signal source from `feature_units`:
   - friendly marines: sum and min health
   - enemy roaches: count, sum health
2. Keep reward terms minimal and interpretable:
   - `+k_damage * enemy_health_delta` (clipped)
   - `-k_taken * friendly_health_delta` (clipped)
   - `+k_kill` per enemy removed
   - `+k_win` / `-k_loss` on terminal
3. Add anti-stall term:
   - small negative step penalty when no combat progress for N steps.
4. Add strict reward logging schema:
   - one field per component used in training
   - no placeholders mapped from unrelated component names.
5. Validate with offline replay checks:
   - run reward over saved episode traces and confirm signs/magnitudes match expectations.

Suggested weights to start:
- `k_damage=0.2`, `k_taken=0.3`, `k_kill=8.0`, `k_win=30.0`, `k_loss=-30.0`, `stall_penalty=-0.01`.

## 2) Observation space cleanup
Current issues:
- `obs_space_2.py` has duplicate method definitions.
- Feature semantics are mixed and some fields are likely unstable/noisy.
- 100-D vector is history-packed without explicit structure metadata.

Plan:
1. Remove duplicate methods and enforce one deterministic extractor path.
2. Build structured vector groups:
   - self: health, position, velocity
   - nearest enemy: health, position, distance, relative angle
   - global: enemy_count, friendly_count, normalized step progress
3. Normalize all scalar features to stable ranges:
   - coordinates by screen size
   - health by expected max
   - distances by map diagonal
4. Keep temporal context explicit:
   - either stack `K` vectors in channel/time order, or use deltas only.
5. Add extractor unit tests with fixed mock inputs asserting exact numeric outputs.

## 3) PPO/training loop reliability
1. Remove unused imports and dead functions (`reset_environment` if unused).
2. Add checkpoint schema version and safe defaults on load.
3. Log best reward and update count per checkpoint.
4. Add deterministic seed wiring (`numpy`, `torch`, env).
5. Add integration test for one mini rollout + update pass with mocked env.

## 4) Dashboard/data contract consistency
1. Ensure dashboard expects exactly the same reward component names produced by trainer.
2. Add schema check at load time with clear warning when columns are missing.
3. For large DBs, aggregate in SQL before loading into memory.

## 5) Environment setup correctness
1. Re-check map/player configuration for `DefeatRoaches` scenario (usually single-agent minigame, bot setup may be unnecessary or harmful).
2. Make `setup_env` defaults match `config.yaml` target map.
3. Add a smoke test: reset + 10 steps + close.
