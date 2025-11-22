# Bugs and Fixes Identified

## 1. Test Suite Deprecation
The existing tests in `tests/` were written for an older version of the agent and PPO implementation. They fail due to:
- **Signature Mismatches:** `agent.step_with_data` now returns 8 values (including tensors for logging), while tests expect fewer values.
- **Type Mismatches:** The new implementation heavily utilizes `torch.Tensor` on the configured device (GPU/CPU), whereas tests expect standard Python types or NumPy arrays.
- **Missing Mock Attributes:** The `RewardFunctionV2` requires access to `feature_units` to calculate rewards based on enemy/agent positions and health, which were missing in the mock observations.

## 2. Tensor vs Numpy Confusion in `PPO_CNN_run.py`
In `PPO_CNN_run.py`:
```python
# spatial/vector are still numpy, store_transition will handle them (or we can convert here)
agent.ppo.store_transition(...)
```
This comment is outdated. `ObservationExtractor` (in `obs_space/obs_space_2.py`) explicitly returns tensors:
```python
spatial_obs = torch.as_tensor(feature_screen / 255.0, dtype=torch.float32, device=self.device)
# ...
return spatial_obs, vector_obs
```
This is actually **good** for performance (keeping data on the device), but the comment and potentially some downstream handling (if it expects numpy) could be misleading. The fix is to ensure `store_transition` is robust to receiving tensors, which it already appears to be, but we should verify the flow.

## 3. `PPO.select_action` Return Signature
The `PPO.select_action` method returns a tuple containing:
`(action, xy_env, log_prob_total, state_value, next_state, xy_raw_sample)`
The tests `test_PPO.py` were not updated to handle the `next_state` or `xy_raw_sample` return values, causing unpacking errors.

## 4. `RewardFunctionV2` Dependency on Raw Observations
The reward function relies on iterating over `obs.observation.feature_units`. This means we cannot purely use tensors for the entire pipeline; we must retain the raw `pysc2` observation object at least until the reward is calculated. This is acceptable but requires our mocks to correctly simulate the `pysc2` object structure.

## Plan for Resolution
1.  **Update Mocks:** Enhance `_create_mock_obs` in test files to include `feature_units` with mock `alliance`, `health`, `x`, and `y` attributes.
2.  **Update Test Assertions:** Modify `test_agent.py` and `test_training_loop.py` to unpack the correct number of return values and assert `torch.Tensor` types.
3.  **Update PPO Tests:** Fix `MockPolicyNet` in `test_PPO.py` to return the expected 4 values (logits, xy_mean, value, state) and update test calls to match the new `select_action` signature.
