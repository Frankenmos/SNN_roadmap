import numpy as np
import pytest

from agent_core.policy_protocol import (
    AGENT_ACTION_TOKEN_DIM,
    AGENT_LAST_ACTION_OFFSET,
    BRIDGE_ACTION_SMART,
    KILLED_VALUE_DELTA_OFFSET,
    LAST_ANY_ACTION_EXECUTED_OFFSET,
    LAST_SMART_EXECUTED_OFFSET,
    META_VECTOR_DIM,
    META_LAST_ACTION_INDEX_OFFSET,
    NO_ACTION_SENTINEL_INDEX,
    SCORE_PENALTY_BIT_OFFSET,
    SCORE_TOTAL_DELTA_OFFSET,
    UNKNOWN_LAST_ACTION_INDEX,
)
from obs_space import obs_space_2


def test_running_feature_normalizer_skips_low_variance_dims_and_clips_active_dims():
    normalizer = obs_space_2.RunningFeatureNormalizer(
        field_names=("health", "shield"),
        normalized_fields=("health", "shield"),
        min_count_for_normalize=4.0,
        min_std=1.0e-2,
        output_clip=10.0,
    )

    for _ in range(4):
        normalizer.update(
            np.asarray(
                [
                    [100.0, 0.0],
                    [101.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

    normalized = normalizer.normalize(
        np.asarray([[150.0, 1.0]], dtype=np.float32),
    )

    assert abs(float(normalized[0, 0])) <= 10.0
    assert float(normalized[0, 1]) == pytest.approx(1.0)


def test_observation_extractor_fails_fast_on_unknown_feature_unit_field(monkeypatch):
    patched_index = dict(obs_space_2._FEATURE_UNIT_INDEX)
    patched_index.pop("weapon_cooldown", None)
    monkeypatch.setattr(obs_space_2, "_FEATURE_UNIT_INDEX", patched_index)

    with pytest.raises(ValueError, match="Unknown FeatureUnit field"):
        obs_space_2.ObservationExtractor()


def test_observation_extractor_fails_fast_on_unknown_selection_field(monkeypatch):
    patched_index = dict(obs_space_2._UNIT_LAYER_INDEX)
    patched_index.pop("energy", None)
    monkeypatch.setattr(obs_space_2, "_UNIT_LAYER_INDEX", patched_index)

    with pytest.raises(ValueError, match="Unknown UnitLayer field"):
        obs_space_2.ObservationExtractor()


def test_last_action_indices_keep_no_action_no_op_and_unknown_distinct(make_obs):
    extractor = obs_space_2.ObservationExtractor()

    no_action_batch = extractor.peek_observation(
        make_obs(last_actions=np.zeros((0,), dtype=np.int32)),
    )
    no_op_batch = extractor.peek_observation(
        make_obs(last_actions=np.asarray([0], dtype=np.int32)),
    )
    unknown_batch = extractor.peek_observation(
        make_obs(last_actions=np.asarray([999], dtype=np.int32)),
    )

    no_action = float(no_action_batch.meta_vec[0, META_LAST_ACTION_INDEX_OFFSET].item())
    no_op = float(no_op_batch.meta_vec[0, META_LAST_ACTION_INDEX_OFFSET].item())
    unknown = float(unknown_batch.meta_vec[0, META_LAST_ACTION_INDEX_OFFSET].item())

    assert no_action == pytest.approx(float(NO_ACTION_SENTINEL_INDEX))
    assert no_op == pytest.approx(float(obs_space_2._LAST_ACTION_TO_INDEX[0]))
    assert unknown == pytest.approx(float(UNKNOWN_LAST_ACTION_INDEX))
    assert len({int(no_action), int(no_op), int(unknown)}) == 3


def test_observation_extractor_appends_last_action_bridge_token(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    batch = extractor.peek_observation(
        make_obs(),
        last_action_token=np.asarray([BRIDGE_ACTION_SMART, 42, 21, 0], dtype=np.int32),
    )

    token = batch.meta_vec[
        0,
        AGENT_LAST_ACTION_OFFSET : AGENT_LAST_ACTION_OFFSET + AGENT_ACTION_TOKEN_DIM,
    ]
    assert batch.meta_vec.shape[-1] == META_VECTOR_DIM
    assert float(token[0].item()) == pytest.approx(float(BRIDGE_ACTION_SMART))
    assert float(token[1].item()) == pytest.approx(42.0 / 83.0)
    assert float(token[2].item()) == pytest.approx(21.0 / 83.0)
    assert float(token[3].item()) == pytest.approx(0.0)


def test_action_history_marks_empty_and_smart_last_actions(make_obs, fake_actions):
    extractor = obs_space_2.ObservationExtractor()

    empty_batch = extractor.peek_observation(
        make_obs(last_actions=np.zeros((0,), dtype=np.int32)),
    )
    smart_batch = extractor.peek_observation(
        make_obs(last_actions=np.asarray([fake_actions.Smart_screen.id], dtype=np.int32)),
    )

    assert float(empty_batch.meta_vec[0, LAST_ANY_ACTION_EXECUTED_OFFSET].item()) == 0.0
    assert float(empty_batch.meta_vec[0, LAST_SMART_EXECUTED_OFFSET].item()) == 0.0
    assert float(smart_batch.meta_vec[0, LAST_ANY_ACTION_EXECUTED_OFFSET].item()) == 1.0
    assert float(smart_batch.meta_vec[0, LAST_SMART_EXECUTED_OFFSET].item()) == 1.0


def test_action_history_encodes_score_delta_clipping_and_penalty(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    extractor.extract_observation(
        make_obs(score_cumulative=[0] * 13),
    )

    score_up = [0] * 13
    score_up[0] = 15
    score_up[5] = 120
    up_batch = extractor.extract_observation(
        make_obs(score_cumulative=score_up),
    )

    assert float(up_batch.meta_vec[0, SCORE_TOTAL_DELTA_OFFSET].item()) == pytest.approx(1.0)
    assert float(up_batch.meta_vec[0, KILLED_VALUE_DELTA_OFFSET].item()) == pytest.approx(1.0)
    assert float(up_batch.meta_vec[0, SCORE_PENALTY_BIT_OFFSET].item()) == 0.0

    score_down = list(score_up)
    score_down[0] = 5
    down_batch = extractor.extract_observation(
        make_obs(score_cumulative=score_down),
    )

    assert float(down_batch.meta_vec[0, SCORE_TOTAL_DELTA_OFFSET].item()) == pytest.approx(-1.0)
    assert float(down_batch.meta_vec[0, KILLED_VALUE_DELTA_OFFSET].item()) == pytest.approx(0.0)
    assert float(down_batch.meta_vec[0, SCORE_PENALTY_BIT_OFFSET].item()) == 1.0


def test_action_history_score_delta_resets_between_episodes(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    extractor.extract_observation(
        make_obs(score_cumulative=[10] + [0] * 12),
    )

    extractor.reset()
    batch = extractor.extract_observation(
        make_obs(score_cumulative=[30] + [0] * 12),
    )

    assert float(batch.meta_vec[0, SCORE_TOTAL_DELTA_OFFSET].item()) == pytest.approx(0.0)
    assert float(batch.meta_vec[0, SCORE_PENALTY_BIT_OFFSET].item()) == 0.0


def test_peek_observation_does_not_consume_action_history_score_delta(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    extractor.extract_observation(
        make_obs(score_cumulative=[10] + [0] * 12),
    )

    next_score = [0] * 13
    next_score[0] = 20
    next_score[5] = 100
    peek_batch = extractor.peek_observation(
        make_obs(score_cumulative=next_score),
    )
    actual_batch = extractor.extract_observation(
        make_obs(score_cumulative=next_score),
    )

    assert float(peek_batch.meta_vec[0, SCORE_TOTAL_DELTA_OFFSET].item()) == pytest.approx(1.0)
    assert float(peek_batch.meta_vec[0, KILLED_VALUE_DELTA_OFFSET].item()) == pytest.approx(1.0)
    assert float(actual_batch.meta_vec[0, SCORE_TOTAL_DELTA_OFFSET].item()) == pytest.approx(1.0)
    assert float(actual_batch.meta_vec[0, KILLED_VALUE_DELTA_OFFSET].item()) == pytest.approx(1.0)
