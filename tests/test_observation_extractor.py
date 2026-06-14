from types import SimpleNamespace

import numpy as np
import pytest

from agent_core.policy_protocol import (
    ACTION_FEEDBACK_ANY_EXECUTED_OFFSET,
    ACTION_FEEDBACK_BRIDGE_TYPE_OFFSET,
    ACTION_FEEDBACK_ENEMY_HEALTH_DROP_OFFSET,
    ACTION_FEEDBACK_EXECUTED_SMART_OFFSET,
    ACTION_FEEDBACK_FRIENDLY_HEALTH_DROP_OFFSET,
    ACTION_FEEDBACK_KILL_DELTA_OFFSET,
    ACTION_FEEDBACK_MOVED_TOWARD_TARGET_OFFSET,
    ACTION_FEEDBACK_PENALTY_BIT_OFFSET,
    ACTION_FEEDBACK_SCORE_DELTA_OFFSET,
    ACTION_FEEDBACK_TARGET_NEAR_ENEMY_OFFSET,
    ACTION_FEEDBACK_TOKEN_DIM,
    ACTION_FEEDBACK_X_NORM_OFFSET,
    ACTION_FEEDBACK_Y_NORM_OFFSET,
    BRIDGE_ACTION_SMART,
    CURATED_FEATURE_UNIT_FIELDS,
    MAX_ENTITY_TOKENS,
    META_VECTOR_DIM,
    META_LAST_ACTION_INDEX_OFFSET,
    NO_ACTION_SENTINEL_INDEX,
    UNKNOWN_LAST_ACTION_INDEX,
)
from obs_space import obs_space_2


def _unit(*, alliance, health, x, y, tag=None):
    kwargs = {
        "alliance": alliance,
        "health": health,
        "x": x,
        "y": y,
        "unit_type": 48 if alliance == 1 else 110,
        "attack_range": 5,
    }
    if tag is not None:
        kwargs["tag"] = tag
    return SimpleNamespace(**kwargs)


def _obs_with_units(make_obs, units, **kwargs):
    obs = make_obs(**kwargs)
    obs.observation.feature_units = list(units)
    return obs


def _smart_token(x, y):
    return np.asarray([BRIDGE_ACTION_SMART, x, y, 0], dtype=np.float32)


def _feedback_value(batch, offset):
    return float(batch.action_feedback_tokens[0, 0, offset].item())


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


def test_entity_sorting_keeps_tail_enemies_under_token_cap(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    friendlies = [
        _unit(alliance=1, health=45, x=float(idx), y=0.0)
        for idx in range(MAX_ENTITY_TOKENS + 4)
    ]
    enemies = [
        _unit(alliance=4, health=145, x=60.0, y=10.0),
        _unit(alliance=4, health=145, x=62.0, y=10.0),
    ]

    batch = extractor.peek_observation(
        _obs_with_units(make_obs, friendlies + enemies),
    )

    alliance_idx = CURATED_FEATURE_UNIT_FIELDS.index("alliance")
    alliances = batch.entity_features[0, :, alliance_idx].detach().cpu().tolist()
    active_count = int(batch.entity_mask[0].sum().item())
    assert active_count == MAX_ENTITY_TOKENS
    assert alliances[:2] == [4.0, 4.0]
    assert alliances.count(4.0) == 2


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


def test_observation_extractor_emits_action_feedback_token(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    batch = extractor.peek_observation(
        make_obs(),
        last_action_token=np.asarray([BRIDGE_ACTION_SMART, 42, 21, 0], dtype=np.int32),
    )

    token = batch.action_feedback_tokens[0, 0]
    assert batch.meta_vec.shape[-1] == META_VECTOR_DIM
    assert batch.action_feedback_tokens.shape[-1] == ACTION_FEEDBACK_TOKEN_DIM
    assert float(token[ACTION_FEEDBACK_BRIDGE_TYPE_OFFSET].item()) == pytest.approx(float(BRIDGE_ACTION_SMART))
    assert float(token[ACTION_FEEDBACK_X_NORM_OFFSET].item()) == pytest.approx(42.0 / 83.0)
    assert float(token[ACTION_FEEDBACK_Y_NORM_OFFSET].item()) == pytest.approx(21.0 / 83.0)


def test_action_history_marks_empty_and_smart_last_actions(make_obs, fake_actions):
    extractor = obs_space_2.ObservationExtractor()

    empty_batch = extractor.peek_observation(
        make_obs(last_actions=np.zeros((0,), dtype=np.int32)),
    )
    smart_batch = extractor.peek_observation(
        make_obs(last_actions=np.asarray([fake_actions.Smart_screen.id], dtype=np.int32)),
    )

    assert float(empty_batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_ANY_EXECUTED_OFFSET].item()) == 0.0
    assert float(empty_batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_EXECUTED_SMART_OFFSET].item()) == 0.0
    assert float(smart_batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_ANY_EXECUTED_OFFSET].item()) == 1.0
    assert float(smart_batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_EXECUTED_SMART_OFFSET].item()) == 1.0


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

    assert float(up_batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_SCORE_DELTA_OFFSET].item()) == pytest.approx(1.0)
    assert float(up_batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_KILL_DELTA_OFFSET].item()) == pytest.approx(1.0)
    assert float(up_batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_PENALTY_BIT_OFFSET].item()) == 0.0

    score_down = list(score_up)
    score_down[0] = 5
    down_batch = extractor.extract_observation(
        make_obs(score_cumulative=score_down),
    )

    assert float(down_batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_SCORE_DELTA_OFFSET].item()) == pytest.approx(-1.0)
    assert float(down_batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_KILL_DELTA_OFFSET].item()) == pytest.approx(0.0)
    assert float(down_batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_PENALTY_BIT_OFFSET].item()) == 1.0


def test_action_history_score_delta_resets_between_episodes(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    extractor.extract_observation(
        make_obs(score_cumulative=[10] + [0] * 12),
    )

    extractor.reset()
    batch = extractor.extract_observation(
        make_obs(score_cumulative=[30] + [0] * 12),
    )

    assert float(batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_SCORE_DELTA_OFFSET].item()) == pytest.approx(0.0)
    assert float(batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_PENALTY_BIT_OFFSET].item()) == 0.0


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

    assert float(peek_batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_SCORE_DELTA_OFFSET].item()) == pytest.approx(1.0)
    assert float(peek_batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_KILL_DELTA_OFFSET].item()) == pytest.approx(1.0)
    assert float(actual_batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_SCORE_DELTA_OFFSET].item()) == pytest.approx(1.0)
    assert float(actual_batch.action_feedback_tokens[0, 0, ACTION_FEEDBACK_KILL_DELTA_OFFSET].item()) == pytest.approx(1.0)


def test_action_effect_feedback_is_zero_without_previous_frame(make_obs):
    extractor = obs_space_2.ObservationExtractor()

    batch = extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=10, y=10, tag=1),
                _unit(alliance=4, health=100, x=20, y=20, tag=2),
            ],
        ),
        last_action_token=_smart_token(20, 20),
    )

    assert _feedback_value(batch, ACTION_FEEDBACK_TARGET_NEAR_ENEMY_OFFSET) == 0.0
    assert _feedback_value(batch, ACTION_FEEDBACK_MOVED_TOWARD_TARGET_OFFSET) == 0.0
    assert _feedback_value(batch, ACTION_FEEDBACK_ENEMY_HEALTH_DROP_OFFSET) == 0.0
    assert _feedback_value(batch, ACTION_FEEDBACK_FRIENDLY_HEALTH_DROP_OFFSET) == 0.0


def test_smart_near_enemy_sets_target_near_enemy(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=10, y=10, tag=1),
                _unit(alliance=4, health=100, x=30, y=30, tag=2),
            ],
        ),
    )

    batch = extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=10, y=10, tag=1),
                _unit(alliance=4, health=100, x=30, y=30, tag=2),
            ],
        ),
        last_action_token=_smart_token(33, 30),
    )

    assert _feedback_value(batch, ACTION_FEEDBACK_TARGET_NEAR_ENEMY_OFFSET) == 1.0


def test_tag_matched_surviving_marines_moving_toward_target_set_movement(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=10, y=10, tag=101),
                _unit(alliance=1, health=100, x=12, y=10, tag=102),
                _unit(alliance=4, health=100, x=40, y=10, tag=201),
            ],
        ),
    )

    batch = extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=14, y=10, tag=101),
                _unit(alliance=1, health=100, x=16, y=10, tag=102),
                _unit(alliance=4, health=100, x=40, y=10, tag=201),
            ],
        ),
        last_action_token=_smart_token(40, 10),
    )

    assert _feedback_value(batch, ACTION_FEEDBACK_MOVED_TOWARD_TARGET_OFFSET) == 1.0


def test_dead_tagged_marine_does_not_create_false_movement(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=10, y=10, tag=101),
                _unit(alliance=1, health=100, x=60, y=10, tag=102),
                _unit(alliance=4, health=100, x=10, y=10, tag=201),
            ],
        ),
    )

    batch = extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=10, y=10, tag=101),
                _unit(alliance=4, health=100, x=10, y=10, tag=201),
            ],
        ),
        last_action_token=_smart_token(10, 10),
    )

    assert _feedback_value(batch, ACTION_FEEDBACK_MOVED_TOWARD_TARGET_OFFSET) == 0.0


def test_tagless_unchanged_count_fallback_uses_median_position(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=10, y=10),
                _unit(alliance=1, health=100, x=20, y=10),
                _unit(alliance=4, health=100, x=40, y=10),
            ],
        ),
    )

    batch = extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=15, y=10),
                _unit(alliance=1, health=100, x=25, y=10),
                _unit(alliance=4, health=100, x=40, y=10),
            ],
        ),
        last_action_token=_smart_token(40, 10),
    )

    assert _feedback_value(batch, ACTION_FEEDBACK_MOVED_TOWARD_TARGET_OFFSET) == 1.0


def test_tagless_changed_count_fallback_suppresses_movement(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=10, y=10),
                _unit(alliance=1, health=100, x=60, y=10),
                _unit(alliance=4, health=100, x=10, y=10),
            ],
        ),
    )

    batch = extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=10, y=10),
                _unit(alliance=4, health=100, x=10, y=10),
            ],
        ),
        last_action_token=_smart_token(10, 10),
    )

    assert _feedback_value(batch, ACTION_FEEDBACK_MOVED_TOWARD_TARGET_OFFSET) == 0.0


def test_action_effect_health_drops_are_clipped_and_normalized(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=10, y=10, tag=101),
                _unit(alliance=4, health=150, x=20, y=20, tag=201),
            ],
        ),
    )

    batch = extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=50, x=10, y=10, tag=101),
                _unit(alliance=4, health=40, x=20, y=20, tag=201),
            ],
        ),
        last_action_token=_smart_token(20, 20),
    )

    assert _feedback_value(batch, ACTION_FEEDBACK_ENEMY_HEALTH_DROP_OFFSET) == pytest.approx(1.0)
    assert _feedback_value(batch, ACTION_FEEDBACK_FRIENDLY_HEALTH_DROP_OFFSET) == pytest.approx(0.5)


def test_peek_observation_does_not_consume_action_effect_state(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=10, y=10, tag=101),
                _unit(alliance=4, health=100, x=20, y=20, tag=201),
            ],
        ),
    )

    current = _obs_with_units(
        make_obs,
        [
            _unit(alliance=1, health=100, x=10, y=10, tag=101),
            _unit(alliance=4, health=80, x=20, y=20, tag=201),
        ],
    )
    peek_batch = extractor.peek_observation(
        current,
        last_action_token=_smart_token(20, 20),
    )
    actual_batch = extractor.extract_observation(
        current,
        last_action_token=_smart_token(20, 20),
    )

    assert _feedback_value(peek_batch, ACTION_FEEDBACK_ENEMY_HEALTH_DROP_OFFSET) == pytest.approx(0.2)
    assert _feedback_value(actual_batch, ACTION_FEEDBACK_ENEMY_HEALTH_DROP_OFFSET) == pytest.approx(0.2)


def test_action_effect_state_resets_between_episodes(make_obs):
    extractor = obs_space_2.ObservationExtractor()
    extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=10, y=10, tag=101),
                _unit(alliance=4, health=100, x=20, y=20, tag=201),
            ],
        ),
    )
    extractor.reset()

    batch = extractor.extract_observation(
        _obs_with_units(
            make_obs,
            [
                _unit(alliance=1, health=100, x=10, y=10, tag=101),
                _unit(alliance=4, health=80, x=20, y=20, tag=201),
            ],
        ),
        last_action_token=_smart_token(20, 20),
    )

    assert _feedback_value(batch, ACTION_FEEDBACK_TARGET_NEAR_ENEMY_OFFSET) == 0.0
    assert _feedback_value(batch, ACTION_FEEDBACK_ENEMY_HEALTH_DROP_OFFSET) == 0.0
