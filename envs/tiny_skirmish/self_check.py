from __future__ import annotations

import numpy as np

from .env import TinySkirmishEnv
from .protocol import (
    ACTION_LEFT_CLICK,
    ACTION_NO_OP,
    ACTION_RIGHT_CLICK,
    ACTION_FEEDBACK_TOKEN_DIM,
    ENTITY_FEATURE_DIM,
    MAX_ENTITY_TOKENS,
    MAX_SELECTION_TOKENS,
    META_VECTOR_DIM,
    SCREEN_SIZE,
    SELECTION_FEATURE_DIM,
    SPATIAL_CHANNELS,
    SkirmishAction,
)
from .rollout import scripted_action


def _assert_shapes() -> None:
    env = TinySkirmishEnv(seed=11)
    obs = env.reset()
    obs.validate()
    assert obs.spatial_obs.shape == (SPATIAL_CHANNELS, SCREEN_SIZE, SCREEN_SIZE)
    assert obs.entity_features.shape == (MAX_ENTITY_TOKENS, ENTITY_FEATURE_DIM)
    assert obs.entity_mask.shape == (MAX_ENTITY_TOKENS,)
    assert obs.selection_features.shape == (MAX_SELECTION_TOKENS, SELECTION_FEATURE_DIM)
    assert obs.selection_mask.shape == (MAX_SELECTION_TOKENS,)
    assert obs.action_feedback_tokens.shape == (1, ACTION_FEEDBACK_TOKEN_DIM)
    assert obs.meta_vec.shape == (META_VECTOR_DIM,)


def _assert_reset_determinism() -> None:
    left = TinySkirmishEnv(seed=123).reset()
    right = TinySkirmishEnv(seed=123).reset()
    assert np.array_equal(left.spatial_obs, right.spatial_obs)
    assert np.array_equal(left.entity_features, right.entity_features)
    assert np.array_equal(left.meta_vec, right.meta_vec)


def _assert_actions_are_handled() -> None:
    env = TinySkirmishEnv(seed=3)
    env.reset()
    actions = [
        SkirmishAction(ACTION_NO_OP, 0, 0),
        SkirmishAction(ACTION_LEFT_CLICK, 10, 10),
        SkirmishAction(ACTION_RIGHT_CLICK, 70, 35),
    ]
    for action in actions:
        result = env.step(action)
        result.observation.validate()
        result.reward.validate()


def _assert_scripted_rollout_rewards_are_legible() -> None:
    env = TinySkirmishEnv(seed=9, max_steps=80)
    env.reset()
    saw_named_positive_event = False
    saw_nonzero_reward = False
    for _ in range(80):
        result = env.step(scripted_action(env))
        result.reward.validate()
        if abs(result.reward.total) > 1.0e-9:
            saw_nonzero_reward = True
            assert result.reward.compact_parts()
        if {"damage_dealt", "kill", "win"} & set(result.reward.events):
            saw_named_positive_event = True
        if result.done or result.truncated:
            break
    assert saw_nonzero_reward
    assert saw_named_positive_event


def _assert_random_rollouts_do_not_crash() -> None:
    for seed in range(5):
        env = TinySkirmishEnv(seed=seed, max_steps=30)
        env.reset()
        for _ in range(30):
            result = env.step(env.random_action())
            result.observation.validate()
            result.reward.validate()
            if result.done or result.truncated:
                break


def main() -> int:
    checks = [
        _assert_shapes,
        _assert_reset_determinism,
        _assert_actions_are_handled,
        _assert_scripted_rollout_rewards_are_legible,
        _assert_random_rollouts_do_not_crash,
    ]
    for check in checks:
        check()
        print(f"ok {check.__name__}")
    print("TinySkirmish self-check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
