from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import torch

import train as run_mod

from MockedEnv.policy_batch import make_dummy_state, make_policy_batch
from agent_core.policy_protocol import (
    META_VECTOR_DIM,
)
from agent_core.rewards import build_reward_function
from agent_core.rewards.defeat_roaches_v2 import RewardFunctionV2
from agent_core.rewards.defeat_roaches_v3 import RewardFunctionV3
from obs_space.obs_space_2 import get_friendly_health


class DummyReward:
    def calculate_reward(self, obs, _):
        return 1.0

    def get_last_reward_components(self):
        return None

    def reset(self):
        return None


class DummyPPO:
    def __init__(self):
        self.memory = []
        self.final_next = None
        self.pending_fragments = []
        self.update_count = 0
        self.store_calls = 0
        self.final_next_calls = 0

    def store_transition(self, *args, **kwargs):
        self.store_calls += 1
        self.memory.append(1)

    def set_final_next(self, *args, **kwargs):
        self.final_next_calls += 1
        self.final_next = True

    def finalize_fragment(self, **kwargs):
        del kwargs
        if not self.memory:
            return None
        fragment = list(self.memory)
        self.pending_fragments.append(fragment)
        self.memory.clear()
        self.final_next = None
        return fragment

    def consume_pending_fragments(self):
        fragments = list(self.pending_fragments)
        self.pending_fragments.clear()
        return fragments

    def pending_rollout_steps(self, include_current=True):
        steps = sum(len(fragment) for fragment in self.pending_fragments)
        if include_current:
            steps += len(self.memory)
        return steps

    def has_pending_rollout(self):
        return bool(self.memory or self.pending_fragments)


class DummyPolicy:
    device = "cpu"


def _dummy_state():
    return make_dummy_state()


def _dummy_batch():
    batch = make_policy_batch(batch_size=1, meta_dim=META_VECTOR_DIM, zeros=True)
    return batch.with_state(None)


class DummyAgent:
    def __init__(self):
        self.policy = DummyPolicy()
        self.ppo = DummyPPO()
        self.reward_function = DummyReward()
        self._step = 0
        self.update_calls = 0
        self.snn_state = _dummy_state()

    def reset(self):
        self._step = 0
        self.reward_function.reset()
        self.snn_state = _dummy_state()

    def peek_observation(self, obs):
        return _dummy_batch()

    def step(self, obs):
        self._step += 1
        self.snn_state = _dummy_state()
        return (
            "noop",
            0,
            0,
            0,
            _dummy_state(),
            0.0,
            0.0,
            _dummy_batch().with_state(_dummy_state()),
            True,
        )

    def update_policy(self, fragments=None):
        del fragments
        self.update_calls += 1
        self.ppo.memory.clear()
        self.ppo.pending_fragments.clear()
        self.ppo.update_count += 1
        return {
            "mean_policy_loss": 0.0,
            "mean_value_loss": 0.0,
            "mean_entropy": 0.0,
            "mean_kl": 0.0,
            "clip_fraction": 0.0,
            "explained_variance": 1.0,
            "grad_norm": 0.0,
            "lr": 1e-4,
            "nonfinite_grad_steps": 0,
            "skipped_optimizer_steps": 0,
            "transitions_in_update": 4,
            "learnable_transitions_in_update": 4,
            "fragments_in_update": 1,
            "return_mean": 0.0,
            "return_std": 0.0,
            "return_p10": 0.0,
            "return_p50": 0.0,
            "return_p90": 0.0,
            "epochs_ran": 1,
        }


class DummyObs:
    def __init__(self, max_steps, is_last=False):
        self.max_steps = max_steps
        self._last = is_last
        self.reward = 0.0

    def last(self):
        return self._last


class DummyEnv:
    def __init__(self, episode_lengths):
        self.episode_lengths = list(episode_lengths)
        self.episode_idx = -1
        self.step_in_episode = 0
        self.current_max = None

    def reset(self):
        self.episode_idx += 1
        self.current_max = self.episode_lengths[self.episode_idx]
        self.step_in_episode = 0
        return [DummyObs(self.current_max, is_last=False)]

    def step(self, actions):
        self.step_in_episode += 1
        return [
            DummyObs(
                self.current_max,
                is_last=self.step_in_episode >= self.current_max,
            )
        ]


class DummyQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _make_reward_obs(friendly_health, enemy_health=100, enemy_count=1, last=False):
    friendly = SimpleNamespace(
        alliance=1,
        health=friendly_health,
        x=0,
        y=0,
        unit_type=48,
        attack_range=5,
    )
    enemies = [
        SimpleNamespace(
            alliance=4,
            health=enemy_health // enemy_count,
            x=10,
            y=10,
            unit_type=110,
            attack_range=5,
        )
        for _ in range(enemy_count)
    ]
    return SimpleNamespace(
        observation=SimpleNamespace(
            player=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            feature_units=[friendly] + enemies,
            score_cumulative=[0] * 13,
        ),
        reward=0,
        last=lambda: last,
    )


def test_get_friendly_health_reads_feature_units():
    assert get_friendly_health(_make_reward_obs(100)) == 100.0
    assert get_friendly_health(_make_reward_obs(90)) == 90.0


def test_reward_health_penalty_fires_on_health_drop():
    reward_fn = RewardFunctionV2()

    reward_fn.calculate_reward(_make_reward_obs(100), None)
    total_reward = reward_fn.calculate_reward(_make_reward_obs(90), None)
    components = reward_fn.get_last_reward_components()

    assert total_reward == -4.0
    assert components["health_reward"] == -4.0
    assert components["engagement_reward"] == 0.0


def test_reward_v3_uses_enemy_count_for_terminal_win_detection():
    reward_fn = RewardFunctionV3(
        damage_dealt_coef=0.0,
        damage_taken_coef=0.0,
        kill_reward_coef=0.0,
        win_reward=60.0,
        loss_penalty=30.0,
        step_penalty=0.0,
        distance_reward_coef=0.0,
        distance_hold_bonus=0.0,
    )

    win_obs = SimpleNamespace(
        observation=SimpleNamespace(
            player=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            feature_units=[
                SimpleNamespace(
                    alliance=1,
                    health=100,
                    x=0,
                    y=0,
                    unit_type=48,
                    attack_range=5,
                ),
            ],
            score_cumulative=[0] * 13,
        ),
        reward=0,
        last=lambda: True,
    )

    total_reward = reward_fn.calculate_reward(win_obs, None)
    components = reward_fn.get_last_reward_components()

    assert total_reward == 60.0
    assert components["end_of_episode_reward"] == 60.0


def test_reward_v3_positioning_reward_is_positive_when_entering_band():
    reward_fn = RewardFunctionV3(
        damage_dealt_coef=0.0,
        damage_taken_coef=0.0,
        kill_reward_coef=0.0,
        win_reward=0.0,
        loss_penalty=0.0,
        step_penalty=0.0,
        target_distance=9.0,
        distance_band_low=7.0,
        distance_band_high=11.0,
        distance_reward_coef=0.5,
        distance_reward_clip=2.0,
        distance_hold_bonus=0.1,
        distance_gate=18.0,
    )

    far_obs = _make_reward_obs(100, enemy_health=100, enemy_count=1, last=False)
    close_obs = SimpleNamespace(
        observation=SimpleNamespace(
            player=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            feature_units=[
                SimpleNamespace(
                    alliance=1,
                    health=100,
                    x=0,
                    y=0,
                    unit_type=48,
                    attack_range=5,
                ),
                SimpleNamespace(
                    alliance=4,
                    health=100,
                    x=9,
                    y=0,
                    unit_type=110,
                    attack_range=5,
                ),
            ],
            score_cumulative=[0] * 13,
        ),
        reward=0,
        last=lambda: False,
    )

    reward_fn.calculate_reward(far_obs, None)
    total_reward = reward_fn.calculate_reward(close_obs, None)
    components = reward_fn.get_last_reward_components()

    assert components["positioning_reward"] > 0.0
    assert total_reward > 0.0


def test_reward_factory_builds_v3():
    reward_fn = build_reward_function("defeat_roaches_v3", step_penalty=0.0)
    assert isinstance(reward_fn, RewardFunctionV3)


def test_rollout_budget_triggers_update_on_transition_count(monkeypatch):
    env = DummyEnv([2, 2])
    agent = DummyAgent()
    queue = DummyQueue()

    monkeypatch.setattr(run_mod.cfg.environment, "total_episodes", 2, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "steps_per_episode", 10, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "reward_window", 10, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "log_frequency", 999, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "eval_frequency", 0, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "eval_episodes", 0, raising=False)
    monkeypatch.setattr(run_mod.cfg.hyperparameters, "rollout_steps", 4, raising=False)
    monkeypatch.setattr(run_mod.cfg.hyperparameters, "reward_scale", 1.0, raising=False)

    with patch.object(
        run_mod,
        "load_checkpoint",
        return_value=(0, float("-inf"), deque(maxlen=10)),
    ):
        best = run_mod.train_agent(env, agent, None, queue)

    assert best == float("-inf")
    assert agent.update_calls == 1
    assert agent.ppo.memory == []


def test_train_agent_flushes_partial_rollout_at_shutdown(monkeypatch):
    env = DummyEnv([2])
    agent = DummyAgent()
    queue = DummyQueue()

    monkeypatch.setattr(run_mod.cfg.environment, "total_episodes", 1, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "steps_per_episode", 10, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "reward_window", 10, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "log_frequency", 999, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "eval_frequency", 0, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "eval_episodes", 0, raising=False)
    monkeypatch.setattr(run_mod.cfg.hyperparameters, "rollout_steps", 4, raising=False)
    monkeypatch.setattr(run_mod.cfg.hyperparameters, "reward_scale", 1.0, raising=False)

    with patch.object(
        run_mod,
        "load_checkpoint",
        return_value=(0, float("-inf"), deque(maxlen=10)),
    ):
        best = run_mod.train_agent(env, agent, None, queue)

    assert best == float("-inf")
    assert agent.update_calls == 1
    assert agent.ppo.memory == []


def test_train_agent_stores_helper_steps_for_recurrent_replay(monkeypatch):
    class HelperMixAgent(DummyAgent):
        def step(self, obs):
            self._step += 1
            self.snn_state = _dummy_state()
            learnable = self._step != 1
            return (
                "noop",
                0,
                0,
                0,
                _dummy_state(),
                0.0,
                0.0,
                _dummy_batch().with_state(_dummy_state()),
                learnable,
            )

    env = DummyEnv([2])
    agent = HelperMixAgent()
    queue = DummyQueue()

    monkeypatch.setattr(run_mod.cfg.environment, "total_episodes", 1, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "steps_per_episode", 10, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "reward_window", 10, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "log_frequency", 999, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "eval_frequency", 0, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "eval_episodes", 0, raising=False)
    monkeypatch.setattr(run_mod.cfg.hyperparameters, "rollout_steps", 4, raising=False)
    monkeypatch.setattr(run_mod.cfg.hyperparameters, "reward_scale", 1.0, raising=False)

    with patch.object(
        run_mod,
        "load_checkpoint",
        return_value=(0, float("-inf"), deque(maxlen=10)),
    ):
        best = run_mod.train_agent(env, agent, None, queue)

    assert best == float("-inf")
    assert agent.ppo.store_calls == 2
    assert agent.ppo.final_next_calls == 2


def test_train_agent_flushes_rollout_inside_long_episode(monkeypatch):
    env = DummyEnv([6])
    agent = DummyAgent()
    queue = DummyQueue()

    monkeypatch.setattr(run_mod.cfg.environment, "total_episodes", 1, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "steps_per_episode", 10, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "reward_window", 10, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "log_frequency", 999, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "eval_frequency", 0, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "eval_episodes", 0, raising=False)
    monkeypatch.setattr(run_mod.cfg.hyperparameters, "rollout_steps", 4, raising=False)
    monkeypatch.setattr(run_mod.cfg.hyperparameters, "reward_scale", 1.0, raising=False)

    with patch.object(
        run_mod,
        "load_checkpoint",
        return_value=(0, float("-inf"), deque(maxlen=10)),
    ):
        best = run_mod.train_agent(env, agent, None, queue)

    assert best == float("-inf")
    assert agent.update_calls == 2
    assert agent.ppo.memory == []


def test_train_agent_skips_bootstrap_steps_outside_ppo_memory(monkeypatch):
    class BootstrapAgent(DummyAgent):
        def step(self, obs):
            self._step += 1
            self.snn_state = _dummy_state()
            if self._step == 1:
                return (
                    "select_army",
                    None,
                    0,
                    0,
                    _dummy_state(),
                    0.0,
                    0.0,
                    None,
                    False,
                )
            return (
                "noop",
                1,
                3,
                4,
                _dummy_state(),
                0.0,
                0.0,
                _dummy_batch().with_state(_dummy_state()),
                True,
            )

    env = DummyEnv([2])
    agent = BootstrapAgent()
    queue = DummyQueue()

    monkeypatch.setattr(run_mod.cfg.environment, "total_episodes", 1, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "steps_per_episode", 10, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "reward_window", 10, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "log_frequency", 999, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "eval_frequency", 0, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "eval_episodes", 0, raising=False)
    monkeypatch.setattr(run_mod.cfg.hyperparameters, "rollout_steps", 4, raising=False)
    monkeypatch.setattr(run_mod.cfg.hyperparameters, "reward_scale", 1.0, raising=False)

    with patch.object(
        run_mod,
        "load_checkpoint",
        return_value=(0, float("-inf"), deque(maxlen=10)),
    ):
        best = run_mod.train_agent(env, agent, None, queue)

    assert best == float("-inf")
    assert agent.ppo.store_calls == 1
    assert agent.ppo.final_next_calls == 1


def test_train_agent_updates_before_eval_and_best_checkpoint(monkeypatch):
    env = DummyEnv([1])
    agent = DummyAgent()
    queue = DummyQueue()
    call_order = []

    monkeypatch.setattr(run_mod.cfg.environment, "total_episodes", 1, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "steps_per_episode", 10, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "reward_window", 10, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "log_frequency", 999, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "eval_frequency", 1, raising=False)
    monkeypatch.setattr(run_mod.cfg.environment, "eval_episodes", 1, raising=False)
    monkeypatch.setattr(run_mod.cfg.hyperparameters, "rollout_steps", 1, raising=False)
    monkeypatch.setattr(run_mod.cfg.hyperparameters, "reward_scale", 1.0, raising=False)

    def _fake_update(agent, log_queue, episode_index):
        del log_queue, episode_index
        call_order.append("update")
        agent.ppo.memory.clear()
        return None

    def _fake_eval(**kwargs):
        del kwargs
        call_order.append("eval")
        return {
            "num_episodes": 1,
            "mean_reward": 1.0,
            "std_reward": 0.0,
            "min_reward": 1.0,
            "max_reward": 1.0,
            "deterministic": True,
        }

    def _fake_best(*args, **kwargs):
        del args, kwargs
        call_order.append("best")
        return 1.0

    with patch.object(
        run_mod,
        "load_checkpoint",
        return_value=(0, float("-inf"), deque(maxlen=10)),
    ), patch.object(
        run_mod,
        "maybe_run_policy_update",
        side_effect=_fake_update,
    ), patch.object(
        run_mod,
        "run_eval_sweep",
        side_effect=_fake_eval,
    ), patch.object(
        run_mod,
        "maybe_save_best_checkpoint",
        side_effect=_fake_best,
    ):
        best = run_mod.train_agent(env, agent, None, queue)

    assert best == 1.0
    assert call_order[:3] == ["update", "eval", "best"]
