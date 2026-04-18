from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import PPO_CNN_run as run_mod

from PPO_CNN.reward_function_2 import RewardFunctionV2
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

    def store_transition(self, *args, **kwargs):
        self.memory.append(1)

    def set_final_next(self, *args, **kwargs):
        self.final_next = True


class DummyPolicy:
    device = "cpu"


class DummyAgent:
    def __init__(self):
        self.policy = DummyPolicy()
        self.ppo = DummyPPO()
        self.reward_function = DummyReward()
        self._step = 0
        self.update_calls = 0
        self.snn_state = ("syn_next", "mem_next")

    def reset(self):
        self._step = 0
        self.reward_function.reset()
        self.snn_state = ("syn_next", "mem_next")

    def peek_observation(self, obs):
        return ("next_spatial", "next_vector")

    def step(self, obs):
        self._step += 1
        self.snn_state = ("syn_next", "mem_next")
        return (
            "noop",
            0,
            0,
            0,
            ("syn", "mem"),
            0.0,
            0.0,
            "spatial",
            "vector",
            True,
        )

    def update_policy(self):
        self.update_calls += 1
        self.ppo.memory.clear()
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
