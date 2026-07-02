import torch

import eval as eval_mod

from MockedEnv.policy_batch import make_policy_batch
from agent_core.policy_protocol import (
    ACTION_FEEDBACK_TOKEN_COUNT,
    ACTION_FEEDBACK_TOKEN_DIM,
    META_VECTOR_DIM,
    POLICY_INPUT_SCHEMA,
    POLICY_PROTOCOL_VERSION,
    SPATIAL_OBS_SHAPE,
)


class _DummyPolicy:
    def __init__(self):
        self.device = "cpu"
        self.loaded_state = None
        self.was_eval = False

    def load_state_dict(self, state):
        self.loaded_state = state

    def eval(self):
        self.was_eval = True


class _DummyExtractor:
    def __init__(self):
        self.loaded_state = None

    def load_state_dict(self, state):
        self.loaded_state = state


class _DummyAgent:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.policy = _DummyPolicy()
        self.extractor = _DummyExtractor()
        self._step = 0

    def reset(self):
        self._step = 0

    def step(self, obs, deterministic=True):
        del obs, deterministic
        self._step += 1
        batch = make_policy_batch(
            batch_size=1,
            spatial_shape=SPATIAL_OBS_SHAPE,
            meta_dim=META_VECTOR_DIM,
            zeros=True,
        ).with_state(None)
        if self._step == 1:
            from pysc2.lib import actions

            return (
                actions.FUNCTIONS.Smart_screen("now", [12, 34]),
                1,
                12,
                34,
                None,
                -0.25,
                1.5,
                batch,
                True,
            )
        from pysc2.lib import actions

        return (
            actions.FUNCTIONS.no_op(),
            0,
            0,
            0,
            None,
            -0.10,
            0.75,
            batch,
            True,
        )


class _DummyObservationExtractor:
    def get_observation_dimensions(self, obs):
        del obs
        return SPATIAL_OBS_SHAPE, META_VECTOR_DIM


class _DummyTimeStep:
    def __init__(self, reward=0.0, is_last=False):
        self.reward = reward
        self._is_last = is_last

    def last(self):
        return self._is_last


class _DummyEnv:
    def __init__(self):
        self.step_in_episode = 0

    def reset(self):
        self.step_in_episode = 0
        return [_DummyTimeStep(reward=0.0, is_last=False)]

    def step(self, action_list):
        del action_list
        self.step_in_episode += 1
        return [
            _DummyTimeStep(
                reward=1.0,
                is_last=self.step_in_episode >= 2,
            )
        ]

    def close(self):
        return None


def test_eval_play_can_write_episode_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_mod, "create_env", lambda **kwargs: _DummyEnv())
    monkeypatch.setattr(eval_mod, "ObservationExtractor", _DummyObservationExtractor)
    monkeypatch.setattr(eval_mod, "DefeatRoaches", _DummyAgent)
    monkeypatch.setattr(
        eval_mod,
        "_load_checkpoint_state",
        lambda checkpoint_path, device: {
            "agent_state": {"dummy": 1},
            "extractor_state": {"dummy": 2},
            "episode": 123,
            "policy_protocol_version": POLICY_PROTOCOL_VERSION,
            "policy_input_schema": POLICY_INPUT_SCHEMA,
        },
    )
    monkeypatch.setattr(eval_mod.cfg.environment, "steps_per_episode", 10, raising=False)
    monkeypatch.setattr(eval_mod.cfg.environment, "map_name", "DefeatRoaches", raising=False)
    monkeypatch.setattr(eval_mod.cfg.model, "action_dim", 2, raising=False)

    eval_mod.play(
        checkpoint_path="models/dummy/checkpoint.pth",
        episodes=1,
        visualize=False,
        deterministic=True,
        trace_episodes=1,
        trace_output_dir=str(tmp_path),
        run_name="dummy-run",
    )

    trace_files = sorted(tmp_path.glob("*.pt"))
    assert len(trace_files) == 1

    payload = torch.load(trace_files[0], map_location="cpu", weights_only=False)
    assert payload["format_version"] == 1
    assert payload["run_name"] == "dummy-run"
    assert payload["checkpoint_episode"] == 123
    assert payload["deterministic"] is True
    assert payload["steps"] == 2
    assert payload["total_reward"] == 2.0

    records = payload["records"]
    assert len(records) == 2
    assert records[0]["action"] == 1
    assert records[0]["dispatched_action"]["function_name"] == "Smart_screen"
    assert tuple(records[0]["policy_input"]["spatial_obs"].shape) == SPATIAL_OBS_SHAPE
    assert tuple(records[0]["policy_input"]["action_feedback_tokens"].shape) == (
        ACTION_FEEDBACK_TOKEN_COUNT,
        ACTION_FEEDBACK_TOKEN_DIM,
    )
    assert tuple(records[0]["policy_input"]["meta_vec"].shape) == (META_VECTOR_DIM,)
    assert records[1]["action"] == 0
    assert records[1]["dispatched_action"]["function_name"] == "no_op"


def test_eval_diagnostic_paths_split_by_mode():
    det_path = eval_mod._resolve_eval_jsonl_path(
        None,
        enabled=True,
        filename="score_diagnostics.jsonl",
        run_name="banana",
        deterministic=True,
        split_by_mode=True,
    )
    stoch_path = eval_mod._resolve_eval_jsonl_path(
        None,
        enabled=True,
        filename="score_diagnostics.jsonl",
        run_name="banana",
        deterministic=False,
        split_by_mode=True,
    )
    explicit_path = eval_mod._resolve_eval_jsonl_path(
        "analysis_results/banana/custom.jsonl",
        enabled=True,
        filename="score_diagnostics.jsonl",
        run_name="banana",
        deterministic=False,
        split_by_mode=True,
    )
    already_split = eval_mod._resolve_eval_jsonl_path(
        "analysis_results/banana/custom_det.jsonl",
        enabled=True,
        filename="score_diagnostics.jsonl",
        run_name="banana",
        deterministic=True,
        split_by_mode=True,
    )

    assert det_path == "analysis_results/banana/score_diagnostics_det.jsonl"
    assert stoch_path == "analysis_results/banana/score_diagnostics_stoch.jsonl"
    assert explicit_path == "analysis_results/banana/custom_stoch.jsonl"
    assert already_split == "analysis_results/banana/custom_det.jsonl"
