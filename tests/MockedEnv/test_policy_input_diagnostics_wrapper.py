import json
from types import SimpleNamespace

import numpy as np

from agent_core.policy_protocol import (
    ACTION_FEEDBACK_FIELD_NAMES,
    ACTION_FEEDBACK_TOKEN_DIM,
    META_VECTOR_DIM,
)
from MockedEnv.fake_pysc2 import build_mock_obs
from Utility.policy_input_diagnostics_wrapper import PolicyInputDiagnosticsWrapper


class DummyEnv:
    def __init__(self, timesteps):
        self.timesteps = list(timesteps)
        self.idx = -1

    def reset(self, *args, **kwargs):
        self.idx = 0
        return [self.timesteps[self.idx]]

    def step(self, *args, **kwargs):
        self.idx += 1
        return [self.timesteps[self.idx]]


def _to_timestep(obs):
    return SimpleNamespace(observation=obs.observation)


def test_policy_input_diagnostics_wrapper_logs_raw_and_batch_fields(
    tmp_path,
    fake_actions,
):
    reset_obs = build_mock_obs(
        fake_actions=fake_actions,
        multi_select=np.asarray([[48, 1, 45, 0, 0, 0, 100]], dtype=np.int32),
        last_actions=np.asarray([fake_actions.select_army.id], dtype=np.int32),
    )
    step_obs = build_mock_obs(
        fake_actions=fake_actions,
        multi_select=np.asarray(
            [
                [48, 1, 45, 0, 0, 0, 100],
                [48, 1, 30, 0, 0, 0, 100],
            ],
            dtype=np.int32,
        ),
        last_actions=np.asarray([fake_actions.Smart_screen.id], dtype=np.int32),
    )
    env = DummyEnv([_to_timestep(reset_obs), _to_timestep(step_obs)])
    output_path = tmp_path / "policy_input_diagnostics.jsonl"

    wrapped = PolicyInputDiagnosticsWrapper(
        env=env,
        output_path=str(output_path),
        log_every_n_steps=1,
    )

    wrapped.reset()
    wrapped.step([fake_actions.no_op()])

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 2
    first = records[0]
    second = records[1]

    assert set(first["raw"]["available_action_ids"]) == {
        fake_actions.Smart_screen.id,
        fake_actions.select_army.id,
    }
    assert first["raw"]["last_action_ids"] == [fake_actions.select_army.id]
    assert first["raw"]["selection_source"] == "multi_select"
    assert first["batch"]["entity_count"] >= 2
    assert first["batch"]["selection_count"] == 1
    assert first["batch"]["meta_last_action_index"] > 0
    assert second["batch"]["selection_count"] == 2
    assert second["batch"]["meta_available_action_mask_active"] == 2
    assert second["batch"]["meta_dim"] == META_VECTOR_DIM
    assert second["batch"]["action_feedback_token_dim"] == ACTION_FEEDBACK_TOKEN_DIM
    assert len(second["batch"]["action_feedback_token"]) == ACTION_FEEDBACK_TOKEN_DIM
    assert second["batch"]["action_feedback_provenance"] == (
        "wrapper_local_extractor_without_agent_last_action_token"
    )
    assert set(second["batch"]["action_feedback_named"]) == set(
        ACTION_FEEDBACK_FIELD_NAMES,
    )
    assert "action_feedback_effect_class" not in second["batch"]
    assert "not agent-faithful" in second["diagnostic_scope"]["action_feedback"]
