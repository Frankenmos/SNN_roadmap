import json
from types import SimpleNamespace

from Utility.feedback_diagnostics_wrapper import (
    LastActionDiagnosticsWrapper,
    ScoreDiagnosticsWrapper,
)


class _DummyEnv:
    def __init__(self, reset_obs, step_obs, action_spec=None):
        self._reset_obs = [reset_obs]
        self._step_obs = [step_obs]
        self._action_spec_value = [action_spec or SimpleNamespace(functions={})]

    def action_spec(self):
        return self._action_spec_value

    def reset(self, *args, **kwargs):
        del args, kwargs
        return self._reset_obs

    def step(self, actions):
        self.last_actions = actions
        return self._step_obs


def test_last_action_diagnostics_logs_action_result_alerts_and_dispatch(
    tmp_path,
    make_obs,
    fake_actions,
):
    reset_obs = make_obs(
        available_actions={fake_actions.Smart_screen.id, fake_actions.select_army.id},
        last_actions=[],
        action_result=[],
        alerts=[],
        game_loop=[0],
    )
    step_obs = make_obs(
        available_actions={fake_actions.Smart_screen.id},
        last_actions=[fake_actions.Smart_screen.id],
        action_result=[0],
        alerts=[1],
        game_loop=[6],
    )
    action_spec = SimpleNamespace(
        functions={
            fake_actions.Smart_screen.id: fake_actions.Smart_screen,
            fake_actions.select_army.id: fake_actions.select_army,
        },
    )
    output_path = tmp_path / "last_action_diagnostics.jsonl"
    wrapper = LastActionDiagnosticsWrapper(
        env=_DummyEnv(reset_obs, step_obs, action_spec),
        output_path=str(output_path),
        log_every_n_steps=1,
    )

    wrapper.reset()
    wrapper.step(
        [
            SimpleNamespace(
                function=fake_actions.Smart_screen.id,
                arguments=[[0], [20, 20]],
            ),
        ],
    )

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 2
    step_record = records[1]
    assert step_record["current_frame"]["last_action_ids"] == [
        fake_actions.Smart_screen.id,
    ]
    assert step_record["current_frame"]["last_action_names"] == ["Smart_screen"]
    assert step_record["current_frame"]["action_result"] == [0]
    assert step_record["current_frame"]["alerts"] == [1]
    assert step_record["dispatched_action"]["function_id"] == fake_actions.Smart_screen.id
    assert step_record["dispatched_action_in_previous_available"] is True
    assert (
        step_record["feedback_summary"][
            "dispatched_function_seen_in_current_last_actions"
        ]
        is True
    )


def test_score_diagnostics_logs_score_delta_and_episode_reward(
    tmp_path,
    make_obs,
):
    reset_obs = make_obs(
        score_cumulative=[10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        reward=0.0,
        game_loop=[0],
    )
    step_obs = make_obs(
        score_cumulative=[15, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
        reward=2.5,
        game_loop=[6],
    )
    output_path = tmp_path / "score_diagnostics.jsonl"
    wrapper = ScoreDiagnosticsWrapper(
        env=_DummyEnv(reset_obs, step_obs),
        output_path=str(output_path),
        log_every_n_steps=1,
    )

    wrapper.reset()
    wrapper.step([SimpleNamespace(function=0, arguments=[])])

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 2
    reset_record = records[0]
    step_record = records[1]
    assert reset_record["score_delta"] is None
    assert step_record["score_total"] == 15.0
    assert step_record["score_total_delta"] == 5.0
    assert step_record["score_delta"][5] == 5.0
    assert set(step_record["score_nonzero_delta_indices"]) == {0, 5}
    assert step_record["score_cumulative_named"]["score"] == 15.0
    assert step_record["score_delta_named"]["killed_value_units"] == 5.0
    assert step_record["reward"] == 2.5
    assert step_record["episode_reward"] == 2.5
