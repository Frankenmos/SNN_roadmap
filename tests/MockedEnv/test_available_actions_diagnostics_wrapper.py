import json
from types import SimpleNamespace

from Utility.available_actions_wrapper import AvailableActionsDiagnosticsWrapper


class _DummyEnv:
    def __init__(self, reset_obs, step_obs, action_spec):
        self._reset_obs = [reset_obs]
        self._step_obs = [step_obs]
        self._action_spec_value = [action_spec]

    def action_spec(self):
        return self._action_spec_value

    def reset(self, *args, **kwargs):
        del args, kwargs
        return self._reset_obs

    def step(self, actions):
        self.last_actions = actions
        return self._step_obs


def test_available_actions_diagnostics_wrapper_logs_previous_current_and_dispatch(
    tmp_path,
    make_obs,
    fake_actions,
):
    reset_obs = make_obs(
        available_actions={
            fake_actions.Attack_screen.id,
            fake_actions.Move_screen.id,
            fake_actions.select_army.id,
        },
        last_actions=[fake_actions.select_army.id],
    )
    step_obs = make_obs(
        available_actions={
            fake_actions.Move_screen.id,
            fake_actions.no_op.id,
        },
        last_actions=[fake_actions.Attack_screen.id],
    )
    action_spec = SimpleNamespace(
        functions={
            fake_actions.no_op.id: fake_actions.no_op,
            fake_actions.Attack_screen.id: fake_actions.Attack_screen,
            fake_actions.Move_screen.id: fake_actions.Move_screen,
            fake_actions.select_army.id: fake_actions.select_army,
        },
    )
    env = _DummyEnv(reset_obs, step_obs, action_spec)
    output_path = tmp_path / "available_actions_diagnostics.jsonl"
    wrapper = AvailableActionsDiagnosticsWrapper(
        env=env,
        output_path=str(output_path),
        log_every_n_steps=1,
    )

    wrapper.reset()
    wrapper.step(
        [
            SimpleNamespace(
                function=fake_actions.Attack_screen.id,
                arguments=["now", [20, 20]],
            )
        ],
    )

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    reset_record = json.loads(lines[0])
    step_record = json.loads(lines[1])

    assert reset_record["event"] == "reset"
    assert set(reset_record["current_frame"]["available_action_ids"]) == {
        fake_actions.Attack_screen.id,
        fake_actions.Move_screen.id,
        fake_actions.select_army.id,
    }

    assert step_record["event"] == "step"
    assert set(step_record["previous_frame"]["available_action_ids"]) == {
        fake_actions.Attack_screen.id,
        fake_actions.Move_screen.id,
        fake_actions.select_army.id,
    }
    assert set(step_record["current_frame"]["available_action_ids"]) == {
        fake_actions.Move_screen.id,
        fake_actions.no_op.id,
    }
    assert step_record["dispatched_action"]["function_id"] == fake_actions.Attack_screen.id
    assert step_record["dispatched_action"]["function_name"] == "Attack_screen"
    assert step_record["dispatched_action_in_previous_available"] is True
