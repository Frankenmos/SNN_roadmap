import pytest

from MockedEnv.fake_pysc2 import build_mock_obs, install_fake_pysc2


install_fake_pysc2()


@pytest.fixture
def fake_actions():
    from pysc2.lib import actions

    return actions.FUNCTIONS


@pytest.fixture
def make_obs(fake_actions):
    def _make_obs(**kwargs):
        return build_mock_obs(fake_actions=fake_actions, **kwargs)

    return _make_obs
