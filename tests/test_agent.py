import numpy as np
import torch

from MockedEnv.policy_batch import make_policy_batch
from PPO_CNN.policy_input import (
    META_VECTOR_DIM,
    PolicyInputBatch,
    SPATIAL_OBS_SHAPE,
)
from PPO_CNN.policy_network import (
    EntityEncoder,
    MetaEncoder,
    PolicyNetwork,
    SelectionEncoder,
)
from PPO_CNN_agent import DefeatRoaches


def _small_policy():
    net = PolicyNetwork(
        SPATIAL_OBS_SHAPE,
        vector_input_dim=META_VECTOR_DIM,
        action_dim=3,
        num_steps=2,
        screen_size=16,
        attention_embed_dim=32,
        attention_pool_size=4,
    )
    net.device = torch.device("cpu")
    net.to("cpu")
    net.use_amp = False
    net.amp_dtype = torch.float32
    net.scaler = torch.amp.GradScaler("cuda", enabled=False)
    return net


def _policy_batch(
    batch_size=2,
    spatial_shape=SPATIAL_OBS_SHAPE,
    meta_dim=META_VECTOR_DIM,
):
    return make_policy_batch(
        batch_size=batch_size,
        spatial_shape=spatial_shape,
        meta_dim=meta_dim,
        with_state=True,
        state_shape=(
            batch_size,
            4 * 4 + 24 + 20 + 1,
            32,
        ),
    )


def test_agent_step_returns_current_training_tuple(make_obs):
    agent = DefeatRoaches()
    obs = make_obs()

    (
        action_func,
        action,
        move_x,
        move_y,
        pre_step_state,
        log_prob,
        value,
        policy_input,
        learnable,
    ) = agent.step(obs)

    assert action_func.name in {
        "no_op",
        "Attack_screen",
        "Move_screen",
        "select_army",
    }
    assert action in {0, 1, 2}
    assert 0 <= move_x < 84
    assert 0 <= move_y < 84
    assert len(pre_step_state) == 2
    assert pre_step_state[0].shape[0] == 1
    assert pre_step_state[1].shape == pre_step_state[0].shape
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    assert isinstance(policy_input, PolicyInputBatch)
    assert tuple(policy_input.spatial_obs.shape) == (1, 27, 84, 84)
    assert tuple(policy_input.entity_features.shape) == (1, 24, 21)
    assert tuple(policy_input.selection_features.shape) == (1, 20, 7)
    assert tuple(policy_input.meta_vec.shape) == (1, META_VECTOR_DIM)
    assert policy_input.state_in is not None
    assert isinstance(learnable, bool)


def test_attack_targets_nearest_enemy_unit_center(make_obs, monkeypatch):
    agent = DefeatRoaches()
    next_state = agent.policy.init_concrete_state(batch_size=1)

    monkeypatch.setattr(
        agent.ppo,
        "select_action",
        lambda observations, state=None, deterministic=False: (
            0,
            0,
            0,
            -0.25,
            0.5,
            next_state,
        ),
    )

    obs = make_obs(
        friendly_positions=[(10, 10), (11, 10)],
        enemy_positions=[(50, 50), (12, 12), (18, 18)],
    )
    action_func, action, *_rest, learnable = agent.step(obs)

    assert action == 0
    assert action_func.name == "Attack_screen"
    assert action_func.args == ("now", [12, 12])
    assert learnable is True


def test_helper_fallback_marks_transition_non_learnable(make_obs, monkeypatch, fake_actions):
    agent = DefeatRoaches()
    next_state = agent.policy.init_concrete_state(batch_size=1)

    monkeypatch.setattr(
        agent.ppo,
        "select_action",
        lambda observations, state=None, deterministic=False: (
            1,
            30,
            40,
            -0.1,
            0.2,
            next_state,
        ),
    )

    obs = make_obs(available_actions={fake_actions.select_army.id})
    action_func, *_prefix, learnable = agent.step(obs)

    assert action_func.name == "select_army"
    assert action_func.args == ("select",)
    assert learnable is False


def test_deterministic_step_does_not_update_extractor_stats(make_obs, monkeypatch):
    agent = DefeatRoaches()
    next_state = agent.policy.init_concrete_state(batch_size=1)

    monkeypatch.setattr(
        agent.ppo,
        "select_action",
        lambda observations, deterministic=False: (
            2,
            0,
            0,
            -0.1,
            0.2,
            next_state,
        ),
    )

    obs = make_obs(
        multi_select=np.asarray([[48, 1, 45, 0, 0, 0, 1]], dtype=np.int32),
    )

    assert agent.extractor.entity_normalizer.count == 0.0
    assert agent.extractor.selection_normalizer.count == 0.0

    agent.step(obs, deterministic=True)

    assert agent.extractor.entity_normalizer.count == 0.0
    assert agent.extractor.selection_normalizer.count == 0.0

    agent.step(obs, deterministic=False)

    assert agent.extractor.entity_normalizer.count > 0.0
    assert agent.extractor.selection_normalizer.count > 0.0


def test_policy_forward_shapes_and_state_continuity():
    net = _small_policy()
    batch = _policy_batch(batch_size=2, spatial_shape=SPATIAL_OBS_SHAPE)
    batch_no_state = PolicyInputBatch(
        spatial_obs=batch.spatial_obs,
        entity_features=batch.entity_features,
        entity_mask=batch.entity_mask,
        selection_features=batch.selection_features,
        selection_mask=batch.selection_mask,
        meta_vec=batch.meta_vec,
    )

    action_logits, move_x_logits, move_y_logits, state_value, state1 = net(batch_no_state)
    action_logits_2, move_x_logits_2, move_y_logits_2, state_value_2, state2 = net(
        PolicyInputBatch(
            spatial_obs=batch.spatial_obs,
            entity_features=batch.entity_features,
            entity_mask=batch.entity_mask,
            selection_features=batch.selection_features,
            selection_mask=batch.selection_mask,
            meta_vec=batch.meta_vec,
            state_in=state1,
        )
    )

    assert action_logits.shape == (2, 3)
    assert move_x_logits.shape == (2, 16)
    assert move_y_logits.shape == (2, 16)
    assert state_value.shape == (2,)
    assert action_logits_2.shape == action_logits.shape
    assert move_x_logits_2.shape == move_x_logits.shape
    assert move_y_logits_2.shape == move_y_logits.shape
    assert state_value_2.shape == state_value.shape
    assert len(state2) == 2
    assert state2[0].shape == state1[0].shape
    assert state2[1].shape == state1[1].shape


def test_learnable_time_constants_receive_gradients():
    net = _small_policy()
    batch = _policy_batch(batch_size=1, spatial_shape=SPATIAL_OBS_SHAPE)

    action_logits, _, _, state_value, _ = net(
        PolicyInputBatch(
            spatial_obs=batch.spatial_obs,
            entity_features=batch.entity_features,
            entity_mask=batch.entity_mask,
            selection_features=batch.selection_features,
            selection_mask=batch.selection_mask,
            meta_vec=batch.meta_vec,
        )
    )
    loss = state_value.mean() - torch.softmax(action_logits, dim=-1).mean()
    loss.backward()

    for name, param in [
        ("token_snn.snn.alpha", net.token_snn.snn.alpha),
        ("token_snn.snn.beta", net.token_snn.snn.beta),
        ("attention.lif_q.beta", net.attention.lif_q.beta),
        ("attention.lif_k.beta", net.attention.lif_k.beta),
        ("attention.lif_v.beta", net.attention.lif_v.beta),
    ]:
        assert param.grad is not None, f"{name} is missing gradients"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"


def test_policy_parameter_count_stays_below_one_million():
    net = PolicyNetwork((27, 84, 84), META_VECTOR_DIM, 3)
    param_count = sum(param.numel() for param in net.parameters())
    assert param_count < 1_000_000


def test_entity_encoder_zeroes_padded_slots():
    encoder = EntityEncoder(feature_dim=21, embed_dim=32)
    features = torch.randn(2, 24, 21)
    mask = torch.tensor(
        [
            [True] * 24,
            [True] * 5 + [False] * 19,
        ],
        dtype=torch.bool,
    )
    features[1, 5:, 0] = 0.0

    encoded = encoder(features, mask)

    assert encoded.shape == (2, 24, 32)
    assert torch.count_nonzero(encoded[1, 5:]) == 0


def test_selection_encoder_zeroes_padded_slots():
    encoder = SelectionEncoder(feature_dim=7, embed_dim=32)
    features = torch.randn(2, 20, 7)
    mask = torch.tensor(
        [
            [True] * 20,
            [True] * 3 + [False] * 17,
        ],
        dtype=torch.bool,
    )
    features[1, 3:, 0] = 0.0

    encoded = encoder(features, mask)

    assert encoded.shape == (2, 20, 32)
    assert torch.count_nonzero(encoded[1, 3:]) == 0


def test_meta_encoder_returns_single_token():
    encoder = MetaEncoder(meta_input_dim=28, embed_dim=32)
    meta_vec = torch.randn(4, 28)
    meta_vec[:, -1] = torch.tensor([0.0, 1.0, 5.0, 17.0])

    encoded = encoder(meta_vec)

    assert encoded.shape == (4, 1, 32)
