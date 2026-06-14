import numpy as np
import torch

from MockedEnv.policy_batch import make_policy_batch
from agent_core.policy_protocol import (
    ACTION_FEEDBACK_TOKEN_DIM,
    BRIDGE_ACTION_BOOTSTRAP_SELECT,
    META_VECTOR_DIM,
    META_LAST_ACTION_INDEX_OFFSET,
    POLICY_ACTION_DIM,
    POLICY_ACTION_RIGHT_CLICK,
    PolicyInputBatch,
    SPATIAL_OBS_SHAPE,
)
from agent_core.target_heads import TargetHeadState
from agent_core.spiking_policy import (
    EntityEncoder,
    MetaEncoder,
    PolicyNetwork,
    SelectionEncoder,
    SpikingSelfAttention,
)
from agent import DefeatRoaches


def _small_policy():
    net = PolicyNetwork(
        SPATIAL_OBS_SHAPE,
        vector_input_dim=META_VECTOR_DIM,
        action_dim=POLICY_ACTION_DIM,
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


def test_bf16_amp_uses_autocast_without_grad_scaler():
    use_amp, dtype = PolicyNetwork._resolve_amp_settings(
        "bf16",
        torch.device("cuda"),
    )
    scaler = PolicyNetwork._make_grad_scaler(use_amp=use_amp, amp_dtype=dtype)

    assert use_amp is True
    assert dtype == torch.bfloat16
    assert scaler.is_enabled() is False


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
            2,
            4 * 4 + 24 + 20 + 1 + 1,
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
        "Smart_screen",
    }
    assert action in {0, 2}
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
    assert tuple(policy_input.action_feedback_tokens.shape) == (
        1,
        1,
        ACTION_FEEDBACK_TOKEN_DIM,
    )
    assert tuple(policy_input.meta_vec.shape) == (1, META_VECTOR_DIM)
    assert policy_input.state_in is not None
    assert isinstance(learnable, bool)


def test_smart_uses_policy_coordinates(make_obs, monkeypatch):
    agent = DefeatRoaches()
    next_state = agent.policy.init_concrete_state(batch_size=1)

    monkeypatch.setattr(
        agent.ppo,
        "select_action",
        lambda observations, state=None, deterministic=False: (
            POLICY_ACTION_RIGHT_CLICK,
            17,
            19,
            -0.25,
            0.5,
            next_state,
        ),
    )

    obs = make_obs()
    action_func, action, *_rest, learnable = agent.step(obs)

    assert action == POLICY_ACTION_RIGHT_CLICK
    assert action_func.name == "Smart_screen"
    assert action_func.args == ("now", [17, 19])
    assert learnable is True


def test_bootstrap_selection_stays_outside_policy_memory(make_obs, fake_actions):
    agent = DefeatRoaches()

    obs = make_obs(available_actions={fake_actions.select_army.id})
    (
        action_func,
        action,
        move_x,
        move_y,
        _pre_step_state,
        log_prob,
        value,
        policy_input,
        learnable,
    ) = agent.step(obs)

    assert action_func.name == "select_army"
    assert action_func.args == ("select",)
    assert action is None
    assert move_x == 0
    assert move_y == 0
    assert log_prob == 0.0
    assert value == 0.0
    assert policy_input is None
    assert learnable is False
    assert agent.last_action_token[0] == BRIDGE_ACTION_BOOTSTRAP_SELECT


def test_deterministic_step_does_not_update_extractor_stats(make_obs, monkeypatch):
    agent = DefeatRoaches()
    next_state = agent.policy.init_concrete_state(batch_size=1)

    monkeypatch.setattr(
        agent.ppo,
        "select_action",
        lambda observations, deterministic=False: (
            POLICY_ACTION_RIGHT_CLICK,
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

    action_logits, target_head_state, state_value, state1 = net(batch_no_state)
    action_logits_2, target_head_state_2, state_value_2, state2 = net(
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

    assert action_logits.shape == (2, POLICY_ACTION_DIM)
    assert isinstance(target_head_state, TargetHeadState)
    assert target_head_state.primary_logits.shape == (2, 16)
    assert target_head_state.secondary_logits is None
    assert state_value.shape == (2,)
    assert action_logits_2.shape == action_logits.shape
    assert target_head_state_2.primary_logits.shape == target_head_state.primary_logits.shape
    assert target_head_state_2.secondary_logits is None
    assert state_value_2.shape == state_value.shape
    assert len(state2) == 2
    assert state2[0].shape == state1[0].shape
    assert state2[1].shape == state1[1].shape


def test_target_head_changes_with_action_id():
    net = _small_policy()
    batch = _policy_batch(batch_size=2, spatial_shape=SPATIAL_OBS_SHAPE)
    latent, _value, _next_state, spatial_context = net.encode_step_tensors(
        spatial_obs=batch.spatial_obs,
        entity_features=batch.entity_features,
        entity_mask=batch.entity_mask,
        selection_features=batch.selection_features,
        selection_mask=batch.selection_mask,
        meta_vec=batch.meta_vec,
        state_in=batch.state_in,
    )

    right_click_head = net.build_target_head(
        latent,
        spatial_context,
        torch.full((2,), POLICY_ACTION_RIGHT_CLICK, dtype=torch.long),
    )
    no_op_head = net.build_target_head(
        latent,
        spatial_context,
        torch.zeros((2,), dtype=torch.long),
    )

    assert right_click_head.primary_logits.shape == (2, 16)
    assert right_click_head.secondary_logits is None
    assert no_op_head.primary_logits.shape == right_click_head.primary_logits.shape
    assert not torch.allclose(
        right_click_head.primary_logits,
        no_op_head.primary_logits,
    )


def test_encode_step_tensors_returns_structured_spatial_context():
    net = _small_policy()
    batch = _policy_batch(batch_size=2, spatial_shape=SPATIAL_OBS_SHAPE)

    latent, state_value, next_state, spatial_context = net.encode_step_tensors(
        spatial_obs=batch.spatial_obs,
        entity_features=batch.entity_features,
        entity_mask=batch.entity_mask,
        selection_features=batch.selection_features,
        selection_mask=batch.selection_mask,
        meta_vec=batch.meta_vec,
        state_in=batch.state_in,
    )

    assert latent.shape == (2, 64)
    assert state_value.shape == (2,)
    assert next_state[0].shape == (2, 2, 4 * 4 + 24 + 20 + 1 + 1, 32)
    assert spatial_context.shape == (2, 32, 4, 4)


def test_policy_zeroes_entity_recurrent_state_between_env_steps():
    net = _small_policy()
    state = net.init_concrete_state(batch_size=1, device=torch.device("cpu"))
    state = (torch.ones_like(state[0]), torch.ones_like(state[1]))
    batch = _policy_batch(batch_size=1, spatial_shape=SPATIAL_OBS_SHAPE)

    _, _, _state_value, next_state = net(
        PolicyInputBatch(
            spatial_obs=batch.spatial_obs,
            entity_features=batch.entity_features,
            entity_mask=batch.entity_mask,
            selection_features=batch.selection_features,
            selection_mask=batch.selection_mask,
            meta_vec=batch.meta_vec,
            state_in=state,
        )
    )

    assert torch.count_nonzero(
        next_state[0][:, :, net._entity_start : net._entity_end, :],
    ) == 0
    assert torch.count_nonzero(
        next_state[1][:, :, net._entity_start : net._entity_end, :],
    ) == 0


def test_learnable_time_constants_receive_gradients():
    net = _small_policy()
    batch = _policy_batch(batch_size=1, spatial_shape=SPATIAL_OBS_SHAPE)

    action_logits, _target_head_state, state_value, _ = net(
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
        ("slow_token_snn.snn.alpha", net.slow_token_snn.snn.alpha),
        ("slow_token_snn.snn.beta", net.slow_token_snn.snn.beta),
        ("attention.lif_q.beta", net.attention.lif_q.beta),
        ("attention.lif_k.beta", net.attention.lif_k.beta),
        ("attention.lif_v.beta", net.attention.lif_v.beta),
    ]:
        assert param.grad is not None, f"{name} is missing gradients"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"


def test_policy_accepts_legacy_single_timescale_state():
    net = _small_policy()
    batch = make_policy_batch(
        batch_size=1,
        spatial_shape=SPATIAL_OBS_SHAPE,
        meta_dim=META_VECTOR_DIM,
        with_state=True,
        state_shape=(1, 4 * 4 + 24 + 20 + 1 + 1, 32),
        zeros=True,
    )

    action_logits, target_head_state, state_value, next_state = net(batch)

    assert action_logits.shape == (1, POLICY_ACTION_DIM)
    assert target_head_state.primary_logits.shape == (1, 16)
    assert target_head_state.secondary_logits is None
    assert state_value.shape == (1,)
    assert next_state[0].shape == (1, 2, 4 * 4 + 24 + 20 + 1 + 1, 32)
    assert next_state[1].shape == next_state[0].shape


def test_policy_forward_handles_batch_size_not_equal_temporal_pathways():
    net = _small_policy()
    batch = _policy_batch(batch_size=4, spatial_shape=SPATIAL_OBS_SHAPE)

    action_logits, target_head_state, state_value, next_state = net(batch)

    assert action_logits.shape == (4, POLICY_ACTION_DIM)
    assert target_head_state.primary_logits.shape == (4, 16)
    assert target_head_state.secondary_logits is None
    assert state_value.shape == (4,)
    assert next_state[0].shape == (4, 2, 4 * 4 + 24 + 20 + 1 + 1, 32)
    assert next_state[1].shape == next_state[0].shape


def test_spiking_self_attention_sdpa_respects_padding_mask():
    attention = SpikingSelfAttention(embed_dim=8)
    tokens = torch.randn(2, 4, 8, requires_grad=True)
    token_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )

    output = attention(tokens, token_mask=token_mask)
    loss = output.sum()
    loss.backward()

    assert output.shape == (2, 4, 8)
    assert torch.count_nonzero(output[1, 2:]) == 0
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_policy_parameter_count_stays_below_one_million():
    net = PolicyNetwork((27, 84, 84), META_VECTOR_DIM, POLICY_ACTION_DIM)
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
    encoder = MetaEncoder(meta_input_dim=META_VECTOR_DIM, embed_dim=32)
    meta_vec = torch.randn(4, META_VECTOR_DIM)
    meta_vec[:, META_LAST_ACTION_INDEX_OFFSET] = torch.tensor([0.0, 1.0, 2.0, 3.0])

    encoded = encoder(meta_vec)

    assert encoded.shape == (4, 1, 32)
