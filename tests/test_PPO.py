import math

import pytest
import torch
import torch.nn as nn

from MockedEnv.policy_batch import make_policy_batch
from agent_core.policy_protocol import (
    ACTION_FEEDBACK_BRIDGE_TYPE_OFFSET,
    ACTION_FEEDBACK_ENEMY_HEALTH_DROP_OFFSET,
    ACTION_FEEDBACK_EXECUTED_SMART_OFFSET,
    ACTION_FEEDBACK_MOVED_TOWARD_TARGET_OFFSET,
    ACTION_FEEDBACK_TARGET_NEAR_ENEMY_OFFSET,
    BRIDGE_ACTION_RIGHT_CLICK,
    META_AVAILABLE_ACTION_OFFSET,
    META_VECTOR_DIM,
    POLICY_ACTION_LEFT_CLICK,
    POLICY_ACTION_NO_OP,
    POLICY_ACTION_RIGHT_CLICK,
    SPATIAL_OBS_SHAPE,
)
from agent_core.ppo_trainer import PPO
from agent_core.spiking_policy import PolicyNetwork


class FakeNet(nn.Module):
    def __init__(self, action_logits=None, move_x_logits=None, move_y_logits=None):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.device = torch.device("cpu")
        self.use_amp = False
        self.amp_dtype = torch.float32
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self._action_logits = action_logits
        self._move_x_logits = move_x_logits
        self._move_y_logits = move_y_logits

    @staticmethod
    def _repeat_logits(logits, batch_size):
        if logits is None:
            return None
        return logits.expand(batch_size, -1).clone()

    def forward(self, batch):
        return self.forward_step_tensors(
            spatial_obs=batch.spatial_obs,
            entity_features=batch.entity_features,
            entity_mask=batch.entity_mask,
            selection_features=batch.selection_features,
            selection_mask=batch.selection_mask,
            meta_vec=batch.meta_vec,
            state_in=batch.state_in,
        )

    def encode_step_tensors(
        self,
        spatial_obs,
        entity_features,
        entity_mask,
        selection_features,
        selection_mask,
        meta_vec,
        state_in,
    ):
        del spatial_obs, entity_features, entity_mask
        del selection_features, selection_mask
        latent = self.fc(meta_vec[:, :4].float())
        state_value = latent.sum(dim=-1) * 0.1
        spatial_context = latent.unsqueeze(-1).unsqueeze(-1)
        return latent, state_value, state_in, spatial_context

    def action_head(self, latent):
        batch_size = latent.size(0)
        base = latent.sum(dim=-1)
        action_logits = self._repeat_logits(self._action_logits, batch_size)
        if action_logits is None:
            action_logits = torch.randn(batch_size, 2, device=self.device)
        action_logits = action_logits.to(self.device) + base.unsqueeze(-1) * 0

        return action_logits

    def conditioned_spatial_head(self, latent, spatial_context, action_ids):
        batch_size = latent.size(0)
        del spatial_context
        base = latent.sum(dim=-1) + action_ids.to(latent.device, dtype=latent.dtype) * 0
        move_x_logits = self._repeat_logits(self._move_x_logits, batch_size)
        if move_x_logits is None:
            move_x_logits = torch.randn(batch_size, 8, device=self.device)
        move_x_logits = move_x_logits.to(self.device) + base.unsqueeze(-1) * 0

        move_y_logits = self._repeat_logits(self._move_y_logits, batch_size)
        if move_y_logits is None:
            move_y_logits = torch.randn(batch_size, 8, device=self.device)
        move_y_logits = move_y_logits.to(self.device) + base.unsqueeze(-1) * 0

        return move_x_logits, move_y_logits

    def forward_step_tensors(
        self,
        spatial_obs,
        entity_features,
        entity_mask,
        selection_features,
        selection_mask,
        meta_vec,
        state_in,
        action_ids=None,
    ):
        latent, state_value, next_state, spatial_context = self.encode_step_tensors(
            spatial_obs,
            entity_features,
            entity_mask,
            selection_features,
            selection_mask,
            meta_vec,
            state_in,
        )
        action_logits = self.action_head(latent)
        if action_ids is None:
            action_ids = action_logits.argmax(dim=-1)
        move_x_logits, move_y_logits = self.conditioned_spatial_head(
            latent,
            spatial_context,
            action_ids,
        )
        return action_logits, move_x_logits, move_y_logits, state_value, next_state

    def init_concrete_state(self, batch_size=1, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = torch.float32
        zeros = torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)
        return zeros.clone(), zeros.clone()


class SequenceCarryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.device = torch.device("cpu")
        self.use_amp = False
        self.amp_dtype = torch.float32
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.seen_states = []

    def init_concrete_state(self, batch_size=1, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = torch.float32
        zeros = torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)
        return zeros.clone(), zeros.clone()

    def forward(self, batch):
        return self.forward_step_tensors(
            spatial_obs=batch.spatial_obs,
            entity_features=batch.entity_features,
            entity_mask=batch.entity_mask,
            selection_features=batch.selection_features,
            selection_mask=batch.selection_mask,
            meta_vec=batch.meta_vec,
            state_in=batch.state_in,
        )

    def encode_step_tensors(
        self,
        spatial_obs,
        entity_features,
        entity_mask,
        selection_features,
        selection_mask,
        meta_vec,
        state_in,
    ):
        del spatial_obs, entity_features, entity_mask
        del selection_features, selection_mask, meta_vec
        batch_size = 1 if state_in is None else state_in[0].size(0)
        if state_in is None:
            state = self.init_concrete_state(
                batch_size=batch_size,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            state = state_in
        syn, mem = state
        current = syn[:, 0, 0]
        self.seen_states.append(float(current[0].detach().cpu().item()))
        base = self.weight * current
        latent = base.unsqueeze(-1)
        next_value = (current + 1.0).view(-1, 1, 1)
        next_state = (
            next_value.expand(batch_size, 1, 1),
            next_value.expand(batch_size, 1, 1),
        )
        spatial_context = latent.unsqueeze(-1).unsqueeze(-1)
        return latent, base, next_state, spatial_context

    def action_head(self, latent):
        base = latent[:, 0]
        return torch.stack((base, base + 0.1), dim=-1)

    def conditioned_spatial_head(self, latent, spatial_context, action_ids):
        base = latent[:, 0]
        del spatial_context, action_ids
        move_x_logits = torch.stack([base + 0.01 * i for i in range(8)], dim=-1)
        move_y_logits = torch.stack([base - 0.01 * i for i in range(8)], dim=-1)
        return move_x_logits, move_y_logits

    def forward_step_tensors(
        self,
        spatial_obs,
        entity_features,
        entity_mask,
        selection_features,
        selection_mask,
        meta_vec,
        state_in,
        action_ids=None,
    ):
        latent, state_value, next_state, spatial_context = self.encode_step_tensors(
            spatial_obs,
            entity_features,
            entity_mask,
            selection_features,
            selection_mask,
            meta_vec,
            state_in,
        )
        action_logits = self.action_head(latent)
        if action_ids is None:
            action_ids = action_logits.argmax(dim=-1)
        move_x_logits, move_y_logits = self.conditioned_spatial_head(
            latent,
            spatial_context,
            action_ids,
        )
        return action_logits, move_x_logits, move_y_logits, state_value, next_state


class CountingReplayNet(FakeNet):
    def __init__(self):
        super().__init__()
        self.encode_step_calls = 0

    def encode_step_tensors(
        self,
        spatial_obs,
        entity_features,
        entity_mask,
        selection_features,
        selection_mask,
        meta_vec,
        state_in,
    ):
        self.encode_step_calls += 1
        return super().encode_step_tensors(
            spatial_obs,
            entity_features,
            entity_mask,
            selection_features,
            selection_mask,
            meta_vec,
            state_in,
        )


class RowResetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.device = torch.device("cpu")
        self.use_amp = False
        self.amp_dtype = torch.float32
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.row_history = []

    def init_concrete_state(self, batch_size=1, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = torch.float32
        zeros = torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)
        return zeros.clone(), zeros.clone()

    def forward(self, batch):
        return self.forward_step_tensors(
            spatial_obs=batch.spatial_obs,
            entity_features=batch.entity_features,
            entity_mask=batch.entity_mask,
            selection_features=batch.selection_features,
            selection_mask=batch.selection_mask,
            meta_vec=batch.meta_vec,
            state_in=batch.state_in,
        )

    def encode_step_tensors(
        self,
        spatial_obs,
        entity_features,
        entity_mask,
        selection_features,
        selection_mask,
        meta_vec,
        state_in,
    ):
        del spatial_obs, entity_features, entity_mask
        del selection_features, selection_mask, meta_vec
        batch_size = 1 if state_in is None else state_in[0].size(0)
        if state_in is None:
            state = self.init_concrete_state(batch_size=batch_size)
        else:
            state = state_in
        syn, mem = state
        current = syn[:, 0, 0]
        self.row_history.append(current.detach().cpu().clone())
        base = self.weight * current
        latent = base.unsqueeze(-1)
        next_value = (current + 1.0).view(-1, 1, 1)
        next_state = (
            next_value.expand(current.size(0), 1, 1),
            next_value.expand(current.size(0), 1, 1),
        )
        spatial_context = latent.unsqueeze(-1).unsqueeze(-1)
        return latent, base, next_state, spatial_context

    def action_head(self, latent):
        base = latent[:, 0]
        return torch.stack((base, base + 0.1), dim=-1)

    def conditioned_spatial_head(self, latent, spatial_context, action_ids):
        base = latent[:, 0]
        del spatial_context, action_ids
        move_x_logits = torch.stack([base + 0.01 * i for i in range(8)], dim=-1)
        move_y_logits = torch.stack([base - 0.01 * i for i in range(8)], dim=-1)
        return move_x_logits, move_y_logits

    def forward_step_tensors(
        self,
        spatial_obs,
        entity_features,
        entity_mask,
        selection_features,
        selection_mask,
        meta_vec,
        state_in,
        action_ids=None,
    ):
        latent, state_value, next_state, spatial_context = self.encode_step_tensors(
            spatial_obs,
            entity_features,
            entity_mask,
            selection_features,
            selection_mask,
            meta_vec,
            state_in,
        )
        action_logits = self.action_head(latent)
        if action_ids is None:
            action_ids = action_logits.argmax(dim=-1)
        move_x_logits, move_y_logits = self.conditioned_spatial_head(
            latent,
            spatial_context,
            action_ids,
        )
        return action_logits, move_x_logits, move_y_logits, state_value, next_state


def _make_state(fill_value):
    tensor = torch.full((1, 1, 1), fill_value, dtype=torch.float32)
    return tensor.clone(), tensor.clone()


def _build_chunks_from_memory(ppo):
    actions = torch.stack(
        [transition["action"].to(ppo.device) for transition in ppo.memory],
    )
    move_xs = torch.stack(
        [transition["move_x"].to(ppo.device) for transition in ppo.memory],
    )
    move_ys = torch.stack(
        [transition["move_y"].to(ppo.device) for transition in ppo.memory],
    )
    log_probs_old = torch.stack(
        [transition["log_prob"].to(ppo.device) for transition in ppo.memory],
    )
    advantages = torch.stack(
        [
            torch.tensor(0.25 + 0.1 * idx, device=ppo.device)
            for idx, _ in enumerate(ppo.memory)
        ],
    )
    returns = torch.stack(
        [
            torch.tensor(1.0 + 0.2 * idx, device=ppo.device)
            for idx, _ in enumerate(ppo.memory)
        ],
    )
    dones = torch.stack(
        [transition["done"].to(ppo.device) for transition in ppo.memory],
    )
    policy_masks = torch.stack(
        [transition["policy_mask"].to(ppo.device) for transition in ppo.memory],
    )
    return ppo._build_tbptt_chunks(
        actions=actions,
        move_xs=move_xs,
        move_ys=move_ys,
        log_probs_old=log_probs_old,
        advantages=advantages,
        returns=returns,
        dones=dones,
        policy_masks=policy_masks,
    )


def test_scheduler_decays_lr():
    net = FakeNet()
    ppo = PPO(net, lr=1e-4, total_updates=1000, lr_min=1e-5)

    lrs = {}
    for step in range(1000):
        loss = net.fc(torch.randn(1, 4)).sum()
        ppo.optimizer.zero_grad()
        loss.backward()
        ppo.optimizer.step()
        if step in {0, 100, 500, 900, 999}:
            lrs[step] = ppo.optimizer.param_groups[0]["lr"]
        ppo.scheduler.step()

    assert lrs[0] == 1e-4
    assert lrs[999] > 1e-5
    assert lrs[999] < lrs[0]
    assert lrs[100] > lrs[500] > lrs[900] > lrs[999]


def test_select_action_deterministic_uses_argmax_for_move_heads():
    action_logits = torch.tensor([[-2.0, 3.0]])
    move_x_logits = torch.tensor([[0.1, -0.1, 0.0, 0.2, 0.3, 1.4, 0.5, -0.7]])
    move_y_logits = torch.tensor([[-0.8, 0.0, 0.2, -0.4, 0.1, 0.3, 0.9, 1.7]])
    ppo = PPO(FakeNet(action_logits, move_x_logits, move_y_logits), lr=1e-4)

    batch = make_policy_batch(batch_size=1, meta_dim=META_VECTOR_DIM)
    sample = ppo.select_action(
        batch, deterministic=True,
    )

    expected_log_prob = (
        torch.log_softmax(action_logits, dim=-1)[0, 1]
        + torch.log_softmax(move_x_logits, dim=-1)[0, 5]
        + torch.log_softmax(move_y_logits, dim=-1)[0, 7]
    )

    assert sample.action_id == 1
    assert sample.x == 5
    assert sample.y == 7
    assert sample.log_prob == pytest.approx(float(expected_log_prob.item()), abs=1e-6)
    assert isinstance(sample.value, float)
    assert sample.next_state is None


def test_select_action_no_op_ignores_spatial_log_prob():
    action_logits = torch.tensor([[3.0, -1.0]])
    move_x_logits = torch.tensor([[0.4, -0.1, 1.2, 0.0, -0.6, 0.7, 0.5, -0.2]])
    move_y_logits = torch.tensor([[-0.3, 0.0, 0.1, 0.2, 1.8, -0.5, 0.6, 0.7]])
    ppo = PPO(FakeNet(action_logits, move_x_logits, move_y_logits), lr=1e-4)

    batch = make_policy_batch(batch_size=1, meta_dim=META_VECTOR_DIM)
    sample = ppo.select_action(
        batch,
        deterministic=True,
    )

    expected_log_prob = torch.log_softmax(action_logits, dim=-1)[0, 0]

    assert sample.action_id == 0
    assert sample.x == 0
    assert sample.y == 0
    assert sample.log_prob == pytest.approx(float(expected_log_prob.item()), abs=1e-6)


def test_right_click_curriculum_temporarily_penalizes_no_op_when_smart_available():
    action_logits = torch.tensor([[0.0, -10.0, -1.0]])
    move_x_logits = torch.zeros(1, 8)
    move_y_logits = torch.zeros(1, 8)
    ppo = PPO(
        FakeNet(action_logits, move_x_logits, move_y_logits),
        lr=1e-4,
        right_click_curriculum_updates=10,
        right_click_curriculum_noop_logit_penalty=2.0,
    )

    batch = make_policy_batch(batch_size=1, meta_dim=META_VECTOR_DIM)
    batch.meta_vec[:, META_AVAILABLE_ACTION_OFFSET + POLICY_ACTION_LEFT_CLICK] = 0.0

    sample = ppo.select_action(batch, deterministic=True)
    assert sample.action_id == POLICY_ACTION_RIGHT_CLICK

    ppo.update_count = 10
    sample = ppo.select_action(batch, deterministic=True)
    assert sample.action_id == POLICY_ACTION_NO_OP


def test_compute_advantages_matches_hand_calculation():
    ppo = PPO(FakeNet(), lr=1e-4)
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0], device=ppo.device)
    values = torch.tensor([0.5, 0.5, 0.5, 0.5], device=ppo.device)
    dones = torch.tensor([0.0, 0.0, 0.0, 1.0], device=ppo.device)

    advantages = ppo._compute_advantages(
        rewards,
        values,
        dones,
        last_next_value=torch.tensor(0.0, device=ppo.device),
    ).cpu()

    expected_adv_3 = 0.5
    expected_adv_2 = 0.995 + 0.99 * 0.95 * expected_adv_3
    expected_adv_1 = 0.995 + 0.99 * 0.95 * expected_adv_2
    expected_adv_0 = 0.995 + 0.99 * 0.95 * expected_adv_1
    expected = torch.tensor([
        expected_adv_0,
        expected_adv_1,
        expected_adv_2,
        expected_adv_3,
    ])

    assert advantages.shape == rewards.cpu().shape
    assert torch.allclose(advantages, expected, atol=1e-5)


def test_update_policy_bootstraps_each_fragment_independently():
    ppo = PPO(SequenceCarryNet(), lr=1e-4, gamma=0.99, total_updates=0, lr_min=0.0)

    first_batch = make_policy_batch(
        batch_size=1,
        meta_dim=META_VECTOR_DIM,
        zeros=True,
    ).with_state(_make_state(0.0))
    ppo.store_transition(
        first_batch,
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(-0.1),
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
        policy_mask=torch.tensor(1.0),
    )
    tail = make_policy_batch(
        batch_size=1,
        meta_dim=META_VECTOR_DIM,
        zeros=True,
    ).with_state(_make_state(20.0))
    ppo.set_final_next(tail)
    first_fragment = ppo.finalize_fragment()

    second_batch = make_policy_batch(
        batch_size=1,
        meta_dim=META_VECTOR_DIM,
        zeros=True,
    ).with_state(_make_state(0.0))
    ppo.store_transition(
        second_batch,
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(-0.1),
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.tensor(1.0),
        policy_mask=torch.tensor(1.0),
    )
    second_fragment = ppo.finalize_fragment()

    assert first_fragment is not None
    assert second_fragment is not None
    losses, stats = ppo.update_policy(
        fragments=[first_fragment, second_fragment],
        batch_size=2,
        epochs=1,
    )

    assert losses
    assert stats["fragments_in_update"] == 2
    assert stats["transitions_in_update"] == 2
    assert stats["return_mean"] == pytest.approx((1.0 + 0.99 * 10.0 + 1.0) / 2.0)


def test_calculate_losses_reports_normalized_entropy():
    ppo = PPO(FakeNet(), lr=1e-4)
    batch_size = 128
    action_logits = torch.randn(batch_size, 2)
    target_logits = torch.randn(batch_size, 84)
    target_dist = torch.distributions.Categorical(logits=target_logits.float())
    sampled_target = torch.randint(0, 84, (batch_size,))

    policy_loss, value_loss, entropy_loss, diag = ppo._calculate_losses(
        action_logits,
        target_dist.log_prob(sampled_target),
        target_dist.entropy() / math.log(84.0),
        torch.randn(batch_size),
        torch.ones(batch_size, dtype=torch.long),
        torch.randn(batch_size),
        torch.randn(batch_size),
        torch.randn(batch_size),
    )

    assert policy_loss.ndim == 0
    assert value_loss.ndim == 0
    assert entropy_loss.ndim == 0
    assert 0.5 <= float(diag["entropy_mean"].item()) <= 2.0
    assert torch.isfinite(diag["approx_kl"])
    assert 0.0 <= float(diag["clip_frac"].item()) <= 1.0


def test_calculate_losses_no_op_samples_do_not_backprop_spatial_heads():
    ppo = PPO(FakeNet(), lr=1e-4)
    action_logits = torch.randn(8, 2, requires_grad=True)
    target_logits = torch.randn(8, 84, requires_grad=True)
    target_dist = torch.distributions.Categorical(logits=target_logits.float())
    sampled_target = torch.randint(0, 84, (8,))
    state_values = torch.randn(8, requires_grad=True)

    policy_loss, value_loss, entropy_loss, _diag = ppo._calculate_losses(
        action_logits,
        target_dist.log_prob(sampled_target),
        target_dist.entropy() / math.log(84.0),
        state_values,
        torch.zeros(8, dtype=torch.long),
        torch.zeros(8),
        torch.ones(8),
        torch.zeros(8),
    )
    total = policy_loss + value_loss - entropy_loss
    total.backward()

    if target_logits.grad is not None:
        assert torch.count_nonzero(target_logits.grad) == 0


def test_calculate_losses_masks_critic_loss_with_policy_mask():
    ppo = PPO(FakeNet(), lr=1e-4, critic_loss_coef=1.0)

    _policy_loss, value_loss, _entropy_loss, diag = ppo._calculate_losses(
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        torch.zeros(2),
        torch.zeros(2),
        torch.tensor([0.0, 0.0]),
        torch.tensor([1, 1], dtype=torch.long),
        torch.zeros(2),
        torch.zeros(2),
        torch.tensor([1.0, 100.0]),
        torch.tensor([1.0, 0.0]),
    )

    assert value_loss.item() == pytest.approx(1.0, abs=1e-6)
    assert diag["value_count"].item() == pytest.approx(1.0, abs=1e-6)


def test_store_transition_requires_pre_step_recurrent_state():
    ppo = PPO(FakeNet(), lr=1e-4)
    batch = make_policy_batch(
        batch_size=1,
        meta_dim=META_VECTOR_DIM,
        with_state=False,
    )

    with pytest.raises(
        ValueError,
        match="Stored transition must carry the pre-step recurrent state",
    ):
        ppo.store_transition(
            batch,
            torch.tensor(1),
            torch.tensor(2),
            torch.tensor(3),
            torch.tensor(-0.5),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )


def test_set_final_next_requires_bootstrap_recurrent_state():
    ppo = PPO(FakeNet(), lr=1e-4)
    batch = make_policy_batch(
        batch_size=1,
        meta_dim=META_VECTOR_DIM,
        with_state=False,
    )

    with pytest.raises(
        ValueError,
        match="Bootstrap observation must carry the recurrent state after the final rollout step",
    ):
        ppo.set_final_next(batch)


def test_finalize_fragment_records_action_effect_feedback_counters():
    ppo = PPO(FakeNet(), lr=1e-4)

    action_ids = [
        POLICY_ACTION_NO_OP,
        POLICY_ACTION_RIGHT_CLICK,
        POLICY_ACTION_RIGHT_CLICK,
    ]
    feedback_overrides = [
        {},
        {
            ACTION_FEEDBACK_BRIDGE_TYPE_OFFSET: BRIDGE_ACTION_RIGHT_CLICK,
            ACTION_FEEDBACK_EXECUTED_SMART_OFFSET: 1.0,
            ACTION_FEEDBACK_TARGET_NEAR_ENEMY_OFFSET: 1.0,
            ACTION_FEEDBACK_ENEMY_HEALTH_DROP_OFFSET: 0.5,
        },
        {
            ACTION_FEEDBACK_BRIDGE_TYPE_OFFSET: BRIDGE_ACTION_RIGHT_CLICK,
            ACTION_FEEDBACK_EXECUTED_SMART_OFFSET: 1.0,
            ACTION_FEEDBACK_MOVED_TOWARD_TARGET_OFFSET: 0.0,
            ACTION_FEEDBACK_ENEMY_HEALTH_DROP_OFFSET: 0.0,
        },
    ]

    for step, action_id in enumerate(action_ids):
        batch = make_policy_batch(batch_size=1, with_state=True, zeros=True)
        for offset, value in feedback_overrides[step].items():
            batch.action_feedback_tokens[0, 0, offset] = float(value)
        ppo.store_transition(
            batch,
            torch.tensor(action_id),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(-1.0),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(float(step == len(action_ids) - 1)),
        )

    fragment = ppo.finalize_fragment()

    assert fragment is not None
    assert fragment.step_counters["rollout_policy_no_op_count"] == 1
    assert fragment.step_counters["rollout_policy_left_click_count"] == 0
    assert fragment.step_counters["rollout_policy_right_click_count"] == 2
    assert fragment.step_counters["rollout_feedback_smart_executed_count"] == 2
    assert fragment.step_counters["rollout_feedback_near_enemy_smart_count"] == 1
    assert fragment.step_counters["rollout_feedback_moved_toward_target_count"] == 0
    assert (
        fragment.step_counters[
            "rollout_feedback_enemy_health_drop_after_smart_count"
        ]
        == 1
    )
    assert fragment.step_counters["rollout_feedback_null_unclear_smart_count"] == 1


def test_update_policy_replays_state_and_clears_memory():
    torch.manual_seed(0)
    net = PolicyNetwork(
        SPATIAL_OBS_SHAPE,
        vector_input_dim=META_VECTOR_DIM,
        action_dim=2,
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

    ppo = PPO(net, lr=1e-4, total_updates=0, lr_min=0.0, tbptt_window=3)
    rollout_steps = 6

    for step in range(rollout_steps):
        state = net.init_concrete_state(batch_size=1, device=torch.device("cpu"))
        batch = make_policy_batch(
            batch_size=1,
            meta_dim=META_VECTOR_DIM,
            with_state=True,
            state_shape=state[0].shape,
        )
        with torch.no_grad():
            _, _, state_value, _ = net(batch.with_state(state))
        ppo.store_transition(
            batch.with_state(state),
            torch.tensor(1),
            torch.tensor(5),
            torch.tensor(10),
            torch.tensor(-1.0),
            torch.tensor(1.0),
            torch.tensor(state_value.item()),
            torch.tensor(float(step == rollout_steps - 1)),
        )

    initial_params = [param.detach().clone() for param in net.parameters()]
    losses, stats = ppo.update_policy(batch_size=3, epochs=1)

    changed_params = sum(
        not torch.equal(before, after)
        for before, after in zip(initial_params, net.parameters())
    )

    assert len(losses) == 2
    assert stats is not None
    assert stats["transitions_in_update"] == rollout_steps
    assert stats["epochs_ran"] == 1
    assert stats["nonfinite_grad_steps"] == 0
    assert stats["skipped_optimizer_steps"] == 0
    assert 0.0 <= stats["entity_mask_utilization"] <= 1.0
    assert 0.0 <= stats["selection_mask_utilization"] <= 1.0
    assert stats["entity_count_p50"] >= 0.0
    assert stats["entity_count_p99"] >= stats["entity_count_p50"]
    assert stats["update_wall_seconds"] >= 0.0
    assert stats["tbptt_chunks"] >= 1
    assert stats["tbptt_chunk_groups"] >= 1
    assert stats["tbptt_window"] == 3
    assert stats["tbptt_group_max_steps"] >= 1
    assert stats["tbptt_group_mean_active_chunks"] >= 1.0
    assert stats["tbptt_forward_calls"] >= 1
    for timing_key in (
        "fragment_tensor_build_wall_seconds",
        "cpu_to_gpu_transfer_wall_seconds",
        "bootstrap_value_wall_seconds",
        "gae_wall_seconds",
        "tbptt_chunk_build_wall_seconds",
        "chunk_pack_wall_seconds",
        "replay_forward_wall_seconds",
        "loss_eval_wall_seconds",
        "backward_optimizer_wall_seconds",
        "ppo_epoch_wall_seconds",
    ):
        assert timing_key in stats
        assert stats[timing_key] >= 0.0
    assert stats["payload_spatial_bytes"] > 0
    assert stats["payload_state_bytes"] > 0
    assert stats["payload_total_bytes"] >= stats["payload_spatial_bytes"]
    assert stats["payload_total_mib"] > 0.0
    assert stats["cuda_peak_allocated_bytes"] == 0
    assert stats["cuda_peak_reserved_bytes"] == 0
    assert stats["rollout_cache_spatial_dtype"] == "float32"
    assert changed_params > 0
    assert ppo.memory == []
    assert ppo.final_next is None


def test_update_policy_caches_fragment_observations_once_per_fragment(monkeypatch):
    torch.manual_seed(0)
    net = SequenceCarryNet()
    ppo = PPO(net, lr=1e-3, total_updates=0, lr_min=0.0, tbptt_window=2)

    for step in range(4):
        batch = make_policy_batch(
            batch_size=1,
            meta_dim=META_VECTOR_DIM,
            zeros=True,
        ).with_state(_make_state(float(step)))
        ppo.store_transition(
            batch,
            torch.tensor(1),
            torch.tensor(2),
            torch.tensor(3),
            torch.tensor(-0.5),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(float(step == 3)),
            policy_mask=torch.tensor(1.0),
        )
    fragment = ppo.finalize_fragment()
    assert fragment is not None

    move_calls = 0
    original_move = ppo._move_policy_input_to_device

    def wrapped_move(*args, **kwargs):
        nonlocal move_calls
        move_calls += 1
        return original_move(*args, **kwargs)

    monkeypatch.setattr(ppo, "_move_policy_input_to_device", wrapped_move)
    losses, stats = ppo.update_policy(fragments=[fragment], batch_size=2, epochs=3)

    assert losses
    assert stats is not None
    assert move_calls == 1
    assert stats["epochs_ran"] == 3


def test_device_cached_chunk_pack_preserves_fragment_protocol_tensors():
    ppo = PPO(FakeNet(), lr=1e-4, total_updates=0, lr_min=0.0, tbptt_window=8)

    for step in range(2):
        batch = make_policy_batch(
            batch_size=1,
            meta_dim=META_VECTOR_DIM,
            zeros=True,
        ).with_state(_make_state(float(step)))
        batch.action_feedback_tokens.fill_(10.0 + step)
        batch.meta_vec.fill_(20.0 + step)
        ppo.store_transition(
            batch,
            torch.tensor(1),
            torch.tensor(2 + step),
            torch.tensor(3 + step),
            torch.tensor(-0.5),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(float(step == 1)),
            policy_mask=torch.tensor(1.0),
            target_index=torch.tensor(4 + step),
            coarse_index=torch.tensor(5 + step),
            fine_index=torch.tensor(6 + step),
        )
    fragment = ppo.finalize_fragment()
    assert fragment is not None

    item = ppo._fragment_tensors(fragment)
    chunks = ppo._build_tbptt_chunks(
        observations=item["observations"],
        pre_step_snn_state=item["pre_step_snn_state"],
        actions=item["actions"],
        move_xs=item["move_x"],
        move_ys=item["move_y"],
        target_indices=item["target_index"],
        coarse_indices=item["coarse_index"],
        fine_indices=item["fine_index"],
        log_probs_old=item["log_probs_old"],
        advantages=torch.ones(2, device=ppo.device),
        returns=torch.ones(2, device=ppo.device),
        dones=item["dones"],
        episode_reset_mask=item["episode_reset_mask"],
        sample_masks=item["sample_masks"],
    )
    packed = ppo._pack_chunk_group(chunks)

    assert torch.allclose(
        packed["action_feedback_tokens"][:, 0].cpu(),
        fragment.action_feedback_tokens,
    )
    assert torch.allclose(packed["meta_vec"][:, 0].cpu(), fragment.meta_vec)
    assert packed["target_index"][:, 0].cpu().tolist() == [4, 5]
    assert packed["coarse_index"][:, 0].cpu().tolist() == [5, 6]
    assert packed["fine_index"][:, 0].cpu().tolist() == [6, 7]
    assert packed["initial_state"][0].shape == (1, 1, 1)
    assert packed["initial_state"][0].cpu().item() == pytest.approx(0.0)


def test_update_policy_uses_chunk_state_carry_and_resets_on_done():
    torch.manual_seed(0)
    net = SequenceCarryNet()
    ppo = PPO(net, lr=1e-3, total_updates=0, lr_min=0.0, tbptt_window=8)

    def _state(fill_value):
        tensor = torch.full((1, 1, 1), fill_value, dtype=torch.float32)
        return tensor.clone(), tensor.clone()

    stored_states = [0.0, 99.0, 7.0]
    dones = [0.0, 1.0, 1.0]
    for stored_state, done in zip(stored_states, dones):
        batch = make_policy_batch(
            batch_size=1,
            meta_dim=META_VECTOR_DIM,
            zeros=True,
        ).with_state(_state(stored_state))
        ppo.store_transition(
            batch,
            torch.tensor(1),
            torch.tensor(2),
            torch.tensor(3),
            torch.tensor(-0.5),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(done),
            policy_mask=torch.tensor(1.0),
        )

    losses, stats = ppo.update_policy(batch_size=8, epochs=1)

    assert losses
    assert stats is not None
    assert net.seen_states[:2] == pytest.approx([0.0, 1.0])


def test_pack_chunk_group_shapes_and_masks():
    ppo = PPO(FakeNet(), lr=1e-4, total_updates=0, lr_min=0.0, tbptt_window=8)

    for stored_state, done, policy_mask in (
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (7.0, 1.0, 1.0),
    ):
        batch = make_policy_batch(
            batch_size=1,
            meta_dim=META_VECTOR_DIM,
            zeros=True,
        ).with_state(_make_state(stored_state))
        ppo.store_transition(
            batch,
            torch.tensor(1),
            torch.tensor(2),
            torch.tensor(3),
            torch.tensor(-0.5),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(done),
            policy_mask=torch.tensor(policy_mask),
        )

    chunks = _build_chunks_from_memory(ppo)
    packed = ppo._pack_chunk_group(chunks)

    assert packed["spatial_obs"].shape[:2] == (2, 2)
    assert packed["entity_features"].shape[:2] == (2, 2)
    assert packed["selection_features"].shape[:2] == (2, 2)
    assert packed["meta_vec"].shape[:2] == (2, 2)
    assert packed["alive_mask"].tolist() == [[True, True], [True, False]]
    assert packed["policy_mask"].tolist() == [[1.0, 1.0], [0.0, 0.0]]
    assert packed["done"].tolist() == [[False, True], [True, False]]


def test_packed_replay_matches_reference_replay():
    net = SequenceCarryNet()
    ppo = PPO(net, lr=1e-3, total_updates=0, lr_min=0.0, tbptt_window=8)

    for stored_state, done in ((0.0, 0.0), (99.0, 1.0), (7.0, 1.0)):
        batch = make_policy_batch(
            batch_size=1,
            meta_dim=META_VECTOR_DIM,
            zeros=True,
        ).with_state(_make_state(stored_state))
        ppo.store_transition(
            batch,
            torch.tensor(1),
            torch.tensor(2),
            torch.tensor(3),
            torch.tensor(-0.5),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(done),
            policy_mask=torch.tensor(1.0),
        )

    chunks = _build_chunks_from_memory(ppo)
    reference = ppo._replay_chunk_group_reference(chunks)
    packed = ppo._replay_packed_chunk_group(ppo._pack_chunk_group(chunks))

    for column, chunk in enumerate(chunks):
        length = int(chunk["length"])
        ref_action, ref_target_log_prob, ref_target_entropy, ref_values = reference[column]
        assert torch.allclose(
            packed["action_logits"][:length, column],
            ref_action,
            atol=1e-6,
        )
        assert torch.allclose(
            packed["target_log_prob"][:length, column],
            ref_target_log_prob,
            atol=1e-6,
        )
        assert torch.allclose(
            packed["target_entropy"][:length, column],
            ref_target_entropy,
            atol=1e-6,
        )
        assert torch.allclose(
            packed["state_values"][:length, column],
            ref_values,
            atol=1e-6,
        )


def test_packed_replay_uses_one_forward_per_timestep_group():
    net = CountingReplayNet()
    ppo = PPO(net, lr=1e-4, total_updates=0, lr_min=0.0, tbptt_window=8)

    for stored_state, done in ((0.0, 0.0), (1.0, 1.0), (7.0, 1.0)):
        batch = make_policy_batch(
            batch_size=1,
            meta_dim=META_VECTOR_DIM,
            zeros=True,
        ).with_state(_make_state(stored_state))
        ppo.store_transition(
            batch,
            torch.tensor(1),
            torch.tensor(2),
            torch.tensor(3),
            torch.tensor(-0.5),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(done),
            policy_mask=torch.tensor(1.0),
        )

    packed = ppo._pack_chunk_group(_build_chunks_from_memory(ppo))
    replayed = ppo._replay_packed_chunk_group(packed)

    assert net.encode_step_calls == packed["max_steps"]
    assert replayed["forward_calls"] == packed["max_steps"]


def test_packed_replay_resets_only_done_rows_inside_group():
    net = RowResetNet()
    ppo = PPO(net, lr=1e-4, total_updates=0, lr_min=0.0, tbptt_window=2)

    stored_states = (0.0, 0.0, 5.0, 5.0)
    for stored_state in stored_states:
        batch = make_policy_batch(
            batch_size=1,
            meta_dim=META_VECTOR_DIM,
            zeros=True,
        ).with_state(_make_state(stored_state))
        ppo.store_transition(
            batch,
            torch.tensor(1),
            torch.tensor(2),
            torch.tensor(3),
            torch.tensor(-0.5),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            policy_mask=torch.tensor(1.0),
        )

    packed = ppo._pack_chunk_group(_build_chunks_from_memory(ppo))
    packed["done"][0, 0] = True
    ppo._replay_packed_chunk_group(packed)

    assert len(net.row_history) == 2
    assert net.row_history[0].tolist() == pytest.approx([0.0, 5.0])
    assert net.row_history[1].tolist() == pytest.approx([0.0, 6.0])
