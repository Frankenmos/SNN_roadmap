import pytest
import torch
import torch.nn as nn

from MockedEnv.policy_batch import make_policy_batch
from PPO_CNN.PPO import PPO
from PPO_CNN.policy_input import (
    META_VECTOR_DIM,
    SPATIAL_OBS_SHAPE,
)
from PPO_CNN.policy_network import PolicyNetwork


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
        return latent, state_value, state_in

    def action_head(self, latent):
        batch_size = latent.size(0)
        base = latent.sum(dim=-1)
        action_logits = self._repeat_logits(self._action_logits, batch_size)
        if action_logits is None:
            action_logits = torch.randn(batch_size, 3, device=self.device)
        action_logits = action_logits.to(self.device) + base.unsqueeze(-1) * 0

        return action_logits

    def conditioned_spatial_head(self, latent, action_ids):
        batch_size = latent.size(0)
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
        latent, state_value, next_state = self.encode_step_tensors(
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
        move_x_logits, move_y_logits = self.conditioned_spatial_head(latent, action_ids)
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
        return latent, base, next_state

    def action_head(self, latent):
        base = latent[:, 0]
        return torch.stack((base, base + 0.1, base - 0.1), dim=-1)

    def conditioned_spatial_head(self, latent, action_ids):
        base = latent[:, 0]
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
        latent, state_value, next_state = self.encode_step_tensors(
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
        move_x_logits, move_y_logits = self.conditioned_spatial_head(latent, action_ids)
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
        return latent, base, next_state

    def action_head(self, latent):
        base = latent[:, 0]
        return torch.stack((base, base + 0.1, base - 0.1), dim=-1)

    def conditioned_spatial_head(self, latent, action_ids):
        base = latent[:, 0]
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
        latent, state_value, next_state = self.encode_step_tensors(
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
        move_x_logits, move_y_logits = self.conditioned_spatial_head(latent, action_ids)
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
    action_logits = torch.tensor([[-2.0, 3.0, 0.5]])
    move_x_logits = torch.tensor([[0.1, -0.1, 0.0, 0.2, 0.3, 1.4, 0.5, -0.7]])
    move_y_logits = torch.tensor([[-0.8, 0.0, 0.2, -0.4, 0.1, 0.3, 0.9, 1.7]])
    ppo = PPO(FakeNet(action_logits, move_x_logits, move_y_logits), lr=1e-4)

    batch = make_policy_batch(batch_size=1, meta_dim=8)
    action, move_x, move_y, log_prob, value, next_state = ppo.select_action(
        batch, deterministic=True,
    )

    expected_log_prob = (
        torch.log_softmax(action_logits, dim=-1)[0, 1]
        + torch.log_softmax(move_x_logits, dim=-1)[0, 5]
        + torch.log_softmax(move_y_logits, dim=-1)[0, 7]
    )

    assert action == 1
    assert move_x == 5
    assert move_y == 7
    assert log_prob == pytest.approx(float(expected_log_prob.item()), abs=1e-6)
    assert isinstance(value, float)
    assert next_state is None


def test_select_action_deterministic_attack_also_uses_spatial_log_prob():
    action_logits = torch.tensor([[-2.0, 0.5, 3.0]])
    move_x_logits = torch.tensor([[0.4, -0.1, 1.2, 0.0, -0.6, 0.7, 0.5, -0.2]])
    move_y_logits = torch.tensor([[-0.3, 0.0, 0.1, 0.2, 1.8, -0.5, 0.6, 0.7]])
    ppo = PPO(FakeNet(action_logits, move_x_logits, move_y_logits), lr=1e-4)

    batch = make_policy_batch(batch_size=1, meta_dim=8)
    action, move_x, move_y, log_prob, *_rest = ppo.select_action(
        batch,
        deterministic=True,
    )

    expected_log_prob = (
        torch.log_softmax(action_logits, dim=-1)[0, 2]
        + torch.log_softmax(move_x_logits, dim=-1)[0, 2]
        + torch.log_softmax(move_y_logits, dim=-1)[0, 4]
    )

    assert action == 2
    assert move_x == 2
    assert move_y == 4
    assert log_prob == pytest.approx(float(expected_log_prob.item()), abs=1e-6)


def test_select_action_no_op_ignores_spatial_log_prob():
    action_logits = torch.tensor([[3.0, -1.0, -2.0]])
    move_x_logits = torch.tensor([[0.4, -0.1, 1.2, 0.0, -0.6, 0.7, 0.5, -0.2]])
    move_y_logits = torch.tensor([[-0.3, 0.0, 0.1, 0.2, 1.8, -0.5, 0.6, 0.7]])
    ppo = PPO(FakeNet(action_logits, move_x_logits, move_y_logits), lr=1e-4)

    batch = make_policy_batch(batch_size=1, meta_dim=8)
    action, move_x, move_y, log_prob, *_rest = ppo.select_action(
        batch,
        deterministic=True,
    )

    expected_log_prob = torch.log_softmax(action_logits, dim=-1)[0, 0]

    assert action == 0
    assert move_x == 0
    assert move_y == 0
    assert log_prob == pytest.approx(float(expected_log_prob.item()), abs=1e-6)


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


def test_calculate_losses_reports_normalized_entropy():
    ppo = PPO(FakeNet(), lr=1e-4)
    batch_size = 128

    policy_loss, value_loss, entropy_loss, diag = ppo._calculate_losses(
        torch.randn(batch_size, 3),
        torch.randn(batch_size, 84),
        torch.randn(batch_size, 84),
        torch.randn(batch_size),
        torch.ones(batch_size, dtype=torch.long),
        torch.randint(0, 84, (batch_size,)),
        torch.randint(0, 84, (batch_size,)),
        torch.randn(batch_size),
        torch.randn(batch_size),
        torch.randn(batch_size),
    )

    assert policy_loss.ndim == 0
    assert value_loss.ndim == 0
    assert entropy_loss.ndim == 0
    assert 1.0 <= float(diag["entropy_mean"].item()) <= 3.0
    assert torch.isfinite(diag["approx_kl"])
    assert 0.0 <= float(diag["clip_frac"].item()) <= 1.0


def test_calculate_losses_no_op_samples_do_not_backprop_spatial_heads():
    ppo = PPO(FakeNet(), lr=1e-4)
    action_logits = torch.randn(8, 3, requires_grad=True)
    move_x_logits = torch.randn(8, 84, requires_grad=True)
    move_y_logits = torch.randn(8, 84, requires_grad=True)
    state_values = torch.randn(8, requires_grad=True)

    policy_loss, value_loss, entropy_loss, _diag = ppo._calculate_losses(
        action_logits,
        move_x_logits,
        move_y_logits,
        state_values,
        torch.zeros(8, dtype=torch.long),
        torch.randint(0, 84, (8,)),
        torch.randint(0, 84, (8,)),
        torch.zeros(8),
        torch.ones(8),
        torch.zeros(8),
    )
    total = policy_loss + value_loss - entropy_loss
    total.backward()

    if move_x_logits.grad is not None:
        assert torch.count_nonzero(move_x_logits.grad) == 0
    if move_y_logits.grad is not None:
        assert torch.count_nonzero(move_y_logits.grad) == 0


def test_update_policy_replays_state_and_clears_memory():
    torch.manual_seed(0)
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
            _, _, _, state_value, _ = net(batch.with_state(state))
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
    assert changed_params > 0
    assert ppo.memory == []
    assert ppo.final_next is None


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
            meta_dim=8,
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
            meta_dim=8,
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
            meta_dim=8,
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
        ref_action, ref_move_x, ref_move_y, ref_values = reference[column]
        assert torch.allclose(
            packed["action_logits"][:length, column],
            ref_action,
            atol=1e-6,
        )
        assert torch.allclose(
            packed["move_x_logits"][:length, column],
            ref_move_x,
            atol=1e-6,
        )
        assert torch.allclose(
            packed["move_y_logits"][:length, column],
            ref_move_y,
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
            meta_dim=8,
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
            meta_dim=8,
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
