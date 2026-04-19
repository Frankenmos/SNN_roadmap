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
        self.fc = nn.Linear(4, 2)
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
        batch_size = batch.batch_size
        base = self.fc(batch.meta_vec[:, :4].float()).sum(dim=-1)

        action_logits = self._repeat_logits(self._action_logits, batch_size)
        if action_logits is None:
            action_logits = torch.randn(batch_size, 3, device=self.device)
        action_logits = action_logits.to(self.device) + base.unsqueeze(-1) * 0

        move_x_logits = self._repeat_logits(self._move_x_logits, batch_size)
        if move_x_logits is None:
            move_x_logits = torch.randn(batch_size, 8, device=self.device)
        move_x_logits = move_x_logits.to(self.device) + base.unsqueeze(-1) * 0

        move_y_logits = self._repeat_logits(self._move_y_logits, batch_size)
        if move_y_logits is None:
            move_y_logits = torch.randn(batch_size, 8, device=self.device)
        move_y_logits = move_y_logits.to(self.device) + base.unsqueeze(-1) * 0

        state_value = base * 0.1
        return action_logits, move_x_logits, move_y_logits, state_value, batch.state_in


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
    assert log_prob == pytest.approx(float(expected_log_prob.item()))
    assert isinstance(value, float)
    assert next_state is None


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

    ppo = PPO(net, lr=1e-4, total_updates=0, lr_min=0.0)
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
    assert changed_params > 0
    assert ppo.memory == []
    assert ppo.final_next is None
