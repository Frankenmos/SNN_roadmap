import torch
import torch.nn as nn
import pytest
from PPO_CNN.PPO import PPO
import numpy as np

# Add a mock policy network for testing
class MockPolicyNet(nn.Module):
    def __init__(self, action_dim=10):
        super(MockPolicyNet, self).__init__()
        self.action_dim = action_dim
        self.device = torch.device('cpu')
        # Add a dummy parameter to be recognized by the optimizer
        self.fc = nn.Linear(1, 1)


    def to(self, device):
        self.device = device
        return self

    def forward(self, spatial_obs, vector_obs, state=None):
        batch_size = spatial_obs.shape[0]
        # Route through self.fc so outputs have grad_fn for backprop
        dummy = self.fc(spatial_obs.flatten(1)[:, :1]).squeeze(-1)  # [B]
        action_logits = torch.randn(batch_size, self.action_dim, device=self.device) + dummy.unsqueeze(-1) * 0
        angle = torch.randn(batch_size, device=self.device) + dummy * 0
        state_value = dummy * 0.1
        return action_logits, angle, state_value, None

@pytest.fixture
def ppo_agent():
    policy_net = MockPolicyNet()
    agent = PPO(policy_net)
    return agent

def test_init(ppo_agent):
    assert ppo_agent.gamma == 0.99
    assert ppo_agent.clip_epsilon == 0.2
    assert ppo_agent.critic_loss_coef == 0.5
    assert ppo_agent.entropy_coef == 0.01
    assert isinstance(ppo_agent.optimizer, torch.optim.Adam)

def test_select_action(ppo_agent):
    spatial_obs = torch.randn(1, 3, 84, 84)
    vector_obs = torch.randn(1, 10)
    action, angle, log_prob, value, next_state = ppo_agent.select_action((spatial_obs, vector_obs))
    assert isinstance(action, int)
    assert 0 <= action < ppo_agent.policy_net.action_dim
    assert isinstance(angle, float)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)

def test_store_transition(ppo_agent):
    spatial_obs = torch.randn(3, 84, 84)
    vector_obs = torch.randn(10)
    ppo_agent.store_transition(
        spatial_obs, vector_obs,
        torch.tensor(1), torch.tensor(-0.5), torch.tensor(1.0), torch.tensor(0.5), torch.tensor(False),
    )
    assert len(ppo_agent.memory) == 1
    transition = ppo_agent.memory[0]
    assert torch.equal(transition['spatial_obs'], spatial_obs)
    assert torch.equal(transition['vector_obs'], vector_obs)
    assert transition['action'] == 1
    assert transition['log_prob'] == -0.5
    assert transition['reward'] == 1.0
    assert transition['value'] == 0.5
    assert not transition['done']

def test_compute_advantages(ppo_agent):
    # Test case: a single trajectory of 4 steps ending in a terminal state.
    # rewards = [r1, r2, r3, r4]
    # values = [v1, v2, v3, v4]
    # dones = [False, False, False, True]
    # GAE(lambda=0.95, gamma=0.99)
    # delta_t = r_t + gamma * V(s_{t+1}) * (1-d_t) - V(s_t)
    # A_t = delta_t + gamma * lambda * A_{t+1} * (1-d_t)
    # For t=3 (last step):
    # delta_3 = 1.0 + 0.99 * 0.5 * 0 - 0.5 = 0.5
    # A_3 = delta_3 = 0.5
    # For t=2:
    # delta_2 = 1.0 + 0.99 * 0.5 * 1 - 0.5 = 0.995
    # A_2 = delta_2 + 0.99 * 0.95 * A_3 = 0.995 + 0.9405 * 0.5 = 1.46525
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5, 0.5])
    dones = torch.tensor([False, False, False, True])

    advantages = ppo_agent._compute_advantages(rewards, values, dones, last_next_value=torch.tensor(0.0))

    expected_adv_3 = 0.5
    expected_adv_2 = 0.995 + 0.99 * 0.95 * expected_adv_3
    expected_adv_1 = 0.995 + 0.99 * 0.95 * expected_adv_2
    expected_adv_0 = 0.995 + 0.99 * 0.95 * expected_adv_1

    expected_advantages = torch.tensor([expected_adv_0, expected_adv_1, expected_adv_2, expected_adv_3])

    assert advantages.shape == rewards.shape
    assert torch.allclose(advantages, expected_advantages, atol=1e-5)


def test_calculate_losses(ppo_agent):
    action_logits = torch.randn(4, 10)
    state_values = torch.randn(4)
    actions = torch.randint(0, 10, (4,))
    old_log_probs = torch.randn(4)
    advantages = torch.randn(4)
    returns = torch.randn(4)
    policy_loss, value_loss, entropy_loss = ppo_agent._calculate_losses(
        action_logits, state_values, actions, old_log_probs, advantages, returns
    )
    assert isinstance(policy_loss, torch.Tensor)
    assert isinstance(value_loss, torch.Tensor)
    assert isinstance(entropy_loss, torch.Tensor)
    assert policy_loss.shape == torch.Size([])
    assert value_loss.shape == torch.Size([])
    assert entropy_loss.shape == torch.Size([])

def test_update_policy(ppo_agent):
    # Store some transitions
    for _ in range(10):
        spatial_obs = torch.randn(3, 84, 84)
        vector_obs = torch.randn(10)
        action, _, log_prob, value, _ = ppo_agent.select_action((spatial_obs.unsqueeze(0), vector_obs.unsqueeze(0)))
        ppo_agent.store_transition(
            spatial_obs, vector_obs,
            torch.tensor(action), torch.tensor(log_prob), torch.tensor(1.0), torch.tensor(value), torch.tensor(False),
        )

    # Get the initial model parameters
    initial_params = [p.clone() for p in ppo_agent.policy_net.parameters()]

    # Run the update
    ppo_agent.update_policy(batch_size=4, epochs=1)

    # Check if memory is cleared
    assert len(ppo_agent.memory) == 0

    # Check if the model parameters have been updated
    for initial, final in zip(initial_params, ppo_agent.policy_net.parameters()):
        assert not torch.equal(initial, final)
