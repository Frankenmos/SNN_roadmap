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

        # Make sure outputs depend on self.fc so gradients can flow
        # create a dummy input for fc
        dummy_in = torch.ones((batch_size, 1), device=self.device)
        dummy_out = self.fc(dummy_in) # [batch, 1]

        # Use dummy_out to influence outputs
        action_logits = torch.randn(batch_size, self.action_dim, device=self.device) + dummy_out
        xy_mean_raw = torch.randn(batch_size, 2, device=self.device) + dummy_out
        state_value = torch.randn(batch_size, 1, device=self.device) + dummy_out

        next_state = None # Mock state
        return action_logits, xy_mean_raw, state_value, next_state

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
    vector_obs = torch.randn(1, 100)

    # Returns 6 values now: action, xy_env, log_prob_total, value, next_state, xy_raw_sample
    action, xy_env, log_prob, value, next_state, xy_raw = ppo_agent.select_action((spatial_obs, vector_obs))

    assert isinstance(action, torch.Tensor)
    assert action.shape == (1,)

    assert isinstance(xy_env, torch.Tensor)
    assert xy_env.shape == (1, 2)

    assert isinstance(log_prob, torch.Tensor)

    assert isinstance(value, torch.Tensor)

    assert isinstance(xy_raw, torch.Tensor)

def test_store_transition(ppo_agent):
    spatial_obs = torch.randn(3, 84, 84)
    vector_obs = torch.randn(100)

    action = torch.tensor([1])
    xy_raw = torch.tensor([0.1, 0.2])
    log_prob = torch.tensor([-0.5])
    reward = torch.tensor([1.0])
    value = torch.tensor([0.5])
    done = torch.tensor([0.0])

    ppo_agent.store_transition(spatial_obs, vector_obs, action, xy_raw, log_prob, reward, value, done)

    assert len(ppo_agent.memory) == 1
    transition = ppo_agent.memory[0]
    assert torch.equal(transition['spatial_obs'], spatial_obs)
    assert torch.equal(transition['vector_obs'], vector_obs)
    assert torch.equal(transition['action'], action)
    assert torch.equal(transition['xy_raw'], xy_raw)
    assert torch.equal(transition['log_prob'], log_prob)
    assert torch.equal(transition['reward'], reward)
    assert torch.equal(transition['value'], value)
    assert torch.equal(transition['done'], done)

def test_compute_advantages(ppo_agent):
    # Test case: single trajectory of 4 steps
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 0.0, 0.0, 1.0]) # Use float 0.0/1.0
    last_next_value = torch.tensor(0.0)

    advantages = ppo_agent._compute_advantages(rewards, values, dones, last_next_value)

    assert advantages.shape == rewards.shape
    assert advantages[-1] == 0.5

def test_calculate_losses(ppo_agent):
    batch_size = 4
    action_logits = torch.randn(batch_size, 10, requires_grad=True)
    xy_mean_raw = torch.randn(batch_size, 2, requires_grad=True)
    state_values = torch.randn(batch_size, requires_grad=True)

    actions = torch.randint(0, 10, (batch_size,))
    xy_raw = torch.randn(batch_size, 2)
    old_log_probs = torch.randn(batch_size)
    advantages = torch.randn(batch_size)
    returns = torch.randn(batch_size)

    policy_loss, value_loss, entropy_loss = ppo_agent._calculate_losses(
        action_logits, xy_mean_raw, state_values, actions, xy_raw, old_log_probs, advantages, returns
    )
    assert isinstance(policy_loss, torch.Tensor)
    assert isinstance(value_loss, torch.Tensor)
    assert isinstance(entropy_loss, torch.Tensor)

def test_update_policy(ppo_agent):
    # Store some transitions
    for _ in range(10):
        spatial_obs = torch.randn(3, 84, 84)
        vector_obs = torch.randn(100)

        # select_action returns tensors on device
        action, xy_env, log_prob, value, _, xy_raw = ppo_agent.select_action((spatial_obs.unsqueeze(0), vector_obs.unsqueeze(0)))

        reward = torch.tensor([1.0])
        done = torch.tensor([0.0])

        ppo_agent.store_transition(
            spatial_obs, vector_obs,
            action.squeeze(0),
            xy_raw.squeeze(0),
            log_prob.squeeze(0),
            reward,
            value.squeeze(0),
            done
        )

    initial_params = [p.clone() for p in ppo_agent.policy_net.parameters()]

    # Run the update
    losses = ppo_agent.update_policy(batch_size=4, epochs=1)

    assert len(ppo_agent.memory) == 0
    assert len(losses) > 0

    # Check param update
    for initial, final in zip(initial_params, ppo_agent.policy_net.parameters()):
        assert not torch.equal(initial, final)
