import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PPO:
    def __init__(
        self,
        policy_net,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        critic_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        """Proximal Policy Optimization for a discrete action + angle policy."""
        self.policy_net = policy_net
        self.device = policy_net.device
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef

        # Each item is:
        #  'spatial_obs', 'vector_obs', 'action', 'log_prob', 'reward', 'value', 'done'
        self.memory = []

    # ------------------------------------------------------------------
    # ACTING
    # ------------------------------------------------------------------
    def select_action(self, observations, state=None):
        """
        Args:
            observations: (spatial_obs, vector_obs) as np arrays or tensors.
            state: SNN hidden state.
        Returns:
            action, angle, log_prob, value, next_state
        """
        spatial_obs, vector_obs = observations

        # Spatial obs -> [1, C, H, W] on device
        if isinstance(spatial_obs, torch.Tensor):
            spatial_tensor = spatial_obs.to(self.device).unsqueeze(0)
        else:
            spatial_tensor = torch.tensor(
                spatial_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        # Vector obs -> [1, D] on device
        if isinstance(vector_obs, torch.Tensor):
            vector_tensor = vector_obs.to(self.device).unsqueeze(0)
        else:
            vector_tensor = torch.tensor(
                vector_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        with torch.no_grad():
            action_logits, angle, state_value, next_state = self.policy_net(
                spatial_tensor, vector_tensor, state=state
            )

            action_probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()          # [1]
            log_prob = dist.log_prob(action)  # [1]

        return (
            int(action.item()),
            float(angle.item()),
            float(log_prob.item()),
            float(state_value.squeeze(-1).item()),
            next_state,
        )

    def store_transition(
        self,
        spatial_obs,
        vector_obs,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        spatial_obs/vector_obs are kept as raw numpy/CPU; they’re moved
        to GPU in bulk inside update_policy. Scalars are tensors already.
        """
        self.memory.append(
            {
                "spatial_obs": spatial_obs,
                "vector_obs": vector_obs,
                "action": action.detach(),
                "log_prob": log_prob.detach(),
                "reward": reward.detach(),
                "value": value.detach(),
                "done": done.detach(),
            }
        )

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    def update_policy(self, batch_size: int = 64, epochs: int = 10):
        """Run a PPO update on all rollouts in memory (on self.device)."""
        if not self.memory:
            return []

        # ---- 1) Convert memory to tensors on device ----
        spatial_list = []
        vector_list = []
        for t in self.memory:
            s = t["spatial_obs"]
            v = t["vector_obs"]
            if isinstance(s, torch.Tensor):
                spatial_list.append(s.to(self.device))
            else:
                spatial_list.append(
                    torch.tensor(s, dtype=torch.float32, device=self.device)
                )
            if isinstance(v, torch.Tensor):
                vector_list.append(v.to(self.device))
            else:
                vector_list.append(
                    torch.tensor(v, dtype=torch.float32, device=self.device)
                )

        spatial_obs = torch.stack(spatial_list, dim=0)   # [T, C, H, W]
        vector_obs = torch.stack(vector_list, dim=0)     # [T, D]

        actions = torch.stack([t["action"].to(self.device) for t in self.memory])
        log_probs_old = torch.stack(
            [t["log_prob"].to(self.device) for t in self.memory]
        )
        rewards = torch.stack([t["reward"].to(self.device) for t in self.memory])
        values = torch.stack([t["value"].to(self.device) for t in self.memory])
        dones = torch.stack([t["done"].to(self.device) for t in self.memory])

        # ---- 2) Bootstrap from last state ----
        with torch.no_grad():
            if dones[-1].item() == 1.0:
                last_next_value = torch.zeros((), device=self.device)
            else:
                last_spatial = spatial_obs[-1].unsqueeze(0)
                last_vector = vector_obs[-1].unsqueeze(0)
                _, _, last_next_value, _ = self.policy_net(
                    last_spatial, last_vector, state=None
                )
                last_next_value = last_next_value.squeeze(-1)

        # ---- 3) GAE advantages + returns ----
        advantages = self._compute_advantages(
            rewards, values, dones, last_next_value
        )  # [T]
        returns = (advantages + values).detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- 4) PPO minibatch training ----
        T = spatial_obs.size(0)
        losses = []

        for _ in range(epochs):
            perm = torch.randperm(T, device=self.device)
            for start in range(0, T, batch_size):
                idx = perm[start : start + batch_size]

                batch_spatial = spatial_obs[idx]
                batch_vector = vector_obs[idx]
                batch_actions = actions[idx]
                batch_old_logp = log_probs_old[idx]
                batch_adv = advantages[idx]
                batch_returns = returns[idx]

                # Stateless SNN during training
                action_logits, _, state_values, _ = self.policy_net(
                    batch_spatial, batch_vector, state=None
                )
                state_values = state_values.squeeze(-1)

                policy_loss, value_loss, entropy_loss = self._calculate_losses(
                    action_logits,
                    state_values,
                    batch_actions,
                    batch_old_logp,
                    batch_adv,
                    batch_returns,
                )

                loss = policy_loss + value_loss - entropy_loss
                losses.append(float(loss.item()))

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.optimizer.step()

        self.memory = []
        return losses

    # ------------------------------------------------------------------
    # INTERNAL: advantage + losses
    # ------------------------------------------------------------------
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_next_value: torch.Tensor,
        gae_lambda: float = 0.95,
    ) -> torch.Tensor:
        """GAE(λ) on self.device, shapes [T]."""
        gamma = self.gamma
        T = rewards.size(0)

        advantages = torch.zeros_like(rewards, device=self.device)
        running_advantage = torch.zeros((), device=self.device)
        next_value = last_next_value

        for t in reversed(range(T)):
            not_done = 1.0 - dones[t].float()
            delta = rewards[t] + gamma * next_value * not_done - values[t]
            running_advantage = delta + gamma * gae_lambda * not_done * running_advantage
            advantages[t] = running_advantage
            next_value = values[t]

        return advantages

    def _calculate_losses(
        self,
        action_logits: torch.Tensor,
        state_values: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ):
        """PPO loss for one minibatch (on GPU)."""
        probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)

        # Policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = self.critic_loss_coef * (returns - state_values).pow(2).mean()

        # Entropy bonus
        entropy_loss = self.entropy_coef * dist.entropy().mean()

        return policy_loss, value_loss, entropy_loss
