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
    # ------------------------------------------------------------------
    # ACTING
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ACTING
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ACTING
    # ------------------------------------------------------------------
    def select_action(self, observations, state=None):
        """
        Args:
            observations: (spatial_obs, vector_obs) as np arrays or tensors.
            state: SNN hidden state.
        Returns:
            action (Tensor), xy_env (Tensor), log_prob_total (Tensor), value (Tensor), next_state, xy_raw_sample (Tensor)
        """
        spatial_obs, vector_obs = observations

        # Ensure inputs are tensors on device
        if not isinstance(spatial_obs, torch.Tensor):
            spatial_tensor = torch.tensor(
                spatial_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        else:
            spatial_tensor = spatial_obs
            if spatial_tensor.dim() == 3:
                spatial_tensor = spatial_tensor.unsqueeze(0)

        if not isinstance(vector_obs, torch.Tensor):
            vector_tensor = torch.tensor(
                vector_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        else:
            vector_tensor = vector_obs
            if vector_tensor.dim() == 1:
                vector_tensor = vector_tensor.unsqueeze(0)

        with torch.no_grad():
            action_logits, xy_mean_raw, state_value, next_state = self.policy_net(
                spatial_tensor, vector_tensor, state=state
            )

            # 1) Discrete action
            action_probs = torch.softmax(action_logits, dim=-1)
            dist_a = torch.distributions.Categorical(action_probs)
            action = dist_a.sample()          # [1]
            logp_a = dist_a.log_prob(action)  # [1]

            # 2) Continuous XY (raw)
            if hasattr(self.policy_net, "log_std_xy"):
                std = self.policy_net.log_std_xy.exp().unsqueeze(0).expand_as(xy_mean_raw)
            else:
                # Fallback if parameter missing (shouldn't happen with new policy_net)
                std = torch.full_like(xy_mean_raw, 0.3)

            dist_xy = torch.distributions.Normal(xy_mean_raw, std)
            xy_raw_sample = dist_xy.sample()      # [1, 2]
            logp_xy = dist_xy.log_prob(xy_raw_sample).sum(dim=-1)  # sum over x,y → [1]

            # 3) Env action in [0,1]
            xy_env = torch.sigmoid(xy_raw_sample)

            logp_total = logp_a + logp_xy

        # Return TENSORS (detached) to avoid CPU sync
        return (
            action.detach(),         # [1]
            xy_env.detach(),         # [1, 2] used by action_space
            logp_total.detach(),     # [1]
            state_value.detach(),    # [1]
            next_state,
            xy_raw_sample.detach(),  # [1, 2] for training
        )

    def store_transition(
        self,
        spatial_obs,
        vector_obs,
        action: torch.Tensor,
        xy_raw: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Store transition data. 
        Optimized: Expects inputs to be Tensors on Device (GPU) where possible.
        """
        # Ensure spatial/vector are tensors on device to save transfer time later
        if not isinstance(spatial_obs, torch.Tensor):
             spatial_obs = torch.tensor(spatial_obs, dtype=torch.float32, device=self.device)
        if not isinstance(vector_obs, torch.Tensor):
             vector_obs = torch.tensor(vector_obs, dtype=torch.float32, device=self.device)
             
        # Ensure scalar tensors have correct shape/device
        if action.device != self.device: action = action.to(self.device)
        if xy_raw.device != self.device: xy_raw = xy_raw.to(self.device)
        if log_prob.device != self.device: log_prob = log_prob.to(self.device)
        if reward.device != self.device: reward = reward.to(self.device)
        if value.device != self.device: value = value.to(self.device)
        if done.device != self.device: done = done.to(self.device)

        self.memory.append(
            {
                "spatial_obs": spatial_obs,
                "vector_obs": vector_obs,
                "action": action,
                "xy_raw": xy_raw,
                "log_prob": log_prob,      # this is TOTAL log prob
                "reward": reward,
                "value": value,
                "done": done,
            }
        )

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    def update_policy(self, batch_size: int = 64, epochs: int = 10):
        """Run a PPO update on all rollouts in memory (on self.device)."""
        if not self.memory:
            return []

        # ---- 1) Stack tensors (Already on GPU) ----
        # This is much faster than converting from list of numpy arrays
        spatial_obs = torch.stack([t["spatial_obs"] for t in self.memory])
        vector_obs = torch.stack([t["vector_obs"] for t in self.memory])
        actions = torch.stack([t["action"] for t in self.memory]).squeeze() # [T]
        xy_raws = torch.stack([t["xy_raw"] for t in self.memory]) # [T, 2]
        log_probs_old = torch.stack([t["log_prob"] for t in self.memory]).squeeze() # [T]
        rewards = torch.stack([t["reward"] for t in self.memory]).squeeze() # [T]
        values = torch.stack([t["value"] for t in self.memory]).squeeze() # [T]
        dones = torch.stack([t["done"] for t in self.memory]).squeeze() # [T]

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

        # ---- 4) PPO minibatch training (RECURRENT UNROLL) ----
        T = spatial_obs.size(0)
        losses = []

        for _ in range(epochs):
            perm = torch.randperm(T, device=self.device)
            for start in range(0, T, batch_size):
                idx = perm[start : start + batch_size]

                batch_spatial = spatial_obs[idx]
                batch_vector = vector_obs[idx]
                batch_actions = actions[idx]
                batch_xy_raw = xy_raws[idx]
                batch_old_logp = log_probs_old[idx]
                batch_adv = advantages[idx]
                batch_returns = returns[idx]

                # Vectorized forward pass (Stateless between steps, Stateful within SNN internal steps)
                # This restores full GPU parallelism (Batch=256)
                action_logits, xy_mean_raw, state_values, _ = self.policy_net(
                    batch_spatial, batch_vector, state=None
                )
                state_values = state_values.squeeze(-1)

                policy_loss, value_loss, entropy_loss = self._calculate_losses(
                    action_logits,
                    xy_mean_raw,
                    state_values,
                    batch_actions,
                    batch_xy_raw,
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
        xy_mean_raw: torch.Tensor,
        state_values: torch.Tensor,
        actions: torch.Tensor,
        xy_raw: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ):
        """PPO loss for one minibatch (on GPU)."""
        # Discrete
        probs = torch.softmax(action_logits, dim=-1)
        dist_a = torch.distributions.Categorical(probs)
        new_logp_a = dist_a.log_prob(actions)

        # Continuous XY
        if hasattr(self.policy_net, "log_std_xy"):
            std = self.policy_net.log_std_xy.exp().unsqueeze(0).expand_as(xy_mean_raw)
        else:
            std = torch.full_like(xy_mean_raw, 0.3)

        dist_xy = torch.distributions.Normal(xy_mean_raw, std)
        new_logp_xy = dist_xy.log_prob(xy_raw).sum(dim=-1)

        new_log_probs = new_logp_a + new_logp_xy

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
        entropy_a = dist_a.entropy().mean()
        entropy_xy = dist_xy.entropy().sum(-1).mean()
        entropy_loss = self.entropy_coef * (entropy_a + 0.01 * entropy_xy)

        return policy_loss, value_loss, entropy_loss
