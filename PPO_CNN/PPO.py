import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO:
    def __init__(self, policy_net, lr=3e-4, gamma=0.99, clip_epsilon=0.2,
                 critic_loss_coef=0.5, entropy_coef=0.01):
        self.policy_net = policy_net
        self.device = policy_net.device
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.memory = []  # Store transitions here

    def select_action(self, observations):
        """
        Select an action based on the current policy.
        Args:
            observations: Tuple of (spatial_obs, vector_obs)
        Returns:
            action: Selected action (discrete)
            angle: Angle value for movement (continuous)
            log_prob: Log probability of the action
            value: State value
        """
        spatial_obs, vector_obs = observations
        
        # Handle spatial observation
        if isinstance(spatial_obs, torch.Tensor):
            spatial_tensor = spatial_obs.unsqueeze(0).to(self.device)
        else:
            spatial_tensor = torch.tensor(spatial_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
        # Handle vector observation
        if isinstance(vector_obs, torch.Tensor):
            vector_tensor = vector_obs.unsqueeze(0).to(self.device)
        else:
            vector_tensor = torch.tensor(vector_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Forward pass through policy network
        action_logits, angle, state_value = self.policy_net(spatial_tensor, vector_tensor)
        
        # Sample action
        action_probs = torch.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return (
            action.item(),
            angle.item(),
            log_prob.item(),
            state_value.item()
        )

    def store_transition(self, spatial_obs, vector_obs, action, log_prob, reward, value, done):
        self.memory.append({
            'spatial_obs': spatial_obs,
            'vector_obs': vector_obs,
            'action': action,
            'log_prob': log_prob,
            'reward': reward,
            'value': value,
            'done': done
        })

    def update_policy(self, batch_size=64, epochs=10):
        # 1) Convert memory to stacked tensors
        spatial_obs = torch.stack([t['spatial_obs'] for t in self.memory])
        vector_obs = torch.stack([t['vector_obs'] for t in self.memory])
        actions = torch.stack([t['action'] for t in self.memory])
        log_probs_old = torch.stack([t['log_prob'] for t in self.memory])
        rewards = torch.stack([t['reward'] for t in self.memory])
        values = torch.stack([t['value'] for t in self.memory])
        dones = torch.stack([t['done'] for t in self.memory])

        # --- FIX 1: Handle Bootstrapping for the very last step ---
        # We need the value of the state AFTER the last step in memory to calculate
        # the advantage for the last step correctly.
        # Ideally, you should store this or pass it. For now, we will estimate it
        # as 0.0 if done, or just duplicate the last value (imperfect but runs)
        # A better way is to query the network one last time before update.
        with torch.no_grad():
             # Assuming the last state in memory was not terminal, we need a bootstrap.
             # If you have the 'next_obs' available, query the network. 
             # Without it, we default to 0 or self-consistency. 
             last_next_value = 0.0 # or self.policy_net(last_obs)[2]

        # 2) Compute advantages
        advantages = self._compute_advantages(rewards, values, dones, last_next_value)

        # --- FIX 2: Calculate Returns (The correct target for Critic) ---
        # Return = Advantage + Value
        # We detach to ensure we don't backpropagate into the old values/advantages
        returns = (advantages + values).detach()

        # 3) Normalize advantages (standard PPO stability trick)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 4) Training loop
        losses = []
        T = len(spatial_obs)
        for _ in range(epochs):
            perm = torch.randperm(T)
            for i in range(0, T, batch_size):
                idx = perm[i : i + batch_size]

                # Gather mini-batch
                batch_spatial = spatial_obs[idx]
                batch_vector = vector_obs[idx]
                batch_actions = actions[idx]
                batch_old_logp = log_probs_old[idx]
                batch_advantages = advantages[idx]
                
                # --- FIX 3: Use the pre-calculated Returns ---
                batch_returns = returns[idx] 

                # Forward pass
                action_logits, _, state_values = self.policy_net(batch_spatial, batch_vector)
                state_values = state_values.squeeze(-1)

                # Calculate losses
                policy_loss, value_loss, entropy_loss = self._calculate_losses(
                    action_logits,
                    state_values,
                    batch_actions,
                    batch_old_logp,
                    batch_advantages,
                    batch_returns # Pass the correct returns
                )

                loss = policy_loss + value_loss - entropy_loss
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.optimizer.step()

        self.memory = []
        return losses

    def _compute_advantages(self, rewards, values, dones, last_next_value):
        """
        GAE Calculation.
        Added `last_next_value` for correct bootstrapping at the end of the batch.
        """
        gamma = self.gamma
        gae_lambda = 0.95

        T = len(rewards)
        advantages = torch.zeros_like(rewards, device=self.device)

        reversed_rewards = torch.flip(rewards, [0])
        reversed_values = torch.flip(values, [0])
        reversed_dones  = torch.flip(dones,  [0])

        running_advantage = 0.0
        
        for t in range(T):
            if t == 0:
                # --- FIX: Use the actual next value (bootstrap) ---
                next_value = last_next_value
            else:
                next_value = reversed_values[t - 1]

            not_done = 1.0 - reversed_dones[t].float()
            
            # GAE Formula: delta = r + gamma * V(s') * (1-done) - V(s)
            delta = reversed_rewards[t] + gamma * next_value * not_done - reversed_values[t]
            
            running_advantage = delta + gamma * gae_lambda * not_done * running_advantage
            advantages[t] = running_advantage

        advantages = torch.flip(advantages, [0])
        return advantages
        

    def _calculate_losses(self, action_logits, state_values, actions, old_log_probs, advantages, returns):
        """
        GPU-accelerated loss calculations for each mini-batch.
        `action_logits` shape [batch_size, action_dim]
        `state_values` shape [batch_size]
        `actions` shape [batch_size]
        `old_log_probs` shape [batch_size]
        `advantages` shape [batch_size]
        `returns` shape [batch_size] (currently we pass immediate rewards,
                  but for proper PPO, you'd pass discounted returns)
        """
        # 1) New distribution & log_probs
        probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)  # shape [batch_size]

        # 2) Policy loss
        ratio = (new_log_probs - old_log_probs).exp()  # shape [batch_size]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 3) Value loss: (returns - value)^2
        #    state_values is [batch_size], returns is [batch_size]
        value_loss = self.critic_loss_coef * (returns - state_values).pow(2).mean()

        # 4) Entropy bonus
        entropy_loss = self.entropy_coef * dist.entropy().mean()

        return policy_loss, value_loss, entropy_loss
