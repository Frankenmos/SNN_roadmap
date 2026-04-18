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
        total_updates: int = 0,
        lr_min: float = 0.0,
    ):
        """Proximal Policy Optimization with screen-point action head.

        total_updates: number of `update_policy` calls expected across the
            whole run. Used to size the CosineAnnealingLR schedule so lr
            decays from `lr` to `lr_min` over the full training run.
            Pass 0 to disable the schedule.
        """
        self.policy_net = policy_net
        self.device = policy_net.device
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef

        if total_updates > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_updates, eta_min=lr_min,
            )
        else:
            self.scheduler = None

        # Each item is:
        #  'spatial_obs', 'vector_obs', 'action', 'log_prob', 'reward', 'value', 'done'
        self.memory = []

    # ------------------------------------------------------------------
    # ACTING
    # ------------------------------------------------------------------
    # Action id for "move" in the high-level action head.
    MOVE_ACTION_ID = 1

    def select_action(self, observations, state=None):
        """
        Args:
            observations: (spatial_obs, vector_obs) as np arrays or tensors.
            state: SNN hidden state.
        Returns:
            action, move_x, move_y, log_prob, value, next_state
            log_prob = logp(action) + is_move * (logp(move_x) + logp(move_y))
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

        with torch.no_grad(), torch.amp.autocast(
            'cuda',
            dtype=self.policy_net.amp_dtype,
            enabled=self.policy_net.use_amp,
        ):
            action_logits, move_x_logits, move_y_logits, state_value, next_state = \
                self.policy_net(spatial_tensor, vector_tensor, state=state)

            # softmax in fp32 for numerical stability of the probability sum
            action_dist = torch.distributions.Categorical(
                logits=action_logits.float()
            )
            move_x_dist = torch.distributions.Categorical(
                logits=move_x_logits.float()
            )
            move_y_dist = torch.distributions.Categorical(
                logits=move_y_logits.float()
            )

            action = action_dist.sample()   # [1]
            move_x = move_x_dist.sample()   # [1]
            move_y = move_y_dist.sample()   # [1]

            # Joint log-prob: move coords only count when action == move.
            is_move = (action == self.MOVE_ACTION_ID).float()
            log_prob = (
                action_dist.log_prob(action)
                + is_move * (
                    move_x_dist.log_prob(move_x)
                    + move_y_dist.log_prob(move_y)
                )
            )

        return (
            int(action.item()),
            int(move_x.item()),
            int(move_y.item()),
            float(log_prob.item()),
            float(state_value.squeeze(-1).item()),
            next_state,
        )

    def store_transition(
        self,
        spatial_obs,
        vector_obs,
        action: torch.Tensor,
        move_x: torch.Tensor,
        move_y: torch.Tensor,
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
                "move_x": move_x.detach(),
                "move_y": move_y.detach(),
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
        move_xs = torch.stack([t["move_x"].to(self.device) for t in self.memory])
        move_ys = torch.stack([t["move_y"].to(self.device) for t in self.memory])
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
                _, _, _, last_next_value, _ = self.policy_net(
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
        # Per-minibatch accumulators for end-of-update logging.
        acc_policy = []
        acc_value = []
        acc_entropy = []
        acc_kl = []
        acc_clip_frac = []
        acc_grad_norm = []

        for _ in range(epochs):
            perm = torch.randperm(T, device=self.device)
            for start in range(0, T, batch_size):
                idx = perm[start : start + batch_size]

                batch_spatial = spatial_obs[idx]
                batch_vector = vector_obs[idx]
                batch_actions = actions[idx]
                batch_move_x = move_xs[idx]
                batch_move_y = move_ys[idx]
                batch_old_logp = log_probs_old[idx]
                batch_adv = advantages[idx]
                batch_returns = returns[idx]

                # Forward + loss in autocast so matmuls/convs run in fp16 on GPU.
                with torch.amp.autocast(
                    'cuda',
                    dtype=self.policy_net.amp_dtype,
                    enabled=self.policy_net.use_amp,
                ):
                    # Stateless SNN during training
                    action_logits, move_x_logits, move_y_logits, state_values, _ = \
                        self.policy_net(batch_spatial, batch_vector, state=None)
                    state_values = state_values.squeeze(-1)

                    policy_loss, value_loss, entropy_loss, diag = \
                        self._calculate_losses(
                            action_logits,
                            move_x_logits,
                            move_y_logits,
                            state_values,
                            batch_actions,
                            batch_move_x,
                            batch_move_y,
                            batch_old_logp,
                            batch_adv,
                            batch_returns,
                        )
                    loss = policy_loss + value_loss - entropy_loss

                losses.append(float(loss.item()))
                acc_policy.append(float(policy_loss.item()))
                acc_value.append(float(value_loss.item()))
                acc_entropy.append(float(diag["entropy_mean"].item()))
                acc_kl.append(float(diag["approx_kl"].item()))
                acc_clip_frac.append(float(diag["clip_frac"].item()))

                self.optimizer.zero_grad()
                # Scale loss so tiny fp16 grads don't underflow to zero.
                self.policy_net.scaler.scale(loss).backward()
                # Unscale before clipping so the 0.5 threshold is in real units.
                self.policy_net.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(), 0.5
                )
                acc_grad_norm.append(float(grad_norm.item()))
                # Step skipped automatically if grads contained inf/nan.
                self.policy_net.scaler.step(self.optimizer)
                self.policy_net.scaler.update()

        # Explained variance on the full rollout: 1 - var(returns - values) / var(returns)
        with torch.no_grad():
            var_returns = returns.var(unbiased=False)
            explained_var = 1.0 - (returns - values).var(unbiased=False) / (
                var_returns + 1e-8
            )

        stats = {
            "mean_policy_loss": float(np.mean(acc_policy)),
            "mean_value_loss": float(np.mean(acc_value)),
            "mean_entropy": float(np.mean(acc_entropy)),
            "mean_kl": float(np.mean(acc_kl)),
            "clip_fraction": float(np.mean(acc_clip_frac)),
            "explained_variance": float(explained_var.item()),
            "grad_norm": float(np.mean(acc_grad_norm)),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

        # Step the LR schedule once per rollout update.
        if self.scheduler is not None:
            self.scheduler.step()

        self.memory = []
        return losses, stats

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
        move_x_logits: torch.Tensor,
        move_y_logits: torch.Tensor,
        state_values: torch.Tensor,
        actions: torch.Tensor,
        move_x: torch.Tensor,
        move_y: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ):
        """PPO loss for one minibatch (on GPU). Returns loss terms plus
        diagnostic scalars used by the logger."""
        # fp32 for numerically stable softmax / log_softmax under autocast.
        action_dist = torch.distributions.Categorical(
            logits=action_logits.float()
        )
        move_x_dist = torch.distributions.Categorical(
            logits=move_x_logits.float()
        )
        move_y_dist = torch.distributions.Categorical(
            logits=move_y_logits.float()
        )

        # Joint log-prob must match how it was stored in the rollout.
        is_move = (actions == self.MOVE_ACTION_ID).float()
        new_log_probs = (
            action_dist.log_prob(actions)
            + is_move * (
                move_x_dist.log_prob(move_x) + move_y_dist.log_prob(move_y)
            )
        )

        # Policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = self.critic_loss_coef * (returns - state_values).pow(2).mean()

        import math
        # Entropy bonus: Normalize each head's entropy by its log(n) so all heads contribute to [0, 1]
        H_action_norm = action_dist.entropy() / math.log(action_logits.size(-1))
        H_x_norm = move_x_dist.entropy() / math.log(move_x_logits.size(-1))
        H_y_norm = move_y_dist.entropy() / math.log(move_y_logits.size(-1))

        # Entropy bonus: same masking — move-head entropy only when moving.
        entropy = (
            H_action_norm
            + is_move * (H_x_norm + H_y_norm)
        )
        entropy_loss = self.entropy_coef * entropy.mean()

        # Diagnostics (detached, scalar tensors).
        with torch.no_grad():
            # Schulman's k3 approximation of KL(old || new).
            approx_kl = ((ratio - 1.0) - (new_log_probs - old_log_probs)).mean()
            clip_frac = (
                (ratio - 1.0).abs() > self.clip_epsilon
            ).float().mean()
            entropy_mean = entropy.mean()

        return policy_loss, value_loss, entropy_loss, {
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "entropy_mean": entropy_mean,
        }
