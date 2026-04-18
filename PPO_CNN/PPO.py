import math

import numpy as np
import torch
import torch.optim as optim


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
        target_kl: float | None = None,
    ):
        """
        Proximal Policy Optimization with a screen-point action head.

        total_updates is the estimated number of PPO updates expected
        across the run and is used to size the cosine LR schedule.
        """

        self.policy_net = policy_net
        self.device = policy_net.device
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl
        self.initial_lr = lr
        self.lr_min = lr_min

        if total_updates > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_updates, eta_min=lr_min,
            )
        else:
            self.scheduler = None

        self.memory = []
        self.final_next = None

    MOVE_ACTION_ID = 1

    def resolved_config(self):
        return {
            "gamma": float(self.gamma),
            "clip_epsilon": float(self.clip_epsilon),
            "critic_loss_coef": float(self.critic_loss_coef),
            "entropy_coef": float(self.entropy_coef),
            "target_kl": (
                None if self.target_kl is None else float(self.target_kl)
            ),
            "lr": float(self.initial_lr),
            "lr_min": float(self.lr_min),
            "scheduler_enabled": bool(self.scheduler is not None),
        }

    def select_action(self, observations, state=None, deterministic: bool = False):
        spatial_obs, vector_obs = observations

        if isinstance(spatial_obs, torch.Tensor):
            spatial_tensor = spatial_obs.to(self.device).unsqueeze(0)
        else:
            spatial_tensor = torch.tensor(
                spatial_obs, dtype=torch.float32, device=self.device,
            ).unsqueeze(0)

        if isinstance(vector_obs, torch.Tensor):
            vector_tensor = vector_obs.to(self.device).unsqueeze(0)
        else:
            vector_tensor = torch.tensor(
                vector_obs, dtype=torch.float32, device=self.device,
            ).unsqueeze(0)

        with torch.no_grad(), torch.amp.autocast(
            "cuda",
            dtype=self.policy_net.amp_dtype,
            enabled=self.policy_net.use_amp,
        ):
            action_logits, move_x_logits, move_y_logits, state_value, next_state = (
                self.policy_net(spatial_tensor, vector_tensor, state=state)
            )

            action_dist = torch.distributions.Categorical(
                logits=action_logits.float(),
            )
            move_x_dist = torch.distributions.Categorical(
                logits=move_x_logits.float(),
            )
            move_y_dist = torch.distributions.Categorical(
                logits=move_y_logits.float(),
            )

            if deterministic:
                action = action_logits.float().argmax(dim=-1)
                move_x = move_x_logits.float().argmax(dim=-1)
                move_y = move_y_logits.float().argmax(dim=-1)
            else:
                action = action_dist.sample()
                move_x = move_x_dist.sample()
                move_y = move_y_dist.sample()

            is_move = (action == self.MOVE_ACTION_ID).float()
            log_prob = (
                action_dist.log_prob(action)
                + is_move
                * (
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

    @staticmethod
    def _state_to_cpu(snn_state):
        syn, mem = snn_state
        return (syn.detach().to("cpu"), mem.detach().to("cpu"))

    @staticmethod
    def _state_to_device(snn_state, device):
        syn, mem = snn_state
        return (syn.to(device), mem.to(device))

    @staticmethod
    def _tensor_to_cpu(tensor_like):
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like.detach().to("cpu")
        return torch.tensor(tensor_like, dtype=torch.float32)

    def set_final_next(self, spatial_obs, vector_obs, snn_state):
        self.final_next = {
            "spatial_obs": self._tensor_to_cpu(spatial_obs),
            "vector_obs": self._tensor_to_cpu(vector_obs),
            "snn_state": self._state_to_cpu(snn_state),
        }

    def _clear_rollout_cache(self):
        self.memory = []
        self.final_next = None

    def store_transition(
        self,
        spatial_obs,
        vector_obs,
        action: torch.Tensor,
        move_x: torch.Tensor,
        move_y: torch.Tensor,
        snn_state,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
    ):
        self.memory.append(
            {
                "spatial_obs": spatial_obs,
                "vector_obs": vector_obs,
                "action": action.detach(),
                "move_x": move_x.detach(),
                "move_y": move_y.detach(),
                "snn_state": self._state_to_cpu(snn_state),
                "log_prob": log_prob.detach(),
                "reward": reward.detach(),
                "value": value.detach(),
                "done": done.detach(),
            }
        )

    def update_policy(self, batch_size: int = 64, epochs: int = 10):
        if not self.memory:
            return [], None

        spatial_list = []
        vector_list = []
        for transition in self.memory:
            s = transition["spatial_obs"]
            v = transition["vector_obs"]
            if isinstance(s, torch.Tensor):
                spatial_list.append(s.to(self.device))
            else:
                spatial_list.append(
                    torch.tensor(s, dtype=torch.float32, device=self.device),
                )
            if isinstance(v, torch.Tensor):
                vector_list.append(v.to(self.device))
            else:
                vector_list.append(
                    torch.tensor(v, dtype=torch.float32, device=self.device),
                )

        spatial_obs = torch.stack(spatial_list, dim=0)
        vector_obs = torch.stack(vector_list, dim=0)
        actions = torch.stack(
            [transition["action"].to(self.device) for transition in self.memory],
        )
        move_xs = torch.stack(
            [transition["move_x"].to(self.device) for transition in self.memory],
        )
        move_ys = torch.stack(
            [transition["move_y"].to(self.device) for transition in self.memory],
        )
        log_probs_old = torch.stack(
            [transition["log_prob"].to(self.device) for transition in self.memory],
        )
        rewards = torch.stack(
            [transition["reward"].to(self.device) for transition in self.memory],
        )
        values = torch.stack(
            [transition["value"].to(self.device) for transition in self.memory],
        )
        dones = torch.stack(
            [transition["done"].to(self.device) for transition in self.memory],
        )

        state_parts = [transition["snn_state"] for transition in self.memory]
        stacked_state = (
            torch.cat([part[0] for part in state_parts], dim=0),
            torch.cat([part[1] for part in state_parts], dim=0),
        )
        state_bytes = sum(t.numel() * t.element_size() for t in stacked_state)
        print(
            f"[PPO] stacked SNN state: {state_bytes / 1e9:.4f} GB on CPU "
            f"for {len(self.memory)} transitions",
        )

        with torch.no_grad():
            if dones[-1].item() == 1.0:
                last_next_value = torch.zeros((), device=self.device)
            else:
                if self.final_next is None:
                    raise RuntimeError(
                        "Non-terminal rollout tail is missing bootstrap data. "
                        "Call PPO.set_final_next() before update_policy().",
                    )
                last_spatial = self.final_next["spatial_obs"].to(self.device).unsqueeze(0)
                last_vector = self.final_next["vector_obs"].to(self.device).unsqueeze(0)
                last_state = self._state_to_device(
                    self.final_next["snn_state"], self.device,
                )
                _, _, _, last_next_value, _ = self.policy_net(
                    last_spatial, last_vector, state=last_state,
                )
                last_next_value = last_next_value.squeeze(-1)

        advantages = self._compute_advantages(
            rewards, values, dones, last_next_value,
        )
        returns = (advantages + values).detach()
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

        rollout_size = spatial_obs.size(0)
        losses = []
        acc_policy = []
        acc_value = []
        acc_entropy = []
        acc_kl = []
        acc_clip_frac = []
        acc_grad_norm = []
        nonfinite_grad_steps = 0
        skipped_optimizer_steps = 0
        epochs_ran = 0

        params = [
            param for param in self.policy_net.parameters() if param.requires_grad
        ]

        for _ in range(epochs):
            epoch_kls = []
            perm = torch.randperm(rollout_size, device=self.device)
            for start in range(0, rollout_size, batch_size):
                idx = perm[start : start + batch_size]

                batch_spatial = spatial_obs[idx]
                batch_vector = vector_obs[idx]
                batch_actions = actions[idx]
                batch_move_x = move_xs[idx]
                batch_move_y = move_ys[idx]
                batch_old_logp = log_probs_old[idx]
                batch_adv = advantages[idx]
                batch_returns = returns[idx]

                idx_cpu = idx.to("cpu")
                syn_s, mem_s = stacked_state
                batch_state = (
                    syn_s[idx_cpu].to(self.device),
                    mem_s[idx_cpu].to(self.device),
                )

                with torch.amp.autocast(
                    "cuda",
                    dtype=self.policy_net.amp_dtype,
                    enabled=self.policy_net.use_amp,
                ):
                    action_logits, move_x_logits, move_y_logits, state_values, _ = (
                        self.policy_net(
                            batch_spatial, batch_vector, state=batch_state,
                        )
                    )
                    state_values = state_values.squeeze(-1)
                    policy_loss, value_loss, entropy_loss, diag = (
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
                    )
                    loss = policy_loss + value_loss - entropy_loss

                losses.append(float(loss.item()))
                acc_policy.append(float(policy_loss.item()))
                acc_value.append(float(value_loss.item()))
                acc_entropy.append(float(diag["entropy_mean"].item()))
                acc_kl.append(float(diag["approx_kl"].item()))
                acc_clip_frac.append(float(diag["clip_frac"].item()))
                epoch_kls.append(float(diag["approx_kl"].item()))

                self.optimizer.zero_grad(set_to_none=True)
                self.policy_net.scaler.scale(loss).backward()
                self.policy_net.scaler.unscale_(self.optimizer)

                grads_finite = True
                for param in params:
                    grad = param.grad
                    if grad is not None and not torch.isfinite(grad).all():
                        grads_finite = False
                        break

                grad_norm_value = float("inf")
                if grads_finite:
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, 0.5)
                    grad_norm_value = float(grad_norm.item())

                if grads_finite and math.isfinite(grad_norm_value):
                    self.policy_net.scaler.step(self.optimizer)
                else:
                    nonfinite_grad_steps += 1
                    skipped_optimizer_steps += 1

                acc_grad_norm.append(grad_norm_value)
                self.policy_net.scaler.update()

            epochs_ran += 1
            if self.target_kl is not None and epoch_kls:
                if float(np.mean(epoch_kls)) > float(self.target_kl):
                    break

        with torch.no_grad():
            var_returns = returns.var(unbiased=False)
            explained_var = 1.0 - (returns - values).var(unbiased=False) / (
                var_returns + 1e-8
            )

        returns_cpu = returns.detach().to("cpu").float()
        stats = {
            "mean_policy_loss": float(np.mean(acc_policy)),
            "mean_value_loss": float(np.mean(acc_value)),
            "mean_entropy": float(np.mean(acc_entropy)),
            "mean_kl": float(np.mean(acc_kl)),
            "clip_fraction": float(np.mean(acc_clip_frac)),
            "explained_variance": float(explained_var.item()),
            "grad_norm": float(np.mean(acc_grad_norm)),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "nonfinite_grad_steps": int(nonfinite_grad_steps),
            "skipped_optimizer_steps": int(skipped_optimizer_steps),
            "transitions_in_update": int(rollout_size),
            "return_mean": float(returns_cpu.mean().item()),
            "return_std": float(returns_cpu.std(unbiased=False).item()),
            "return_p10": float(torch.quantile(returns_cpu, 0.10).item()),
            "return_p50": float(torch.quantile(returns_cpu, 0.50).item()),
            "return_p90": float(torch.quantile(returns_cpu, 0.90).item()),
            "epochs_ran": int(epochs_ran),
        }

        if self.scheduler is not None:
            self.scheduler.step()

        self._clear_rollout_cache()
        return losses, stats

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_next_value: torch.Tensor,
        gae_lambda: float = 0.95,
    ) -> torch.Tensor:
        gamma = self.gamma
        rollout_size = rewards.size(0)

        advantages = torch.zeros_like(rewards, device=self.device)
        running_advantage = torch.zeros((), device=self.device)
        next_value = last_next_value

        for t in reversed(range(rollout_size)):
            not_done = 1.0 - dones[t].float()
            delta = rewards[t] + gamma * next_value * not_done - values[t]
            running_advantage = (
                delta + gamma * gae_lambda * not_done * running_advantage
            )
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
        action_dist = torch.distributions.Categorical(
            logits=action_logits.float(),
        )
        move_x_dist = torch.distributions.Categorical(
            logits=move_x_logits.float(),
        )
        move_y_dist = torch.distributions.Categorical(
            logits=move_y_logits.float(),
        )

        is_move = (actions == self.MOVE_ACTION_ID).float()
        new_log_probs = (
            action_dist.log_prob(actions)
            + is_move
            * (
                move_x_dist.log_prob(move_x) + move_y_dist.log_prob(move_y)
            )
        )

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon,
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = (
            self.critic_loss_coef * (returns - state_values).pow(2).mean()
        )

        action_dim = action_logits.size(-1)
        screen_x = move_x_logits.size(-1)
        screen_y = move_y_logits.size(-1)
        inv_log_action = 1.0 / math.log(action_dim)
        inv_log_x = 1.0 / math.log(screen_x)
        inv_log_y = 1.0 / math.log(screen_y)
        entropy = (
            action_dist.entropy() * inv_log_action
            + is_move
            * (
                move_x_dist.entropy() * inv_log_x
                + move_y_dist.entropy() * inv_log_y
            )
        )
        entropy_loss = self.entropy_coef * entropy.mean()

        with torch.no_grad():
            approx_kl = (
                (ratio - 1.0) - (new_log_probs - old_log_probs)
            ).mean()
            clip_frac = (
                (ratio - 1.0).abs() > self.clip_epsilon
            ).float().mean()
            entropy_mean = entropy.mean()

        return policy_loss, value_loss, entropy_loss, {
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "entropy_mean": entropy_mean,
        }
