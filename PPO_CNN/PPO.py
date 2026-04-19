import math

import numpy as np
import torch
import torch.optim as optim

from PPO_CNN.policy_input import (
    MAX_ENTITY_TOKENS,
    MAX_SELECTION_TOKENS,
    PolicyInputBatch,
)


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
        tbptt_window: int | None = None,
    ):
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
        self.tbptt_window = (
            None if tbptt_window is None else max(1, int(tbptt_window))
        )

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
            "tbptt_window": (
                None if self.tbptt_window is None else int(self.tbptt_window)
            ),
        }

    def select_action(self, observations, state=None, deterministic: bool = False):
        if not isinstance(observations, PolicyInputBatch):
            raise TypeError(
                f"PPO.select_action expects PolicyInputBatch, got {type(observations)!r}",
            )
        if state is not None:
            observations = observations.with_state(state)
        batch = observations.to(device=self.device, dtype=torch.float32)

        with torch.no_grad(), torch.amp.autocast(
            "cuda",
            dtype=self.policy_net.amp_dtype,
            enabled=self.policy_net.use_amp,
        ):
            action_logits, move_x_logits, move_y_logits, state_value, next_state = (
                self.policy_net(batch)
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

    def set_final_next(self, observation_batch: PolicyInputBatch):
        self.final_next = observation_batch.detach().to(device="cpu")

    def _clear_rollout_cache(self):
        self.memory = []
        self.final_next = None

    def store_transition(
        self,
        observation_batch: PolicyInputBatch,
        action: torch.Tensor,
        move_x: torch.Tensor,
        move_y: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
        policy_mask: torch.Tensor | None = None,
    ):
        if policy_mask is None:
            policy_mask = torch.tensor(1.0, dtype=torch.float32)
        self.memory.append(
            {
                "observation_batch": observation_batch.detach().to(device="cpu"),
                "action": action.detach().to(device="cpu"),
                "move_x": move_x.detach().to(device="cpu"),
                "move_y": move_y.detach().to(device="cpu"),
                "log_prob": log_prob.detach().to(device="cpu"),
                "reward": reward.detach().to(device="cpu"),
                "value": value.detach().to(device="cpu"),
                "done": done.detach().to(device="cpu"),
                "policy_mask": policy_mask.detach().to(device="cpu"),
            }
        )

    def update_policy(self, batch_size: int = 64, epochs: int = 10):
        if not self.memory:
            return [], None

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
        policy_masks = torch.stack(
            [
                transition.get(
                    "policy_mask",
                    torch.tensor(1.0, dtype=torch.float32),
                ).to(self.device)
                for transition in self.memory
            ],
        ).float()

        with torch.no_grad():
            if dones[-1].item() == 1.0:
                last_next_value = torch.zeros((), device=self.device)
            else:
                if self.final_next is None:
                    raise RuntimeError(
                        "Non-terminal rollout tail is missing bootstrap data. "
                        "Call PPO.set_final_next() before update_policy().",
                    )
                last_batch = self.final_next.to(
                    device=self.device,
                    dtype=torch.float32,
                )
                _, _, _, last_next_value, _ = self.policy_net(last_batch)
                last_next_value = last_next_value.squeeze(-1)

        raw_advantages = self._compute_advantages(
            rewards, values, dones, last_next_value,
        )
        returns = (raw_advantages + values).detach()
        if bool((policy_masks > 0.0).any().item()):
            valid_advantages = raw_advantages[policy_masks > 0.0]
            advantages = (
                raw_advantages - valid_advantages.mean()
            ) / (valid_advantages.std(unbiased=False) + 1e-8)
        else:
            advantages = torch.zeros_like(raw_advantages, device=self.device)

        chunks = self._build_tbptt_chunks(
            actions=actions,
            move_xs=move_xs,
            move_ys=move_ys,
            log_probs_old=log_probs_old,
            advantages=advantages,
            returns=returns,
            dones=dones,
            policy_masks=policy_masks,
        )

        rollout_size = len(self.memory)
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
            for chunk_group in self._iter_chunk_groups(chunks, batch_size):
                policy_num = torch.zeros((), device=self.device)
                policy_den = torch.zeros((), device=self.device)
                value_num = torch.zeros((), device=self.device)
                value_den = torch.zeros((), device=self.device)
                entropy_num = torch.zeros((), device=self.device)
                diag_kl_num = 0.0
                diag_clip_num = 0.0
                diag_entropy_num = 0.0
                diag_policy_count = 0.0

                with torch.amp.autocast(
                    "cuda",
                    dtype=self.policy_net.amp_dtype,
                    enabled=self.policy_net.use_amp,
                ):
                    for chunk in chunk_group:
                        (
                            action_logits,
                            move_x_logits,
                            move_y_logits,
                            state_values,
                        ) = self._replay_chunk(chunk)
                        policy_loss, value_loss, entropy_loss, diag = (
                            self._calculate_losses(
                                action_logits,
                                move_x_logits,
                                move_y_logits,
                                state_values,
                                chunk["actions"],
                                chunk["move_x"],
                                chunk["move_y"],
                                chunk["old_log_prob"],
                                chunk["advantages"],
                                chunk["returns"],
                                chunk["policy_mask"],
                            )
                        )
                        policy_num = policy_num + policy_loss * diag["policy_count"]
                        policy_den = policy_den + diag["policy_count"]
                        value_num = value_num + value_loss * diag["value_count"]
                        value_den = value_den + diag["value_count"]
                        entropy_num = entropy_num + entropy_loss * diag["policy_count"]
                        policy_weight = float(diag["policy_count"].item())
                        diag_kl_num += float(diag["approx_kl"].item()) * policy_weight
                        diag_clip_num += (
                            float(diag["clip_frac"].item()) * policy_weight
                        )
                        diag_entropy_num += (
                            float(diag["entropy_mean"].item()) * policy_weight
                        )
                        diag_policy_count += policy_weight

                    policy_loss = policy_num / policy_den.clamp_min(1.0)
                    value_loss = value_num / value_den.clamp_min(1.0)
                    entropy_loss = entropy_num / policy_den.clamp_min(1.0)
                    loss = policy_loss + value_loss - entropy_loss

                approx_kl = (
                    0.0
                    if diag_policy_count <= 0.0
                    else diag_kl_num / diag_policy_count
                )
                clip_frac = (
                    0.0
                    if diag_policy_count <= 0.0
                    else diag_clip_num / diag_policy_count
                )
                entropy_mean = (
                    0.0
                    if diag_policy_count <= 0.0
                    else diag_entropy_num / diag_policy_count
                )

                losses.append(float(loss.item()))
                acc_policy.append(float(policy_loss.item()))
                acc_value.append(float(value_loss.item()))
                acc_entropy.append(float(entropy_mean))
                acc_kl.append(float(approx_kl))
                acc_clip_frac.append(float(clip_frac))
                epoch_kls.append(float(approx_kl))

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
        entity_counts = torch.tensor(
            [
                float(
                    transition["observation_batch"].entity_mask.sum().item(),
                )
                for transition in self.memory
            ],
            dtype=torch.float32,
        )
        selection_counts = torch.tensor(
            [
                float(
                    transition["observation_batch"].selection_mask.sum().item(),
                )
                for transition in self.memory
            ],
            dtype=torch.float32,
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
            "nonfinite_grad_steps": int(nonfinite_grad_steps),
            "skipped_optimizer_steps": int(skipped_optimizer_steps),
            "transitions_in_update": int(rollout_size),
            "return_mean": float(returns_cpu.mean().item()),
            "return_std": float(returns_cpu.std(unbiased=False).item()),
            "return_p10": float(torch.quantile(returns_cpu, 0.10).item()),
            "return_p50": float(torch.quantile(returns_cpu, 0.50).item()),
            "return_p90": float(torch.quantile(returns_cpu, 0.90).item()),
            "entity_mask_utilization": float(
                (entity_counts / MAX_ENTITY_TOKENS).mean().item(),
            ),
            "entity_count_p50": float(torch.quantile(entity_counts, 0.50).item()),
            "entity_count_p99": float(torch.quantile(entity_counts, 0.99).item()),
            "selection_mask_utilization": float(
                (selection_counts / MAX_SELECTION_TOKENS).mean().item(),
            ),
            "epochs_ran": int(epochs_ran),
        }

        if self.scheduler is not None:
            self.scheduler.step()

        self._clear_rollout_cache()
        return losses, stats

    def _build_tbptt_chunks(
        self,
        actions: torch.Tensor,
        move_xs: torch.Tensor,
        move_ys: torch.Tensor,
        log_probs_old: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        dones: torch.Tensor,
        policy_masks: torch.Tensor,
    ):
        chunks = []
        rollout_size = len(self.memory)
        window = rollout_size if self.tbptt_window is None else self.tbptt_window
        start = 0
        while start < rollout_size:
            end = min(start + window, rollout_size)
            done_indices = torch.nonzero(dones[start:end] > 0.5, as_tuple=False)
            if len(done_indices) > 0:
                end = start + int(done_indices[0].item()) + 1

            step_batches = [
                self.memory[idx]["observation_batch"].with_state(None)
                for idx in range(start, end)
            ]
            chunks.append(
                {
                    "observations": PolicyInputBatch.stack(step_batches),
                    "initial_state": self.memory[start][
                        "observation_batch"
                    ].state_in,
                    "actions": actions[start:end],
                    "move_x": move_xs[start:end],
                    "move_y": move_ys[start:end],
                    "old_log_prob": log_probs_old[start:end],
                    "advantages": advantages[start:end],
                    "returns": returns[start:end],
                    "dones": dones[start:end],
                    "policy_mask": policy_masks[start:end],
                    "length": int(end - start),
                }
            )
            start = end
        return chunks

    @staticmethod
    def _iter_chunk_groups(chunks, batch_size: int):
        if not chunks:
            return
        max_steps = max(1, int(batch_size))
        order = torch.randperm(len(chunks)).tolist()
        group = []
        steps_in_group = 0
        for idx in order:
            chunk = chunks[idx]
            chunk_len = int(chunk["length"])
            if group and steps_in_group + chunk_len > max_steps:
                yield group
                group = []
                steps_in_group = 0
            group.append(chunk)
            steps_in_group += chunk_len
        if group:
            yield group

    def _replay_chunk(self, chunk):
        state = chunk["initial_state"]
        if state is not None:
            state = (
                state[0].to(device=self.device, dtype=torch.float32).detach(),
                state[1].to(device=self.device, dtype=torch.float32).detach(),
            )

        action_logits_seq = []
        move_x_logits_seq = []
        move_y_logits_seq = []
        state_values_seq = []

        for t in range(int(chunk["length"])):
            step_observation = chunk["observations"].index_select([t]).to(
                device=self.device,
                dtype=torch.float32,
            ).with_state(state)
            action_logits, move_x_logits, move_y_logits, state_values, state = (
                self.policy_net(step_observation)
            )
            action_logits_seq.append(action_logits)
            move_x_logits_seq.append(move_x_logits)
            move_y_logits_seq.append(move_y_logits)
            state_values_seq.append(state_values)

            if (
                bool(chunk["dones"][t].item() > 0.5)
                and t + 1 < int(chunk["length"])
            ):
                state = self.policy_net.init_concrete_state(
                    batch_size=step_observation.batch_size,
                    device=self.device,
                    dtype=step_observation.spatial_obs.dtype,
                )

        return (
            torch.cat(action_logits_seq, dim=0),
            torch.cat(move_x_logits_seq, dim=0),
            torch.cat(move_y_logits_seq, dim=0),
            torch.cat(state_values_seq, dim=0),
        )

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
        policy_mask: torch.Tensor | None = None,
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

        if policy_mask is None:
            policy_mask = torch.ones_like(advantages)
        policy_mask = policy_mask.to(device=advantages.device, dtype=advantages.dtype)
        policy_count = policy_mask.sum()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon,
        ) * advantages
        policy_loss = -self._masked_mean(torch.min(surr1, surr2), policy_mask)

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
        entropy_loss = self.entropy_coef * self._masked_mean(entropy, policy_mask)

        with torch.no_grad():
            approx_kl = self._masked_mean(
                (ratio - 1.0) - (new_log_probs - old_log_probs),
                policy_mask,
            )
            clip_frac = self._masked_mean(
                ((ratio - 1.0).abs() > self.clip_epsilon).float(),
                policy_mask,
            )
            entropy_mean = self._masked_mean(entropy, policy_mask)

        return policy_loss, value_loss, entropy_loss, {
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "entropy_mean": entropy_mean,
            "policy_count": policy_count.detach(),
            "value_count": torch.tensor(
                float(state_values.numel()),
                device=state_values.device,
            ),
        }

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=values.device, dtype=values.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom
