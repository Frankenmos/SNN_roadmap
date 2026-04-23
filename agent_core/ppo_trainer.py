import math
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim

from agent_core.policy_protocol import (
    ACTION_REQUIRES_TARGET,
    ActionSample,
    MAX_ENTITY_TOKENS,
    MAX_SELECTION_TOKENS,
    META_AVAILABLE_ACTION_DIM,
    META_AVAILABLE_ACTION_OFFSET,
    POLICY_ACTION_LEFT_CLICK,
    POLICY_ACTION_NO_OP,
    POLICY_ACTION_RIGHT_CLICK,
    PolicyInputBatch,
    SEMANTIC_AVAILABLE_NO_OP_INDEX,
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

    NO_OP_ACTION_ID = POLICY_ACTION_NO_OP
    LEFT_CLICK_ACTION_ID = POLICY_ACTION_LEFT_CLICK
    RIGHT_CLICK_ACTION_ID = POLICY_ACTION_RIGHT_CLICK
    SPATIAL_ACTION_IDS = ACTION_REQUIRES_TARGET

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

    def _spatial_action_mask(self, actions: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(actions, dtype=torch.bool)
        for action_id in self.SPATIAL_ACTION_IDS:
            mask = mask | (actions == int(action_id))
        return mask.to(dtype=torch.float32)

    def _policy_action_availability(self, meta_vec: torch.Tensor) -> torch.Tensor:
        required_width = META_AVAILABLE_ACTION_OFFSET + META_AVAILABLE_ACTION_DIM
        if int(meta_vec.size(-1)) < required_width:
            raise ValueError(
                f"meta_vec width too small for availability slice: "
                f"got {int(meta_vec.size(-1))}, require at least {required_width}",
            )
        if torch.isnan(meta_vec).any():
            raise ValueError("meta_vec contains NaN values; protocol validation failed")
        available = meta_vec[
            ...,
            META_AVAILABLE_ACTION_OFFSET : META_AVAILABLE_ACTION_OFFSET
            + META_AVAILABLE_ACTION_DIM,
        ]
        available = available > 0.5
        available[..., SEMANTIC_AVAILABLE_NO_OP_INDEX] = True
        return available

    def _mask_action_logits(
        self,
        action_logits: torch.Tensor,
        meta_vec: torch.Tensor,
    ) -> torch.Tensor:
        available = self._policy_action_availability(meta_vec)
        if int(available.size(-1)) != int(action_logits.size(-1)):
            if int(available.size(-1)) < int(action_logits.size(-1)):
                raise ValueError(
                    "meta availability width is smaller than action_logits width: "
                    f"{int(available.size(-1))} < {int(action_logits.size(-1))}",
                )
            available = available[..., : action_logits.size(-1)]
        return action_logits.masked_fill(~available, -1.0e4)

    def _build_target_head_state(
        self,
        latent: torch.Tensor,
        spatial_context: torch.Tensor,
        action_ids: torch.Tensor,
    ):
        if hasattr(self.policy_net, "build_target_head"):
            return self.policy_net.build_target_head(latent, spatial_context, action_ids)
        move_x_logits, move_y_logits = self.policy_net.conditioned_spatial_head(
            latent,
            spatial_context,
            action_ids,
        )
        return {
            "head_type": "factorized_xy_legacy",
            "move_x_logits": move_x_logits,
            "move_y_logits": move_y_logits,
        }

    def _sample_target(
        self,
        target_head_state,
        action_ids: torch.Tensor,
        *,
        deterministic: bool = False,
    ):
        if hasattr(self.policy_net, "sample_target") and not isinstance(
            target_head_state,
            dict,
        ):
            return self.policy_net.sample_target(
                target_head_state,
                action_ids,
                deterministic=deterministic,
            )
        move_x_dist = torch.distributions.Categorical(
            logits=target_head_state["move_x_logits"].float(),
        )
        move_y_dist = torch.distributions.Categorical(
            logits=target_head_state["move_y_logits"].float(),
        )
        if deterministic:
            x = target_head_state["move_x_logits"].float().argmax(dim=-1)
            y = target_head_state["move_y_logits"].float().argmax(dim=-1)
        else:
            x = move_x_dist.sample()
            y = move_y_dist.sample()
        entropy = (
            move_x_dist.entropy() / math.log(float(target_head_state["move_x_logits"].size(-1)))
            + move_y_dist.entropy() / math.log(float(target_head_state["move_y_logits"].size(-1)))
        )
        return SimpleNamespace(
            x=x,
            y=y,
            target_index=None,
            coarse_index=None,
            fine_index=None,
            log_prob=move_x_dist.log_prob(x) + move_y_dist.log_prob(y),
            entropy=entropy,
        )

    def _evaluate_target(
        self,
        target_head_state,
        recorded_target: dict[str, torch.Tensor | None],
        action_ids: torch.Tensor,
    ):
        if hasattr(self.policy_net, "evaluate_target") and not isinstance(
            target_head_state,
            dict,
        ):
            return self.policy_net.evaluate_target(
                target_head_state,
                recorded_target,
                action_ids,
            )
        move_x_dist = torch.distributions.Categorical(
            logits=target_head_state["move_x_logits"].float(),
        )
        move_y_dist = torch.distributions.Categorical(
            logits=target_head_state["move_y_logits"].float(),
        )
        entropy = (
            move_x_dist.entropy() / math.log(float(target_head_state["move_x_logits"].size(-1)))
            + move_y_dist.entropy() / math.log(float(target_head_state["move_y_logits"].size(-1)))
        )
        return SimpleNamespace(
            log_prob=move_x_dist.log_prob(recorded_target["x"].long())
            + move_y_dist.log_prob(recorded_target["y"].long()),
            entropy=entropy,
        )

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
            latent, state_value, next_state, spatial_context = (
                self.policy_net.encode_step_tensors(
                    spatial_obs=batch.spatial_obs,
                    entity_features=batch.entity_features,
                    entity_mask=batch.entity_mask,
                    selection_features=batch.selection_features,
                    selection_mask=batch.selection_mask,
                    meta_vec=batch.meta_vec,
                    state_in=batch.state_in,
                )
            )
            action_logits = self._mask_action_logits(
                self.policy_net.action_head(latent),
                batch.meta_vec,
            )

            action_dist = torch.distributions.Categorical(
                logits=action_logits.float(),
            )

            if deterministic:
                action = action_logits.float().argmax(dim=-1)
            else:
                action = action_dist.sample()
            target_head_state = self._build_target_head_state(
                latent,
                spatial_context,
                action,
            )
            target_sample = self._sample_target(
                target_head_state,
                action,
                deterministic=deterministic,
            )

            is_spatial = self._spatial_action_mask(action)
            move_x = torch.where(
                is_spatial.bool(),
                target_sample.x,
                torch.zeros_like(target_sample.x),
            )
            move_y = torch.where(
                is_spatial.bool(),
                target_sample.y,
                torch.zeros_like(target_sample.y),
            )
            log_prob = (
                action_dist.log_prob(action)
                + is_spatial * target_sample.log_prob
            )

        target_index = (
            None
            if target_sample.target_index is None
            else int(target_sample.target_index.item())
        )
        coarse_index = (
            None
            if target_sample.coarse_index is None
            else int(target_sample.coarse_index.item())
        )
        fine_index = (
            None
            if target_sample.fine_index is None
            else int(target_sample.fine_index.item())
        )
        return ActionSample(
            action_id=int(action.item()),
            x=int(move_x.item()),
            y=int(move_y.item()),
            target_index=target_index,
            coarse_index=coarse_index,
            fine_index=fine_index,
            log_prob=float(log_prob.item()),
            value=float(state_value.squeeze(-1).item()),
            next_state=next_state,
        )

    def set_final_next(self, observation_batch: PolicyInputBatch):
        if observation_batch.state_in is None:
            raise ValueError(
                "Bootstrap observation must carry the recurrent state after the final rollout step.",
            )
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
        sample_mask: torch.Tensor | None = None,
        policy_mask: torch.Tensor | None = None,
        *,
        target_index: torch.Tensor | None = None,
        coarse_index: torch.Tensor | None = None,
        fine_index: torch.Tensor | None = None,
    ):
        if observation_batch.state_in is None:
            raise ValueError(
                "Stored transition must carry the pre-step recurrent state.",
            )
        if sample_mask is None:
            if policy_mask is None:
                sample_mask = torch.tensor(1.0, dtype=torch.float32)
            else:
                sample_mask = policy_mask
        sample_mask = sample_mask.to(device="cpu").detach()
        default_target = torch.tensor(-1, dtype=torch.long)
        self.memory.append(
            {
                "observation_batch": observation_batch.detach().to(device="cpu"),
                "action": action.detach().to(device="cpu"),
                "move_x": move_x.detach().to(device="cpu"),
                "move_y": move_y.detach().to(device="cpu"),
                "target_index": (
                    default_target
                    if target_index is None
                    else target_index.detach().to(device="cpu", dtype=torch.long)
                ),
                "coarse_index": (
                    default_target
                    if coarse_index is None
                    else coarse_index.detach().to(device="cpu", dtype=torch.long)
                ),
                "fine_index": (
                    default_target
                    if fine_index is None
                    else fine_index.detach().to(device="cpu", dtype=torch.long)
                ),
                "log_prob": log_prob.detach().to(device="cpu"),
                "reward": reward.detach().to(device="cpu"),
                "value": value.detach().to(device="cpu"),
                "done": done.detach().to(device="cpu"),
                "sample_mask": sample_mask,
                "policy_mask": sample_mask,
            }
        )

    def update_policy(self, batch_size: int = 64, epochs: int = 10):
        if not self.memory:
            return [], None
        update_started = time.perf_counter()

        actions = torch.stack(
            [transition["action"].to(self.device) for transition in self.memory],
        )
        move_xs = torch.stack(
            [transition["move_x"].to(self.device) for transition in self.memory],
        )
        move_ys = torch.stack(
            [transition["move_y"].to(self.device) for transition in self.memory],
        )
        target_indices = torch.stack(
            [transition["target_index"].to(self.device) for transition in self.memory],
        )
        coarse_indices = torch.stack(
            [transition["coarse_index"].to(self.device) for transition in self.memory],
        )
        fine_indices = torch.stack(
            [transition["fine_index"].to(self.device) for transition in self.memory],
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
        sample_masks = torch.stack(
            [
                transition.get(
                    "sample_mask",
                    transition.get(
                        "policy_mask",
                        torch.tensor(1.0, dtype=torch.float32, device=self.device),
                    ),
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
                _, _, last_next_value, _ = self.policy_net(last_batch)
                last_next_value = last_next_value.squeeze(-1)

        raw_advantages = self._compute_advantages(
            rewards, values, dones, last_next_value,
        )
        returns = (raw_advantages + values).detach()
        if bool((sample_masks > 0.0).any().item()):
            valid_advantages = raw_advantages[sample_masks > 0.0]
            advantages = (
                raw_advantages - valid_advantages.mean()
            ) / (valid_advantages.std(unbiased=False) + 1e-8)
        else:
            advantages = torch.zeros_like(raw_advantages, device=self.device)

        chunks = self._build_tbptt_chunks(
            actions=actions,
            move_xs=move_xs,
            move_ys=move_ys,
            target_indices=target_indices,
            coarse_indices=coarse_indices,
            fine_indices=fine_indices,
            log_probs_old=log_probs_old,
            advantages=advantages,
            returns=returns,
            dones=dones,
            sample_masks=sample_masks,
        )
        tbptt_chunks = int(len(chunks))

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
        tbptt_chunk_groups = 0
        tbptt_group_max_steps = 0
        tbptt_forward_calls = 0
        active_chunk_sum = 0.0
        active_chunk_steps = 0

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
                tbptt_chunk_groups += 1

                with torch.amp.autocast(
                    "cuda",
                    dtype=self.policy_net.amp_dtype,
                    enabled=self.policy_net.use_amp,
                ):
                    replayed_group = self._replay_packed_chunk_group(
                        self._pack_chunk_group(chunk_group),
                    )
                    tbptt_group_max_steps = max(
                        tbptt_group_max_steps,
                        int(replayed_group["max_steps"]),
                    )
                    tbptt_forward_calls += int(replayed_group["forward_calls"])
                    active_chunk_sum += float(replayed_group["active_chunks_sum"])
                    active_chunk_steps += int(replayed_group["active_steps"])

                    active_mask = replayed_group["alive_mask"].reshape(-1)
                    action_logits = replayed_group["action_logits"].reshape(
                        -1,
                        replayed_group["action_logits"].size(-1),
                    )[active_mask]
                    target_log_prob = replayed_group["target_log_prob"].reshape(-1)[
                        active_mask
                    ]
                    target_entropy = replayed_group["target_entropy"].reshape(-1)[
                        active_mask
                    ]
                    state_values = replayed_group["state_values"].reshape(-1)[
                        active_mask
                    ]
                    policy_loss, value_loss, entropy_loss, diag = (
                        self._calculate_losses(
                            action_logits,
                            target_log_prob,
                            target_entropy,
                            state_values,
                            replayed_group["actions"].reshape(-1)[active_mask],
                            replayed_group["old_log_prob"].reshape(-1)[active_mask],
                            replayed_group["advantages"].reshape(-1)[active_mask],
                            replayed_group["returns"].reshape(-1)[active_mask],
                             replayed_group["sample_mask"].reshape(-1)[active_mask],
                        )
                    )
                    policy_num = policy_num + policy_loss * diag["sample_count"]
                    policy_den = policy_den + diag["sample_count"]
                    value_num = value_num + value_loss * diag["sample_count"]
                    value_den = value_den + diag["sample_count"]
                    entropy_num = entropy_num + entropy_loss * diag["sample_count"]
                    policy_weight = float(diag["sample_count"].item())
                    diag_kl_num += float(diag["approx_kl"].item()) * policy_weight
                    diag_clip_num += float(diag["clip_frac"].item()) * policy_weight
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
        update_wall_seconds = time.perf_counter() - update_started
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
            "update_wall_seconds": float(update_wall_seconds),
            "tbptt_chunks": int(tbptt_chunks),
            "tbptt_chunk_groups": int(tbptt_chunk_groups),
            "tbptt_window": (
                None if self.tbptt_window is None else int(self.tbptt_window)
            ),
            "tbptt_group_max_steps": int(tbptt_group_max_steps),
            "tbptt_group_mean_active_chunks": float(
                0.0
                if active_chunk_steps <= 0
                else active_chunk_sum / float(active_chunk_steps)
            ),
            "tbptt_forward_calls": int(tbptt_forward_calls),
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
        sample_masks: torch.Tensor | None = None,
        policy_masks: torch.Tensor | None = None,
        target_indices: torch.Tensor | None = None,
        coarse_indices: torch.Tensor | None = None,
        fine_indices: torch.Tensor | None = None,
    ):
        if sample_masks is None:
            if policy_masks is not None:
                sample_masks = policy_masks
            else:
                sample_masks = torch.ones(
                    len(self.memory),
                    dtype=torch.float32,
                    device=self.device,
                )
        if target_indices is None:
            target_indices = torch.full_like(actions, -1, dtype=torch.long)
        if coarse_indices is None:
            coarse_indices = torch.full_like(actions, -1, dtype=torch.long)
        if fine_indices is None:
            fine_indices = torch.full_like(actions, -1, dtype=torch.long)
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
                    "target_index": target_indices[start:end],
                    "coarse_index": coarse_indices[start:end],
                    "fine_index": fine_indices[start:end],
                    "old_log_prob": log_probs_old[start:end],
                    "advantages": advantages[start:end],
                    "returns": returns[start:end],
                    "dones": dones[start:end],
                    "sample_mask": sample_masks[start:end],
                    "policy_mask": sample_masks[start:end],
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

    def _forward_replay_step_tensors(
        self,
        spatial_obs: torch.Tensor,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
        selection_features: torch.Tensor,
        selection_mask: torch.Tensor,
        meta_vec: torch.Tensor,
        action_ids: torch.Tensor | None,
        move_x: torch.Tensor,
        move_y: torch.Tensor,
        target_index: torch.Tensor,
        coarse_index: torch.Tensor,
        fine_index: torch.Tensor,
        state_in: tuple[torch.Tensor, torch.Tensor] | None,
    ):
        latent, state_value, next_state, spatial_context = (
            self.policy_net.encode_step_tensors(
                spatial_obs=spatial_obs,
                entity_features=entity_features,
                entity_mask=entity_mask,
                selection_features=selection_features,
                selection_mask=selection_mask,
                meta_vec=meta_vec,
                state_in=state_in,
            )
        )
        action_logits = self._mask_action_logits(
            self.policy_net.action_head(latent),
            meta_vec,
        )
        target_head_state = self._build_target_head_state(
            latent,
            spatial_context,
            action_ids,
        )
        target_eval = self._evaluate_target(
            target_head_state,
            {
                "x": move_x.long(),
                "y": move_y.long(),
                "target_index": target_index.long(),
                "coarse_index": coarse_index.long(),
                "fine_index": fine_index.long(),
            },
            action_ids,
        )
        return (
            action_logits,
            target_eval.log_prob,
            target_eval.entropy,
            state_value,
            next_state,
        )

    def _reset_replay_state_rows(
        self,
        state: tuple[torch.Tensor, torch.Tensor] | None,
        reset_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if state is None or reset_mask is None:
            return state
        if hasattr(self.policy_net, "reset_state_rows"):
            return self.policy_net.reset_state_rows(state, reset_mask)
        if not bool(reset_mask.any().item()):
            return state
        keep_mask = (~reset_mask).to(
            device=state[0].device,
            dtype=state[0].dtype,
        ).view(-1, 1, 1)
        return state[0] * keep_mask, state[1] * keep_mask

    def _pack_chunk_group(self, chunk_group):
        max_len = max(int(chunk["length"]) for chunk in chunk_group)
        group_size = len(chunk_group)
        sample_obs = chunk_group[0]["observations"]

        spatial_obs = torch.zeros(
            (max_len, group_size, *sample_obs.spatial_obs.shape[1:]),
            device=self.device,
            dtype=torch.float32,
        )
        entity_features = torch.zeros(
            (max_len, group_size, *sample_obs.entity_features.shape[1:]),
            device=self.device,
            dtype=torch.float32,
        )
        entity_mask = torch.zeros(
            (max_len, group_size, sample_obs.entity_mask.shape[-1]),
            device=self.device,
            dtype=torch.bool,
        )
        selection_features = torch.zeros(
            (max_len, group_size, *sample_obs.selection_features.shape[1:]),
            device=self.device,
            dtype=torch.float32,
        )
        selection_mask = torch.zeros(
            (max_len, group_size, sample_obs.selection_mask.shape[-1]),
            device=self.device,
            dtype=torch.bool,
        )
        meta_vec = torch.zeros(
            (max_len, group_size, sample_obs.meta_vec.shape[-1]),
            device=self.device,
            dtype=torch.float32,
        )
        actions = torch.zeros(
            (max_len, group_size),
            device=self.device,
            dtype=chunk_group[0]["actions"].dtype,
        )
        move_x = torch.zeros(
            (max_len, group_size),
            device=self.device,
            dtype=chunk_group[0]["move_x"].dtype,
        )
        move_y = torch.zeros(
            (max_len, group_size),
            device=self.device,
            dtype=chunk_group[0]["move_y"].dtype,
        )
        target_index = torch.full(
            (max_len, group_size),
            -1,
            device=self.device,
            dtype=chunk_group[0]["target_index"].dtype,
        )
        coarse_index = torch.full(
            (max_len, group_size),
            -1,
            device=self.device,
            dtype=chunk_group[0]["coarse_index"].dtype,
        )
        fine_index = torch.full(
            (max_len, group_size),
            -1,
            device=self.device,
            dtype=chunk_group[0]["fine_index"].dtype,
        )
        old_log_prob = torch.zeros(
            (max_len, group_size),
            device=self.device,
            dtype=chunk_group[0]["old_log_prob"].dtype,
        )
        advantages = torch.zeros(
            (max_len, group_size),
            device=self.device,
            dtype=chunk_group[0]["advantages"].dtype,
        )
        returns = torch.zeros(
            (max_len, group_size),
            device=self.device,
            dtype=chunk_group[0]["returns"].dtype,
        )
        sample_mask = torch.zeros(
            (max_len, group_size),
            device=self.device,
            dtype=chunk_group[0]["sample_mask"].dtype,
        )
        policy_mask = torch.zeros(
            (max_len, group_size),
            device=self.device,
            dtype=chunk_group[0]["sample_mask"].dtype,
        )
        done = torch.zeros(
            (max_len, group_size),
            device=self.device,
            dtype=torch.bool,
        )
        alive_mask = torch.zeros(
            (max_len, group_size),
            device=self.device,
            dtype=torch.bool,
        )
        lengths = []

        states = [chunk["initial_state"] for chunk in chunk_group]
        if all(state is None for state in states):
            initial_state = None
        elif any(state is None for state in states):
            raise ValueError(
                "Either every chunk must have an initial_state, or none may",
            )
        else:
            initial_state = (
                torch.cat([state[0] for state in states], dim=0).to(
                    device=self.device,
                    dtype=torch.float32,
                ).detach(),
                torch.cat([state[1] for state in states], dim=0).to(
                    device=self.device,
                    dtype=torch.float32,
                ).detach(),
            )

        for column, chunk in enumerate(chunk_group):
            length = int(chunk["length"])
            lengths.append(length)
            obs = chunk["observations"].to(device=self.device, dtype=torch.float32)
            spatial_obs[:length, column] = obs.spatial_obs
            entity_features[:length, column] = obs.entity_features
            entity_mask[:length, column] = obs.entity_mask
            selection_features[:length, column] = obs.selection_features
            selection_mask[:length, column] = obs.selection_mask
            meta_vec[:length, column] = obs.meta_vec
            actions[:length, column] = chunk["actions"]
            move_x[:length, column] = chunk["move_x"]
            move_y[:length, column] = chunk["move_y"]
            target_index[:length, column] = chunk["target_index"]
            coarse_index[:length, column] = chunk["coarse_index"]
            fine_index[:length, column] = chunk["fine_index"]
            old_log_prob[:length, column] = chunk["old_log_prob"]
            advantages[:length, column] = chunk["advantages"]
            returns[:length, column] = chunk["returns"]
            chunk_mask = chunk.get(
                "sample_mask",
                chunk.get(
                    "policy_mask",
                    torch.ones((length,), device=self.device, dtype=sample_mask.dtype),
                ),
            )
            sample_mask[:length, column] = chunk_mask
            policy_mask[:length, column] = chunk_mask
            done[:length, column] = chunk["dones"] > 0.5
            alive_mask[:length, column] = True

        return {
            "spatial_obs": spatial_obs,
            "entity_features": entity_features,
            "entity_mask": entity_mask,
            "selection_features": selection_features,
            "selection_mask": selection_mask,
            "meta_vec": meta_vec,
            "actions": actions,
            "move_x": move_x,
            "move_y": move_y,
            "target_index": target_index,
            "coarse_index": coarse_index,
            "fine_index": fine_index,
            "old_log_prob": old_log_prob,
            "advantages": advantages,
            "returns": returns,
            "sample_mask": sample_mask,
            "policy_mask": policy_mask,
            "done": done,
            "alive_mask": alive_mask,
            "initial_state": initial_state,
            "lengths": lengths,
            "max_steps": int(max_len),
        }

    def _replay_packed_chunk_group(self, packed_group):
        alive_mask = packed_group["alive_mask"]
        max_len = int(packed_group["max_steps"])
        state = packed_group["initial_state"]
        action_logits_buf = None
        target_log_prob_buf = None
        target_entropy_buf = None
        state_values_buf = None
        forward_calls = 0
        active_chunks_sum = 0.0
        active_steps = 0

        for step_index in range(max_len):
            active_indices = torch.nonzero(
                alive_mask[step_index],
                as_tuple=False,
            ).flatten()
            if int(active_indices.numel()) == 0:
                continue

            if state is not None and int(state[0].size(0)) != int(active_indices.numel()):
                raise ValueError(
                    "Packed replay state rows must match the active chunk count",
                )

            action_logits, target_log_prob, target_entropy, state_values, next_state = (
                self._forward_replay_step_tensors(
                    spatial_obs=packed_group["spatial_obs"][step_index].index_select(
                        0,
                        active_indices,
                    ),
                    entity_features=packed_group["entity_features"][
                        step_index
                    ].index_select(0, active_indices),
                    entity_mask=packed_group["entity_mask"][step_index].index_select(
                        0,
                        active_indices,
                    ),
                    selection_features=packed_group["selection_features"][
                        step_index
                    ].index_select(0, active_indices),
                    selection_mask=packed_group["selection_mask"][
                        step_index
                    ].index_select(0, active_indices),
                    meta_vec=packed_group["meta_vec"][step_index].index_select(
                        0,
                        active_indices,
                    ),
                    action_ids=packed_group["actions"][step_index].index_select(
                        0,
                        active_indices,
                    ),
                    move_x=packed_group["move_x"][step_index].index_select(
                        0,
                        active_indices,
                    ),
                    move_y=packed_group["move_y"][step_index].index_select(
                        0,
                        active_indices,
                    ),
                    target_index=packed_group["target_index"][step_index].index_select(
                        0,
                        active_indices,
                    ),
                    coarse_index=packed_group["coarse_index"][step_index].index_select(
                        0,
                        active_indices,
                    ),
                    fine_index=packed_group["fine_index"][step_index].index_select(
                        0,
                        active_indices,
                    ),
                    state_in=state,
                )
            )

            if action_logits_buf is None:
                group_size = alive_mask.size(1)
                action_logits_buf = torch.zeros(
                    max_len,
                    group_size,
                    action_logits.size(-1),
                    device=self.device,
                    dtype=action_logits.dtype,
                )
                target_log_prob_buf = torch.zeros(
                    max_len,
                    group_size,
                    device=self.device,
                    dtype=target_log_prob.dtype,
                )
                target_entropy_buf = torch.zeros(
                    max_len,
                    group_size,
                    device=self.device,
                    dtype=target_entropy.dtype,
                )
                state_values_buf = torch.zeros(
                    max_len,
                    group_size,
                    device=self.device,
                    dtype=state_values.dtype,
                )

            action_logits_buf[step_index].index_copy_(
                0,
                active_indices,
                action_logits,
            )
            target_log_prob_buf[step_index].index_copy_(
                0,
                active_indices,
                target_log_prob,
            )
            target_entropy_buf[step_index].index_copy_(
                0,
                active_indices,
                target_entropy,
            )
            state_values_buf[step_index].index_copy_(
                0,
                active_indices,
                state_values,
            )

            forward_calls += 1
            active_chunks_sum += float(active_indices.numel())
            active_steps += 1

            if next_state is None:
                state = None
            else:
                done_active = packed_group["done"][step_index].index_select(
                    0,
                    active_indices,
                )
                state = self._reset_replay_state_rows(next_state, done_active)

            if step_index + 1 < max_len and state is not None:
                keep_mask = alive_mask[step_index + 1].index_select(
                    0,
                    active_indices,
                )
                if bool(keep_mask.any().item()):
                    state = (
                        state[0][keep_mask],
                        state[1][keep_mask],
                    )
                else:
                    state = None

        if action_logits_buf is None:
            raise RuntimeError("Packed replay produced no active timesteps")

        return {
            **packed_group,
            "action_logits": action_logits_buf,
            "target_log_prob": target_log_prob_buf,
            "target_entropy": target_entropy_buf,
            "state_values": state_values_buf,
            "forward_calls": int(forward_calls),
            "active_chunks_sum": float(active_chunks_sum),
            "active_steps": int(active_steps),
        }

    def _replay_chunk_group_reference(self, chunk_group):
        prepared_chunks = []
        for chunk in chunk_group:
            state = chunk["initial_state"]
            if state is not None:
                state = (
                    state[0].to(device=self.device, dtype=torch.float32).detach(),
                    state[1].to(device=self.device, dtype=torch.float32).detach(),
                )
            prepared_chunks.append(
                {
                    "observations": chunk["observations"].to(
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    "actions": chunk["actions"].to(
                        device=self.device,
                        dtype=torch.long,
                    ),
                    "move_x": chunk["move_x"].to(
                        device=self.device,
                        dtype=torch.long,
                    ),
                    "move_y": chunk["move_y"].to(
                        device=self.device,
                        dtype=torch.long,
                    ),
                    "target_index": chunk["target_index"].to(
                        device=self.device,
                        dtype=torch.long,
                    ),
                    "coarse_index": chunk["coarse_index"].to(
                        device=self.device,
                        dtype=torch.long,
                    ),
                    "fine_index": chunk["fine_index"].to(
                        device=self.device,
                        dtype=torch.long,
                    ),
                    "state": state,
                    "dones": chunk["dones"],
                    "length": int(chunk["length"]),
                    "action_logits": [],
                    "target_log_prob": [],
                    "target_entropy": [],
                    "state_values": [],
                }
            )

        max_len = max(chunk["length"] for chunk in prepared_chunks)
        for t in range(max_len):
            active_indices = [
                idx
                for idx, chunk in enumerate(prepared_chunks)
                if t < chunk["length"]
            ]
            if not active_indices:
                continue

            step_batch = self._stack_active_step(
                [prepared_chunks[idx] for idx in active_indices],
                t,
            )
            action_ids = torch.stack(
                [prepared_chunks[idx]["actions"][t] for idx in active_indices],
                dim=0,
            )
            action_logits, target_log_prob, target_entropy, state_values, next_state = (
                self._forward_replay_step_tensors(
                    spatial_obs=step_batch.spatial_obs,
                    entity_features=step_batch.entity_features,
                    entity_mask=step_batch.entity_mask,
                    selection_features=step_batch.selection_features,
                    selection_mask=step_batch.selection_mask,
                    meta_vec=step_batch.meta_vec,
                    action_ids=action_ids,
                    move_x=torch.stack(
                        [prepared_chunks[idx]["move_x"][t] for idx in active_indices],
                        dim=0,
                    ),
                    move_y=torch.stack(
                        [prepared_chunks[idx]["move_y"][t] for idx in active_indices],
                        dim=0,
                    ),
                    target_index=torch.stack(
                        [
                            prepared_chunks[idx]["target_index"][t]
                            for idx in active_indices
                        ],
                        dim=0,
                    ),
                    coarse_index=torch.stack(
                        [
                            prepared_chunks[idx]["coarse_index"][t]
                            for idx in active_indices
                        ],
                        dim=0,
                    ),
                    fine_index=torch.stack(
                        [
                            prepared_chunks[idx]["fine_index"][t]
                            for idx in active_indices
                        ],
                        dim=0,
                    ),
                    state_in=step_batch.state_in,
                )
            )

            for offset, chunk_idx in enumerate(active_indices):
                chunk = prepared_chunks[chunk_idx]
                chunk["action_logits"].append(action_logits[offset : offset + 1])
                chunk["target_log_prob"].append(target_log_prob[offset : offset + 1])
                chunk["target_entropy"].append(target_entropy[offset : offset + 1])
                chunk["state_values"].append(state_values[offset : offset + 1])

                if next_state is None:
                    chunk["state"] = None
                else:
                    chunk["state"] = (
                        next_state[0][offset : offset + 1],
                        next_state[1][offset : offset + 1],
                    )

                if (
                    bool(chunk["dones"][t].item() > 0.5)
                    and t + 1 < chunk["length"]
                ):
                    chunk["state"] = self.policy_net.init_concrete_state(
                        batch_size=1,
                        device=self.device,
                        dtype=step_batch.spatial_obs.dtype,
                    )

        replayed = []
        for chunk in prepared_chunks:
            replayed.append(
                (
                    torch.cat(chunk["action_logits"], dim=0),
                    torch.cat(chunk["target_log_prob"], dim=0),
                    torch.cat(chunk["target_entropy"], dim=0),
                    torch.cat(chunk["state_values"], dim=0),
                )
            )
        return replayed

    def _stack_active_step(self, active_chunks, step_index: int) -> PolicyInputBatch:
        observations = [chunk["observations"] for chunk in active_chunks]
        states = [chunk["state"] for chunk in active_chunks]
        if all(state is None for state in states):
            state_in = None
        elif any(state is None for state in states):
            raise ValueError(
                "Mixed recurrent state presence inside TBPTT replay group",
            )
        else:
            state_in = (
                torch.cat([state[0] for state in states], dim=0),
                torch.cat([state[1] for state in states], dim=0),
            )

        return PolicyInputBatch(
            spatial_obs=torch.cat(
                [
                    obs.spatial_obs[step_index : step_index + 1]
                    for obs in observations
                ],
                dim=0,
            ),
            entity_features=torch.cat(
                [
                    obs.entity_features[step_index : step_index + 1]
                    for obs in observations
                ],
                dim=0,
            ),
            entity_mask=torch.cat(
                [
                    obs.entity_mask[step_index : step_index + 1]
                    for obs in observations
                ],
                dim=0,
            ),
            selection_features=torch.cat(
                [
                    obs.selection_features[step_index : step_index + 1]
                    for obs in observations
                ],
                dim=0,
            ),
            selection_mask=torch.cat(
                [
                    obs.selection_mask[step_index : step_index + 1]
                    for obs in observations
                ],
                dim=0,
            ),
            meta_vec=torch.cat(
                [
                    obs.meta_vec[step_index : step_index + 1]
                    for obs in observations
                ],
                dim=0,
            ),
            state_in=state_in,
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
        target_log_probs: torch.Tensor,
        target_entropy: torch.Tensor,
        state_values: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        policy_mask: torch.Tensor | None = None,
    ):
        action_dist = torch.distributions.Categorical(
            logits=action_logits.float(),
        )

        is_spatial = self._spatial_action_mask(actions)
        new_log_probs = (
            action_dist.log_prob(actions)
            + is_spatial * target_log_probs
        )

        if sample_mask is None:
            if policy_mask is None:
                sample_mask = torch.ones_like(advantages)
            else:
                sample_mask = policy_mask
        sample_mask = sample_mask.to(
            device=advantages.device,
            dtype=advantages.dtype,
        )
        sample_count = sample_mask.sum()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon,
        ) * advantages
        policy_loss = -self._masked_mean(torch.min(surr1, surr2), sample_mask)

        value_loss = self.critic_loss_coef * self._masked_mean(
            (returns - state_values).pow(2),
            sample_mask,
        )

        action_dim = action_logits.size(-1)
        inv_log_action = 1.0 / math.log(action_dim)
        entropy = (
            action_dist.entropy() * inv_log_action
            + is_spatial * target_entropy
        )
        entropy_loss = self.entropy_coef * self._masked_mean(entropy, sample_mask)

        with torch.no_grad():
            approx_kl = self._masked_mean(
                (ratio - 1.0) - (new_log_probs - old_log_probs),
                sample_mask,
            )
            clip_frac = self._masked_mean(
                ((ratio - 1.0).abs() > self.clip_epsilon).float(),
                sample_mask,
            )
            entropy_mean = self._masked_mean(entropy, sample_mask)

        return policy_loss, value_loss, entropy_loss, {
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "entropy_mean": entropy_mean,
            "sample_count": sample_count.detach(),
            "value_count": sample_count.detach(),
        }

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=values.device, dtype=values.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom
