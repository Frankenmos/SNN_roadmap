import math
import inspect
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim

from agent_core.policy_protocol import (
    ACTION_FEEDBACK_BRIDGE_TYPE_OFFSET,
    ACTION_FEEDBACK_ENEMY_HEALTH_DROP_OFFSET,
    ACTION_FEEDBACK_EXECUTED_SMART_OFFSET,
    ACTION_FEEDBACK_MOVED_TOWARD_TARGET_OFFSET,
    ACTION_FEEDBACK_TARGET_NEAR_ENEMY_OFFSET,
    ACTION_REQUIRES_TARGET,
    ActionSample,
    BRIDGE_ACTION_RIGHT_CLICK,
    MAX_ENTITY_TOKENS,
    MAX_SELECTION_TOKENS,
    META_AVAILABLE_ACTION_DIM,
    META_AVAILABLE_ACTION_OFFSET,
    POLICY_ACTION_LEFT_CLICK,
    POLICY_ACTION_NO_OP,
    POLICY_ACTION_RIGHT_CLICK,
    PolicyInputBatch,
    SEMANTIC_AVAILABLE_NO_OP_INDEX,
    SEMANTIC_AVAILABLE_RIGHT_CLICK_INDEX,
)
from distributed.protocol import RolloutFragment


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
        rollout_cache_spatial_dtype: str | torch.dtype = "float32",
        right_click_curriculum_updates: int = 0,
        right_click_curriculum_noop_logit_penalty: float = 0.0,
        sil_enabled: bool = False,
        sil_buffer_size: int = 5000,
        sil_batch_fraction: float = 0.25,
        sil_coef: float = 0.0,
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
        self.right_click_curriculum_updates = max(
            0,
            int(right_click_curriculum_updates or 0),
        )
        self.right_click_curriculum_noop_logit_penalty = max(
            0.0,
            float(right_click_curriculum_noop_logit_penalty or 0.0),
        )
        self.sil_enabled = bool(sil_enabled)
        self.sil_buffer_size = max(1, int(sil_buffer_size or 5000))
        self.sil_batch_fraction = max(0.0, float(sil_batch_fraction or 0.0))
        self.sil_coef = float(sil_coef or 0.0)
        self.sil_buffer: deque = deque(maxlen=self.sil_buffer_size)
        self.rollout_cache_spatial_dtype = self._resolve_spatial_cache_dtype(
            rollout_cache_spatial_dtype,
        )

        if total_updates > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_updates, eta_min=lr_min,
            )
        else:
            self.scheduler = None

        self.memory = []
        self.final_next = None
        self.pending_fragments = []
        self._next_fragment_id = 0
        self.update_count = 0
        self._policy_accepts_action_feedback = (
            "action_feedback_tokens"
            in inspect.signature(policy_net.encode_step_tensors).parameters
        )

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
            "rollout_cache_spatial_dtype": str(self.rollout_cache_spatial_dtype).replace(
                "torch.",
                "",
            ),
            "right_click_curriculum_updates": int(
                self.right_click_curriculum_updates,
            ),
            "right_click_curriculum_noop_logit_penalty": float(
                self.right_click_curriculum_noop_logit_penalty,
            ),
            "sil_enabled": bool(self.sil_enabled),
            "sil_buffer_size": int(self.sil_buffer_size),
            "sil_batch_fraction": float(self.sil_batch_fraction),
            "sil_coef": float(self.sil_coef),
        }

    @staticmethod
    def _resolve_spatial_cache_dtype(value: str | torch.dtype) -> torch.dtype:
        if isinstance(value, torch.dtype):
            if value in (torch.float16, torch.float32):
                return value
            raise ValueError(
                "rollout_cache_spatial_dtype must be float32 or float16, "
                f"got {value}",
            )
        normalized = str(value).strip().lower().replace("torch.", "")
        aliases = {
            "float": torch.float32,
            "float32": torch.float32,
            "fp32": torch.float32,
            "single": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
        }
        if normalized not in aliases:
            raise ValueError(
                "rollout_cache_spatial_dtype must be float32 or float16, "
                f"got {value!r}",
            )
        return aliases[normalized]

    def _device_spatial_cache_dtype(self) -> torch.dtype:
        if torch.device(self.device).type == "cuda":
            return self.rollout_cache_spatial_dtype
        return torch.float32

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
        if not isinstance(tensor, torch.Tensor):
            return 0
        return int(tensor.numel() * tensor.element_size())

    @classmethod
    def _state_nbytes(cls, state) -> int:
        if state is None:
            return 0
        return sum(cls._tensor_nbytes(part) for part in state)

    @classmethod
    def _policy_input_nbytes(cls, batch: PolicyInputBatch | None) -> int:
        if batch is None:
            return 0
        total = (
            cls._tensor_nbytes(batch.spatial_obs)
            + cls._tensor_nbytes(batch.entity_features)
            + cls._tensor_nbytes(batch.entity_mask)
            + cls._tensor_nbytes(batch.selection_features)
            + cls._tensor_nbytes(batch.selection_mask)
            + cls._tensor_nbytes(batch.action_feedback_tokens)
            + cls._tensor_nbytes(batch.meta_vec)
        )
        return int(total + cls._state_nbytes(batch.state_in))

    @classmethod
    def _fragment_payload_stats(cls, fragments: list[RolloutFragment]) -> dict:
        spatial_bytes = 0
        state_bytes = 0
        total_bytes = 0
        for fragment in fragments:
            spatial_bytes += cls._tensor_nbytes(fragment.spatial_obs)
            state_bytes += cls._state_nbytes(fragment.pre_step_snn_state)
            state_bytes += cls._state_nbytes(
                None
                if fragment.tail_next_policy_input is None
                else fragment.tail_next_policy_input.state_in
            )
            tensor_fields = (
                "spatial_obs",
                "entity_features",
                "entity_mask",
                "selection_features",
                "selection_mask",
                "action_feedback_tokens",
                "meta_vec",
                "actions",
                "move_x",
                "move_y",
                "target_index",
                "coarse_index",
                "fine_index",
                "old_log_probs",
                "values",
                "rewards",
                "dones",
                "truncateds",
                "episode_reset_mask",
                "sample_mask",
            )
            total_bytes += sum(
                cls._tensor_nbytes(getattr(fragment, name))
                for name in tensor_fields
            )
            total_bytes += cls._state_nbytes(fragment.pre_step_snn_state)
            total_bytes += cls._policy_input_nbytes(fragment.tail_next_policy_input)

        return {
            "payload_spatial_bytes": int(spatial_bytes),
            "payload_state_bytes": int(state_bytes),
            "payload_total_bytes": int(total_bytes),
            "payload_total_mib": float(total_bytes / (1024**2)),
        }

    @staticmethod
    def _fragment_step_counters(
        *,
        actions: torch.Tensor,
        action_feedback_tokens: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> dict[str, int]:
        actions = actions.detach().cpu().long().reshape(-1)
        sample_mask = sample_mask.detach().cpu().float().reshape(-1) > 0.0
        feedback = action_feedback_tokens.detach().cpu().float()
        feedback = feedback.reshape(feedback.shape[0], -1)
        smart_feedback = (
            feedback[:, ACTION_FEEDBACK_BRIDGE_TYPE_OFFSET].round().long()
            == BRIDGE_ACTION_RIGHT_CLICK
        )
        smart_executed = feedback[:, ACTION_FEEDBACK_EXECUTED_SMART_OFFSET] > 0.5
        near_enemy = feedback[:, ACTION_FEEDBACK_TARGET_NEAR_ENEMY_OFFSET] > 0.5
        moved = feedback[:, ACTION_FEEDBACK_MOVED_TOWARD_TARGET_OFFSET] > 0.5
        enemy_drop = feedback[:, ACTION_FEEDBACK_ENEMY_HEALTH_DROP_OFFSET] > 0.0
        null_unclear = smart_feedback & ~moved & ~enemy_drop

        return {
            "steps": int(actions.numel()),
            "learnable_steps": int(sample_mask.sum().item()),
            "rollout_policy_no_op_count": int(
                (actions[sample_mask] == POLICY_ACTION_NO_OP).sum().item(),
            ),
            "rollout_policy_left_click_count": int(
                (actions[sample_mask] == POLICY_ACTION_LEFT_CLICK).sum().item(),
            ),
            "rollout_policy_right_click_count": int(
                (actions[sample_mask] == POLICY_ACTION_RIGHT_CLICK).sum().item(),
            ),
            "rollout_feedback_smart_executed_count": int(
                (smart_executed & sample_mask).sum().item(),
            ),
            "rollout_feedback_near_enemy_smart_count": int(
                (near_enemy & smart_feedback & sample_mask).sum().item(),
            ),
            "rollout_feedback_moved_toward_target_count": int(
                (moved & smart_feedback & sample_mask).sum().item(),
            ),
            "rollout_feedback_enemy_health_drop_after_smart_count": int(
                (enemy_drop & smart_feedback & sample_mask).sum().item(),
            ),
            "rollout_feedback_null_unclear_smart_count": int(
                (null_unclear & sample_mask).sum().item(),
            ),
        }

    @staticmethod
    def _aggregate_fragment_step_counters(
        fragments: list[RolloutFragment],
    ) -> dict[str, int]:
        totals: dict[str, int] = {}
        for fragment in fragments:
            for key, value in fragment.step_counters.items():
                if key.startswith("rollout_"):
                    totals[key] = totals.get(key, 0) + int(value)
        return totals

    def _move_state_to_device(
        self,
        state,
        *,
        dtype: torch.dtype = torch.float32,
    ):
        if state is None:
            return None
        return (
            state[0].to(device=self.device, dtype=dtype).detach(),
            state[1].to(device=self.device, dtype=dtype).detach(),
        )

    def _move_policy_input_to_device(
        self,
        batch: PolicyInputBatch,
        *,
        spatial_dtype: torch.dtype | None = None,
    ) -> PolicyInputBatch:
        if spatial_dtype is None:
            spatial_dtype = self._device_spatial_cache_dtype()
        state_in = self._move_state_to_device(batch.state_in, dtype=torch.float32)
        return PolicyInputBatch(
            spatial_obs=batch.spatial_obs.to(
                device=self.device,
                dtype=spatial_dtype,
            ),
            entity_features=batch.entity_features.to(
                device=self.device,
                dtype=torch.float32,
            ),
            entity_mask=batch.entity_mask.to(device=self.device),
            selection_features=batch.selection_features.to(
                device=self.device,
                dtype=torch.float32,
            ),
            selection_mask=batch.selection_mask.to(device=self.device),
            action_feedback_tokens=batch.action_feedback_tokens.to(
                device=self.device,
                dtype=torch.float32,
            ),
            meta_vec=batch.meta_vec.to(device=self.device, dtype=torch.float32),
            state_in=state_in,
        )

    @staticmethod
    def _sync_cuda_if_needed(device: torch.device | str) -> None:
        device = torch.device(device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _record_elapsed(
        self,
        timings: dict[str, float] | None,
        key: str,
        started: float,
        *,
        sync_cuda: bool = False,
    ) -> None:
        if timings is None:
            return
        if sync_cuda:
            self._sync_cuda_if_needed(self.device)
        timings[key] = float(timings.get(key, 0.0) + time.perf_counter() - started)

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
        masked_logits = action_logits.masked_fill(~available, -1.0e4)
        return self._apply_right_click_curriculum(masked_logits, available)

    def _grad_component_norms(self) -> dict[str, float]:
        """Pre-clip gradient L2 norms split by module group: the actor head,
        the critic head, the spatial target head, and everything else (trunk).
        Diagnoses which loss component dominates the shared-trunk gradient.
        Call after unscale_ and before clip_grad_norm_ (clip mutates grads)."""
        sums: dict[str, torch.Tensor] = {}
        for name, param in self.policy_net.named_parameters():
            grad = param.grad
            if not param.requires_grad or grad is None:
                continue
            if name.startswith("critic_fc"):
                key = "critic_head"
            elif name.startswith("actor_fc"):
                key = "actor_head"
            elif name.startswith("target_head"):
                key = "target_head"
            else:
                key = "trunk"
            contribution = grad.detach().float().pow(2).sum()
            if key in sums:
                sums[key] = sums[key] + contribution
            else:
                sums[key] = contribution
        return {key: float(value.sqrt().item()) for key, value in sums.items()}

    def _right_click_curriculum_penalty(self) -> float:
        if (
            self.right_click_curriculum_updates <= 0
            or self.right_click_curriculum_noop_logit_penalty <= 0.0
            or self.update_count >= self.right_click_curriculum_updates
        ):
            return 0.0

        progress = float(self.update_count) / float(
            max(1, self.right_click_curriculum_updates),
        )
        return float(
            self.right_click_curriculum_noop_logit_penalty
            * max(0.0, 1.0 - progress)
        )

    def _apply_right_click_curriculum(
        self,
        action_logits: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        penalty = self._right_click_curriculum_penalty()
        if penalty <= 0.0:
            return action_logits
        if int(action_logits.size(-1)) <= max(
            POLICY_ACTION_NO_OP,
            POLICY_ACTION_RIGHT_CLICK,
        ):
            return action_logits

        right_click_available = available[..., SEMANTIC_AVAILABLE_RIGHT_CLICK_INDEX]
        if not bool(right_click_available.any().item()):
            return action_logits

        adjusted = action_logits.clone()
        adjusted[..., POLICY_ACTION_NO_OP] = torch.where(
            right_click_available,
            adjusted[..., POLICY_ACTION_NO_OP] - penalty,
            adjusted[..., POLICY_ACTION_NO_OP],
        )
        return adjusted

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

    def _encode_step_tensors(
        self,
        *,
        spatial_obs: torch.Tensor,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
        selection_features: torch.Tensor,
        selection_mask: torch.Tensor,
        meta_vec: torch.Tensor,
        state_in: tuple[torch.Tensor, torch.Tensor] | None,
        action_feedback_tokens: torch.Tensor | None = None,
    ):
        kwargs = {
            "spatial_obs": spatial_obs,
            "entity_features": entity_features,
            "entity_mask": entity_mask,
            "selection_features": selection_features,
            "selection_mask": selection_mask,
            "meta_vec": meta_vec,
            "state_in": state_in,
        }
        if self._policy_accepts_action_feedback:
            kwargs["action_feedback_tokens"] = action_feedback_tokens
        return self.policy_net.encode_step_tensors(**kwargs)

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
                self._encode_step_tensors(
                    spatial_obs=batch.spatial_obs,
                    entity_features=batch.entity_features,
                    entity_mask=batch.entity_mask,
                    selection_features=batch.selection_features,
                    selection_mask=batch.selection_mask,
                    action_feedback_tokens=batch.action_feedback_tokens,
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
        self.pending_fragments = []

    def has_pending_rollout(self) -> bool:
        return bool(self.memory or self.pending_fragments)

    def pending_rollout_steps(self, *, include_current: bool = True) -> int:
        steps = sum(fragment.num_steps for fragment in self.pending_fragments)
        if include_current:
            steps += len(self.memory)
        return int(steps)

    def pending_learnable_steps(self, *, include_current: bool = True) -> int:
        steps = sum(
            fragment.num_learnable_steps for fragment in self.pending_fragments
        )
        if include_current:
            steps += sum(
                int(float(transition["sample_mask"].item()) > 0.0)
                for transition in self.memory
            )
        return int(steps)

    def consume_pending_fragments(self) -> list[RolloutFragment]:
        fragments = list(self.pending_fragments)
        self.pending_fragments = []
        return fragments

    @staticmethod
    def _stack_states_from_transitions(transitions):
        states = [transition["observation_batch"].state_in for transition in transitions]
        if all(state is None for state in states):
            return None
        if any(state is None for state in states):
            raise ValueError("Either every transition must carry state_in, or none may")
        return (
            torch.cat([state[0] for state in states], dim=0),
            torch.cat([state[1] for state in states], dim=0),
        )

    @staticmethod
    def _slice_state_row(state, row_index: int):
        if state is None:
            return None
        return (
            state[0][row_index : row_index + 1].detach(),
            state[1][row_index : row_index + 1].detach(),
        )

    def finalize_fragment(
        self,
        *,
        actor_id: int = 0,
        fragment_id: int | None = None,
        policy_version: int | None = None,
        episode_summaries=(),
        reward_component_summaries=(),
    ) -> RolloutFragment | None:
        if not self.memory:
            return None

        last_done = float(self.memory[-1]["done"].item()) > 0.5
        if not last_done and self.final_next is None:
            raise RuntimeError(
                "Non-terminal rollout fragment is missing bootstrap data. "
                "Call PPO.set_final_next() before finalize_fragment().",
            )

        if fragment_id is None:
            fragment_id = self._next_fragment_id
            self._next_fragment_id += 1
        if policy_version is None:
            policy_version = int(self.update_count)

        step_batches = [
            transition["observation_batch"].with_state(None)
            for transition in self.memory
        ]
        observations = PolicyInputBatch.stack(step_batches)
        pre_step_snn_state = self._stack_states_from_transitions(self.memory)
        default_false = torch.tensor(False, dtype=torch.bool)
        actions = torch.stack([transition["action"] for transition in self.memory])
        sample_mask = torch.stack(
            [transition["sample_mask"] for transition in self.memory],
        ).float()
        step_counters = self._fragment_step_counters(
            actions=actions,
            action_feedback_tokens=observations.action_feedback_tokens,
            sample_mask=sample_mask,
        )

        fragment = RolloutFragment(
            actor_id=int(actor_id),
            fragment_id=int(fragment_id),
            policy_version=int(policy_version),
            spatial_obs=observations.spatial_obs,
            entity_features=observations.entity_features,
            entity_mask=observations.entity_mask,
            selection_features=observations.selection_features,
            selection_mask=observations.selection_mask,
            action_feedback_tokens=observations.action_feedback_tokens,
            meta_vec=observations.meta_vec,
            actions=actions,
            move_x=torch.stack([transition["move_x"] for transition in self.memory]),
            move_y=torch.stack([transition["move_y"] for transition in self.memory]),
            target_index=torch.stack(
                [transition["target_index"] for transition in self.memory],
            ),
            coarse_index=torch.stack(
                [transition["coarse_index"] for transition in self.memory],
            ),
            fine_index=torch.stack(
                [transition["fine_index"] for transition in self.memory],
            ),
            old_log_probs=torch.stack(
                [transition["log_prob"] for transition in self.memory],
            ),
            values=torch.stack([transition["value"] for transition in self.memory]),
            rewards=torch.stack([transition["reward"] for transition in self.memory]),
            dones=torch.stack([transition["done"] for transition in self.memory]),
            truncateds=torch.stack(
                [
                    transition.get(
                        "truncated",
                        torch.zeros((), dtype=torch.float32),
                    )
                    for transition in self.memory
                ],
            ),
            episode_reset_mask=torch.stack(
                [
                    transition.get(
                        "episode_reset",
                        transition.get("done", default_false).bool(),
                    ).bool()
                    for transition in self.memory
                ],
            ),
            sample_mask=sample_mask,
            pre_step_snn_state=pre_step_snn_state,
            tail_next_policy_input=self.final_next,
            tail_next_snn_state=(
                None if self.final_next is None else self.final_next.state_in
            ),
            episode_summaries=tuple(episode_summaries),
            reward_component_summaries=tuple(reward_component_summaries),
            step_counters=step_counters,
        )
        self.pending_fragments.append(fragment)
        self.memory = []
        self.final_next = None
        return fragment

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
        truncated: torch.Tensor | None = None,
        episode_reset: torch.Tensor | None = None,
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
        if truncated is None:
            truncated = torch.tensor(0.0, dtype=torch.float32)
        if episode_reset is None:
            episode_reset = done
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
                "truncated": truncated.detach().to(device="cpu", dtype=torch.float32),
                "episode_reset": episode_reset.detach().to(device="cpu").bool(),
                "sample_mask": sample_mask,
                "policy_mask": sample_mask,
            }
        )

    def _bootstrap_fragment_tail_value(
        self,
        fragment: RolloutFragment,
        timings: dict[str, float] | None = None,
    ) -> torch.Tensor:
        transfer_started = time.perf_counter()
        dones = fragment.dones.to(self.device).float().view(-1)
        self._record_elapsed(
            timings,
            "cpu_to_gpu_transfer_wall_seconds",
            transfer_started,
            sync_cuda=True,
        )
        if float(dones[-1].item()) > 0.5:
            return torch.zeros((), device=self.device)
        if fragment.tail_next_policy_input is None:
            raise RuntimeError(
                "Non-terminal rollout fragment is missing bootstrap data.",
            )
        transfer_started = time.perf_counter()
        last_batch = self._move_policy_input_to_device(
            fragment.tail_next_policy_input,
            spatial_dtype=torch.float32,
        )
        self._record_elapsed(
            timings,
            "cpu_to_gpu_transfer_wall_seconds",
            transfer_started,
            sync_cuda=True,
        )
        bootstrap_started = time.perf_counter()
        with torch.no_grad():
            _, state_value, _, _ = self._encode_step_tensors(
                spatial_obs=last_batch.spatial_obs,
                entity_features=last_batch.entity_features,
                entity_mask=last_batch.entity_mask,
                selection_features=last_batch.selection_features,
                selection_mask=last_batch.selection_mask,
                action_feedback_tokens=last_batch.action_feedback_tokens,
                meta_vec=last_batch.meta_vec,
                state_in=last_batch.state_in,
            )
        value = state_value.reshape(-1)[0].detach()
        self._record_elapsed(
            timings,
            "bootstrap_value_wall_seconds",
            bootstrap_started,
            sync_cuda=True,
        )
        return value

    def _fragment_tensors(
        self,
        fragment: RolloutFragment,
        timings: dict[str, float] | None = None,
    ) -> dict:
        transfer_started = time.perf_counter()
        observations = self._move_policy_input_to_device(
            fragment.as_policy_input_batch(state_in=None),
        )
        pre_step_snn_state = self._move_state_to_device(fragment.pre_step_snn_state)
        actions = fragment.actions.to(self.device).long().view(-1)
        move_x = fragment.move_x.to(self.device).long().view(-1)
        move_y = fragment.move_y.to(self.device).long().view(-1)
        target_index = fragment.target_index.to(self.device).long().view(-1)
        coarse_index = fragment.coarse_index.to(self.device).long().view(-1)
        fine_index = fragment.fine_index.to(self.device).long().view(-1)
        log_probs_old = fragment.old_log_probs.to(self.device).float().view(-1)
        rewards = fragment.rewards.to(self.device).float().view(-1)
        values = fragment.values.to(self.device).float().view(-1)
        dones = fragment.dones.to(self.device).float().view(-1)
        truncateds = fragment.truncateds.to(self.device).float().view(-1)
        episode_reset_mask = (
            fragment.episode_reset_mask.to(self.device).bool().view(-1)
        )
        sample_masks = fragment.sample_mask.to(self.device).float().view(-1)
        self._record_elapsed(
            timings,
            "cpu_to_gpu_transfer_wall_seconds",
            transfer_started,
            sync_cuda=True,
        )
        return {
            "fragment": fragment,
            "observations": observations,
            "pre_step_snn_state": pre_step_snn_state,
            "actions": actions,
            "move_x": move_x,
            "move_y": move_y,
            "target_index": target_index,
            "coarse_index": coarse_index,
            "fine_index": fine_index,
            "log_probs_old": log_probs_old,
            "rewards": rewards,
            "values": values,
            "dones": dones,
            "truncateds": truncateds,
            "episode_reset_mask": episode_reset_mask,
            "sample_masks": sample_masks,
        }

    def update_policy(
        self,
        fragments: list[RolloutFragment] | None = None,
        batch_size: int = 64,
        epochs: int = 10,
    ):
        if fragments is None:
            if self.memory:
                self.finalize_fragment()
            fragments = self.consume_pending_fragments()
        else:
            fragments = list(fragments)

        if not fragments:
            return [], None
        update_started = time.perf_counter()
        timings = {
            "fragment_tensor_build_wall_seconds": 0.0,
            "cpu_to_gpu_transfer_wall_seconds": 0.0,
            "bootstrap_value_wall_seconds": 0.0,
            "gae_wall_seconds": 0.0,
            "tbptt_chunk_build_wall_seconds": 0.0,
            "chunk_pack_wall_seconds": 0.0,
            "replay_forward_wall_seconds": 0.0,
            "loss_eval_wall_seconds": 0.0,
            "backward_optimizer_wall_seconds": 0.0,
            "ppo_epoch_wall_seconds": 0.0,
        }
        payload_stats = self._fragment_payload_stats(fragments)
        use_cuda_memory_stats = (
            torch.device(self.device).type == "cuda" and torch.cuda.is_available()
        )
        if use_cuda_memory_stats:
            torch.cuda.reset_peak_memory_stats(self.device)

        fragment_tensor_started = time.perf_counter()
        fragment_tensors = [
            self._fragment_tensors(fragment, timings) for fragment in fragments
        ]
        self._record_elapsed(
            timings,
            "fragment_tensor_build_wall_seconds",
            fragment_tensor_started,
            sync_cuda=True,
        )
        raw_advantages_by_fragment = []
        returns_by_fragment = []
        gae_started = time.perf_counter()
        for item in fragment_tensors:
            if int(item["dones"].numel()) > 1:
                unsupported_resets = (
                    item["episode_reset_mask"][:-1]
                    & (item["dones"][:-1] <= 0.5)
                )
                if bool(unsupported_resets.any().item()):
                    raise ValueError(
                        "RolloutFragment contains an internal non-terminal "
                        "episode reset. Split time-limit fragments at the reset "
                        "boundary, or store per-boundary bootstrap values.",
                    )
            last_next_value = self._bootstrap_fragment_tail_value(
                item["fragment"],
                timings,
            )
            raw_advantages = self._compute_advantages(
                item["rewards"],
                item["values"],
                item["dones"],
                last_next_value,
            )
            raw_advantages_by_fragment.append(raw_advantages)
            returns_by_fragment.append((raw_advantages + item["values"]).detach())

        if self.sil_enabled:
            self._admit_to_sil_buffer(fragments, returns_by_fragment)

        all_raw_advantages = torch.cat(raw_advantages_by_fragment, dim=0)
        all_sample_masks = torch.cat(
            [item["sample_masks"] for item in fragment_tensors],
            dim=0,
        )
        if bool((all_sample_masks > 0.0).any().item()):
            valid_advantages = all_raw_advantages[all_sample_masks > 0.0]
            normalized_advantages = (
                all_raw_advantages - valid_advantages.mean()
            ) / (valid_advantages.std(unbiased=False) + 1e-8)
        else:
            normalized_advantages = torch.zeros_like(
                all_raw_advantages,
                device=self.device,
            )
        self._record_elapsed(
            timings,
            "gae_wall_seconds",
            gae_started,
            sync_cuda=True,
        )

        chunks = []
        cursor = 0
        chunk_build_started = time.perf_counter()
        for item, returns in zip(fragment_tensors, returns_by_fragment):
            length = int(item["actions"].size(0))
            advantages = normalized_advantages[cursor : cursor + length]
            cursor += length
            chunks.extend(
                self._build_tbptt_chunks(
                    observations=item["observations"],
                    pre_step_snn_state=item["pre_step_snn_state"],
                    actions=item["actions"],
                    move_xs=item["move_x"],
                    move_ys=item["move_y"],
                    target_indices=item["target_index"],
                    coarse_indices=item["coarse_index"],
                    fine_indices=item["fine_index"],
                    log_probs_old=item["log_probs_old"],
                    advantages=advantages,
                    returns=returns,
                    dones=item["dones"],
                    episode_reset_mask=item["episode_reset_mask"],
                    sample_masks=item["sample_masks"],
                ),
            )
        tbptt_chunks = int(len(chunks))
        self._record_elapsed(
            timings,
            "tbptt_chunk_build_wall_seconds",
            chunk_build_started,
            sync_cuda=True,
        )

        rollout_size = int(sum(fragment.num_steps for fragment in fragments))
        losses = []
        acc_policy = []
        acc_value = []
        acc_entropy = []
        acc_kl = []
        acc_clip_frac = []
        acc_grad_norm = []
        acc_grad_component_norms: dict[str, list[float]] = {
            "trunk": [],
            "actor_head": [],
            "critic_head": [],
            "target_head": [],
        }
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

        epoch_loop_started = time.perf_counter()
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
                    pack_started = time.perf_counter()
                    packed_group = self._pack_chunk_group(chunk_group)
                    self._record_elapsed(
                        timings,
                        "chunk_pack_wall_seconds",
                        pack_started,
                        sync_cuda=True,
                    )

                    replay_started = time.perf_counter()
                    replayed_group = self._replay_packed_chunk_group(packed_group)
                    self._record_elapsed(
                        timings,
                        "replay_forward_wall_seconds",
                        replay_started,
                        sync_cuda=True,
                    )
                    tbptt_group_max_steps = max(
                        tbptt_group_max_steps,
                        int(replayed_group["max_steps"]),
                    )
                    tbptt_forward_calls += int(replayed_group["forward_calls"])
                    active_chunk_sum += float(replayed_group["active_chunks_sum"])
                    active_chunk_steps += int(replayed_group["active_steps"])

                    loss_started = time.perf_counter()
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
                    self._record_elapsed(
                        timings,
                        "loss_eval_wall_seconds",
                        loss_started,
                        sync_cuda=True,
                    )

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

                backward_started = time.perf_counter()
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
                    for key, value in self._grad_component_norms().items():
                        acc_grad_component_norms[key].append(value)
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, 0.5)
                    grad_norm_value = float(grad_norm.item())

                if grads_finite and math.isfinite(grad_norm_value):
                    self.policy_net.scaler.step(self.optimizer)
                else:
                    nonfinite_grad_steps += 1
                    skipped_optimizer_steps += 1

                acc_grad_norm.append(grad_norm_value)
                self.policy_net.scaler.update()
                self._record_elapsed(
                    timings,
                    "backward_optimizer_wall_seconds",
                    backward_started,
                    sync_cuda=True,
                )

            epochs_ran += 1
            if self.target_kl is not None and epoch_kls:
                if float(np.mean(epoch_kls)) > float(self.target_kl):
                    break
        self._record_elapsed(
            timings,
            "ppo_epoch_wall_seconds",
            epoch_loop_started,
            sync_cuda=True,
        )

        sil_stats = (
            self._run_sil_pass(rollout_size, params, timings)
            if self.sil_enabled
            else {}
        )

        with torch.no_grad():
            returns = torch.cat(returns_by_fragment, dim=0)
            values = torch.cat([item["values"] for item in fragment_tensors], dim=0)
            var_returns = returns.var(unbiased=False)
            explained_var = 1.0 - (returns - values).var(unbiased=False) / (
                var_returns + 1e-8
            )

        returns_cpu = returns.detach().to("cpu").float()
        update_wall_seconds = time.perf_counter() - update_started
        entity_counts = torch.cat(
            [
                fragment.entity_mask.sum(dim=1).detach().to("cpu").float()
                for fragment in fragments
            ],
            dim=0,
        )
        selection_counts = torch.cat(
            [
                fragment.selection_mask.sum(dim=1).detach().to("cpu").float()
                for fragment in fragments
            ],
            dim=0,
        )
        if use_cuda_memory_stats:
            self._sync_cuda_if_needed(self.device)
            cuda_peak_allocated_bytes = int(torch.cuda.max_memory_allocated(self.device))
            cuda_peak_reserved_bytes = int(torch.cuda.max_memory_reserved(self.device))
        else:
            cuda_peak_allocated_bytes = 0
            cuda_peak_reserved_bytes = 0
        stats = {
            "mean_policy_loss": float(np.mean(acc_policy)),
            "mean_value_loss": float(np.mean(acc_value)),
            "mean_entropy": float(np.mean(acc_entropy)),
            "mean_kl": float(np.mean(acc_kl)),
            "clip_fraction": float(np.mean(acc_clip_frac)),
            "explained_variance": float(explained_var.item()),
            "grad_norm": float(np.mean(acc_grad_norm)),
            **{
                f"grad_norm_{key}": (
                    float(np.mean(values)) if values else 0.0
                )
                for key, values in acc_grad_component_norms.items()
            },
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "nonfinite_grad_steps": int(nonfinite_grad_steps),
            "skipped_optimizer_steps": int(skipped_optimizer_steps),
            "transitions_in_update": int(rollout_size),
            "learnable_transitions_in_update": int(
                sum(fragment.num_learnable_steps for fragment in fragments),
            ),
            "fragments_in_update": int(len(fragments)),
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
            "cuda_peak_allocated_bytes": int(cuda_peak_allocated_bytes),
            "cuda_peak_reserved_bytes": int(cuda_peak_reserved_bytes),
            "rollout_cache_spatial_dtype": str(
                self._device_spatial_cache_dtype(),
            ).replace("torch.", ""),
        }
        stats.update(payload_stats)
        stats.update(self._aggregate_fragment_step_counters(fragments))
        stats.update(timings)
        if self.sil_enabled:
            stats.update(
                {
                    "sil_loss": 0.0,
                    "sil_gate_open_fraction": 0.0,
                    "sil_buffer_size": len(self.sil_buffer),
                    "sil_steps_replayed": 0,
                    "sil_groups": 0,
                },
            )
            stats.update(sil_stats)

        if self.scheduler is not None:
            self.scheduler.step()
        self.update_count += 1
        stats["global_update_index"] = int(self.update_count)

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
        observations: PolicyInputBatch | None = None,
        pre_step_snn_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        episode_reset_mask: torch.Tensor | None = None,
        sample_masks: torch.Tensor | None = None,
        policy_masks: torch.Tensor | None = None,
        target_indices: torch.Tensor | None = None,
        coarse_indices: torch.Tensor | None = None,
        fine_indices: torch.Tensor | None = None,
    ):
        rollout_size = int(actions.size(0))
        if sample_masks is None:
            if policy_masks is not None:
                sample_masks = policy_masks
            else:
                sample_masks = torch.ones(
                    rollout_size,
                    dtype=torch.float32,
                    device=self.device,
                )
        if episode_reset_mask is None:
            episode_reset_mask = torch.zeros(
                rollout_size,
                dtype=torch.bool,
                device=dones.device,
            )
        else:
            episode_reset_mask = episode_reset_mask.to(
                device=dones.device,
                dtype=torch.bool,
            )
        if target_indices is None:
            target_indices = torch.full_like(actions, -1, dtype=torch.long)
        if coarse_indices is None:
            coarse_indices = torch.full_like(actions, -1, dtype=torch.long)
        if fine_indices is None:
            fine_indices = torch.full_like(actions, -1, dtype=torch.long)
        chunks = []
        window = rollout_size if self.tbptt_window is None else self.tbptt_window
        start = 0
        while start < rollout_size:
            end = min(start + window, rollout_size)
            split_mask = (dones[start:end] > 0.5) | episode_reset_mask[start:end]
            split_indices = torch.nonzero(split_mask, as_tuple=False)
            if len(split_indices) > 0:
                end = start + int(split_indices[0].item()) + 1

            if observations is None:
                step_batches = [
                    self.memory[idx]["observation_batch"].with_state(None)
                    for idx in range(start, end)
                ]
                chunk_observations = PolicyInputBatch.stack(step_batches)
                initial_state = self.memory[start]["observation_batch"].state_in
            else:
                index = torch.arange(
                    start,
                    end,
                    device=observations.spatial_obs.device,
                )
                chunk_observations = observations.index_select(index).with_state(None)
                initial_state = self._slice_state_row(pre_step_snn_state, start)

            chunks.append(
                {
                    "observations": chunk_observations,
                    "initial_state": initial_state,
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
        action_feedback_tokens: torch.Tensor | None,
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
            self._encode_step_tensors(
                spatial_obs=spatial_obs,
                entity_features=entity_features,
                entity_mask=entity_mask,
                selection_features=selection_features,
                selection_mask=selection_mask,
                action_feedback_tokens=action_feedback_tokens,
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
        target_device = torch.device(self.device)
        spatial_dtype = (
            sample_obs.spatial_obs.dtype
            if sample_obs.spatial_obs.is_floating_point()
            else torch.float32
        )

        spatial_obs = torch.zeros(
            (max_len, group_size, *sample_obs.spatial_obs.shape[1:]),
            device=self.device,
            dtype=spatial_dtype,
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
        action_feedback_tokens = torch.zeros(
            (max_len, group_size, *sample_obs.action_feedback_tokens.shape[1:]),
            device=self.device,
            dtype=torch.float32,
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
            obs = chunk["observations"]
            if obs.spatial_obs.device != target_device:
                obs = self._move_policy_input_to_device(
                    obs,
                    spatial_dtype=spatial_dtype,
                )
            spatial_obs[:length, column] = obs.spatial_obs
            entity_features[:length, column] = obs.entity_features
            entity_mask[:length, column] = obs.entity_mask
            selection_features[:length, column] = obs.selection_features
            selection_mask[:length, column] = obs.selection_mask
            action_feedback_tokens[:length, column] = obs.action_feedback_tokens
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
            "action_feedback_tokens": action_feedback_tokens,
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
                    action_feedback_tokens=packed_group["action_feedback_tokens"][
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
                    action_feedback_tokens=step_batch.action_feedback_tokens,
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
            action_feedback_tokens=torch.cat(
                [
                    obs.action_feedback_tokens[step_index : step_index + 1]
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

    def _admit_to_sil_buffer(
        self,
        fragments: list[RolloutFragment],
        returns_by_fragment: list[torch.Tensor],
    ) -> None:
        """Admit committed-attack transitions to the SIL trophy buffer.

        The action_feedback_tokens at step i describe the action taken at step
        i-1 (their effect is what is visible in observation i; see
        ``action_feedback_encoder.py``). So a RIGHT_CLICK at step j is confirmed
        to have engaged an enemy by the feedback token at step j+1 (target near
        an enemy, or an enemy health drop). We admit step j (the click itself,
        with its own observation / pre-step SNN state / return R) only when that
        next-step feedback confirms engagement. Idle steps and missed clicks are
        never admitted, so the buffer cannot fill with the auto-attack-confounded
        idle behavior that dominates winning episodes.
        """
        for fragment, returns in zip(fragments, returns_by_fragment):
            actions = fragment.actions.detach().cpu().long().reshape(-1)
            num_steps = int(actions.shape[0])
            if num_steps < 2:
                continue  # need a following step's feedback to confirm engagement
            feedback = fragment.action_feedback_tokens.detach().cpu().float()
            feedback = feedback.reshape(num_steps, -1)
            if int(feedback.shape[0]) != num_steps:
                continue
            returns_cpu = returns.detach().cpu().float().reshape(-1)
            sample_mask = (
                fragment.sample_mask.detach().cpu().float().reshape(-1) > 0.0
            )
            if int(sample_mask.shape[0]) != num_steps:
                continue
            obs_batch = fragment.as_policy_input_batch(state_in=None)
            pre_state = fragment.pre_step_snn_state
            for j in range(num_steps - 1):
                if not bool(sample_mask[j].item()):
                    continue
                if int(actions[j].item()) != POLICY_ACTION_RIGHT_CLICK:
                    continue
                fb_next = feedback[j + 1]
                near_enemy = bool(
                    fb_next[ACTION_FEEDBACK_TARGET_NEAR_ENEMY_OFFSET].item() > 0.5
                )
                health_drop = bool(
                    fb_next[ACTION_FEEDBACK_ENEMY_HEALTH_DROP_OFFSET].item() > 0.0
                )
                if not (near_enemy or health_drop):
                    continue
                self.sil_buffer.append(
                    self._build_sil_entry(
                        obs_batch, pre_state, fragment, returns_cpu, j,
                    ),
                )

    def _build_sil_entry(
        self,
        obs_batch: PolicyInputBatch,
        pre_state,
        fragment: RolloutFragment,
        returns_cpu: torch.Tensor,
        i: int,
    ) -> dict:
        index = torch.tensor([int(i)], dtype=torch.long)
        obs_step = obs_batch.index_select(index).with_state(None)
        state_step = self._slice_state_row(pre_state, i)
        if state_step is not None:
            state_step = (state_step[0].clone(), state_step[1].clone())
        actions_row = fragment.actions.detach().cpu().long().reshape(-1)
        move_x_row = fragment.move_x.detach().cpu().long().reshape(-1)
        move_y_row = fragment.move_y.detach().cpu().long().reshape(-1)
        target_row = fragment.target_index.detach().cpu().long().reshape(-1)
        coarse_row = fragment.coarse_index.detach().cpu().long().reshape(-1)
        fine_row = fragment.fine_index.detach().cpu().long().reshape(-1)
        old_lp_row = fragment.old_log_probs.detach().cpu().float().reshape(-1)
        return {
            "observations": obs_step,
            "initial_state": state_step,
            "actions": actions_row[i : i + 1].clone(),
            "move_x": move_x_row[i : i + 1].clone(),
            "move_y": move_y_row[i : i + 1].clone(),
            "target_index": target_row[i : i + 1].clone(),
            "coarse_index": coarse_row[i : i + 1].clone(),
            "fine_index": fine_row[i : i + 1].clone(),
            "old_log_prob": old_lp_row[i : i + 1].clone(),
            "advantages": torch.zeros(1, dtype=torch.float32),
            "returns": returns_cpu[i : i + 1].clone(),
            "dones": torch.ones(1, dtype=torch.float32),
            "sample_mask": torch.ones(1, dtype=torch.float32),
            "policy_mask": torch.ones(1, dtype=torch.float32),
            "length": 1,
        }

    def _run_sil_pass(
        self,
        rollout_size: int,
        params,
        timings: dict[str, float] | None,
    ) -> dict:
        """One SIL auxiliary update: replay sampled trophy transitions through
        the current policy and apply the gated imitation loss
        ``sil_coef * mean( (R - V_current)+ . -log pi(a|s) )`` via its own
        backward+step. Returns SIL diagnostics, or {} if SIL did not run.
        """
        if (
            not self.sil_enabled
            or self.sil_coef <= 0.0
            or self.sil_batch_fraction <= 0.0
            or len(self.sil_buffer) == 0
        ):
            return {}

        sil_started = time.perf_counter()
        buffer_list = list(self.sil_buffer)
        sil_batch = max(
            1,
            int(self.sil_batch_fraction * max(1, int(rollout_size))),
        )
        sil_batch = min(sil_batch, len(buffer_list))
        order = torch.randperm(len(buffer_list))[:sil_batch].tolist()
        sampled = [buffer_list[k] for k in order]

        total_sil_loss = 0.0
        gate_open_sum = 0.0
        gate_count = 0.0
        sil_groups = 0
        sil_grad_norms: list[float] = []

        for chunk_group in self._iter_chunk_groups(sampled, sil_batch):
            with torch.amp.autocast(
                "cuda",
                dtype=self.policy_net.amp_dtype,
                enabled=self.policy_net.use_amp,
            ):
                pack_started = time.perf_counter()
                packed_group = self._pack_chunk_group(chunk_group)
                self._record_elapsed(
                    timings,
                    "sil_chunk_pack_wall_seconds",
                    pack_started,
                    sync_cuda=True,
                )
                replay_started = time.perf_counter()
                replayed_group = self._replay_packed_chunk_group(packed_group)
                self._record_elapsed(
                    timings,
                    "sil_replay_forward_wall_seconds",
                    replay_started,
                    sync_cuda=True,
                )

                active_mask = replayed_group["alive_mask"].reshape(-1)
                action_logits = replayed_group["action_logits"].reshape(
                    -1,
                    replayed_group["action_logits"].size(-1),
                )[active_mask]
                target_log_prob = replayed_group["target_log_prob"].reshape(-1)[
                    active_mask
                ]
                state_values = replayed_group["state_values"].reshape(-1)[
                    active_mask
                ]
                actions = replayed_group["actions"].reshape(-1)[active_mask]
                returns_r = (
                    replayed_group["returns"].reshape(-1)[active_mask].float()
                )
                sample_mask = replayed_group["sample_mask"].reshape(-1)[
                    active_mask
                ]

                action_dist = torch.distributions.Categorical(
                    logits=action_logits.float(),
                )
                is_spatial = self._spatial_action_mask(actions)
                new_log_prob = (
                    action_dist.log_prob(actions)
                    + is_spatial * target_log_prob
                )
                # Gate = (R - V_current)+, detached: a fixed per-sample weight,
                # not something we backprop through the critic.
                weight = (returns_r - state_values.float()).clamp(min=0.0).detach()
                sil_loss = -self.sil_coef * self._masked_mean(
                    weight * new_log_prob,
                    sample_mask,
                )

            backward_started = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            self.policy_net.scaler.scale(sil_loss).backward()
            self.policy_net.scaler.unscale_(self.optimizer)

            grads_finite = True
            for param in params:
                grad = param.grad
                if grad is not None and not torch.isfinite(grad).all():
                    grads_finite = False
                    break
            if grads_finite:
                sil_grad_norm = torch.nn.utils.clip_grad_norm_(params, 0.5)
                sil_grad_norms.append(float(sil_grad_norm.item()))
                self.policy_net.scaler.step(self.optimizer)
            self.policy_net.scaler.update()
            self._record_elapsed(
                timings,
                "sil_backward_optimizer_wall_seconds",
                backward_started,
                sync_cuda=True,
            )

            total_sil_loss += float(sil_loss.item())
            gate_open_sum += float(
                ((weight > 0.0).float() * sample_mask).sum().item(),
            )
            gate_count += float(sample_mask.sum().item())
            sil_groups += 1

        self._record_elapsed(
            timings, "sil_wall_seconds", sil_started, sync_cuda=True,
        )
        return {
            "sil_loss": total_sil_loss / max(1, sil_groups),
            "sil_gate_open_fraction": gate_open_sum / max(1.0, gate_count),
            "sil_buffer_size": len(self.sil_buffer),
            "sil_steps_replayed": int(sil_batch),
            "sil_groups": int(sil_groups),
            "sil_grad_norm": (
                float(np.mean(sil_grad_norms)) if sil_grad_norms else 0.0
            ),
        }
