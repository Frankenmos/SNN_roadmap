from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from distributed.protocol import EpisodeSummary, RolloutFragment
from distributed.sc2_runtime import sc2_create_game_lock


@dataclass(slots=True)
class RolloutCounters:
    env_steps: int = 0
    learnable_steps: int = 0
    helper_steps: int = 0
    episodes_completed: int = 0


class LocalRolloutWorker:
    """Owns one env/agent pair and emits rollout fragments.

    This class intentionally has no Ray dependency. The Ray actor wrapper is
    just transport around this stateful collector.
    """

    def __init__(
        self,
        *,
        actor_id: int,
        env,
        agent,
        steps_per_episode: int,
        reward_scale: float = 1.0,
        serialize_env_resets: bool = False,
    ) -> None:
        self.actor_id = int(actor_id)
        self.env = env
        self.agent = agent
        self.steps_per_episode = int(steps_per_episode)
        self.reward_scale = float(reward_scale)
        self.serialize_env_resets = bool(serialize_env_resets)
        self.fragment_id = 0
        self.episode_index = -1
        self.current_obs = None
        self.episode_reward = 0.0
        self.cumulative_reward = 0.0
        self.step_count = 0
        self.counters = RolloutCounters()

    def collect_fragment(
        self,
        *,
        target_steps: int,
        policy_version: int,
        max_env_steps: int | None = None,
    ) -> RolloutFragment:
        target_steps = max(1, int(target_steps))
        policy_version = int(policy_version)
        if max_env_steps is None:
            max_env_steps = max(target_steps * 4, self.steps_per_episode + 1)

        episode_summaries: list[EpisodeSummary] = []
        env_steps_in_call = 0
        self._ensure_episode(policy_version)

        while self.agent.ppo.pending_rollout_steps() < target_steps:
            if env_steps_in_call >= max_env_steps:
                raise RuntimeError(
                    "Rollout worker exceeded max_env_steps before producing a "
                    "fragment. This usually means the env is stuck in helper-only "
                    "steps or SC2 is not advancing.",
                )

            (
                action_func,
                action_id,
                move_x,
                move_y,
                _pre_step_state,
                log_prob,
                value,
                policy_input,
                learnable,
            ) = self.agent.step(self.current_obs)

            next_obs = self.env.step([action_func])[0]
            env_steps_in_call += 1
            self.counters.env_steps += 1
            self.step_count += 1

            env_done = bool(next_obs.last())
            time_cap = self.step_count >= self.steps_per_episode
            terminal = env_done
            done = terminal
            truncated = bool(time_cap and not terminal)
            episode_reset = bool(env_done or time_cap)

            raw_reward = self.agent.reward_function.calculate_reward(next_obs, None)
            raw_reward = float(
                raw_reward.item() if isinstance(raw_reward, torch.Tensor) else raw_reward,
            )
            scaled_reward = raw_reward * self.reward_scale

            # Bootstrap steps (select_army) return policy_input=None - skip these
            if policy_input is None:
                self.counters.helper_steps += 1
                if episode_reset:
                    episode_summaries.append(
                        EpisodeSummary(
                            actor_id=self.actor_id,
                            episode_index=self.episode_index,
                            total_reward=float(self.episode_reward),
                            steps=int(self.step_count),
                            terminated=bool(terminal),
                            truncated=bool(truncated),
                            policy_version=policy_version,
                        ),
                    )
                    self.counters.episodes_completed += 1
                    self.current_obs = None
                    break
                self.current_obs = next_obs
                continue

            action_sample = getattr(self.agent, "last_action_sample", None)
            device = self.agent.policy.device
            self.agent.ppo.store_transition(
                policy_input,
                torch.tensor(action_id, device=device),
                torch.tensor(move_x, device=device),
                torch.tensor(move_y, device=device),
                torch.tensor(log_prob, device=device),
                torch.tensor(scaled_reward, device=device),
                torch.tensor(value, device=device),
                torch.tensor(done, dtype=torch.float32, device=device),
                sample_mask=torch.tensor(
                    1.0 if learnable else 0.0,
                    dtype=torch.float32,
                    device=device,
                ),
                truncated=torch.tensor(
                    truncated,
                    dtype=torch.float32,
                    device=device,
                ),
                episode_reset=torch.tensor(
                    episode_reset,
                    dtype=torch.bool,
                    device=device,
                ),
                target_index=(
                    None
                    if action_sample is None or action_sample.target_index is None
                    else torch.tensor(action_sample.target_index, device=device)
                ),
                coarse_index=(
                    None
                    if action_sample is None or action_sample.coarse_index is None
                    else torch.tensor(action_sample.coarse_index, device=device)
                ),
                fine_index=(
                    None
                    if action_sample is None or action_sample.fine_index is None
                    else torch.tensor(action_sample.fine_index, device=device)
                ),
            )
            next_policy_input = self.agent.peek_observation(next_obs).with_state(
                self.agent.snn_state,
            )
            self.agent.ppo.set_final_next(next_policy_input)
            if learnable:
                self.counters.learnable_steps += 1
            else:
                self.counters.helper_steps += 1

            self.episode_reward += raw_reward
            self.cumulative_reward += raw_reward

            if episode_reset:
                episode_summaries.append(
                    EpisodeSummary(
                        actor_id=self.actor_id,
                        episode_index=self.episode_index,
                        total_reward=float(self.episode_reward),
                        steps=int(self.step_count),
                        terminated=bool(terminal),
                        truncated=bool(truncated),
                        policy_version=policy_version,
                    ),
                )
                self.counters.episodes_completed += 1
                self.current_obs = None
                break

            self.current_obs = next_obs

        if not self.agent.ppo.memory:
            # All transitions were already moved to fragments via finalize_fragment
            # This can happen when pending_rollout_steps() >= target_steps from
            # previously collected fragments, and the loop didn't add new steps.
            # Return None to signal no new fragment is ready.
            return None

        fragment = self.agent.ppo.finalize_fragment(
            actor_id=self.actor_id,
            fragment_id=self.fragment_id,
            policy_version=policy_version,
            episode_summaries=episode_summaries,
        )
        self.fragment_id += 1
        if fragment is None:
            raise RuntimeError("Rollout worker failed to finalize a fragment")
        self.agent.ppo.consume_pending_fragments()
        return fragment

    def stats(self) -> dict[str, Any]:
        return {
            "actor_id": int(self.actor_id),
            "episode_index": int(self.episode_index),
            "fragment_id": int(self.fragment_id),
            "env_steps": int(self.counters.env_steps),
            "learnable_steps": int(self.counters.learnable_steps),
            "helper_steps": int(self.counters.helper_steps),
            "episodes_completed": int(self.counters.episodes_completed),
        }

    def _ensure_episode(self, policy_version: int) -> None:
        del policy_version
        if self.current_obs is not None:
            return
        with sc2_create_game_lock(enabled=self.serialize_env_resets):
            self.current_obs = self.env.reset()[0]
        self.agent.reset()
        self.agent.reward_function.calculate_reward(self.current_obs, None)
        self.episode_index += 1
        self.episode_reward = 0.0
        self.cumulative_reward = 0.0
        self.step_count = 0
