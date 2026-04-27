from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

from agent_core.policy_protocol import POLICY_INPUT_SCHEMA, POLICY_PROTOCOL_VERSION
from distributed.learner import move_agent_to_device
from distributed.protocol import RolloutFragment, validate_policy_protocol


def _ensure_absl_flags_parsed(argv0: str = "ray_rollout_actor") -> None:
    """Parse absl flags in Ray workers before PySC2 reads FLAGS values."""
    from absl import flags

    absl_flags = flags.FLAGS
    if not absl_flags.is_parsed():
        absl_flags([argv0])


class RolloutActor:
    """Ray-transport wrapper around LocalRolloutWorker.

    The class is intentionally undecorated so unit tests can import it without
    starting Ray; `distributed.ray_train` applies `ray.remote(...)`.
    """

    def __init__(
        self,
        *,
        actor_id: int,
        repo_root: str,
        config_path: str,
        run_name: str | None = None,
        visualize: bool = False,
    ) -> None:
        self.actor_id = int(actor_id)
        self.policy_version = -1
        self.repo_root = Path(repo_root).resolve()
        self.config_path = Path(config_path).resolve()
        os.environ["SNN_CONFIG_PATH"] = str(self.config_path)
        os.chdir(self.repo_root)
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        from Utility.config import cfg

        cfg.reload(self.config_path)
        if run_name:
            cfg.environment.run_name = run_name
        actor_device_name = str(getattr(cfg.distributed, "actor_device", "cpu"))
        if torch.device(actor_device_name).type == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        from agent import DefeatRoaches
        from distributed.rollout import LocalRolloutWorker
        from envs.setup_env import create_env

        _ensure_absl_flags_parsed()

        self.env = create_env(
            map_name=cfg.environment.map_name,
            visualize=bool(visualize),
            use_action_printer=False,
            use_available_actions_diagnostics=False,
            use_last_action_diagnostics=False,
            use_score_diagnostics=False,
            use_observation_inspector=False,
            use_policy_input_diagnostics=False,
        )
        self.agent = DefeatRoaches()
        self.device = move_agent_to_device(
            self.agent,
            actor_device_name,
        )
        self.agent.policy.eval()
        runtime_profile = str(
            getattr(cfg.distributed, "sc2_runtime_profile", "windows_local"),
        ).lower()
        serialize_env_resets = bool(
            getattr(
                cfg.distributed,
                "serialize_env_resets",
                runtime_profile.startswith("windows"),
            ),
        )
        self.worker = LocalRolloutWorker(
            actor_id=self.actor_id,
            env=self.env,
            agent=self.agent,
            steps_per_episode=int(cfg.environment.steps_per_episode),
            reward_scale=float(getattr(cfg.hyperparameters, "reward_scale", 1.0)),
            serialize_env_resets=serialize_env_resets,
        )

    def set_weights(self, payload: dict[str, Any]) -> dict[str, Any]:
        validate_policy_protocol(
            policy_protocol_version=payload.get("policy_protocol_version"),
            policy_input_schema=payload.get("policy_input_schema"),
        )
        self.agent.policy.load_state_dict(payload["state_dict"])
        extractor_state = payload.get("extractor_state")
        if extractor_state is not None:
            self.agent.extractor.load_state_dict(extractor_state)
        self.policy_version = int(payload["policy_version"])
        self.agent.ppo.update_count = int(self.policy_version)
        self.agent.snn_state = self.agent.policy.init_concrete_state(
            batch_size=1,
            device=self.device,
        )
        self.agent.policy.eval()
        return {
            "actor_id": int(self.actor_id),
            "policy_version": int(self.policy_version),
            "device": str(self.device),
            "policy_protocol_version": POLICY_PROTOCOL_VERSION,
            "policy_input_schema": POLICY_INPUT_SCHEMA,
        }

    def collect_fragment(
        self,
        target_steps: int,
        policy_version: int,
    ) -> RolloutFragment:
        if int(policy_version) != int(self.policy_version):
            raise ValueError(
                "RolloutActor policy_version mismatch. Broadcast learner weights "
                f"before collecting: actor={self.policy_version}, "
                f"requested={policy_version}",
            )
        with torch.no_grad():
            return self.worker.collect_fragment(
                target_steps=int(target_steps),
                policy_version=int(policy_version),
            )

    def get_stats(self) -> dict[str, Any]:
        stats = self.worker.stats()
        stats["policy_version"] = int(self.policy_version)
        stats["device"] = str(self.device)
        return stats

    def close(self) -> None:
        self.env.close()
