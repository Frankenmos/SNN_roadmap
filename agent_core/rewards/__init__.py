"""Reward implementations for supported environments."""

from agent_core.rewards.defeat_roaches_v2 import RewardFunctionV2
from agent_core.rewards.defeat_roaches_v3 import RewardFunctionV3


def build_reward_function(name="defeat_roaches_v3", **kwargs):
    normalized = str(name or "defeat_roaches_v3").strip().lower()
    if normalized in {"defeat_roaches_v2", "v2"}:
        return RewardFunctionV2()
    if normalized in {"defeat_roaches_v3", "v3"}:
        return RewardFunctionV3(**kwargs)
    raise ValueError(f"Unknown reward function name: {name}")
