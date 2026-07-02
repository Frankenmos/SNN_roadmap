"""TinySkirmish protocol lab.

Legible, numpy-only skirmish environment emitting the same policy-input
envelope as the PySC2 pipeline (protocol v3 shapes), for verifying the
architecture and reward attribution without StarCraft. The base env imports
neither PySC2 nor Torch; the torch_* and real_snn_* modules bridge into the
real `agent_core` PolicyNetwork/PPO.
"""

from .env import TinySkirmishEnv
from .protocol import (
    ACTION_LEFT_CLICK,
    ACTION_NO_OP,
    ACTION_RIGHT_CLICK,
    ACTION_NAMES,
    ObservationBatch,
    RewardInfo,
    SkirmishAction,
    StepResult,
)

__all__ = [
    "ACTION_LEFT_CLICK",
    "ACTION_NAMES",
    "ACTION_NO_OP",
    "ACTION_RIGHT_CLICK",
    "ObservationBatch",
    "RewardInfo",
    "SkirmishAction",
    "StepResult",
    "TinySkirmishEnv",
]
