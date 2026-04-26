from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch


SPATIAL_OBS_SHAPE: Final[tuple[int, int, int]] = (27, 84, 84)
SPATIAL_TOKEN_COUNT: Final[int] = 49
MAX_ENTITY_TOKENS: Final[int] = 24
MAX_SELECTION_TOKENS: Final[int] = 20
SELECTION_FEATURE_DIM: Final[int] = 7
TOKEN_TYPE_SPATIAL: Final[int] = 0
TOKEN_TYPE_ENTITY: Final[int] = 1
TOKEN_TYPE_SELECTION: Final[int] = 2
TOKEN_TYPE_ACTION_FEEDBACK: Final[int] = 3
TOKEN_TYPE_META: Final[int] = 4
TOKEN_TYPE_GROUPS: Final[int] = 5
ACTION_FEEDBACK_TOKEN_COUNT: Final[int] = 1
ACTION_FEEDBACK_TOKEN_DIM: Final[int] = 9
TOTAL_TOKEN_COUNT: Final[int] = (
    SPATIAL_TOKEN_COUNT
    + MAX_ENTITY_TOKENS
    + MAX_SELECTION_TOKENS
    + ACTION_FEEDBACK_TOKEN_COUNT
    + 1
)

CURATED_FEATURE_UNIT_FIELDS: Final[tuple[str, ...]] = (
    "unit_type",
    "alliance",
    "health",
    "health_ratio",
    "shield",
    "shield_ratio",
    "energy",
    "energy_ratio",
    "weapon_cooldown",
    "x",
    "y",
    "radius",
    "build_progress",
    "order_id_0",
    "order_id_1",
    "is_selected",
    "is_in_cargo",
    "assigned_harvesters",
    "ideal_harvesters",
    "active",
    "hallucination",
)
UNIT_TYPE_FEATURE_INDEX: Final[int] = 0

SELECTION_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "unit_type",
    "player_relative",
    "health",
    "shields",
    "energy",
    "transport_slots_taken",
    "build_progress",
)
SELECTION_UNIT_TYPE_INDEX: Final[int] = 0

META_PLAYER_FEATURE_DIM: Final[int] = 11
DEFEAT_ROACHES_ACTION_IDS: Final[tuple[int, ...]] = (
    0,
    1,
    2,
    3,
    4,
    7,
    12,
    13,
    274,
    331,
    332,
    333,
    334,
    451,
    452,
    453,
)
RAW_AVAILABLE_ACTION_DIM: Final[int] = len(DEFEAT_ROACHES_ACTION_IDS)
META_AVAILABLE_ACTION_DIM: Final[int] = 3
META_LAST_ACTION_INDEX_DIM: Final[int] = 1
META_PLAYER_FEATURE_OFFSET: Final[int] = 0
META_AVAILABLE_ACTION_OFFSET: Final[int] = META_PLAYER_FEATURE_OFFSET + META_PLAYER_FEATURE_DIM
META_LAST_ACTION_INDEX_OFFSET: Final[int] = (
    META_AVAILABLE_ACTION_OFFSET + META_AVAILABLE_ACTION_DIM
)
AGENT_ACTION_TOKEN_DIM: Final[int] = 4
ACTION_HISTORY_DIM: Final[int] = 5
AGENT_LAST_ACTION_DIM: Final[int] = AGENT_ACTION_TOKEN_DIM + ACTION_HISTORY_DIM
AGENT_LAST_ACTION_OFFSET: Final[int] = (
    META_LAST_ACTION_INDEX_OFFSET + META_LAST_ACTION_INDEX_DIM
)
ACTION_HISTORY_OFFSET: Final[int] = AGENT_LAST_ACTION_OFFSET + AGENT_ACTION_TOKEN_DIM
LAST_ANY_ACTION_EXECUTED_OFFSET: Final[int] = ACTION_HISTORY_OFFSET
LAST_SMART_EXECUTED_OFFSET: Final[int] = ACTION_HISTORY_OFFSET + 1
SCORE_TOTAL_DELTA_OFFSET: Final[int] = ACTION_HISTORY_OFFSET + 2
KILLED_VALUE_DELTA_OFFSET: Final[int] = ACTION_HISTORY_OFFSET + 3
SCORE_PENALTY_BIT_OFFSET: Final[int] = ACTION_HISTORY_OFFSET + 4
META_VECTOR_DIM: Final[int] = (
    META_LAST_ACTION_INDEX_OFFSET + META_LAST_ACTION_INDEX_DIM
)
POLICY_PROTOCOL_VERSION: Final[int] = 2
NO_ACTION_SENTINEL_INDEX: Final[int] = 0
UNKNOWN_LAST_ACTION_INDEX: Final[int] = RAW_AVAILABLE_ACTION_DIM + 1

ACTION_FEEDBACK_BRIDGE_TYPE_OFFSET: Final[int] = 0
ACTION_FEEDBACK_X_NORM_OFFSET: Final[int] = 1
ACTION_FEEDBACK_Y_NORM_OFFSET: Final[int] = 2
ACTION_FEEDBACK_EXECUTED_SMART_OFFSET: Final[int] = 3
ACTION_FEEDBACK_ANY_EXECUTED_OFFSET: Final[int] = 4
ACTION_FEEDBACK_SCORE_DELTA_OFFSET: Final[int] = 5
ACTION_FEEDBACK_KILL_DELTA_OFFSET: Final[int] = 6
ACTION_FEEDBACK_PENALTY_BIT_OFFSET: Final[int] = 7
ACTION_FEEDBACK_RESERVED_OFFSET: Final[int] = 8

POLICY_ACTION_NO_OP: Final[int] = 0
POLICY_ACTION_LEFT_CLICK: Final[int] = 1
POLICY_ACTION_RIGHT_CLICK: Final[int] = 2
POLICY_ACTION_DIM: Final[int] = 3
SPATIAL_ACTION_IDS: Final[tuple[int, ...]] = (
    POLICY_ACTION_LEFT_CLICK,
    POLICY_ACTION_RIGHT_CLICK,
)
ACTION_REQUIRES_TARGET: Final[tuple[int, ...]] = SPATIAL_ACTION_IDS
SEMANTIC_AVAILABLE_NO_OP_INDEX: Final[int] = POLICY_ACTION_NO_OP
SEMANTIC_AVAILABLE_LEFT_CLICK_INDEX: Final[int] = POLICY_ACTION_LEFT_CLICK
SEMANTIC_AVAILABLE_RIGHT_CLICK_INDEX: Final[int] = POLICY_ACTION_RIGHT_CLICK
# Legacy aliases kept so older helpers can still import the names while the
# learned action space becomes semantic NO_OP / LEFT_CLICK / RIGHT_CLICK.
POLICY_ACTION_SMART: Final[int] = POLICY_ACTION_RIGHT_CLICK
POLICY_ACTION_MOVE: Final[int] = POLICY_ACTION_RIGHT_CLICK
POLICY_ACTION_ATTACK: Final[int] = POLICY_ACTION_RIGHT_CLICK

BRIDGE_ACTION_NO_OP: Final[int] = 0
BRIDGE_ACTION_LEFT_CLICK: Final[int] = 1
BRIDGE_ACTION_RIGHT_CLICK: Final[int] = 2
BRIDGE_ACTION_BOOTSTRAP_SELECT: Final[int] = 3
BRIDGE_ACTION_VOCAB_SIZE: Final[int] = 4
BRIDGE_ACTION_SMART: Final[int] = BRIDGE_ACTION_RIGHT_CLICK
BRIDGE_ACTION_MOVE: Final[int] = BRIDGE_ACTION_RIGHT_CLICK
BRIDGE_ACTION_ATTACK: Final[int] = BRIDGE_ACTION_RIGHT_CLICK

ATTACK_SCREEN_FUNCTION_ID: Final[int] = 12
MOVE_SCREEN_FUNCTION_ID: Final[int] = 13
SMART_SCREEN_FUNCTION_ID: Final[int] = 451
SELECT_ARMY_FUNCTION_ID: Final[int] = 7

ATTACK_AVAILABLE_ACTION_INDEX: Final[int] = DEFEAT_ROACHES_ACTION_IDS.index(
    ATTACK_SCREEN_FUNCTION_ID,
)
MOVE_AVAILABLE_ACTION_INDEX: Final[int] = DEFEAT_ROACHES_ACTION_IDS.index(
    MOVE_SCREEN_FUNCTION_ID,
)
SMART_AVAILABLE_ACTION_INDEX: Final[int] = DEFEAT_ROACHES_ACTION_IDS.index(
    SMART_SCREEN_FUNCTION_ID,
)

SNNState = tuple[torch.Tensor, torch.Tensor]


@dataclass(slots=True)
class ActionSample:
    action_id: int
    x: int
    y: int
    target_index: int | None
    coarse_index: int | None
    fine_index: int | None
    log_prob: float
    value: float
    next_state: SNNState | None


@dataclass(slots=True)
class PolicyInputBatch:
    """
    Frozen observation protocol for Fix 3 hybrid tokenization.

    Shapes follow docs/archive/NEXT_FIXES_PLAN_3.md §3.5 exactly:
      spatial_obs:        [B, 27, 84, 84]
      entity_features:    [B, 24, F_unit]
      entity_mask:        [B, 24]
      selection_features: [B, 20, 7]
      selection_mask:     [B, 20]
      action_feedback_tokens: [B, 1, 9]
      meta_vec:           [B, F_meta]
      state_in:           optional SNN state tuple
    """

    spatial_obs: torch.Tensor
    entity_features: torch.Tensor
    entity_mask: torch.Tensor
    selection_features: torch.Tensor
    selection_mask: torch.Tensor
    meta_vec: torch.Tensor
    state_in: SNNState | None = None
    action_feedback_tokens: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.action_feedback_tokens is None:
            dtype = (
                self.spatial_obs.dtype
                if self.spatial_obs.is_floating_point()
                else torch.float32
            )
            self.action_feedback_tokens = torch.zeros(
                (
                    int(self.spatial_obs.shape[0]),
                    ACTION_FEEDBACK_TOKEN_COUNT,
                    ACTION_FEEDBACK_TOKEN_DIM,
                ),
                dtype=dtype,
                device=self.spatial_obs.device,
            )
        self._validate()

    @property
    def batch_size(self) -> int:
        return int(self.spatial_obs.shape[0])

    @property
    def feature_unit_dim(self) -> int:
        return int(self.entity_features.shape[-1])

    @property
    def meta_dim(self) -> int:
        return int(self.meta_vec.shape[-1])

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "PolicyInputBatch":
        float_kwargs = {}
        mask_kwargs = {}
        if device is not None:
            float_kwargs["device"] = device
            mask_kwargs["device"] = device
        if dtype is not None:
            float_kwargs["dtype"] = dtype

        moved_state = None
        if self.state_in is not None:
            syn, mem = self.state_in
            moved_state = (
                syn.to(**float_kwargs),
                mem.to(**float_kwargs),
            )

        return PolicyInputBatch(
            spatial_obs=self.spatial_obs.to(**float_kwargs),
            entity_features=self.entity_features.to(**float_kwargs),
            entity_mask=self.entity_mask.to(**mask_kwargs),
            selection_features=self.selection_features.to(**float_kwargs),
            selection_mask=self.selection_mask.to(**mask_kwargs),
            meta_vec=self.meta_vec.to(**float_kwargs),
            action_feedback_tokens=self.action_feedback_tokens.to(**float_kwargs),
            state_in=moved_state,
        )

    def detach(self) -> "PolicyInputBatch":
        detached_state = None
        if self.state_in is not None:
            syn, mem = self.state_in
            detached_state = (syn.detach(), mem.detach())

        return PolicyInputBatch(
            spatial_obs=self.spatial_obs.detach(),
            entity_features=self.entity_features.detach(),
            entity_mask=self.entity_mask.detach(),
            selection_features=self.selection_features.detach(),
            selection_mask=self.selection_mask.detach(),
            meta_vec=self.meta_vec.detach(),
            action_feedback_tokens=self.action_feedback_tokens.detach(),
            state_in=detached_state,
        )

    def with_state(self, state_in: SNNState | None) -> "PolicyInputBatch":
        return PolicyInputBatch(
            spatial_obs=self.spatial_obs,
            entity_features=self.entity_features,
            entity_mask=self.entity_mask,
            selection_features=self.selection_features,
            selection_mask=self.selection_mask,
            meta_vec=self.meta_vec,
            action_feedback_tokens=self.action_feedback_tokens,
            state_in=state_in,
        )

    @classmethod
    def stack(cls, batches: list["PolicyInputBatch"]) -> "PolicyInputBatch":
        if not batches:
            raise ValueError("batches must be non-empty")

        states = [batch.state_in for batch in batches]
        if all(state is None for state in states):
            state_in = None
        elif any(state is None for state in states):
            raise ValueError("Either every stacked batch has state_in, or none do")
        else:
            syn_parts = [state[0] for state in states if state is not None]
            mem_parts = [state[1] for state in states if state is not None]
            state_in = (torch.cat(syn_parts, dim=0), torch.cat(mem_parts, dim=0))

        return cls(
            spatial_obs=torch.cat([batch.spatial_obs for batch in batches], dim=0),
            entity_features=torch.cat(
                [batch.entity_features for batch in batches], dim=0,
            ),
            entity_mask=torch.cat([batch.entity_mask for batch in batches], dim=0),
            selection_features=torch.cat(
                [batch.selection_features for batch in batches], dim=0,
            ),
            selection_mask=torch.cat(
                [batch.selection_mask for batch in batches], dim=0,
            ),
            meta_vec=torch.cat([batch.meta_vec for batch in batches], dim=0),
            action_feedback_tokens=torch.cat(
                [batch.action_feedback_tokens for batch in batches],
                dim=0,
            ),
            state_in=state_in,
        )

    def index_select(self, index: torch.Tensor | list[int]) -> "PolicyInputBatch":
        if not isinstance(index, torch.Tensor):
            index = torch.as_tensor(
                index,
                dtype=torch.long,
                device=self.spatial_obs.device,
            )
        else:
            index = index.to(device=self.spatial_obs.device, dtype=torch.long)

        if index.ndim == 0:
            index = index.view(1)
        if index.ndim != 1:
            raise ValueError(
                f"index must be 1D for batch slicing, got shape={tuple(index.shape)}",
            )

        sliced_state = None
        if self.state_in is not None:
            syn, mem = self.state_in
            sliced_state = (
                syn.index_select(0, index),
                mem.index_select(0, index),
            )

        return PolicyInputBatch(
            spatial_obs=self.spatial_obs.index_select(0, index),
            entity_features=self.entity_features.index_select(0, index),
            entity_mask=self.entity_mask.index_select(0, index),
            selection_features=self.selection_features.index_select(0, index),
            selection_mask=self.selection_mask.index_select(0, index),
            meta_vec=self.meta_vec.index_select(0, index),
            action_feedback_tokens=self.action_feedback_tokens.index_select(0, index),
            state_in=sliced_state,
        )

    def _validate(self) -> None:
        self._validate_float_tensor(
            "spatial_obs",
            self.spatial_obs,
            expected_ndim=4,
            expected_tail=SPATIAL_OBS_SHAPE,
        )
        self._validate_float_tensor(
            "entity_features",
            self.entity_features,
            expected_ndim=3,
            expected_tail=(MAX_ENTITY_TOKENS, None),
        )
        self._validate_mask(
            "entity_mask",
            self.entity_mask,
            expected_shape=(self.batch_size, MAX_ENTITY_TOKENS),
        )
        self._validate_float_tensor(
            "selection_features",
            self.selection_features,
            expected_ndim=3,
            expected_tail=(MAX_SELECTION_TOKENS, SELECTION_FEATURE_DIM),
        )
        self._validate_mask(
            "selection_mask",
            self.selection_mask,
            expected_shape=(self.batch_size, MAX_SELECTION_TOKENS),
        )
        self._validate_float_tensor(
            "action_feedback_tokens",
            self.action_feedback_tokens,
            expected_ndim=3,
            expected_tail=(ACTION_FEEDBACK_TOKEN_COUNT, ACTION_FEEDBACK_TOKEN_DIM),
        )
        self._validate_float_tensor(
            "meta_vec",
            self.meta_vec,
            expected_ndim=2,
            expected_tail=(META_VECTOR_DIM,),
        )

        expected_batch = self.batch_size
        for name, tensor in (
            ("entity_features", self.entity_features),
            ("entity_mask", self.entity_mask),
            ("selection_features", self.selection_features),
            ("selection_mask", self.selection_mask),
            ("action_feedback_tokens", self.action_feedback_tokens),
            ("meta_vec", self.meta_vec),
        ):
            if int(tensor.shape[0]) != expected_batch:
                raise ValueError(
                    f"{name} batch dimension must match spatial_obs: "
                    f"{tensor.shape[0]} != {expected_batch}",
                )

        if self.state_in is not None:
            if not isinstance(self.state_in, tuple) or len(self.state_in) != 2:
                raise TypeError("state_in must be a (syn, mem) tensor tuple or None")
            syn, mem = self.state_in
            if not isinstance(syn, torch.Tensor) or not isinstance(mem, torch.Tensor):
                raise TypeError("state_in must contain tensors")
            if syn.shape != mem.shape:
                raise ValueError(
                    f"state_in tensors must share a shape, got {syn.shape} and {mem.shape}",
                )
            if syn.ndim not in (3, 4):
                raise ValueError(
                    "state_in tensors must be rank-3 legacy or rank-4 multi-timescale, "
                    f"got ndim={syn.ndim}",
                )
            if syn.ndim < 1 or int(syn.shape[0]) != expected_batch:
                raise ValueError(
                    f"state_in batch dimension must match spatial_obs: "
                    f"{syn.shape[0] if syn.ndim else 'scalar'} != {expected_batch}",
                )

    @staticmethod
    def _validate_float_tensor(
        name: str,
        tensor: torch.Tensor,
        expected_ndim: int,
        expected_tail: tuple[int | None, ...],
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != expected_ndim:
            raise ValueError(
                f"{name} must have ndim={expected_ndim}, got shape={tuple(tensor.shape)}",
            )
        if not tensor.dtype.is_floating_point:
            raise TypeError(f"{name} must use a floating dtype, got {tensor.dtype}")

        tail = tensor.shape[1:]
        if len(tail) != len(expected_tail):
            raise ValueError(f"{name} has unexpected tail shape {tuple(tail)}")
        for actual, expected in zip(tail, expected_tail):
            if expected is not None and int(actual) != int(expected):
                raise ValueError(
                    f"{name} has unexpected shape {tuple(tensor.shape)}; "
                    f"expected tail {expected_tail}",
                )

    @staticmethod
    def _validate_mask(
        name: str,
        tensor: torch.Tensor,
        expected_shape: tuple[int, int],
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bool:
            raise TypeError(f"{name} must use torch.bool, got {tensor.dtype}")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}",
            )
