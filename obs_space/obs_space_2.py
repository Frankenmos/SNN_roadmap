import random

import numpy as np

# ---- PySC2 colors.py fix for Python 3.11+ (random.shuffle signature) ----
# The same monkey-patch lives at PPO_CNN_agent.py:8-15. We replicate it
# here so that any path that imports obs_space_2 WITHOUT going through
# the training entrypoint (e.g. unit tests, smoke scripts) is still safe
# — otherwise the `from pysc2.lib import features` below triggers
# SCREEN_FEATURES construction → colors.shuffled_hue → the broken
# random.shuffle(palette, lambda: 0.5) call.
from pysc2.lib import colors as _colors


def _shuffled_hue_fixed(scale):
    palette = list(_colors.smooth_hue_palette(scale))
    random_keys = [random.random() for _ in palette]
    palette = [x for _, x in sorted(zip(random_keys, palette))]
    return np.array(palette)


_colors.shuffled_hue = _shuffled_hue_fixed
# -------------------------------------------------------------------------

from pysc2.lib import features  # noqa: E402,F401  (kept for downstream consumers)
import torch  # noqa: E402

from PPO_CNN.policy_input import (
    AGENT_LAST_ACTION_DIM,
    AGENT_LAST_ACTION_OFFSET,
    BRIDGE_ACTION_NO_OP,
    CURATED_FEATURE_UNIT_FIELDS,
    DEFEAT_ROACHES_ACTION_IDS,
    MAX_ENTITY_TOKENS,
    MAX_SELECTION_TOKENS,
    META_AVAILABLE_ACTION_OFFSET,
    META_LAST_ACTION_INDEX_OFFSET,
    META_PLAYER_FEATURE_DIM,
    META_VECTOR_DIM,
    NO_ACTION_SENTINEL_INDEX,
    PolicyInputBatch,
    SELECTION_FEATURE_NAMES,
    SPATIAL_OBS_SHAPE,
    UNKNOWN_LAST_ACTION_INDEX,
)

_PLAYER_FRIENDLY = 1
_FEATURE_UNIT_INDEX = {
    name: int(field.value)
    for name, field in features.FeatureUnit.__members__.items()
}
_UNIT_LAYER_INDEX = {
    name: int(field.value)
    for name, field in features.UnitLayer.__members__.items()
}
_LAST_ACTION_TO_INDEX = {
    action_id: idx + 1 for idx, action_id in enumerate(DEFEAT_ROACHES_ACTION_IDS)
}


def _validate_index_fields(field_names, index_map, index_name):
    missing = [name for name in field_names if name not in index_map]
    if missing:
        raise ValueError(
            f"Unknown {index_name} field(s): {', '.join(sorted(missing))}",
        )


class RunningFeatureNormalizer:
    def __init__(
        self,
        field_names,
        normalized_fields,
        min_count_for_normalize=32.0,
        min_std=1.0e-2,
        min_variance=None,
        output_clip=10.0,
    ):
        self.field_names = tuple(field_names)
        self.normalized_fields = tuple(normalized_fields)
        self.normalized_indices = [
            self.field_names.index(name) for name in self.normalized_fields
        ]
        self.min_count_for_normalize = float(min_count_for_normalize)
        self.min_std = float(min_std)
        if min_variance is None:
            min_variance = self.min_std ** 2
        self.min_variance = float(min_variance)
        self.output_clip = float(output_clip)
        size = len(self.normalized_indices)
        self.count = 0.0
        self.mean = np.zeros(size, dtype=np.float64)
        self.m2 = np.zeros(size, dtype=np.float64)

    def update(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0 or not self.normalized_indices:
            return
        if values.ndim == 1:
            values = values.reshape(1, -1)

        selected = values[:, self.normalized_indices].astype(np.float64)
        batch_count = float(selected.shape[0])
        batch_mean = selected.mean(axis=0)
        batch_m2 = ((selected - batch_mean) ** 2).sum(axis=0)
        if self.count == 0.0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_m2 + (delta ** 2) * self.count * batch_count / total
        self.count = total

    def normalize(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0 or not self.normalized_indices:
            return values
        squeeze_back = False
        if values.ndim == 1:
            values = values.reshape(1, -1)
            squeeze_back = True
        out = values.copy()
        if self.count < self.min_count_for_normalize:
            return out[0] if squeeze_back else out

        denom = np.maximum(self.count - 1.0, 1.0)
        variance = np.maximum(self.m2 / denom, 0.0)
        std = np.sqrt(variance)
        active = np.isfinite(std) & (std >= self.min_std) & (
            variance >= self.min_variance
        )
        if not np.any(active):
            return out[0] if squeeze_back else out

        selected = out[:, self.normalized_indices]
        mean = self.mean.astype(np.float32, copy=False)
        std_safe = np.maximum(std, self.min_std).astype(np.float32, copy=False)
        normalized = (selected - mean) / std_safe
        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=self.output_clip,
            neginf=-self.output_clip,
        )
        normalized = np.clip(normalized, -self.output_clip, self.output_clip)
        selected[:, active] = normalized[:, active]
        out[:, self.normalized_indices] = selected
        return out[0] if squeeze_back else out

    def state_dict(self):
        return {
            "count": float(self.count),
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
        }

    def load_state_dict(self, state):
        self.count = float(state.get("count", 0.0))
        self.mean = np.asarray(state.get("mean", self.mean.tolist()), dtype=np.float64)
        self.m2 = np.asarray(state.get("m2", self.m2.tolist()), dtype=np.float64)


def get_friendly_health(obs):
    """Sum of health across all friendly units."""
    feature_units = getattr(obs.observation, "feature_units", None)
    if feature_units is None or len(feature_units) == 0:
        return 0.0
    return float(sum(
        u.health for u in feature_units
        if getattr(u, "alliance", 0) == _PLAYER_FRIENDLY
    ))


class ObservationExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _validate_index_fields(
            CURATED_FEATURE_UNIT_FIELDS,
            _FEATURE_UNIT_INDEX,
            "FeatureUnit",
        )
        _validate_index_fields(
            SELECTION_FEATURE_NAMES,
            _UNIT_LAYER_INDEX,
            "UnitLayer",
        )
        self.entity_normalizer = RunningFeatureNormalizer(
            CURATED_FEATURE_UNIT_FIELDS,
            normalized_fields=(
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
                "assigned_harvesters",
                "ideal_harvesters",
            ),
        )
        self.selection_normalizer = RunningFeatureNormalizer(
            SELECTION_FEATURE_NAMES,
            normalized_fields=(
                "health",
                "shields",
                "energy",
                "transport_slots_taken",
                "build_progress",
            ),
        )

    def extract_observation(self, obs, update_stats=True, last_action_token=None):
        feature_screen = getattr(obs.observation, "feature_screen", None)
        if feature_screen is not None and getattr(feature_screen, "size", 0) > 0:
            spatial_obs = torch.as_tensor(
                np.asarray(feature_screen) / 255.0,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
        else:
            spatial_obs = torch.zeros(
                (1, *SPATIAL_OBS_SHAPE),
                dtype=torch.float32,
                device=self.device,
            )

        entity_rows = self._extract_entity_rows(obs)
        if update_stats:
            self.entity_normalizer.update(entity_rows)
        entity_rows = self.entity_normalizer.normalize(entity_rows)

        selection_rows = self._extract_selection_rows(obs)
        if update_stats:
            self.selection_normalizer.update(selection_rows)
        selection_rows = self.selection_normalizer.normalize(selection_rows)

        entity_features, entity_mask = self._pad_rows(
            entity_rows,
            max_rows=MAX_ENTITY_TOKENS,
            width=len(CURATED_FEATURE_UNIT_FIELDS),
        )
        selection_features, selection_mask = self._pad_rows(
            selection_rows,
            max_rows=MAX_SELECTION_TOKENS,
            width=len(SELECTION_FEATURE_NAMES),
        )
        meta_vec = self._extract_meta_vector(
            obs,
            last_action_token=last_action_token,
        )

        return PolicyInputBatch(
            spatial_obs=spatial_obs,
            entity_features=entity_features.unsqueeze(0),
            entity_mask=entity_mask.unsqueeze(0),
            selection_features=selection_features.unsqueeze(0),
            selection_mask=selection_mask.unsqueeze(0),
            meta_vec=meta_vec.unsqueeze(0),
        )

    def peek_observation(self, obs, last_action_token=None):
        return self.extract_observation(
            obs,
            update_stats=False,
            last_action_token=last_action_token,
        )

    def _extract_entity_rows(self, obs):
        feature_units = getattr(obs.observation, "feature_units", None)
        numeric = self._coerce_numeric_rows(feature_units)
        if numeric is not None:
            return self._project_numeric_rows(
                numeric,
                field_names=CURATED_FEATURE_UNIT_FIELDS,
                index_map=_FEATURE_UNIT_INDEX,
            )
        return self._project_object_rows(
            feature_units,
            field_names=CURATED_FEATURE_UNIT_FIELDS,
        )

    def _extract_selection_rows(self, obs):
        selection = getattr(obs.observation, "multi_select", None)
        if selection is None or len(selection) == 0:
            selection = getattr(obs.observation, "single_select", None)

        numeric = self._coerce_numeric_rows(selection)
        if numeric is not None:
            return self._project_numeric_rows(
                numeric,
                field_names=SELECTION_FEATURE_NAMES,
                index_map=_UNIT_LAYER_INDEX,
            )
        return self._project_object_rows(
            selection,
            field_names=SELECTION_FEATURE_NAMES,
        )

    def _extract_meta_vector(self, obs, last_action_token=None):
        player = getattr(obs.observation, "player", None)
        if player is None:
            player_vec = np.zeros(META_PLAYER_FEATURE_DIM, dtype=np.float32)
        else:
            player_vec = np.asarray(player, dtype=np.float32).reshape(-1)
            if player_vec.size < META_PLAYER_FEATURE_DIM:
                player_vec = np.pad(
                    player_vec,
                    (0, META_PLAYER_FEATURE_DIM - player_vec.size),
                    mode="constant",
                )
            else:
                player_vec = player_vec[:META_PLAYER_FEATURE_DIM]

        available_actions = getattr(obs.observation, "available_actions", None)
        available_set = (
            {int(action_id) for action_id in list(available_actions)}
            if available_actions is not None
            else set()
        )
        available_mask = np.asarray(
            [
                1.0 if action_id in available_set else 0.0
                for action_id in DEFEAT_ROACHES_ACTION_IDS
            ],
            dtype=np.float32,
        )

        last_actions = getattr(obs.observation, "last_actions", None)
        if last_actions is None or len(last_actions) == 0:
            last_action_index = float(NO_ACTION_SENTINEL_INDEX)
        else:
            raw_last_action = int(list(last_actions)[0])
            last_action_index = float(
                _LAST_ACTION_TO_INDEX.get(raw_last_action, UNKNOWN_LAST_ACTION_INDEX),
            )

        agent_last = self._normalize_last_action_token(last_action_token)
        full = np.concatenate(
            (
                player_vec,
                available_mask,
                np.asarray([last_action_index], dtype=np.float32),
                agent_last,
            ),
        )
        return torch.as_tensor(
            full,
            dtype=torch.float32,
            device=self.device,
        )

    def _normalize_last_action_token(self, token):
        if token is None:
            token = np.asarray(
                [BRIDGE_ACTION_NO_OP, 0, 0, 0],
                dtype=np.float32,
            )
        else:
            token = np.asarray(token, dtype=np.float32).reshape(-1)
        if token.size < AGENT_LAST_ACTION_DIM:
            token = np.pad(
                token,
                (0, AGENT_LAST_ACTION_DIM - token.size),
                mode="constant",
            )
        else:
            token = token[:AGENT_LAST_ACTION_DIM]

        max_coord = float(SPATIAL_OBS_SHAPE[-1] - 1)
        out = token.astype(np.float32, copy=True)
        out[0] = float(max(0.0, out[0]))
        out[1] = float(np.clip(out[1], 0.0, max_coord) / max_coord)
        out[2] = float(np.clip(out[2], 0.0, max_coord) / max_coord)
        out[3] = float(out[3])
        return out

    def _coerce_numeric_rows(self, rows):
        if rows is None:
            return np.zeros((0, 0), dtype=np.float32)
        if isinstance(rows, torch.Tensor):
            arr = rows.detach().cpu().numpy()
        else:
            arr = np.asarray(rows)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.dtype != object:
            return arr.astype(np.float32, copy=False)
        return None

    def _project_numeric_rows(self, rows, field_names, index_map):
        if rows is None or rows.size == 0:
            return np.zeros((0, len(field_names)), dtype=np.float32)
        projected = np.zeros((rows.shape[0], len(field_names)), dtype=np.float32)
        for idx, field_name in enumerate(field_names):
            source_idx = index_map[field_name]
            if source_idx >= rows.shape[1]:
                raise ValueError(
                    f"Numeric rows for '{field_name}' expected column {source_idx} "
                    f"but width is only {rows.shape[1]}",
                )
            projected[:, idx] = rows[:, source_idx]
        return projected

    def _project_object_rows(self, rows, field_names):
        if rows is None or len(rows) == 0:
            return np.zeros((0, len(field_names)), dtype=np.float32)
        projected = np.zeros((len(rows), len(field_names)), dtype=np.float32)
        for row_idx, row in enumerate(list(rows)):
            for field_idx, field_name in enumerate(field_names):
                value = getattr(row, field_name, 0.0)
                projected[row_idx, field_idx] = float(
                    0.0 if value is None else value,
                )
        return projected

    def _pad_rows(self, rows, max_rows, width):
        rows = np.asarray(rows, dtype=np.float32)
        features = torch.zeros(
            (max_rows, width),
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.zeros((max_rows,), dtype=torch.bool, device=self.device)
        if rows.size == 0:
            return features, mask

        actual_rows = min(int(rows.shape[0]), int(max_rows))
        features[:actual_rows] = torch.as_tensor(
            rows[:actual_rows],
            dtype=torch.float32,
            device=self.device,
        )
        mask[:actual_rows] = True
        return features, mask

    def get_observation_dimensions(self, obs):
        del obs
        return SPATIAL_OBS_SHAPE, META_VECTOR_DIM

    def reset(self):
        return None

    def state_dict(self):
        return {
            "entity_normalizer": self.entity_normalizer.state_dict(),
            "selection_normalizer": self.selection_normalizer.state_dict(),
        }

    def load_state_dict(self, state):
        if not state:
            return
        entity_state = state.get("entity_normalizer")
        if entity_state:
            self.entity_normalizer.load_state_dict(entity_state)
        selection_state = state.get("selection_normalizer")
        if selection_state:
            self.selection_normalizer.load_state_dict(selection_state)
