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

_PLAYER_FRIENDLY = 1
_PLAYER_ENEMY = 4


def get_friendly_health(obs):
    """Sum of health across all friendly units.

    The PySC2 `player` observation array does NOT contain health:
    index 0 is `player_id` (a constant within an episode) and index 3
    is `food_used`. Reading either as "agent health" gives either a
    constant or a proxy for unit count. Aggregate from feature_units
    instead so reward shaping and vector features agree on the same
    definition of health.
    """
    feature_units = getattr(obs.observation, "feature_units", None)
    if feature_units is None or len(feature_units) == 0:
        return 0.0
    return float(sum(
        u.health for u in feature_units
        if u.alliance == _PLAYER_FRIENDLY
    ))


class ObservationExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unit_features_printed = True
        self.previous_position = None
        self.history_length = 3
        self.history = []

    def extract_observation(self, obs):
        """Ensure spatial observations are 3D [channels, height, width]"""
        feature_screen = getattr(obs.observation, "feature_screen", None)
        if feature_screen is not None and feature_screen.size > 0:
            spatial_obs = torch.as_tensor(feature_screen / 255.0, dtype=torch.float32, device=self.device)
        else:
            spatial_obs = torch.zeros((27, 84, 84), dtype=torch.float32, device=self.device)

        # Vector observations as 1D
        vector_obs = self._extract_vector_features(obs).flatten()  # Ensure 1D
        return spatial_obs, vector_obs

    def peek_observation(self, obs):
        """Extract the next observation without mutating history state."""
        saved_prev_position = self.previous_position
        saved_history = [list(item) for item in self.history]
        spatial_obs, vector_obs = self.extract_observation(obs)
        self.previous_position = saved_prev_position
        self.history = saved_history
        return spatial_obs, vector_obs

    def _extract_vector_features(self, obs):
        """Modified to handle tensor comparisons safely"""
        observation_vector = []

        # Agent data. NOTE: obs.observation.player[3] was previously
        # misread as "agent health" — it's actually `food_used` per
        # PySC2's Player enum. Health is aggregated from feature_units
        # via the shared helper so the vector feature and the reward's
        # health penalty stay on the same definition.
        agent_health = get_friendly_health(obs)
        agent_position = self.get_agent_position(obs)

        # Handle None comparison safely
        if self.previous_position is None:
            self.previous_position = agent_position

        # Store positions as floats
        observation_vector.extend([
            float(agent_health),
            float(agent_position[0]),
            float(agent_position[1])
        ])

        # Enemy data with tensor-safe checks
        feature_units = getattr(obs.observation, "feature_units", None)
        if feature_units is not None and len(feature_units) > 0:  # Check list length, not tensor
            enemies = [unit for unit in feature_units if unit.alliance == _PLAYER_ENEMY]
            if enemies:  # Check list existence, not tensor
                nearest_enemy = min(enemies, key=lambda e: self.compute_distance(agent_position, (e.x, e.y)))
                enemy_health = nearest_enemy.health
                enemy_pos = (nearest_enemy.x, nearest_enemy.y)

                distance = self.compute_distance(agent_position, enemy_pos)
                attack_range = self.get_auto_attack_range(nearest_enemy)
                rel_distance = abs(distance - attack_range)
                tangent_dist = self.compute_tangent_distance(agent_position, *enemy_pos, attack_range)

                velocity = self.compute_velocity(agent_position)
                direction = 0.0
                if velocity is not None:  # Handle None safely
                    rel_pos = (enemy_pos[0] - agent_position[0], enemy_pos[1] - agent_position[1])
                    direction = self.compute_scalar_direction(velocity, rel_pos)

                observation_vector.extend([
                    float(enemy_health), float(enemy_pos[0]), float(enemy_pos[1]),
                    float(distance), float(attack_range), float(rel_distance),
                    float(tangent_dist), float(direction)
                ])
            else:
                observation_vector.extend([0.0] * 8)
        else:
            observation_vector.extend([0.0] * 8)

        # Update position history
        self.previous_position = agent_position
        self._update_history(observation_vector)

        # Convert to tensor with explicit float casting
        vector_tensor = torch.zeros(100, device=self.device, dtype=torch.float32)
        valid_length = min(100, len(self.history) * len(observation_vector))
        vector_tensor[:valid_length] = torch.tensor(
            self.history,
            device=self.device,
            dtype=torch.float32
        ).view(-1)[:valid_length]

        return vector_tensor

    def _update_history(self, current_obs):
        """Maintain observation history with explicit list handling"""
        self.history.append(current_obs)
        if len(self.history) > self.history_length:
            self.history.pop(0)

    # Keep other utility methods unchanged but ensure float returns
    def compute_distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def compute_velocity(self, current_pos):
        if self.previous_position is None:
            return None
        delta = np.array([
            current_pos[0] - self.previous_position[0],
            current_pos[1] - self.previous_position[1]
        ])
        return delta if np.linalg.norm(delta) > 1e-8 else None

    def find_units(self, player_relative, condition):
        """Added missing method from original implementation"""
        # Check if input is a tensor and convert to numpy
        if isinstance(player_relative, torch.Tensor):
            player_relative = player_relative.cpu().numpy()

        unit_positions = np.argwhere(player_relative == condition)
        return [(int(pos[1]), int(pos[0])) for pos in unit_positions] if unit_positions.size > 0 else []

    # Keep all original utility methods
    def get_auto_attack_range(self, unit):
        return getattr(unit, 'attack_range', 5.0)

    def compute_tangent_distance(self, agent_pos, enemy_x, enemy_y, attack_range):
        distance = self.compute_distance(agent_pos, (enemy_x, enemy_y))
        return max(0.0, distance - attack_range)

    def compute_scalar_direction(self, velocity, relative_pos):
        vel = np.array(velocity, dtype=np.float32)
        rel = np.array(relative_pos, dtype=np.float32)
        return np.dot(vel, rel) / (np.linalg.norm(vel) * np.linalg.norm(rel) + 1e-8)

    def get_agent_position(self, obs):
        # Handle different observation formats
        feature_units = getattr(obs.observation, 'feature_units', None)
        if feature_units is None:
            return (0, 0)

        # Add type conversion safety
        if isinstance(feature_units, torch.Tensor):
            feature_units = feature_units.cpu().numpy()
        # Rest of your logic...
        if feature_units is not None and len(feature_units) > 0:
            friendly = [u for u in feature_units
                        if u.alliance == _PLAYER_FRIENDLY and u.unit_type in [48, 51]]
            if friendly:
                return (friendly[0].x, friendly[0].y)
        return (0, 0)

    def get_observation_dimensions(self, obs):
        return (27, 84, 84), 100  # Hardcode based on known structure

    def reset(self):
        self.history = []
        self.previous_position = None
