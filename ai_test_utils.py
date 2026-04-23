"""
AI-Compatible Testing Utilities

This module provides factory functions to create fake PySC2 observations.
This allows AI agents (or CI/CD pipelines) to test pure-Python RL logic
(like RewardFunctions or ObservationExtractors) without needing access to
the StarCraft II game client or PySC2 dependencies.
"""
import random

class MockFeatureUnit:
    def __init__(self, unit_type, alliance, health, x, y, attack_range=5.0):
        self.unit_type = unit_type
        self.alliance = alliance
        self.health = health
        self.x = x
        self.y = y
        self.attack_range = attack_range

class MockObservation:
    def __init__(self):
        # 1 = Friendly, 4 = Enemy
        self.player = [100, 0, 0, 500]  # e.g., health at index 3
        self.feature_screen = None # Normally a numpy array
        self.feature_units = []

    def add_unit(self, unit_type, alliance, health, x, y):
        self.feature_units.append(
            MockFeatureUnit(unit_type, alliance, health, x, y)
        )

class MockTimeStep:
    def __init__(self):
        self.observation = MockObservation()
        self.reward = 0.0
        self._last = False

    def last(self):
        return self._last

def generate_random_combat_timestep(num_marines=3, num_roaches=3):
    """
    Generates a fake PySC2 TimeStep containing marines and roaches.
    Useful for testing the RewardFunction.
    """
    timestep = MockTimeStep()

    # Add Friendly Marines (Alliance = 1, Type = 48)
    for _ in range(num_marines):
        timestep.observation.add_unit(
            unit_type=48,
            alliance=1,
            health=random.randint(10, 45),
            x=random.randint(0, 83),
            y=random.randint(0, 83)
        )

    # Add Enemy Roaches (Alliance = 4, Type = 51)
    for _ in range(num_roaches):
        timestep.observation.add_unit(
            unit_type=51,
            alliance=4,
            health=random.randint(10, 145),
            x=random.randint(0, 83),
            y=random.randint(0, 83)
        )

    return timestep

if __name__ == "__main__":
    # Quick sanity check
    ts = generate_random_combat_timestep()
    print(f"Generated timestep with {len(ts.observation.feature_units)} units.")
    print(f"Is last step? {ts.last()}")
