
class ObservationInspector:
    def __init__(self):
        self.observation_printed = False  # To avoid redundant output

    def print_all_features(self, obs):
        """
        Print all available features in the observation space.
        """
        if not self.observation_printed:
            print("\n=== Observation Space Features ===")
            for key, value in vars(obs.observation).items():
                print(f"Feature: {key}")
                if isinstance(value, (list, tuple)):
                    print(f" - Length: {len(value)}")
                elif hasattr(value, "shape"):
                    print(f" - Shape: {value.shape}")
                elif value is not None:
                    print(f" - Value: {value}")
            self.observation_printed = True  # Avoid repeating this output

    def reset(self):
        """Reset the inspector for new episodes."""
        self.observation_printed = False


class FeatureUnitInspector:
    """
    Helper class to log and monitor the dynamic state of `feature_units` in the observation space.

    Tracks and prints details like unit type, position, and health for each step during the environment run.
    This is useful for debugging and understanding the evolution of unit-level data over time.

    Note: This class is designed for continuous logging and does not suppress repeated outputs.
    """
    def __init__(self):
        self.unit_features_printed = False  # Track whether unit features have been printed

    def print_unit_features(self, obs):
        """
        Print the structure of the unit features from an observation object.
        """
        if hasattr(obs.observation, "feature_units"):
            feature_units = obs.observation.feature_units
            if feature_units is not None and len(feature_units) > 0:
                if not self.unit_features_printed:
                    print("\nUnit Features Observation Space:")
                    print(f"Dtype of feature_units: {feature_units.dtype}")  # Debug field names and types

                # Access fields dynamically
                for unit in feature_units[:5]:  # Limit to first 5 units for brevity
                    print(f"Unit Type: {unit['unit_type']}, Position: ({unit['x']}, {unit['y']}), Health: {unit['health']}")

                self.unit_features_printed = True  # Avoid repeating this for every step
            else:
                print("No units found in the current observation.")
        else:
            print("Unit features are not available. Ensure `use_feature_units=True` is set in the environment.")

    def reset(self):
        """Reset the inspector (e.g., for a new episode)."""
        self.unit_features_printed = False

