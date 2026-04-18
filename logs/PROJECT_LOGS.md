# Project Logs: PySC2 Exploration

## Overview
This project log outlines the step-by-step progression of our PySC2 project, detailing how we set up the environment, modularized components, and prepared the foundation for transitioning to reinforcement learning agents.

---

### **[2025-01-18]**
---
- **Objective**:
  - Set up the PySC2 environment to mimic DeepMind’s original configuration.
  - Ensure compatibility by resolving dependency and versioning issues.
- **Steps Taken**:
  - Created a custom Python environment to align with PySC2 requirements.
  - Downgraded `protobuf` and related dependencies to match DeepMind’s setup.
  - Tested environment setup by running the default `DefeatRoaches` map with a scripted agent.
- **Challenges**:
  - Compatibility issues with newer versions of `protobuf` and `pysc2`.
  - Debugging issues related to `sc2_env` initialization.
- **Results**:
  - Verified that the environment and basic scripted agent were functioning as expected.
- **Next Steps**:
  - Begin modularizing components for clarity and scalability.

---

### **[2025-01-19] Morning**
---
- **Objective**:
  - Modularize the project structure by separating the environment, main logic, and utility functions.
- **Steps Taken**:
  - Created the `setup_env.py` file to encapsulate environment creation logic.
  - Updated `main.py` to call `setup_env.create_env()` instead of defining the environment directly.
  - Separated agent logic into `Agents/scripted_agent.py`.
  - Organized utility wrappers and tools into a `Utility` directory for debugging and preprocessing.
- **Challenges**:
  - Ensuring compatibility between the modularized components (e.g., passing environment flags).
  - Debugging broken imports after reorganization.
- **Results**:
  - Established a clean directory structure:
    ```
    RL-Ler/
    ├── Agents/
    ├── envs/
    ├── Utility/
    ├── main.py
    └── setup_env.py
    ```
  - Successfully ran the environment with modularized components.
- **Next Steps**:
  - Implement a wrapper to inspect and debug the action space.

---

### **[2025-01-19] Afternoon**
---
- **Objective**:
  - Create a wrapper to log the available actions for debugging purposes.
- **Steps Taken**:
  - Added `Utility/available_actions_wrapper.py`.
  - Defined a class `AvailableActionsPrinter` to log newly available actions dynamically.
  - Integrated the wrapper into `setup_env.create_env()` using a conditional flag (`use_action_printer`).
  - Updated `main.py` to allow enabling the wrapper via command-line arguments.
- **Challenges**:
  - Debugging wrapper stacking to ensure it did not interfere with other components.
- **Results**:
  - Verified that available actions are correctly logged during gameplay.
- **Next Steps**:
  - Implement an observation space wrapper to explore and preprocess environment observations.

---

### **Key Learnings**
1. **Environment Setup**:
   - Downgrading dependencies and creating a custom Python environment ensured compatibility with DeepMind’s PySC2 setup.
2. **Modularization**:
   - Separating `main`, environment setup, agents, and utilities created a scalable and maintainable structure.
3. **Wrapper Integration**:
   - Wrappers like `AvailableActionsPrinter` proved invaluable for debugging and understanding the environment dynamics.

---

### **Next Steps**
1. Develop an observation space wrapper to preprocess environment observations.
2. Finalize the input-output design for the agent.
3. Transition from scripted agents to reinforcement learning using PPO or DQN.


---

### **[2025-01-19] Evening**
---
#### **Objective:**
To dynamically explore and document the observation space in PySC2, focusing on the `feature_units`

---
#### **Work Completed:**

**1. Developed Two Inspectors for Observation Parsing:**

- **ObservationInspector:**
  - Purpose: Dynamically explore and log the entire observation space once per episode.
  - Functionality:
    - Logs all available features in the observation space.
    - Identifies spatial features (e.g., `feature_screen`, `feature_minimap`) and their shapes.
    - Captures high-level stats (e.g., `player`, `score_cumulative`, `available_actions`).
  - Example Output:
    ```
    === Observation Space Features ===
    Feature: single_select
     - Shape: (0, 7)
    Feature: feature_screen
     - Shape: (27, 84, 84)
    Feature: feature_units
     - Shape: (13, 46)
    Feature: available_actions
     - Shape: (6,)
    ```

- **FeatureUnitInspector:**
  - Purpose: Focus on the `feature_units` feature for unit-specific data.
  - Functionality:
    - Parses and logs key unit fields: `unit_type`, position (`x, y`), and `health`.
    - Dynamically accesses fields and limits verbosity by logging only the first five units.
  - Example Output:
    ```
    Unit Features Observation Space:
    Dtype of feature_units: int64
    Unit Type: 48, Position: (24, 39), Health: 45
    Unit Type: 48, Position: (24, 32), Health: 45
    Unit Type: 48, Position: (24, 29), Health: 45
    ```

---

**2. Observed and Parsed the Entire Observation Space:**
- Explored all features in the observation space, including:
  - **Spatial Features:**
    - `feature_screen`: `(27, 84, 84)`
    - `feature_minimap`: `(11, 64, 64)`
  - **Tabular Features:**
    - `feature_units`: `(13, 46)`
    - `player`, `score_cumulative`, and other non-spatial data.

- Avoided using pixel-based spatial features (`feature_screen`, `feature_minimap`) to simplify the observation space.

**3. Focused on Unit-Level Observations:**
- Parsed fields in `feature_units` dynamically to understand the available unit-level data.
- Logged critical fields like:
  - `unit_type`: Identifies the type of unit (e.g., Roach, Marine).
  - `x, y`: Spatial position on the map.
  - `health`: Current health of the unit.

---

#### **Challenges and Solutions:**

- **Challenge:**
  - `feature_units` occasionally returned invalid or empty structures, causing runtime errors.
  - Fields like `dtype.names` were sometimes `None`.

- **Solution:**
  - Added robust checks to ensure `feature_units` and its fields were valid before accessing attributes.
  - Logged raw `feature_units` data during debugging to identify the structure and potential issues.

---

#### **Results:**

- Successfully parsed and logged:
  - All features in the observation space.
  - Unit-level data dynamically for `feature_units`.

- Modularized the inspectors to:
  - Dynamically handle both spatial and non-spatial features.
  - Reuse across different maps and scenarios.

---

#### **Next Steps:**

1. **Feature Selection:**
   - Finalize which features to use for the agent based on the logged observation space.
   - Focus on:
     - `feature_units` for unit-level data.
     - `player` and `available_actions` for high-level decision-making.

2. **Preprocessing:**
   - Normalize selected features (e.g., scale positions, health).
   - Aggregate spatial data to simplify input for the agent.

3. **Agent Development:**
   - Transition from scripted logic to reinforcement learning (e.g., PPO or DQN).
   - Define the observation space dynamically based on parsed features.

4. **Documentation:**
   - Document the finalized observation space and preprocessing pipeline.
   - Include logs and insights into decisions made for feature selection.

