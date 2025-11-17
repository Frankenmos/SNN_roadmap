# Project Overview: PySC2 Environment Exploration and Wrappers

This document serves as a comprehensive overview of the project to explore, debug, and extend the PySC2 environment, with a focus on modularity and custom agent design.

## Environment Setup

### 1. **System Preparation**

To ensure compatibility with PySC2, we matched the environment configuration DeepMind used during the original library development:

- **Python Interpreter:** A custom Python environment was created to ensure exact compatibility.
- **Library Downgrades:** Downgraded to specific versions of `protobuf` and related dependencies to maintain consistency with the PySC2 environment.

### 2. **Dependencies**

Installed the following dependencies:

- `pysc2`
- `numpy`
- `absl-py`
- Additional tools for debugging and wrappers as needed.

### 3. **Environment Directory Structure**

We organized the project into modular components to enhance clarity, simplify debugging, and support scalable development. This structure allows for easier management of individual functionalities, such as agents, environment setup, and utility wrappers, ensuring modularity and maintainability as the project grows.

```
RL-Ler/
├── Agents/
│   ├── ML_agent.py
│   └── scripted_agent.py
├── envs/
│   └── setup_env.py
├── Utility/
│   ├── available_actions_wrapper.py
│   ├── valid_actions.py
│   └── obs_wrapper.py
├── main.py
├── README.md
└── STARCRAFT 2 ML.ipynb
```

- ``**Agents**`` Contains scripted agents and placeholders for machine learning agents.
- ``**envs**`` Environment setup and configuration files.
- ``**Utility**`` Wrappers and utility scripts for debugging and environment extension.

### 4. **PySC2 Configuration**

- Map: `DefeatRoaches`
- Agent: `Terran` (scripted logic)
- Opponent: Zerg bot with `easy` difficulty
- Customizations:
  - Step multiplier: 20
  - Feature dimensions: 84x84 for screen and minimap.

---

## Current Features and Wrappers

### 1. **Available Actions Printer**

- **Purpose:** Dynamically logs all available actions in the environment for debugging and exploration.
- **Implementation:**
  - Defined in `available_actions_wrapper.py`.
  - Tracks and prints newly available actions during gameplay.
- **Usage:** Enabled via the `use_action_printer` flag in `main.py`.

```bash
python main.py --use_action_printer=True
```

### 2. **Observation Space Inspector** (In Progress)

- **Purpose:** Explore and debug the observation space returned by the environment.
- **Planned Features:**
  - Log keys and shapes of all components in the observation dictionary.
  - Provide insights into player stats, feature screens, and available actions.
- **Integration:** Will be stacked with the Available Actions Printer to enhance modular debugging by combining insights from both available actions and observation components, ensuring a comprehensive understanding of the environment dynamics.

---

## Transition Plan: From Scripted to ML Agents

After finalizing the exploration of the observation space:

1. **Scripted Agent Improvements:**
   - Ensure the `DefeatRoaches` agent is robust and performs well within the environment.
2. **Machine Learning Agent:**
   - Transition to reinforcement learning using frameworks like TensorFlow or PyTorch, chosen for their robust ecosystem, ease of implementation, and extensive support for deep reinforcement learning algorithms.
   - Train agents to outperform scripted logic.

---

## Challenges Faced

1. **API Quirks:**
   - Adapting to the structure and limitations of the PySC2 library.
   - Debugging step-related issues when stacking wrappers.
2. **Compatibility Issues:**
   - Ensuring the library dependencies match the original DeepMind environment setup.
3. **Wrapper Interactions:**
   - Managing dependencies and preserving functionality across multiple wrappers.

---

## Next Steps

1. **Complete Observation Space Wrapper:**
   - Verify and log all components of the observation space.
2. **Stack Multiple Wrappers:**
   - Combine available actions and observation inspectors.
3. **Prepare for RL Agent Integration:**
   - Define reward functions and action policies for training.

---

### Notes:

This project has evolved from a single script into a modular system, incorporating real-world programming workflows to enhance scalability and maintainability. The current design ensures scalability and serves as a strong foundation for future ML-based exploration.

