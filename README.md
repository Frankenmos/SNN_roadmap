# PySC2 Spiking PPO Agent (DefeatRoaches)

Welcome to the **PySC2 Spiking PPO Agent** repository. This project implements a modular reinforcement learning (RL) stack on top of the PySC2 (StarCraft II Learning Environment), specifically targeting the `DefeatRoaches` mini-game.

It combines state-of-the-art Spiking Neural Networks (SNNs) with Proximal Policy Optimization (PPO) to explore how temporal, event-driven architectures perform in a complex, multi-agent micro-management scenario.

---

## 🎯 Project Goals

- **Explore SNNs in RL:** Test the viability of Spiking Neural Networks (`snntorch`) as the policy backbone for a PPO agent in a continuous, complex environment.
- **Modular Architecture:** Maintain a clean separation between the PySC2 environment wrappers, the observation extraction, the reward shaping, and the core PPO math.
- **Robust Diagnostics:** Move beyond basic CLI logs by utilizing a SQLite-backed logging system and an interactive Streamlit dashboard to monitor policy collapse, entropy asymmetry, and reward distribution.

---

## 🏗 Architecture

The repository is structured into several logical components:

### 1. The Environment (`envs/`)
- `setup_env.py`: Encapsulates the PySC2 environment initialization, ensuring the correct map (`DefeatRoaches`), step multipliers, and observation dimensions are loaded.

### 2. Observation & Action Space (`obs_space/`, `action_space/`)
- `ObservationExtractor`: Converts the raw PySC2 observations (spatial screen features and tabular unit features) into stable, normalized tensors (a 3D spatial tensor and a 1D vector history).
- `ActionSpace`: Maps the network's continuous and discrete outputs (e.g., move coordinates, attack targets) into PySC2's specific `FunctionCall` formats.

### 3. The Agent & Policy (`PPO_CNN/`)
- **`DefeatRoaches` (Agent):** The orchestrator that binds the observation extractor, the reward function, and the PPO update logic.
- **`PolicyNetwork`:** A custom neural network combining CNNs (for spatial features), Linear layers (for vector features), and `snntorch` Leaky Integrate-and-Fire (LIF) neurons for temporal processing. It utilizes a Spiking Self-Attention mechanism to process the multi-modal inputs.
- **`PPO`:** The core implementation of Proximal Policy Optimization, adapted to handle the unique constraints of SNNs (like maintaining membrane potential state across rollouts) and multi-head discrete/continuous action spaces.

### 4. Utilities & Analysis (`Utility/`, Root)
- `results.py` / `dashboard.py`: Tools to parse the `training_logs.db`, detect plateaus, and visualize training metrics like KL divergence, clip fraction, and action entropy.
- `config.yaml`: Centralized configuration for hyperparameters, network dimensions, and training settings.

---

## 🚀 Installation & Setup

This project uses a standard `requirements.txt` file. We recommend using Python 3.10 - 3.12 within a virtual environment.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-name>

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

*(Note: PySC2 requires a working installation of StarCraft II. Please refer to the [PySC2 documentation](https://github.com/google-deepmind/pysc2) for instructions on installing the game client and maps.)*

---

## 🏃‍♂️ Running the Project

### 1. Training the Agent
To start training the agent from scratch (or resume from the latest checkpoint if configured):

```bash
python PPO_CNN_run.py
```

### 2. Resuming from the Best Checkpoint
If your policy collapses and you want to resume training from the historical peak:

```bash
python resume_from_best.py
```

### 3. Monitoring Training
To generate a comprehensive markdown report and static plots of the training progress:

```bash
python results.py --run-name <your_run_name> --report
```

To launch the interactive Streamlit dashboard:

```bash
streamlit run dashboard.py
```

---

## 🧪 Testing

The repository includes a suite of unit tests located in the `tests/` directory. Because the environment requires heavy dependencies (`torch`, `pysc2`, `snntorch`), running the tests in a minimal environment requires mocking.

```bash
pytest -v tests/
```

---

## 📝 Planning & Future Work

We actively maintain our development roadmap and architectural debugging logs in the repository:
- `NEXT_FIXES_PLAN.md`: Detailed Socratic reasoning for upcoming architectural fixes (e.g., SNN state mismatch, Entropy asymmetry).
- `plan.md`: High-level roadmap for reward redesign, observation space cleanup, and loop reliability.
- `PROJECT_LOGS.md` / `SESSION_LOG_*.md`: Historical context on the project's evolution.