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
- `ObservationExtractor`: Converts raw PySC2 observations into the current hybrid policy input: spatial screen tensor, padded entity tokens, padded selection tokens, and a 32-dim meta vector that now includes the executed-action bridge token.
- `ActionSpace`: Maps the current 2-way policy (`NO_OP`, `SMART`) into explicit PySC2 `FunctionCall`s (`no_op`, `Smart_screen`). This keeps the learned click primitive honest to in-game right-click semantics instead of pretending `MOVE` and `ATTACK` are cleanly separated when `Attack_screen` can still behave like attack-move on empty ground.

### 3. The Agent & Policy (`agent_core/`)
- **`DefeatRoaches` (Agent):** The orchestrator that binds the observation extractor, the reward function, and the PPO update logic.
- **`PolicyNetwork`:** A hybrid CNN + token encoder policy with spiking attention and dual-timescale token memory. Spatial features become pooled spatial tokens with explicit 2D positional encoding; unit, selection, and meta context become additional token groups before attention and the fast/slow temporal SNN pathways. The action/value heads still read a global latent, but the `SMART` click head now keeps a structured spatial branch alive for localization instead of predicting coordinates only from a pooled latent.
- **`PPO`:** The current PPO path includes Stage-1 TBPTT with ordered chunk replay, helper-step masking, packed replay, and the SDPA attention fast path.

Canonical entrypoints are now `train.py`, `eval.py`, `agent.py`, and
the `agent_core/` package. The root-level `PPO_CNN_*` launchers remain
as compatibility wrappers for old commands, but the `PPO_CNN/` package
itself is now treated as a legacy snapshot of the older architecture
rather than a live shim into `agent_core/`.

### 4. Utilities & Analysis (`Utility/`, `tools/analysis/`, Root)
- `tools/analysis/`: Home for the actual analysis implementations.
- Root launchers like `results.py`, `dashboard.py`, `analyze_run.py`, and `analyze_pth.py` remain as thin wrappers so the old commands still work.
- `results.py` / `dashboard.py`: Tools to parse `training_logs.db`, inspect `ppo_updates` / `eval_runs`, detect plateaus, and visualize training metrics like KL divergence, clip fraction, and action entropy.
- `analyze_eval_trace.py`: Sidecar eval-trace analyzer for `.pt` episode traces, with per-step summaries, spatial panels, and optional conv activation maps from a chosen checkpoint.
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
python train.py
```

### 2. Resuming from the Best Checkpoint
If your policy collapses and you want to resume training from the historical peak:

```bash
python resume_from_best.py
```

### 3. Monitoring Training
To generate the static analysis bundle for a run:

```bash
python results.py --run-name <your_run_name> --report
```

To also export the high-signal panels as separate PNGs for easy sharing
back into text-only workflows:

<<<<<<< ours
```bash
python results.py --run-name <your_run_name> --report --aismart
```

This writes the usual files under `analysis_results/<run_name>/`, including:
- `training_progress.png`
- `reward_components.png`
- `win_rate.png`
- `training_metrics.csv`
- `instability_report.txt`
=======
Install the Python dependencies before running training scripts or tests:

```bash
pip install -r requirements.txt
```

This covers the core stack (`numpy`, `torch`), PySC2 tooling (`pysc2`, `absl-py`), and utilities used by the helpers (`pyyaml`, `matplotlib`).
>>>>>>> theirs

When `--aismart` is enabled it also writes:
- `analysis_results/<run_name>/ai_friendly_results/`
  focused dashboard-style PNG panels such as reward trajectory, reward
  efficiency, action entropy, action mix, phase-of-episode action mix,
  TBPTT/speed, eval split, and other high-signal views depending on the
  data available in the DB

To launch the interactive Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The root launchers stay in place for convenience, but the real implementations now live in `tools/analysis/`.

The dashboard can inspect either:
- a local run directly from `models/<run_name>/training_logs.db`
- an uploaded `training_logs.db`
- an optional uploaded or local `.pth` checkpoint alongside the DB

The current dashboard is worth using for this branch because it now exposes:
- **Overview:** reward trajectory, episode length, oscillation score, and reward-per-step efficiency
- **Policy:** whole-run action mix, early/mid/late episode action mix, action entropy, action heatmap, and move-target heatmap
- **PPO / Eval:** PPO metrics, TBPTT/speed metrics, deterministic vs stochastic eval curves, and eval reward gap
- **Reward Shaping:** reward-component trends and distributions
- **Checkpoint:** tensor inspection plus checkpoint metadata, extractor normalizer stats, and learned SNN `alpha` / `beta` parameters

For quick summaries and checkpoint inspection:

```bash
python analyze_run.py --mode db --db models/<run_name>/training_logs.db
python analyze_pth.py models/<run_name>/checkpoint.pth --no-map
```

Useful variants:

```bash
python analyze_run.py --mode db --run-name <your_run_name>
python analyze_pth.py --run-name <your_run_name> --which best --no-map
python analyze_pth.py --run-name <your_run_name> --which best --max-points 2000
```

### 4. Evaluation & Diagnostics
Basic eval from the latest or best checkpoint:

```bash
python eval.py --run_name <your_run_name> --best --episodes 10 --nodeterministic
python eval.py --run_name <your_run_name> --best --episodes 10
```

High-signal eval flags:
- `--best`
  prefer `best_checkpoint.pth` over `checkpoint.pth`
- `--checkpoint <path>`
  explicit checkpoint path; overrides `--run_name`
- `--episodes <N>`
  how many eval episodes to play
- `--deterministic` / `--nodeterministic`
  argmax vs sampled evaluation
- `--visualize` / `--novisualize`
  toggle the SC2 renderer

Inspection flags that write JSONL diagnostics:
- `--inspect`
  raw observation schema/stats via `ObservationInspectorWrapper`
- `--inspect_output <path>`
  output path for the observation inspector
- `--inspect_policy_input`
  raw obs plus extracted hybrid batch summaries via `PolicyInputDiagnosticsWrapper`
- `--policy_input_output <path>`
  output path for policy-input diagnostics
- `--policy_input_every <N>`
  log every `N` env steps for policy-input diagnostics
- `--inspect_actions`
  available-action and dispatched-call logging via `AvailableActionsDiagnosticsWrapper`
- `--actions_output <path>`
  output path for action-space diagnostics
- `--actions_every <N>`
  log every `N` env steps for action-space diagnostics
- `--inspect_last_action`
  post-action feedback logging via `LastActionDiagnosticsWrapper`
- `--last_action_output <path>`
  output path for last-action diagnostics
- `--last_action_every <N>`
  log every `N` env steps for last-action diagnostics
- `--inspect_score`
  reward and `score_cumulative` delta logging via `ScoreDiagnosticsWrapper`
- `--score_output <path>`
  output path for score diagnostics
- `--score_every <N>`
  log every `N` env steps for score diagnostics

Trace flags that write per-episode `.pt` sidecar artifacts:
- `--trace_episodes <N>`
  save the first `N` eval episodes as replayable trace files for later
  activation / checkpoint inspection
- `--trace_output_dir <path>`
  directory for those episode trace files; defaults to
  `analysis_results/<run_name>/episode_traces/`

To analyze one of those saved traces as a separate bundle:

```bash
python analyze_eval_trace.py --run-name <your_run_name> --mode det
python analyze_eval_trace.py --trace analysis_results/<your_run_name>/episode_traces/det/episode_0001_det.pt --activations
```

This writes a small analysis bundle next to the trace by default, including:
- `trace_report.txt`
- reward and action timelines
- dispatched action counts
- spatial target scatter
- spatial input planes for one selected policy step
- optional `conv1` / `conv2` / `conv3` activation grids when `--activations` is enabled

One useful combined command:

```bash
python eval.py --run_name <your_run_name> --best --episodes 5 --inspect --inspect_policy_input --inspect_actions --inspect_last_action --inspect_score --inspect_output analysis_results/<your_run_name>/eval_observation_space.jsonl --policy_input_output analysis_results/<your_run_name>/policy_input_diagnostics.jsonl --actions_output analysis_results/<your_run_name>/available_actions_diagnostics.jsonl --last_action_output analysis_results/<your_run_name>/last_action_diagnostics.jsonl --score_output analysis_results/<your_run_name>/score_diagnostics.jsonl
```

Training-side defaults for these diagnostics can also be centralized in `config.yaml`:
- `use_observation_inspector`
- `observation_inspector_*`
- `use_policy_input_diagnostics`
- `policy_input_diagnostics_*`
- `use_available_actions_diagnostics`
- `available_actions_diagnostics_*`
- `use_last_action_diagnostics`
- `last_action_diagnostics_*`
- `use_score_diagnostics`
- `score_diagnostics_*`

---

## 🧪 Testing

The repository includes a suite of unit tests located in the `tests/` directory. Because the environment requires heavy dependencies (`torch`, `pysc2`, `snntorch`), running the tests in a minimal environment requires mocking.

```bash
pytest -v tests/
```

---

## 📝 Planning & Future Work

The repo root is intentionally lighter now. Planning notes and
historical docs live under [`docs/`](docs/README.md).

Recommended starting points:
- [`docs/current/REPO_STATE.md`](docs/current/REPO_STATE.md): current repo state and open questions
- [`docs/current/action_refactor.md`](docs/current/action_refactor.md): what the Stage-1 action refactor landed and what remains
- [`docs/current/THE_BPTT.md`](docs/current/THE_BPTT.md): current BPTT/TBPTT reasoning note
- [`docs/current/working_log.md`](docs/current/working_log.md): compressed implementation log
- [`docs/README.md`](docs/README.md): full doc index
- `logs/PROJECT_LOGS.md` / `logs/SESSION_LOG_*.md`: longer historical narrative
