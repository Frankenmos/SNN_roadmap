# Repo Notebook

**Last Updated:** 2026-04-21

This document serves as a living technical memory of the project's evolution. It synthesizes the architecture, the decisions that shaped it, and the known bottlenecks.

---

## How To Onboard A New Agent Fast

If you are a coding agent freshly dropped into this repository, read this section to quickly establish context without falling into stale rabbit holes.

### 1. Where to look for Truth
*   **The absolute operational truth:** [`docs/current/REPO_STATE.md`](../current/REPO_STATE.md). Always start here to understand the *immediate* state of the code and the current priorities.
*   **Active refactor status:** [`docs/current/action_refactor.md`](../current/action_refactor.md) and [`docs/current/THE_BPTT.md`](../current/THE_BPTT.md).
*   **Code defaults:** `config.yaml`.
*   **Recent execution context:** `logs/PROJECT_LOGS.md` or `docs/current/working_log.md` (the latter is compressed but more recent).
*   **This directory (`docs/notebook/`):** Use for deep context on *why* things are structured this way, what experiments were run, and historical decision-making.

### 2. What is Historical/Archive (Read with Caution)
*   **`docs/archive/`:** Old plans (e.g., `NEXT_FIXES_PLAN.md`) are explicitly stale. They contain useful ideation but are **not** the current execution plan.
*   **`docs/current/Claude_rapport.md`:** Independent 2026-04-20 review. Useful for reward/eval reasoning, but superseded by `REPO_STATE.md` as of 2026-04-21.

### 3. Conceptual Traps in this Repo
*   **Argmax-Trap / Deterministic Flatline:** The policy has a known issue where it learns to score points stochastically during training but achieves `0.0` reward during deterministic evaluation (`argmax`). This is because it gets stuck taking `NO_OP` or `ATTACK` (without `MOVE`) when forced into deterministic exploitation. It is a policy-shape/reward-shaping issue, **not** a broken action availability issue.
*   **The Evaluator and the DB:** The "best" checkpoint logic is tied to *deterministic evaluation*. Because deterministic eval is stuck at 0.0, `best_checkpoint.pth` is often stale compared to the real learning progress stored in the training database. **Always check the live DB (e.g., via `analysis_results/` plots) instead of relying solely on `best_checkpoint.pth`.**
*   **PPO Tensor Semantics:** This repo batch-converts native Python scalars to tensors inside `PPO.update_policy` rather than inside the step loop. This avoids CUDA sync overhead. **Do not refactor step-level storage to use tensors for scalars.**
*   **Fake PySC2 FunctionCalls:** Unit tests use fake `FunctionCall` objects to mock the action space. The matcher logic must robustly check `.function`, then `.id`, then `.name` because the test environment cannot perfectly simulate the real PySC2 C++ backend.
*   **SNN Recurrent State:** The recurrent state of the Spiking Neural Network (SNN) is handled delicately across Truncated Backpropagation Through Time (TBPTT) chunks. Terminated episodes zero-out the state differentiably (`torch.where`). **Do not break the differentiability of the SNN state reset.**

### 4. Local Vocabulary
*   **TBPTT:** Truncated Backpropagation Through Time. We chunk episodes into lengths of e.g. 32 steps to backpropagate through the SNN without exhausting memory or encountering vanishing gradients. Currently at "Stage-1" (ordered chunk replay + packed replay).
*   **Fix 3:** A major historical refactor that introduced hybrid observation tokenization (handling spatial CNN outputs, tabular entity features, and selection state as a unified token stream).
*   **Bridge Token:** A 4-float vector added to `meta_vec` to communicate the *executed action* back to the policy on the next step.
*   **Reset Bootstrap:** A non-learned step at the start of an episode (outside PPO memory) to select the army, ensuring the policy begins with `MOVE` and `ATTACK` available.

---

## Project Identity

This project is a Reinforcement Learning agent built to solve the PySC2 (StarCraft II) `DefeatRoaches` minigame.

The core architecture combines:
1.  **Proximal Policy Optimization (PPO)** for the reinforcement learning algorithm.
2.  **Spiking Neural Networks (SNN)** powered by `snntorch` for the recurrent policy backbone. This is an explicit, deliberate design choice to explore neuromorphic/temporal processing for RL sequence modeling.
3.  **Hybrid Tokenization** utilizing a Transformer-style architecture (specifically using SDPA - Scaled Dot-Product Attention) to process heterogeneous observations (spatial maps, tabular unit data, global metadata).

## Architecture in Plain English

The agent needs to process visual maps, lists of units, and global game state, and then output actions (`NO_OP`, `MOVE`, `ATTACK`) along with spatial coordinates `(x, y)` when applicable.

### 1. Observation Pipeline

The environment state is processed into a sequence of tokens (`Fix 3` architecture):
*   **Spatial Tokens:** The minimap/screen feature layers (e.g., `feature_screen`) are passed through a lightweight CNN, pooled, and flattened into tokens.
*   **Entity Tokens:** The tabular `feature_units` array is parsed directly into entity representations.
*   **Selection Tokens:** Information about currently selected units (`single_select`, `multi_select`).
*   **Meta Vector:** A fixed-size vector (`32` dimensions) capturing global state: player stats, available actions, the last executed PySC2 action ID, and the **Bridge Token** (which records the last action taken by the agent itself).

These tokens are combined, embedded with type identifiers, and passed through a Self-Attention block (using `torch.nn.functional.scaled_dot_product_attention`).

### 2. The Spiking Recurrent Core (Dual-Timescale)

After attention, the tokens enter the SNN layer. As of the multi-timescale patch (2026-04-21), the architecture uses two parallel pathways:
*   **Fast SNN Pathway:** Uses Leaky Integrate-and-Fire (LIF) neurons with lower decay rates (e.g., `alpha=0.55`, `beta=0.65`) to react quickly to immediate changes.
*   **Slow SNN Pathway:** Uses LIF neurons with higher decay rates (e.g., `alpha=0.92`, `beta=0.97`) to retain longer-term sequence memory.

The membrane potentials from these two pathways are combined (e.g., averaged) to form a single latent control representation.

### 3. Action Pipeline

The action space has undergone a "Stage-1 Refactor" to enforce clean semantics.
*   **Vocab:** The policy strictly outputs one of three action IDs: `0: NO_OP`, `1: MOVE`, `2: ATTACK`.
*   **Factorization:** The policy first samples the `action_type`. It then uses the *same* latent state, conditioned on the chosen action type, to sample the spatial coordinates `(move_x, move_y)` if the action is `MOVE` or `ATTACK`.
*   **Execution:**
    *   `MOVE` translates directly to PySC2's `Move_screen(x, y)`.
    *   `ATTACK` translates to `Attack_screen(x, y)`.
*   **Availability Masking:** The policy explicitly masks out invalid actions based on PySC2's `available_actions`.
*   **The Reset Bootstrap:** To prevent the agent from getting permanently stuck at the start of an episode without `MOVE`/`ATTACK` available, an automatic `select_army` action is taken on step 0, *outside* the PPO replay buffer.

### 4. PPO and TBPTT Evolution

Training SNNs over long sequences (e.g., 600 steps) is intractable with standard BPTT due to VRAM limits and gradient degradation. We implemented **Truncated Backpropagation Through Time (TBPTT)**.
*   The replay buffer stores rollout steps.
*   During update, episodes are sliced into overlapping chunks (e.g., 32 steps).
*   The SNN internal state is preserved across chunk boundaries but explicitly detached from the computation graph to stop gradients from flowing too far back.
*   "Packed replay" was introduced to drastically speed up the processing of these chunks by avoiding redundant zero-padding for variable-length episodes.

## Reward Evolution

The reward signal has been a constant source of iteration. The current iteration (`defeat_roaches_v3`) shapes the reward based on:
*   Damage dealt to enemies (+).
*   Damage taken by allies (-).
*   Kills (+ heavily weighted).
*   Distance to target (encouraging engagement).
*   Step penalty (discouraging stalling/`NO_OP` abuse).

**Current Status:** The reward function is currently out of sync with the recent Stage-1 action refactor. The wrapper-driven environment insights necessitate a refactoring of the reward calculation (especially fixing terminal win/loss detection), which is an immediate priority.

## Analysis and Diagnostics Ecosystem

Because standard RL metrics (like `reward`) are deeply flawed when the agent is stuck in local optima, we built a robust diagnostics layer:
*   **SQLite DB:** All training steps, episodes, and evaluations are logged directly to a SQLite database (`training_logs.db`).
*   **Analysis Scripts:** Python scripts (`analyze_run.py`, `results.py`) query the DB and generate plots in `analysis_results/` (e.g., `win_rate.png`, `reward_components.png`).
*   **Trace Tools:** `analyze_eval_trace.py` allows detailed inspection of specific evaluation episodes, tracking exactly which actions were available, what the policy chose, and the subsequent reward.
*   **JSONL Dumps:** The agent periodically dumps observation structures, available actions, and policy inputs to `.jsonl` files for detailed post-mortem debugging.

## Current Bottlenecks and Failures

1.  **The Argmax-Trap:** During deterministic evaluation, the agent collapses to outputting `NO_OP` and occasionally `ATTACK` (but never `MOVE`). As a result, it scores `0.0` in evaluation despite learning to increase shaped reward during stochastic training. This is the biggest current bottleneck.
2.  **Stale Reward Logic:** The reward function needs an overhaul to properly encourage aggressive movement and penalize passive `NO_OP` behavior.
3.  **Entity Identity (Tag Pinning):** The policy currently struggles to track specific enemy units across time steps because PySC2 does not inherently stabilize unit indices. We need to implement tag-pinned entity tracking.

## Near-Term likely Next Steps

1.  **Reward Refactor:** Rebalance the reward to punish `NO_OP` strongly and reward successful engagement, then observe if deterministic eval recovers.
2.  **Diagnostic Regeneration:** Re-run the main analysis bundle against the live `BPTT-1` checkpoint.
3.  **Evaluate Action Space Expansion (Stage 2):** If deterministic behavior recovers after the reward pass, move towards Stage 2: introducing a dedicated action-history token group and learnable selection actions (`SELECT_POINT`, `SELECT_RECT`).

## Long-Term Branch Ideas

*   **Reward-as-Neuromodulator:** Injecting the reward signal directly into the SNN recurrent state to dynamically modulate learning.
*   **ALIF Neurons:** Swapping standard LIF neurons for Adaptive Leaky Integrate-and-Fire neurons for more complex temporal dynamics.
*   **Multi-Minigame / Full Game Branch:** Scaling the architecture to handle more complex PySC2 tasks beyond `DefeatRoaches`.

---
*Reference: [docs/current/REPO_STATE.md](../current/REPO_STATE.md)*