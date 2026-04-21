# Decision Timeline

**Last Updated:** 2026-04-21

This document tracks the chronological history of major architectural decisions in the repository. It captures *what* changed, *why* it changed, the *trade-offs*, and the current *stability* of the decision.

---

### [2025-01-18] Initial Setup & Scripted Agents
*   **What changed:** Custom Python environment created to match DeepMind’s PySC2 requirements (downgrading protobuf, etc.).
*   **Why:** PySC2 is notoriously fragile with modern dependencies.
*   **Status:** Stable. The environment setup (`envs/setup_env.py`) remains the bedrock.

### [2025-01-19] Observation Inspectors & Wrapper Pattern
*   **What changed:** Introduced `AvailableActionsPrinter` and `ObservationInspector` wrappers.
*   **Why:** To dynamically explore PySC2's complex observation space (`feature_screen`, `feature_units`, `available_actions`) before designing the RL policy.
*   **Status:** Stable. The wrapper pattern evolved into the current diagnostic JSONL dumpers.

### [Fix 3] Hybrid Observation Tokenization
*   **What changed:** Replaced monolithic flattened observations with a tokenized stream. Spatial features (via CNN), tabular entity data (`feature_units`), and selection state are all converted to tokens, embedded with type IDs, and processed via self-attention.
*   **Why:** PySC2 state is inherently heterogeneous. A dense linear layer cannot handle variable numbers of units and spatial data effectively.
*   **Trade-off:** Computationally more expensive; requires an attention mechanism before the SNN.
*   **Status:** Stable. This is the core observation pipeline.

### [Pre-2026-04-20] The SNN + PPO Integration
*   **What changed:** Adopted `snntorch` for the recurrent policy backbone within a PPO framework.
*   **Why:** To explore neuromorphic computing capabilities for sequential decision-making in RL.
*   **Trade-off:** SNNs are notoriously difficult to train with standard RL due to non-differentiable spiking mechanisms (requiring surrogate gradients) and complex temporal credit assignment.
*   **Status:** Stable (Core Research Bet).

### [2026-04-20] Stage-1 Action Refactor
*   **What changed:**
    *   Hardcoded the learned policy vocabulary to exactly 3 discrete actions: `NO_OP`, `MOVE`, `ATTACK`.
    *   Removed `Smart_screen` and the learned `select_army` actions.
    *   Introduced a **Reset Bootstrap**: a forced, non-learned `select_army` step at the beginning of the episode, *outside* PPO memory.
    *   Introduced the **Bridge Token**: a 4-float vector in `meta_vec` to pass the previously executed action back into the policy.
    *   Implemented strict action availability masking.
*   **Why:** The agent was thrashing in a massive action space. It needed a constrained environment to learn basic micro (moving and attacking) before learning complex selections. The reset bootstrap ensures the agent starts with `MOVE` and `ATTACK` actually available.
*   **Trade-off:** The agent cannot currently learn to select sub-groups or use spells.
*   **Status:** Stable. (Stage 2 expansion is deferred pending reward refactoring).

### [Post-2026-04-20] Stage-1 TBPTT & Packed Replay
*   **What changed:** Implemented Truncated Backpropagation Through Time (TBPTT) with a window of 32 steps. Added packed replay to handle variable-length chunks efficiently.
*   **Why:** Full-episode BPTT (600 steps) was blowing up VRAM and causing vanishing gradients in the SNN.
*   **Trade-off:** SNN state must be detached at chunk boundaries, slightly breaking the exactness of the temporal gradient.
*   **Status:** Stable, but the long-term effectiveness of SNN + TBPTT for this task is still under evaluation.

### [2026-04-21] Multi-timescale Token Memory
*   **What changed:** Split the SNN temporal processing into two parallel pathways: a "fast" path (low decay) and a "slow" path (high decay). Both combine into a single latent readout.
*   **Why:** The policy needed to react to immediate tactical changes (fast path) while remembering longer sequence context (slow path).
*   **Trade-off:** Increases the size of the recurrent state and computational cost per step.
*   **Status:** Provisional. The first major runs (`BPTT-1`) are evaluating its impact. ALIF neurons and reward-driven neuromodulation were intentionally deferred to avoid compounding risk.