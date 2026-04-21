# Jules Memory Notebook

**Project:** SNN_roadmap
**Date:** 2026-04-21

## State of the Repo

I just mapped out the entire `docs/notebook/` memory layer for the repository based on the current architecture. Here are the core insights I need to remember for future sessions:

### 1. The PPO+SNN Architecture
This is not a standard RL repo. It uses PyTorch + `snntorch`.
- **Hybrid Obs (Fix 3):** Heterogeneous observations (spatial, tabular, selection, meta) are tokenized and passed through SDPA.
- **Dual-Timescale SNN:** Temporal processing uses parallel "fast" and "slow" LIF neuron pathways.
- **TBPTT:** The replay mechanism unrolls the SNN over chunks (e.g., 32 steps) to backpropagate. The SNN internal state must be carefully maintained and detached at chunk boundaries, and zeroed out differentiably via `torch.where` when an episode terminates.

### 2. The Stage-1 Action Refactor
- The agent is tightly constrained. The vocab is exactly 3 actions: `NO_OP(0)`, `MOVE(1)`, `ATTACK(2)`.
- **Reset Bootstrap:** To ensure `MOVE` and `ATTACK` are actually available during training, the agent forces a `select_army` action outside of PPO memory at step 0.
- **Bridge Token:** A 4-float vector injected into `meta_vec` to tell the policy what action it took on the previous step.

### 3. The "Argmax-Trap" (Current Biggest Bottleneck)
- **Symptom:** Deterministic evaluation scores `0.0`. The `best_checkpoint.pth` stops updating early.
- **Cause:** The policy learns a stochastic shape that can score, but when forced to `argmax`, it defaults entirely to `NO_OP` or `ATTACK` and never chooses `MOVE`.
- **Correction:** Do not "fix" this by tearing apart the action availability masking. The action plumbing is correct. The fix requires reward refactoring (currently `v3`) to heavily penalize `NO_OP` and encourage movement.
- **Diagnostic Habit:** Always check `available_actions_diagnostics.jsonl` vs `policy_input_diagnostics.jsonl` to see if the agent *could* move but *chose* not to.

### 4. Sandbox Quirks
- Standard pip installs fail (restricted internet).
- `pytest` commands often fail locally because `numpy`, `torch`, etc., are missing from the default `python3` path.
- Standalone mock scripts or `sys.modules` mocking are required to run local tests.
- Tensor conversion in PPO happens *during the batch update*, not during the inner step loop. Avoid changing this.

### Next Immediate Tasks (If I am assigned them)
1. **Reward Refactor:** Fix terminal win/loss check in `defeat_roaches_v3.py` and rebalance penalties to fix the argmax-trap.
2. **Analysis Regeneration:** Re-run the main `BPTT-1` static report bundle against the live DB.