# Open Questions

**Last Updated:** 2026-04-21

This is the project's backlog of unresolved technical questions.

---

## 1. Urgent

*   **Reward Refactoring:** How aggressively should we penalize `NO_OP` to break the deterministic argmax-trap? Is the current `reward_scale=0.1` suppressing the learning signal too much with the new shaped reward magnitudes?
*   **Terminal Detection:** Why is the terminal win/loss check in `agent_core/rewards/defeat_roaches_v2.py` (or `v3`) broken, and how does it skew the final step value estimation in PPO?
*   **Diagnostic Refresh:** The main analysis bundle for `BPTT-1` needs to be regenerated against the live DB/checkpoint state to stop lagging behind the actual run.

## 2. Important But Not Urgent

*   **The SNN/TBPTT Verdict:** Are SNNs + TBPTT actually providing a benefit for this task over a dense transformer or standard LSTM? We are paying the computational cost, but have we seen the sample efficiency / temporal abstraction benefits yet?
*   **Action History Tokens:** Is the 4-float bridge token sufficient, or is the lack of a deeper action-history token group preventing the agent from understanding its own trajectory?
*   **Entity Tracking Strategy:** PySC2 shuffles the `feature_units` array. Will we solve the "entity identity" problem by pinning to `raw_units.tag`, or does the attention mechanism naturally learn to track entities without explicit pinning?

## 3. Research Branch

*   **Neuromodulation via Reward:** Can the reward signal be piped back into the SNN's recurrent state (e.g., modulating membrane decay or synaptic weights) to dynamically influence exploration/exploitation?
*   **ALIF Neurons:** Would swapping to Adaptive Leaky Integrate-and-Fire neurons provide better temporal dynamics than the current static-threshold LIF neurons?
*   **Multi-Minigame Scaling:** Can this architecture learn `CollectMineralShards` or `FindAndDefeatZerglings` without architectural changes?

## 4. Tooling / Docs / Memory

*   **Eval Determinism Check:** Can we add an `--eval_epsilon` flag to `eval.py` to run near-deterministic (e.g., $\epsilon=0.05$) evaluations? This would conclusively prove if the policy can score points but is just trapped by strict argmax.
*   **DB vs. Checkpoint Sync:** Can we decouple the `best_checkpoint` saving logic from the flawed deterministic eval reward, and instead tie it to moving averages of the stochastic training reward stored in the DB?