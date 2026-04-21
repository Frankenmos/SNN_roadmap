# Experiment Memory

**Last Updated:** 2026-04-21

This document captures the history of major training runs, what hypotheses they tested, and how to correctly interpret the diagnostics in this repository.

---

## How to Read Experiments Without Fooling Yourself

In this repo, standard RL metrics can be highly misleading.

1.  **Do not trust `best_checkpoint.pth` blindly.**
    *   The evaluator saves the "best" model based on *deterministic* (argmax) reward.
    *   If the agent learns to score using stochastic sampling but collapses to `NO_OP` during deterministic evaluation (the "Argmax-Trap"), the `best_checkpoint.pth` will stop updating very early in training.
    *   **Rule:** Always check the SQLite DB (`training_logs.db`) and the plots in `analysis_results/` to see the actual stochastic training curve.

2.  **Verify Action Availability vs. Policy Intent.**
    *   If the agent is only outputting `NO_OP`, is it because PySC2 disabled `MOVE` and `ATTACK`, or because the policy *chose* `NO_OP`?
    *   **Rule:** Use the `available_actions_diagnostics.jsonl` and `policy_input_diagnostics.jsonl` dumps to confirm. If `Move_screen` is in the available actions list but the policy outputs `action_id=0`, it's a policy/reward issue, not a plumbing issue.

3.  **Trace the Eval Step-by-Step.**
    *   Use `--trace_episodes` in `eval.py` to dump full episode traces. Inspect them with `analyze_eval_trace.py`. This tells you exactly what the agent saw and did on step 45, which is infinitely more useful than an average score of `0.0`.

---

## Important Runs

### `BPTT-1` (Current Mainline Evidence)
*   **What was tested:** The integration of the Stage-1 Action Refactor, Stage-1 TBPTT, packed replay, and the Multi-timescale SNN patch.
*   **Status:** Trained to ~5260 episodes.
*   **The Findings:**
    *   **Stochastic Training:** The training DB shows a positive trend. The shaped reward is increasing (last-100 episode average ~223).
    *   **Deterministic Evaluation (The Trap):** The deterministic evaluation score flatlined at `0.0` very early on. The late-stage deterministic action mix is heavily dominated by `NO_OP` and `ATTACK`. `MOVE` has almost vanished.
    *   **Root Cause Analysis:**
        *   Did the action refactor break availability? **No.** `available_actions_diagnostics` shows `Move_screen` and `Attack_screen` are available >99% of the time after the step 0 reset bootstrap.
        *   Did TBPTT break learning? **Unlikely.** Stochastic training is improving.
        *   **Conclusion:** The agent is stuck in an argmax-trap. It likely learned that `NO_OP` is "safe" (avoids damage penalties) and only uses `ATTACK` when an enemy wanders close. It hasn't learned to actively `MOVE` to seek out enemies. The reward shaping favors conservative behavior over aggressive completion.
*   **Resulting Action Item:** Refactor the reward function (currently `v3`) to heavily penalize `NO_OP` stalling and reward active engagement, rather than blaming the action-space plumbing.

### `BPTT_test` & `BPTT_test1`
*   **What was tested:** Shorter diagnostic runs to verify the plumbing of TBPTT and the DB logging infrastructure before launching `BPTT-1`.
*   **The Findings:** Confirmed that SQLite batch insertions and TBPTT packed chunking were computationally stable and didn't crash PySC2.

### `fix3_hybrid_obs_*`
*   **What was tested:** The transition from dense linear observations to the hybrid tokenized architecture.
*   **The Findings:** Proved that the agent could process heterogeneous inputs (spatial CNN + tabular units) via self-attention without blowing up memory. Established the baseline that `BPTT-1` built upon.