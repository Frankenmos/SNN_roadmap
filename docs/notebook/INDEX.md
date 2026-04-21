# Repo Notebook Index

Welcome to the durable technical memory of this project. If you are an agent or a human trying to understand *why* this repo looks the way it does, this is your starting point.

This notebook sits one layer above the operational docs in `docs/current/`. While `docs/current/` describes the *immediate code state and open tasks*, the notebook captures the *trajectory, experiments, and architectural reasoning*.

## 🧭 Navigation Guide

Depending on your goal, jump to the relevant section:

### "I need to understand the current repo truth and architecture"
**Go to: [REPO_NOTEBOOK.md](REPO_NOTEBOOK.md)**
- This is the main lab notebook.
- It covers the core project identity (SNN + PPO + PySC2), the observation and action pipelines, the evolution of the reward function, and a high-level summary of the diagnostics ecosystem.
- Start here if you are newly onboarding.

### "I need to onboard a new agent fast"
**Go to: [REPO_NOTEBOOK.md#how-to-onboard-a-new-agent-fast](REPO_NOTEBOOK.md#how-to-onboard-a-new-agent-fast)**
- A dedicated cheat-sheet for autonomous coding agents detailing which files are the current source of truth, what conceptual traps exist, and local vocabulary definitions.

### "I need to understand the history of major design decisions"
**Go to: [DECISION_TIMELINE.md](DECISION_TIMELINE.md)**
- A chronological timeline of structural shifts (e.g., Fix 3 hybrid tokenization, Stage-1 Action Refactor, Stage-1 TBPTT, Multi-timescale Token Memory).
- Explains the trade-offs introduced and whether a component is stable or provisional.

### "I need experiment history and how to interpret runs"
**Go to: [EXPERIMENT_MEMORY.md](EXPERIMENT_MEMORY.md)**
- Summarizes the major named runs like `BPTT-1` (and `BPTT_test`).
- Explains the symptoms observed in analysis (like the argmax-trap where the deterministic evaluation drops to 0.0), and provides guidance on how to avoid misleading interpretations of the logs.

### "I need context on research ideas, literature, and future branches"
**Go to: [LITERATURE_AND_IDEAS.md](LITERATURE_AND_IDEAS.md)**
- Maps broader machine learning and SNN ideas to the concrete code reality of this repo.
- Separates what is *already implemented* from *plausible future branches* and things *intentionally deferred*.

### "I need to know the open questions and next steps"
**Go to: [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)**
- A categorized backlog of unresolved technical questions.
- Delineates urgent fixes from important architectural debates.

---
*Note: For the operational plan, active execution state, and exact file mappings, always cross-reference with [docs/current/](../current/REPO_STATE.md).*