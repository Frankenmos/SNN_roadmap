# Session Log — 2026-04-15

SNN + PPO on DefeatRoaches. Dense session covering architecture surgery,
new diagnostics, a clean per-run directory layout, and the first round of
post-collapse recovery knobs. This log lives separately from
`logs/PROJECT_LOGS.md` because that file predates most of the current code.

---

## 1. Policy-network changes

### 1.1 `num_steps` 8 → 2

- **Problem**: the inner SNN time loop fed the *same* `spatial_input` for
  all 8 steps in [policy_network.py](PPO_CNN/policy_network.py). With a
  static input and identical conv weights, only membrane state varied —
  we were paying 8× the compute for the LIFs to "settle" on one frame.
- **Solution**: dropped the default to 2. Kept the learnable
  `alpha/beta` per layer; we can raise `num_steps` again once we feed a
  true temporal stack.
- **Where**: `num_steps` default in `PolicyNetwork.__init__` and in
  `config.yaml`.

### 1.2 Device probe cleanup

- Paranoid nested try/except around `torch.cuda.is_available()` plus a
  separate test allocation removed. `is_available()` already performs the
  runtime check.

### 1.3 Angle head → screen-point move head

- **Problem / constraint**: original design emitted a 4-logit "angle"
  head, then used two of the four logits as `(sin, cos)` to compute a
  movement direction with fixed magnitude. This is a reasonable first-env
  inductive bias, but wastes half the logits and collapses
  screen-point intent onto a single circle around the agent.
- **Solution**: replaced with two 84-way categorical heads
  (`move_x_fc`, `move_y_fc`), one per screen axis. Follows the AlphaStar
  / PySC2-baseline convention.
- **PPO adaptation (subtle)**: the *joint* log-prob is

    ```text
    log_prob = logp(action) + is_move * (logp(move_x) + logp(move_y))
    ```

  So move-head gradients only flow on rollouts where the sampled
  high-level action was "move". Attack/no-op rollouts give zero move-head
  gradient, so irrelevant move-coord samples don't pollute training.
  Same masking used in the entropy bonus. The rollout and training
  log-probs must compute this identically — both do.
- **Plumbing**: `select_action`, `store_transition`, `update_policy`,
  `_calculate_losses`, `action_space.move(obs, x, y)`, `agent.step`, and
  the rollout tuple in `PPO_CNN_run.py` all updated.

### 1.4 LayerNorm — two targeted spots

- **Problem**: post-concat we feed
  `cat([3136-d time-summed spike tokens, 100-d vector obs])` into a
  `Linear(3236, 128)`. The two sides live on wildly different scales and
  the token-side magnitude drifts as the SNN's learnable time constants
  change during training.
- **Solution**:
  1. **Pre-LN inside `SpikingSelfAttention`**: normalize tokens before
     QKV projection; residual stays unnormalized (Pre-LN transformer
     convention).
  2. **`combined_norm` before the shared FC head**: LayerNorm across the
     3236-dim concatenated vector.
- **Clarification the user asked about**: LayerNorm does not "lower
  extremes" — it rescales each sample to zero-mean unit-variance and
  applies a learned affine, so downstream layers see a stable input
  distribution regardless of upstream drift.

### 1.5 AMP wired end-to-end

- **Problem**: a `GradScaler` and `amp_dtype` attribute existed on the
  policy but were never used. No `autocast` around forward; no
  `scaler.scale(loss).backward()` in PPO.
- **Solution**: `torch.amp.autocast('cuda', dtype=fp16, enabled=use_amp)`
  wraps both the rollout forward in `select_action` and the forward+loss
  in `update_policy`. Backward uses
  `scaler.scale(loss).backward() → unscale_ → clip_grad_norm_ → step →
  update`. Softmax runs in fp32 for numerical stability.
- **Explanation (from the chat)**: AMP runs matmuls/convs in fp16 for
  tensor-core throughput while keeping reductions + optimizer state in
  fp32. It is *not* quantization: no grid snapping, just a smaller float.
  The scaler multiplies the loss before backward so tiny fp16 gradients
  don't underflow to zero.

---

## 2. Training-run analyzer — `results.py`

### 2.1 What was there

`TrainingAnalyzer` class that loaded `episodes` / `steps` /
`reward_components` from the SQLite log and plotted reward curve,
episode length, and a flat action histogram. Had a bug where the
`reward_components` loader queried `episode` instead of `episode_id`,
which silently broke the component plot.

### 2.2 What we added

- **Bug fix** on the column name.
- **Plateau detection**: sliding linear fit on the rolling-mean reward;
  first window where |slope| < threshold AND the center is near the peak.
- **Rolling oscillation score**: coefficient of variation of reward,
  window=50. CoV > 0.5 = instability flag.
- **Empirical action-entropy** over episode-bins as a proxy for policy
  entropy. Needed because the logger did not (yet) store true entropy.
- **Action-mix drift**: L1 distance between adjacent bins' action
  distributions. Sharp jumps → possible catastrophic forgetting.
- **Win-rate proxy**: fraction of episodes above a reward threshold.
- **`diagnose()`** that runs everything and emits a plain-text
  `instability_report.txt` mapping observed signals to concrete
  hyperparameter knobs.

### 2.3 First diagnosis of the existing run (`run_collapse_e3884`)

- No plateau detected (CoV too high to ever settle).
- Two flags fired:
  - **No-op dominates**: 55.2% of late-training steps are no-op.
  - **Sharp action-mix shift** at bins 385 / 1001 / 1232 — kiting
    learned, then twice collapsed.
- Supplemented by **architectural suspects** (listed separately in the
  report because they cannot be confirmed from logs alone):
  1. **Stateful/stateless SNN mismatch** between rollout
     (`self.snn_state` carried across env steps,
     [PPO_CNN_agent.py:83-85](PPO_CNN_agent.py#L83-L85)) and training
     (`state=None`, [PPO.py:179-181](PPO_CNN/PPO.py#L179-L181)). PPO's
     importance ratio becomes inconsistent; bias grows with episode
     length. Very plausible cause of the drift after peak.
  2. **Move-head entropy asymmetry**. A move sample contributes
     `H ≈ log 3 + 2 log 84 ≈ 9.9` to entropy; an attack sample
     contributes `H ≈ 1.1`. With `entropy_coef=0.01` the move action
     gets a ~10× entropy bonus — a systematic bias toward movement that
     matches the observed shift away from attack late in training.

---

## 3. Best-checkpoint saving

### 3.1 Problem

[PPO_CNN_run.py](PPO_CNN_run.py) only saved a latest checkpoint; every
save overwrote the previous one. The policy peaked around episode 1000
at reward ~600, then degraded to ~42 by episode 3800. **The peak policy
is unrecoverable for that run** — there was no separate best file. The
forgetting pattern is very consistent with an LR that was appropriate
early and too aggressive post-convergence.

### 3.2 Solution

- `config.yaml` gained `best_checkpoint_path` and `best_min_episodes`.
- `maybe_save_best_checkpoint()` in [PPO_CNN_run.py](PPO_CNN_run.py)
  writes a separate `best_checkpoint.pth` only when the rolling average
  reward beats the stored best, and only after `best_min_episodes` to
  prevent early noise from setting an unbeatable floor.
- New helper [resume_from_best.py](resume_from_best.py): copies
  `best_checkpoint.pth` over `checkpoint.pth` (with a `.before_resume`
  backup) so the next `PPO_CNN_run.py` invocation resumes from the peak.

---

## 4. Per-run folder layout (housekeeping)

### 4.1 Problem

Everything (`checkpoint.pth`, `training_logs.db`, analysis output) lived
at the repo root. After multiple runs, the root filled up with suffixed
copies (`checkpoint2.pth`, `training_logs_old.db`, etc.) with no clear
mapping from run → artifacts → analysis.

### 4.2 Solution

Canonical layout:

```text
models/<run_name>/
    checkpoint.pth
    best_checkpoint.pth
    training_logs.db
analysis_results/<run_name>/
    training_progress.png
    reward_components.png
    win_rate.png
    training_metrics.csv
    instability_report.txt
```

- `config.yaml` gained `run_name` (`""` = auto-generate
  `run_YYYYMMDD_HHMMSS`), `models_dir`, `analysis_dir`. Existing
  `checkpoint_path` / `best_checkpoint_path` / `db_path` are now bare
  filenames joined under `{models_dir}/{run_name}/`.
- [PPO_CNN_run.py](PPO_CNN_run.py) added `_run_dir()` / `_run_path()`
  helpers; auto-creates the per-run folder; all checkpoint and DB paths
  route through them.
- [results.py](results.py) CLI added `--run-name`; defaults read from
  config; `--db` / `--out` auto-derive. Explicit flags still override.
- **Migration done**: the existing run was moved to
  `models/run_collapse_e3884/` and the analyzer re-verified against it.

---

## 5. Logger instrumentation

### 5.1 Problem

The SQLite schema logged only the high-level action id (`0/1/2`), raw
reward, and reward components. No policy-level signals (entropy, KL,
clip-fraction, value loss) and no move coordinates for the new
screen-point action space. So the analyzer was forced to build empirical
proxies.

### 5.2 Solution

- **`steps` table**: two new nullable columns `move_x INTEGER` and
  `move_y INTEGER`. `ALTER TABLE` migrations included for legacy DBs.
- **New table `ppo_updates`**: one row per `update_policy` call, with
  `mean_policy_loss`, `mean_value_loss`, `mean_entropy`, `mean_kl`
  (Schulman k3 approx), `clip_fraction`, `explained_variance`,
  `grad_norm`, `lr`, `timestamp`.
- **Producer**: `PPO._calculate_losses` now returns a `diag` dict;
  `PPO.update_policy` accumulates per-minibatch stats, computes
  `explained_variance` from the full rollout (`1 - var(returns - values)
  / var(returns)`), and returns `(losses, stats)`.
- **Consumer**: new `UPDATE` message branch in `LogListener`,
  flushed on the same cadence as the other buffers.
- **Analyzer**: `get_update_metrics()` loader; three new diagnosis
  rules (sustained `clip_fraction > 0.3` / `mean_kl > 0.05` /
  `explained_variance < 0`); new "Instrumentation status" section in
  the report that prints real stats when present and gracefully falls
  back to "using empirical proxies" otherwise.

---

## 6. Recovery knobs (LR decay + clip + epochs)

### 6.1 Problem

Flat `lr=1e-4`, `clip_eps=0.18`, and `epochs=20` per update (over ~3000
samples of rollout) are all aggressive in late training. Combined with
the architectural suspects from §2, they plausibly explain the
post-peak degradation via repeated overshoot updates.

### 6.2 Solution (config + scheduler)

- `config.yaml`: `clip_eps: 0.18 → 0.10`, `epochs: 20 → 8`, added
  `lr_min: 1.0e-5`.
- `PPO.__init__` accepts `total_updates` + `lr_min`; builds a
  `CosineAnnealingLR(T_max=total_updates, eta_min=lr_min)`. Scheduler
  stepped once at the end of every `update_policy`.
- `PPO_CNN_agent` computes `total_updates = total_episodes //
  update_frequency = 1000` from config and passes it in, so LR
  cosine-anneals from 1e-4 down to 1e-5 across the full run.
- Checkpoint save/load now includes `scheduler_state`, so a mid-run
  resume keeps the decay phase consistent. Legacy checkpoints (no
  `scheduler_state`) are handled without error.

---

## 7. Diagnosed but not yet addressed

Ordered by estimated impact.

- **Stateful/stateless SNN mismatch** (HIGH). Most likely cause of the
  post-peak drift. Three possible fixes considered; decision deferred.
- **Move-head entropy asymmetry** (HIGH) - **FIXED**. The ~10× entropy bonus for
  moving vs attacking has been normalized using log(n) per head.
- **No-op spam** (MED). Reward function rewards survival passively. Ties
  into the deferred reward-redesign carryover.
- **Broken tests** (`tests/test_PPO.py`, `tests/test_agent.py`) — still
  expect the old `(action, angle, …)` signature from before the
  screen-point refactor.
- **Env setup bug** in [envs/setup_env.py](envs/setup_env.py):
  `Bot(zerg)` configured for DefeatRoaches (single-agent mini-game),
  and `map_name` default mismatches config.
- **Dashboard contract** ([dashboard.py](dashboard.py)): no schema-missing
  warnings, no SQL-side aggregation for very large `steps` tables.

---

## 8. TL;DR

We turned a multi-mechanism SNN+PPO project with an opaque collapse
into a project with a clear per-run layout, a data-driven diagnostic
tool, proper best-checkpoint saving, real PPO-side instrumentation, and
a first round of recovery knobs queued to actually test the recovery
hypothesis in the next run.

### Fixed

`num_steps 8→2`, AMP end-to-end, screen-point move head, joint-logprob
with `is_move` masking, LayerNorm in the two places that mattered,
plateau/entropy/oscillation/drift diagnostic, best-checkpoint saving +
resume helper, per-run folder layout, full PPO logging schema with
KL/clip-fraction/explained-variance/grad-norm/LR, cosine LR decay 1e-4
→ 1e-5 + clip 0.10 + epochs 8.

### Not fixed yet (ordered)

1. Recovery verification — run training with the new knobs, interpret
   the new `ppo_updates` signals.
2. Structural fixes: SNN stateful/stateless mismatch.
3. Reward redesign (no-op spam).
4. Test + env-setup fixes.
5. EventProp dive (paused mid-Socratic; three math forks queued).

### Next-step anchor

Run the new training. When it stops (reaches total episodes, peaks, or
collapses), run:

```text
python results.py --run-name run_<timestamp> --report
```

Compare the "Instrumentation status" section's late-stage
`clip_fraction`, `mean_kl`, `explained_variance`, and `lr` against
expectations (clip_fraction should drop, kl should stay < 0.02,
explained_variance should trend up, lr should visibly decay).
Then revisit the deferred bucket list in order.
