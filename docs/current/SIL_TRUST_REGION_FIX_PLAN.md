# SIL Trust-Region Fix Plan — V7 post-mortem → V8 staged fixes

Written: 2026-07-07. Companion teaching doc: `learning/MERGING_LOSSES.md`.
Status: PLANNED, not implemented. Per the pause rule, Stage 0 is
measurement-only (allowed now); Stage 1 gates on Unit 5, Stage 2 on Unit 3.

## The evidence this plan answers (v7 @ ep 5500, run DB, 316 updates)

1. Approx KL grows 0.004 → ~1.0 over 300 updates; clip fraction saturates
   ~0.88 at clip_eps 0.10. Curriculum ended at update 120; KL kept climbing.
2. `target_kl=0.03` early-stop (`ppo_trainer.py:1331`) therefore fires after
   epoch 1 — the run has been training at EFFECTIVE EPOCHS = 1 for most of
   its life.
3. SIL gate never closes: `sil_gate_open_fraction` flat 0.52–0.66, buffer
   pinned at 5000, `sil_grad_norm` (pre-clip) rising 3 → 13. Cause: critic
   miscalibrated (EV 0.12–0.20; V ≈ −25 vs episodes landing ≈ +1), so the
   raw-scale gate (R−V)+ ≈ 26 stays wide open forever.
4. Grad decomposition: `grad_norm_critic_head` 11–160, `grad_norm_trunk`
   16–78, `grad_norm_actor_head` DECAYED to 0.002–0.03. With ~88% of samples
   clipped (zero policy grad), PPO barely trains the actor at all; SIL's
   separate pass (8 optimizer steps/update, consistent direction, shared
   Adam) is the dominant policy-training signal. That is the runaway.
5. Behavior: det eval −3.6 → −5.4 → −9.0 (n=5, std 0.00 = stereotyped);
   3254/3274 det actions are Smart_screen. Aiming (WHERE) good; dosage
   (WHEN) broken. Training shaped reward climbs while native eval falls.

## Stage 0 — Ratio diagnostic (measurement-only; DO THIS FIRST)

**Claim being tested:** at the first chunk group of the first epoch, the
weights are IDENTICAL to the collection weights (collection is synchronous,
`_collect_sync_fragments`), so ratio = π_new/π_old must be ≈ 1 and approx_kl
≈ 0. Anything else is a replay-fidelity artifact, not policy movement.

**Patch (small, logging-only), in the epoch loop of `update_policy`:**

- Record `approx_kl` / `clip_frac` for (epoch 1, group 1) BEFORE any
  optimizer step this update → new DB columns `kl_update_start`,
  `clip_frac_update_start`.
- Decompose the composite log-prob into its two parts and log each part's
  approx_kl separately:
  - action head: `action_dist.log_prob(actions)` vs the stored action-head
    old component — requires storing old log-prob per component; cheaper
    proxy: compute KL on the FULL composite and on the action head only
    (old composite minus old spatial is not stored → log action-head KL
    against a recomputed no-grad reference at update start instead).
  - spatial head: `is_spatial * target_log_prob` share.
- Optional deeper cut (second iteration): per-position-in-chunk mean
  |log-ratio| for positions 1–16 vs rest — a staleness gradient along the
  chunk would implicate stored-state drift; flat implicates numerics.

**How to read it:**

| Observation | Meaning | Consequence |
|---|---|---|
| `kl_update_start` ≈ 0 (<0.01), KL grows across groups/epochs | real intra-update policy movement | optimizer steps too large / Adam contamination → Stage 1 is the right fix and will show up in the KL metric |
| `kl_update_start` ≈ 0.5–1.0, flat | replay path does not reproduce collection log-probs (numerics: bf16 chunked replay vs collection forward; or weight-sync gap) | PPO ratios are lies; fix replay fidelity FIRST or Stage 1's effect will be invisible in KL |
| KL concentrated in spatial-head component | fine-head sharpening/quantization sensitivity (144-way logits under bf16) | the notes' opinion 3 becomes live; consider fp32 fine logits |
| KL concentrated in action head | 3-way head genuinely being yanked | consistent with SIL push on P(RIGHT) |

**How to run:** requires a restart, so piggyback on the next natural pause
of the live run: stop training → apply patch on `tooling-sprint` (or a
`ratio-probe` branch) → relaunch the SAME run (resume loads
`checkpoint.pth`) → 5 updates is enough → query:

```sql
SELECT update_id, kl_update_start, mean_kl, clip_frac_update_start,
       clip_fraction FROM ppo_updates ORDER BY update_id DESC LIMIT 5;
```

Zero-risk offline variant (more work, only if pausing the run is
unacceptable): standalone `--ratio-probe` mode that copies the checkpoint
to a scratch run dir, collects one rollout, runs the epoch loop with
`optimizer.step()` skipped, prints the table, exits.

## Stage 1 — Merge SIL into the PPO backward (fix 1)

Full why-and-how in `learning/MERGING_LOSSES.md`. Design, minimal diff:

1. **Sample trophies once per update** (start of `update_policy`), build
   packed trophy chunk groups once (reuse `_pack_chunk_group`).
2. **Inside the epoch loop, per chunk group:** forward one trophy sub-group
   (round-robin over the update's trophy groups), compute `sil_loss` exactly
   as today (gate detached), then

   ```python
   loss = policy_loss + value_loss - entropy_loss + sil_loss
   ```

   ONE backward, the EXISTING clip (0.5) and the EXISTING `optimizer.step()`.
   Delete the separate stepping in `_run_sil_pass` (keep its stats).
3. **Tame the gate scale.** Today (R−V)+ ≈ 26 while PPO's policy loss is
   O(1) (standardized advantages). Merged, relative scale matters directly.
   Interim (pre-Stage-2): `weight = (R - V).clamp(min=0).clamp(max=W_MAX)`
   with W_MAX ≈ 2. After Stage 2 the normalization makes this ~automatic.
4. **Drop `sil_coef` 0.5 → 0.1 initially** — the term now shares the
   gradient budget with PPO instead of getting its own 8 clipped steps.
5. Adam moment contamination disappears as a side effect (single step).
6. NOT in this stage (parked, separate concerns): trophy stale-state
   recompute/burn-in; PPO-style ratio clipping of the SIL term against an
   update-start reference (low value while effective epochs = 1; revisit if
   the epoch early-stop stops firing).

**Tests:** adapt `tests/test_sil.py` — (a) merged update performs exactly
one optimizer step per chunk group (count via a spy); (b) trophy log-prob
still rises after a merged update on synthetic data; (c) `sil_enabled:
false` regression unchanged.

**Success metrics (fresh run name, e.g. `v8_silmerge`):** mean KL back
below ~0.05 with the early-stop no longer firing at epoch 1 (epochs_ran
≈ 4); clip fraction < 30%; `grad_norm_actor_head` no longer decaying to
1e-3; `sil_gate_open_fraction` still open early (that's fine) but det eval
no longer declining.

## Stage 2 — Critic recalibration via return scaling (fix 2)

**Problem:** returns live at raw shaped-reward scale (tens); value MSE with
`critic_loss_coef` on the shared trunk gives the critic 3–4 orders of
magnitude more gradient than the actor head; V is pinned ≈ −25 vs realized
≈ +1 (EV 0.12–0.20); and the raw-scale (R−V)+ gate is why SIL never
extinguishes.

**Plan: running return normalization (scale-only, PopArt-lite):**

1. Maintain a running std σ_R of (discounted) returns — reuse the Welford
   pattern from `RunningFeatureNormalizer`.
2. Scale REWARDS by 1/σ_R before GAE so returns, advantages, V, and the SIL
   gate all live in the same O(1) normalized space. (Advantages are already
   re-standardized afterwards; unaffected.)
3. σ_R MUST travel with checkpoints (the count=0 normalizer lesson) and be
   recorded in `effective_config.json` / logs.
4. Do NOT also lower `critic_loss_coef` in the same run (one variable).

**Success metrics:** EV climbing well above 0.2; V(s) tracks cumulative
reward on a trace replay; `grad_norm_critic_head` within ~10× of actor
head; `sil_gate_open_fraction` finally TAPERS as V catches up — the gate's
designed self-extinguishing behavior, observed for the first time.

**Interaction note:** Stage 1's W_MAX clamp and Stage 2's normalization
both address gate scale — after Stage 2, revisit W_MAX (likely keep ~2 in
normalized units as a safety rail).

## Sequencing (one variable at a time)

| Run | Change | Question it answers |
|---|---|---|
| A (probe) | Stage 0 logging only, resume live run | where does KL ≈ 0.9 actually come from? |
| B | Stage 1 merged SIL (+ gate clamp) | does trust-region health return? does det eval stop declining? |
| C | Stage 2 return scaling | does the critic calibrate and the gate self-extinguish? |

Raw-score validation (queued elsewhere) remains the follow-up after B/C —
shaped-up/native-down divergence is the deeper alignment question.
