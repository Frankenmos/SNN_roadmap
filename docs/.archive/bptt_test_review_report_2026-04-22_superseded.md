# BPTT_test branch review

> Archived 2026-06-26.
>
> This review captured real issues at the time, but several headline items are
> now fixed: the click path keeps spatial structure, PPO masks the critic with
> the training mask, rollout cadence is checked inside the step loop, and eval
> happens after pending updates. Use `docs/current/REPO_STATE.md` for the live
> state.

## Executive summary
- The recurrent bookkeeping is better than it first looked. The current agent stores the pre-action recurrent state with each observation, the trainer replays chunks in temporal order, and replay conditions the spatial head on the recorded action IDs. Those parts are mostly coherent.
- The main weaknesses are not in basic TBPTT plumbing. The largest likely performance limit is the policy architecture for spatial action selection: the model compresses all spatial tokens into a global summary before predicting click coordinates. That makes localization unnecessarily hard.
- The biggest correctness issue in the trainer is semantic ambiguity around policy_mask: the actor and entropy losses are masked, but the critic still trains on masked transitions. If masked transitions mean “invalid for learning,” that is a bug. If they mean “actor-only mask,” the name is misleading and the choice should be explicit.
- There are also a few experiment-hygiene issues: evaluation and best-checkpoint selection happen before a pending PPO update, protocol validation is too permissive in places, rollout_steps is only checked at episode boundaries, and checkpoint resume drops any in-flight rollout context.

## What currently looks correct
- The agent attaches the pre-step recurrent state to the PolicyInputBatch before action selection, and that same stored state is used as the chunk initial state during replay.
- The replay path passes recorded action IDs into conditioned_spatial_head(), so the x/y logits are recomputed under the same high-level action that generated the rollout.
- Done rows are reset during packed replay, and the current acting path uses concrete zero states rather than mixing None and tensor states.
- The bootstrap tail observation is prepared with agent.snn_state attached, so the intended recurrent bootstrap contract is present in the training loop.

## Confirmed issues and recommended repairs
### [High] Spatial click head collapses location information too early
- The policy mean-pools each semantic token group, including all spatial tokens, into a single vector before the shared latent and coordinate heads. The x and y logits are then predicted from a global 64-dimensional latent rather than from a spatial map or spatial-token readout.
- That is a strong bottleneck for Smart_screen. The network must compress the full battlefield into one vector and then reconstruct a target location from that summary.
- Recommendation: keep a spatial pathway alive for the coordinate head. Good options are (a) a 2D heatmap head over the screen, (b) a pointer-style head over spatial tokens, or (c) a coarse spatial-token head plus a small refinement module. Let the value head use pooled/global features, but let the click head read real spatial structure.

### [High] Spatial tokens have no explicit positional embedding
- All spatial tokens receive the same token-type embedding. The conv backbone preserves some implicit position, but once the pooled grid is flattened into tokens, there is no explicit row/column or 2D positional signal added before attention.
- This makes it harder for attention and downstream heads to reason cleanly about “where” a token came from.
- Recommendation: add learned 2D positional embeddings (or separable row/column embeddings) to the 7×7 spatial tokens before attention.

### [High] The coordinate head factorizes x and y instead of modeling a joint 2D target
- The policy predicts move_x and move_y independently from the same latent. That cannot represent an arbitrary joint distribution over screen points.
- For targeting tasks, a joint 2D head is usually a better fit than independent x and y marginals.
- Recommendation: replace the separate x/y heads with a joint spatial distribution. A full 84×84 heatmap is the cleanest version; a 7×7 token pointer with optional refinement is the lighter-weight version.

### [High] policy_mask only masks the actor path, not the critic
- In PPO._calculate_losses(), policy_mask gates the policy loss, entropy, KL, and clip diagnostics. But value_loss is still computed as an unconditional mean over all active timesteps.
- If masked transitions mean “do not learn from this transition,” then the critic is still learning from transitions the actor intentionally ignores. In this codebase that matters because unlearnable SMART mismatches are masked out of the actor, but the critic still trains on the resulting rewards and next states.
- Recommendation: decide the semantics explicitly. If these are actor-only masks, rename policy_mask to actor_mask and document it. If these are true training masks, also mask the critic loss and think carefully about whether GAE/returns should exclude masked transitions or use a separate validity channel.

### [Medium] Evaluation and best-checkpoint selection happen before a pending PPO update
- At the end of each episode, the loop may run deterministic evaluation and maybe_save_best_checkpoint() before maybe_run_policy_update().
- That means the “best” checkpoint can correspond to the pre-update policy even when enough rollout data is already waiting in memory for an update.
- Recommendation: run the PPO update before evaluation/checkpoint selection, or explicitly label current evaluation as pre-update and keep a separate post-update checkpoint policy.

### [Medium] Protocol validation is too permissive and can hide bugs
- The protocol is treated as fixed in comments and constants, but several components silently fall back instead of failing fast. Examples: PolicyInputBatch only checks that meta_vec is 2D float, not that it matches the expected protocol width; MetaEncoder falls back to raw meta_vec when structured fields are missing; PPO._policy_action_availability() returns all actions available if meta_vec is too short.
- Those fallbacks are convenient in generic code, but in an experimental branch they can hide observation-protocol drift and make bugs look like learning failures.
- Recommendation: in BPTT_test, prefer strict assertions. Validate exact meta dimensions, validate structured-meta availability, and fail fast if the availability slice is missing.

### [Medium] rollout_steps is only enforced at episode boundaries
- The training loop checks len(agent.ppo.memory) >= rollout_steps only after an episode ends. So rollout_steps behaves as a minimum threshold, not an exact update cadence.
- If episodes are long, the actual rollout size can overshoot the configured value by a large margin.
- Recommendation: either document this clearly and rename it mentally as “minimum rollout steps before an end-of-episode update,” or change the collection loop if you truly want tighter rollout-size control.

### [Medium] Time-cap semantics need an explicit decision
- The code sets done = env_done or time_cap. When time_cap is reached, advantages and bootstrap logic treat the transition as terminal.
- That is correct if steps_per_episode is meant to define a real finite-horizon task boundary. It is incorrect if the cap is only a training convenience and you actually want time-limit truncation with bootstrap.
- Recommendation: decide this explicitly. If the cap is artificial, track separate env_done and time_limit_truncated flags and bootstrap through time-limit truncation.

### [Medium] Checkpoint resume drops in-flight rollout context
- Checkpoints save the model, optimizer, scheduler, episode counters, reward window, and extractor state, but not PPO memory, final_next, or the current recurrent state.
- That means resuming from a checkpoint discards any partially collected on-policy rollout. It is not catastrophic, but it breaks exact continuity.
- Recommendation: either checkpoint only when PPO memory is empty, or also persist the rollout buffer state and current recurrent state if exact resume matters.

### [Medium] The action space is intentionally collapsed to NO_OP + SMART
- This is not a bug, but it is a strong inductive bias. It may be sufficient for DefeatRoaches, but it limits what the policy can express and can blur tactical distinctions between move-like and attack-like behavior.
- Recommendation: if performance plateaus, consider expanding the learned action set or keeping the high-level action small while making the spatial targeting head stronger.

### [Medium] num_steps is SNN micro-time, not environment-time memory
- Inside encode_step_tensors(), num_steps replays the same attended token tensor through the temporal SNN multiple times. It does not extend the environment-time recurrent horizon.
- That is fine if it is intentional, but the name makes it easy to confuse micro-steps with TBPTT sequence length.
- Recommendation: rename it to something like snn_micro_steps and document that TBPTT horizon is controlled by rollout structure and tbptt_window, not by this parameter.

### [Conditional] AMP on recurrent spike state may be too aggressive
- The acting and training paths use autocast with float16 on CUDA, and the recurrent token state is carried through that path. This can be fine, but recurrent/spiking state can be more numerically sensitive than ordinary feed-forward activations.
- Recommendation: if you see instability or NaNs, first test with AMP disabled or with a more conservative mixed-precision policy (for example, keeping recurrent-state updates in float32).

### [Config footgun] Protocol constants assume 7×7 spatial tokens, while the model exposes attention_pool_size as configurable
- policy_protocol.py defines SPATIAL_TOKEN_COUNT = 49 and TOTAL_TOKEN_COUNT accordingly, but PolicyNetwork lets attention_pool_size vary and derives its own token count from that setting.
- As long as attention_pool_size stays 7 this is harmless. If it changes, protocol constants, saved-state assumptions, and comments drift apart.
- Recommendation: either lock attention_pool_size to 7 in this branch or derive the protocol constants from the same source of truth.

## Recommended repair order
### Phase 1 — correctness and experiment hygiene
- Make protocol handling strict: assert exact meta-vector dimensions and remove silent “all actions available” fallbacks in the experiment branch.
- Decide whether policy_mask is really an actor-only mask or a full training-validity mask. Then rename it or extend the masking to the critic and return computation accordingly.
- Move maybe_run_policy_update() ahead of evaluation and best-checkpoint selection if you want “best checkpoint” to refer to the latest trained weights.
- Decide whether steps_per_episode is a true task horizon or a time-limit truncation, and implement the bootstrap logic accordingly.
- Checkpoint only when PPO memory is empty, or extend checkpointing to cover rollout buffer state if exact resume matters.

### Phase 2 — repair the spatial action architecture
- Add explicit 2D positional embeddings to the 7×7 spatial tokens.
- Do not collapse the spatial token group into one mean-pooled vector before the coordinate head.
- Use a joint spatial head (heatmap or token pointer) conditioned on the chosen high-level action.
- Keep the value head global if you like, but let the click head see spatial structure directly.

### Phase 3 — clarity and tests
- Rename num_steps to snn_micro_steps.
- Rename policy_mask to actor_mask if the current semantics are intentional.
- Add a small unit test that compares _replay_packed_chunk_group() against _replay_chunk_group_reference() on synthetic chunks; the reference path already exists and is a good oracle.
- Separate “bootstrap helper steps” from “policy chose SMART but environment fallback occurred” in logging, so helper statistics are easier to interpret.

## Items I did not audit from the provided code
- ObservationExtractor feature scaling, normalization, and exact meta-vector construction.
- ActionSpace.smart() fallback behavior beyond the learnable/non-learnable signal.
- Reward shaping details inside build_reward_function(...).
- Environment wrapper behavior and any PySC2-specific edge cases not visible in the pasted files.

## Bottom line
The recurrent/TBPTT side is in better shape than the branch name might suggest. The most likely reason this agent underperforms is not “fake BPTT,” but that the policy throws away too much spatial information before asking itself to click on the screen. Fix the masking semantics and experiment hygiene first, then repair the spatial head.
