# Planned Fixes

Critical read of the recent review notes against the current Fix-3 code.

Goal: separate real correctness issues from stylistic debt and from
review comments that sound plausible but are already handled correctly.

This file is intentionally Socratic: each section starts with the
question we should ask before changing code.

---

## 1. What is actually broken right now?

### Q1. Did we accidentally make `"no action"` and `"no_op"` collide?

**Answer: no, the current indexing scheme is correct.**

Current contract:

- `NO_ACTION_SENTINEL_INDEX = 0`
- real action ids map through `_LAST_ACTION_TO_INDEX = {action_id: idx + 1 ...}`
- `UNKNOWN_LAST_ACTION_INDEX = len(DEFEAT_ROACHES_ACTION_IDS) + 1`

So the index space is:

- `0` = no action recorded this step
- `1..16` = one of the 16 known DefeatRoaches action ids
- `17` = unknown action id

That means `"no_op"` as a real action id is **not** `0` in the embedding
space; it is `1`, because all real actions are offset by `+1`.

**Verdict:** no logic fix needed.

**But:** this contract is too implicit. We should add one small
regression test and one code comment so nobody re-breaks it later.

---

### Q2. Is the last-action meta slot being treated as a scalar magnitude?

**Answer: no, the model currently handles it correctly.**

`obs_space_2.py` stores the last-action slot in `meta_vec` as an
integer-valued float, but `MetaEncoder.forward()` does:

- slice player features
- slice available-action mask
- read the last slot as ids
- `round().long()`
- feed it through `self.last_action_embedding`

So the last-action channel is **not** fed into the meta MLP as a raw
continuous magnitude.

**Verdict:** no correctness fix needed.

**But:** this deserves a regression test because the representation is
heterogeneous by design.

---

### Q3. Is `state_in` validation assuming the wrong memory layout?

**Answer: no, the current SNN state is batch-first.**

The critique is good in general, but our actual state creator is
`TokenTemporalSNN.init_state()` in `policy_network.py`, which returns:

- `syn: [B, N, D]`
- `mem: [B, N, D]`

So the current validation in `PolicyInputBatch._validate()` checking
`shape[0] == batch_size` matches the real implementation.

**Verdict:** no immediate fix needed.

**But:** this is still contract debt. We should document in code that
`state_in` is explicitly `[B, N, D]`, not a generic recurrent-state
tuple.

---

### Q3b. Does the `49 -> 94` token-count change make SNN state or old
checkpoints incompatible?

**Answer: yes for architecture compatibility, no for serialized live
state.**

Two different questions were getting mixed together:

1. **Does the concrete recurrent state shape change?**
   Yes. `TokenTemporalSNN` state is per token:
   - `syn: [B, N, D]`
   - `mem: [B, N, D]`

   and `N` comes from `PolicyNetwork._num_tokens`, which is now:

   - spatial tokens
   - `+ MAX_ENTITY_TOKENS`
   - `+ MAX_SELECTION_TOKENS`
   - `+ 1` meta token

   So the concrete state shape is architecture-dependent and absolutely
   changed when the token budget changed.

2. **Did we miss a checkpoint migration for that live state?**
   No. The runtime `self.snn_state` is not saved in checkpoints.

   Current checkpoints save:
   - `agent_state`
   - optimizer / scheduler state
   - `extractor_state`
   - episode / reward bookkeeping

   On reset, the agent recreates token state from the current
   architecture with `policy.init_concrete_state(...)`.

So the real conclusion is:

- the wording "`state_in` unchanged" was too loose
- the **field** is unchanged in the batch protocol
- the **tensor shape** is not fixed and changes with `N`
- pre-Fix-3 checkpoints are unsupported anyway because the policy
  architecture changed, not because a serialized SNN state migration was
  forgotten

There is one adjacent UX risk, though:

- training resume currently catches a failed `load_state_dict(...)` and
  can rename the checkpoint as `.corrupted`
- for a pre-Fix-3 checkpoint, that failure may mean **incompatible**,
  not corrupted

**Verdict:** not a live Fix-3 runtime bug, but the docs should say this
precisely and checkpoint loading should ideally distinguish
incompatibility from corruption.

**Planned change:**

- update wording so `state_in` means the same protocol slot, not a
  shape-invariant tensor
- explicitly document `state_in` as `[B, N, D]` with `N = current token
  count`
- note that pre-Fix-3 checkpoints are unsupported
- later, improve checkpoint-load messaging so incompatible checkpoints do
  not get mislabeled as corrupted

---

### Q4. Are the running normalizers in `obs_space_2.py` a real footgun?

**Answer: yes, this is the highest-priority real bug.**

Current behavior:

- normalization activates as soon as `count > 1`
- std floor is `1e-6`
- outputs are not clipped

This is dangerous for near-constant fields in DefeatRoaches:

- `energy`, `shield`, `shield_ratio`
- `transport_slots_taken`
- `build_progress`
- `assigned_harvesters`, `ideal_harvesters`

If a field is constant zero for a long time and then becomes slightly
nonzero, dividing by `1e-6` can explode it.

**Verdict:** must fix.

**Planned change:**

1. Add a warm-up threshold before normalization turns on.
2. Use a safer std floor such as `1e-2`.
3. Skip normalization for dimensions whose running variance stays below
   a threshold.
4. Clip normalized outputs to a sane range, e.g. `[-10, 10]`.

This is the first code change we should make.

---

### Q5. Are evaluation episodes mutating the running observation stats?

**Answer: yes, this is a real bug.**

Right now:

- `ObservationExtractor.extract_observation(..., update_stats=True)` is
  the default
- `DefeatRoaches.step(... deterministic=...)` always calls
  `self.extractor.extract_observation(obs)` without changing that default
- deterministic eval in both `run_eval_sweep()` and `PPO_CNN_eval.py`
  therefore updates the same running normalizer state

Consequences:

- eval episodes can change the normalization seen by later training
- eval itself is no longer fully “fixed-policy on fixed-preprocessing”

**Verdict:** must fix.

**Planned change:**

- thread an explicit `update_stats` decision through the agent path
- default to `False` during deterministic eval
- keep `peek_observation()` stateless as it already is

This should happen immediately after the normalizer hardening.

---

### Q6. Can misnamed curated fields silently become all-zero columns?

**Answer: yes, that is a real robustness gap.**

Current behavior in `_project_numeric_rows()`:

- if `index_map.get(field_name)` is `None`
- that feature column just stays zero

That means a misspelling like `hallucinated` vs `hallucination` would
quietly disable a feature instead of failing fast.

For the current code:

- `hallucination` is correct for local `FeatureUnit`
- selection field names also appear to match `UnitLayer`

So there is no proven live bug **today**, but the failure mode is bad.

**Verdict:** must harden.

**Planned change:**

- validate all names in `CURATED_FEATURE_UNIT_FIELDS` against
  `features.FeatureUnit.__members__`
- validate all names in `SELECTION_FEATURE_NAMES` against
  `features.UnitLayer.__members__`
- fail at extractor construction time if a field name is unknown

This should land in the same pass as the normalizer fix.

---

## 2. What is not broken, but still worth cleaning up?

### Q7. Is `vector_input_dim` now a misleading name?

**Answer: yes, but this is clarity debt, not a live correctness bug.**

Today it really means “meta input dim”:

- config still calls it `vector_input_dim`
- `PolicyNetwork.__init__` uses it as `self._meta_input_dim`
- the old vector branch no longer exists

The code works, but the name is stale and confusing.

**Verdict:** change, but after correctness fixes.

**Planned change:**

- rename constructor/config usage to `meta_input_dim`
- keep a short compatibility shim only if needed

This is a medium-priority cleanup, not step 1.

---

### Q8. Should we profile `PolicyInputBatch` validation overhead before
adding an unchecked fast path?

**Answer: yes.**

The review is right that validation runs on many internal transforms:

- `to()`
- `detach()`
- `with_state()`
- `stack()`
- `index_select()`

But there is no evidence yet that this is a bottleneck.

**Verdict:** do not prematurely optimize.

**Planned change:**

- add no fast path yet
- only introduce `_trusted_construct(...)` if profiling says this is hot

---

### Q9. Should we normalize the 11 `player` meta scalars?

**Answer: maybe, but not in the first corrective pass.**

This is a sample-efficiency question, not a correctness bug.

In DefeatRoaches these values are already low-range, and the more urgent
issues are the unsafe entity/selection normalizers and eval-stat drift.

**Verdict:** defer.

Possible later options:

- fixed divide-by-20 for player counts
- tanh squash
- tiny dedicated player normalizer that explicitly excludes the last-action slot

---

### Q10. Should we change `MAX_SELECTION_TOKENS` from 20?

**Answer: no.**

The plan’s observed maximum was 19 and current `K_max = 20` already
adds margin.

There is no evidence that selection truncation is a live issue.

**Verdict:** keep as-is.

---

## 3. What should we change in documentation/comments only?

### Q11. Are the reward comments now partly stale because the old health
term was previously inert?

**Answer: yes.**

The reward function itself is not obviously wrong from this critique
alone, but some of the comment rationale is weaker than it reads,
because it cites runs collected before the health bugfix.

That means:

- the coefficients may still be reasonable
- the evidence text in comments is not as trustworthy as written

**Verdict:** documentation cleanup, not immediate reward redesign.

**Planned change:**

- update comments in `reward_function_2.py`
- clearly say the coefficient choice is now first-principles plus
  post-fix validation, not “proven by the cited old run”

---

## 4. In what order should we change things?

## Phase 1 — correctness hardening in the extractor

Files:

- `obs_space/obs_space_2.py`

Changes:

1. Add enum-name validation for curated entity and selection fields.
2. Harden `RunningFeatureNormalizer`:
   warm-up threshold, safer std floor, variance gating, output clip.
3. Replace `getattr(... ) or 0.0` with an explicit `None` check for clarity.

Why first:

- this is the biggest numerical risk
- it affects every timestep
- it is independent of larger refactors

---

## Phase 2 — isolate eval from training-time observation stats

Files:

- `PPO_CNN_agent.py`
- possibly `PPO_CNN_run.py`
- possibly `PPO_CNN_eval.py`

Changes:

1. Make stat updates explicit in the agent/extractor path.
2. Disable stat updates for deterministic eval.
3. Keep training updates on.

Why second:

- it is a real reproducibility bug
- once normalizers are safer, we should also stop eval from moving them

---

## Phase 3 — lock the implicit contracts with tests

Files:

- `tests/test_policy_input.py`
- `tests/test_agent.py`
- `tests/test_PPO.py`
- new small tests if needed

Changes:

1. Test that `no_action`, `no_op`, and `unknown_action` map to distinct
   last-action indices.
2. Test that `MetaEncoder` accepts the max unknown index without OOB.
3. Test that near-constant normalized features do not explode.
4. Test that deterministic eval does not update extractor stats.

Why third:

- these tests make the hidden contracts explicit
- they reduce the chance of quietly reintroducing the same bugs later

---

## Phase 4 — naming and comment cleanup

Files:

- `PPO_CNN_agent.py`
- `PPO_CNN/policy_network.py`
- `config.yaml`
- `PPO_CNN/reward_function_2.py`

Changes:

1. Rename `vector_input_dim` to `meta_input_dim`.
2. Clarify state layout as `[B, N, D]`.
3. Refresh stale reward comments.

Why fourth:

- these changes improve comprehension
- they are lower-risk and easier once correctness is settled

---

## 5. Which review points do we explicitly reject?

These are the ones we should **not** “fix” just to satisfy the review:

- `TOKEN_TYPE_GROUPS` is not orphaned in practice; it is consumed by
  `policy_network.py`.
- last-action sentinel collision is **not** happening in the current
  implementation.
- the meta last-action slot is **not** being treated as a raw scalar by
  the policy.
- `state_in` validation is not using the wrong dimension for the current
  SNN implementation.
- `MAX_SELECTION_TOKENS = 20` is not an issue worth changing now.
- adding an unchecked `PolicyInputBatch` fast path before profiling would
  be premature.

---

## 6. What should the next coding pass do?

Concrete next pass:

1. Harden `RunningFeatureNormalizer`.
2. Add curated-field validation.
3. Freeze extractor stat updates during deterministic eval.
4. Add regression tests for last-action index separation and normalizer stability.

That is the smallest next patch that materially reduces real Fix-3 risk.
