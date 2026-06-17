"""Tests for merging per-actor running normalizer stats into one set.

Regression cover for the Ray-checkpoint confound: the learner's extractor
never sees observations (count stays 0), so checkpoints must fold the actors'
cumulative stats back in. The merge must be numerically equivalent to having
observed the concatenation of every actor's samples.
"""

import numpy as np
import pytest

from obs_space.obs_space_2 import (
    ObservationExtractor,
    RunningFeatureNormalizer,
)
from distributed.ray_train import _merge_extractor_sync_states


def _make_normalizer():
    return RunningFeatureNormalizer(
        field_names=("a", "b", "c"),
        normalized_fields=("a", "b", "c"),
        min_count_for_normalize=4.0,
    )


def _fed_normalizer(rng, n_rows):
    norm = _make_normalizer()
    rows = rng.normal(loc=[3.0, -1.0, 10.0], scale=[2.0, 0.5, 5.0],
                      size=(n_rows, 3)).astype(np.float32)
    norm.update(rows)
    return norm, rows


def test_merge_matches_single_normalizer_over_concatenated_data():
    rng = np.random.default_rng(0)
    actors = [_fed_normalizer(rng, n) for n in (37, 51, 29, 44)]

    merged = RunningFeatureNormalizer.merge_state_dicts(
        [norm.state_dict() for norm, _ in actors],
    )

    # Ground truth: one normalizer that saw all rows at once.
    reference = _make_normalizer()
    reference.update(np.concatenate([rows for _, rows in actors], axis=0))

    assert merged["count"] == pytest.approx(reference.count)
    np.testing.assert_allclose(merged["mean"], reference.mean, rtol=1e-6)
    np.testing.assert_allclose(merged["m2"], reference.m2, rtol=1e-6)


def test_merge_ignores_empty_and_zero_count_states():
    rng = np.random.default_rng(1)
    real, rows = _fed_normalizer(rng, 40)
    learner_placeholder = _make_normalizer().state_dict()  # count == 0

    merged = RunningFeatureNormalizer.merge_state_dicts(
        [learner_placeholder, None, {}, real.state_dict()],
    )

    assert merged["count"] == pytest.approx(real.count)
    np.testing.assert_allclose(merged["mean"], real.mean, rtol=1e-6)


def test_merge_all_empty_returns_none():
    assert RunningFeatureNormalizer.merge_state_dicts([None, {}]) is None
    assert (
        RunningFeatureNormalizer.merge_state_dicts(
            [_make_normalizer().state_dict()],
        )
        is None
    )


def test_merged_stats_actually_normalize_after_load():
    """End-to-end: a loaded merged state lifts count past the threshold so
    normalize() stops being a passthrough (the eval-time bug)."""
    rng = np.random.default_rng(2)
    actors = [_fed_normalizer(rng, n) for n in (20, 25)]
    merged = RunningFeatureNormalizer.merge_state_dicts(
        [norm.state_dict() for norm, _ in actors],
    )

    fresh = _make_normalizer()
    sample = np.array([[3.0, -1.0, 10.0]], dtype=np.float32)
    # Before loading: count==0, passthrough returns the raw values.
    np.testing.assert_array_equal(fresh.normalize(sample), sample)

    fresh.load_state_dict(merged)
    out = fresh.normalize(sample)
    # After loading real stats, the near-mean row is pulled toward ~0.
    assert np.all(np.abs(out) < np.abs(sample))


def test_extractor_merge_combines_both_normalizers():
    rng = np.random.default_rng(3)
    states = []
    for n in (33, 41):
        ent, _ = _fed_normalizer(rng, n)
        sel, _ = _fed_normalizer(rng, n)
        states.append(
            {
                "entity_normalizer": ent.state_dict(),
                "selection_normalizer": sel.state_dict(),
            },
        )

    merged = ObservationExtractor.merge_state_dicts(states)

    assert merged["entity_normalizer"]["count"] == pytest.approx(74.0)
    assert merged["selection_normalizer"]["count"] == pytest.approx(74.0)


def test_subtract_state_recovers_post_baseline_samples():
    rng = np.random.default_rng(4)
    baseline_norm, baseline_rows = _fed_normalizer(rng, 50)
    actor_norm = _make_normalizer()
    actor_norm.update(baseline_rows)
    new_rows = rng.normal(
        loc=[5.0, 2.0, -3.0],
        scale=[1.0, 2.0, 0.25],
        size=(25, 3),
    ).astype(np.float32)
    actor_norm.update(new_rows)

    delta = RunningFeatureNormalizer.subtract_state_dict(
        actor_norm.state_dict(),
        baseline_norm.state_dict(),
    )
    reference = _make_normalizer()
    reference.update(new_rows)

    assert delta["count"] == pytest.approx(reference.count)
    np.testing.assert_allclose(delta["mean"], reference.mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(delta["m2"], reference.m2, rtol=1e-5, atol=1e-5)


def test_sync_merge_counts_resume_baseline_once():
    rng = np.random.default_rng(5)
    baseline_norm, baseline_rows = _fed_normalizer(rng, 40)
    baseline = {
        "entity_normalizer": baseline_norm.state_dict(),
        "selection_normalizer": baseline_norm.state_dict(),
    }
    actor_states = []
    all_new_rows = []
    for n_rows in (15, 23, 31):
        actor_norm = _make_normalizer()
        actor_norm.update(baseline_rows)
        new_rows = rng.normal(
            loc=[-2.0, 3.0, 7.0],
            scale=[0.75, 1.25, 3.0],
            size=(n_rows, 3),
        ).astype(np.float32)
        actor_norm.update(new_rows)
        all_new_rows.append(new_rows)
        current = {
            "entity_normalizer": actor_norm.state_dict(),
            "selection_normalizer": actor_norm.state_dict(),
        }
        actor_states.append({"baseline": baseline, "current": current})

    merged = _merge_extractor_sync_states(actor_states)

    reference = _make_normalizer()
    reference.update(baseline_rows)
    reference.update(np.concatenate(all_new_rows, axis=0))
    assert merged["entity_normalizer"]["count"] == pytest.approx(reference.count)
    assert merged["selection_normalizer"]["count"] == pytest.approx(reference.count)
    np.testing.assert_allclose(
        merged["entity_normalizer"]["mean"],
        reference.mean,
        rtol=1e-6,
        atol=1e-6,
    )


def test_sync_merge_fresh_run_uses_actor_currents_directly():
    rng = np.random.default_rng(6)
    actor_states = []
    rows_seen = []
    for n_rows in (12, 18):
        norm, rows = _fed_normalizer(rng, n_rows)
        rows_seen.append(rows)
        current = {
            "entity_normalizer": norm.state_dict(),
            "selection_normalizer": norm.state_dict(),
        }
        actor_states.append({"baseline": {}, "current": current})

    merged = _merge_extractor_sync_states(actor_states)

    reference = _make_normalizer()
    reference.update(np.concatenate(rows_seen, axis=0))
    assert merged["entity_normalizer"]["count"] == pytest.approx(reference.count)
    np.testing.assert_allclose(
        merged["entity_normalizer"]["mean"],
        reference.mean,
        rtol=1e-6,
        atol=1e-6,
    )
