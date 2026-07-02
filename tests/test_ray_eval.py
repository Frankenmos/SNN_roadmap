"""Tests for deterministic evaluation in the Ray training path.

Covers the three things that, if subtly wrong, would silently produce
misleading eval numbers:

1. Borrowing a training actor for eval must leave it in a clean state so the
   next collect_fragment cannot splice pre-eval transitions onto a fresh
   episode (RolloutActor.run_eval reset contract).
2. Eval must never mutate the policy version / update count.
3. The extractor sync MUST run before the best-checkpoint save, otherwise
   best_checkpoint.pth re-ships the count=0 normalizer confound this whole
   feature is meant to make measurable.

All tests use fakes - no Ray, no SC2, no torch model.
"""

import numpy as np
import pytest

from distributed.ray_actor import RolloutActor
from distributed.ray_train import (
    _aggregate_eval_summaries,
    _run_eval_and_maybe_save,
    _split_episodes,
)


def _summary(mean, *, n=1, std=0.0, lo=None, hi=None, deterministic=True):
    return {
        "num_episodes": n,
        "mean_reward": mean,
        "std_reward": std,
        "min_reward": mean if lo is None else lo,
        "max_reward": mean if hi is None else hi,
        "deterministic": deterministic,
    }


# --------------------------------------------------------------------------- #
# RolloutActor.run_eval reset contract
# --------------------------------------------------------------------------- #


class _FakePPO:
    def __init__(self):
        self.memory = [{"transition": 1}]
        self.final_next = object()
        self.pending_fragments = [{"fragment": 1}]
        self.update_count = 7

    def _clear_rollout_cache(self):
        self.memory = []
        self.final_next = None
        self.pending_fragments = []


class _FakePolicy:
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1

    def init_concrete_state(self, batch_size, device):
        return ("snn_state", batch_size, device)


class _FakeAgent:
    def __init__(self):
        self.ppo = _FakePPO()
        self.policy = _FakePolicy()
        self.snn_state = "stale_state"


class _FakeWorker:
    def __init__(self, serialize=False):
        self.current_obs = object()
        self.step_count = 99
        self.episode_reward = 5.0
        self.cumulative_reward = 9.0
        self.serialize_env_resets = serialize


def _make_actor(agent, worker, device="cpu"):
    # Bypass __init__ (it needs Ray/SC2/config); set only what run_eval touches.
    actor = RolloutActor.__new__(RolloutActor)
    actor.agent = agent
    actor.worker = worker
    actor.env = object()
    actor.device = device
    actor.policy_version = 3
    return actor


def test_run_eval_resets_worker_and_ppo(monkeypatch):
    import train

    captured = {}

    def fake_sweep(env, agent, episodes, steps, deterministic=True,
                   reset_lock=None):
        captured["call"] = (episodes, steps, deterministic)
        captured["reset_lock"] = reset_lock
        return _summary(2.0)

    monkeypatch.setattr(train, "run_eval_sweep", fake_sweep)

    agent = _FakeAgent()
    worker = _FakeWorker()
    actor = _make_actor(agent, worker)

    out = actor.run_eval(1, 200)

    assert out == _summary(2.0)
    assert captured["call"] == (1, 200, True)
    # The cross-process SC2 reset lock factory is threaded into the sweep.
    assert callable(captured["reset_lock"])

    # Worker is forced to start a fresh episode on the next collect_fragment.
    assert worker.current_obs is None
    assert worker.step_count == 0
    assert worker.episode_reward == 0.0
    assert worker.cumulative_reward == 0.0

    # PPO rollout cache is cleared so no pre-eval transition can splice in.
    assert agent.ppo.memory == []
    assert agent.ppo.pending_fragments == []
    assert agent.ppo.final_next is None

    # Recurrent state re-initialized for batch=1 on the actor device.
    assert agent.snn_state == ("snn_state", 1, "cpu")
    assert agent.policy.eval_calls >= 1


def test_run_eval_resets_even_when_sweep_raises(monkeypatch):
    import train

    def boom(*args, **kwargs):
        raise RuntimeError("eval boom")

    monkeypatch.setattr(train, "run_eval_sweep", boom)

    agent = _FakeAgent()
    worker = _FakeWorker()
    actor = _make_actor(agent, worker)

    with pytest.raises(RuntimeError, match="eval boom"):
        actor.run_eval(1, 200)

    # The finally block must still restore a clean state.
    assert worker.current_obs is None
    assert agent.ppo.memory == []
    assert agent.ppo.final_next is None


def test_run_eval_preserves_versions(monkeypatch):
    import train

    monkeypatch.setattr(train, "run_eval_sweep",
                        lambda *a, **k: _summary(0.0))

    agent = _FakeAgent()
    actor = _make_actor(agent, _FakeWorker())

    actor.run_eval(1, 50)

    assert agent.ppo.update_count == 7      # only update_policy may bump this
    assert actor.policy_version == 3        # eval never re-versions the actor


# --------------------------------------------------------------------------- #
# Summary helpers
# --------------------------------------------------------------------------- #


def test_split_episodes_even_and_clamped():
    assert _split_episodes(5, 5) == [1, 1, 1, 1, 1]

    counts = _split_episodes(7, 5)
    assert sum(counts) == 7
    assert len(counts) == 5
    assert max(counts) - min(counts) <= 1

    # More actors than episodes -> clamp to one slot per episode.
    assert _split_episodes(3, 5) == [1, 1, 1]
    # Fewer actors than episodes -> pack evenly, biggest slots first.
    assert _split_episodes(10, 3) == [4, 3, 3]


def test_aggregate_matches_numpy_one_episode_per_actor():
    rewards = [10.0, 20.0, 30.0]
    summaries = [_summary(r) for r in rewards]

    agg = _aggregate_eval_summaries(summaries)
    arr = np.asarray(rewards)

    assert agg["num_episodes"] == 3
    assert agg["mean_reward"] == pytest.approx(arr.mean())
    assert agg["std_reward"] == pytest.approx(arr.std())
    assert agg["min_reward"] == pytest.approx(arr.min())
    assert agg["max_reward"] == pytest.approx(arr.max())
    assert agg["deterministic"] is True


def test_aggregate_weighted_mean_and_empty():
    summaries = [
        _summary(0.0, n=1),
        _summary(8.0, n=3, std=1.0, lo=5.0, hi=11.0),
    ]
    agg = _aggregate_eval_summaries(summaries)

    assert agg["num_episodes"] == 4
    assert agg["mean_reward"] == pytest.approx((0.0 * 1 + 8.0 * 3) / 4)
    assert agg["min_reward"] == pytest.approx(0.0)
    assert agg["max_reward"] == pytest.approx(11.0)

    # None/empty placeholders collapse to safe zeros, no crash.
    empty = _aggregate_eval_summaries([None, {}])
    assert empty["num_episodes"] == 0
    assert empty["mean_reward"] == 0.0


# --------------------------------------------------------------------------- #
# Driver wiring + confound guard
# --------------------------------------------------------------------------- #


class _FakeExtractor:
    def __init__(self):
        self.count = 0  # starts confounded (count=0), like a fresh learner


class _FakeLearnerAgent:
    def __init__(self):
        self.extractor = _FakeExtractor()


class _FakeLearner:
    def __init__(self):
        self.agent = _FakeLearnerAgent()
        self.policy_version = 42


class _FakeRemote:
    def __init__(self, summary, calls):
        self.summary = summary
        self.calls = calls

    def remote(self, count, steps):
        self.calls.append(("run_eval", count, steps))
        return self.summary


class _FakeActor:
    def __init__(self, summary, calls):
        self.run_eval = _FakeRemote(summary, calls)


class _FakeRay:
    @staticmethod
    def get(refs):
        return list(refs)


class _FakeQueue:
    def __init__(self):
        self.records = []

    def put(self, record):
        self.records.append(record)


def test_eval_syncs_before_best_save_and_threads_reward(monkeypatch):
    import train
    import distributed.ray_train as rt

    calls = []

    # sync spy: records order AND de-confounds the extractor (count 0 -> 74),
    # so a later best-save observing count==74 proves the sync ran first.
    def fake_sync(ray, actors, learner):
        calls.append(("sync", learner.agent.extractor.count))
        learner.agent.extractor.count = 74

    monkeypatch.setattr(rt, "_sync_extractor_state_from_actors", fake_sync)

    seen = {}

    def fake_best(agent, episode, avg_reward, eval_summary, best,
                  episode_rewards):
        calls.append(("best", agent.extractor.count))
        seen["best_in"] = best
        seen["episode"] = episode
        return 999.0  # report a new best

    monkeypatch.setattr(train, "maybe_save_best_checkpoint", fake_best)

    summary = _summary(12.0)
    actors = [_FakeActor(summary, calls) for _ in range(3)]
    learner = _FakeLearner()
    queue = _FakeQueue()

    new_best = _run_eval_and_maybe_save(
        _FakeRay,
        actors,
        learner,
        log_queue=queue,
        episode_index=100,
        episode_rewards=[1.0, 2.0, 3.0],
        best_eval_reward=5.0,
        num_actors=3,
        eval_episodes=2,
        eval_steps=200,
        update_index=49,
    )

    # Best reward is threaded back out for the next checkpoint save.
    assert new_best == 999.0
    assert seen["best_in"] == 5.0
    assert seen["episode"] == 100

    # Exactly one EVAL record, carrying the pooled summary + protocol fields.
    assert len(queue.records) == 1
    record = queue.records[0]
    assert record["type"] == "EVAL"
    assert record["policy_version"] == 42
    assert record["episode_index"] == 100
    assert record["mean_reward"] == pytest.approx(12.0)

    # CRITICAL: sync precedes best-save, and best-save sees a de-confounded
    # (count>0) extractor - no count=0 normalizer can reach best_checkpoint.pth.
    order = [c[0] for c in calls if c[0] in ("sync", "best")]
    assert order == ["sync", "best"]
    best_call = next(c for c in calls if c[0] == "best")
    assert best_call[1] == 74

    # Eval borrowed min(eval_episodes, num_actors) = 2 actors, 1 episode each.
    run_eval_calls = [c for c in calls if c[0] == "run_eval"]
    assert len(run_eval_calls) == 2
    assert all(c[1] == 1 and c[2] == 200 for c in run_eval_calls)
