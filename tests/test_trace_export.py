"""Unit tests for tools.analysis.trace_export against a synthetic trace
written by the real recorder (Utility.eval_trace.EpisodeTraceRecorder)."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest
from pysc2.lib import actions as sc2_actions

from MockedEnv.policy_batch import make_policy_batch
from Utility.eval_trace import EpisodeTraceRecorder
from agent_core.policy_protocol import META_VECTOR_DIM, SPATIAL_OBS_SHAPE
from tools.analysis.trace_export import (
    PLAYER_RELATIVE_CHANNEL,
    PR_ENEMY,
    PR_FRIENDLY,
    SCREEN_SIZE,
    SELECTED_CHANNEL,
    export_trace,
    list_traces,
    resolve_trace_path,
    write_trace_json,
)


def _unpack(encoded: str) -> np.ndarray:
    bits = np.unpackbits(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
    )
    return bits[: SCREEN_SIZE * SCREEN_SIZE]


def _batch_with_pixels():
    """Friendly at (x=10, y=20) (also selected), enemy at (x=30, y=40),
    stored the way the extractor stores feature_screen: value / 255."""
    batch = make_policy_batch(
        batch_size=1,
        spatial_shape=SPATIAL_OBS_SHAPE,
        meta_dim=META_VECTOR_DIM,
        zeros=True,
    ).with_state(None)
    batch.spatial_obs[0, PLAYER_RELATIVE_CHANNEL, 20, 10] = PR_FRIENDLY / 255.0
    batch.spatial_obs[0, PLAYER_RELATIVE_CHANNEL, 40, 30] = PR_ENEMY / 255.0
    batch.spatial_obs[0, SELECTED_CHANNEL, 20, 10] = 1.0 / 255.0
    return batch


def _write_trace(tmp_path):
    recorder = EpisodeTraceRecorder(
        output_dir=tmp_path / "run_a" / "episode_traces",
        run_name="run_a",
        checkpoint_path="models/run_a/best_checkpoint.pth",
        checkpoint_episode=99,
        deterministic=True,
    )
    recorder.add_step(
        step_index=0,
        action_func=sc2_actions.FUNCTIONS.select_army("select"),
        action=None,
        move_x=0,
        move_y=0,
        log_prob=0.0,
        value=0.0,
        reward=0.0,
        cumulative_reward=0.0,
        done=False,
        learnable=False,
        policy_input=None,
    )
    recorder.add_step(
        step_index=1,
        action_func=sc2_actions.FUNCTIONS.Smart_screen("now", [10, 20]),
        action=2,
        move_x=10,
        move_y=20,
        log_prob=-0.30123,
        value=1.25,
        reward=0.5,
        cumulative_reward=0.5,
        done=False,
        learnable=True,
        policy_input=_batch_with_pixels(),
    )
    recorder.add_step(
        step_index=2,
        action_func=sc2_actions.FUNCTIONS.no_op(),
        action=0,
        move_x=0,
        move_y=0,
        log_prob=-0.1,
        value=0.75,
        reward=0.0,
        cumulative_reward=0.5,
        done=True,
        learnable=True,
        policy_input=_batch_with_pixels(),
    )
    return recorder.save(episode_index=1, total_reward=0.5, steps=3)


def test_export_trace_schema_and_masks(tmp_path):
    trace_path = _write_trace(tmp_path)
    data = export_trace(trace_path)

    assert data["schema_version"] == 1
    assert data["kind"] == "arch-explorer-trace"
    assert data["run"] == "run_a"
    assert data["mode"] == "det"
    assert data["checkpoint_episode"] == 99
    assert data["total_reward"] == 0.5
    assert len(data["steps"]) == 3

    bootstrap, click, noop = data["steps"]
    # non-policy bootstrap step carries no masks
    assert bootstrap["policy"] is False
    assert "friendly" not in bootstrap
    assert bootstrap["func"] == "select_army"

    assert click["action"] == 2
    assert (click["x"], click["y"]) == (10, 20)
    assert click["func"] == "Smart_screen"
    assert click["learnable"] is True
    assert click["log_prob"] == pytest.approx(-0.3012, abs=1e-4)
    # make_policy_batch(zeros=True) keeps masks all-True: 24 entity slots
    assert click["entities"] == 24
    assert len(click["feedback"]) == 12

    friendly = _unpack(click["friendly"])
    enemy = _unpack(click["enemy"])
    selected = _unpack(click["selected"])
    assert friendly.sum() == 1
    assert friendly[20 * SCREEN_SIZE + 10] == 1  # bit index = y*84 + x
    assert enemy.sum() == 1
    assert enemy[40 * SCREEN_SIZE + 30] == 1
    assert selected.sum() == 1
    assert selected[20 * SCREEN_SIZE + 10] == 1

    assert noop["action"] == 0
    assert noop["done"] is True


def test_write_list_and_resolve(tmp_path):
    _write_trace(tmp_path)

    traces = list_traces("run_a", traces_root=tmp_path)
    assert [path.name for path in traces] == ["episode_0001_det.pt"]

    resolved = resolve_trace_path("run_a", 1, "det", traces_root=tmp_path)
    assert resolved.name == "episode_0001_det.pt"
    with pytest.raises(FileNotFoundError, match="available"):
        resolve_trace_path("run_a", 2, "det", traces_root=tmp_path)

    out_path = tmp_path / "out" / "trace_data.json"
    written = write_trace_json(
        "run_a", episode=1, mode="det", traces_root=tmp_path, out_path=out_path,
    )
    assert written == out_path
    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    assert parsed["kind"] == "arch-explorer-trace"
    assert parsed["steps"][1]["x"] == 10
