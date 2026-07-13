import sqlite3

import pytest

from agent_core.policy_protocol import POLICY_INPUT_SCHEMA, POLICY_PROTOCOL_VERSION
from Utility.logger_utils import LogListener, _safe_add_column, initialize_db


class _StaticQueue:
    def __init__(self, records):
        self._records = list(records)

    def get(self, timeout=1):
        del timeout
        if not self._records:
            raise RuntimeError("queue exhausted before KILL")
        return self._records.pop(0)


def test_log_listener_persists_tbptt_update_metrics(tmp_path):
    db_path = tmp_path / "metrics.db"
    queue = _StaticQueue(
        [
            {
                "type": "EPISODE_START",
                "internal_ep": 0,
                "phase_id": 3,
                "actor_id": 3,
                "policy_version": 4,
            },
            {
                "type": "STEP",
                "internal_ep": 0,
                "step": 1,
                "act": 2,
                "move_x": 10,
                "move_y": 20,
                "actor_id": 3,
                "policy_version": 4,
                "fragment_id": 5,
                "policy_protocol_version": POLICY_PROTOCOL_VERSION,
                "policy_input_schema": POLICY_INPUT_SCHEMA,
                "rew": 1.5,
                "cum_rew": 1.5,
            },
            {
                "type": "UPDATE",
                "phase_id": 3,
                "internal_ep": 0,
                "episode_index": 12,
                "global_update_index": 2,
                "policy_version": 4,
                "policy_protocol_version": POLICY_PROTOCOL_VERSION,
                "policy_input_schema": POLICY_INPUT_SCHEMA,
                "mean_policy_loss": 0.1,
                "mean_value_loss": 0.2,
                "mean_entropy": 0.3,
                "mean_kl": 0.4,
                "clip_fraction": 0.5,
                "epochs_ran": 2,
                "kl_update_start": 0.04,
                "log_ratio_update_start_p99": 0.09,
                "explained_variance": 0.6,
                "grad_norm": 0.7,
                "lr": 1e-4,
                "nonfinite_grad_steps": 0,
                "skipped_optimizer_steps": 0,
                "transitions_in_update": 64,
                "learnable_transitions_in_update": 60,
                "fragments_in_update": 3,
                "return_mean": 1.0,
                "return_std": 2.0,
                "return_p10": 0.1,
                "return_p50": 1.0,
                "return_p90": 2.0,
                "entity_mask_utilization": 0.4,
                "entity_count_p50": 10.0,
                "entity_count_p99": 20.0,
                "selection_mask_utilization": 0.2,
                "update_wall_seconds": 3.5,
                "tbptt_chunks": 8,
                "tbptt_chunk_groups": 16,
                "tbptt_window": 32,
                "tbptt_group_max_steps": 4,
                "tbptt_group_mean_active_chunks": 2.5,
                "tbptt_forward_calls": 24,
                "rollout_wall_seconds": 9.0,
                "ray_get_wall_seconds": 8.0,
                "fragment_validation_wall_seconds": 0.01,
                "cpu_to_gpu_transfer_wall_seconds": 0.25,
                "chunk_pack_wall_seconds": 0.5,
                "replay_forward_wall_seconds": 1.5,
                "backward_optimizer_wall_seconds": 2.5,
                "payload_spatial_bytes": 1024,
                "payload_state_bytes": 256,
                "payload_total_bytes": 2048,
                "payload_total_mib": 2.0 / 1024.0,
                "cuda_peak_allocated_bytes": 4096,
                "cuda_peak_reserved_bytes": 8192,
                "rollout_cache_spatial_dtype": "float32",
            },
            {
                "type": "EPISODE_END",
                "internal_ep": 0,
                "total": 7.5,
                "native_reward": 1.0,
                "avg": 7.5,
                "steps": 12,
                "terminated": False,
                "truncated": True,
                "reward_components": {"engagement_reward": 2.5},
                "policy_no_op_count": 2,
                "policy_left_click_count": 1,
                "policy_right_click_count": 9,
            },
            {
                "type": "EVAL",
                "phase_id": 3,
                "eval_group_id": "eval-abc",
                "episode_index": 12,
                "num_episodes": 1,
                "mean_reward": 5.0,
                "std_reward": 0.0,
                "min_reward": 5.0,
                "max_reward": 5.0,
                "deterministic": True,
                "policy_version": 4,
                "policy_protocol_version": POLICY_PROTOCOL_VERSION,
                "policy_input_schema": POLICY_INPUT_SCHEMA,
                "episode_results": [
                    {
                        "episode_number": 0,
                        "actor_id": 3,
                        "native_reward": 5.0,
                        "steps": 18,
                        "terminated": True,
                        "truncated": False,
                        "policy_right_click_count": 6,
                    }
                ],
            },
            {"type": "KILL"},
        ],
    )

    listener = LogListener(queue, str(db_path))
    listener.run()

    conn = sqlite3.connect(db_path)
    update_row = conn.execute(
        "SELECT global_update_index, policy_version, policy_protocol_version, "
        "policy_input_schema, transitions_in_update, "
        "learnable_transitions_in_update, fragments_in_update, "
        "update_wall_seconds, tbptt_chunks, tbptt_chunk_groups, tbptt_window, "
        "tbptt_group_max_steps, tbptt_group_mean_active_chunks, tbptt_forward_calls, "
        "ray_get_wall_seconds, cpu_to_gpu_transfer_wall_seconds, "
        "chunk_pack_wall_seconds, replay_forward_wall_seconds, "
        "backward_optimizer_wall_seconds, payload_total_bytes, "
        "cuda_peak_allocated_bytes, rollout_cache_spatial_dtype "
        "FROM ppo_updates",
    ).fetchone()
    step_row = conn.execute(
        "SELECT actor_id, policy_version, fragment_id, policy_protocol_version, "
        "policy_input_schema FROM steps",
    ).fetchone()
    episode_row = conn.execute(
        "SELECT phase_id, shaped_reward, native_reward, terminated, truncated, "
        "reward_components_json, policy_right_click_count FROM episodes",
    ).fetchone()
    eval_run_row = conn.execute(
        "SELECT eval_group_id, phase_id, num_episodes FROM eval_runs",
    ).fetchone()
    eval_episode_row = conn.execute(
        "SELECT eval_group_id, phase_id, actor_id, native_reward, steps, "
        "terminated, truncated, policy_right_click_count FROM eval_episodes",
    ).fetchone()
    conn.close()

    assert update_row == (
        2,
        4,
        POLICY_PROTOCOL_VERSION,
        POLICY_INPUT_SCHEMA,
        64,
        60,
        3,
        3.5,
        8,
        16,
        32,
        4,
        2.5,
        24,
        8.0,
        0.25,
        0.5,
        1.5,
        2.5,
        2048,
        4096,
        "float32",
    )
    assert step_row == (3, 4, 5, POLICY_PROTOCOL_VERSION, POLICY_INPUT_SCHEMA)
    assert episode_row == (3, 7.5, 1.0, 0, 1, '{"engagement_reward": 2.5}', 9)
    assert eval_run_row == ("eval-abc", 3, 1)
    assert eval_episode_row == ("eval-abc", 3, 3, 5.0, 18, 1, 0, 6)


def test_safe_add_column_does_not_hide_real_schema_errors(tmp_path):
    db_path = tmp_path / "schema.db"
    conn = initialize_db(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            _safe_add_column(conn, "missing_table", "bad INTEGER")
    finally:
        conn.close()


def test_initialize_db_migrates_legacy_tables_without_losing_rows(tmp_path):
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db_path)
    with legacy:
        legacy.execute(
            "CREATE TABLE episodes (episode_id INTEGER PRIMARY KEY, "
            "total_reward REAL, average_reward REAL, steps INTEGER)",
        )
        legacy.execute(
            "INSERT INTO episodes VALUES (1, 12.5, 6.25, 2)",
        )
        legacy.execute(
            "CREATE TABLE ppo_updates (update_id INTEGER PRIMARY KEY, "
            "episode_id INTEGER)",
        )
        legacy.execute(
            "INSERT INTO ppo_updates VALUES (1, 1)",
        )
        legacy.execute(
            "CREATE TABLE eval_runs (eval_id INTEGER PRIMARY KEY, "
            "episode_index INTEGER, num_episodes INTEGER, mean_reward REAL, "
            "std_reward REAL, min_reward REAL, max_reward REAL, "
            "deterministic INTEGER)",
        )
        legacy.execute(
            "INSERT INTO eval_runs VALUES (1, 10, 1, 3.0, 0.0, 3.0, 3.0, 1)",
        )
    legacy.close()

    migrated = initialize_db(db_path)
    try:
        episode_columns = {
            row[1] for row in migrated.execute("PRAGMA table_info(episodes)")
        }
        update_columns = {
            row[1] for row in migrated.execute("PRAGMA table_info(ppo_updates)")
        }
        eval_columns = {
            row[1] for row in migrated.execute("PRAGMA table_info(eval_runs)")
        }
        tables = {
            row[0]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            )
        }
        assert {"phase_id", "shaped_reward", "native_reward"} <= episode_columns
        assert {
            "epochs_ran",
            "update_start_scope",
            "mean_kl",
            "sil_age_p90",
        } <= update_columns
        assert {"eval_group_id", "phase_id", "policy_version"} <= eval_columns
        assert "eval_episodes" in tables
        assert migrated.execute(
            "SELECT episode_id, total_reward, average_reward, steps FROM episodes",
        ).fetchone() == (1, 12.5, 6.25, 2)
        assert migrated.execute(
            "SELECT eval_id, mean_reward FROM eval_runs",
        ).fetchone() == (1, 3.0)
    finally:
        migrated.close()
