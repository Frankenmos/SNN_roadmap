import sqlite3

import pytest

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
            {"type": "EPISODE_START", "internal_ep": 0},
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
                "policy_protocol_version": 2,
                "policy_input_schema": "stream_action_feedback_v1",
                "rew": 1.5,
                "cum_rew": 1.5,
            },
            {
                "type": "UPDATE",
                "internal_ep": 0,
                "episode_index": 12,
                "global_update_index": 2,
                "policy_version": 4,
                "policy_protocol_version": 2,
                "policy_input_schema": "stream_action_feedback_v1",
                "mean_policy_loss": 0.1,
                "mean_value_loss": 0.2,
                "mean_entropy": 0.3,
                "mean_kl": 0.4,
                "clip_fraction": 0.5,
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
    conn.close()

    assert update_row == (
        2,
        4,
        2,
        "stream_action_feedback_v1",
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
    assert step_row == (3, 4, 5, 2, "stream_action_feedback_v1")


def test_safe_add_column_does_not_hide_real_schema_errors(tmp_path):
    db_path = tmp_path / "schema.db"
    conn = initialize_db(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            _safe_add_column(conn, "missing_table", "bad INTEGER")
    finally:
        conn.close()
