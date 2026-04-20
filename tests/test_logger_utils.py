import sqlite3

from Utility.logger_utils import LogListener


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
                "type": "UPDATE",
                "internal_ep": 0,
                "episode_index": 12,
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
            },
            {"type": "KILL"},
        ],
    )

    listener = LogListener(queue, str(db_path))
    listener.run()

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT update_wall_seconds, tbptt_chunks, tbptt_chunk_groups, "
        "tbptt_window, tbptt_group_max_steps, tbptt_group_mean_active_chunks, "
        "tbptt_forward_calls FROM ppo_updates",
    ).fetchone()
    conn.close()

    assert row == (3.5, 8, 16, 32, 4, 2.5, 24)
