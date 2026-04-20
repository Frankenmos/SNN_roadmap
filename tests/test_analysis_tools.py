import sqlite3

import pytest
import torch

from Utility.logger_utils import initialize_db
from tools.analysis.analyze_pth import (
    collect_checkpoint_metadata,
    collect_extractor_state_rows,
    collect_time_constant_rows,
)
from tools.analysis.results import TrainingAnalyzer


def test_training_analyzer_loads_tbptt_and_phase_metrics(tmp_path):
    db_path = tmp_path / "analysis.db"
    conn = initialize_db(str(db_path))
    with conn:
        conn.execute(
            "INSERT INTO episodes (episode_id, total_reward, average_reward, steps) "
            "VALUES (1, 12.0, 2.0, 6)",
        )
        conn.execute(
            "INSERT INTO episodes (episode_id, total_reward, average_reward, steps) "
            "VALUES (2, 9.0, 3.0, 3)",
        )
        step_rows = [
            (1, 0, 0, None, None, 1.0, 1.0),
            (1, 1, 1, 10, 10, 1.0, 2.0),
            (1, 2, 0, None, None, 1.0, 3.0),
            (1, 3, 1, 20, 20, 1.0, 4.0),
            (1, 4, 2, None, None, 1.0, 5.0),
            (1, 5, 0, None, None, 1.0, 6.0),
            (2, 0, 1, 5, 6, 1.0, 1.0),
            (2, 1, 0, None, None, 1.0, 2.0),
            (2, 2, 2, None, None, 1.0, 3.0),
        ]
        conn.executemany(
            "INSERT INTO steps "
            "(episode_id, step_number, action, move_x, move_y, reward, cumulative_reward) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            step_rows,
        )
        conn.execute(
            "INSERT INTO ppo_updates ("
            "episode_id, episode_index, mean_policy_loss, mean_value_loss, "
            "mean_entropy, mean_kl, clip_fraction, explained_variance, grad_norm, lr, "
            "nonfinite_grad_steps, skipped_optimizer_steps, transitions_in_update, "
            "return_mean, return_std, return_p10, return_p50, return_p90, "
            "entity_mask_utilization, entity_count_p50, entity_count_p99, "
            "selection_mask_utilization, update_wall_seconds, tbptt_chunks, "
            "tbptt_chunk_groups, tbptt_window, tbptt_group_max_steps, "
            "tbptt_group_mean_active_chunks, tbptt_forward_calls"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                2,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                1.0e-4,
                0,
                0,
                64,
                1.0,
                0.5,
                0.1,
                0.9,
                1.2,
                0.4,
                10.0,
                20.0,
                0.2,
                3.5,
                8,
                16,
                32,
                4,
                2.5,
                24,
            ),
        )
        conn.execute(
            "INSERT INTO reward_components ("
            "episode_id, step, health_reward, engagement_reward, "
            "positioning_reward, score_reward, bonus_reward, "
            "end_of_episode_reward, total_reward"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, 0, -1.0, 2.0, 0.0, 0.5, 0.0, 0.0, 1.5),
        )
        conn.execute(
            "INSERT INTO eval_runs ("
            "episode_index, num_episodes, mean_reward, std_reward, "
            "min_reward, max_reward, deterministic"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (2, 5, 12.0, 1.0, 10.0, 13.0, 0),
        )
        conn.execute(
            "INSERT INTO eval_runs ("
            "episode_index, num_episodes, mean_reward, std_reward, "
            "min_reward, max_reward, deterministic"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (2, 5, 8.0, 1.5, 6.0, 9.0, 1),
        )
    conn.close()

    analyzer = TrainingAnalyzer(str(db_path))
    try:
        steps_df = analyzer.get_step_metrics()
        updates_df = analyzer.get_update_metrics()
        phase_mix_df = analyzer.action_mix_by_episode_phase(num_bins=2)
    finally:
        analyzer.close()

    assert {"move_x", "move_y"}.issubset(set(steps_df.columns))
    assert {
        "update_wall_seconds",
        "tbptt_chunks",
        "tbptt_chunk_groups",
        "tbptt_window",
        "tbptt_group_max_steps",
        "tbptt_group_mean_active_chunks",
        "tbptt_forward_calls",
    }.issubset(set(updates_df.columns))
    assert set(phase_mix_df["phase"].astype(str).unique()) == {"early", "mid", "late"}
    probs = phase_mix_df.groupby(["phase", "bin"], observed=False)["prob"].sum()
    assert probs.apply(lambda value: pytest.approx(1.0) == value).all()


def test_training_analyzer_exports_ai_friendly_panels(tmp_path):
    db_path = tmp_path / "analysis.db"
    conn = initialize_db(str(db_path))
    with conn:
        conn.execute(
            "INSERT INTO episodes (episode_id, total_reward, average_reward, steps) "
            "VALUES (1, 12.0, 2.0, 6)",
        )
        conn.execute(
            "INSERT INTO episodes (episode_id, total_reward, average_reward, steps) "
            "VALUES (2, 18.0, 3.0, 9)",
        )
        conn.executemany(
            "INSERT INTO steps "
            "(episode_id, step_number, action, move_x, move_y, reward, cumulative_reward) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 0, 0, None, None, 1.0, 1.0),
                (1, 1, 1, 10, 11, 1.0, 2.0),
                (1, 2, 2, None, None, 1.0, 3.0),
                (1, 3, 0, None, None, 1.0, 4.0),
                (1, 4, 1, 15, 16, 1.0, 5.0),
                (1, 5, 0, None, None, 1.0, 6.0),
                (2, 0, 0, None, None, 1.0, 1.0),
                (2, 1, 1, 20, 21, 1.0, 2.0),
                (2, 2, 0, None, None, 1.0, 3.0),
                (2, 3, 2, None, None, 1.0, 4.0),
                (2, 4, 0, None, None, 1.0, 5.0),
                (2, 5, 1, 22, 23, 1.0, 6.0),
                (2, 6, 0, None, None, 1.0, 7.0),
                (2, 7, 2, None, None, 1.0, 8.0),
                (2, 8, 0, None, None, 1.0, 9.0),
            ],
        )
        conn.execute(
            "INSERT INTO ppo_updates ("
            "episode_id, episode_index, mean_policy_loss, mean_value_loss, "
            "mean_entropy, mean_kl, clip_fraction, explained_variance, grad_norm, lr, "
            "nonfinite_grad_steps, skipped_optimizer_steps, transitions_in_update, "
            "return_mean, return_std, return_p10, return_p50, return_p90, "
            "entity_mask_utilization, entity_count_p50, entity_count_p99, "
            "selection_mask_utilization, update_wall_seconds, tbptt_chunks, "
            "tbptt_chunk_groups, tbptt_window, tbptt_group_max_steps, "
            "tbptt_group_mean_active_chunks, tbptt_forward_calls"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                2,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                1.0e-4,
                0,
                0,
                64,
                1.0,
                0.5,
                0.1,
                0.9,
                1.2,
                0.4,
                10.0,
                20.0,
                0.2,
                3.5,
                8,
                16,
                32,
                4,
                2.5,
                24,
            ),
        )
        conn.execute(
            "INSERT INTO reward_components ("
            "episode_id, step, health_reward, engagement_reward, "
            "positioning_reward, score_reward, bonus_reward, "
            "end_of_episode_reward, total_reward"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, 0, -1.0, 2.0, 0.0, 0.5, 0.0, 0.0, 1.5),
        )
        conn.execute(
            "INSERT INTO eval_runs ("
            "episode_index, num_episodes, mean_reward, std_reward, "
            "min_reward, max_reward, deterministic"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (2, 5, 12.0, 1.0, 10.0, 13.0, 0),
        )
        conn.execute(
            "INSERT INTO eval_runs ("
            "episode_index, num_episodes, mean_reward, std_reward, "
            "min_reward, max_reward, deterministic"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (2, 5, 8.0, 1.5, 6.0, 9.0, 1),
        )
    conn.close()

    analyzer = TrainingAnalyzer(str(db_path))
    try:
        out_dir = tmp_path / "ai_friendly_results"
        exported = analyzer.export_ai_friendly_panels(
            str(out_dir),
            window=2,
            num_bins=2,
            plateau_ep=1,
        )
    finally:
        analyzer.close()

    assert "01_reward_trajectory.png" in exported
    assert "07_phase_action_mix.png" in exported
    assert "09_tbptt_speed.png" in exported
    assert "10_eval_split.png" in exported
    assert "11_eval_gap.png" in exported
    assert (out_dir / "manifest.txt").exists()


def test_checkpoint_helpers_surface_metadata_extractor_state_and_time_constants():
    ckpt = {
        "episode": 42,
        "best_eval_reward": 123.0,
        "avg_reward_at_save": 100.0,
        "optimizer_state": {"param_groups": []},
        "scheduler_state": {"last_epoch": 3},
        "extractor_state": {
            "entity_normalizer": {
                "count": 64.0,
                "mean": [1.0, 2.0],
                "m2": [63.0, 252.0],
            },
            "selection_normalizer": {
                "count": 16.0,
                "mean": [3.0],
                "m2": [15.0],
            },
        },
        "agent_state": {
            "token_snn.snn.alpha": torch.tensor([0.8]),
            "token_snn.snn.beta": torch.tensor([0.9]),
            "attention.lif_q.beta": torch.tensor([0.5]),
            "fc.weight": torch.randn(2, 3),
        },
    }

    metadata = collect_checkpoint_metadata(ckpt)
    extractor_rows = collect_extractor_state_rows(ckpt)
    time_constant_rows = collect_time_constant_rows(ckpt["agent_state"])

    assert metadata["episode"] == 42
    assert metadata["has_extractor_state"] is True
    assert metadata["has_optimizer_state"] is True
    assert metadata["state_tensor_count"] == 4
    assert metadata["parameter_count"] == 9

    entity_summary = extractor_rows["entity_normalizer"]
    assert entity_summary["count"] == 64.0
    assert entity_summary["warm"] is True
    assert len(entity_summary["rows"]) == 2
    assert entity_summary["rows"][0]["field"] == "health"
    assert entity_summary["rows"][0]["std"] > 0.0

    selection_summary = extractor_rows["selection_normalizer"]
    assert selection_summary["count"] == 16.0
    assert selection_summary["warm"] is False

    assert {row["name"] for row in time_constant_rows} == {
        "token_snn.snn.alpha",
        "token_snn.snn.beta",
        "attention.lif_q.beta",
    }
