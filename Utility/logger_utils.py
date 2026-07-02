import multiprocessing
import sqlite3
import time
from queue import Empty


PPO_UPDATE_COLUMNS = [
    ("episode_index", "INTEGER"),
    ("global_update_index", "INTEGER"),
    ("policy_version", "INTEGER"),
    ("policy_protocol_version", "INTEGER"),
    ("policy_input_schema", "TEXT"),
    ("mean_policy_loss", "REAL"),
    ("mean_value_loss", "REAL"),
    ("mean_entropy", "REAL"),
    ("mean_kl", "REAL"),
    ("clip_fraction", "REAL"),
    ("explained_variance", "REAL"),
    ("grad_norm", "REAL"),
    ("grad_norm_trunk", "REAL"),
    ("grad_norm_actor_head", "REAL"),
    ("grad_norm_critic_head", "REAL"),
    ("grad_norm_target_head", "REAL"),
    ("sil_loss", "REAL"),
    ("sil_gate_open_fraction", "REAL"),
    ("sil_buffer_size", "INTEGER"),
    ("sil_steps_replayed", "INTEGER"),
    ("sil_groups", "INTEGER"),
    ("sil_grad_norm", "REAL"),
    ("lr", "REAL"),
    ("nonfinite_grad_steps", "INTEGER"),
    ("skipped_optimizer_steps", "INTEGER"),
    ("transitions_in_update", "INTEGER"),
    ("learnable_transitions_in_update", "INTEGER"),
    ("fragments_in_update", "INTEGER"),
    ("return_mean", "REAL"),
    ("return_std", "REAL"),
    ("return_p10", "REAL"),
    ("return_p50", "REAL"),
    ("return_p90", "REAL"),
    ("entity_mask_utilization", "REAL"),
    ("entity_count_p50", "REAL"),
    ("entity_count_p99", "REAL"),
    ("selection_mask_utilization", "REAL"),
    ("update_wall_seconds", "REAL"),
    ("tbptt_chunks", "INTEGER"),
    ("tbptt_chunk_groups", "INTEGER"),
    ("tbptt_window", "INTEGER"),
    ("tbptt_group_max_steps", "INTEGER"),
    ("tbptt_group_mean_active_chunks", "REAL"),
    ("tbptt_forward_calls", "INTEGER"),
    ("rollout_wall_seconds", "REAL"),
    ("ray_get_wall_seconds", "REAL"),
    ("ray_submit_wall_seconds", "REAL"),
    ("rollout_collect_overhead_wall_seconds", "REAL"),
    ("rollout_collect_waves", "INTEGER"),
    ("rollout_empty_waves", "INTEGER"),
    ("rollout_steps_collected", "INTEGER"),
    ("rollout_policy_no_op_count", "INTEGER"),
    ("rollout_policy_left_click_count", "INTEGER"),
    ("rollout_policy_right_click_count", "INTEGER"),
    ("rollout_feedback_smart_executed_count", "INTEGER"),
    ("rollout_feedback_near_enemy_smart_count", "INTEGER"),
    ("rollout_feedback_moved_toward_target_count", "INTEGER"),
    ("rollout_feedback_enemy_health_drop_after_smart_count", "INTEGER"),
    ("rollout_feedback_null_unclear_smart_count", "INTEGER"),
    ("rollout_actor_count", "INTEGER"),
    ("rollout_fragments_collected", "INTEGER"),
    ("fragment_validation_wall_seconds", "REAL"),
    ("learner_update_from_fragments_wall_seconds", "REAL"),
    ("fragment_tensor_build_wall_seconds", "REAL"),
    ("cpu_to_gpu_transfer_wall_seconds", "REAL"),
    ("bootstrap_value_wall_seconds", "REAL"),
    ("gae_wall_seconds", "REAL"),
    ("tbptt_chunk_build_wall_seconds", "REAL"),
    ("chunk_pack_wall_seconds", "REAL"),
    ("replay_forward_wall_seconds", "REAL"),
    ("loss_eval_wall_seconds", "REAL"),
    ("backward_optimizer_wall_seconds", "REAL"),
    ("ppo_epoch_wall_seconds", "REAL"),
    ("payload_spatial_bytes", "INTEGER"),
    ("payload_state_bytes", "INTEGER"),
    ("payload_total_bytes", "INTEGER"),
    ("payload_total_mib", "REAL"),
    ("cuda_peak_allocated_bytes", "INTEGER"),
    ("cuda_peak_reserved_bytes", "INTEGER"),
    ("rollout_cache_spatial_dtype", "TEXT"),
    ("episode_log_enqueue_wall_seconds", "REAL"),
    ("episodes_logged_in_update", "INTEGER"),
    ("checkpoint_wall_seconds", "REAL"),
]


def _safe_add_column(conn, table_name, column_sql):
    column_name = column_sql.split()[0]
    try:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if "duplicate column name" in message and column_name.lower() in message:
            return
        raise


def initialize_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_id INTEGER,
                policy_version INTEGER,
                total_reward REAL,
                average_reward REAL,
                steps INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS steps (
                step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                step_number INTEGER,
                action INTEGER,
                move_x INTEGER,
                move_y INTEGER,
                actor_id INTEGER,
                policy_version INTEGER,
                fragment_id INTEGER,
                policy_protocol_version INTEGER,
                policy_input_schema TEXT,
                reward REAL,
                cumulative_reward REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reward_components (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                step INTEGER,
                health_reward REAL,
                engagement_reward REAL,
                positioning_reward REAL,
                score_reward REAL,
                bonus_reward REAL,
                end_of_episode_reward REAL,
                total_reward REAL
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS ppo_updates (
                update_id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                {", ".join(f"{name} {column_type}" for name, column_type in PPO_UPDATE_COLUMNS)},
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_runs (
                eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_index INTEGER,
                num_episodes INTEGER,
                mean_reward REAL,
                std_reward REAL,
                min_reward REAL,
                max_reward REAL,
                deterministic INTEGER,
                policy_version INTEGER,
                policy_protocol_version INTEGER,
                policy_input_schema TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        _safe_add_column(conn, "episodes", "actor_id INTEGER")
        _safe_add_column(conn, "episodes", "policy_version INTEGER")
        _safe_add_column(conn, "steps", "move_x INTEGER")
        _safe_add_column(conn, "steps", "move_y INTEGER")
        _safe_add_column(conn, "steps", "actor_id INTEGER")
        _safe_add_column(conn, "steps", "policy_version INTEGER")
        _safe_add_column(conn, "steps", "fragment_id INTEGER")
        _safe_add_column(conn, "steps", "policy_protocol_version INTEGER")
        _safe_add_column(conn, "steps", "policy_input_schema TEXT")
        for column_name, column_type in PPO_UPDATE_COLUMNS:
            _safe_add_column(conn, "ppo_updates", f"{column_name} {column_type}")
        _safe_add_column(conn, "eval_runs", "policy_version INTEGER")
        _safe_add_column(conn, "eval_runs", "policy_protocol_version INTEGER")
        _safe_add_column(conn, "eval_runs", "policy_input_schema TEXT")

    return conn


class LogListener(multiprocessing.Process):
    def __init__(self, queue, db_path):
        super().__init__()
        self.queue = queue
        self.db_path = db_path
        self.batch_size = 2000
        self.timeout = 5.0

    def run(self):
        conn = initialize_db(self.db_path)
        cursor = conn.cursor()

        buffer_steps = []
        buffer_rewards = []
        buffer_updates = []
        buffer_evals = []
        last_commit = time.time()

        def flush_buffers():
            nonlocal buffer_steps, buffer_rewards, buffer_updates, buffer_evals, last_commit
            if buffer_steps:
                cursor.executemany(
                    "INSERT INTO steps (episode_id, step_number, action, "
                    "move_x, move_y, actor_id, policy_version, fragment_id, "
                    "policy_protocol_version, policy_input_schema, reward, "
                    "cumulative_reward) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    buffer_steps,
                )
                buffer_steps = []
            if buffer_rewards:
                cursor.executemany(
                    "INSERT INTO reward_components (episode_id, step, "
                    "health_reward, engagement_reward, positioning_reward, "
                    "score_reward, bonus_reward, end_of_episode_reward, "
                    "total_reward) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    buffer_rewards,
                )
                buffer_rewards = []
            if buffer_updates:
                update_columns = [
                    "episode_id",
                    *[name for name, _column_type in PPO_UPDATE_COLUMNS],
                ]
                placeholders = ", ".join("?" for _column in update_columns)
                cursor.executemany(
                    f"INSERT INTO ppo_updates ({', '.join(update_columns)}) "
                    f"VALUES ({placeholders})",
                    buffer_updates,
                )
                buffer_updates = []
            if buffer_evals:
                cursor.executemany(
                    "INSERT INTO eval_runs (episode_index, num_episodes, "
                    "mean_reward, std_reward, min_reward, max_reward, "
                    "deterministic, policy_version, policy_protocol_version, "
                    "policy_input_schema) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    buffer_evals,
                )
                buffer_evals = []
            conn.commit()
            last_commit = time.time()

        episode_map = {}
        running = True
        while running:
            try:
                record = self.queue.get(timeout=1)

                if record["type"] == "EPISODE_START":
                    cursor.execute(
                        "INSERT INTO episodes (actor_id, policy_version, "
                        "total_reward, average_reward, steps) "
                        "VALUES (?, ?, 0, 0, 0)",
                        (
                            record.get("actor_id"),
                            record.get("policy_version"),
                        ),
                    )
                    episode_map[record["internal_ep"]] = cursor.lastrowid

                elif record["type"] == "EPISODE_END":
                    db_id = episode_map.get(record["internal_ep"])
                    if db_id:
                        cursor.execute(
                            "UPDATE episodes SET total_reward=?, average_reward=?, steps=? WHERE episode_id=?",
                            (
                                record["total"],
                                record["avg"],
                                record["steps"],
                                db_id,
                            ),
                        )
                        del episode_map[record["internal_ep"]]

                elif record["type"] == "STEP":
                    db_id = episode_map.get(record["internal_ep"])
                    if db_id:
                        buffer_steps.append(
                            (
                                db_id,
                                record["step"],
                                record["act"],
                                record.get("move_x"),
                                record.get("move_y"),
                                record.get("actor_id"),
                                record.get("policy_version"),
                                record.get("fragment_id"),
                                record.get("policy_protocol_version"),
                                record.get("policy_input_schema"),
                                record["rew"],
                                record["cum_rew"],
                            )
                        )

                elif record["type"] == "UPDATE":
                    db_id = episode_map.get(record["internal_ep"])
                    buffer_updates.append(
                        (
                            db_id,
                            *[
                                record.get(name)
                                for name, _column_type in PPO_UPDATE_COLUMNS
                            ],
                        )
                    )

                elif record["type"] == "REWARD_COMP":
                    db_id = episode_map.get(record["internal_ep"])
                    if db_id:
                        buffer_rewards.append(
                            (
                                db_id,
                                record["step"],
                                record["h_rew"],
                                record["e_rew"],
                                record["p_rew"],
                                record["s_rew"],
                                record["b_rew"],
                                record["end_rew"],
                                record["tot_rew"],
                            )
                        )

                elif record["type"] == "EVAL":
                    buffer_evals.append(
                        (
                            record.get("episode_index"),
                            record.get("num_episodes"),
                            record.get("mean_reward"),
                            record.get("std_reward"),
                            record.get("min_reward"),
                            record.get("max_reward"),
                            int(bool(record.get("deterministic", False))),
                            record.get("policy_version"),
                            record.get("policy_protocol_version"),
                            record.get("policy_input_schema"),
                        )
                    )

                elif record["type"] == "KILL":
                    flush_buffers()
                    running = False

                current_time = time.time()
                if (
                    len(buffer_steps) >= self.batch_size
                    or current_time - last_commit > self.timeout
                ):
                    flush_buffers()

            except Empty:
                continue
            except KeyboardInterrupt:
                running = False
            except Exception as exc:
                print(f"LOGGER ERROR: {exc}")

        flush_buffers()
        conn.close()
