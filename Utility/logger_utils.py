import multiprocessing
import sqlite3
import time
from queue import Empty


def _safe_add_column(conn, table_name, column_sql):
    column_name = column_sql.split()[0]
    try:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")
    except sqlite3.OperationalError as exc:
        if column_name not in str(exc):
            pass


def initialize_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            """
            CREATE TABLE IF NOT EXISTS ppo_updates (
                update_id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                episode_index INTEGER,
                mean_policy_loss REAL,
                mean_value_loss REAL,
                mean_entropy REAL,
                mean_kl REAL,
                clip_fraction REAL,
                explained_variance REAL,
                grad_norm REAL,
                lr REAL,
                nonfinite_grad_steps INTEGER,
                skipped_optimizer_steps INTEGER,
                transitions_in_update INTEGER,
                return_mean REAL,
                return_std REAL,
                return_p10 REAL,
                return_p50 REAL,
                return_p90 REAL,
                entity_mask_utilization REAL,
                entity_count_p50 REAL,
                entity_count_p99 REAL,
                selection_mask_utilization REAL,
                update_wall_seconds REAL,
                tbptt_chunks INTEGER,
                tbptt_chunk_groups INTEGER,
                tbptt_window INTEGER,
                tbptt_group_max_steps INTEGER,
                tbptt_group_mean_active_chunks REAL,
                tbptt_forward_calls INTEGER,
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
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        _safe_add_column(conn, "steps", "move_x INTEGER")
        _safe_add_column(conn, "steps", "move_y INTEGER")
        _safe_add_column(conn, "ppo_updates", "episode_index INTEGER")
        _safe_add_column(conn, "ppo_updates", "nonfinite_grad_steps INTEGER")
        _safe_add_column(conn, "ppo_updates", "skipped_optimizer_steps INTEGER")
        _safe_add_column(conn, "ppo_updates", "transitions_in_update INTEGER")
        _safe_add_column(conn, "ppo_updates", "return_mean REAL")
        _safe_add_column(conn, "ppo_updates", "return_std REAL")
        _safe_add_column(conn, "ppo_updates", "return_p10 REAL")
        _safe_add_column(conn, "ppo_updates", "return_p50 REAL")
        _safe_add_column(conn, "ppo_updates", "return_p90 REAL")
        _safe_add_column(conn, "ppo_updates", "entity_mask_utilization REAL")
        _safe_add_column(conn, "ppo_updates", "entity_count_p50 REAL")
        _safe_add_column(conn, "ppo_updates", "entity_count_p99 REAL")
        _safe_add_column(conn, "ppo_updates", "selection_mask_utilization REAL")
        _safe_add_column(conn, "ppo_updates", "update_wall_seconds REAL")
        _safe_add_column(conn, "ppo_updates", "tbptt_chunks INTEGER")
        _safe_add_column(conn, "ppo_updates", "tbptt_chunk_groups INTEGER")
        _safe_add_column(conn, "ppo_updates", "tbptt_window INTEGER")
        _safe_add_column(conn, "ppo_updates", "tbptt_group_max_steps INTEGER")
        _safe_add_column(conn, "ppo_updates", "tbptt_group_mean_active_chunks REAL")
        _safe_add_column(conn, "ppo_updates", "tbptt_forward_calls INTEGER")

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
                    "move_x, move_y, reward, cumulative_reward) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
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
                cursor.executemany(
                    "INSERT INTO ppo_updates (episode_id, episode_index, "
                    "mean_policy_loss, mean_value_loss, mean_entropy, mean_kl, "
                    "clip_fraction, explained_variance, grad_norm, lr, "
                    "nonfinite_grad_steps, skipped_optimizer_steps, "
                    "transitions_in_update, return_mean, return_std, "
                    "return_p10, return_p50, return_p90, "
                    "entity_mask_utilization, entity_count_p50, entity_count_p99, "
                    "selection_mask_utilization, update_wall_seconds, "
                    "tbptt_chunks, tbptt_chunk_groups, tbptt_window, "
                    "tbptt_group_max_steps, tbptt_group_mean_active_chunks, "
                    "tbptt_forward_calls) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    buffer_updates,
                )
                buffer_updates = []
            if buffer_evals:
                cursor.executemany(
                    "INSERT INTO eval_runs (episode_index, num_episodes, "
                    "mean_reward, std_reward, min_reward, max_reward, deterministic) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
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
                        "INSERT INTO episodes (total_reward, average_reward, steps) VALUES (0, 0, 0)"
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
                                record["rew"],
                                record["cum_rew"],
                            )
                        )

                elif record["type"] == "UPDATE":
                    db_id = episode_map.get(record["internal_ep"])
                    buffer_updates.append(
                        (
                            db_id,
                            record.get("episode_index"),
                            record.get("mean_policy_loss"),
                            record.get("mean_value_loss"),
                            record.get("mean_entropy"),
                            record.get("mean_kl"),
                            record.get("clip_fraction"),
                            record.get("explained_variance"),
                            record.get("grad_norm"),
                            record.get("lr"),
                            record.get("nonfinite_grad_steps"),
                            record.get("skipped_optimizer_steps"),
                            record.get("transitions_in_update"),
                            record.get("return_mean"),
                            record.get("return_std"),
                            record.get("return_p10"),
                            record.get("return_p50"),
                            record.get("return_p90"),
                            record.get("entity_mask_utilization"),
                            record.get("entity_count_p50"),
                            record.get("entity_count_p99"),
                            record.get("selection_mask_utilization"),
                            record.get("update_wall_seconds"),
                            record.get("tbptt_chunks"),
                            record.get("tbptt_chunk_groups"),
                            record.get("tbptt_window"),
                            record.get("tbptt_group_max_steps"),
                            record.get("tbptt_group_mean_active_chunks"),
                            record.get("tbptt_forward_calls"),
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
            except Exception as exc:
                print(f"LOGGER ERROR: {exc}")

        flush_buffers()
        conn.close()
