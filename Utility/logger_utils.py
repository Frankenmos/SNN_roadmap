import sqlite3
import multiprocessing
import time
from queue import Empty

def initialize_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_reward REAL,
                average_reward REAL,
                steps INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
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
        """)
        conn.execute("""
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
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ppo_updates (
                update_id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                mean_policy_loss REAL,
                mean_value_loss REAL,
                mean_entropy REAL,
                mean_kl REAL,
                clip_fraction REAL,
                explained_variance REAL,
                grad_norm REAL,
                lr REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migration for DBs created before move_x/move_y existed.
        try:
            conn.execute("ALTER TABLE steps ADD COLUMN move_x INTEGER")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE steps ADD COLUMN move_y INTEGER")
        except sqlite3.OperationalError:
            pass
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
        last_commit = time.time()

        def flush_buffers():
            nonlocal buffer_steps, buffer_rewards, buffer_updates, last_commit
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
                    "INSERT INTO ppo_updates (episode_id, mean_policy_loss, "
                    "mean_value_loss, mean_entropy, mean_kl, clip_fraction, "
                    "explained_variance, grad_norm, lr) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    buffer_updates,
                )
                buffer_updates = []
            conn.commit()
            last_commit = time.time()

        episode_map = {}
        running = True
        while running:
            try:
                record = self.queue.get(timeout=1)

                if record['type'] == 'EPISODE_START':
                    cursor.execute(
                        "INSERT INTO episodes (total_reward, average_reward, steps) VALUES (0, 0, 0)"
                    )
                    episode_map[record['internal_ep']] = cursor.lastrowid

                elif record['type'] == 'EPISODE_END':
                    db_id = episode_map.get(record['internal_ep'])
                    if db_id:
                        cursor.execute(
                            "UPDATE episodes SET total_reward=?, average_reward=?, steps=? WHERE episode_id=?",
                            (record['total'], record['avg'], record['steps'], db_id),
                        )
                        del episode_map[record['internal_ep']]

                elif record['type'] == 'STEP':
                    db_id = episode_map.get(record['internal_ep'])
                    if db_id:
                        buffer_steps.append(
                            (
                                db_id,
                                record['step'],
                                record['act'],
                                record.get('move_x'),
                                record.get('move_y'),
                                record['rew'],
                                record['cum_rew'],
                            )
                        )

                elif record['type'] == 'UPDATE':
                    db_id = episode_map.get(record['internal_ep'])
                    # Updates may arrive after the corresponding episode was
                    # removed from episode_map — still log with NULL episode_id.
                    buffer_updates.append(
                        (
                            db_id,
                            record.get('mean_policy_loss'),
                            record.get('mean_value_loss'),
                            record.get('mean_entropy'),
                            record.get('mean_kl'),
                            record.get('clip_fraction'),
                            record.get('explained_variance'),
                            record.get('grad_norm'),
                            record.get('lr'),
                        )
                    )

                elif record['type'] == 'REWARD_COMP':
                    db_id = episode_map.get(record['internal_ep'])
                    if db_id:
                        buffer_rewards.append(
                            (
                                db_id,
                                record['step'],
                                record['h_rew'],
                                record['e_rew'],
                                record['p_rew'],
                                record['s_rew'],
                                record['b_rew'],
                                record['end_rew'],
                                record['tot_rew'],
                            )
                        )

                elif record['type'] == 'KILL':
                    flush_buffers()
                    running = False

                current_time = time.time()
                if len(buffer_steps) >= self.batch_size or (current_time - last_commit > self.timeout):
                    flush_buffers()

            except Empty:
                continue
            except Exception as e:
                print(f"LOGGER ERROR: {e}")

        flush_buffers()
        conn.close()
