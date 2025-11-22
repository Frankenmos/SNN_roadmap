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
                                record['rew'],
                                record['cum_rew'],
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
                    running = False

                current_time = time.time()
                if len(buffer_steps) >= self.batch_size or (current_time - last_commit > self.timeout):
                    if buffer_steps:
                        cursor.executemany(
                            "INSERT INTO steps (episode_id, step_number, action, reward, cumulative_reward) VALUES (?, ?, ?, ?, ?)",
                            buffer_steps,
                        )
                        buffer_steps = []
                    if buffer_rewards:
                        cursor.executemany(
                            "INSERT INTO reward_components (episode_id, step, health_reward, engagement_reward, positioning_reward, score_reward, bonus_reward, end_of_episode_reward, total_reward) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            buffer_rewards,
                        )
                        buffer_rewards = []
                    conn.commit()
                    last_commit = current_time

            except Empty:
                continue
            except Exception as e:
                print(f"LOGGER ERROR: {e}")

        conn.close()
