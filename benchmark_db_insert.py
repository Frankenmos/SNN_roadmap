import sqlite3
import time
import os

def setup_db(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
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
                cumulative_reward REAL,
                FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reward_components (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode INTEGER,
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

def run_n_plus_one(conn, num_episodes=10, steps_per_episode=600):
    start_time = time.time()
    for episode in range(num_episodes):
        with conn:
            cursor = conn.execute(
                "INSERT INTO episodes (total_reward, average_reward, steps) VALUES (0, 0, 0)"
            )
            episode_id = cursor.lastrowid

        cumulative_reward = 0.0
        for step in range(steps_per_episode):
            reward = 1.0
            cumulative_reward += reward
            with conn:
                conn.execute(
                    "INSERT INTO steps (episode_id, step_number, action, reward, cumulative_reward) VALUES (?, ?, ?, ?, ?)",
                    (int(episode_id), int(step + 1), 0, float(reward), float(cumulative_reward))
                )
                conn.execute(
                    """INSERT INTO reward_components
                       (episode, step, health_reward, engagement_reward, positioning_reward,
                        score_reward, bonus_reward, end_of_episode_reward, total_reward)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (int(episode), int(step + 1), 0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 1.0)
                )
        with conn:
            conn.execute(
                "UPDATE episodes SET total_reward=?, average_reward=?, steps=? WHERE episode_id=?",
                (float(cumulative_reward), float(cumulative_reward), int(steps_per_episode), int(episode_id))
            )
    return time.time() - start_time

def run_batched(conn, num_episodes=10, steps_per_episode=600, batch_size=100):
    start_time = time.time()
    for episode in range(num_episodes):
        with conn:
            cursor = conn.execute(
                "INSERT INTO episodes (total_reward, average_reward, steps) VALUES (0, 0, 0)"
            )
            episode_id = cursor.lastrowid

        cumulative_reward = 0.0
        step_buffer = []
        reward_comp_buffer = []

        for step in range(steps_per_episode):
            reward = 1.0
            cumulative_reward += reward

            step_buffer.append(
                (int(episode_id), int(step + 1), 0, float(reward), float(cumulative_reward))
            )
            reward_comp_buffer.append(
                (int(episode), int(step + 1), 0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 1.0)
            )

            if len(step_buffer) >= batch_size:
                with conn:
                    conn.executemany(
                        "INSERT INTO steps (episode_id, step_number, action, reward, cumulative_reward) VALUES (?, ?, ?, ?, ?)",
                        step_buffer
                    )
                    conn.executemany(
                        """INSERT INTO reward_components
                           (episode, step, health_reward, engagement_reward, positioning_reward,
                            score_reward, bonus_reward, end_of_episode_reward, total_reward)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        reward_comp_buffer
                    )
                step_buffer.clear()
                reward_comp_buffer.clear()

        if step_buffer:
            with conn:
                conn.executemany(
                    "INSERT INTO steps (episode_id, step_number, action, reward, cumulative_reward) VALUES (?, ?, ?, ?, ?)",
                    step_buffer
                )
                conn.executemany(
                    """INSERT INTO reward_components
                       (episode, step, health_reward, engagement_reward, positioning_reward,
                        score_reward, bonus_reward, end_of_episode_reward, total_reward)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    reward_comp_buffer
                )
            step_buffer.clear()
            reward_comp_buffer.clear()

        with conn:
            conn.execute(
                "UPDATE episodes SET total_reward=?, average_reward=?, steps=? WHERE episode_id=?",
                (float(cumulative_reward), float(cumulative_reward), int(steps_per_episode), int(episode_id))
            )
    return time.time() - start_time

if __name__ == "__main__":
    db_path = "benchmark.db"

    print("Setting up DB for N+1 benchmark...")
    conn = setup_db(db_path)
    time_n_plus_one = run_n_plus_one(conn, num_episodes=10, steps_per_episode=600)
    conn.close()

    print("Setting up DB for Batched benchmark...")
    conn = setup_db(db_path)
    time_batched = run_batched(conn, num_episodes=10, steps_per_episode=600, batch_size=100)
    conn.close()

    print(f"N+1 Insert Time: {time_n_plus_one:.4f} seconds")
    print(f"Batched Insert Time: {time_batched:.4f} seconds")
    print(f"Improvement: {time_n_plus_one / time_batched:.2f}x faster")

    if os.path.exists(db_path):
        os.remove(db_path)
