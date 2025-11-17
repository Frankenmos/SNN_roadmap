import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingAnalyzer:
    def __init__(self, db_path: str):
        """Initialize the analyzer with database path."""
        self.db_path = db_path
        self.conn = None
        self._connect()

    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def get_episode_metrics(self) -> pd.DataFrame:
        """Retrieve and process episode-level metrics."""
        query = """
        SELECT 
            episode_id,
            total_reward,
            average_reward,
            steps,
            timestamp
        FROM episodes
        ORDER BY episode_id
        """
        return pd.read_sql_query(query, self.conn)

    def get_step_metrics(self) -> pd.DataFrame:
        """Retrieve step-level metrics."""
        query = """
        SELECT 
            episode_id,
            step_number,
            action,
            reward,
            cumulative_reward
        FROM steps
        ORDER BY episode_id, step_number
        """
        return pd.read_sql_query(query, self.conn)

    def get_reward_components(self) -> pd.DataFrame:
        """Retrieve reward component data."""
        query = """
        SELECT 
            episode,
            step,
            health_reward,
            engagement_reward,
            positioning_reward,
            score_reward,
            bonus_reward,
            end_of_episode_reward,
            total_reward
        FROM reward_components
        ORDER BY episode, step
        """
        return pd.read_sql_query(query, self.conn)

    def calculate_summary_statistics(self) -> dict:
        """Calculate summary statistics for the training run."""
        episodes_df = self.get_episode_metrics()
        steps_df = self.get_step_metrics()
        
        summary = {
            "total_episodes": len(episodes_df),
            "avg_episode_length": episodes_df['steps'].mean(),
            "max_total_reward": episodes_df['total_reward'].max(),
            "avg_total_reward": episodes_df['total_reward'].mean(),
            "reward_std": episodes_df['total_reward'].std(),
            "final_100_avg_reward": episodes_df['total_reward'].tail(100).mean() if len(episodes_df) >= 100 else episodes_df['total_reward'].mean(),
            "completion_rate": len(episodes_df[episodes_df['total_reward'] > 0]) / len(episodes_df) * 100
        }

        # Add reward component statistics if available
        try:
            reward_components_df = self.get_reward_components()
            component_cols = [
                'health_reward', 'engagement_reward', 'positioning_reward',
                'score_reward', 'bonus_reward', 'end_of_episode_reward'
            ]
            component_means = reward_components_df[component_cols].mean()
            summary.update({f"avg_{k}": v for k, v in component_means.items()})
        except:
            logger.warning("Could not calculate reward component statistics")

        return summary

    def plot_training_progress(self, window_size: int = 100, save_path: Optional[str] = None):
        """Plot training progress metrics."""
        episodes_df = self.get_episode_metrics()
        steps_df = self.get_step_metrics()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Plot 1: Total Reward Over Time
        episodes_df['rolling_reward'] = episodes_df['total_reward'].rolling(window=window_size).mean()
        ax1.plot(episodes_df['episode_id'], episodes_df['total_reward'], 'b-', alpha=0.3, label='Raw Reward')
        ax1.plot(episodes_df['episode_id'], episodes_df['rolling_reward'], 'r-', label=f'{window_size}-Episode Moving Average')
        ax1.set_title('Training Progress: Total Reward per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Episode Length
        episodes_df['rolling_steps'] = episodes_df['steps'].rolling(window=window_size).mean()
        ax2.plot(episodes_df['episode_id'], episodes_df['steps'], 'g-', alpha=0.3, label='Episode Length')
        ax2.plot(episodes_df['episode_id'], episodes_df['rolling_steps'], 'r-', label=f'{window_size}-Episode Moving Average')
        ax2.set_title('Episode Length Over Time')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Action Distribution
        action_counts = steps_df['action'].value_counts()
        ax3.bar(range(len(action_counts)), action_counts.values)
        ax3.set_title('Action Distribution')
        ax3.set_xlabel('Action ID')
        ax3.set_ylabel('Count')
        ax3.set_xticks(range(len(action_counts)))
        ax3.set_xticklabels([f'Action {i}' for i in action_counts.index])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training progress plots saved to {save_path}")
        else:
            plt.show()

    def plot_reward_components(self, save_path: Optional[str] = None):
        """Plot reward component analysis."""
        try:
            reward_components_df = self.get_reward_components()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

            # Plot 1: Component Distribution
            component_cols = [
                'health_reward', 'engagement_reward', 'positioning_reward',
                'score_reward', 'bonus_reward', 'end_of_episode_reward'
            ]
            sns.boxplot(data=reward_components_df[component_cols], ax=ax1)
            ax1.set_title('Distribution of Reward Components')
            ax1.set_xlabel('Component Type')
            ax1.set_ylabel('Reward Value')
            plt.xticks(rotation=45)

            # Plot 2: Component Evolution
            for col in component_cols:
                rolling_mean = reward_components_df.groupby('episode')[col].mean().rolling(window=50).mean()
                ax2.plot(rolling_mean.index, rolling_mean.values, label=col)
            
            ax2.set_title('Reward Components Evolution Over Episodes')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Average Reward')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                logger.info(f"Reward component plots saved to {save_path}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"Could not plot reward components: {e}")

    def export_metrics(self, output_path: str):
        """Export summary metrics to a file."""
        summary = self.calculate_summary_statistics()
        summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
        summary_df.to_csv(output_path, index=False)
        logger.info(f"Summary metrics exported to {output_path}")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    """Main execution function."""
    db_path = "training_logs.db"
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)

    try:
        analyzer = TrainingAnalyzer(db_path)

        # Generate and save plots
        analyzer.plot_training_progress(
            save_path=str(output_dir / "training_progress.png")
        )
        
        analyzer.plot_reward_components(
            save_path=str(output_dir / "reward_components.png")
        )

        # Export metrics
        analyzer.export_metrics(str(output_dir / "training_metrics.csv"))

        # Print summary statistics
        summary = analyzer.calculate_summary_statistics()
        print("\nTraining Summary:")
        for metric, value in summary.items():
            print(f"{metric}: {value:.4f}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
    finally:
        if 'analyzer' in locals():
            analyzer.close()

if __name__ == "__main__":
    main()
