#!/usr/bin/env python3
"""
Training Logger for Federated DRL
===================================
Helper module for logging training metrics, events, and debugging info

Usage:
    from training_logger import TrainingLogger
    
    logger = TrainingLogger('drl_outputs')
    logger.log_round(round=1, client=0, reward=0.5, loss=0.1)
    logger.log_action(round=1, client=0, actions=[1, 2, 0])
    logger.save()
"""

import os
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np


class TrainingLogger:
    """
    Comprehensive logger for federated DRL training
    
    Features:
        - CSV logging for metrics (rewards, losses, episodes)
        - JSON logging for configurations and events
        - Action distribution tracking
        - Checkpoint management
        - Real-time file writing for monitoring
    """
    
    def __init__(self, output_dir: str = 'drl_outputs', experiment_name: Optional[str] = None):
        """
        Initialize training logger
        
        Args:
            output_dir: Directory to save logs
            experiment_name: Name of experiment (default: timestamp)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set experiment name
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = experiment_name
        
        # Create log files
        self.metrics_file = self.output_dir / 'metrics.csv'
        self.actions_file = self.output_dir / 'actions.csv'
        self.events_file = self.output_dir / 'training_log.txt'
        self.config_file = self.output_dir / 'config.json'
        
        # Initialize CSV files with headers
        self._init_metrics_csv()
        self._init_actions_csv()
        
        # In-memory buffers
        self.metrics_buffer = []
        self.actions_buffer = []
        self.events_buffer = []
        
        self._log_event("Logger initialized", level='INFO')
    
    def _init_metrics_csv(self):
        """Initialize metrics CSV file with headers"""
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'round', 'client_id', 'mean_reward', 
                    'std_reward', 'episode_length', 'loss', 'global_reward'
                ])
    
    def _init_actions_csv(self):
        """Initialize actions CSV file with headers"""
        if not self.actions_file.exists():
            with open(self.actions_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'round', 'client_id',
                    'skip_0', 'skip_1', 'skip_2', 'skip_3', 'skip_4',
                    'bitrate_low', 'bitrate_med', 'bitrate_high', 'bitrate_auto',
                    'prefetch_off', 'prefetch_short', 'prefetch_long'
                ])
    
    def _log_event(self, message: str, level: str = 'INFO'):
        """Log event to text file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] [{level}] {message}\n"
        
        # Write to file immediately
        with open(self.events_file, 'a') as f:
            f.write(log_line)
        
        # Also keep in buffer
        self.events_buffer.append(log_line)
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log training configuration
        
        Args:
            config: Dictionary with training configuration
        """
        config['experiment_name'] = self.experiment_name
        config['timestamp'] = datetime.now().isoformat()
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self._log_event(f"Configuration saved: {config}", level='INFO')
    
    def log_round(self, round_num: int, client_id: int, mean_reward: float,
                  std_reward: float = 0.0, episode_length: int = 0,
                  loss: float = 0.0, global_reward: Optional[float] = None):
        """
        Log metrics for a training round
        
        Args:
            round_num: Federated round number
            client_id: Client identifier
            mean_reward: Average reward for this round
            std_reward: Standard deviation of rewards
            episode_length: Average episode length
            loss: Training loss
            global_reward: Global policy reward (if available)
        """
        timestamp = datetime.now().isoformat()
        
        row = [
            timestamp, round_num, client_id, mean_reward,
            std_reward, episode_length, loss,
            global_reward if global_reward is not None else ''
        ]
        
        # Write immediately to CSV
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Log event
        self._log_event(
            f"Round {round_num} | Client {client_id} | Reward: {mean_reward:.4f} | Loss: {loss:.4f}",
            level='METRIC'
        )
        
        # Keep in buffer
        self.metrics_buffer.append(row)
    
    def log_action_distribution(self, round_num: int, client_id: int,
                                action_counts: List[int]):
        """
        Log action distribution for a round
        
        Args:
            round_num: Federated round number
            client_id: Client identifier
            action_counts: List of action counts [skip×5, bitrate×4, prefetch×3]
                          Should have length 12
        """
        timestamp = datetime.now().isoformat()
        
        # Ensure we have 12 values
        if len(action_counts) != 12:
            action_counts = action_counts + [0] * (12 - len(action_counts))
        
        row = [timestamp, round_num, client_id] + action_counts
        
        # Write immediately to CSV
        with open(self.actions_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        self.actions_buffer.append(row)
    
    def log_episode(self, round_num: int, client_id: int, episode_num: int,
                   total_reward: float, length: int):
        """
        Log individual episode results
        
        Args:
            round_num: Federated round number
            client_id: Client identifier
            episode_num: Episode number within round
            total_reward: Total episode reward
            length: Episode length
        """
        self._log_event(
            f"Round {round_num} | Client {client_id} | Episode {episode_num} | "
            f"Reward: {total_reward:.4f} | Length: {length}",
            level='EPISODE'
        )
    
    def log_aggregation(self, round_num: int, num_clients: int,
                       pre_avg_variance: float, post_avg_variance: float):
        """
        Log federated aggregation statistics
        
        Args:
            round_num: Federated round number
            num_clients: Number of clients aggregated
            pre_avg_variance: Variance before aggregation
            post_avg_variance: Variance after aggregation
        """
        self._log_event(
            f"Round {round_num} | FedAvg | Clients: {num_clients} | "
            f"Variance: {pre_avg_variance:.4f} → {post_avg_variance:.4f}",
            level='AGGREGATION'
        )
    
    def log_error(self, message: str, exception: Optional[Exception] = None):
        """
        Log error message
        
        Args:
            message: Error description
            exception: Optional exception object
        """
        error_msg = message
        if exception is not None:
            error_msg += f" | Exception: {str(exception)}"
        
        self._log_event(error_msg, level='ERROR')
    
    def log_checkpoint(self, round_num: int, checkpoint_path: str):
        """
        Log checkpoint save event
        
        Args:
            round_num: Federated round number
            checkpoint_path: Path where checkpoint was saved
        """
        self._log_event(
            f"Round {round_num} | Checkpoint saved: {checkpoint_path}",
            level='CHECKPOINT'
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of logged metrics
        
        Returns:
            Dictionary with summary statistics
        """
        if len(self.metrics_buffer) == 0:
            return {'status': 'No metrics logged yet'}
        
        # Extract rewards
        rewards = [row[3] for row in self.metrics_buffer if isinstance(row[3], (int, float))]
        
        summary = {
            'total_rounds': len(set(row[1] for row in self.metrics_buffer)),
            'total_clients': len(set(row[2] for row in self.metrics_buffer)),
            'total_metrics': len(self.metrics_buffer),
            'avg_reward': np.mean(rewards) if rewards else 0,
            'max_reward': np.max(rewards) if rewards else 0,
            'min_reward': np.min(rewards) if rewards else 0,
            'std_reward': np.std(rewards) if rewards else 0,
        }
        
        return summary
    
    def print_summary(self):
        """Print training summary to console"""
        summary = self.get_metrics_summary()
        
        print("\n" + "="*60)
        print("  TRAINING SUMMARY")
        print("="*60)
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("="*60 + "\n")
    
    def save(self):
        """Force save all buffers (files are already written in real-time)"""
        self._log_event("Training session completed", level='INFO')
        self.print_summary()
        print(f"✓ Logs saved to: {self.output_dir}")


# Convenience functions
def create_logger(output_dir: str = 'drl_outputs', 
                 experiment_name: Optional[str] = None) -> TrainingLogger:
    """
    Create and return a training logger
    
    Args:
        output_dir: Directory for logs
        experiment_name: Name of experiment
    
    Returns:
        TrainingLogger instance
    """
    return TrainingLogger(output_dir, experiment_name)


# Example usage
if __name__ == '__main__':
    # Test the logger
    logger = TrainingLogger('test_outputs', 'test_experiment')
    
    # Log configuration
    logger.log_config({
        'algorithm': 'PPO',
        'n_rounds': 10,
        'n_clients': 5,
        'timesteps_per_round': 5000
    })
    
    # Simulate training
    for round_num in range(1, 4):
        for client_id in range(3):
            logger.log_round(
                round_num=round_num,
                client_id=client_id,
                mean_reward=np.random.uniform(0.3, 0.7),
                std_reward=np.random.uniform(0.05, 0.15),
                episode_length=np.random.randint(50, 150),
                loss=np.random.uniform(0.01, 0.5)
            )
            
            # Log action distribution
            action_counts = np.random.randint(0, 100, size=12).tolist()
            logger.log_action_distribution(round_num, client_id, action_counts)
    
    # Save and print summary
    logger.save()
    
    print("✓ Test completed successfully!")
