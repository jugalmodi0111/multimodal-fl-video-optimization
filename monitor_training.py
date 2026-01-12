#!/usr/bin/env python3
"""
Real-Time Federated DRL Training Monitor
=========================================
Professional monitoring dashboard for tracking training metrics, accuracy,
and performance across multiple RL agents in federated learning setup.

Usage:
    python monitor_training.py [results_directory]
    
Features:
    - Live metrics tracking (accuracy, reward, loss)
    - Per-agent performance comparison
    - Training progress visualization
    - Resource utilization monitoring
    - Automated report generation
"""

import os
import sys
import time
import csv
from pathlib import Path
from datetime import datetime

# Configuration
DEFAULT_RESULTS_DIR = 'results'
REFRESH_INTERVAL = 2  # seconds
MAX_DISPLAY_ROUNDS = 20

class TrainingMonitor:
    """
    Real-time monitoring system for federated DRL training.
    """
    
    def __init__(self, results_dir=DEFAULT_RESULTS_DIR):
        """
        Initialize training monitor.
        
        Args:
            results_dir: Path to results directory containing CSV files
        """
        self.results_dir = Path(results_dir)
        
        if not self.results_dir.exists():
            print(f"Warning: Results directory '{results_dir}' not found")
            print(f"Creating directory...")
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected CSV files
        self.metric_files = {
            'PPO': self.results_dir / 'PPO_metrics.csv',
            'SAC': self.results_dir / 'SAC_metrics.csv',
            'TD3': self.results_dir / 'TD3_metrics.csv',
            'Random': self.results_dir / 'Random_metrics.csv'
        }
        
        self.summary_file = self.results_dir / 'summary.csv'
        
        self.start_time = time.time()
        self.iteration = 0
    
    def check_files_exist(self):
        """Check which metric files are available."""
        available = {}
        for agent, filepath in self.metric_files.items():
            available[agent] = filepath.exists()
        return available
    
    def load_metrics(self, agent_name):
        """
        Load metrics from CSV file for specific agent.
        
        Args:
            agent_name: Name of the agent (PPO, SAC, TD3, Random)
            
        Returns:
            List of dictionaries with metrics or None if file not found
        """
        filepath = self.metric_files.get(agent_name)
        
        if filepath and filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                    if len(data) > 0:
                        return data
            except Exception as e:
                print(f"Error loading {agent_name}: {e}")
        
        return None
    
    def load_summary(self):
        """Load summary statistics."""
        if self.summary_file.exists():
            try:
                with open(self.summary_file, 'r') as f:
                    reader = csv.DictReader(f)
                    return list(reader)
            except:
                pass
        return None
    
    def get_latest_metrics(self, data):
        """Extract latest metrics from data list."""
        if data is None or len(data) == 0:
            return {'round': 0, 'accuracy': 0.0, 'reward': 0.0}
        
        latest = data[-1]
        return {
            'round': int(float(latest.get('round', 0))),
            'accuracy': float(latest.get('accuracy', 0.0)),
            'reward': float(latest.get('reward', 0.0))
        }
    
    def get_statistics(self, data, column='accuracy'):
        """Calculate statistics for a column."""
        if data is None or len(data) == 0:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}
        
        try:
            values = [float(row[column]) for row in data if column in row and row[column]]
            if len(values) == 0:
                return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}
            
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            std_val = variance ** 0.5
            
            return {
                'mean': mean_val,
                'std': std_val,
                'max': max(values),
                'min': min(values)
            }
        except:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}
    
    def display_header(self):
        """Display monitoring header."""
        elapsed = time.time() - self.start_time
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        
        print("\033[H\033[J")  # Clear screen
        print("=" * 90)
        print(" " * 20 + "FEDERATED DRL TRAINING MONITOR")
        print("=" * 90)
        print(f"Results Directory: {self.results_dir}")
        print(f"Monitor Runtime: {elapsed_min}m {elapsed_sec}s")
        print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Refresh Rate: {REFRESH_INTERVAL}s")
        print("=" * 90)
    
    def display_training_progress(self):
        """Display training progress for all agents."""
        print("\nTRAINING PROGRESS")
        print("-" * 90)
        
        available = self.check_files_exist()
        
        if not any(available.values()):
            print("No training data found. Waiting for training to start...")
            return
        
        # Table header
        print(f"{'Agent':<10} {'Round':<8} {'Latest Acc':<12} {'Mean Acc':<12} {'Best Acc':<12} {'Latest Reward':<15}")
        print("-" * 90)
        
        all_metrics = {}
        
        for agent in ['PPO', 'SAC', 'TD3', 'Random']:
            if not available.get(agent, False):
                print(f"{agent:<10} {'N/A':<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}")
                continue
            
            df = self.load_metrics(agent)
            if df is None or len(df) == 0:
                print(f"{agent:<10} {'N/A':<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}")
                continue
            
            latest = self.get_latest_metrics(df)
            stats = self.get_statistics(df, 'accuracy')
            
            all_metrics[agent] = {'latest': latest, 'stats': stats}
            
            print(f"{agent:<10} "
                  f"{latest['round']:<8} "
                  f"{latest['accuracy']:<12.4f} "
                  f"{stats['mean']:<12.4f} "
                  f"{stats['max']:<12.4f} "
                  f"{latest['reward']:<15.4f}")
        
        # Find best performer
        if all_metrics:
            best_agent = max(all_metrics.keys(), 
                           key=lambda x: all_metrics[x]['latest']['accuracy'])
            print("\n" + "-" * 90)
            print(f"Best Current Performance: {best_agent} "
                  f"(Accuracy: {all_metrics[best_agent]['latest']['accuracy']:.4f})")
    
    def display_summary(self):
        """Display summary statistics if available."""
        summary_data = self.load_summary()
        
        if summary_data is not None and len(summary_data) > 0:
            print("\n" + "=" * 90)
            print("FINAL SUMMARY STATISTICS")
            print("-" * 90)
            
            for row in summary_data:
                agent = row.get('agent', 'Unknown')
                final_acc = float(row.get('final_accuracy', 0.0))
                mean_acc = float(row.get('mean_accuracy', 0.0))
                std_acc = float(row.get('std_accuracy', 0.0))
                time_taken = float(row.get('training_time_seconds', 0.0))
                
                print(f"{agent:<10} "
                      f"Final: {final_acc:.4f} | "
                      f"Mean: {mean_acc:.4f} | "
                      f"Std: {std_acc:.4f} | "
                      f"Time: {time_taken:.1f}s")
    
    def display_recent_history(self):
        """Display recent training history for all agents."""
        print("\n" + "=" * 90)
        print("RECENT ACCURACY HISTORY (Last 5 Rounds)")
        print("-" * 90)
        
        for agent in ['PPO', 'SAC', 'TD3', 'Random']:
            data = self.load_metrics(agent)
            
            if data is None or len(data) == 0:
                print(f"{agent:<10} No data available")
                continue
            
            # Get last 5 rounds
            recent = data[-5:] if len(data) >= 5 else data
            
            try:
                acc_str = ', '.join([
                    f"R{int(float(row['round']))}:{float(row['accuracy']):.3f}" 
                    for row in recent if 'round' in row and 'accuracy' in row
                ])
                print(f"{agent:<10} {acc_str}")
            except:
                print(f"{agent:<10} Data format error")
    
    def run(self):
        """Main monitoring loop."""
        print(f"Starting training monitor...")
        print(f"Monitoring: {self.results_dir}")
        print(f"Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.iteration += 1
                
                self.display_header()
                self.display_training_progress()
                self.display_recent_history()
                self.display_summary()
                
                print("\n" + "=" * 90)
                print(f"Refresh #{self.iteration} | Press Ctrl+C to exit")
                print("=" * 90)
                
                time.sleep(REFRESH_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            print(f"Total runtime: {int(time.time() - self.start_time)}s")
            print("Monitor terminated successfully")


def main():
    """Main entry point."""
    # Get results directory from command line or use default
    results_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    
    print("=" * 90)
    print(" " * 25 + "TRAINING MONITOR v2.0")
    print("=" * 90)
    print()
    
    # Create and run monitor
    monitor = TrainingMonitor(results_dir)
    monitor.run()


if __name__ == '__main__':
    main()
