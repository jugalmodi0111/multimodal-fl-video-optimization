#!/usr/bin/env python3
"""
Real-Time DRL Training Plotter
================================
Live visualization of federated DRL training metrics

Usage:
    python plot_realtime.py [output_dir]
    
Dependencies:
    - matplotlib
    - pandas
    - numpy

Features:
    - Live reward curves (per client + global)
    - Action distribution evolution
    - Loss curves
    - Episode statistics
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime

# Configuration
REFRESH_INTERVAL = 2000  # milliseconds
WINDOW_SIZE = 100  # Number of data points to show

class DRLTrainingPlotter:
    """Real-time plotter for DRL training metrics"""
    
    def __init__(self, output_dir='drl_outputs'):
        self.output_dir = Path(output_dir)
        self.metrics_file = self.output_dir / 'metrics.csv'
        self.actions_file = self.output_dir / 'actions.csv'
        
        # Check if directory exists
        if not self.output_dir.exists():
            print(f"Error: Directory '{output_dir}' not found")
            print("Run the training cell first to create output directory")
            sys.exit(1)
        
        # Setup plot
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Federated DRL Training - Real-Time Monitor', 
                         fontsize=16, fontweight='bold')
        
        # Create subplot grid
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.3, wspace=0.3)
        
        self.ax1 = self.fig.add_subplot(gs[0, :])  # Reward curves (full width)
        self.ax2 = self.fig.add_subplot(gs[1, 0])  # Episode lengths
        self.ax3 = self.fig.add_subplot(gs[1, 1])  # Loss curves
        self.ax4 = self.fig.add_subplot(gs[2, 0])  # Action distribution
        self.ax5 = self.fig.add_subplot(gs[2, 1])  # Training statistics
        
        # Initialize data storage
        self.reward_data = {}
        self.loss_data = {}
        self.episode_data = {}
        self.action_data = {}
        
        print(f"✓ Monitoring: {self.output_dir}")
        print(f"✓ Refresh interval: {REFRESH_INTERVAL/1000}s")
        print("✓ Press Ctrl+C to stop")
    
    def read_metrics(self):
        """Read latest metrics from CSV file"""
        try:
            if self.metrics_file.exists():
                df = pd.read_csv(self.metrics_file)
                return df
            return None
        except Exception as e:
            print(f"Warning: Error reading metrics: {e}")
            return None
    
    def read_actions(self):
        """Read action distribution data"""
        try:
            if self.actions_file.exists():
                df = pd.read_csv(self.actions_file)
                return df
            return None
        except Exception as e:
            return None
    
    def update_plot(self, frame):
        """Update all plots with latest data"""
        
        # Read latest data
        metrics_df = self.read_metrics()
        actions_df = self.read_actions()
        
        if metrics_df is None or len(metrics_df) == 0:
            # No data yet - show waiting message
            self.ax1.clear()
            self.ax1.text(0.5, 0.5, 'Waiting for training data...', 
                         ha='center', va='center', fontsize=14)
            self.ax1.set_title('Reward Progression')
            return
        
        # --- Plot 1: Reward Curves ---
        self.ax1.clear()
        
        # Group by client
        for client_id in metrics_df['client_id'].unique():
            client_data = metrics_df[metrics_df['client_id'] == client_id]
            self.ax1.plot(client_data['round'], client_data['mean_reward'], 
                         marker='o', label=f'Client {client_id}', alpha=0.7)
        
        # Plot global average if available
        if 'global_reward' in metrics_df.columns:
            global_data = metrics_df.groupby('round')['global_reward'].first()
            self.ax1.plot(global_data.index, global_data.values, 
                         linestyle='--', linewidth=3, color='black', 
                         marker='s', label='Global', alpha=0.9)
        
        self.ax1.set_xlabel('Round', fontweight='bold')
        self.ax1.set_ylabel('Mean Reward', fontweight='bold')
        self.ax1.set_title('Training Reward Progression', fontweight='bold')
        self.ax1.legend(loc='best', fontsize=9)
        self.ax1.grid(True, alpha=0.3)
        
        # --- Plot 2: Episode Lengths ---
        self.ax2.clear()
        
        if 'episode_length' in metrics_df.columns:
            for client_id in metrics_df['client_id'].unique():
                client_data = metrics_df[metrics_df['client_id'] == client_id]
                self.ax2.plot(client_data['round'], client_data['episode_length'], 
                             label=f'Client {client_id}', alpha=0.6)
            
            self.ax2.set_xlabel('Round', fontweight='bold')
            self.ax2.set_ylabel('Episode Length', fontweight='bold')
            self.ax2.set_title('Episode Length Over Time', fontweight='bold')
            self.ax2.legend(loc='best', fontsize=8)
            self.ax2.grid(True, alpha=0.3)
        else:
            self.ax2.text(0.5, 0.5, 'Episode data not available', 
                         ha='center', va='center')
        
        # --- Plot 3: Loss Curves ---
        self.ax3.clear()
        
        if 'loss' in metrics_df.columns:
            for client_id in metrics_df['client_id'].unique():
                client_data = metrics_df[metrics_df['client_id'] == client_id]
                self.ax3.plot(client_data['round'], client_data['loss'], 
                             label=f'Client {client_id}', alpha=0.6)
            
            self.ax3.set_xlabel('Round', fontweight='bold')
            self.ax3.set_ylabel('Loss', fontweight='bold')
            self.ax3.set_title('Training Loss', fontweight='bold')
            self.ax3.legend(loc='best', fontsize=8)
            self.ax3.grid(True, alpha=0.3)
            self.ax3.set_yscale('log')  # Log scale for loss
        else:
            self.ax3.text(0.5, 0.5, 'Loss data not available', 
                         ha='center', va='center')
        
        # --- Plot 4: Action Distribution ---
        self.ax4.clear()
        
        if actions_df is not None and len(actions_df) > 0:
            # Get latest action distribution
            latest_actions = actions_df.iloc[-1]
            action_names = [col for col in actions_df.columns if col not in ['round', 'client_id']]
            action_values = [latest_actions[name] for name in action_names]
            
            bars = self.ax4.barh(action_names, action_values, color='steelblue')
            self.ax4.set_xlabel('Frequency', fontweight='bold')
            self.ax4.set_title('Current Action Distribution', fontweight='bold')
            self.ax4.grid(True, axis='x', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, action_values):
                self.ax4.text(val, bar.get_y() + bar.get_height()/2, 
                             f'{val:.3f}', va='center', ha='left', fontsize=9)
        else:
            self.ax4.text(0.5, 0.5, 'Action data not available', 
                         ha='center', va='center')
        
        # --- Plot 5: Training Statistics ---
        self.ax5.clear()
        self.ax5.axis('off')
        
        # Calculate statistics
        latest_round = metrics_df['round'].max()
        n_clients = metrics_df['client_id'].nunique()
        
        latest_rewards = metrics_df[metrics_df['round'] == latest_round]['mean_reward']
        avg_reward = latest_rewards.mean()
        best_reward = latest_rewards.max()
        worst_reward = latest_rewards.min()
        
        # Create statistics text
        stats_text = f"""
        TRAINING STATISTICS
        ═══════════════════════════════
        
        Current Round: {latest_round}
        Number of Clients: {n_clients}
        
        REWARDS:
        • Average: {avg_reward:.4f}
        • Best Client: {best_reward:.4f}
        • Worst Client: {worst_reward:.4f}
        • Std Dev: {latest_rewards.std():.4f}
        
        PROGRESS:
        • Total Timesteps: {latest_round * 5000 * n_clients:,}
        • Elapsed Time: {datetime.now().strftime('%H:%M:%S')}
        
        STATUS: 🟢 TRAINING IN PROGRESS
        """
        
        self.ax5.text(0.1, 0.95, stats_text, transform=self.ax5.transAxes,
                     fontsize=10, verticalalignment='top', 
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
    
    def start(self):
        """Start the real-time plotting"""
        ani = animation.FuncAnimation(self.fig, self.update_plot, 
                                     interval=REFRESH_INTERVAL,
                                     cache_frame_data=False)
        plt.show()


def main():
    """Main entry point"""
    # Get output directory from command line or use default
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'drl_outputs'
    
    print("="*60)
    print("  FEDERATED DRL TRAINING - REAL-TIME PLOTTER")
    print("="*60)
    
    # Create plotter and start
    try:
        plotter = DRLTrainingPlotter(output_dir)
        plotter.start()
    except KeyboardInterrupt:
        print("\n✓ Plotting stopped by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
