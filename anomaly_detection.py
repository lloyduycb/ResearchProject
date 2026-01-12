"""
Anomaly Detection & Health Index Module
========================================
This module provides Health Index calculation for turbofan engines
using reconstruction-based anomaly detection.

The key insight: Train on "healthy" data (early life), then measure
how much current data deviates from healthy patterns.

Author: Research Project CYS6001-20
Date: January 12, 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class HealthIndexCalculator:
    """
    Calculate Health Index (0-100) for each time step of an engine.
    
    Approach:
    1. Train an Isolation Forest on "healthy" data (first X% of each engine's life)
    2. Use anomaly scores as inverse health measure
    3. Normalize to 0-100 scale (100 = perfectly healthy)
    """
    
    def __init__(self, contamination: float = 0.1, healthy_fraction: float = 0.2):
        """
        Args:
            contamination: Expected proportion of outliers in training data
            healthy_fraction: Fraction of early life to consider "healthy" (default 20%)
        """
        self.contamination = contamination
        self.healthy_fraction = healthy_fraction
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.sensor_cols = None
        self.fitted = False
        
    def _get_sensor_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract sensor columns from dataframe."""
        return [col for col in df.columns if col.startswith('sensor_')]
    
    def _extract_healthy_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the first healthy_fraction of each engine's life cycle.
        These early cycles represent "healthy" operation.
        """
        healthy_rows = []
        
        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id].copy()
            max_cycles = unit_data['time_cycles'].max()
            healthy_threshold = int(max_cycles * self.healthy_fraction)
            
            healthy_portion = unit_data[unit_data['time_cycles'] <= healthy_threshold]
            healthy_rows.append(healthy_portion)
        
        return pd.concat(healthy_rows, ignore_index=True)
    
    def fit(self, train_df: pd.DataFrame) -> 'HealthIndexCalculator':
        """
        Fit the anomaly detector on healthy data from training set.
        
        Args:
            train_df: Training dataframe with unit_id, time_cycles, and sensor columns
        """
        self.sensor_cols = self._get_sensor_columns(train_df)
        
        if not self.sensor_cols:
            raise ValueError("No sensor columns found. Expected columns like 'sensor_1', 'sensor_2', etc.")
        
        print(f"[HealthIndex] Extracting healthy data (first {self.healthy_fraction*100:.0f}% of each engine)...")
        healthy_df = self._extract_healthy_data(train_df)
        
        print(f"[HealthIndex] Training on {len(healthy_df)} healthy samples...")
        X_healthy = healthy_df[self.sensor_cols].values
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X_healthy)
        
        # Fit Isolation Forest
        self.model.fit(X_scaled)
        self.fitted = True
        
        print(f"[HealthIndex] Model trained successfully on {len(self.sensor_cols)} sensors.")
        return self
    
    def compute_health_index(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute Health Index for each row in the dataframe.
        
        Returns:
            Array of health indices (0-100), where 100 is healthiest
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = df[self.sensor_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores (negative = more anomalous)
        raw_scores = self.model.decision_function(X_scaled)
        
        # Convert to 0-100 scale (higher = healthier)
        # decision_function returns negative for outliers, positive for inliers
        # We normalize to [0, 100] range
        min_score, max_score = raw_scores.min(), raw_scores.max()
        
        if max_score - min_score > 0:
            health_index = (raw_scores - min_score) / (max_score - min_score) * 100
        else:
            health_index = np.full_like(raw_scores, 50.0)
        
        return health_index
    
    def compute_engine_health_trajectory(self, df: pd.DataFrame, 
                                         unit_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get health trajectory for a specific engine.
        
        Returns:
            (time_cycles, health_indices) for the specified engine
        """
        engine_data = df[df['unit_id'] == unit_id].copy()
        engine_data = engine_data.sort_values('time_cycles')
        
        time_cycles = engine_data['time_cycles'].values
        health_index = self.compute_health_index(engine_data)
        
        return time_cycles, health_index


class AnomalyVisualizer:
    """
    Publication-quality visualizations for Health Index analysis.
    """
    
    @staticmethod
    def plot_health_degradation(time_cycles: np.ndarray, health_index: np.ndarray,
                                unit_id: int, save_path: Optional[str] = None):
        """
        Plot health index degradation over time for a single engine.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create color gradient based on health
        colors = plt.cm.RdYlGn(health_index / 100)
        
        ax.scatter(time_cycles, health_index, c=colors, s=30, alpha=0.7, edgecolors='k', linewidth=0.3)
        ax.plot(time_cycles, health_index, 'b-', alpha=0.3, linewidth=1)
        
        # Add critical threshold line
        ax.axhline(y=50, color='orange', linestyle='--', linewidth=2, label='Warning Threshold')
        ax.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Critical Threshold')
        
        ax.set_xlabel('Time Cycles', fontweight='bold')
        ax.set_ylabel('Health Index (0-100)', fontweight='bold')
        ax.set_title(f'Engine {unit_id}: Health Degradation Over Time', fontweight='bold', pad=15)
        ax.set_ylim(-5, 105)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 100))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Health Level', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[HealthIndex] Saved plot to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_fleet_health_summary(df: pd.DataFrame, health_calculator: HealthIndexCalculator,
                                  save_path: Optional[str] = None):
        """
        Plot health summary across all engines in the fleet.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Fleet Health Analysis', fontweight='bold', fontsize=14)
        
        # Compute health for all data
        df = df.copy()
        df['health_index'] = health_calculator.compute_health_index(df)
        
        # 1. Health distribution histogram
        axes[0, 0].hist(df['health_index'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].axvline(df['health_index'].mean(), color='red', linestyle='--', 
                           linewidth=2, label=f"Mean: {df['health_index'].mean():.1f}")
        axes[0, 0].set_xlabel('Health Index')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Health Index Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Health by lifecycle stage
        df['lifecycle_stage'] = pd.cut(df['time_cycles'], 
                                       bins=[0, 50, 100, 150, float('inf')],
                                       labels=['0-50', '51-100', '101-150', '150+'])
        stage_health = df.groupby('lifecycle_stage')['health_index'].mean()
        
        bars = axes[0, 1].bar(stage_health.index.astype(str), stage_health.values, 
                              edgecolor='black', alpha=0.7)
        for bar, val in zip(bars, stage_health.values):
            bar.set_color('green' if val > 70 else 'orange' if val > 40 else 'red')
        axes[0, 1].set_xlabel('Lifecycle Stage (cycles)')
        axes[0, 1].set_ylabel('Mean Health Index')
        axes[0, 1].set_title('Health by Lifecycle Stage')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Sample engine trajectories
        sample_engines = df['unit_id'].unique()[:5]
        for engine_id in sample_engines:
            engine_data = df[df['unit_id'] == engine_id].sort_values('time_cycles')
            axes[1, 0].plot(engine_data['time_cycles'], engine_data['health_index'], 
                           alpha=0.7, label=f'Engine {engine_id}')
        
        axes[1, 0].set_xlabel('Time Cycles')
        axes[1, 0].set_ylabel('Health Index')
        axes[1, 0].set_title('Sample Engine Trajectories')
        axes[1, 0].legend(loc='lower left', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Engines in critical state
        critical_engines = df[df['health_index'] < 25]['unit_id'].nunique()
        warning_engines = df[(df['health_index'] >= 25) & (df['health_index'] < 50)]['unit_id'].nunique()
        healthy_engines = df[df['health_index'] >= 50]['unit_id'].nunique()
        
        categories = ['Critical\n(<25)', 'Warning\n(25-50)', 'Healthy\n(>50)']
        counts = [critical_engines, warning_engines, healthy_engines]
        colors = ['red', 'orange', 'green']
        
        axes[1, 1].bar(categories, counts, color=colors, edgecolor='black', alpha=0.7)
        axes[1, 1].set_ylabel('Number of Engines')
        axes[1, 1].set_title('Fleet Health Status')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for i, (cat, count) in enumerate(zip(categories, counts)):
            axes[1, 1].text(i, count + 0.5, str(count), ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[HealthIndex] Saved fleet summary to {save_path}")
        
        return fig


def demo_anomaly_detection():
    """
    Demonstration of the anomaly detection module using C-MAPSS data.
    """
    import os
    from run_evaluation import CMAPSSDataLoader
    
    print("\n" + "="*70)
    print("ANOMALY DETECTION & HEALTH INDEX DEMO")
    print("="*70)
    
    # Load data
    loader = CMAPSSDataLoader('./data')
    try:
        train_df, test_df, true_rul = loader.load_fd_dataset(1)
        train_df = loader.add_rul_column(train_df)
    except FileNotFoundError:
        print("[ERROR] C-MAPSS data not found. Place data in ./data/")
        return
    
    # Create and fit health calculator
    print("\n[1/3] Training Health Index model...")
    calculator = HealthIndexCalculator(healthy_fraction=0.2)
    calculator.fit(train_df)
    
    # Compute health for training data
    print("\n[2/3] Computing Health Index for all engines...")
    train_df['health_index'] = calculator.compute_health_index(train_df)
    
    # Visualize
    print("\n[3/3] Generating visualizations...")
    os.makedirs('./results', exist_ok=True)
    
    viz = AnomalyVisualizer()
    
    # Single engine
    sample_engine = train_df['unit_id'].unique()[0]
    time_cycles, health = calculator.compute_engine_health_trajectory(train_df, sample_engine)
    viz.plot_health_degradation(time_cycles, health, sample_engine, 
                                f'./results/health_degradation_engine_{sample_engine}.png')
    
    # Fleet summary
    viz.plot_fleet_health_summary(train_df, calculator, './results/fleet_health_summary.png')
    
    print("\n" + "="*70)
    print("[OK] Demo complete! Check ./results/ for visualizations.")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Anomaly Detection & Health Index Calculator")
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--fd', type=int, default=1, choices=[1, 2, 3, 4],
                        help='FD dataset number (default: 1)')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_anomaly_detection()
    else:
        print("Use --demo to run the demonstration.")
        print("Example: python anomaly_detection.py --demo")
