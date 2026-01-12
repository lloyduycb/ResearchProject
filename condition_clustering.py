"""
Operating Condition Clustering Module
=====================================
Cluster engine operating conditions to enable condition-aware normalization.

Critical for FD002 and FD004 datasets which have 6 different operating regimes.

Author: Research Project CYS6001-20
Date: January 12, 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class OperatingConditionClusterer:
    """
    Cluster engine data by operating conditions (altitude, mach, throttle).
    
    This is essential for FD002/FD004 which have 6 operating conditions.
    Normalization should be done PER CONDITION for best results.
    """
    
    def __init__(self, n_clusters: int = 6, random_state: int = 42):
        """
        Args:
            n_clusters: Number of operating conditions to detect (default 6 for FD002/FD004)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()
        self.setting_cols = ['setting_1', 'setting_2', 'setting_3']
        self.fitted = False
        self.cluster_centers_ = None
        
    def fit(self, df: pd.DataFrame) -> 'OperatingConditionClusterer':
        """
        Fit the clusterer on the settings columns.
        
        Args:
            df: Dataframe with setting_1, setting_2, setting_3 columns
        """
        # Verify columns exist
        missing = [col for col in self.setting_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing setting columns: {missing}")
        
        X = df[self.setting_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"[Clustering] Fitting K-Means with {self.n_clusters} clusters...")
        self.kmeans.fit(X_scaled)
        self.cluster_centers_ = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        self.fitted = True
        
        # Report cluster sizes
        labels = self.kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        print(f"[Clustering] Cluster distribution:")
        for cluster, count in zip(unique, counts):
            print(f"  Cluster {cluster}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Returns:
            Array of cluster labels (0 to n_clusters-1)
        """
        if not self.fitted:
            raise RuntimeError("Clusterer not fitted. Call fit() first.")
        
        X = df[self.setting_cols].values
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
    
    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and predict in one step.
        """
        self.fit(df)
        return self.predict(df)
    
    def add_cluster_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'operating_condition' column to the dataframe.
        """
        df = df.copy()
        df['operating_condition'] = self.predict(df)
        return df
    
    def normalize_by_condition(self, df: pd.DataFrame, 
                               sensor_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize sensor values WITHIN each operating condition.
        
        This removes the effect of different operating regimes and 
        makes the data more suitable for degradation modeling.
        
        Args:
            df: Dataframe with sensor columns
            sensor_cols: List of sensor columns to normalize (auto-detected if None)
            
        Returns:
            Dataframe with normalized sensor values
        """
        if sensor_cols is None:
            sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        
        df = df.copy()
        
        # Add cluster labels if not present
        if 'operating_condition' not in df.columns:
            df['operating_condition'] = self.predict(df)
        
        # Normalize within each cluster
        print(f"[Clustering] Normalizing {len(sensor_cols)} sensors by operating condition...")
        
        for condition in range(self.n_clusters):
            mask = df['operating_condition'] == condition
            if mask.sum() > 0:
                condition_data = df.loc[mask, sensor_cols]
                
                # Z-score normalization within condition
                mean_vals = condition_data.mean()
                std_vals = condition_data.std().replace(0, 1)  # Avoid division by zero
                
                df.loc[mask, sensor_cols] = (condition_data - mean_vals) / std_vals
        
        print(f"[Clustering] Normalization complete.")
        return df
    
    def get_cluster_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics for each cluster/operating condition.
        """
        df = df.copy()
        if 'operating_condition' not in df.columns:
            df['operating_condition'] = self.predict(df)
        
        stats = []
        for condition in range(self.n_clusters):
            mask = df['operating_condition'] == condition
            condition_data = df[mask]
            
            if len(condition_data) > 0:
                stats.append({
                    'Condition': condition,
                    'Count': len(condition_data),
                    'Pct': len(condition_data) / len(df) * 100,
                    'Setting_1_Mean': condition_data['setting_1'].mean(),
                    'Setting_2_Mean': condition_data['setting_2'].mean(),
                    'Setting_3_Mean': condition_data['setting_3'].mean(),
                })
        
        return pd.DataFrame(stats)


class ClusteringVisualizer:
    """
    Visualizations for operating condition clustering.
    """
    
    @staticmethod
    def plot_3d_clusters(df: pd.DataFrame, labels: np.ndarray,
                         save_path: Optional[str] = None):
        """
        3D scatter plot of operating conditions colored by cluster.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            df['setting_1'], 
            df['setting_2'], 
            df['setting_3'],
            c=labels,
            cmap='tab10',
            alpha=0.5,
            s=10
        )
        
        ax.set_xlabel('Setting 1 (Altitude)', fontweight='bold')
        ax.set_ylabel('Setting 2 (Mach)', fontweight='bold')
        ax.set_zlabel('Setting 3 (Throttle)', fontweight='bold')
        ax.set_title('Operating Condition Clusters', fontweight='bold', pad=15)
        
        # Add legend
        legend = ax.legend(*scatter.legend_elements(), title="Clusters", loc='upper left')
        ax.add_artist(legend)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Clustering] Saved 3D plot to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_cluster_distribution(labels: np.ndarray, save_path: Optional[str] = None):
        """
        Bar chart of cluster distribution.
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(unique, counts, edgecolor='black', alpha=0.7)
        
        # Color bars
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Operating Condition Cluster', fontweight='bold')
        ax.set_ylabel('Number of Samples', fontweight='bold')
        ax.set_title('Distribution of Operating Conditions', fontweight='bold')
        ax.set_xticks(unique)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for i, count in enumerate(counts):
            ax.text(unique[i], count + 50, f'{count}\n({count/sum(counts)*100:.1f}%)', 
                    ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Clustering] Saved distribution plot to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_sensor_by_condition(df: pd.DataFrame, sensor_name: str,
                                 save_path: Optional[str] = None):
        """
        Box plot of a sensor's values across different operating conditions.
        """
        if 'operating_condition' not in df.columns:
            raise ValueError("Dataframe must have 'operating_condition' column")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        conditions = sorted(df['operating_condition'].unique())
        data = [df[df['operating_condition'] == c][sensor_name].values for c in conditions]
        
        bp = ax.boxplot(data, labels=[f'Cond {c}' for c in conditions], patch_artist=True)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Operating Condition', fontweight='bold')
        ax.set_ylabel(f'{sensor_name} Value', fontweight='bold')
        ax.set_title(f'{sensor_name} Distribution by Operating Condition', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Clustering] Saved sensor plot to {save_path}")
        
        return fig


def demo_clustering():
    """
    Demonstration of operating condition clustering.
    """
    import os
    from run_evaluation import CMAPSSDataLoader
    
    print("\n" + "="*70)
    print("OPERATING CONDITION CLUSTERING DEMO")
    print("="*70)
    
    # Load FD002 or FD004 (they have multiple operating conditions)
    loader = CMAPSSDataLoader('./data')
    
    # Try FD002 first (6 operating conditions)
    try:
        train_df, test_df, true_rul = loader.load_fd_dataset(2)
        dataset_name = "FD002"
    except FileNotFoundError:
        try:
            train_df, test_df, true_rul = loader.load_fd_dataset(1)
            dataset_name = "FD001"
            print("[WARNING] FD002 not found, using FD001 (fewer conditions)")
        except FileNotFoundError:
            print("[ERROR] C-MAPSS data not found. Place data in ./data/")
            return
    
    print(f"\nUsing dataset: {dataset_name}")
    
    # Create clusterer
    print("\n[1/4] Fitting K-Means clusterer...")
    n_clusters = 6 if dataset_name != "FD001" else 1
    clusterer = OperatingConditionClusterer(n_clusters=n_clusters)
    labels = clusterer.fit_predict(train_df)
    
    # Add labels to dataframe
    train_df = clusterer.add_cluster_labels(train_df)
    
    # Get statistics
    print("\n[2/4] Cluster statistics:")
    stats = clusterer.get_cluster_statistics(train_df)
    print(stats.to_string(index=False))
    
    # Normalize by condition
    print("\n[3/4] Normalizing sensors by operating condition...")
    normalized_df = clusterer.normalize_by_condition(train_df)
    
    # Visualize
    print("\n[4/4] Generating visualizations...")
    os.makedirs('./results', exist_ok=True)
    
    viz = ClusteringVisualizer()
    viz.plot_3d_clusters(train_df, labels, f'./results/{dataset_name}_operating_conditions_3d.png')
    viz.plot_cluster_distribution(labels, f'./results/{dataset_name}_condition_distribution.png')
    
    if 'sensor_2' in train_df.columns:
        viz.plot_sensor_by_condition(train_df, 'sensor_2', 
                                     f'./results/{dataset_name}_sensor2_by_condition.png')
    
    print("\n" + "="*70)
    print(f"[OK] Demo complete! Check ./results/ for {dataset_name} visualizations.")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Operating Condition Clustering")
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--clusters', type=int, default=6, help='Number of clusters (default: 6)')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_clustering()
    else:
        print("Use --demo to run the demonstration.")
        print("Example: python condition_clustering.py --demo")
