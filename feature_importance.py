"""
Feature Importance & Interpretability Module
============================================
Understand which sensors drive RUL predictions using 
permutation importance and correlation analysis.

Author: Research Project CYS6001-20
Date: January 12, 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureAnalyzer:
    """
    Analyze feature importance for RUL prediction.
    
    Uses:
    1. Permutation importance (model-agnostic)
    2. Correlation analysis (fast, interpretable)
    3. Random Forest feature importance (built-in)
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.importance_scores = None
        self.fitted = False
        
    def _get_sensor_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract sensor columns from dataframe."""
        return [col for col in df.columns if col.startswith('sensor_')]
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for importance analysis.
        Uses the last observation per engine with its RUL as target.
        """
        sensor_cols = self._get_sensor_columns(df)
        
        # Get last observation per engine (test-like scenario)
        last_obs = df.groupby('unit_id').last().reset_index()
        
        X = last_obs[sensor_cols].values
        y = last_obs['RUL'].values
        
        return X, y, sensor_cols
    
    def fit(self, train_df: pd.DataFrame) -> 'FeatureAnalyzer':
        """
        Fit a Random Forest to analyze feature importance.
        """
        if 'RUL' not in train_df.columns:
            raise ValueError("DataFrame must have 'RUL' column")
        
        X, y, self.feature_names = self._prepare_features(train_df)
        
        print(f"[FeatureImportance] Training RF on {len(X)} samples, {len(self.feature_names)} features...")
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)
        
        # Store built-in importance
        self.importance_scores = dict(zip(self.feature_names, self.model.feature_importances_))
        self.fitted = True
        
        print(f"[FeatureImportance] Model trained. R² = {self.model.score(X, y):.3f}")
        return self
    
    def compute_permutation_importance(self, df: pd.DataFrame, 
                                       n_repeats: int = 10) -> pd.DataFrame:
        """
        Compute permutation importance (more reliable than built-in).
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X, y, _ = self._prepare_features(df)
        
        print(f"[FeatureImportance] Computing permutation importance ({n_repeats} repeats)...")
        
        result = permutation_importance(
            self.model, X, y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance_Mean': result.importances_mean,
            'Importance_Std': result.importances_std
        })
        
        importance_df = importance_df.sort_values('Importance_Mean', ascending=False)
        return importance_df
    
    def compute_correlation_with_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute correlation between sensors and RUL.
        Fast and interpretable alternative to model-based importance.
        """
        sensor_cols = self._get_sensor_columns(df)
        
        correlations = []
        for sensor in sensor_cols:
            corr = df[sensor].corr(df['RUL'])
            correlations.append({
                'Feature': sensor,
                'Correlation': corr,
                'Abs_Correlation': abs(corr)
            })
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
        return corr_df
    
    def get_top_features(self, n: int = 10) -> List[str]:
        """
        Get top N most important features.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        sorted_features = sorted(self.importance_scores.items(), 
                                 key=lambda x: x[1], reverse=True)
        return [f[0] for f in sorted_features[:n]]
    
    def analyze_feature_degradation(self, df: pd.DataFrame, 
                                    feature: str) -> pd.DataFrame:
        """
        Analyze how a feature changes as RUL decreases (degradation pattern).
        """
        # Bin RUL into stages
        df = df.copy()
        df['RUL_Stage'] = pd.cut(df['RUL'], 
                                 bins=[0, 25, 50, 75, 100, 125, float('inf')],
                                 labels=['<25', '25-50', '50-75', '75-100', '100-125', '>125'])
        
        # Get stats per stage
        stats = df.groupby('RUL_Stage')[feature].agg(['mean', 'std', 'count'])
        stats = stats.reset_index()
        stats.columns = ['RUL_Stage', 'Mean', 'Std', 'Count']
        
        return stats


class ImportanceVisualizer:
    """
    Visualizations for feature importance analysis.
    """
    
    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, 
                                top_n: int = 15,
                                save_path: Optional[str] = None):
        """
        Bar chart of top feature importances.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        data = importance_df.head(top_n)
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(data)), data['Importance_Mean'].values, 
                       xerr=data['Importance_Std'].values if 'Importance_Std' in data.columns else None,
                       edgecolor='black', alpha=0.7, color='steelblue')
        
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['Feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontweight='bold')
        ax.set_title('Feature Importance for RUL Prediction', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[FeatureImportance] Saved plot to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, 
                                top_n: int = 15,
                                save_path: Optional[str] = None):
        """
        Heatmap of correlations between top sensors and RUL.
        """
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        
        # Get correlations with RUL and select top sensors
        correlations = df[sensor_cols].corrwith(df['RUL']).abs()
        top_sensors = correlations.nlargest(top_n).index.tolist()
        
        # Compute correlation matrix for these sensors + RUL
        cols_to_plot = top_sensors + ['RUL']
        corr_matrix = df[cols_to_plot].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=ax,
                    square=True, linewidths=0.5)
        
        ax.set_title('Sensor Correlation with RUL', fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[FeatureImportance] Saved heatmap to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_feature_vs_rul(df: pd.DataFrame, feature: str,
                           save_path: Optional[str] = None):
        """
        Scatter plot of a feature vs RUL with trend line.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(df[feature], df['RUL'], alpha=0.3, s=10)
        
        # Add trend line
        z = np.polyfit(df[feature], df['RUL'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[feature].min(), df[feature].max(), 100)
        axes[0].plot(x_line, p(x_line), 'r-', linewidth=2, label='Trend')
        
        corr = df[feature].corr(df['RUL'])
        axes[0].set_xlabel(feature, fontweight='bold')
        axes[0].set_ylabel('RUL', fontweight='bold')
        axes[0].set_title(f'{feature} vs RUL (r={corr:.3f})', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Distribution by RUL stage
        df_temp = df.copy()
        df_temp['RUL_Stage'] = pd.cut(df['RUL'], 
                                      bins=[0, 50, 100, float('inf')],
                                      labels=['Critical (<50)', 'Warning (50-100)', 'Healthy (>100)'])
        
        for stage in ['Healthy (>100)', 'Warning (50-100)', 'Critical (<50)']:
            stage_data = df_temp[df_temp['RUL_Stage'] == stage][feature]
            if len(stage_data) > 0:
                axes[1].hist(stage_data, bins=30, alpha=0.5, label=stage, density=True)
        
        axes[1].set_xlabel(feature, fontweight='bold')
        axes[1].set_ylabel('Density', fontweight='bold')
        axes[1].set_title(f'{feature} Distribution by Health Stage', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[FeatureImportance] Saved feature plot to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_degradation_pattern(stats_df: pd.DataFrame, feature: str,
                                save_path: Optional[str] = None):
        """
        Plot how a feature degrades as RUL decreases.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(stats_df))
        
        ax.bar(x, stats_df['Mean'].values, yerr=stats_df['Std'].values,
               edgecolor='black', alpha=0.7, capsize=3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df['RUL_Stage'].values, rotation=45, ha='right')
        ax.set_xlabel('RUL Stage (cycles)', fontweight='bold')
        ax.set_ylabel(f'{feature} Mean Value', fontweight='bold')
        ax.set_title(f'{feature} Degradation Pattern', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[FeatureImportance] Saved degradation plot to {save_path}")
        
        return fig


def demo_feature_importance():
    """
    Demonstration of feature importance analysis.
    """
    import os
    from run_evaluation import CMAPSSDataLoader
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE & INTERPRETABILITY DEMO")
    print("="*70)
    
    # Load data
    loader = CMAPSSDataLoader('./data')
    try:
        train_df, test_df, true_rul = loader.load_fd_dataset(1)
        train_df = loader.add_rul_column(train_df)
        train_df, _ = loader.remove_constant_sensors(train_df, test_df)
    except FileNotFoundError:
        print("[ERROR] C-MAPSS data not found. Place data in ./data/")
        return
    
    # Create analyzer
    print("\n[1/5] Training feature importance model...")
    analyzer = FeatureAnalyzer()
    analyzer.fit(train_df)
    
    # Get permutation importance
    print("\n[2/5] Computing permutation importance...")
    perm_importance = analyzer.compute_permutation_importance(train_df)
    print("\nTop 10 Important Features:")
    print(perm_importance.head(10).to_string(index=False))
    
    # Get correlations
    print("\n[3/5] Computing correlations with RUL...")
    correlations = analyzer.compute_correlation_with_rul(train_df)
    print("\nTop 10 Correlated Features:")
    print(correlations.head(10).to_string(index=False))
    
    # Analyze top feature degradation
    print("\n[4/5] Analyzing degradation patterns...")
    top_feature = analyzer.get_top_features(1)[0]
    degradation = analyzer.analyze_feature_degradation(train_df, top_feature)
    print(f"\n{top_feature} Degradation Pattern:")
    print(degradation.to_string(index=False))
    
    # Visualize
    print("\n[5/5] Generating visualizations...")
    os.makedirs('./results', exist_ok=True)
    
    viz = ImportanceVisualizer()
    viz.plot_feature_importance(perm_importance, top_n=15, 
                                save_path='./results/feature_importance.png')
    viz.plot_correlation_heatmap(train_df, top_n=10,
                                 save_path='./results/correlation_heatmap.png')
    viz.plot_feature_vs_rul(train_df, top_feature,
                            save_path=f'./results/{top_feature}_vs_rul.png')
    viz.plot_degradation_pattern(degradation, top_feature,
                                 save_path=f'./results/{top_feature}_degradation.png')
    
    print("\n" + "="*70)
    print("[OK] Demo complete! Check ./results/ for visualizations.")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Importance Analysis")
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_feature_importance()
    else:
        print("Use --demo to run the demonstration.")
        print("Example: python feature_importance.py --demo")
