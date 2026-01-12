"""
Advanced RUL Analysis Toolkit for NASA C-MAPSS Dataset
======================================================
This toolkit provides comprehensive analysis capabilities beyond standard metrics,
including robustness testing, uncertainty quantification, and interpretability.

Author: Research Project CYS6001-20
Date: January 12, 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting defaults
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.figsize'] = (12, 8)


class RULAnalyzer:
    """
    Advanced analyzer for Remaining Useful Life predictions with 
    security-focused robustness metrics.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results = {}
        
    def phm_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate PHM08 Challenge asymmetric scoring function.
        Penalizes late predictions more heavily than early ones.
        """
        diff = y_pred - y_true
        score = np.sum(np.where(diff < 0, 
                                np.exp(-diff/13) - 1,
                                np.exp(diff/10) - 1))
        return score
    
    def compute_standard_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate standard regression metrics."""
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'PHM_Score': self.phm_score(y_true, y_pred),
            'Max_Error': np.max(np.abs(y_true - y_pred)),
            'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        }
    
    def compute_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    confidence: float = 0.95) -> Dict:
        """
        Bootstrap confidence intervals for metrics.
        Critical for statistical significance claims.
        """
        n_bootstrap = 1000
        rmse_boots = []
        mae_boots = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            rmse_boots.append(np.sqrt(mean_squared_error(y_true[indices], y_pred[indices])))
            mae_boots.append(mean_absolute_error(y_true[indices], y_pred[indices]))
        
        alpha = 1 - confidence
        return {
            'RMSE_CI': (np.percentile(rmse_boots, alpha/2 * 100),
                       np.percentile(rmse_boots, (1-alpha/2) * 100)),
            'MAE_CI': (np.percentile(mae_boots, alpha/2 * 100),
                      np.percentile(mae_boots, (1-alpha/2) * 100))
        }
    
    def analyze_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze error distribution characteristics.
        Important for understanding model bias and variance.
        """
        errors = y_pred - y_true
        return {
            'Mean_Error': np.mean(errors),
            'Std_Error': np.std(errors),
            'Skewness': stats.skew(errors),
            'Kurtosis': stats.kurtosis(errors),
            'Q25_Error': np.percentile(errors, 25),
            'Q75_Error': np.percentile(errors, 75),
            'IQR': np.percentile(errors, 75) - np.percentile(errors, 25)
        }
    
    def early_vs_late_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 threshold: int = 50) -> Dict:
        """
        Compare performance on early vs late life predictions.
        Critical for maintenance planning.
        """
        early_mask = y_true > threshold
        late_mask = y_true <= threshold
        
        return {
            'Early_Life_RMSE': np.sqrt(mean_squared_error(y_true[early_mask], 
                                                          y_pred[early_mask])),
            'Late_Life_RMSE': np.sqrt(mean_squared_error(y_true[late_mask], 
                                                         y_pred[late_mask])),
            'Early_Life_MAE': mean_absolute_error(y_true[early_mask], y_pred[early_mask]),
            'Late_Life_MAE': mean_absolute_error(y_true[late_mask], y_pred[late_mask])
        }
    
    def robustness_to_noise(self, y_true: np.ndarray, clean_pred: np.ndarray,
                           noisy_pred: np.ndarray, noise_level: float) -> Dict:
        """
        Quantify robustness by comparing clean vs noisy predictions.
        CRITICAL for cyber-security assessment.
        """
        clean_rmse = np.sqrt(mean_squared_error(y_true, clean_pred))
        noisy_rmse = np.sqrt(mean_squared_error(y_true, noisy_pred))
        
        degradation = (noisy_rmse - clean_rmse) / clean_rmse * 100
        
        # Correlation between clean and noisy predictions
        correlation = np.corrcoef(clean_pred, noisy_pred)[0, 1]
        
        return {
            'Clean_RMSE': clean_rmse,
            'Noisy_RMSE': noisy_rmse,
            'Degradation_Percent': degradation,
            'Prediction_Correlation': correlation,
            'Noise_Level': noise_level,
            'Robustness_Score': 100 - degradation  # Higher is better
        }


class VisualizationSuite:
    """
    Publication-quality visualization suite for RUL analysis.
    """
    
    @staticmethod
    def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                                  model_name: str, save_path: Optional[str] = None):
        """Scatter plot with regression line and confidence bands."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val, max_val = 0, max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
                label='Perfect Prediction')
        
        # Add regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_true, p(y_true), "b-", alpha=0.8, lw=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Metrics annotation
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        textstr = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel('True RUL (cycles)', fontweight='bold')
        ax.set_ylabel('Predicted RUL (cycles)', fontweight='bold')
        ax.set_title(f'{model_name}: Prediction vs Actual RUL', fontweight='bold', pad=20)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    @staticmethod
    def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str, save_path: Optional[str] = None):
        """Comprehensive error distribution analysis."""
        errors = y_pred - y_true
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{model_name}: Error Distribution Analysis', 
                     fontweight='bold', fontsize=14)
        
        # 1. Histogram with KDE
        axes[0, 0].hist(errors, bins=50, density=True, alpha=0.7, 
                       edgecolor='black', label='Histogram')
        kde_x = np.linspace(errors.min(), errors.max(), 100)
        kde = stats.gaussian_kde(errors)
        axes[0, 0].plot(kde_x, kde(kde_x), 'r-', lw=2, label='KDE')
        axes[0, 0].axvline(0, color='green', linestyle='--', lw=2, label='Zero Error')
        axes[0, 0].set_xlabel('Prediction Error (cycles)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q Plot for normality
        stats.probplot(errors, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normality Check)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error vs True RUL (heteroscedasticity check)
        axes[1, 0].scatter(y_true, errors, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
        axes[1, 0].axhline(0, color='red', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('True RUL (cycles)')
        axes[1, 0].set_ylabel('Prediction Error (cycles)')
        axes[1, 0].set_title('Residual Plot (Heteroscedasticity Check)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Boxplot
        axes[1, 1].boxplot(errors, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[1, 1].axhline(0, color='red', linestyle='--', lw=2)
        axes[1, 1].set_ylabel('Prediction Error (cycles)')
        axes[1, 1].set_title('Error Boxplot')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = (f'Mean: {np.mean(errors):.2f}\n'
                     f'Std: {np.std(errors):.2f}\n'
                     f'Skew: {stats.skew(errors):.2f}\n'
                     f'Kurt: {stats.kurtosis(errors):.2f}')
        axes[1, 1].text(1.15, 0.5, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    @staticmethod
    def plot_robustness_comparison(robustness_results: Dict[str, Dict],
                                  save_path: Optional[str] = None):
        """
        Compare robustness across models - CRITICAL for security analysis.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Model Robustness to Sensor Noise: Security Assessment',
                     fontweight='bold', fontsize=14)
        
        models = list(robustness_results.keys())
        degradations = [robustness_results[m]['Degradation_Percent'] for m in models]
        robustness_scores = [robustness_results[m]['Robustness_Score'] for m in models]
        
        # 1. Performance Degradation
        bars1 = axes[0].bar(models, degradations, alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('Performance Degradation (%)', fontweight='bold')
        axes[0].set_title('Performance Degradation under Noise\n(Lower is Better)')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Color bars based on severity
        for i, (bar, deg) in enumerate(zip(bars1, degradations)):
            if deg < 10:
                bar.set_color('green')
            elif deg < 25:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add value labels
        for i, (model, deg) in enumerate(zip(models, degradations)):
            axes[0].text(i, deg + 1, f'{deg:.1f}%', ha='center', va='bottom', 
                        fontweight='bold')
        
        # 2. Robustness Score
        bars2 = axes[1].bar(models, robustness_scores, alpha=0.7, 
                           edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('Robustness Score', fontweight='bold')
        axes[1].set_title('Overall Robustness Score\n(Higher is Better)')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Color bars
        for bar, score in zip(bars2, robustness_scores):
            if score > 90:
                bar.set_color('green')
            elif score > 75:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add value labels
        for i, (model, score) in enumerate(zip(models, robustness_scores)):
            axes[1].text(i, score + 1, f'{score:.1f}', ha='center', va='bottom',
                        fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    @staticmethod
    def plot_lifecycle_performance(y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray],
                                  save_path: Optional[str] = None):
        """
        Analyze performance across engine lifecycle stages.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Performance Across Engine Lifecycle', 
                     fontweight='bold', fontsize=14)
        
        # Define lifecycle stages
        stages = {
            'Early Life (>100 cycles)': y_true > 100,
            'Mid Life (50-100 cycles)': (y_true >= 50) & (y_true <= 100),
            'Late Life (20-50 cycles)': (y_true >= 20) & (y_true < 50),
            'Critical (<20 cycles)': y_true < 20
        }
        
        for idx, (stage_name, mask) in enumerate(stages.items()):
            ax = axes[idx // 2, idx % 2]
            
            for model_name, y_pred in predictions_dict.items():
                if mask.sum() > 0:
                    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                    ax.bar(model_name, rmse, alpha=0.7, label=model_name)
            
            ax.set_ylabel('RMSE (cycles)', fontweight='bold')
            ax.set_title(f'{stage_name}\n(n={mask.sum()} samples)')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


class ComprehensiveReport:
    """
    Generate a comprehensive analysis report with all metrics and visualizations.
    """
    
    def __init__(self, output_dir: str = './results'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_comparison_table(self, results: Dict[str, Dict], latex_format: bool = False) -> pd.DataFrame:
        """
        Generate comparison table.
        
        Args:
            results: Dictionary of model results
            latex_format: If True, add LaTeX bold formatting for best values
        """
        df = pd.DataFrame(results).T
        df = df.round(2)
        
        # Only add LaTeX formatting if requested
        if latex_format:
            # Highlight best performance
            for col in df.columns:
                if col in ['RMSE', 'MAE', 'PHM_Score', 'Max_Error', 'MAPE']:
                    best_idx = df[col].idxmin()
                else:  # R2, Robustness_Score
                    best_idx = df[col].idxmax()
                df.loc[best_idx, col] = f"\\textbf{{{df.loc[best_idx, col]}}}"
        
        return df
    
    def export_latex_table(self, df: pd.DataFrame, filename: str):
        """Export results as LaTeX table."""
        latex_str = df.to_latex(escape=False, caption='Model Performance Comparison',
                               label='tab:results')
        
        with open(f'{self.output_dir}/{filename}', 'w') as f:
            f.write(latex_str)
        
        print(f"LaTeX table saved to {self.output_dir}/{filename}")
    
    def generate_statistical_significance_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Perform pairwise statistical significance testing.
        """
        models = list(results.keys())
        n_models = len(models)
        
        # Create significance matrix
        sig_matrix = np.zeros((n_models, n_models))
        
        # This is simplified - in real analysis, you'd use bootstrap or permutation tests
        for i, model_i in enumerate(models):
            for j, model_j in enumerate(models):
                if i != j:
                    # Calculate effect size (Cohen's d)
                    rmse_i = results[model_i]['RMSE']
                    rmse_j = results[model_j]['RMSE']
                    sig_matrix[i, j] = abs(rmse_i - rmse_j)
        
        df = pd.DataFrame(sig_matrix, index=models, columns=models)
        return df.round(3)


# Example usage function
def run_complete_analysis():
    """
    Example of running complete analysis pipeline.
    Replace with your actual data and predictions.
    """
    
    # Simulated data (replace with your actual data)
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.exponential(50, n_samples)
    
    # Simulated predictions for different models
    predictions = {
        'SVR': y_true + np.random.normal(0, 10, n_samples),
        'LSTM': y_true + np.random.normal(0, 7, n_samples),
        'TCN': y_true + np.random.normal(0, 5, n_samples),
        'Transformer': y_true + np.random.normal(0, 5.5, n_samples)
    }
    
    # Noisy predictions for robustness analysis
    noisy_predictions = {
        model: pred + np.random.normal(0, 3, n_samples) 
        for model, pred in predictions.items()
    }
    
    # Analysis pipeline
    report = ComprehensiveReport('./results')
    viz = VisualizationSuite()
    
    all_results = {}
    robustness_results = {}
    
    for model_name, y_pred in predictions.items():
        print(f"\n{'='*50}")
        print(f"Analyzing {model_name}")
        print(f"{'='*50}")
        
        analyzer = RULAnalyzer(model_name)
        
        # Standard metrics
        metrics = analyzer.compute_standard_metrics(y_true, y_pred)
        print("\nStandard Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        # Confidence intervals
        ci = analyzer.compute_confidence_intervals(y_true, y_pred)
        print("\n95% Confidence Intervals:")
        print(f"  RMSE: [{ci['RMSE_CI'][0]:.2f}, {ci['RMSE_CI'][1]:.2f}]")
        print(f"  MAE: [{ci['MAE_CI'][0]:.2f}, {ci['MAE_CI'][1]:.2f}]")
        
        # Error distribution
        error_stats = analyzer.analyze_error_distribution(y_true, y_pred)
        print("\nError Distribution:")
        for stat, value in error_stats.items():
            print(f"  {stat}: {value:.3f}")
        
        # Lifecycle performance
        lifecycle = analyzer.early_vs_late_performance(y_true, y_pred)
        print("\nLifecycle Performance:")
        for stage, value in lifecycle.items():
            print(f"  {stage}: {value:.3f}")
        
        # Robustness
        robustness = analyzer.robustness_to_noise(
            y_true, y_pred, noisy_predictions[model_name], noise_level=0.01
        )
        print("\nRobustness Analysis:")
        for metric, value in robustness.items():
            print(f"  {metric}: {value:.3f}")
        
        # Store results
        all_results[model_name] = {**metrics, **lifecycle}
        robustness_results[model_name] = robustness
        
        # Generate visualizations
        viz.plot_prediction_vs_actual(
            y_true, y_pred, model_name,
            f'./results/{model_name}_prediction_vs_actual.png'
        )
        
        viz.plot_error_distribution(
            y_true, y_pred, model_name,
            f'./results/{model_name}_error_distribution.png'
        )
    
    # Comparative visualizations
    viz.plot_robustness_comparison(
        robustness_results,
        './results/robustness_comparison.png'
    )
    
    viz.plot_lifecycle_performance(
        y_true, predictions,
        './results/lifecycle_performance.png'
    )
    
    # Generate reports
    comparison_df = report.generate_comparison_table(all_results)
    print("\n" + "="*50)
    print("FINAL COMPARISON TABLE")
    print("="*50)
    print(comparison_df)
    
    report.export_latex_table(comparison_df, 'results_table.tex')
    
    significance_df = report.generate_statistical_significance_table(all_results)
    print("\n" + "="*50)
    print("STATISTICAL SIGNIFICANCE (RMSE Differences)")
    print("="*50)
    print(significance_df)
    
    return all_results, robustness_results


if __name__ == "__main__":
    print("RUL Analysis Toolkit - Advanced Evaluation Suite")
    print("="*60)
    results, robustness = run_complete_analysis()
    print("\n✓ Analysis complete! Check the './results' directory for outputs.")
