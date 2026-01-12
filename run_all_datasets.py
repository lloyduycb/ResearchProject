import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from typing import Dict, List

# Fix Windows encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from run_evaluation import run_comprehensive_evaluation
from rul_analysis_toolkit import VisualizationSuite

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300


class MultiDatasetAnalyzer:
    """
    Analyze model performance across all four FD datasets.
    """
    
    def __init__(self, output_dir: str = './results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.all_results = {}
        self.all_robustness = {}
        
    def run_all_datasets(self, data_path: str = './data', 
                        predictions_path: str = './predictions'):
        """
        Run evaluation on all four FD datasets.
        """
        print("\n" + "="*80)
        print(" "*20 + "MULTI-DATASET COMPREHENSIVE ANALYSIS")
        print("="*80)
        print("\nThis will analyze performance across all C-MAPSS datasets:")
        print("  FD001: Single condition, single fault (BASELINE)")
        print("  FD002: Multi conditions, single fault")
        print("  FD003: Single condition, multi faults")
        print("  FD004: Multi conditions, multi faults (MOST COMPLEX)")
        print("="*80)
        
        fd_datasets = [1, 2, 3, 4]
        
        for fd_num in fd_datasets:
            print(f"\n{'='*80}")
            print(f" "*25 + f"ANALYZING FD00{fd_num}")
            print(f"{'='*80}")
            
            try:
                results, robustness = run_comprehensive_evaluation(
                    fd_number=fd_num,
                    data_path=data_path,
                    predictions_path=predictions_path
                )
                
                self.all_results[f'FD00{fd_num}'] = results
                self.all_robustness[f'FD00{fd_num}'] = robustness
                
                print(f"\n[OK] FD00{fd_num} analysis complete!")
                
            except Exception as e:
                print(f"\n[WARNING] Error analyzing FD00{fd_num}: {e}")
                print(f"  Continuing with other datasets...")
                continue
        
        # Generate cross-dataset comparisons
        if len(self.all_results) > 0:
            self._generate_cross_dataset_analysis()
    
    def _generate_cross_dataset_analysis(self):
        """
        Generate comparative analysis across datasets.
        """
        print("\n" + "="*80)
        print(" "*20 + "CROSS-DATASET COMPARATIVE ANALYSIS")
        print("="*80)
        
        # 1. Performance across datasets
        self._plot_performance_across_datasets()
        
        # 2. Robustness across datasets
        self._plot_robustness_across_datasets()
        
        # 3. Complexity impact analysis
        self._analyze_complexity_impact()
        
        # 4. Generate summary tables
        self._generate_summary_tables()
        
        print("\n[OK] Cross-dataset analysis complete!")
    
    def _plot_performance_across_datasets(self):
        """
        Plot model performance (RMSE) across all datasets.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Across C-MAPSS Datasets', 
                     fontweight='bold', fontsize=16)
        
        datasets = list(self.all_results.keys())
        models = list(self.all_results[datasets[0]].keys())
        
        # Metrics to plot
        metrics = [
            ('RMSE', 'Root Mean Squared Error (cycles)', 0),
            ('MAE', 'Mean Absolute Error (cycles)', 1),
            ('R2', 'R^2 Score', 2),
            ('PHM_Score', 'PHM Score (lower is better)', 3)
        ]
        
        for metric_name, ylabel, idx in metrics:
            ax = axes[idx // 2, idx % 2]
            
            # Reshape for grouped bar plot
            x = np.arange(len(models))
            width = 0.2
            
            for i, dataset in enumerate(datasets):
                values = [self.all_results[dataset][model].get(metric_name, 0) 
                         for model in models if model in self.all_results[dataset]]
                ax.bar(x + i*width, values, width, 
                      label=dataset, alpha=0.8, edgecolor='black')
            
            ax.set_xlabel('Model', fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(f'{metric_name} Comparison', fontweight='bold')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend(title='Dataset')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/all_datasets_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: all_datasets_performance_comparison.png")
        plt.close()
    
    def _plot_robustness_across_datasets(self):
        """
        Plot robustness scores across all datasets.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Model Robustness Across C-MAPSS Datasets', 
                     fontweight='bold', fontsize=16)
        
        datasets = list(self.all_robustness.keys())
        models = list(self.all_robustness[datasets[0]].keys())
        
        # 1. Robustness scores
        x = np.arange(len(models))
        width = 0.2
        
        for i, dataset in enumerate(datasets):
            scores = [self.all_robustness[dataset][model]['Robustness_Score'] 
                     for model in models if model in self.all_robustness[dataset]]
            bars = ax1.bar(x + i*width, scores, width, 
                          label=dataset, alpha=0.8, edgecolor='black')
            
            # Color code by robustness
            for bar, score in zip(bars, scores):
                if score > 90:
                    bar.set_color('green')
                elif score > 75:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        ax1.set_xlabel('Model', fontweight='bold')
        ax1.set_ylabel('Robustness Score', fontweight='bold')
        ax1.set_title('Robustness Scores (Higher = Better)', fontweight='bold')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend(title='Dataset')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Excellent (>90)')
        ax1.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='Good (>75)')
        
        # 2. Performance degradation
        for i, dataset in enumerate(datasets):
            degradations = [self.all_robustness[dataset][model]['Degradation_Percent'] 
                           for model in models if model in self.all_robustness[dataset]]
            ax2.bar(x + i*width, degradations, width, 
                   label=dataset, alpha=0.8, edgecolor='black')
        
        ax2.set_xlabel('Model', fontweight='bold')
        ax2.set_ylabel('Performance Degradation (%)', fontweight='bold')
        ax2.set_title('Performance Degradation Under Noise (Lower = Better)', 
                     fontweight='bold')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend(title='Dataset')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/all_datasets_robustness_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: all_datasets_robustness_comparison.png")
        plt.close()
    
    def _analyze_complexity_impact(self):
        """
        Analyze how dataset complexity affects model performance.
        """
        print("\n" + "="*80)
        print("COMPLEXITY IMPACT ANALYSIS")
        print("="*80)
        
        datasets = list(self.all_results.keys())
        models = list(self.all_results[datasets[0]].keys())
        
        # Define complexity order
        complexity = {
            'FD001': 1,  # Simple: Single condition, single fault
            'FD002': 2,  # Medium: Multi conditions, single fault
            'FD003': 2,  # Medium: Single condition, multi faults
            'FD004': 3   # Complex: Multi conditions, multi faults
        }
        
        print("\nDataset Complexity Ranking:")
        print("  FD001: * (Single condition, single fault)")
        print("  FD002: ** (Multiple conditions, single fault)")
        print("  FD003: ** (Single condition, multiple faults)")
        print("  FD004: *** (Multiple conditions, multiple faults)")
        
        # Analyze RMSE degradation from simple to complex
        print("\n" + "-"*80)
        print("RMSE Degradation from FD001 (baseline) to FD004 (complex):")
        print("-"*80)
        
        for model in models:
            if model in self.all_results['FD001'] and model in self.all_results.get('FD004', {}):
                rmse_fd001 = self.all_results['FD001'][model]['RMSE']
                rmse_fd004 = self.all_results.get('FD004', {}).get(model, {}).get('RMSE', 0)
                
                if rmse_fd004 > 0:
                    degradation = ((rmse_fd004 - rmse_fd001) / rmse_fd001) * 100
                    
                    print(f"\n{model}:")
                    print(f"  FD001 RMSE: {rmse_fd001:.2f}")
                    print(f"  FD004 RMSE: {rmse_fd004:.2f}")
                    print(f"  Degradation: {degradation:+.1f}%")
                    
                    if degradation < 20:
                        print(f"  Assessment: [OK] Good generalization")
                    elif degradation < 50:
                        print(f"  Assessment: [WARNING] Moderate degradation")
                    else:
                        print(f"  Assessment: [X] Poor generalization")
        
        # Plot complexity impact
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for model in models:
            rmse_values = []
            dataset_names = []
            
            for dataset in sorted(datasets, key=lambda x: complexity.get(x, 0)):
                if model in self.all_results[dataset]:
                    rmse_values.append(self.all_results[dataset][model]['RMSE'])
                    dataset_names.append(dataset)
            
            ax.plot(dataset_names, rmse_values, marker='o', linewidth=2, 
                   markersize=8, label=model)
        
        ax.set_xlabel('Dataset (Increasing Complexity -->)', fontweight='bold', fontsize=12)
        ax.set_ylabel('RMSE (cycles)', fontweight='bold', fontsize=12)
        ax.set_title('Model Performance vs. Dataset Complexity', 
                    fontweight='bold', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/complexity_impact_analysis.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n[OK] Saved: complexity_impact_analysis.png")
        plt.close()
    
    def _generate_summary_tables(self):
        """
        Generate comprehensive summary tables across all datasets.
        """
        print("\n" + "="*80)
        print("SUMMARY TABLES")
        print("="*80)
        
        datasets = list(self.all_results.keys())
        models = list(self.all_results[datasets[0]].keys())
        
        # 1. RMSE Summary Table
        rmse_data = []
        for model in models:
            row = {'Model': model}
            for dataset in datasets:
                if model in self.all_results[dataset]:
                    row[dataset] = self.all_results[dataset][model]['RMSE']
            rmse_data.append(row)
        
        rmse_df = pd.DataFrame(rmse_data)
        
        print("\nRMSE ACROSS ALL DATASETS:")
        print("="*80)
        print(rmse_df.to_string(index=False))
        
        rmse_df.to_csv(f'{self.output_dir}/all_datasets_rmse_summary.csv', index=False)
        print(f"\n[OK] Saved: all_datasets_rmse_summary.csv")
        
        # 2. Robustness Summary Table
        robustness_data = []
        for model in models:
            row = {'Model': model}
            for dataset in datasets:
                if model in self.all_robustness[dataset]:
                    row[f'{dataset}_Score'] = self.all_robustness[dataset][model]['Robustness_Score']
            robustness_data.append(row)
        
        robustness_df = pd.DataFrame(robustness_data)
        
        print("\nROBUSTNESS SCORES ACROSS ALL DATASETS:")
        print("="*80)
        print(robustness_df.to_string(index=False))
        
        robustness_df.to_csv(f'{self.output_dir}/all_datasets_robustness_summary.csv', 
                            index=False)
        print(f"\n[OK] Saved: all_datasets_robustness_summary.csv")
        
        # 3. Best Model per Dataset
        print("\nBEST MODEL PER DATASET (by RMSE):")
        print("="*80)
        
        for dataset in datasets:
            best_model = min(self.all_results[dataset].items(), 
                           key=lambda x: x[1]['RMSE'])
            print(f"{dataset}: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.2f})")
        
        # 4. Overall Rankings
        print("\nOVERALL MODEL RANKINGS (Average RMSE across datasets):")
        print("="*80)
        
        avg_rmse = {}
        for model in models:
            rmse_values = [self.all_results[dataset][model]['RMSE'] 
                          for dataset in datasets if model in self.all_results[dataset]]
            avg_rmse[model] = np.mean(rmse_values)
        
        ranked_models = sorted(avg_rmse.items(), key=lambda x: x[1])
        
        for rank, (model, rmse) in enumerate(ranked_models, 1):
            medal = "[1st]" if rank == 1 else "[2nd]" if rank == 2 else "[3rd]" if rank == 3 else "    "
            print(f"{rank}. {medal} {model}: {rmse:.2f}")
        
        # Save rankings
        ranking_df = pd.DataFrame([
            {'Rank': i+1, 'Model': model, 'Avg_RMSE': rmse}
            for i, (model, rmse) in enumerate(ranked_models)
        ])
        ranking_df.to_csv(f'{self.output_dir}/model_rankings.csv', index=False)
        print(f"\n[OK] Saved: model_rankings.csv")


def main():
    """
    Main execution function.
    """
    print("\n" + "="*80)
    print(" "*15 + "C-MAPSS COMPREHENSIVE MULTI-DATASET ANALYSIS")
    print("="*80)
    
    analyzer = MultiDatasetAnalyzer('./results')
    
    # Run analysis on all datasets
    analyzer.run_all_datasets(
        data_path='./data',
        predictions_path='./predictions'
    )
    
    print("\n" + "="*80)
    print(" "*25 + "ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\n[FILES] OUTPUTS GENERATED:")
    print("  Individual Dataset Results:")
    print("    * FD001_results.csv, FD002_results.csv, etc.")
    print("    * FD001_robustness.csv, FD002_robustness.csv, etc.")
    print("    * Individual visualizations for each dataset")
    
    print("\n  Cross-Dataset Comparisons:")
    print("    * all_datasets_performance_comparison.png")
    print("    * all_datasets_robustness_comparison.png")
    print("    * complexity_impact_analysis.png")
    print("    * all_datasets_rmse_summary.csv")
    print("    * all_datasets_robustness_summary.csv")
    print("    * model_rankings.csv")
    
    print("\n[TIP] KEY INSIGHTS TO EMPHASIZE IN YOUR PAPER:")
    print("  1. How models perform across different complexities")
    print("  2. Which models generalize best (FD001 --> FD004)")
    print("  3. Robustness consistency across operating conditions")
    print("  4. Overall model rankings based on comprehensive evaluation")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
