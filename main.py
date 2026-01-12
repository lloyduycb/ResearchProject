"""
NASA C-MAPSS RUL Analysis Toolkit - Unified Entry Point
========================================================
Main entry point providing access to all toolkit features via subcommands.

Usage:
    python main.py evaluate --fd 1
    python main.py anomaly --demo
    python main.py cluster --demo
    python main.py forecast --demo
    python main.py features --demo

Author: Research Project CYS6001-20
Date: January 12, 2026
"""

import argparse
import os
import sys
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def cmd_evaluate(args, config):
    """Run RUL evaluation pipeline."""
    from run_evaluation import run_comprehensive_evaluation
    
    # Use config defaults if not specified
    fd = args.fd or config.get('dataset', {}).get('fd_number', 1)
    data_path = args.data_path or config.get('paths', {}).get('data', './data')
    pred_path = args.predictions_path or config.get('paths', {}).get('predictions', './predictions')
    out_path = args.output_path or config.get('paths', {}).get('results', './results')
    
    os.makedirs(out_path, exist_ok=True)
    
    if args.demo:
        # Generate demo predictions
        print("[DEMO MODE] Generating synthetic predictions...")
        import numpy as np
        from save_predictions import save_model_predictions
        
        demo_true_rul = np.random.exponential(50, 100)
        for model in ['SVR', 'LSTM', 'TCN', 'Transformer']:
            noise = {'SVR': 10, 'LSTM': 7, 'TCN': 5, 'Transformer': 5.5}[model]
            fake_pred = np.maximum(demo_true_rul + np.random.normal(0, noise, len(demo_true_rul)), 0)
            save_model_predictions(fake_pred, model, fd, pred_path)
    
    try:
        run_comprehensive_evaluation(fd, data_path, pred_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Use --demo to run with synthetic data")
        sys.exit(1)


def cmd_anomaly(args, config):
    """Run anomaly detection / health index analysis."""
    from anomaly_detection import demo_anomaly_detection, HealthIndexCalculator, AnomalyVisualizer
    from run_evaluation import CMAPSSDataLoader
    
    if args.demo:
        demo_anomaly_detection()
    else:
        fd = args.fd or config.get('dataset', {}).get('fd_number', 1)
        data_path = args.data_path or config.get('paths', {}).get('data', './data')
        out_path = args.output_path or config.get('paths', {}).get('results', './results')
        
        loader = CMAPSSDataLoader(data_path)
        train_df, _, _ = loader.load_fd_dataset(fd)
        train_df = loader.add_rul_column(train_df)
        
        hi_config = config.get('health_index', {})
        calc = HealthIndexCalculator(
            healthy_fraction=hi_config.get('healthy_fraction', 0.2),
            contamination=hi_config.get('contamination', 0.1)
        )
        calc.fit(train_df)
        
        train_df['health_index'] = calc.compute_health_index(train_df)
        
        os.makedirs(out_path, exist_ok=True)
        viz = AnomalyVisualizer()
        viz.plot_fleet_health_summary(train_df, calc, f'{out_path}/FD00{fd}_fleet_health.png')
        print(f"[OK] Health analysis saved to {out_path}/")


def cmd_cluster(args, config):
    """Run operating condition clustering."""
    from condition_clustering import demo_clustering, OperatingConditionClusterer, ClusteringVisualizer
    from run_evaluation import CMAPSSDataLoader
    
    if args.demo:
        demo_clustering()
    else:
        fd = args.fd or config.get('dataset', {}).get('fd_number', 2)  # FD002 has multiple conditions
        data_path = args.data_path or config.get('paths', {}).get('data', './data')
        out_path = args.output_path or config.get('paths', {}).get('results', './results')
        
        loader = CMAPSSDataLoader(data_path)
        train_df, _, _ = loader.load_fd_dataset(fd)
        
        n_clusters = config.get('clustering', {}).get('n_clusters', 6)
        clusterer = OperatingConditionClusterer(n_clusters=n_clusters)
        labels = clusterer.fit_predict(train_df)
        
        os.makedirs(out_path, exist_ok=True)
        viz = ClusteringVisualizer()
        viz.plot_3d_clusters(train_df, labels, f'{out_path}/FD00{fd}_clusters_3d.png')
        viz.plot_cluster_distribution(labels, f'{out_path}/FD00{fd}_cluster_dist.png')
        print(f"[OK] Clustering analysis saved to {out_path}/")


def cmd_forecast(args, config):
    """Run sensor forecasting."""
    from forecasting import demo_forecasting, SensorForecaster, ForecastingVisualizer
    from run_evaluation import CMAPSSDataLoader
    
    if args.demo:
        demo_forecasting()
    else:
        fd = args.fd or config.get('dataset', {}).get('fd_number', 1)
        data_path = args.data_path or config.get('paths', {}).get('data', './data')
        out_path = args.output_path or config.get('paths', {}).get('results', './results')
        
        loader = CMAPSSDataLoader(data_path)
        train_df, test_df, _ = loader.load_fd_dataset(fd)
        train_df = loader.add_rul_column(train_df)
        train_df, _ = loader.remove_constant_sensors(train_df, test_df)
        
        lookback = config.get('forecasting', {}).get('lookback', 10)
        forecaster = SensorForecaster(lookback=lookback)
        forecaster.fit(train_df)
        
        os.makedirs(out_path, exist_ok=True)
        viz = ForecastingVisualizer()
        viz.plot_forecast_accuracy(forecaster, train_df, f'{out_path}/FD00{fd}_forecast_accuracy.png')
        print(f"[OK] Forecasting analysis saved to {out_path}/")


def cmd_features(args, config):
    """Run feature importance analysis."""
    from feature_importance import demo_feature_importance, FeatureAnalyzer, ImportanceVisualizer
    from run_evaluation import CMAPSSDataLoader
    
    if args.demo:
        demo_feature_importance()
    else:
        fd = args.fd or config.get('dataset', {}).get('fd_number', 1)
        data_path = args.data_path or config.get('paths', {}).get('data', './data')
        out_path = args.output_path or config.get('paths', {}).get('results', './results')
        
        loader = CMAPSSDataLoader(data_path)
        train_df, test_df, _ = loader.load_fd_dataset(fd)
        train_df = loader.add_rul_column(train_df)
        train_df, _ = loader.remove_constant_sensors(train_df, test_df)
        
        analyzer = FeatureAnalyzer()
        analyzer.fit(train_df)
        
        importance_df = analyzer.compute_permutation_importance(train_df)
        
        os.makedirs(out_path, exist_ok=True)
        viz = ImportanceVisualizer()
        viz.plot_feature_importance(importance_df, save_path=f'{out_path}/FD00{fd}_feature_importance.png')
        viz.plot_correlation_heatmap(train_df, save_path=f'{out_path}/FD00{fd}_correlation_heatmap.png')
        print(f"[OK] Feature analysis saved to {out_path}/")


def cmd_all(args, config):
    """Run all analyses in sequence."""
    print("\n" + "="*70)
    print("RUNNING ALL ANALYSES")
    print("="*70)
    
    args.demo = True  # Force demo mode for full pipeline
    
    print("\n[1/5] Running Evaluation...")
    cmd_evaluate(args, config)
    
    print("\n[2/5] Running Anomaly Detection...")
    cmd_anomaly(args, config)
    
    print("\n[3/5] Running Clustering...")
    cmd_cluster(args, config)
    
    print("\n[4/5] Running Forecasting...")
    cmd_forecast(args, config)
    
    print("\n[5/5] Running Feature Importance...")
    cmd_features(args, config)
    
    print("\n" + "="*70)
    print("[OK] All analyses complete! Check ./results/ for outputs.")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="NASA C-MAPSS RUL Analysis Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py evaluate --demo         # Run evaluation with synthetic data
  python main.py anomaly --demo          # Run anomaly detection demo
  python main.py cluster --demo          # Run clustering demo
  python main.py forecast --demo         # Run forecasting demo
  python main.py features --demo         # Run feature importance demo
  python main.py all                     # Run all analyses

  python main.py evaluate --fd 3         # Evaluate on FD003
  python main.py cluster --fd 2          # Cluster FD002 (6 conditions)
        """
    )
    
    # Global options
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file (default: config.yaml)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments for all subcommands
    def add_common_args(subparser):
        subparser.add_argument('--fd', type=int, choices=[1, 2, 3, 4],
                              help='FD dataset number')
        subparser.add_argument('--data-path', type=str, help='Path to data')
        subparser.add_argument('--output-path', type=str, help='Path for outputs')
        subparser.add_argument('--demo', action='store_true', help='Run demo mode')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run RUL evaluation')
    add_common_args(eval_parser)
    eval_parser.add_argument('--predictions-path', type=str, help='Path to predictions')
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Anomaly command
    anomaly_parser = subparsers.add_parser('anomaly', help='Anomaly detection / Health Index')
    add_common_args(anomaly_parser)
    anomaly_parser.set_defaults(func=cmd_anomaly)
    
    # Cluster command
    cluster_parser = subparsers.add_parser('cluster', help='Operating condition clustering')
    add_common_args(cluster_parser)
    cluster_parser.set_defaults(func=cmd_cluster)
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Sensor forecasting')
    add_common_args(forecast_parser)
    forecast_parser.set_defaults(func=cmd_forecast)
    
    # Features command
    features_parser = subparsers.add_parser('features', help='Feature importance analysis')
    add_common_args(features_parser)
    features_parser.set_defaults(func=cmd_features)
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run all analyses')
    add_common_args(all_parser)
    all_parser.set_defaults(func=cmd_all)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Load config
    config = load_config(args.config)
    
    print("\n" + "="*70)
    print("NASA C-MAPSS RUL ANALYSIS TOOLKIT")
    print("Research Project CYS6001-20")
    print("="*70)
    
    # Run the command
    args.func(args, config)


if __name__ == "__main__":
    main()
