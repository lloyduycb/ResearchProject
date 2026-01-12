"""
Time-Series Forecasting Module
==============================
Predict future sensor values for "Virtual Sensing" and fault detection.

If predicted != actual, there may be a sensor fault or anomaly.

Author: Research Project CYS6001-20
Date: January 12, 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class SensorForecaster:
    """
    Forecast next-step sensor values using a sliding window approach.
    
    This enables:
    1. Virtual Sensing: Detect when a sensor reading doesn't match expected value
    2. Short-term prediction: Plan maintenance based on predicted trajectories
    """
    
    def __init__(self, lookback: int = 10):
        """
        Args:
            lookback: Number of previous time steps to use for prediction (window size)
        """
        self.lookback = lookback
        self.models = {}  # One model per sensor
        self.scalers = {}  # One scaler per sensor
        self.sensor_cols = None
        self.fitted = False
        
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for training.
        
        Args:
            data: 1D array of sensor values
            
        Returns:
            X: (n_samples, lookback) array of input windows
            y: (n_samples,) array of target values (next step)
        """
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i+self.lookback])
            y.append(data[i+self.lookback])
        return np.array(X), np.array(y)
    
    def _prepare_training_data(self, df: pd.DataFrame, sensor_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from multiple engines.
        """
        all_X, all_y = [], []
        
        for unit_id in df['unit_id'].unique():
            engine_data = df[df['unit_id'] == unit_id].sort_values('time_cycles')
            sensor_values = engine_data[sensor_col].values
            
            if len(sensor_values) > self.lookback:
                X, y = self._create_sequences(sensor_values)
                all_X.append(X)
                all_y.append(y)
        
        if all_X:
            return np.vstack(all_X), np.concatenate(all_y)
        else:
            return np.array([]), np.array([])
    
    def fit(self, train_df: pd.DataFrame, 
            sensor_cols: Optional[List[str]] = None) -> 'SensorForecaster':
        """
        Fit forecasting models for each sensor.
        
        Args:
            train_df: Training dataframe
            sensor_cols: List of sensor columns to forecast (auto-detected if None)
        """
        if sensor_cols is None:
            sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]
        
        self.sensor_cols = sensor_cols
        
        print(f"[Forecasting] Training models for {len(sensor_cols)} sensors...")
        
        for i, sensor in enumerate(sensor_cols):
            X, y = self._prepare_training_data(train_df, sensor)
            
            if len(X) == 0:
                print(f"  [SKIP] {sensor}: Not enough data for lookback={self.lookback}")
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit Ridge regression (fast, robust)
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y)
            
            self.models[sensor] = model
            self.scalers[sensor] = scaler
            
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(sensor_cols)}] Trained {sensor}")
        
        self.fitted = True
        print(f"[Forecasting] Trained {len(self.models)} sensor models.")
        return self
    
    def predict_next(self, sequence: np.ndarray, sensor: str) -> float:
        """
        Predict the next value given a sequence of previous values.
        
        Args:
            sequence: Array of shape (lookback,) with recent sensor values
            sensor: Name of sensor to predict
            
        Returns:
            Predicted next value
        """
        if sensor not in self.models:
            raise ValueError(f"No model for sensor: {sensor}")
        
        if len(sequence) != self.lookback:
            raise ValueError(f"Sequence must have length {self.lookback}, got {len(sequence)}")
        
        X = self.scalers[sensor].transform(sequence.reshape(1, -1))
        return self.models[sensor].predict(X)[0]
    
    def predict_trajectory(self, df: pd.DataFrame, unit_id: int, 
                          sensor: str, steps_ahead: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict trajectory for a specific engine and sensor.
        
        Returns:
            (time_cycles, actual_values, predicted_values)
        """
        engine_data = df[df['unit_id'] == unit_id].sort_values('time_cycles')
        time_cycles = engine_data['time_cycles'].values
        actual = engine_data[sensor].values
        
        predicted = np.full_like(actual, np.nan, dtype=float)
        
        for i in range(self.lookback, len(actual)):
            sequence = actual[i-self.lookback:i]
            predicted[i] = self.predict_next(sequence, sensor)
        
        return time_cycles, actual, predicted
    
    def detect_sensor_faults(self, df: pd.DataFrame, 
                            threshold_std: float = 3.0) -> pd.DataFrame:
        """
        Detect potential sensor faults where actual != predicted significantly.
        
        Args:
            df: Dataframe with sensor data
            threshold_std: Number of standard deviations to flag as anomaly
            
        Returns:
            Dataframe with fault indicators
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        results = []
        
        for unit_id in df['unit_id'].unique():
            for sensor in self.models.keys():
                time_cycles, actual, predicted = self.predict_trajectory(df, unit_id, sensor)
                
                # Calculate prediction errors
                valid_mask = ~np.isnan(predicted)
                errors = actual[valid_mask] - predicted[valid_mask]
                
                # Flag anomalies
                if len(errors) > 0:
                    mean_error = np.mean(errors)
                    std_error = np.std(errors)
                    
                    anomaly_mask = np.abs(errors - mean_error) > threshold_std * std_error
                    
                    if np.any(anomaly_mask):
                        results.append({
                            'unit_id': unit_id,
                            'sensor': sensor,
                            'n_anomalies': np.sum(anomaly_mask),
                            'anomaly_pct': np.mean(anomaly_mask) * 100,
                            'mean_error': mean_error,
                            'std_error': std_error
                        })
        
        return pd.DataFrame(results)


class ForecastingVisualizer:
    """
    Visualizations for sensor forecasting.
    """
    
    @staticmethod
    def plot_prediction_vs_actual(time_cycles: np.ndarray, actual: np.ndarray,
                                  predicted: np.ndarray, sensor: str, unit_id: int,
                                  save_path: Optional[str] = None):
        """
        Plot predicted vs actual sensor values over time.
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f'Sensor Forecasting: {sensor} (Engine {unit_id})', 
                     fontweight='bold', fontsize=14)
        
        # Top: Time series
        axes[0].plot(time_cycles, actual, 'b-', alpha=0.7, label='Actual', linewidth=1)
        axes[0].plot(time_cycles, predicted, 'r--', alpha=0.7, label='Predicted', linewidth=1)
        axes[0].set_ylabel(f'{sensor} Value', fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Actual vs Predicted Values')
        
        # Bottom: Prediction error
        valid_mask = ~np.isnan(predicted)
        errors = np.full_like(actual, np.nan, dtype=float)
        errors[valid_mask] = actual[valid_mask] - predicted[valid_mask]
        
        axes[1].plot(time_cycles, errors, 'g-', alpha=0.7, linewidth=1)
        axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1)
        
        # Add threshold lines
        valid_errors = errors[~np.isnan(errors)]
        if len(valid_errors) > 0:
            std_error = np.std(valid_errors)
            axes[1].axhline(y=3*std_error, color='r', linestyle=':', label='+3σ')
            axes[1].axhline(y=-3*std_error, color='r', linestyle=':', label='-3σ')
            axes[1].legend(loc='upper right')
        
        axes[1].set_xlabel('Time Cycles', fontweight='bold')
        axes[1].set_ylabel('Prediction Error', fontweight='bold')
        axes[1].set_title('Forecasting Error (Actual - Predicted)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Forecasting] Saved plot to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_forecast_accuracy(forecaster: SensorForecaster, df: pd.DataFrame,
                               save_path: Optional[str] = None):
        """
        Bar chart showing forecast accuracy (R²) for each sensor.
        """
        from sklearn.metrics import r2_score
        
        r2_scores = {}
        
        for sensor in forecaster.models.keys():
            all_actual, all_predicted = [], []
            
            for unit_id in df['unit_id'].unique()[:20]:  # Sample 20 engines
                time_cycles, actual, predicted = forecaster.predict_trajectory(df, unit_id, sensor)
                valid_mask = ~np.isnan(predicted)
                
                all_actual.extend(actual[valid_mask])
                all_predicted.extend(predicted[valid_mask])
            
            if len(all_actual) > 0:
                r2_scores[sensor] = r2_score(all_actual, all_predicted)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sensors = list(r2_scores.keys())
        scores = list(r2_scores.values())
        
        bars = ax.bar(range(len(sensors)), scores, edgecolor='black', alpha=0.7)
        
        # Color by performance
        for bar, score in zip(bars, scores):
            bar.set_color('green' if score > 0.8 else 'orange' if score > 0.5 else 'red')
        
        ax.set_xticks(range(len(sensors)))
        ax.set_xticklabels([s.replace('sensor_', 'S') for s in sensors], rotation=45, ha='right')
        ax.set_ylabel('R² Score', fontweight='bold')
        ax.set_xlabel('Sensor', fontweight='bold')
        ax.set_title('Forecasting Accuracy by Sensor', fontweight='bold')
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (R²>0.8)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Forecasting] Saved accuracy plot to {save_path}")
        
        return fig


def demo_forecasting():
    """
    Demonstration of sensor forecasting.
    """
    import os
    from run_evaluation import CMAPSSDataLoader
    
    print("\n" + "="*70)
    print("TIME-SERIES SENSOR FORECASTING DEMO")
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
    
    # Create and fit forecaster
    print("\n[1/4] Training sensor forecasters...")
    forecaster = SensorForecaster(lookback=10)
    forecaster.fit(train_df)
    
    # Predict for sample engine
    print("\n[2/4] Generating predictions for sample engine...")
    sample_engine = train_df['unit_id'].unique()[0]
    sample_sensor = [s for s in forecaster.models.keys()][0]
    
    time_cycles, actual, predicted = forecaster.predict_trajectory(
        train_df, sample_engine, sample_sensor
    )
    
    # Detect faults
    print("\n[3/4] Detecting potential sensor faults...")
    fault_df = forecaster.detect_sensor_faults(train_df)
    print(f"  Found {len(fault_df)} potential fault indicators")
    if len(fault_df) > 0:
        print(fault_df.head(10).to_string(index=False))
    
    # Visualize
    print("\n[4/4] Generating visualizations...")
    os.makedirs('./results', exist_ok=True)
    
    viz = ForecastingVisualizer()
    viz.plot_prediction_vs_actual(
        time_cycles, actual, predicted, sample_sensor, sample_engine,
        f'./results/forecast_{sample_sensor}_engine_{sample_engine}.png'
    )
    viz.plot_forecast_accuracy(forecaster, train_df, './results/forecast_accuracy.png')
    
    print("\n" + "="*70)
    print("[OK] Demo complete! Check ./results/ for visualizations.")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Time-Series Sensor Forecasting")
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--lookback', type=int, default=10, help='Lookback window (default: 10)')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_forecasting()
    else:
        print("Use --demo to run the demonstration.")
        print("Example: python forecasting.py --demo")
