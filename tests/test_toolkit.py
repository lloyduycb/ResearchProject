"""
Unit Tests for NASA C-MAPSS RUL Analysis Toolkit
=================================================
Run with: python -m pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRULAnalyzer:
    """Tests for RULAnalyzer class."""
    
    def test_phm_score_perfect_prediction(self):
        """PHM score should be 0 for perfect predictions."""
        from rul_analysis_toolkit import RULAnalyzer
        
        analyzer = RULAnalyzer("test")
        y_true = np.array([100, 50, 25, 10])
        y_pred = np.array([100, 50, 25, 10])  # Perfect
        
        score = analyzer.phm_score(y_true, y_pred)
        assert score == 0.0
    
    def test_phm_score_asymmetric_penalty(self):
        """Late predictions should be penalized more than early."""
        from rul_analysis_toolkit import RULAnalyzer
        
        analyzer = RULAnalyzer("test")
        y_true = np.array([50.0])
        
        # Early prediction (pred < true): penalized less
        early_score = analyzer.phm_score(y_true, np.array([40.0]))
        
        # Late prediction (pred > true): penalized more
        late_score = analyzer.phm_score(y_true, np.array([60.0]))
        
        assert late_score > early_score, "Late predictions should have higher penalty"
    
    def test_compute_standard_metrics(self):
        """Standard metrics should be computed correctly."""
        from rul_analysis_toolkit import RULAnalyzer
        
        analyzer = RULAnalyzer("test")
        y_true = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
        y_pred = np.array([95.0, 82.0, 58.0, 42.0, 18.0])
        
        metrics = analyzer.compute_standard_metrics(y_true, y_pred)
        
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'R2' in metrics
        assert 'PHM_Score' in metrics
        assert metrics['RMSE'] > 0
        assert metrics['MAE'] > 0


class TestStatisticalAnalyzer:
    """Tests for statistical analysis functions."""
    
    def test_cohens_d_large_effect(self):
        """Cohen's d should detect large effect sizes."""
        from statistical_analysis import StatisticalAnalyzer
        
        # Two very different distributions
        group1 = np.array([10, 11, 12, 13, 14])
        group2 = np.array([50, 51, 52, 53, 54])
        
        d = StatisticalAnalyzer.cohens_d(group1, group2)
        
        assert d > 0.8, "Should detect large effect size"
    
    def test_cohens_d_small_effect(self):
        """Cohen's d should detect small effect sizes."""
        from statistical_analysis import StatisticalAnalyzer
        
        # Very similar distributions (small effect)
        group1 = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        group2 = np.array([10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5])
        
        d = StatisticalAnalyzer.cohens_d(group1, group2)
        
        assert d < 0.5, f"Should detect small effect size, got {d}"
    
    def test_interpret_cohens_d(self):
        """Effect size interpretation should be correct."""
        from statistical_analysis import StatisticalAnalyzer
        
        assert StatisticalAnalyzer.interpret_cohens_d(0.1) == "negligible"
        assert StatisticalAnalyzer.interpret_cohens_d(0.3) == "small"
        assert StatisticalAnalyzer.interpret_cohens_d(0.6) == "medium"
        assert StatisticalAnalyzer.interpret_cohens_d(1.0) == "large"


class TestHealthIndexCalculator:
    """Tests for anomaly detection module."""
    
    def test_health_index_range(self):
        """Health index should be between 0 and 100."""
        from anomaly_detection import HealthIndexCalculator
        
        # Create mock training data
        np.random.seed(42)
        n_samples = 200
        train_df = pd.DataFrame({
            'unit_id': np.repeat([1, 2], n_samples // 2),
            'time_cycles': np.tile(range(n_samples // 2), 2),
            'sensor_1': np.random.randn(n_samples),
            'sensor_2': np.random.randn(n_samples),
        })
        
        calc = HealthIndexCalculator(healthy_fraction=0.3)
        calc.fit(train_df)
        
        health = calc.compute_health_index(train_df)
        
        assert health.min() >= 0, "Health index should be >= 0"
        assert health.max() <= 100, "Health index should be <= 100"


class TestOperatingConditionClusterer:
    """Tests for clustering module."""
    
    def test_cluster_labels_range(self):
        """Cluster labels should be in valid range."""
        from condition_clustering import OperatingConditionClusterer
        
        # Mock data with 3 distinct conditions
        np.random.seed(42)
        df = pd.DataFrame({
            'setting_1': np.concatenate([np.random.normal(0, 0.1, 50),
                                         np.random.normal(10, 0.1, 50),
                                         np.random.normal(20, 0.1, 50)]),
            'setting_2': np.concatenate([np.random.normal(0, 0.1, 50),
                                         np.random.normal(5, 0.1, 50),
                                         np.random.normal(10, 0.1, 50)]),
            'setting_3': np.ones(150) * 100,
        })
        
        clusterer = OperatingConditionClusterer(n_clusters=3)
        labels = clusterer.fit_predict(df)
        
        assert labels.min() >= 0
        assert labels.max() < 3
        assert len(np.unique(labels)) == 3


class TestSensorForecaster:
    """Tests for forecasting module."""
    
    def test_sequence_creation(self):
        """Sequence creation should produce correct shapes."""
        from forecasting import SensorForecaster
        
        forecaster = SensorForecaster(lookback=5)
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        X, y = forecaster._create_sequences(data)
        
        assert X.shape == (5, 5), f"Expected (5, 5), got {X.shape}"
        assert y.shape == (5,), f"Expected (5,), got {y.shape}"
        assert np.array_equal(X[0], [1, 2, 3, 4, 5])
        assert y[0] == 6


class TestFeatureAnalyzer:
    """Tests for feature importance module."""
    
    def test_correlation_calculation(self):
        """Correlation values should be in valid range."""
        from feature_importance import FeatureAnalyzer
        
        # Create data with known correlation
        np.random.seed(42)
        n = 100
        rul = np.linspace(100, 0, n)
        df = pd.DataFrame({
            'RUL': rul,
            'sensor_1': rul + np.random.randn(n) * 5,  # Strong correlation
            'sensor_2': np.random.randn(n),  # No correlation
        })
        
        analyzer = FeatureAnalyzer()
        corr_df = analyzer.compute_correlation_with_rul(df)
        
        # sensor_1 should have higher absolute correlation
        s1_corr = corr_df[corr_df['Feature'] == 'sensor_1']['Abs_Correlation'].values[0]
        s2_corr = corr_df[corr_df['Feature'] == 'sensor_2']['Abs_Correlation'].values[0]
        
        assert s1_corr > s2_corr, "sensor_1 should have higher correlation"
        assert s1_corr > 0.9, "sensor_1 correlation should be very high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
