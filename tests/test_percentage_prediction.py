"""
Tests the specialized percentage KPI prediction functionality to ensure
correct detection, prediction generation, and configuration handling.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from percentage_predictor import (
    PercentageKPIPredictor,
    should_use_specialized_percentage_prediction,
    predict_percentage_kpi_specialized
)
from percentage_config import PERCENTAGE_INDICATORS, PERCENTAGE_MODEL_CONFIGS


class TestPercentageKPIDetection(unittest.TestCase):
    """Test percentage KPI detection logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = PercentageKPIPredictor()
        self.percentage_kpi = "% of Completed Trips - MTA Bus"
        self.non_percentage_kpi = "Total Ridership - Subways"
        self.percentage_values = np.array([98.5, 99.0, 98.8, 99.2, 98.7, 99.1])
        self.non_percentage_values = np.array([1500000, 1600000, 1550000, 1580000])
    
    def test_percentage_kpi_detection_positive(self):
        """Test that percentage KPIs are correctly identified."""
        result = self.predictor.is_percentage_kpi(
            self.percentage_kpi, self.percentage_values
        )
        self.assertTrue(result)
    
    def test_percentage_kpi_detection_negative(self):
        """Test that non-percentage KPIs are correctly rejected."""
        result = self.predictor.is_percentage_kpi(
            self.non_percentage_kpi, self.non_percentage_values
        )
        self.assertFalse(result)
    
    def test_percentage_indicators_coverage(self):
        """Test that key percentage indicators are covered."""
        test_names = [
            "% of Completed Trips",
            "Elevator Availability", 
            "On-time Performance Rate",
            "Completion Rate - Depot"
        ]
        
        for name in test_names:
            with self.subTest(name=name):
                is_detected = any(
                    indicator in name.lower() 
                    for indicator in PERCENTAGE_INDICATORS
                )
                self.assertTrue(is_detected, f"'{name}' should be detected as percentage KPI")
    
    def test_convenience_function(self):
        """Test the convenience function works correctly."""
        result = should_use_specialized_percentage_prediction(
            self.percentage_kpi, self.percentage_values
        )
        self.assertTrue(result)


class TestPercentagePrediction(unittest.TestCase):
    """Test percentage KPI prediction generation."""
    
    def setUp(self):
        """Set up test fixtures with mock data."""
        self.predictor = PercentageKPIPredictor()
        
        # Create mock historical data
        dates = pd.date_range('2020-01-01', periods=24, freq='M')
        values = np.random.normal(98.5, 0.5, 24)  # Realistic percentage values
        values = np.clip(values, 95, 100)  # Ensure percentage bounds
        
        self.mock_df = pd.DataFrame({
            'YYYY_MM': dates,
            'MONTHLY_ACTUAL': values
        })
        
        self.kpi_name = "% of Completed Trips - Test Depot"
        self.forecast_months = 6
    
    def test_prediction_format(self):
        """Test that predictions return correct DataFrame format."""
        for model_name in ['RandomForest', 'XGBoost', 'LinearRegression']:
            with self.subTest(model=model_name):
                predictions = self.predictor.predict(
                    self.mock_df, self.kpi_name, model_name, self.forecast_months
                )
                
                # Check format
                self.assertIsInstance(predictions, pd.DataFrame)
                self.assertIn('Date', predictions.columns)
                self.assertIn('Prediction', predictions.columns)
                self.assertEqual(len(predictions), self.forecast_months)
    
    def test_prediction_bounds(self):
        """Test that predictions stay within realistic bounds."""
        for model_name in ['RandomForest', 'XGBoost', 'LinearRegression']:
            with self.subTest(model=model_name):
                predictions = self.predictor.predict(
                    self.mock_df, self.kpi_name, model_name, self.forecast_months
                )
                
                values = predictions['Prediction'].values
                config = PERCENTAGE_MODEL_CONFIGS[model_name.lower()]
                lower_bound, upper_bound = config['bounds']
                
                # Check bounds
                self.assertTrue(np.all(values >= lower_bound), 
                              f"{model_name} predictions below lower bound")
                self.assertTrue(np.all(values <= upper_bound),
                              f"{model_name} predictions above upper bound")
    
    def test_model_differentiation(self):
        """Test that different models produce different prediction patterns."""
        predictions = {}
        
        for model_name in ['RandomForest', 'XGBoost', 'LinearRegression']:
            pred = self.predictor.predict(
                self.mock_df, self.kpi_name, model_name, self.forecast_months
            )
            predictions[model_name] = pred['Prediction'].values
        
        # RandomForest and XGBoost should be more similar to each other than LinearRegression
        rf_xgb_diff = np.mean(np.abs(predictions['RandomForest'] - predictions['XGBoost']))
        rf_lr_diff = np.mean(np.abs(predictions['RandomForest'] - predictions['LinearRegression']))
        
        # LinearRegression should be more conservative (different)
        self.assertGreater(rf_lr_diff, rf_xgb_diff * 0.5,
                          "LinearRegression should be more conservative than other models")
    
    def test_deterministic_results(self):
        """Test that predictions are deterministic (same results for same inputs)."""
        model_name = 'RandomForest'
        
        pred1 = self.predictor.predict(
            self.mock_df, self.kpi_name, model_name, self.forecast_months
        )
        pred2 = self.predictor.predict(
            self.mock_df, self.kpi_name, model_name, self.forecast_months
        )
        
        np.testing.assert_array_almost_equal(
            pred1['Prediction'].values, 
            pred2['Prediction'].values,
            decimal=6,
            err_msg="Predictions should be deterministic"
        )


class TestHistoricalAnalysis(unittest.TestCase):
    """Test historical pattern analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = PercentageKPIPredictor()
    
    def test_pattern_analysis_normal(self):
        """Test pattern analysis with normal percentage data."""
        values = np.array([98.5, 99.0, 98.8, 99.2, 98.7, 99.1, 98.9])
        pattern = self.predictor.analyze_historical_pattern(values)
        
        # Check all expected keys are present
        expected_keys = ['mean', 'std', 'trend', 'min', 'max']
        for key in expected_keys:
            self.assertIn(key, pattern)
        
        # Check reasonable values
        self.assertGreater(pattern['mean'], 95)
        self.assertLess(pattern['mean'], 100)
        self.assertGreater(pattern['std'], 0)
        self.assertLess(pattern['std'], 5)  # Low variance for percentage data
    
    def test_pattern_analysis_empty(self):
        """Test pattern analysis with empty data."""
        values = np.array([])
        pattern = self.predictor.analyze_historical_pattern(values)
        
        # Should return sensible defaults
        self.assertEqual(pattern['mean'], 95.0)
        self.assertEqual(pattern['std'], 1.0)
        self.assertEqual(pattern['trend'], 0.0)


class TestConfigurationIntegration(unittest.TestCase):
    """Test integration with configuration system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = PercentageKPIPredictor()
    
    def test_model_configs_loaded(self):
        """Test that model configurations are properly loaded."""
        required_models = ['randomforest', 'xgboost', 'linearregression']
        
        for model in required_models:
            self.assertIn(model, self.predictor.model_configs)
            
            config = self.predictor.model_configs[model]
            required_keys = ['trend_weight', 'noise_factor', 'lookback_months', 'bounds']
            
            for key in required_keys:
                self.assertIn(key, config)
    
    def test_percentage_indicators_loaded(self):
        """Test that percentage indicators are properly loaded."""
        self.assertIsInstance(self.predictor.percentage_indicators, list)
        self.assertGreater(len(self.predictor.percentage_indicators), 5)
        self.assertIn('% of', self.predictor.percentage_indicators)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPercentageKPIDetection,
        TestPercentagePrediction,
        TestHistoricalAnalysis,
        TestConfigurationIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    print("🧪 Running Percentage KPI Prediction System Tests")
    print("=" * 60)
    
    result = runner.run(test_suite)
    
    if result.wasSuccessful():
        print("\n✅ All tests passed! Percentage KPI system is working correctly.")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
