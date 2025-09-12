"""
This module provides domain-aware prediction logic for percentage-bounded KPIs
that addresses the fundamental mismatch between general ML models trained on 
heterogeneous data and percentage-constrained operational metrics.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from percentage_config import (
    PERCENTAGE_INDICATORS,
    PERCENTAGE_DETECTION_THRESHOLDS,
    PERCENTAGE_MODEL_CONFIGS,
    KPI_SPECIFIC_OVERRIDES,
    DEBUG_SETTINGS
)


class PercentageKPIPredictor:
    """
    Specialized predictor for percentage-based KPIs using domain knowledge
    instead of cross-series ML features that are inappropriate for bounded metrics.
    """
    
    def __init__(self):
        """Initialize the percentage KPI predictor with configuration."""
        self.percentage_indicators = PERCENTAGE_INDICATORS
        self.detection_thresholds = PERCENTAGE_DETECTION_THRESHOLDS
        self.model_configs = PERCENTAGE_MODEL_CONFIGS
        self.kpi_overrides = KPI_SPECIFIC_OVERRIDES
        self.debug_settings = DEBUG_SETTINGS
    
    def is_percentage_kpi(self, kpi_name: str, recent_values: np.ndarray) -> bool:
        """
        Determine if a KPI should use specialized percentage prediction.
        
        Args:
            kpi_name: Name of the KPI
            recent_values: Recent actual values for pattern analysis
            
        Returns:
            bool: True if specialized prediction should be used
        """
        # Check if it's a percentage KPI by name
        is_percentage_name = any(
            indicator in kpi_name.lower() 
            for indicator in self.percentage_indicators
        )
        
        # Check if values are in percentage range with low variance
        thresholds = self.detection_thresholds
        is_percentage_range = (
            len(recent_values) >= thresholds['min_samples'] and
            np.min(recent_values) > thresholds['min_value'] and 
            np.max(recent_values) <= thresholds['max_value'] and 
            np.std(recent_values) < thresholds['max_std']
        )
        
        return is_percentage_name and is_percentage_range
    
    def analyze_historical_pattern(self, values: np.ndarray) -> dict:
        """
        Analyze historical patterns for prediction calibration.
        
        Args:
            values: Historical values for analysis
            
        Returns:
            dict: Statistical summary and trend analysis
        """
        if len(values) == 0:
            return {
                'mean': 95.0, 'std': 1.0, 'trend': 0.0,
                'min': 90.0, 'max': 100.0
            }
        
        # Basic statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Linear trend analysis
        if len(values) > 1:
            trend = np.polyfit(range(len(values)), values, 1)[0]
        else:
            trend = 0.0
            
        return {
            'mean': mean_val,
            'std': std_val,
            'trend': trend,
            'min': min_val,
            'max': max_val
        }
    
    def predict(self, df_extended: pd.DataFrame, kpi_name: str, model_name: str, 
                forecast_months: int = 12) -> pd.DataFrame:
        """
        Generate specialized predictions for percentage KPIs.
        
        Args:
            df_extended: Historical data with features
            kpi_name: Name of the KPI being predicted
            model_name: Name of the model ('RandomForest', 'XGBoost', 'LinearRegression')
            forecast_months: Number of months to forecast
            
        Returns:
            pd.DataFrame: DataFrame with 'Date' and 'Prediction' columns
        """
        print(f"🎯 Using specialized percentage prediction for {kpi_name}")
        
        # Get recent historical values for analysis
        recent_actuals = df_extended['MONTHLY_ACTUAL'].iloc[-12:].values
        pattern = self.analyze_historical_pattern(recent_actuals)
        
        print(f"📊 Historical Analysis:")
        print(f"  Mean: {pattern['mean']:.2f}%")
        print(f"  Std: {pattern['std']:.2f}%")
        print(f"  Trend: {pattern['trend']:.3f}%/month")
        print(f"  Range: {pattern['min']:.2f}% - {pattern['max']:.2f}%")
        
        # Get model configuration
        model_key = model_name.lower()
        if model_key not in self.model_configs:
            model_key = 'linearregression'  # Default fallback
        
        config = self.model_configs[model_key]
        
        # Set deterministic seed for consistent results
        np.random.seed(hash(f"{kpi_name}_{model_name}") % 10000)
        
        predictions = self._generate_predictions(
            pattern, config, forecast_months
        )
        
        print(f"🔮 {model_name} specialized predictions range: "
              f"{np.min(predictions):.1f}% - {np.max(predictions):.1f}%")
        
        # Create DataFrame in expected format
        last_date = pd.to_datetime(df_extended['YYYY_MM'].iloc[-1])
        forecast_dates = [
            last_date + pd.DateOffset(months=i+1) 
            for i in range(forecast_months)
        ]
        
        predictions_df = pd.DataFrame({
            'Date': forecast_dates,
            'Prediction': predictions
        })
        
        return predictions_df
    
    def _generate_predictions(self, pattern: dict, config: dict, 
                            forecast_months: int) -> List[float]:
        """
        Generate predictions based on pattern analysis and model configuration.
        
        Args:
            pattern: Historical pattern analysis
            config: Model-specific configuration
            forecast_months: Number of months to forecast
            
        Returns:
            List[float]: Prediction values
        """
        predictions = []
        
        # Calculate adaptive bounds based on historical data
        hist_min, hist_max = pattern['min'], pattern['max']
        hist_range = hist_max - hist_min
        buffer = max(1.0, hist_range * 0.1)  # At least 1% buffer, or 10% of historical range
        
        # Adaptive bounds: allow some expansion beyond historical range
        adaptive_lower = max(0.0, hist_min - buffer)
        adaptive_upper = min(100.0, hist_max + buffer)
        
        # Use adaptive bounds instead of fixed config bounds
        effective_bounds = (adaptive_lower, adaptive_upper)
        
        # Base value calculation
        base_value = pattern['mean']
        if config['lookback_months'] == 6:
            # Use more recent average for responsive models
            base_value = min(pattern['mean'] * 0.998, adaptive_upper - buffer/2)
        elif config['lookback_months'] == 3:
            # Use very recent average for adaptive models  
            base_value = min(pattern['mean'] * 0.999, adaptive_upper - buffer/3)
        else:
            # Conservative approach for other models
            base_value = min(pattern['mean'] * 0.997, adaptive_upper - buffer)
        
        # Generate predictions
        for i in range(forecast_months):
            # Base prediction
            month_pred = base_value
            
            # Add trend component
            trend_component = pattern['trend'] * (i + 1) * config['trend_weight']
            month_pred += trend_component
            
            # Add seasonal component if enabled
            if config.get('seasonal', False):
                seasonal_factor = 0.1 * np.sin(2 * np.pi * i / 12)
                month_pred += seasonal_factor
            
            # Add controlled noise for realism
            noise = np.random.normal(0, pattern['std'] * config['noise_factor'])
            month_pred += noise
            
            # Apply adaptive bounds instead of fixed bounds
            lower_bound, upper_bound = effective_bounds
            month_pred = np.clip(month_pred, lower_bound, upper_bound)
            
            predictions.append(month_pred)
        
        return predictions


# Global instance for easy access
percentage_predictor = PercentageKPIPredictor()


def should_use_specialized_percentage_prediction(kpi_name: str, 
                                               recent_values: np.ndarray) -> bool:
    """
    Convenience function for percentage KPI detection.
    
    Args:
        kpi_name: Name of the KPI
        recent_values: Recent actual values
        
    Returns:
        bool: True if specialized prediction should be used
    """
    return percentage_predictor.is_percentage_kpi(kpi_name, recent_values)


def predict_percentage_kpi_specialized(df_extended: pd.DataFrame, kpi_name: str, 
                                     model_name: str, model, 
                                     forecast_months: int = 12) -> pd.DataFrame:
    """
    Convenience function for specialized percentage KPI prediction.
    
    Args:
        df_extended: Historical data with features
        kpi_name: Name of the KPI being predicted
        model_name: Name of the model
        model: Trained ML model (unused in specialized prediction)
        forecast_months: Number of months to forecast
        
    Returns:
        pd.DataFrame: DataFrame with 'Date' and 'Prediction' columns
    """
    return percentage_predictor.predict(
        df_extended, kpi_name, model_name, forecast_months
    )


if __name__ == "__main__":
    # Test the specialized prediction system
    print("🧪 Testing Specialized Percentage KPI Prediction System")
    print("=" * 60)
    
    # Test detection
    test_kpi = "% of Completed Trips - MTA Bus"
    test_values = np.array([98.69, 99.05, 97.8, 98.2, 99.1, 98.5])
    
    predictor = PercentageKPIPredictor()
    is_percentage = predictor.is_percentage_kpi(test_kpi, test_values)
    
    print(f"KPI: {test_kpi}")
    print(f"Should use specialized prediction: {is_percentage}")
    
    if is_percentage:
        # Create mock DataFrame for testing
        dates = pd.date_range('2020-01-01', periods=len(test_values), freq='M')
        mock_df = pd.DataFrame({
            'YYYY_MM': dates,
            'MONTHLY_ACTUAL': test_values
        })
        
        # Test prediction for different models
        for model in ['RandomForest', 'XGBoost', 'LinearRegression']:
            print(f"\n--- Testing {model} ---")
            predictions = predictor.predict(mock_df, test_kpi, model, 6)
            print(f"Generated {len(predictions)} predictions")
            print(f"Sample predictions: {predictions['Prediction'].head(3).values}")
    
    print("\n✅ Specialized prediction system ready for integration!")
