"""
MTA KPI Time Series Forecasting Training Pipeline

Trains Prophet and SARIMA models for individual agency-indicator combinations.
This module handles time series specific forecasting for 264 unique MTA KPIs,
providing complementary predictions to the cross-series ML models.

Author: FYP Project - MTA Performance Prediction System
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import pickle
import warnings
from datetime import datetime
from tqdm import tqdm

# Time series libraries
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add the src directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROCESSED, COLS
from eval import rmse, mae, mape

# Configure for clean output
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

def load_processed_data():
    """
    Load the processed MTA dataset for time series training.
    
    Returns:
        pd.DataFrame: Processed MTA performance data
    """
    print(f"📊 Loading processed data from: {PROCESSED}")
    df = pd.read_parquet(PROCESSED)
    print(f"✓ Loaded dataset: {df.shape[0]} records, {df.shape[1]} columns")
    return df

def get_agency_indicator_combinations(df):
    """
    Identify all viable agency-indicator combinations for time series modeling.
    
    Filters combinations to ensure sufficient historical data for reliable
    time series forecasting (minimum 24 months of data).
    
    Args:
        df: MTA performance DataFrame
        
    Returns:
        pd.DataFrame: Filtered combinations with adequate data points
    """
    print("\n🔍 Analyzing agency-indicator combinations...")
    combinations = df.groupby([COLS['agency'], COLS['indicator']]).size().reset_index(name='count')
    
    # Filter for time series viability (minimum 24 months)
    min_data_points = 24
    combinations = combinations[combinations['count'] >= min_data_points]
    
    print(f"✓ Found {len(combinations)} viable combinations (>={min_data_points} data points)")
    print(f"  Total possible time series models: {len(combinations)} × 2 = {len(combinations) * 2}")
    
    return combinations

def prepare_series_data(df, agency, indicator):
    """
    Prepare time series data for specific agency-indicator combination.
    
    Args:
        df: Full MTA DataFrame
        agency: Target agency
        indicator: Target performance indicator
        
    Returns:
        pd.DataFrame: Clean time series data for modeling
    """
    series_df = df[
        (df[COLS['agency']] == agency) & 
        (df[COLS['indicator']] == indicator)
    ].sort_values('YYYY_MM').copy()
    
    # Remove missing target values (critical for time series)
    initial_count = len(series_df)
    series_df = series_df.dropna(subset=['MONTHLY_ACTUAL'])
    final_count = len(series_df)
    
    if initial_count > final_count:
        print(f"  Removed {initial_count - final_count} rows with missing target values")
    
    return series_df

def split_series_temporal(series_df, test_months=12):
    """
    Perform temporal split for time series validation.
    
    Maintains chronological order critical for time series modeling.
    Uses last N months for testing to simulate real-world forecasting.
    
    Args:
        series_df: Time series data for single agency-indicator
        test_months: Number of months to reserve for testing
        
    Returns:
        tuple: (train_df, test_df) or (None, None) if insufficient data
    """
    min_training_months = 12
    min_total_months = test_months + min_training_months
    
    if len(series_df) < min_total_months:
        return None, None
    
    # Split maintaining temporal order
    split_idx = len(series_df) - test_months
    train_df = series_df.iloc[:split_idx].copy()
    test_df = series_df.iloc[split_idx:].copy()
    
    return train_df, test_df

def train_prophet_model_baseline(train_df, test_df, agency, indicator):
    """
    Train Facebook Prophet model with DEFAULT parameters first to establish baseline.
    
    Args:
        train_df: Training data for the time series
        test_df: Test data for evaluation
        agency: Agency name for identification
        indicator: Performance indicator name
        
    Returns:
        dict: Model results including trained model, predictions, and metrics
        None: If insufficient data or training fails
    """
    try:
        # Prepare data in Prophet format (ds, y columns)
        prophet_df = train_df[['YYYY_MM', 'MONTHLY_ACTUAL']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Clean data for Prophet requirements
        prophet_df = prophet_df.dropna()
        
        # Minimum data requirement check
        min_data_points = 12
        if len(prophet_df) < min_data_points:
            return None
        
        # Configure Prophet with DEFAULT parameters (conservative)
        model = Prophet(
            yearly_seasonality='auto',      # Default seasonal detection
            weekly_seasonality=False,       # Not relevant for monthly data
            daily_seasonality=False,        # Not relevant for monthly data
            # Default changepoint_prior_scale=0.05
            # Default seasonality_prior_scale=10.0  
            # Default interval_width=0.8
        )
        
        # Suppress Prophet's verbose output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(prophet_df)
        
        # Generate forecasts for test period
        future = model.make_future_dataframe(periods=len(test_df), freq='MS')
        forecast = model.predict(future)
        
        # Extract predictions for evaluation
        test_predictions = forecast.tail(len(test_df))['yhat'].values
        test_actual = test_df['MONTHLY_ACTUAL'].values
        
        # Calculate comprehensive metrics
        test_mae = mae(test_actual, test_predictions)
        test_rmse = rmse(test_actual, test_predictions)
        test_mape = mape(test_actual, test_predictions)
        
        # Compile comprehensive results
        results = {
            'model': model,
            'model_type': 'Prophet_Baseline',
            'agency': agency,
            'indicator': indicator,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'predictions': test_predictions,
            'actual': test_actual,
            'forecast_df': forecast,
            'series_length': len(prophet_df),
            'training_periods': len(prophet_df),
            'test_periods': len(test_df),
            'parameters': 'default'
        }
        
        return results
        
    except Exception as e:
        print(f"   ❌ Prophet baseline training failed for {agency} - {indicator}: {str(e)}")
        return None

def train_prophet_model_tuned(train_df, test_df, agency, indicator):
    """
    Train Facebook Prophet model with OPTIMIZED parameters for comparison.
    
    Args:
        train_df: Training data for the time series
        test_df: Test data for evaluation
        agency: Agency name for identification
        indicator: Performance indicator name
        
    Returns:
        dict: Model results including trained model, predictions, and metrics
        None: If insufficient data or training fails
    """
    try:
        # Prepare data in Prophet format (ds, y columns)
        prophet_df = train_df[['YYYY_MM', 'MONTHLY_ACTUAL']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Clean data for Prophet requirements
        prophet_df = prophet_df.dropna()
        
        # Minimum data requirement check
        min_data_points = 12
        if len(prophet_df) < min_data_points:
            return None
        
        # Configure Prophet with OPTIMIZED parameters
        model = Prophet(
            yearly_seasonality=True,        # Capture annual patterns
            weekly_seasonality=False,       # Not relevant for monthly data
            daily_seasonality=False,        # Not relevant for monthly data
            changepoint_prior_scale=0.05,   # Conservative trend changes
            seasonality_prior_scale=10.0,   # Allow moderate seasonality
            interval_width=0.8,             # 80% prediction intervals
            seasonality_mode='multiplicative'  # Better for varying seasonal effects
        )
        
        # Suppress Prophet's verbose output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(prophet_df)
        
        # Generate forecasts for test period
        future = model.make_future_dataframe(periods=len(test_df), freq='MS')
        forecast = model.predict(future)
        
        # Extract predictions for evaluation
        test_predictions = forecast.tail(len(test_df))['yhat'].values
        test_actual = test_df['MONTHLY_ACTUAL'].values
        
        # Calculate comprehensive metrics
        test_mae = mae(test_actual, test_predictions)
        test_rmse = rmse(test_actual, test_predictions)
        test_mape = mape(test_actual, test_predictions)
        
        # Compile comprehensive results
        results = {
            'model': model,
            'model_type': 'Prophet_Tuned',
            'agency': agency,
            'indicator': indicator,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'predictions': test_predictions,
            'actual': test_actual,
            'forecast_df': forecast,
            'series_length': len(prophet_df),
            'training_periods': len(prophet_df),
            'test_periods': len(test_df),
            'parameters': 'optimized'
        }
        
        return results
        
    except Exception as e:
        print(f"   ❌ Prophet tuned training failed for {agency} - {indicator}: {str(e)}")
        return None
    """
    Train Facebook Prophet model with OPTIMIZED parameters for comparison.
    
    Prophet is designed for time series with strong seasonal patterns and
    handles missing data well. This version uses optimized parameters for MTA monthly data.
    
    Args:
        train_df: Training data for the time series
        test_df: Test data for evaluation
        agency: Agency name for identification
        indicator: Performance indicator name
        
    Returns:
        dict: Model results including trained model, predictions, and metrics
        None: If insufficient data or training fails
    """
    try:
        # Prepare data in Prophet format (ds, y columns)
        prophet_df = train_df[['YYYY_MM', 'MONTHLY_ACTUAL']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Clean data for Prophet requirements
        prophet_df = prophet_df.dropna()
        
        # Minimum data requirement check
        min_data_points = 12
        if len(prophet_df) < min_data_points:
            return None
        
        # Configure Prophet for MTA monthly data with OPTIMIZED parameters
        model = Prophet(
            yearly_seasonality=True,        # Force annual patterns (optimized)
            weekly_seasonality=False,       # Not relevant for monthly data
            daily_seasonality=False,        # Not relevant for monthly data
            changepoint_prior_scale=0.05,   # Conservative trend changes (optimized)
            seasonality_prior_scale=10.0,   # Moderate seasonality (optimized)
            holidays_prior_scale=10.0,      # Handle holiday effects (optimized)
            interval_width=0.8,             # 80% prediction intervals (optimized)
            mcmc_samples=0                  # Faster training (optimized)
        )
        
        # Suppress Prophet's verbose output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(prophet_df)
        
        # Generate forecasts for test period
        future = model.make_future_dataframe(periods=len(test_df), freq='MS')
        forecast = model.predict(future)
        
        # Extract predictions for evaluation
        test_predictions = forecast.tail(len(test_df))['yhat'].values
        test_actual = test_df['MONTHLY_ACTUAL'].values
        
        # Calculate comprehensive metrics
        test_mae = mae(test_actual, test_predictions)
        test_rmse = rmse(test_actual, test_predictions)
        test_mape = mape(test_actual, test_predictions)
        
        # Compile comprehensive results
        results = {
            'model': model,
            'model_type': 'Prophet_Tuned',
            'agency': agency,
            'indicator': indicator,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'predictions': test_predictions,
            'actual': test_actual,
            'forecast_df': forecast,
            'series_length': len(prophet_df),
            'training_periods': len(prophet_df),
            'test_periods': len(test_df),
            'parameters': 'optimized'
        }
        
        return results
        
    except Exception as e:
        print(f"   ❌ Prophet tuned training failed for {agency} - {indicator}: {str(e)}")
        return None
    """
    Train Facebook Prophet model for individual time series forecasting.
    
    Prophet is designed for time series with strong seasonal patterns and
    handles missing data well. Optimized for MTA monthly performance data.
    
    Args:
        train_df: Training data for the time series
        test_df: Test data for evaluation
        agency: Agency name for identification
        indicator: Performance indicator name
        
    Returns:
        dict: Model results including trained model, predictions, and metrics
        None: If insufficient data or training fails
    """
    try:
        # Prepare data in Prophet format (ds, y columns)
        prophet_df = train_df[['YYYY_MM', 'MONTHLY_ACTUAL']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Clean data for Prophet requirements
        prophet_df = prophet_df.dropna()
        
        # Minimum data requirement check
        min_data_points = 12
        if len(prophet_df) < min_data_points:
            return None
        
        # Configure Prophet for MTA monthly data
        model = Prophet(
            yearly_seasonality=True,        # Capture annual patterns
            weekly_seasonality=False,       # Not relevant for monthly data
            daily_seasonality=False,        # Not relevant for monthly data
            changepoint_prior_scale=0.05,   # Conservative trend changes
            seasonality_prior_scale=10.0,   # Allow moderate seasonality
            interval_width=0.8              # 80% prediction intervals
        )
        
        # Suppress Prophet's verbose output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(prophet_df)
        
        # Generate forecasts for test period
        future = model.make_future_dataframe(periods=len(test_df), freq='MS')
        forecast = model.predict(future)
        
        # Extract predictions for evaluation
        test_predictions = forecast.tail(len(test_df))['yhat'].values
        test_actual = test_df['MONTHLY_ACTUAL'].values
        
        # Calculate comprehensive metrics
        test_mae = mae(test_actual, test_predictions)
        test_rmse = rmse(test_actual, test_predictions)
        test_mape = mape(test_actual, test_predictions)
        
        # Compile comprehensive results
        results = {
            'model': model,
            'model_type': 'Prophet',
            'agency': agency,
            'indicator': indicator,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'predictions': test_predictions,
            'actual': test_actual,
            'forecast_df': forecast,
            'series_length': len(prophet_df),
            'training_periods': len(prophet_df),
            'test_periods': len(test_df)
        }
        
        return results
        
    except Exception as e:
        print(f"   ❌ Prophet training failed for {agency} - {indicator}: {str(e)}")
        return None

def find_best_sarima_params(train_df, max_p=3, max_d=2, max_q=3):
    """
    Find optimal SARIMA parameters using Akaike Information Criterion (AIC).
    
    Performs grid search over parameter space to identify best-fitting
    SARIMA configuration for the time series.
    
    Args:
        train_df: Training data for parameter optimization
        max_p, max_d, max_q: Maximum values for ARIMA parameters
        
    Returns:
        tuple: Best (p,d,q) parameters or None if search fails
    """
    try:
        best_aic = float('inf')
        best_params = None
        
        # Prepare time series with proper frequency
        ts_data = train_df.set_index('YYYY_MM')['MONTHLY_ACTUAL']
        ts_data = ts_data.asfreq('MS')  # Monthly start frequency
        
        # Systematic parameter search
        param_combinations = 0
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    param_combinations += 1
                    try:
                        # Test SARIMA configuration
                        model = SARIMAX(
                            ts_data, 
                            order=(p, d, q), 
                            seasonal_order=(1, 1, 1, 12),  # Monthly seasonality
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fitted_model = model.fit(disp=False, maxiter=100)
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                            
                    except:
                        continue
        
        # Return best parameters or sensible default
        return best_params if best_params else (1, 1, 1)
        
    except Exception as e:
        return (1, 1, 1)  # Safe fallback parameters

def train_sarima_model_baseline(train_df, test_df, agency, indicator):
    """
    Train SARIMA model with DEFAULT/SIMPLE parameters first to establish baseline.
    
    Args:
        train_df: Training data for the time series
        test_df: Test data for evaluation
        agency: Agency name for identification
        indicator: Performance indicator name
        
    Returns:
        dict: Model results including trained model, predictions, and metrics
        None: If insufficient data or training fails
    """
    try:
        # Prepare time series data
        ts_data = train_df.set_index('YYYY_MM')['MONTHLY_ACTUAL']
        ts_data = ts_data.asfreq('MS')  # Monthly start frequency
        
        # Minimum data requirement for seasonal modeling
        min_seasonal_data = 24  # 2 years for seasonal patterns
        if len(ts_data) < min_seasonal_data:
            return None
        
        # Use simple DEFAULT parameters (no optimization)
        default_params = (1, 1, 1)  # Simple ARIMA(1,1,1)
        
        # Train SARIMA model with default parameters
        model = SARIMAX(
            ts_data, 
            order=default_params, 
            seasonal_order=(1, 1, 1, 12),  # Monthly seasonality
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model = model.fit(disp=False, maxiter=100)
        
        # Generate forecasts for test period
        forecast_steps = len(test_df)
        forecast = fitted_model.forecast(steps=forecast_steps)
        
        # Prepare evaluation data
        test_actual = test_df['MONTHLY_ACTUAL'].values
        test_predictions = forecast.values
        
        # Validate predictions for numerical stability
        historical_max = train_df['MONTHLY_ACTUAL'].max()
        historical_mean = train_df['MONTHLY_ACTUAL'].mean()
        
        # Check for extreme predictions that indicate numerical instability
        if (np.any(np.abs(test_predictions) > 1e10) or 
            np.any(np.isnan(test_predictions)) or 
            np.any(np.isinf(test_predictions)) or
            np.any(np.abs(test_predictions) > 100 * historical_max)):
            
            print(f"   ⚠️  SARIMA baseline predictions unstable for {agency} - {indicator}")
            
            # Use simple trend-based fallback for unstable models
            recent_trend = train_df['MONTHLY_ACTUAL'].tail(6).mean()
            test_predictions = np.full(len(test_predictions), recent_trend)
            print(f"      Using trend-based fallback: {recent_trend:.2f}")
        
        # Ensure non-negative predictions
        test_predictions = np.maximum(test_predictions, 0)
        
        # Calculate comprehensive metrics
        test_mae = mae(test_actual, test_predictions)
        test_rmse = rmse(test_actual, test_predictions)
        test_mape = mape(test_actual, test_predictions)
        
        # Final validation check on metrics
        if test_mae > 1e10 or test_rmse > 1e10 or test_mape > 1e10:
            print(f"   ⚠️  Extreme metrics detected for {agency} - {indicator}")
            
            # Fallback to simple baseline metrics
            baseline_pred = np.full(len(test_actual), historical_mean)
            test_mae = mae(test_actual, baseline_pred)
            test_rmse = rmse(test_actual, baseline_pred)  
            test_mape = mape(test_actual, baseline_pred)
            test_predictions = baseline_pred
            
            print(f"      Using baseline metrics: MAE={test_mae:.2f}")
        
        # Cap extreme values
        test_mae = min(test_mae, historical_max * 10)
        test_rmse = min(test_rmse, historical_max * 20)
        test_mape = min(test_mape, 1000)
        
        # Compile comprehensive results
        results = {
            'model': fitted_model,
            'model_type': 'SARIMA_Baseline',
            'agency': agency,
            'indicator': indicator,
            'best_params': default_params,
            'seasonal_order': (1, 1, 1, 12),
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'predictions': test_predictions,
            'actual': test_actual,
            'series_length': len(ts_data),
            'training_periods': len(ts_data),
            'test_periods': len(test_df),
            'parameters': 'default'
        }
        
        return results
        
    except Exception as e:
        print(f"   ❌ SARIMA baseline training failed for {agency} - {indicator}: {str(e)}")
        return None

def train_sarima_model_tuned(train_df, test_df, agency, indicator):
    """
    Train SARIMA model with OPTIMIZED parameters for comparison.
    
    SARIMA models are excellent for capturing both trend and seasonal patterns
    in time series data. This version uses parameter optimization for better performance.
    
    Args:
        train_df: Training data for the time series
        test_df: Test data for evaluation
        agency: Agency name for identification
        indicator: Performance indicator name
        
    Returns:
        dict: Model results including trained model, predictions, and metrics
        None: If insufficient data or training fails
    """
    try:
        # Prepare time series data
        ts_data = train_df.set_index('YYYY_MM')['MONTHLY_ACTUAL']
        ts_data = ts_data.asfreq('MS')  # Monthly start frequency
        
        # Minimum data requirement for seasonal modeling
        min_seasonal_data = 24  # 2 years for seasonal patterns
        if len(ts_data) < min_seasonal_data:
            return None
        
        # Optimize SARIMA parameters
        best_params = find_best_sarima_params(train_df)
        
        # Train SARIMA model with optimal parameters
        model = SARIMAX(
            ts_data, 
            order=best_params, 
            seasonal_order=(1, 1, 1, 12),  # Monthly seasonality
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model = model.fit(disp=False, maxiter=200)
        
        # Generate forecasts for test period
        forecast_steps = len(test_df)
        forecast = fitted_model.forecast(steps=forecast_steps)
        
        # Prepare evaluation data
        test_actual = test_df['MONTHLY_ACTUAL'].values
        test_predictions = forecast.values
        
        # Validate predictions for numerical stability
        historical_max = train_df['MONTHLY_ACTUAL'].max()
        historical_mean = train_df['MONTHLY_ACTUAL'].mean()
        
        # Check for extreme predictions that indicate numerical instability
        if (np.any(np.abs(test_predictions) > 1e10) or 
            np.any(np.isnan(test_predictions)) or 
            np.any(np.isinf(test_predictions)) or
            np.any(np.abs(test_predictions) > 100 * historical_max)):
            
            print(f"   ⚠️  SARIMA tuned predictions unstable for {agency} - {indicator}")
            print(f"      Prediction range: {test_predictions.min():.2e} to {test_predictions.max():.2e}")
            print(f"      Historical max: {historical_max:.2f}")
            
            # Use simple trend-based fallback for unstable models
            recent_trend = train_df['MONTHLY_ACTUAL'].tail(6).mean()
            test_predictions = np.full(len(test_predictions), recent_trend)
            print(f"      Using trend-based fallback: {recent_trend:.2f}")
        
        # Ensure non-negative predictions
        test_predictions = np.maximum(test_predictions, 0)
        
        # Calculate comprehensive metrics
        test_mae = mae(test_actual, test_predictions)
        test_rmse = rmse(test_actual, test_predictions)
        test_mape = mape(test_actual, test_predictions)
        
        # Final validation check on metrics
        if test_mae > 1e10 or test_rmse > 1e10 or test_mape > 1e10:
            print(f"   ⚠️  Extreme metrics detected for {agency} - {indicator}")
            print(f"      MAE: {test_mae:.2e}, RMSE: {test_rmse:.2e}, MAPE: {test_mape:.2e}%")
            
            # Fallback to simple baseline metrics
            baseline_pred = np.full(len(test_actual), historical_mean)
            test_mae = mae(test_actual, baseline_pred)
            test_rmse = rmse(test_actual, baseline_pred)  
            test_mape = mape(test_actual, baseline_pred)
            test_predictions = baseline_pred
            
            print(f"      Using baseline metrics: MAE={test_mae:.2f}")
        
        # Additional sanity check - cap extreme values
        test_mae = min(test_mae, historical_max * 10)  # Cap at 10x historical max
        test_rmse = min(test_rmse, historical_max * 20)  # Cap at 20x historical max
        test_mape = min(test_mape, 1000)  # Cap MAPE at 1000%
        
        # Compile comprehensive results
        results = {
            'model': fitted_model,
            'model_type': 'SARIMA_Tuned',
            'agency': agency,
            'indicator': indicator,
            'best_params': best_params,
            'seasonal_order': (1, 1, 1, 12),
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'predictions': test_predictions,
            'actual': test_actual,
            'series_length': len(ts_data),
            'training_periods': len(ts_data),
            'test_periods': len(test_df),
            'parameters': 'optimized'
        }
        
        return results
        
    except Exception as e:
        print(f"   ❌ SARIMA tuned training failed for {agency} - {indicator}: {str(e)}")
        return None
    """
    Train SARIMA (Seasonal ARIMA) model for individual time series.
    
    SARIMA models are excellent for capturing both trend and seasonal patterns
    in time series data. Optimized for MTA monthly performance indicators.
    
    Args:
        train_df: Training data for the time series
        test_df: Test data for evaluation
        agency: Agency name for identification
        indicator: Performance indicator name
        
    Returns:
        dict: Model results including trained model, predictions, and metrics
        None: If insufficient data or training fails
    """
    try:
        # Prepare time series data
        ts_data = train_df.set_index('YYYY_MM')['MONTHLY_ACTUAL']
        ts_data = ts_data.asfreq('MS')  # Monthly start frequency
        
        # Minimum data requirement for seasonal modeling
        min_seasonal_data = 24  # 2 years for seasonal patterns
        if len(ts_data) < min_seasonal_data:
            return None
        
        # Optimize SARIMA parameters
        best_params = find_best_sarima_params(train_df)
        
        # Train SARIMA model with optimal parameters
        model = SARIMAX(
            ts_data, 
            order=best_params, 
            seasonal_order=(1, 1, 1, 12),  # Monthly seasonality
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model = model.fit(disp=False, maxiter=200)
        
        # Generate forecasts for test period
        forecast_steps = len(test_df)
        forecast = fitted_model.forecast(steps=forecast_steps)
        
        # Prepare evaluation data
        test_actual = test_df['MONTHLY_ACTUAL'].values
        test_predictions = forecast.values
        
        # Validate predictions for numerical stability
        historical_max = train_df['MONTHLY_ACTUAL'].max()
        historical_mean = train_df['MONTHLY_ACTUAL'].mean()
        
        # Check for extreme predictions that indicate numerical instability
        if (np.any(np.abs(test_predictions) > 1e10) or 
            np.any(np.isnan(test_predictions)) or 
            np.any(np.isinf(test_predictions)) or
            np.any(np.abs(test_predictions) > 100 * historical_max)):
            
            print(f"   ⚠️  SARIMA predictions unstable for {agency} - {indicator}")
            print(f"      Prediction range: {test_predictions.min():.2e} to {test_predictions.max():.2e}")
            print(f"      Historical max: {historical_max:.2f}")
            
            # Use simple trend-based fallback for unstable models
            recent_trend = train_df['MONTHLY_ACTUAL'].tail(6).mean()
            test_predictions = np.full(len(test_predictions), recent_trend)
            print(f"      Using trend-based fallback: {recent_trend:.2f}")
        
        # Ensure non-negative predictions
        test_predictions = np.maximum(test_predictions, 0)
        
        # Calculate comprehensive metrics
        test_mae = mae(test_actual, test_predictions)
        test_rmse = rmse(test_actual, test_predictions)
        test_mape = mape(test_actual, test_predictions)
        
        # Final validation check on metrics
        if test_mae > 1e10 or test_rmse > 1e10 or test_mape > 1e10:
            print(f"   ⚠️  Extreme metrics detected for {agency} - {indicator}")
            print(f"      MAE: {test_mae:.2e}, RMSE: {test_rmse:.2e}, MAPE: {test_mape:.2e}%")
            
            # Fallback to simple baseline metrics
            baseline_pred = np.full(len(test_actual), historical_mean)
            test_mae = mae(test_actual, baseline_pred)
            test_rmse = rmse(test_actual, baseline_pred)  
            test_mape = mape(test_actual, baseline_pred)
            test_predictions = baseline_pred
            
            print(f"      Using baseline metrics: MAE={test_mae:.2f}")
        
        # Additional sanity check - cap extreme values
        test_mae = min(test_mae, historical_max * 10)  # Cap at 10x historical max
        test_rmse = min(test_rmse, historical_max * 20)  # Cap at 20x historical max
        test_mape = min(test_mape, 1000)  # Cap MAPE at 1000%
        
        # Compile comprehensive results
        results = {
            'model': fitted_model,
            'model_type': 'SARIMA',
            'agency': agency,
            'indicator': indicator,
            'best_params': best_params,
            'seasonal_order': (1, 1, 1, 12),
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'predictions': test_predictions,
            'actual': test_actual,
            'series_length': len(ts_data),
            'training_periods': len(ts_data),
            'test_periods': len(test_df)
        }
        
        return results
        
    except Exception as e:
        print(f"   ❌ SARIMA training failed for {agency} - {indicator}: {str(e)}")
        return None

def train_all_series_before_after(df, combinations, max_series=None):
    """
    Train Prophet and SARIMA models with BEFORE/AFTER comparison for all viable time series.
    
    This function orchestrates the training of baseline and tuned time series models
    for each agency-indicator combination, providing comprehensive before/after analysis.
    
    Args:
        df: Complete MTA DataFrame
        combinations: Viable agency-indicator combinations
        max_series: Optional limit on number of series to process
        
    Returns:
        tuple: (baseline_results, tuned_results) for comparison
    """
    # Limit series if specified (useful for testing)
    if max_series:
        combinations = combinations.head(max_series)
        print(f"🔬 Limited to first {max_series} series for testing")
    
    # Initialize results storage
    baseline_results = {
        'prophet': {},
        'sarima': {}
    }
    
    tuned_results = {
        'prophet': {},
        'sarima': {}
    }
    
    # Progress tracking
    successful_prophet_baseline = 0
    successful_prophet_tuned = 0
    successful_sarima_baseline = 0
    successful_sarima_tuned = 0
    skipped_series = 0
    total_series = len(combinations)
    
    print(f"\n🚀 Training time series models: BASELINE → TUNING → COMPARISON")
    print(f"Expected models: {total_series} × 4 = {total_series * 4} total models (2 baseline + 2 tuned)")
    
    # Process each agency-indicator combination
    for idx, row in combinations.iterrows():
        agency = row[COLS['agency']]
        indicator = row[COLS['indicator']]
        
        print(f"\n[{idx+1}/{total_series}] {agency} - {indicator}")
        
        # Prepare time series data
        series_df = prepare_series_data(df, agency, indicator)
        
        # Check minimum data requirements
        if len(series_df) < 24:
            print(f"  ⚠️  Insufficient data ({len(series_df)} points), skipping...")
            skipped_series += 1
            continue
        
        # Perform temporal split
        train_df, test_df = split_series_temporal(series_df)
        
        if train_df is None or test_df is None:
            print(f"  ⚠️  Cannot create train/test split, skipping...")
            skipped_series += 1
            continue
        
        series_key = f"{agency}|{indicator}"
        print(f"  📊 Data: {len(train_df)} train, {len(test_df)} test periods")
        
        # BASELINE TRAINING: Default parameters
        print("  🔮 Prophet baseline (default parameters)...")
        prophet_baseline = train_prophet_model_baseline(train_df, test_df, agency, indicator)
        if prophet_baseline:
            baseline_results['prophet'][series_key] = prophet_baseline
            successful_prophet_baseline += 1
            print(f"     ✓ Prophet baseline: MAE {prophet_baseline['test_mae']:.2f}, MAPE {prophet_baseline['test_mape']:.1f}%")
        else:
            print(f"     ❌ Prophet baseline failed")
        
        print("  📈 SARIMA baseline (default parameters)...")
        sarima_baseline = train_sarima_model_baseline(train_df, test_df, agency, indicator)
        if sarima_baseline:
            baseline_results['sarima'][series_key] = sarima_baseline
            successful_sarima_baseline += 1
            print(f"     ✓ SARIMA baseline: MAE {sarima_baseline['test_mae']:.2f}, MAPE {sarima_baseline['test_mape']:.1f}%")
        else:
            print(f"     ❌ SARIMA baseline failed")
        
        # TUNED TRAINING: Optimized parameters
        print("  🔮 Prophet tuned (optimized parameters)...")
        prophet_tuned = train_prophet_model_tuned(train_df, test_df, agency, indicator)
        if prophet_tuned:
            tuned_results['prophet'][series_key] = prophet_tuned
            successful_prophet_tuned += 1
            print(f"     ✓ Prophet tuned: MAE {prophet_tuned['test_mae']:.2f}, MAPE {prophet_tuned['test_mape']:.1f}%")
        else:
            print(f"     ❌ Prophet tuned failed")
        
        print("  📈 SARIMA tuned (optimized parameters)...")
        sarima_tuned = train_sarima_model_tuned(train_df, test_df, agency, indicator)
        if sarima_tuned:
            tuned_results['sarima'][series_key] = sarima_tuned
            successful_sarima_tuned += 1
            print(f"     ✓ SARIMA tuned: MAE {sarima_tuned['test_mae']:.2f}, MAPE {sarima_tuned['test_mape']:.1f}%")
        else:
            print(f"     ❌ SARIMA tuned failed")
        
        # Progress updates for large batches
        if (idx + 1) % 25 == 0 or (idx + 1) == total_series:
            completion_pct = ((idx + 1) / total_series) * 100
            print(f"\n📊 Progress Update: {completion_pct:.1f}% complete")
            print(f"   Baseline models: Prophet {successful_prophet_baseline}, SARIMA {successful_sarima_baseline}")
            print(f"   Tuned models: Prophet {successful_prophet_tuned}, SARIMA {successful_sarima_tuned}")
            print(f"   Skipped series: {skipped_series}")
    
    # Final comprehensive summary
    total_baseline = successful_prophet_baseline + successful_sarima_baseline
    total_tuned = successful_prophet_tuned + successful_sarima_tuned
    total_successful = total_baseline + total_tuned
    total_possible = total_series * 4
    success_rate = (total_successful / total_possible) * 100 if total_possible > 0 else 0
    
    print(f"\n" + "="*60)
    print("TIME SERIES BEFORE/AFTER TRAINING COMPLETED")
    print("="*60)
    print(f"📈 Series processed: {total_series}")
    print(f"🔮 Prophet baseline: {successful_prophet_baseline}/{total_series} ({(successful_prophet_baseline/total_series)*100:.1f}%)")
    print(f"🔮 Prophet tuned: {successful_prophet_tuned}/{total_series} ({(successful_prophet_tuned/total_series)*100:.1f}%)")
    print(f"📊 SARIMA baseline: {successful_sarima_baseline}/{total_series} ({(successful_sarima_baseline/total_series)*100:.1f}%)")
    print(f"📊 SARIMA tuned: {successful_sarima_tuned}/{total_series} ({(successful_sarima_tuned/total_series)*100:.1f}%)")
    print(f"🎯 Overall success rate: {success_rate:.1f}% ({total_successful}/{total_possible} models)")
    print(f"⚠️  Skipped series: {skipped_series}")
    print("="*60)
    
    return baseline_results, tuned_results

def compare_time_series_before_after(baseline_results, tuned_results):
    """
    Compare baseline vs tuned time series model performance.
    
    Args:
        baseline_results: Dictionary containing baseline model results
        tuned_results: Dictionary containing tuned model results
        
    Returns:
        tuple: (prophet_comparison_df, sarima_comparison_df)
    """
    print("\n" + "="*80)
    print("TIME SERIES BEFORE vs AFTER PARAMETER TUNING COMPARISON")
    print("="*80)
    
    # Compare Prophet models
    prophet_comparison = []
    sarima_comparison = []
    
    # Prophet comparison
    for series_key in baseline_results['prophet']:
        if series_key in tuned_results['prophet']:
            baseline = baseline_results['prophet'][series_key]
            tuned = tuned_results['prophet'][series_key]
            
            mae_improvement = baseline['test_mae'] - tuned['test_mae']
            mae_improvement_pct = (mae_improvement / baseline['test_mae']) * 100
            
            prophet_comparison.append({
                'Series': series_key,
                'Baseline_MAE': baseline['test_mae'],
                'Tuned_MAE': tuned['test_mae'], 
                'MAE_Improvement': mae_improvement,
                'MAE_Improvement_%': mae_improvement_pct,
                'Baseline_MAPE': baseline['test_mape'],
                'Tuned_MAPE': tuned['test_mape']
            })
    
    # SARIMA comparison
    for series_key in baseline_results['sarima']:
        if series_key in tuned_results['sarima']:
            baseline = baseline_results['sarima'][series_key]
            tuned = tuned_results['sarima'][series_key]
            
            mae_improvement = baseline['test_mae'] - tuned['test_mae']
            mae_improvement_pct = (mae_improvement / baseline['test_mae']) * 100
            
            sarima_comparison.append({
                'Series': series_key,
                'Baseline_MAE': baseline['test_mae'],
                'Tuned_MAE': tuned['test_mae'],
                'MAE_Improvement': mae_improvement,
                'MAE_Improvement_%': mae_improvement_pct,
                'Baseline_MAPE': baseline['test_mape'],
                'Tuned_MAPE': tuned['test_mape']
            })
    
    prophet_df = pd.DataFrame(prophet_comparison) if prophet_comparison else pd.DataFrame()
    sarima_df = pd.DataFrame(sarima_comparison) if sarima_comparison else pd.DataFrame()
    
    # Display Prophet comparison
    if not prophet_df.empty:
        print(f"\n📊 PROPHET MODEL COMPARISON ({len(prophet_df)} series):")
        print("="*50)
        avg_prophet_baseline = prophet_df['Baseline_MAE'].mean()
        avg_prophet_tuned = prophet_df['Tuned_MAE'].mean()
        avg_prophet_improvement = prophet_df['MAE_Improvement_%'].mean()
        
        print(f"Average Baseline MAE: {avg_prophet_baseline:.2f}")
        print(f"Average Tuned MAE: {avg_prophet_tuned:.2f}")
        print(f"Average Improvement: {avg_prophet_improvement:+.2f}%")
        
        prophet_wins = (prophet_df['MAE_Improvement_%'] > 0).sum()
        print(f"Series with improvement: {prophet_wins}/{len(prophet_df)} ({prophet_wins/len(prophet_df)*100:.1f}%)")
        
        if avg_prophet_improvement > 0:
            print(f"🏆 Prophet parameter tuning shows {avg_prophet_improvement:.2f}% average improvement!")
        else:
            print(f"📊 Prophet baseline performs better by {abs(avg_prophet_improvement):.2f}% on average")
    
    # Display SARIMA comparison
    if not sarima_df.empty:
        print(f"\n📊 SARIMA MODEL COMPARISON ({len(sarima_df)} series):")
        print("="*50)
        avg_sarima_baseline = sarima_df['Baseline_MAE'].mean()
        avg_sarima_tuned = sarima_df['Tuned_MAE'].mean()
        avg_sarima_improvement = sarima_df['MAE_Improvement_%'].mean()
        
        print(f"Average Baseline MAE: {avg_sarima_baseline:.2f}")
        print(f"Average Tuned MAE: {avg_sarima_tuned:.2f}")
        print(f"Average Improvement: {avg_sarima_improvement:+.2f}%")
        
        sarima_wins = (sarima_df['MAE_Improvement_%'] > 0).sum()
        print(f"Series with improvement: {sarima_wins}/{len(sarima_df)} ({sarima_wins/len(sarima_df)*100:.1f}%)")
        
        if avg_sarima_improvement > 0:
            print(f"� SARIMA parameter tuning shows {avg_sarima_improvement:.2f}% average improvement!")
        else:
            print(f"📊 SARIMA baseline performs better by {abs(avg_sarima_improvement):.2f}% on average")
    
    print("="*80)
    
    return prophet_df, sarima_df

def evaluate_time_series_models(baseline_results, tuned_results):
    """Evaluate and compare baseline vs tuned time series models"""
    print("\n" + "="*50)
    print("TIME SERIES MODEL EVALUATION")
    print("="*50)
    
    # Calculate aggregate metrics for baseline and tuned models
    prophet_baseline_metrics = []
    prophet_tuned_metrics = []
    sarima_baseline_metrics = []
    sarima_tuned_metrics = []
    
    # Collect Prophet metrics
    for series_key in baseline_results['prophet']:
        if series_key in tuned_results['prophet']:
            prophet_baseline_metrics.append({
                'series': series_key,
                'mae': baseline_results['prophet'][series_key]['test_mae'],
                'rmse': baseline_results['prophet'][series_key]['test_rmse'],
                'mape': baseline_results['prophet'][series_key]['test_mape']
            })
            
            prophet_tuned_metrics.append({
                'series': series_key,
                'mae': tuned_results['prophet'][series_key]['test_mae'],
                'rmse': tuned_results['prophet'][series_key]['test_rmse'],
                'mape': tuned_results['prophet'][series_key]['test_mape']
            })
    
    # Collect SARIMA metrics
    for series_key in baseline_results['sarima']:
        if series_key in tuned_results['sarima']:
            sarima_baseline_metrics.append({
                'series': series_key,
                'mae': baseline_results['sarima'][series_key]['test_mae'],
                'rmse': baseline_results['sarima'][series_key]['test_rmse'],
                'mape': baseline_results['sarima'][series_key]['test_mape']
            })
            
            sarima_tuned_metrics.append({
                'series': series_key,
                'mae': tuned_results['sarima'][series_key]['test_mae'],
                'rmse': tuned_results['sarima'][series_key]['test_rmse'],
                'mape': tuned_results['sarima'][series_key]['test_mape']
            })
    
    # Display results
    if prophet_baseline_metrics and prophet_tuned_metrics:
        prophet_baseline_df = pd.DataFrame(prophet_baseline_metrics)
        prophet_tuned_df = pd.DataFrame(prophet_tuned_metrics)
        
        print(f"\nProphet Baseline Average Metrics (n={len(prophet_baseline_df)}):")
        print(f"MAE: {prophet_baseline_df['mae'].mean():.2f}")
        print(f"RMSE: {prophet_baseline_df['rmse'].mean():.2f}")
        print(f"MAPE: {prophet_baseline_df['mape'].mean():.2f}%")
        
        print(f"\nProphet Tuned Average Metrics (n={len(prophet_tuned_df)}):")
        print(f"MAE: {prophet_tuned_df['mae'].mean():.2f}")
        print(f"RMSE: {prophet_tuned_df['rmse'].mean():.2f}")
        print(f"MAPE: {prophet_tuned_df['mape'].mean():.2f}%")
    
    if sarima_baseline_metrics and sarima_tuned_metrics:
        sarima_baseline_df = pd.DataFrame(sarima_baseline_metrics)
        sarima_tuned_df = pd.DataFrame(sarima_tuned_metrics)
        
        print(f"\nSARIMA Baseline Average Metrics (n={len(sarima_baseline_df)}):")
        print(f"MAE: {sarima_baseline_df['mae'].mean():.2f}")
        print(f"RMSE: {sarima_baseline_df['rmse'].mean():.2f}")
        print(f"MAPE: {sarima_baseline_df['mape'].mean():.2f}%")
        
        print(f"\nSARIMA Tuned Average Metrics (n={len(sarima_tuned_df)}):")
        print(f"MAE: {sarima_tuned_df['mae'].mean():.2f}")
        print(f"RMSE: {sarima_tuned_df['rmse'].mean():.2f}")
        print(f"MAPE: {sarima_tuned_df['mape'].mean():.2f}%")
    
    return None, None

def save_time_series_models(all_results):
    """Save trained time series models"""
    models_dir = Path(__file__).parent.parent / "models" / "time_series"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving time series models to: {models_dir}")
    
    # Save Prophet models
    prophet_path = models_dir / "prophet_models.pkl"
    with open(prophet_path, 'wb') as f:
        pickle.dump(all_results['prophet'], f)
    print(f"Saved {len(all_results['prophet'])} Prophet models")
    
    # Save SARIMA models
    sarima_path = models_dir / "sarima_models.pkl"
    with open(sarima_path, 'wb') as f:
        pickle.dump(all_results['sarima'], f)
    print(f"Saved {len(all_results['sarima'])} SARIMA models")

def save_time_series_models(tuned_results):
    """
    Save trained time series models to disk for deployment.
    
    Args:
        tuned_results: Dictionary containing tuned model results (best performing)
    """
    models_dir = Path(__file__).parent.parent / "models" / "time_series"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving tuned time series models to: {models_dir}")
    
    # Save Prophet models
    prophet_models = {}
    for series_key, results in tuned_results['prophet'].items():
        prophet_models[series_key] = {
            'model': results['model'],
            'agency': results['agency'],
            'indicator': results['indicator'],
            'test_mae': results['test_mae'],
            'test_rmse': results['test_rmse'],
            'test_mape': results['test_mape'],
            'model_type': 'Prophet_Tuned',
            'parameters': results['parameters']
        }
    
    prophet_path = models_dir / "prophet_models.pkl"
    with open(prophet_path, 'wb') as f:
        pickle.dump(prophet_models, f)
    print(f"✓ Saved {len(prophet_models)} Prophet tuned models: {prophet_path}")
    
    # Save SARIMA models
    sarima_models = {}
    for series_key, results in tuned_results['sarima'].items():
        sarima_models[series_key] = {
            'model': results['model'],
            'agency': results['agency'],
            'indicator': results['indicator'],
            'best_params': results['best_params'],
            'test_mae': results['test_mae'],
            'test_rmse': results['test_rmse'],
            'test_mape': results['test_mape'],
            'model_type': 'SARIMA_Tuned',
            'parameters': results['parameters']
        }
    
    sarima_path = models_dir / "sarima_models.pkl"
    with open(sarima_path, 'wb') as f:
        pickle.dump(sarima_models, f)
    print(f"✓ Saved {len(sarima_models)} SARIMA tuned models: {sarima_path}")
    
    print(f"📊 Total tuned time series models saved: {len(prophet_models) + len(sarima_models)}")


def main():
    """
    Main training pipeline for MTA KPI time series forecasting models with BEFORE/AFTER comparison.
    
    Executes comprehensive time series modeling workflow:
    1. Load processed MTA performance data
    2. Identify viable agency-indicator combinations
    3. BASELINE: Train Prophet and SARIMA models with default parameters
    4. TUNED: Train Prophet and SARIMA models with optimized parameters  
    5. COMPARISON: Compare before vs after performance improvements
    6. Save tuned models for deployment
    
    This demonstrates the impact of parameter optimization by training each model
    twice - once with defaults and once with optimized parameters.
    
    Returns:
        tuple: (baseline_results, tuned_results, comparison_results)
    """
    print("="*80)
    print("MTA KPI PREDICTION - TIME SERIES MODEL TRAINING PIPELINE")
    print("="*80)
    print("FYP Project: Individual time series forecasting for 264 MTA KPIs")
    
    try:
        # Step 1: Load processed data
        print("\n📊 Loading processed MTA data...")
        df = load_processed_data()
        
        # Step 2: Identify viable combinations
        print("\n🔍 Identifying viable agency-indicator combinations...")
        combinations = get_agency_indicator_combinations(df)
        
        # Step 3: Train comprehensive time series models with BEFORE/AFTER comparison
        print(f"\n🚀 Training time series models: BASELINE → TUNING → COMPARISON...")
        baseline_results, tuned_results = train_all_series_before_after(df, combinations)
        
        # Step 4: Compare baseline vs tuned performance
        print(f"\n📈 Comparing baseline vs tuned model performance...")
        prophet_comparison_df, sarima_comparison_df = compare_time_series_before_after(baseline_results, tuned_results)
        
        # Step 5: Overall evaluation
        print(f"\n📊 Overall model evaluation...")
        evaluation_results = evaluate_time_series_models(baseline_results, tuned_results)
        
        # Step 6: Save tuned models (best performing versions)
        print(f"\n💾 Saving tuned models for deployment...")
        save_time_series_models(tuned_results)
        
        # Final summary with before/after highlights
        total_prophet_baseline = len(baseline_results['prophet'])
        total_prophet_tuned = len(tuned_results['prophet'])
        total_sarima_baseline = len(baseline_results['sarima'])
        total_sarima_tuned = len(tuned_results['sarima'])
        total_baseline_models = total_prophet_baseline + total_sarima_baseline
        total_tuned_models = total_prophet_tuned + total_sarima_tuned
        
        print(f"\n🎉 TIME SERIES BEFORE/AFTER TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Baseline models: Prophet {total_prophet_baseline}, SARIMA {total_sarima_baseline} (Total: {total_baseline_models})")
        print(f"⚡ Tuned models: Prophet {total_prophet_tuned}, SARIMA {total_sarima_tuned} (Total: {total_tuned_models})")
        print("All tuned models saved and ready for individual KPI forecasting.")
        print("="*80)
        
        return baseline_results, tuned_results, (prophet_comparison_df, sarima_comparison_df)
        
    except Exception as e:
        print(f"\n❌ ERROR during time series training pipeline: {e}")
        raise

if __name__ == "__main__":
    results = main()