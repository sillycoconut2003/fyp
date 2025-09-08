import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import pickle
import warnings
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add the src directory to the path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROCESSED, COLS
from model_ts import fit_prophet
from eval import rmse, mae, mape

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the processed dataset for time series training"""
    print(f"Loading processed data from: {PROCESSED}")
    df = pd.read_parquet(PROCESSED)
    print(f"Loaded dataset with shape: {df.shape}")
    return df

def get_agency_indicator_combinations(df):
    """Get all unique agency-indicator combinations"""
    combinations = df.groupby([COLS['agency'], COLS['indicator']]).size().reset_index(name='count')
    # Filter combinations with sufficient data points (at least 24 months)
    combinations = combinations[combinations['count'] >= 24]
    print(f"Found {len(combinations)} agency-indicator combinations with >=24 data points")
    return combinations

def prepare_series_data(df, agency, indicator):
    """Prepare data for a specific agency-indicator combination"""
    series_df = df[
        (df[COLS['agency']] == agency) & 
        (df[COLS['indicator']] == indicator)
    ].sort_values('YYYY_MM').copy()
    
    # Remove rows with missing target values
    series_df = series_df.dropna(subset=['MONTHLY_ACTUAL'])
    
    return series_df

def split_series_temporal(series_df, test_months=12):
    """Split series data temporally"""
    if len(series_df) < test_months + 12:  # Need at least 12 months for training
        return None, None
    
    split_idx = len(series_df) - test_months
    train_df = series_df.iloc[:split_idx].copy()
    test_df = series_df.iloc[split_idx:].copy()
    
    return train_df, test_df

def train_prophet_model(train_df, test_df, agency, indicator):
    """Train Prophet model for a specific series"""
    try:
        # Prepare data for Prophet
        prophet_df = train_df[['YYYY_MM', 'MONTHLY_ACTUAL']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Remove any remaining NaN values
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) < 12:  # Need at least 12 data points
            return None
        
        # Initialize and fit Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        model.fit(prophet_df)
        
        # Make predictions for test period
        future = model.make_future_dataframe(periods=len(test_df), freq='MS')
        forecast = model.predict(future)
        
        # Extract test predictions
        test_predictions = forecast.tail(len(test_df))['yhat'].values
        test_actual = test_df['MONTHLY_ACTUAL'].values
        
        # Calculate metrics
        test_mae = mae(test_actual, test_predictions)
        test_rmse = rmse(test_actual, test_predictions)
        test_mape = mape(test_actual, test_predictions)
        
        results = {
            'model': model,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'predictions': test_predictions,
            'actual': test_actual,
            'forecast_df': forecast,
            'series_length': len(prophet_df)
        }
        
        return results
        
    except Exception as e:
        print(f"Error training Prophet for {agency} - {indicator}: {str(e)}")
        return None

def find_best_sarima_params(train_df, max_p=3, max_d=2, max_q=3):
    """Find best SARIMA parameters using AIC"""
    try:
        best_aic = float('inf')
        best_params = None
        
        # Prepare data
        ts_data = train_df.set_index('YYYY_MM')['MONTHLY_ACTUAL']
        ts_data = ts_data.asfreq('MS')  # Monthly start frequency
        
        # Grid search for best parameters
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = SARIMAX(ts_data, order=(p, d, q), seasonal_order=(1, 1, 1, 12))
                        fitted_model = model.fit(disp=False)
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        return best_params if best_params else (1, 1, 1)
        
    except:
        return (1, 1, 1)  # Default parameters

def train_sarima_model(train_df, test_df, agency, indicator):
    """Train SARIMA model for a specific series"""
    try:
        # Prepare data
        ts_data = train_df.set_index('YYYY_MM')['MONTHLY_ACTUAL']
        ts_data = ts_data.asfreq('MS')
        
        if len(ts_data) < 24:  # Need at least 24 data points for seasonal ARIMA
            return None
        
        # Find best parameters
        best_params = find_best_sarima_params(train_df)
        
        # Fit SARIMA model
        model = SARIMAX(
            ts_data, 
            order=best_params, 
            seasonal_order=(1, 1, 1, 12)
        )
        fitted_model = model.fit(disp=False)
        
        # Make predictions
        forecast_steps = len(test_df)
        forecast = fitted_model.forecast(steps=forecast_steps)
        
        test_actual = test_df['MONTHLY_ACTUAL'].values
        test_predictions = forecast.values
        
        # Calculate metrics
        test_mae = mae(test_actual, test_predictions)
        test_rmse = rmse(test_actual, test_predictions)
        test_mape = mape(test_actual, test_predictions)
        
        results = {
            'model': fitted_model,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'predictions': test_predictions,
            'actual': test_actual,
            'params': best_params,
            'series_length': len(ts_data)
        }
        
        return results
        
    except Exception as e:
        print(f"Error training SARIMA for {agency} - {indicator}: {str(e)}")
        return None

def train_all_series(df, combinations, max_series=None):
    """Train models for all agency-indicator combinations"""
    if max_series:
        combinations = combinations.head(max_series)
    
    all_results = {
        'prophet': {},
        'sarima': {}
    }
    
    successful_prophet = 0
    successful_sarima = 0
    total_series = len(combinations)
    
    for idx, row in combinations.iterrows():
        agency = row[COLS['agency']]
        indicator = row[COLS['indicator']]
        
        print(f"\n[{idx+1}/{total_series}] Training models for: {agency} - {indicator}")
        
        # Prepare series data
        series_df = prepare_series_data(df, agency, indicator)
        
        if len(series_df) < 24:
            print(f"Insufficient data ({len(series_df)} points), skipping...")
            continue
        
        # Split data
        train_df, test_df = split_series_temporal(series_df)
        
        if train_df is None or test_df is None:
            print("Insufficient data for train/test split, skipping...")
            continue
        
        series_key = f"{agency}|{indicator}"
        
        # Train Prophet
        prophet_results = train_prophet_model(train_df, test_df, agency, indicator)
        if prophet_results:
            all_results['prophet'][series_key] = prophet_results
            successful_prophet += 1
            print(f"Prophet - MAE: {prophet_results['test_mae']:.2f}, MAPE: {prophet_results['test_mape']:.2f}%")
        
        # Train SARIMA
        sarima_results = train_sarima_model(train_df, test_df, agency, indicator)
        if sarima_results:
            all_results['sarima'][series_key] = sarima_results
            successful_sarima += 1
            print(f"SARIMA - MAE: {sarima_results['test_mae']:.2f}, MAPE: {sarima_results['test_mape']:.2f}%")
        
        # Progress update every 10 series
        if (idx + 1) % 10 == 0:
            print(f"\n--- Progress Update ---")
            print(f"Completed: {idx + 1}/{total_series} series")
            print(f"Prophet successes: {successful_prophet}")
            print(f"SARIMA successes: {successful_sarima}")
    
    print(f"\n=== Training Summary ===")
    print(f"Total series processed: {total_series}")
    print(f"Successful Prophet models: {successful_prophet}")
    print(f"Successful SARIMA models: {successful_sarima}")
    
    return all_results

def evaluate_time_series_models(all_results):
    """Evaluate and compare time series models"""
    print("\n" + "="*50)
    print("TIME SERIES MODEL EVALUATION")
    print("="*50)
    
    # Calculate aggregate metrics
    prophet_metrics = []
    sarima_metrics = []
    
    for series_key in all_results['prophet']:
        if series_key in all_results['sarima']:  # Only compare series where both models worked
            prophet_metrics.append({
                'series': series_key,
                'mae': all_results['prophet'][series_key]['test_mae'],
                'rmse': all_results['prophet'][series_key]['test_rmse'],
                'mape': all_results['prophet'][series_key]['test_mape']
            })
            
            sarima_metrics.append({
                'series': series_key,
                'mae': all_results['sarima'][series_key]['test_mae'],
                'rmse': all_results['sarima'][series_key]['test_rmse'],
                'mape': all_results['sarima'][series_key]['test_mape']
            })
    
    if prophet_metrics and sarima_metrics:
        prophet_df = pd.DataFrame(prophet_metrics)
        sarima_df = pd.DataFrame(sarima_metrics)
        
        print(f"\nProphet Average Metrics (n={len(prophet_df)}):")
        print(f"MAE: {prophet_df['mae'].mean():.2f}")
        print(f"RMSE: {prophet_df['rmse'].mean():.2f}")
        print(f"MAPE: {prophet_df['mape'].mean():.2f}%")
        
        print(f"\nSARIMA Average Metrics (n={len(sarima_df)}):")
        print(f"MAE: {sarima_df['mae'].mean():.2f}")
        print(f"RMSE: {sarima_df['rmse'].mean():.2f}")
        print(f"MAPE: {sarima_df['mape'].mean():.2f}%")
        
        # Count wins
        prophet_wins = sum(prophet_df['mae'] < sarima_df['mae'])
        sarima_wins = sum(sarima_df['mae'] < prophet_df['mae'])
        
        print(f"\nModel Comparison (by MAE):")
        print(f"Prophet wins: {prophet_wins}")
        print(f"SARIMA wins: {sarima_wins}")
        
        return prophet_df, sarima_df
    
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

def main():
    """Main training pipeline for time series models"""
    print("Starting Time Series Model Training Pipeline...")
    
    # Load data
    df = load_processed_data()
    
    # Get agency-indicator combinations
    combinations = get_agency_indicator_combinations(df)
    
    # Train all series (removed max_series limit for full training)
    print(f"\nTraining models for all {len(combinations)} series...")
    
    # Train all series
    all_results = train_all_series(df, combinations)
    
    # Evaluate models
    prophet_df, sarima_df = evaluate_time_series_models(all_results)
    
    # Save models
    save_time_series_models(all_results)
    
    print("\nTime Series Training Pipeline Completed!")
    return all_results

if __name__ == "__main__":
    results = main()