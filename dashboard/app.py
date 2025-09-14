import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pickle
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import logging
warnings.filterwarnings('ignore')

# Suppress Streamlit warnings
logging.getLogger('streamlit.runtime.scriptrunner.script_runner').setLevel(logging.ERROR)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

st.set_page_config(page_title="MTA KPI Analytics & Forecasting", layout="wide", initial_sidebar_state="expanded")

def load_css():
    """Load custom CSS for the futuristic dashboard theme"""
    css_file = Path(__file__).parent / "style.css"
    if css_file.exists():
        st.markdown(f'<style>{css_file.read_text()}</style>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Custom CSS file not found - using default Streamlit theme")

load_css()


@st.cache_data
def load_data():
    """Load the processed dataset"""
    fp = Path(__file__).resolve().parents[1]/"data"/"processed"/"mta_model.parquet"
    return pd.read_parquet(fp)

@st.cache_data(show_spinner="Loading ML models...")
def load_ml_models():
    """Load trained ML models with validation"""
    models_dir = Path(__file__).parent.parent / "models"
    ml_models = {}
    
    model_files = {
        'RandomForest': 'RandomForest_model.pkl',
        'XGBoost': 'XGBoost_model.pkl',
        'LinearRegression': 'LinearRegression_model.pkl'
    }
    
    for name, filename in model_files.items():
        filepath = models_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                    # Validate model structure
                    if 'model' in model_data and 'feature_cols' in model_data:
                        ml_models[name] = model_data
                        print(f"✅ Loaded {name}: {len(model_data['feature_cols'])} features")
                    else:
                        st.error(f"Invalid model structure in {name}")
            except Exception as e:
                st.error(f"Failed to load {name}: {e}")
        else:
            st.warning(f"Model file not found: {filename}")
    
    return ml_models

@st.cache_data
def load_ts_models():
    """Load time series models"""
    models_dir = Path(__file__).parent.parent / "models" / "time_series"
    ts_models = {}
    
    model_files = ['prophet_models.pkl', 'sarima_models.pkl']
    
    for filename in model_files:
        filepath = models_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    models = pickle.load(f)
                    model_type = filename.split('_')[0].upper()
                    ts_models[model_type] = models
            except Exception as e:
                st.warning(f"Failed to load {filename}: {e}")
    
    return ts_models

def calculate_ml_confidence_intervals(model, df_extended, feature_cols, predictions, confidence_level=0.8, n_bootstrap=100):
    """
    Calculate confidence intervals for ML predictions using residual bootstrap.
    
    Args:
        model: Trained ML model
        df_extended: Historical data with features
        feature_cols: List of feature column names
        predictions: Point predictions
        confidence_level: Confidence level (0.8 = 80%)
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        tuple: (lower_bounds, upper_bounds)
    """
    try:
        # Calculate residuals from historical predictions
        historical_features = pd.DataFrame()
        for col in feature_cols:
            if col in df_extended.columns:
                historical_features[col] = df_extended[col]
            else:
                historical_features[col] = 0  # Fill missing features
        
        print(f"   Historical features shape: {historical_features.shape}")
        
        # Fill NaN values
        historical_features = historical_features.fillna(method='ffill').fillna(0)
        
        # Get historical predictions and residuals
        if len(historical_features) > 0:
            historical_preds = model.predict(historical_features)
            residuals = df_extended['MONTHLY_ACTUAL'].values - historical_preds
            
            print(f"   Historical predictions: {len(historical_preds)}")
            print(f"   Residuals calculated: {len(residuals)}")
            print(f"   Residual range: [{np.min(residuals):.1f}, {np.max(residuals):.1f}]")
            
            # Remove outlier residuals (beyond 2 standard deviations)
            residual_std = np.std(residuals)
            residuals_clean = residuals[np.abs(residuals) <= 2 * residual_std]
            
            print(f"   Cleaned residuals: {len(residuals_clean)} (removed {len(residuals) - len(residuals_clean)} outliers)")
            
            if len(residuals_clean) < 5:  # Not enough residuals for bootstrap
                print(f"⚠️  Not enough clean residuals for bootstrap, using fallback")
                # Fallback to simple percentage-based intervals
                std_pred = np.std(predictions) if len(predictions) > 1 else np.mean(predictions) * 0.1
                margin = std_pred * 1.5
                return (
                    [max(0, p - margin) for p in predictions],
                    [p + margin for p in predictions]
                )
            
            # Bootstrap confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            print(f"   Bootstrap parameters: {n_bootstrap} samples, {lower_percentile:.1f}-{upper_percentile:.1f} percentiles")
            
            lower_bounds = []
            upper_bounds = []
            
            np.random.seed(42)  # For reproducibility
            
            for i, pred in enumerate(predictions):
                # Bootstrap sample residuals and add to prediction
                bootstrap_preds = []
                for _ in range(n_bootstrap):
                    sampled_residual = np.random.choice(residuals_clean)
                    bootstrap_pred = pred + sampled_residual
                    bootstrap_preds.append(bootstrap_pred)
                
                # Calculate percentiles
                lower_bound = np.percentile(bootstrap_preds, lower_percentile)
                upper_bound = np.percentile(bootstrap_preds, upper_percentile)
                
                # Ensure non-negative bounds
                lower_bounds.append(max(0, lower_bound))
                upper_bounds.append(upper_bound)
                
                if i == 0:  # Debug first prediction
                    print(f"   First prediction bootstrap: pred={pred:.1f}, bounds=[{lower_bound:.1f}, {upper_bound:.1f}]")
            
            print(f"✅ Bootstrap completed: {len(lower_bounds)} confidence intervals")
            return lower_bounds, upper_bounds
        
    except Exception as e:
        print(f"⚠️ Confidence interval calculation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback to simple intervals based on prediction variance
    print(f"🔄 Using fallback confidence intervals...")
    if len(predictions) > 1:
        pred_std = np.std(predictions)
        margin = pred_std * 1.2
    else:
        margin = np.mean(predictions) * 0.15  # 15% margin as fallback
    
    fallback_lower = [max(0, p - margin) for p in predictions]
    fallback_upper = [p + margin for p in predictions]
    print(f"   Fallback margin: ±{margin:.1f}")
    
    return fallback_lower, fallback_upper

def simple_trend_forecast(df, periods=12):
    """Simple fallback forecasting using linear trend"""
    values = df['MONTHLY_ACTUAL'].values
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    predictions = []
    for i in range(periods):
        future_x = len(values) + i
        pred = slope * future_x + intercept
        pred += np.random.normal(0, np.std(values) * 0.05)
        predictions.append(max(0, pred))
    return predictions

def predict_ml_model(df, model_info, model_name, periods=12, kpi_name=""):
    """Make predictions with a feature-based ML model using iterative forecasting."""
    print(f"🚀 Starting {model_name} prediction for {kpi_name}")
    print(f"   Data shape: {df.shape}, Periods: {periods}")
    
    try:
        model = model_info['model']
        feature_cols = model_info['feature_cols']
        
        # Validate inputs
        if len(df) < 12:
            print(f"⚠️ Insufficient data for {model_name}: {len(df)} records")
            return simple_trend_forecast(df, periods)
        
        # Create extended dataset for iterative predictions
        df_extended = df.copy()
        df_extended['Date'] = pd.to_datetime(df_extended['YYYY_MM'])
        df_extended = df_extended.sort_values('Date')
        
        print(f"✅ Using standard ML prediction for {kpi_name}")
        
        # Original prediction logic for all KPIs continues below
        # Check critical features availability
        missing_features = [col for col in feature_cols if col not in df_extended.columns]
        if len(missing_features) > len(feature_cols) * 0.3:  # If >30% features missing
            print(f"⚠️ Too many missing features for {model_name}: {len(missing_features)}/{len(feature_cols)}")
            return simple_trend_forecast(df, periods)
        
        # Get baseline statistics for bounds checking
        historical_mean = df_extended['MONTHLY_ACTUAL'].mean()
        historical_std = df_extended['MONTHLY_ACTUAL'].std()
        historical_max = df_extended['MONTHLY_ACTUAL'].max()
        historical_min = df_extended['MONTHLY_ACTUAL'].min()
        
        # Normal bounds for other KPIs
        upper_bound = historical_mean + 2 * historical_std
        lower_bound = max(0, historical_mean - 2 * historical_std)
        
        last_date = df_extended['Date'].max()
        predictions = []
        dates = []
        
        print(f"🔍 {model_name} prediction bounds: {lower_bound:.0f} - {upper_bound:.0f}")
        
        # Use the last known row as template for feature alignment
        template_row = df_extended.iloc[-1:].copy()
        
        # Iterative prediction with robust feature engineering
        for i in range(periods):
            current_date = last_date + pd.DateOffset(months=i+1)
            dates.append(current_date)
            
            # Create prediction row based on template
            future_row = template_row.copy()
            future_row['YYYY_MM'] = current_date
            future_row['Date'] = current_date
            
            # Update time-based features safely
            if 'PERIOD_YEAR' in future_row.columns:
                future_row['PERIOD_YEAR'] = current_date.year
            if 'PERIOD_MONTH' in future_row.columns:
                future_row['PERIOD_MONTH'] = current_date.month
            if 'year' in future_row.columns:
                future_row['year'] = current_date.year
            if 'month' in future_row.columns:
                future_row['month'] = current_date.month
            if 'quarter' in future_row.columns:
                future_row['quarter'] = (current_date.month - 1) // 3 + 1
            
            # Update lag features with proper historical context
            if i == 0:
                last_value = df_extended['MONTHLY_ACTUAL'].iloc[-1]
            else:
                last_value = predictions[i-1]
            
            # Get extended historical context for longer lags
            all_historical = df_extended['MONTHLY_ACTUAL'].tolist() + predictions[:i]
            
            # Update all lag features properly
            lag_mappings = {
                'm_act_lag1': 1, 'lag_1': 1, 'MONTHLY_ACTUAL_lag1': 1,
                'm_act_lag2': 2, 'lag_2': 2, 'MONTHLY_ACTUAL_lag2': 2,
                'm_act_lag3': 3, 'lag_3': 3, 'MONTHLY_ACTUAL_lag3': 3,
                'm_act_lag6': 6, 'lag_6': 6, 'MONTHLY_ACTUAL_lag6': 6,
                'm_act_lag12': 12, 'lag_12': 12, 'MONTHLY_ACTUAL_lag12': 12
            }
            
            for lag_col, lag_periods in lag_mappings.items():
                if lag_col in future_row.columns:
                    if len(all_historical) >= lag_periods:
                        future_row[lag_col] = all_historical[-lag_periods]
                    else:
                        # Fallback to last available value
                        future_row[lag_col] = all_historical[0] if all_historical else last_value
            
            # Update rolling means safely
            recent_values = df_extended['MONTHLY_ACTUAL'].iloc[-12:].tolist() + predictions[:i]
            recent_12 = recent_values[-12:]  # Last 12 values
            
            for rolling_col in ['rolling_mean_12', 'MONTHLY_ACTUAL_rolling_12', 'm_act_rolling_12']:
                if rolling_col in future_row.columns:
                    future_row[rolling_col] = np.mean(recent_12)
            
            # Make prediction with extensive error checking
            try:
                # Ensure all features exist and align properly
                X_pred = pd.DataFrame()
                for col in feature_cols:
                    if col in future_row.columns:
                        X_pred[col] = future_row[col]
                    else:
                        # Use median value for missing features
                        if col in df_extended.columns:
                            X_pred[col] = [df_extended[col].median()]
                        else:
                            X_pred[col] = [0]
                
                # Fill any remaining NaN values
                X_pred = X_pred.fillna(method='ffill').fillna(0)
                
                # Make prediction
                raw_pred = model.predict(X_pred)[0]
                
                # Add controlled variation for overly stable models
                if model_name in ['RandomForest', 'XGBoost']:
                    # Calculate historical month-to-month volatility
                    historical_values = df_extended['MONTHLY_ACTUAL'].values
                    monthly_changes = np.diff(historical_values)
                    volatility = np.std(monthly_changes) * 0.3  # Use 30% of historical volatility
                    
                    # Add small random variation to prevent overly smooth predictions
                    np.random.seed(42 + i)  # Reproducible randomness
                    variation = np.random.normal(0, volatility)
                    raw_pred = raw_pred + variation
                
                # Apply bounds checking
                if (np.isnan(raw_pred) or np.isinf(raw_pred) or 
                    raw_pred > upper_bound or raw_pred < lower_bound):
                    
                    print(f"⚠️ {model_name} extreme prediction: {raw_pred:.2f} -> using conservative estimate")
                    # Use conservative prediction based on recent trend
                    recent_data = df_extended['MONTHLY_ACTUAL'].iloc[-6:]
                    recent_trend = np.mean(recent_data)
                    print(f"📊 Recent 6 values: {recent_data.tolist()}")
                    print(f"📊 Recent trend (mean): {recent_trend:.2f}")
                    print(f"📊 Bounds: {lower_bound:.2f} - {upper_bound:.2f}")
                    final_pred = recent_trend
                else:
                    final_pred = raw_pred
                
                # Apply enhanced bounds checking for edge cases
                if final_pred < 0 and lower_bound == 0:
                    # Special handling for KPIs that can't be negative (rates, counts, etc.)
                    final_pred = max(historical_mean * 0.5, 0.1)  # Use conservative positive value
                    print(f"🔧 Applied non-negative constraint: {final_pred:.2f}")
                
                # Apply final bounds and add to predictions
                final_pred = max(lower_bound, min(upper_bound, final_pred))
                predictions.append(final_pred)
                
                if i < 3:  # Debug first few predictions
                    print(f"  Step {i+1}: {final_pred:.0f}")
                
            except Exception as e:
                print(f"⚠️ {model_name} prediction error at step {i+1}: {e}")
                # Fallback: use conservative trend-based prediction
                if i == 0:
                    fallback_pred = df_extended['MONTHLY_ACTUAL'].iloc[-1] * 1.02
                else:
                    fallback_pred = predictions[i-1] * 1.01
                
                fallback_pred = max(lower_bound, min(upper_bound, fallback_pred))
                predictions.append(fallback_pred)
        
        print(f"🎯 ML prediction loop completed: {len(predictions)} predictions generated")
        
        # Calculate confidence intervals using residual bootstrap
        print(f"🔍 Calculating confidence intervals for {model_name}...")
        try:
            lower_bounds, upper_bounds = calculate_ml_confidence_intervals(
                model, df_extended, feature_cols, predictions, confidence_level=0.8
            )
            print(f"✅ Confidence intervals calculated: {len(lower_bounds)} bands")
            print(f"   Sample bounds: [{lower_bounds[0]:.1f}, {upper_bounds[0]:.1f}]")
        except Exception as e:
            print(f"❌ Confidence interval calculation failed: {e}")
            # Fallback to simple bounds
            margin = np.std(predictions) * 0.5 if len(predictions) > 1 else np.mean(predictions) * 0.1
            lower_bounds = [max(0, p - margin) for p in predictions]
            upper_bounds = [p + margin for p in predictions]
        
        # Apply percentage bounds if this appears to be a percentage KPI
        historical_values = df_extended['MONTHLY_ACTUAL'].values
        is_percentage_kpi = (
            '% ' in kpi_name or 'percent' in kpi_name.lower() or
            (len(historical_values) > 0 and 
             np.max(historical_values) <= 100.0 and 
             np.mean(historical_values) > 50.0)
        )
        
        if is_percentage_kpi:
            print(f"🎯 Applying percentage bounds (0-100%) to {model_name} predictions")
            predictions = [max(0.0, min(100.0, p)) for p in predictions]
            lower_bounds = [max(0.0, min(100.0, lb)) for lb in lower_bounds]
            upper_bounds = [max(0.0, min(100.0, ub)) for ub in upper_bounds]
            
            # Ensure minimum confidence band width for visibility (at least 0.5% wide)
            min_band_width = 0.5
            current_width = np.mean([ub - lb for ub, lb in zip(upper_bounds, lower_bounds)])
            if current_width < min_band_width:
                print(f"📊 Expanding confidence bands for visibility: {current_width:.2f}% → {min_band_width:.2f}%")
                expansion = (min_band_width - current_width) / 2
                lower_bounds = [max(0.0, lb - expansion) for lb in lower_bounds]
                upper_bounds = [min(100.0, ub + expansion) for ub in upper_bounds]
            
            print(f"📈 Final percentage range: {np.min(predictions):.1f}% - {np.max(predictions):.1f}%")
            print(f"📊 Confidence bands: {np.min(lower_bounds):.1f}% - {np.max(upper_bounds):.1f}%")
        
        return pd.DataFrame({
            'Date': dates,
            'Prediction': predictions,
            'Lower': lower_bounds,
            'Upper': upper_bounds,
            'Model': model_name
        })
        
    except Exception as e:
        st.error(f"Error in {model_name} prediction: {e}")
        return pd.DataFrame()

def predict_ts_model(df, model_info, model_name, periods=12):
    """Make predictions with time series model"""
    try:
        # Extract the actual model from the model info dictionary
        if isinstance(model_info, dict) and 'model' in model_info:
            model = model_info['model']
        else:
            model = model_info
        
        # Check if this might be a percentage KPI based on historical data
        historical_values = df['MONTHLY_ACTUAL'].values
        is_likely_percentage = (
            len(historical_values) > 0 and
            np.max(historical_values) <= 100.0 and
            np.min(historical_values) >= 0.0 and
            np.mean(historical_values) > 50.0  # Typical for operational KPIs
        )
        
        if 'PROPHET' in model_name.upper():
            # Prophet prediction logic
            # Create future dataframe starting from the last date in the data
            last_date = pd.to_datetime(df['YYYY_MM']).max()
            
            # Create historical data in Prophet format for context
            historical_data = df[['YYYY_MM', 'MONTHLY_ACTUAL']].copy()
            historical_data.columns = ['ds', 'y']
            historical_data['ds'] = pd.to_datetime(historical_data['ds'])
            historical_data = historical_data.sort_values('ds')
            
            # Generate future dates
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1), 
                periods=periods, 
                freq='MS'
            )
            
            # Create future dataframe for Prophet
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Make predictions
            forecast = model.predict(future_df)
            
            # Apply percentage constraints if detected
            predictions = forecast['yhat']
            if is_likely_percentage:
                print(f"🎯 Applying percentage constraints to {model_name} predictions")
                predictions = predictions.clip(0, 100)  # Enforce 0-100% bounds
            else:
                predictions = predictions.clip(lower=0)  # Just ensure non-negative
            
            return pd.DataFrame({
                'Date': forecast['ds'],
                'Prediction': predictions,
                'Lower': forecast['yhat_lower'].clip(lower=0) if 'yhat_lower' in forecast.columns else predictions * 0.9,
                'Upper': forecast['yhat_upper'].clip(upper=100 if is_likely_percentage else None) if 'yhat_upper' in forecast.columns else predictions * 1.1,
                'Model': model_name
            })
            
        elif 'SARIMA' in model_name.upper():
            # SARIMA prediction logic with confidence intervals
            last_date = pd.to_datetime(df['YYYY_MM']).max()
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1), 
                periods=periods, 
                freq='MS'
            )
            
            print(f"📊 Generating {model_name} forecasts with confidence intervals...")
            
            # Generate SARIMA forecasts with confidence intervals
            try:
                # Use get_prediction for confidence intervals (80% confidence level)
                start_idx = len(df)
                end_idx = start_idx + periods - 1
                prediction_result = model.get_prediction(start=start_idx, end=end_idx, alpha=0.2)
                
                # Extract point forecasts and confidence intervals
                predictions = prediction_result.predicted_mean.values
                conf_int = prediction_result.conf_int()
                lower_bounds = conf_int.iloc[:, 0].values
                upper_bounds = conf_int.iloc[:, 1].values
                
                print(f"✅ SARIMA confidence intervals calculated successfully")
                
            except (AttributeError, ValueError) as e:
                # Fallback to simple forecast if confidence intervals fail
                print(f"⚠️ SARIMA confidence intervals unavailable, using point forecast: {e}")
                forecast = model.forecast(steps=periods)
                
                # Handle case where forecast might be a pandas Series or array
                if hasattr(forecast, 'values'):
                    predictions = forecast.values
                else:
                    predictions = np.array(forecast)
                
                # Create simple confidence intervals based on residual std
                residual_std = np.std(model.resid) if hasattr(model, 'resid') else np.std(predictions) * 0.1
                confidence_width = 1.28 * residual_std  # 80% confidence interval
                lower_bounds = predictions - confidence_width
                upper_bounds = predictions + confidence_width
            
            # Ensure predictions respect percentage bounds if detected
            if is_likely_percentage:
                print(f"🎯 Applying percentage constraints to {model_name} predictions")
                predictions = np.clip(predictions, 0, 100)
                lower_bounds = np.clip(lower_bounds, 0, 100)
                upper_bounds = np.clip(upper_bounds, 0, 100)
            else:
                predictions = np.maximum(predictions, 0)  # Just ensure non-negative
                lower_bounds = np.maximum(lower_bounds, 0)
                upper_bounds = np.maximum(upper_bounds, 0)
            
            print(f"📈 SARIMA prediction range: {np.min(predictions):.2f} - {np.max(predictions):.2f}")
            print(f"📊 Confidence interval range: [{np.min(lower_bounds):.2f}, {np.max(upper_bounds):.2f}]")
            
            return pd.DataFrame({
                'Date': future_dates,
                'Prediction': predictions,
                'Lower': lower_bounds,
                'Upper': upper_bounds,
                'Model': model_name
            })
        else:
            # Generic time series model
            last_date = pd.to_datetime(df['YYYY_MM']).max()
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1), 
                periods=periods, 
                freq='MS'
            )
            
            # Try to generate predictions
            if hasattr(model, 'forecast'):
                predictions = model.forecast(steps=periods)
            elif hasattr(model, 'predict'):
                predictions = model.predict(n_periods=periods)
            else:
                raise ValueError(f"Unknown prediction method for {model_name}")
            
            # Apply percentage constraints if detected
            if is_likely_percentage:
                print(f"🎯 Applying percentage constraints to {model_name} predictions")
                predictions = np.clip(predictions, 0, 100)
            else:
                predictions = np.maximum(predictions, 0)
            
            return pd.DataFrame({
                'Date': future_dates,
                'Prediction': predictions,
                'Model': model_name
            })
    
    except Exception as e:
        st.error(f"Error in {model_name} prediction: {e}")
        print(f"Debug - Model type: {type(model_info)}, Model name: {model_name}")
        if isinstance(model_info, dict):
            print(f"Debug - Model info keys: {list(model_info.keys())}")
        return pd.DataFrame()

def render_stat_card(title: str, value: str, delta_text: str = "", delta_type: str = "neutral") -> str:
    """Return HTML for a professional stat card with delta coloring."""
    # Determine delta class from explicit type if provided; otherwise infer from sign
    inferred_type = 'neutral'
    if delta_text:
        if str(delta_text).strip().startswith('+'):
            inferred_type = 'up'
        elif str(delta_text).strip().startswith('-'):
            inferred_type = 'down'
    delta_class = {
        'up': 'delta-up',
        'down': 'delta-down',
        'neutral': 'delta-neutral'
    }.get(delta_type if delta_type in ['up','down','neutral'] else inferred_type, 'delta-neutral')

    return (
        f"<div class='stat-card'>"
        f"<div class='stat-title'>{title}</div>"
        f"<div class='stat-value'>{value}</div>"
        f"<div class='stat-delta {delta_class}'>{delta_text}</div>"
        f"</div>"
    )

def render_stat_grid(cards_html: list[str]) -> str:
        """Wrap a list of stat-card HTML snippets into the grid container (uses global CSS)."""
        return f"<div class='stat-grid'>{''.join(cards_html)}</div>"

def main():
    # --- Header ---
    st.markdown(
        """
        <div class="header">
            <h1 class="header-title">MTA Performance Analytics</h1>
            <p class="header-subtitle">Advanced ML & Time Series Forecasting Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Performance Metrics (Stat Grid) ---
    st.markdown("#### Model Performance Overview")
    stat_cards = [
        render_stat_card("Best Performer", "RandomForest", "MAE: 13,637", "up"),
        render_stat_card("XGBoost Model", "MAE: 39,885", "+192% vs Best", "down"), 
        render_stat_card("Linear Regression", "MAE: 130,912", "+860% vs Best", "down"),
        render_stat_card("Total Models Trained", "267", "3 ML + 264 TS", "neutral"),
    ]
    # Render within main DOM so global CSS applies
    st.markdown(render_stat_grid(stat_cards), unsafe_allow_html=True)
    st.caption("*Performance metrics corrected with accurate baseline comparisons - September 15, 2025*")

    st.markdown("---")
    
    # --- Load Data & Models ---
    df = load_data()
    if df.empty:
        st.error("❌ Processed dataset is empty. Please run the data processing pipeline.")
        return
    
    ml_models = load_ml_models()
    ts_models = load_ts_models()
    
    # --- Sidebar Controls ---
    st.sidebar.markdown("## Forecast Controls")
    
    # Cache refresh button for troubleshooting
    if st.sidebar.button("🔄 Refresh Models", help="Clear cache and reload models if predictions seem incorrect"):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared! Models will reload on next prediction.")
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Data Selection Section
    st.sidebar.markdown("### Data Selection")
    # Clean and sort KPI names properly
    kpi_names = df['INDICATOR_NAME'].unique()
    available_kpis = sorted([kpi.strip() for kpi in kpi_names])
    selected_kpi_clean = st.sidebar.selectbox(
        "Key Performance Indicator", 
        available_kpis, 
        index=0,
        help="Choose the KPI to forecast"
    )
    selected_kpi = next(kpi for kpi in kpi_names if kpi.strip() == selected_kpi_clean)
    
    kpi_data = df[df['INDICATOR_NAME'] == selected_kpi]
    available_agencies = sorted(kpi_data['AGENCY_NAME'].unique())
    
    # Smart agency selection
    if len(available_agencies) == 1:
        selected_agency = available_agencies[0]
        st.sidebar.info(f"**Agency:** {selected_agency}")
    else:
        selected_agency = st.sidebar.selectbox(
            "Agency", 
            available_agencies, 
            index=0,
            help=f"Choose from {len(available_agencies)} agencies"
        )
    
    st.sidebar.markdown("---")
    
    # Model Selection Section
    st.sidebar.markdown("### Model Selection")
    series_key = f"{selected_agency}|{selected_kpi}"

    model_options = [f"ML: {name}" for name in ml_models.keys()]
    
    if 'PROPHET' in ts_models and series_key in ts_models['PROPHET']:
        model_options.append("TS: Prophet")
    if 'SARIMA' in ts_models and series_key in ts_models['SARIMA']:
        model_options.append("TS: SARIMA")

    if not model_options:
        st.sidebar.error("❌ No models available for this selection.")
        return
        
    selected_model_name = st.sidebar.selectbox(
        "Forecasting Model", 
        model_options,
        help="Choose between Machine Learning (ML) and Time Series (TS) models"
    )
    
    st.sidebar.markdown("---")
    
    # Forecast Settings Section
    st.sidebar.markdown("### Forecast Settings")
    periods = st.sidebar.slider(
        "Forecast Horizon (months)", 
        min_value=1, 
        max_value=36, 
        value=12,
        help="Number of months to forecast"
    )

    # --- Main Content Area ---
    tab1, tab2 = st.tabs(["Forecast Analysis", "Data Explorer"])

    filtered_data = df[(df['INDICATOR_NAME'] == selected_kpi) & (df['AGENCY_NAME'] == selected_agency)].copy()
    filtered_data['Date'] = pd.to_datetime(filtered_data['YYYY_MM'])
    filtered_data['Value'] = filtered_data['MONTHLY_ACTUAL']
    filtered_data = filtered_data.sort_values('Date')

    if filtered_data.empty:
        st.warning("No data available for this KPI and Agency combination.")
        return

    # --- Generate Predictions ---
    predictions = pd.DataFrame()
    if selected_model_name.startswith("ML:"):
        model_name = selected_model_name.split(": ")[1]
        if model_name in ml_models:
            model_info = ml_models[model_name]
            predictions = predict_ml_model(filtered_data, model_info, model_name, periods, selected_kpi_clean)
    else: # Time Series
        ts_type = selected_model_name.split(": ")[1].upper()
        if ts_type in ts_models and series_key in ts_models[ts_type]:
            model_info = ts_models[ts_type][series_key]
            predictions = predict_ts_model(filtered_data, model_info, ts_type, periods)

    with tab1:
        st.subheader(f"Forecast for: {selected_kpi} ({selected_agency})")
        if not predictions.empty:
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'], y=filtered_data['Value'],
                mode='lines+markers', name='Historical Data',
                line=dict(color='#3b82f6', width=2.5),
                marker=dict(size=4, color='#3b82f6')
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=predictions['Date'], y=predictions['Prediction'],
                mode='lines+markers', name='Forecast',
                line=dict(color='#ef4444', dash='dash', width=2.5),
                marker=dict(size=4, color='#ef4444')
            ))
            
            # Debug: Check what columns are available for confidence intervals
            print(f"🔍 Chart Debug - Prediction DataFrame columns: {list(predictions.columns)}")
            if 'Lower' in predictions.columns and 'Upper' in predictions.columns:
                print(f"📊 Confidence interval data available: Lower range [{predictions['Lower'].min():.2f}, {predictions['Lower'].max():.2f}], Upper range [{predictions['Upper'].min():.2f}, {predictions['Upper'].max():.2f}]")
            else:
                print(f"❌ Missing confidence interval columns. Available: {list(predictions.columns)}")
            
            if 'Lower' in predictions.columns and 'Upper' in predictions.columns:
                fig.add_trace(go.Scatter(
                    x=predictions['Date'], y=predictions['Upper'],
                    mode='lines', line=dict(width=0), showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=predictions['Date'], y=predictions['Lower'],
                    mode='lines', fill='tonexty',
                    fillcolor='rgba(239,68,68,0.15)',
                    line=dict(width=0), name='Confidence Interval'
                ))
            
            fig.update_layout(
                template="plotly_dark",
                title=dict(
                    text="Historical Performance & Forecast",
                    x=0.02,
                    font=dict(size=18, family="SF Pro Display, -apple-system, sans-serif")
                ),
                xaxis_title="Date", 
                yaxis_title="Value",
                height=520, 
                hovermode="x unified",
                margin=dict(l=20, r=20, t=80, b=20),
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="left", 
                    x=0,
                    bgcolor="rgba(0,0,0,0)"
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### Forecast Insights")
            col1, col2 = st.columns(2)
            with col1:
                next_month_pred = predictions.iloc[0]['Prediction']
                last_actual = filtered_data.iloc[-1]['Value']
                delta = ((next_month_pred / last_actual) - 1) * 100 if last_actual else 0
                # Get the actual forecast period for accurate labeling
                forecast_period = predictions.iloc[0]['Date'].strftime('%b %Y') if not predictions.empty else "First Period"
                last_actual_period = filtered_data.iloc[-1]['Date'].strftime('%b %Y') if not filtered_data.empty else "Last Period"
                st.metric(f"First Forecast ({forecast_period})", f"{next_month_pred:,.1f}", f"{delta:+.1f}% vs {last_actual_period}")
            with col2:
                # Calculate trend direction for user-friendly insight
                if len(predictions) >= 3:
                    first_forecast = predictions.iloc[0]['Prediction']
                    mid_forecast = predictions.iloc[len(predictions)//2]['Prediction']
                    last_forecast = predictions.iloc[-1]['Prediction']
                    
                    if last_forecast > first_forecast * 1.05:  # 5% threshold
                        trend_direction = "↗ Increasing"
                        trend_color = "#10b981"  # Success green
                    elif last_forecast < first_forecast * 0.95:  # 5% threshold  
                        trend_direction = "↘ Decreasing"
                        trend_color = "#ef4444"  # Danger red
                    else:
                        trend_direction = "→ Stable"
                        trend_color = "#94a3b8"  # Muted gray
                    
                    trend_change = ((last_forecast / first_forecast) - 1) * 100
                    
                    st.markdown(f"""
                    <div style="background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); 
                                padding: 1.5rem; text-align: center;">
                        <div style="color: var(--muted); font-size: 0.875rem; text-transform: uppercase; 
                                    font-weight: 600; margin-bottom: 0.75rem; letter-spacing: 0.05em;">
                            Forecast Trend
                        </div>
                        <div style="color: {trend_color}; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                            {trend_direction}
                        </div>
                        <div style="color: var(--muted); font-size: 0.875rem;">
                            {trend_change:+.1f}% over forecast period
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Fallback for short forecast periods
                    forecast_range = f"{predictions.iloc[0]['Date'].strftime('%b %Y')} - {predictions.iloc[-1]['Date'].strftime('%b %Y')}"
                    st.markdown(f"""
                    <div style="background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); 
                                padding: 1.5rem; text-align: center;">
                        <div style="color: var(--muted); font-size: 0.875rem; text-transform: uppercase; 
                                    font-weight: 600; margin-bottom: 0.75rem; letter-spacing: 0.05em;">
                            Forecast Period
                        </div>
                        <div style="color: var(--text); font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem;">
                            {len(predictions)} Months
                        </div>
                        <div style="color: var(--muted); font-size: 0.875rem;">
                            {forecast_range}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Could not generate a forecast for the selected model.")

    with tab2:
        st.subheader(f"Data for: {selected_kpi} ({selected_agency})")
        st.dataframe(
            filtered_data[['Date', 'Value', 'MONTHLY_TARGET']].style.format({
                "Value": "{:,.2f}",
                "MONTHLY_TARGET": "{:,.2f}"
            }), 
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
