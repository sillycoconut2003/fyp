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

# Import specialized percentage prediction
from percentage_predictor import (
    should_use_specialized_percentage_prediction,
    predict_percentage_kpi_specialized
)

st.set_page_config(page_title="MTA KPI Analytics & Forecasting", layout="wide", initial_sidebar_state="expanded")

def load_css():
    """Load custom CSS for the futuristic dashboard theme"""
    css_file = Path(__file__).parent / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
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

def simple_trend_forecast(df, periods=12):
    """Simple fallback forecasting using linear trend"""
    values = df['MONTHLY_ACTUAL'].values
    dates = pd.to_datetime(df['YYYY_MM'])
    
    # Calculate linear trend
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    
    # Generate future predictions
    predictions = []
    for i in range(periods):
        future_x = len(values) + i
        pred = slope * future_x + intercept
        # Add some realistic variation
        pred += np.random.normal(0, np.std(values) * 0.05)
        predictions.append(max(0, pred))  # Ensure non-negative
    
    return predictions

def predict_ml_model(df, model_info, model_name, periods=12, kpi_name=""):
    """Make predictions with a feature-based ML model using iterative forecasting."""
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
        
        # Check if this is a percentage KPI that needs specialized handling
        recent_actuals = df_extended['MONTHLY_ACTUAL'].iloc[-12:].values
        if should_use_specialized_percentage_prediction(kpi_name, recent_actuals):
            print(f"🎯 Using specialized percentage prediction for {kpi_name}")
            return predict_percentage_kpi_specialized(
                df_extended, kpi_name, model_name, model, periods
            )
        
        # Original prediction logic for non-percentage KPIs continues below
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
            
            # Update lag features conservatively
            if i == 0:
                last_value = df_extended['MONTHLY_ACTUAL'].iloc[-1]
            else:
                last_value = predictions[i-1]
            
            # Update common lag features if they exist
            for lag_col in ['m_act_lag1', 'lag_1', 'MONTHLY_ACTUAL_lag1']:
                if lag_col in future_row.columns:
                    future_row[lag_col] = last_value
            
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
                
                # Apply bounds checking
                if (np.isnan(raw_pred) or np.isinf(raw_pred) or 
                    raw_pred > upper_bound or raw_pred < lower_bound):
                    
                    print(f"⚠️ {model_name} extreme prediction: {raw_pred:.2f} -> using conservative estimate")
                    # Use conservative prediction based on recent trend
                    recent_trend = np.mean(df_extended['MONTHLY_ACTUAL'].iloc[-6:])
                    final_pred = recent_trend
                else:
                    final_pred = raw_pred
                
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
        
        return pd.DataFrame({
            'Date': dates,
            'Prediction': predictions,
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
            
            return pd.DataFrame({
                'Date': forecast['ds'],
                'Prediction': forecast['yhat'].clip(lower=0),  # Ensure non-negative
                'Lower': forecast['yhat_lower'].clip(lower=0) if 'yhat_lower' in forecast.columns else forecast['yhat'] * 0.9,
                'Upper': forecast['yhat_upper'] if 'yhat_upper' in forecast.columns else forecast['yhat'] * 1.1,
                'Model': model_name
            })
            
        elif 'SARIMA' in model_name.upper():
            # SARIMA prediction logic
            last_date = pd.to_datetime(df['YYYY_MM']).max()
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1), 
                periods=periods, 
                freq='MS'
            )
            
            # Generate SARIMA forecasts
            forecast = model.forecast(steps=periods)
            
            # Handle case where forecast might be a pandas Series or array
            if hasattr(forecast, 'values'):
                predictions = forecast.values
            else:
                predictions = np.array(forecast)
            
            # Ensure non-negative predictions
            predictions = np.maximum(predictions, 0)
            
            return pd.DataFrame({
                'Date': future_dates,
                'Prediction': predictions,
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
            
            return pd.DataFrame({
                'Date': future_dates,
                'Prediction': np.maximum(predictions, 0),
                'Model': model_name
            })
    
    except Exception as e:
        st.error(f"Error in {model_name} prediction: {e}")
        print(f"Debug - Model type: {type(model_info)}, Model name: {model_name}")
        if isinstance(model_info, dict):
            print(f"Debug - Model info keys: {list(model_info.keys())}")
        return pd.DataFrame()

def main():
    # --- Futuristic Header ---
    st.markdown("""
        <div class="header">
            <h1 class="header-title">🚇 MTA KPI Analytics & Forecasting</h1>
            <p class="header-subtitle">FYP 2025: Optimized ML & Time Series Modeling</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Performance Metrics ---
    st.markdown("### 🚀 Model Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏆 Best Model", "RandomForest", "13,637 MAE")
    with col2:
        st.metric("🥈 XGBoost", "39,885 MAE", "+192% vs RF")
    with col3:
        st.metric("🥉 Ridge Regression", "96,886 MAE", "+610% vs RF")
    with col4:
        st.metric("📈 Total Models", "267", "3 ML + 264 TS")

    st.markdown("---")
    
    # --- Load Data & Models ---
    df = load_data()
    if df.empty:
        st.error("❌ Processed dataset is empty. Please run the data processing pipeline.")
        return
    
    ml_models = load_ml_models()
    ts_models = load_ts_models()
    
    # --- Sidebar Controls ---
    st.sidebar.markdown("## ⚙️ Forecast Controls")
    
    # Cache refresh button for troubleshooting
    if st.sidebar.button("🔄 Refresh Models", help="Clear cache and reload models if predictions seem incorrect"):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared! Models will reload on next prediction.")
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Data Selection Section
    st.sidebar.markdown("### 📊 Data Selection")
    # Clean and sort KPI names properly
    kpi_names = df['INDICATOR_NAME'].unique()
    # Remove leading/trailing spaces and sort alphabetically
    available_kpis = sorted([kpi.strip() for kpi in kpi_names])
    selected_kpi_clean = st.sidebar.selectbox(
        "Select KPI", 
        available_kpis, 
        index=0,
        help="Choose the Key Performance Indicator to forecast"
    )
    # Find the original KPI name (with potential spaces) for data filtering
    selected_kpi = next(kpi for kpi in kpi_names if kpi.strip() == selected_kpi_clean)
    
    kpi_data = df[df['INDICATOR_NAME'] == selected_kpi]
    available_agencies = sorted(kpi_data['AGENCY_NAME'].unique())
    
    # Smart agency selection: only show dropdown if multiple agencies available
    if len(available_agencies) == 1:
        selected_agency = available_agencies[0]
        st.sidebar.info(f"📍 **Agency:** {selected_agency}")
    else:
        selected_agency = st.sidebar.selectbox(
            "Select Agency", 
            available_agencies, 
            index=0,
            help=f"Choose from {len(available_agencies)} agencies that have this KPI"
        )
    
    st.sidebar.markdown("---")
    
    # Model Selection Section
    st.sidebar.markdown("### 🤖 Model Selection")
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
        "Select Model", 
        model_options,
        help="Choose between Machine Learning (ML) and Time Series (TS) models"
    )
    
    st.sidebar.markdown("---")
    
    # Forecast Settings Section
    st.sidebar.markdown("### 🔮 Forecast Settings")
    periods = st.sidebar.slider(
        "Forecast Periods (months)", 
        min_value=1, 
        max_value=36, 
        value=12,
        help="Number of months to forecast into the future"
    )

    # --- Main Content Area ---
    tab1, tab2 = st.tabs(["� KPI Forecast", "🗃️ Data Explorer"])

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
                line=dict(color='#00A2FF', width=2),
                marker=dict(size=5)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=predictions['Date'], y=predictions['Prediction'],
                mode='lines+markers', name='Forecast',
                line=dict(color='#FF4B4B', dash='dash', width=2),
                marker=dict(size=5)
            ))
            
            if 'Lower' in predictions.columns and 'Upper' in predictions.columns:
                fig.add_trace(go.Scatter(
                    x=predictions['Date'], y=predictions['Upper'],
                    mode='lines', line=dict(width=0), showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=predictions['Date'], y=predictions['Lower'],
                    mode='lines', fill='tonexty',
                    fillcolor='rgba(255, 75, 75, 0.2)',
                    line=dict(width=0), name='Confidence Interval'
                ))
            
            fig.update_layout(
                template="plotly_dark",
                title=f"Forecast vs. Historical Data",
                xaxis_title="Date", yaxis_title="Value",
                height=500, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📈 Forecast Summary")
            col1, col2 = st.columns(2)
            with col1:
                next_month_pred = predictions.iloc[0]['Prediction']
                last_actual = filtered_data.iloc[-1]['Value']
                delta = ((next_month_pred / last_actual) - 1) * 100 if last_actual else 0
                st.metric("Next Month Forecast", f"{next_month_pred:,.2f}", f"{delta:+.1f}% vs Last")
            with col2:
                avg_forecast = predictions['Prediction'].mean()
                avg_actual = filtered_data['Value'].mean()
                delta_avg = ((avg_forecast / avg_actual) - 1) * 100 if avg_actual else 0
                st.metric("Average Forecast Value", f"{avg_forecast:,.2f}", f"{delta_avg:+.1f}% vs Hist. Avg")

        else:
            st.warning("Could not generate a forecast for the selected model.")

    with tab2:
        st.subheader(f"Data for: {selected_kpi} ({selected_agency})")
        st.dataframe(filtered_data[['Date', 'Value', 'MONTHLY_TARGET']].style.format({
            "Value": "{:,.2f}",
            "MONTHLY_TARGET": "{:,.2f}"
        }), use_container_width=True)


if __name__ == "__main__":
    main()
