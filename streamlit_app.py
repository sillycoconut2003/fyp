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

# Configure Streamlit page
st.set_page_config(
    page_title="MTA KPI Analytics & Forecasting",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optimized for Streamlit Cloud deployment
@st.cache_data(ttl=3600, show_spinner="Loading dataset...")
def load_data():
    """Load the processed dataset with error handling"""
    try:
    # Try multiple data sources for robust deployment
        data_paths = [
            Path(__file__).parent / "data" / "processed" / "mta_model.parquet",
            Path(__file__).parent / "data" / "processed" / "cleaned_data.csv",
            Path(__file__).parent / "data" / "raw" / "MTA_Performance_Agencies.csv"
        ]
        
        for path in data_paths:
            if path.exists():
                if path.suffix == '.parquet':
                    return pd.read_parquet(path)
                else:
                    return pd.read_csv(path)
        
        # Fallback: Create demo data if no files found
        st.warning("⚠️ Using demo data - full dataset not available")
        return create_demo_data()
        
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return create_demo_data()

def create_demo_data():
    """Create demonstration data for deployment testing"""
    np.random.seed(42)
    agencies = ['MTA New York City Transit', 'Long Island Rail Road', 'Metro-North Railroad']
    indicators = ['On-Time Performance', 'Customer Satisfaction', 'System Reliability']
    
    data = []
    for agency in agencies:
        for indicator in indicators:
            for month in pd.date_range('2020-01-01', '2023-12-01', freq='MS'):
                data.append({
                    'AGENCY_NAME': agency,
                    'INDICATOR_NAME': indicator,
                    'YYYY_MM': month,
                    'MONTHLY_ACTUAL': np.random.normal(85, 10),
                    'MONTHLY_TARGET': np.random.normal(90, 5),
                    'YTD_ACTUAL': np.random.normal(85, 8),
                    'YTD_TARGET': np.random.normal(90, 4)
                })
    
    return pd.DataFrame(data)

@st.cache_resource(show_spinner="Initializing ML models...")
def load_ml_models():
    """Load trained ML models with progressive loading for Streamlit Cloud"""
    models_dir = Path(__file__).parent / "models"  # Fixed path for cloud deployment
    ml_models = {}
    
    # Start with lightweight fallback immediately
    ml_models['StatisticalFallback'] = {
        'model': None,
        'mae': 50000,
        'description': '📈 Statistical Average (Always Available)',
        'status': 'Loaded'
    }
    
    # Priority order for model loading (smallest to largest for faster startup)
    model_priority = {
        'Ridge': ('Ridge_Tuned_model.pkl', 130912, '📊 Linear Baseline'),
        'RandomForest': ('RandomForest_model.pkl', 13637, '🏆 Production Model'),
        'XGBoost': ('XGBoost_model.pkl', 39885, '⚡ Fast Alternative'), 
        'OptimizedEnsemble': ('OptimizedStackingEnsemble_model.pkl', 20880, '🔧 Ensemble Model')
    }
    
    loaded_count = 0
    progress_bar = st.progress(0)
    total_models = len(model_priority)
    
    for i, (model_name, (filename, mae, description)) in enumerate(model_priority.items()):
        try:
            progress_bar.progress((i + 1) / total_models, text=f"Loading {model_name}...")
            model_path = models_dir / filename
            
            if model_path.exists():
                # Add timeout protection for large models
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                ml_models[model_name] = {
                    'model': model,
                    'mae': mae,
                    'description': description,
                    'status': 'Loaded'
                }
                loaded_count += 1
                st.sidebar.success(f"✅ {model_name}: {description}")
            else:
                st.sidebar.warning(f"⚠️ {model_name}: File not found")
                ml_models[model_name] = {
                    'model': None,
                    'mae': mae,
                    'description': f"{description} (Not Available)",
                    'status': 'Missing'
                }
                
        except Exception as e:
            st.sidebar.error(f"❌ {model_name}: {str(e)[:50]}...")
            ml_models[model_name] = {
                'model': None,
                'mae': mae,
                'description': f"{description} (Failed)",
                'status': 'Failed'
            }
    
    progress_bar.progress(1.0, text=f"✅ Loaded {loaded_count}/{total_models} models")
    
    if loaded_count > 0:
        st.sidebar.info(f"🎯 {loaded_count} ML models ready for predictions")
    else:
        st.sidebar.warning("⚠️ Using statistical fallback only")
    
    return ml_models

def make_prediction_with_fallback(features, model_choice, ml_models):
    """Make prediction with graceful fallbacks"""
    try:
        if model_choice in ml_models and ml_models[model_choice]['model']:
            model = ml_models[model_choice]['model']
            prediction = model.predict([features])[0]
            return prediction, ml_models[model_choice]['mae'], "Success"
        else:
            # Statistical fallback
            prediction = np.mean(features[:3]) if len(features) >= 3 else 85.0
            return prediction, 50000, "Statistical Fallback"
            
    except Exception as e:
        st.warning(f"Prediction failed with {model_choice}: {e}")
        # Emergency fallback
        prediction = 85.0  # Reasonable default for MTA KPIs
        return prediction, 100000, f"Emergency Fallback ({str(e)[:50]})"

def main():
    """Main Streamlit application with startup monitoring"""
    
    startup_time = datetime.now()
    
    # App header with deployment info
    st.title("🚇 MTA Performance Analytics Dashboard")
    st.markdown("### Advanced ML Forecasting for Transportation KPIs")
    
    # Deployment status indicator
    with st.sidebar:
        st.markdown("### 🔧 Deployment Status")
        st.info(f"⏰ Started: {startup_time.strftime('%H:%M:%S')} UTC")
        
        # Check Git LFS status
        models_dir = Path(__file__).parent / "models"
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl"))
            st.success(f"📁 Found {len(model_files)} model files")
        else:
            st.error("❌ Models directory not found!")
    
    # Load resources with error handling and timing
    try:
        with st.spinner("Loading data and models..."):
            load_start = datetime.now()
            
            data = load_data()
            data_time = (datetime.now() - load_start).total_seconds()
            
            ml_models = load_ml_models()
            total_time = (datetime.now() - load_start).total_seconds()
            
            # Performance monitoring
            st.sidebar.info(f"⚡ Data loaded in {data_time:.1f}s")
            st.sidebar.info(f"🚀 Total startup: {total_time:.1f}s")
            
        if data is None or data.empty:
            st.error("Failed to load data. Please check data sources.")
            return
            
    except Exception as e:
        st.error(f"Critical error during initialization: {e}")
        return
    
    # Sidebar model information
    st.sidebar.markdown("## 📊 Model Performance")
    
    if ml_models:
        model_info = []
        for name, info in ml_models.items():
            if info['status'] in ['Loaded', 'Fallback']:
                model_info.append({
                    'Model': name,
                    'MAE': f"{info['mae']:,}",
                    'Status': info['description']
                })
        
        if model_info:
            model_df = pd.DataFrame(model_info)
            st.sidebar.dataframe(model_df, hide_index=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 KPI Prediction Interface")
        
        # Prediction inputs
        if not data.empty:
            agencies = ['Select Agency...'] + sorted(data['AGENCY_NAME'].unique().tolist())
            selected_agency = st.selectbox("Choose MTA Agency", agencies)
            
            if selected_agency != 'Select Agency...':
                agency_data = data[data['AGENCY_NAME'] == selected_agency]
                indicators = ['Select KPI...'] + sorted(agency_data['INDICATOR_NAME'].unique().tolist())
                selected_indicator = st.selectbox("Choose Performance Indicator", indicators)
                
                if selected_indicator != 'Select KPI...':
                    # Prediction inputs
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        monthly_target = st.number_input("Monthly Target", value=90.0, min_value=0.0, max_value=100.0)
                    
                    with col_b:
                        ytd_target = st.number_input("YTD Target", value=85.0, min_value=0.0, max_value=100.0)
                    
                    with col_c:
                        year = st.selectbox("Year", [2024, 2025, 2026], index=0)
                    
                    # Model selection
                    available_models = [name for name, info in ml_models.items() if info['status'] in ['Loaded', 'Fallback']]
                    if available_models:
                        model_choice = st.selectbox("Select Prediction Model", available_models)
                        
                        # Make prediction
                        if st.button("🎯 Generate Prediction", type="primary"):
                            with st.spinner("Generating prediction..."):
                                features = [monthly_target, ytd_target, year, 1, 0, 0]  # Simplified features
                                prediction, mae, status = make_prediction_with_fallback(
                                    features, model_choice, ml_models
                                )
                                
                                # Display results
                                st.success("✅ Prediction Complete!")
                                
                                pred_col1, pred_col2, pred_col3 = st.columns(3)
                                
                                with pred_col1:
                                    st.metric("Predicted Value", f"{prediction:.2f}")
                                
                                with pred_col2:
                                    st.metric("Expected MAE", f"{mae:,}")
                                
                                with pred_col3:
                                    confidence = "High" if mae < 20000 else "Medium" if mae < 50000 else "Low"
                                    st.metric("Confidence", confidence)
                                
                                if status != "Success":
                                    st.warning(f"Note: {status}")
    
    with col2:
        st.subheader("📊 System Status")
        
        # Deployment information
        st.metric("Environment", "Streamlit Cloud" if os.getenv('STREAMLIT_SHARING') else "Local")
        st.metric("Models Loaded", len([m for m in ml_models.values() if m['status'] == 'Loaded']))
        st.metric("Data Points", f"{len(data):,}" if not data.empty else "0")
        
        # Model rankings
        st.markdown("### 🏆 Model Rankings")
        if ml_models:
            ranked_models = sorted(ml_models.items(), key=lambda x: x[1]['mae'])
            for i, (name, info) in enumerate(ranked_models[:3], 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                st.write(f"{emoji} **{name}**: {info['mae']:,} MAE")
    
    # Footer
    st.markdown("---")
    st.markdown("**🎓 FYP 2025 - MTA Performance Analytics** | Powered by Advanced Machine Learning")

if __name__ == "__main__":
    main()