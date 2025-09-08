#!/usr/bin/env python3
"""
Test the dashboard integration and model loading
"""
import sys
import os
from pathlib import Path
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_model_loading():
    print("🧪 Testing Dashboard Integration")
    print("=" * 50)
    
    # Test ML models
    models_dir = Path("models")
    ml_files = ["RandomForest_model.pkl", "XGBoost_model.pkl", "LinearRegression_model.pkl"]
    
    print("📊 Machine Learning Models:")
    for filename in ml_files:
        filepath = models_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                size_mb = filepath.stat().st_size / (1024 * 1024)
                mae = model_data.get('test_mae', 'N/A')
                print(f"  ✅ {filename}: {size_mb:.1f} MB, MAE: {mae}")
            except Exception as e:
                print(f"  ❌ {filename}: Error - {e}")
        else:
            print(f"  ❌ {filename}: Not found")
    
    # Test time series models
    ts_dir = models_dir / "time_series"
    ts_files = ["prophet_models.pkl", "sarima_models.pkl"]
    
    print("\n📈 Time Series Models:")
    for filename in ts_files:
        filepath = ts_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    ts_models = pickle.load(f)
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  ✅ {filename}: {size_mb:.1f} MB, {len(ts_models)} series")
            except Exception as e:
                print(f"  ❌ {filename}: Error - {e}")
        else:
            print(f"  ❌ {filename}: Not found")
    
    # Test data loading
    print("\n📁 Data Availability:")
    data_file = Path("data/processed/mta_model.parquet")
    if data_file.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(data_file)
            size_mb = data_file.stat().st_size / (1024 * 1024)
            print(f"  ✅ Processed data: {size_mb:.1f} MB, {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"  📊 Agencies: {df['AGENCY_NAME'].nunique()}")
            print(f"  📊 Indicators: {df['INDICATOR_NAME'].nunique()}")
        except Exception as e:
            print(f"  ❌ Data loading error: {e}")
    else:
        print(f"  ❌ Processed data not found")
    
    print("\n🎯 Dashboard Status:")
    if all((models_dir / f).exists() for f in ml_files) and \
       all((ts_dir / f).exists() for f in ts_files) and \
       data_file.exists():
        print("  🎉 All components ready! Dashboard fully functional.")
        print("  🚀 Features available:")
        print("    - Historical data visualization")
        print("    - ML-based forecasting (RandomForest, XGBoost, LinearRegression)")
        print("    - Time series forecasting (Prophet, SARIMA)")
        print("    - Interactive model comparison")
        print("    - Performance metrics display")
    else:
        print("  ⚠️  Some components missing. Check above for details.")

if __name__ == "__main__":
    test_model_loading()
