#!/usr/bin/env python3
"""
Check model accuracies
"""
import pickle
import os
from pathlib import Path

def check_ml_accuracies():
    print("🎯 MODEL ACCURACY REPORT")
    print("=" * 50)
    print()
    
    print("📊 MACHINE LEARNING MODELS:")
    models_dir = Path("models")
    
    ml_models = {
        "RandomForest": "RandomForest_model.pkl",
        "XGBoost": "XGBoost_model.pkl", 
        "LinearRegression": "LinearRegression_model.pkl"
    }
    
    for model_name, filename in ml_models.items():
        filepath = models_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                print(f"\n{model_name}:")
                
                # Try different possible keys for metrics
                mae = data.get('test_mae', data.get('mae', 'N/A'))
                rmse = data.get('test_rmse', data.get('rmse', 'N/A'))
                mape = data.get('test_mape', data.get('mape', 'N/A'))
                
                if mae != 'N/A':
                    print(f"  ✅ MAE (Mean Absolute Error): {mae:,.2f}")
                if rmse != 'N/A':
                    print(f"  ✅ RMSE (Root Mean Square Error): {rmse:,.2f}")
                if mape != 'N/A':
                    print(f"  ✅ MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
                    
            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")
        else:
            print(f"  ❌ {filename} not found")
    
    print("\n" + "=" * 50)
    print("📈 TIME SERIES MODELS:")
    
    ts_dir = models_dir / "time_series"
    ts_models = {
        "Prophet": "prophet_models.pkl",
        "SARIMA": "sarima_models.pkl"
    }
    
    for model_name, filename in ts_models.items():
        filepath = ts_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    models_dict = pickle.load(f)
                
                print(f"\n{model_name}:")
                print(f"  ✅ Number of trained series: {len(models_dict)}")
                
                # Try to get sample metrics if available
                if models_dict:
                    sample_key = list(models_dict.keys())[0]
                    sample_model = models_dict[sample_key]
                    
                    if isinstance(sample_model, dict) and 'mae' in sample_model:
                        print(f"  📊 Sample series metrics available")
                    else:
                        print(f"  📊 Models trained (metrics calculated during forecasting)")
                        
            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")
        else:
            print(f"  ❌ {filename} not found")

if __name__ == "__main__":
    check_ml_accuracies()
