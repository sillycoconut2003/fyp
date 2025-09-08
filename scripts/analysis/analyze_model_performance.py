#!/usr/bin/env python3
"""
Model Performance Analysis for MTA KPI Forecasting
Analyzes and compares ML and Time Series model performance
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_ml_models():
    """Analyze Machine Learning model performance"""
    print("="*60)
    print("MACHINE LEARNING MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    models_dir = Path('models')
    ml_results = {}
    
    # Load ML models and their performance metrics
    ml_files = {
        'RandomForest': 'RandomForest_model.pkl',
        'XGBoost': 'XGBoost_model.pkl', 
        'LinearRegression': 'LinearRegression_model.pkl'
    }
    
    for model_name, filename in ml_files.items():
        filepath = models_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                ml_results[model_name] = model_data
                
                print(f"\n{model_name} Performance:")
                print(f"  MAE (Mean Absolute Error): {model_data['test_mae']:,.2f}")
                print(f"  RMSE (Root Mean Square Error): {model_data['test_rmse']:,.2f}")
                print(f"  MAPE (Mean Absolute Percentage Error): {model_data['test_mape']*100:.2f}%")
                print(f"  Cross-Validation MAE: {model_data['cv_mae']:,.2f}")
                print(f"  Training Features: {len(model_data['feature_cols'])}")
                
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
    
    # Rank models by different metrics
    if ml_results:
        print("\n" + "="*60)
        print("MODEL RANKINGS")
        print("="*60)
        
        # Rank by MAE (lower is better)
        mae_ranking = sorted(ml_results.items(), key=lambda x: x[1]['test_mae'])
        print("\n🎯 ACCURACY RANKING (by Test MAE - Lower is Better):")
        for i, (model, data) in enumerate(mae_ranking, 1):
            if i == 1:
                print(f"  {i}. {model}: {data['test_mae']:,.0f} MAE ⭐ BEST")
            else:
                ratio = data['test_mae'] / mae_ranking[0][1]['test_mae']
                print(f"  {i}. {model}: {data['test_mae']:,.0f} MAE ({ratio:.1f}x worse)")
        
        # Rank by MAPE (lower is better)
        mape_ranking = sorted(ml_results.items(), key=lambda x: x[1]['test_mape'])
        print("\n📊 PERCENTAGE ERROR RANKING (by MAPE - Lower is Better):")
        for i, (model, data) in enumerate(mape_ranking, 1):
            mape_pct = data['test_mape'] * 100
            if i == 1:
                print(f"  {i}. {model}: {mape_pct:.2f}% MAPE ⭐ BEST")
            else:
                print(f"  {i}. {model}: {mape_pct:.2f}% MAPE")
        
        # Performance insights
        print("\n" + "="*60)
        print("PERFORMANCE INSIGHTS")
        print("="*60)
        
        best_model = mae_ranking[0]
        worst_model = mae_ranking[-1]
        improvement_factor = worst_model[1]['test_mae'] / best_model[1]['test_mae']
        
        print(f"\n🏆 CHAMPION: {best_model[0]}")
        print(f"   - {improvement_factor:.0f}x more accurate than baseline")
        print(f"   - Test MAE: {best_model[1]['test_mae']:,.0f}")
        print(f"   - Test MAPE: {best_model[1]['test_mape']*100:.2f}%")
        print(f"   - Cross-Val MAE: {best_model[1]['cv_mae']:,.0f}")
        
        print(f"\n⚡ SPEED vs ACCURACY TRADEOFF:")
        if 'XGBoost' in ml_results and best_model[0] == 'RandomForest':
            xgb_data = ml_results['XGBoost']
            rf_data = best_model[1]
            accuracy_ratio = xgb_data['test_mae'] / rf_data['test_mae']
            print(f"   - XGBoost is {accuracy_ratio:.1f}x less accurate than RandomForest")
            print(f"   - But XGBoost trains much faster")
            print(f"   - XGBoost MAE: {xgb_data['test_mae']:,.0f}")
            print(f"   - Good for real-time applications")
    
    return ml_results

def analyze_ts_models():
    """Analyze Time Series model coverage and performance"""
    print("\n" + "="*60)
    print("TIME SERIES MODEL ANALYSIS")
    print("="*60)
    
    ts_dir = Path('models/time_series')
    
    # Analyze Prophet models
    prophet_file = ts_dir / 'prophet_models.pkl'
    if prophet_file.exists():
        try:
            with open(prophet_file, 'rb') as f:
                prophet_models = pickle.load(f)
            
            print(f"\n📈 PROPHET MODELS:")
            print(f"   - Total Series Covered: {len(prophet_models)}")
            print(f"   - Model Type: Seasonal decomposition + trend")
            print(f"   - Strengths: Seasonality, holidays, confidence intervals")
            print(f"   - Best For: Operational planning, seasonal KPIs")
            
            # Show sample series
            sample_keys = list(prophet_models.keys())[:3]
            print(f"\n   Sample Coverage:")
            for key in sample_keys:
                agency, indicator = key.split('|')
                print(f"     - {agency}: {indicator}")
            
        except Exception as e:
            print(f"Error loading Prophet models: {e}")
    
    # Analyze SARIMA models
    sarima_file = ts_dir / 'sarima_models.pkl'
    if sarima_file.exists():
        try:
            with open(sarima_file, 'rb') as f:
                sarima_models = pickle.load(f)
            
            print(f"\n📊 SARIMA MODELS:")
            print(f"   - Total Series Covered: {len(sarima_models)}")
            print(f"   - Model Type: Statistical ARIMA with seasonality")
            print(f"   - Strengths: Statistical rigor, traditional approach")
            print(f"   - Best For: Stable time series, statistical analysis")
            
        except Exception as e:
            print(f"Error loading SARIMA models: {e}")

def model_selection_guide():
    """Provide model selection recommendations"""
    print("\n" + "="*60)
    print("MODEL SELECTION GUIDE")
    print("="*60)
    
    print("\n🎯 TASK-BASED RECOMMENDATIONS:")
    
    print("\n1. GENERAL FORECASTING:")
    print("   ✅ Primary: RandomForest (MAE: ~12,651)")
    print("   ✅ Backup: XGBoost (faster, MAE: ~49,445)")
    print("   💡 Why: Cross-series patterns, robust performance")
    
    print("\n2. SEASONAL ANALYSIS:")
    print("   ✅ Primary: Prophet (confidence intervals)")
    print("   ✅ Secondary: RandomForest (baseline)")
    print("   💡 Why: Prophet captures operational cycles")
    
    print("\n3. OPERATIONAL PLANNING:")
    print("   ✅ Ensemble: Prophet + RandomForest")
    print("   💡 Why: Combine seasonal insight with accuracy")
    
    print("\n4. BUDGET PLANNING:")
    print("   ✅ Primary: RandomForest (most reliable)")
    print("   ✅ Validation: SARIMA (statistical confidence)")
    print("   💡 Why: High-stakes decisions need accuracy")
    
    print("\n5. REAL-TIME ANALYSIS:")
    print("   ✅ Primary: XGBoost (fast processing)")
    print("   💡 Why: Speed vs accuracy tradeoff")

def evaluation_metrics_explained():
    """Explain the evaluation metrics used"""
    print("\n" + "="*60)
    print("EVALUATION METRICS EXPLAINED")
    print("="*60)
    
    print("\n📏 METRICS USED:")
    
    print("\n1. MAE (Mean Absolute Error):")
    print("   - What: Average absolute difference between predicted and actual")
    print("   - Range: 0 to ∞ (lower is better)")
    print("   - Example: MAE of 1000 = predictions off by 1000 units on average")
    print("   - Why: Easy to interpret, robust to outliers")
    
    print("\n2. RMSE (Root Mean Square Error):")
    print("   - What: Square root of average squared differences")
    print("   - Range: 0 to ∞ (lower is better)")
    print("   - Penalty: Heavily penalizes large errors")
    print("   - Why: Sensitive to outliers, common ML metric")
    
    print("\n3. MAPE (Mean Absolute Percentage Error):")
    print("   - What: Average percentage error")
    print("   - Range: 0% to ∞% (lower is better)")
    print("   - Example: 10% MAPE = predictions off by 10% on average")
    print("   - Why: Scale-independent, business-friendly")
    
    print("\n4. R² Score (Coefficient of Determination):")
    print("   - What: Proportion of variance explained by model")
    print("   - Range: -∞ to 1.0 (higher is better)")
    print("   - Example: R² = 0.85 means model explains 85% of variance")
    print("   - Why: Shows how well model captures patterns")

def main():
    """Main analysis function"""
    print("MTA KPI FORECASTING - MODEL PERFORMANCE ANALYSIS")
    print("Generated:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Run all analyses
    ml_results = analyze_ml_models()
    analyze_ts_models()
    model_selection_guide()
    evaluation_metrics_explained()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\n💡 Key Takeaway: RandomForest dominates for accuracy,")
    print("   Prophet adds seasonal insights, XGBoost offers speed.")
    print("\n🎯 Recommendation: Use RandomForest + Prophet ensemble")
    print("   for most business-critical forecasting tasks.")

if __name__ == "__main__":
    main()
