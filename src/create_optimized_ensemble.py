#!/usr/bin/env python3
"""
Create an optimized stacking ensemble that excludes Ridge regression
Only uses the two best performing models: RandomForest and XGBoost
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def create_optimized_ensemble():
    """Create a new stacking ensemble with only RandomForest and XGBoost"""
    print("🔬 CREATING OPTIMIZED STACKING ENSEMBLE")
    print("=" * 60)
    print("Strategy: Exclude Ridge, use only top 2 performers")
    print("Base learners: RandomForest + XGBoost")
    print("Meta learner: LinearRegression")
    print()
    
    # Load data
    data_path = Path(__file__).parent / "data" / "processed" / "mta_model.parquet"
    df = pd.read_parquet(data_path)
    
    # Prepare features (same as training pipeline)
    exclude_cols = [
        'YYYY_MM', 'AGENCY_NAME', 'INDICATOR_NAME', 'MONTHLY_ACTUAL',
        'period_start', 'period_end', 'Date', 'DESCRIPTION', 'CATEGORY',
        'DESIRED_CHANGE', 'INDICATOR_UNIT'
    ]
    
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    
    print(f"✅ Using {len(feature_cols)} features for training")
    
    # Split data (same as training pipeline)
    df_sorted = df.sort_values('YYYY_MM')
    split_date = '2015-09-01'
    train_data = df_sorted[df_sorted['YYYY_MM'] < split_date]
    test_data = df_sorted[df_sorted['YYYY_MM'] >= split_date]
    
    X_train = train_data[feature_cols]
    y_train = train_data['MONTHLY_ACTUAL']
    X_test = test_data[feature_cols]
    y_test = test_data['MONTHLY_ACTUAL']
    
    print(f"📊 Data split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Load the two best performing tuned models
    models_dir = Path(__file__).parent / "models"
    
    print("\n🔄 Loading base learners...")
    
    # Load RandomForest
    rf_path = models_dir / "RandomForest_Tuned_model.pkl"
    with open(rf_path, 'rb') as f:
        rf_data = pickle.load(f)
        rf_model = rf_data['model']
        print(f"✅ RandomForest loaded - Test MAE: {rf_data['test_mae']:,.0f}")
    
    # Load XGBoost  
    xgb_path = models_dir / "XGBoost_Tuned_model.pkl"
    with open(xgb_path, 'rb') as f:
        xgb_data = pickle.load(f)
        xgb_model = xgb_data['model']
        print(f"✅ XGBoost loaded - Test MAE: {xgb_data['test_mae']:,.0f}")
    
    # Create optimized stacking ensemble
    print("\n🏗️ Building optimized stacking ensemble...")
    
    base_learners = [
        ('randomforest', rf_model),
        ('xgboost', xgb_model)
    ]
    
    # Use LinearRegression as meta-learner (simple and effective)
    meta_learner = LinearRegression()
    
    optimized_ensemble = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,  # 5-fold cross-validation for stacking
        n_jobs=-1
    )
    
    # Train the ensemble
    print("🔧 Training optimized ensemble...")
    optimized_ensemble.fit(X_train, y_train)
    
    # Evaluate performance
    print("\n📈 PERFORMANCE EVALUATION")
    print("=" * 40)
    
    # Cross-validation performance
    cv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        optimized_ensemble, X_train, y_train, 
        cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"Cross-validation MAE: {cv_mae:,.0f} (±{cv_std:,.0f})")
    
    # Test set performance
    y_pred = optimized_ensemble.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    print(f"Test MAE: {test_mae:,.0f}")
    print(f"Test RMSE: {test_rmse:,.0f}")
    print(f"Test MAPE: {test_mape:.1f}%")
    
    # Compare with original ensemble
    original_path = models_dir / "StackingEnsemble_model.pkl"
    if original_path.exists():
        with open(original_path, 'rb') as f:
            original_data = pickle.load(f)
            original_test_mae = original_data['test_mae']
            improvement = original_test_mae - test_mae
            improvement_pct = (improvement / original_test_mae) * 100
            
            print(f"\n🏆 IMPROVEMENT vs ORIGINAL ENSEMBLE")
            print(f"Original (3-model) MAE: {original_test_mae:,.0f}")
            print(f"Optimized (2-model) MAE: {test_mae:,.0f}")
            print(f"Improvement: {improvement:,.0f} ({improvement_pct:+.1f}%)")
    
    # Compare with best individual model (RandomForest)
    rf_improvement = rf_data['test_mae'] - test_mae
    rf_improvement_pct = (rf_improvement / rf_data['test_mae']) * 100
    
    print(f"\n🥇 vs BEST INDIVIDUAL MODEL (RandomForest)")
    print(f"RandomForest MAE: {rf_data['test_mae']:,.0f}")
    print(f"Optimized Ensemble MAE: {test_mae:,.0f}")
    if rf_improvement > 0:
        print(f"Improvement: {rf_improvement:,.0f} ({rf_improvement_pct:+.1f}%)")
        print("🎉 SUCCESS: Ensemble beats best individual model!")
    else:
        print(f"Difference: {rf_improvement:,.0f} ({rf_improvement_pct:+.1f}%)")
        print("📝 Note: RandomForest still performs better individually")
    
    # Save the optimized ensemble
    model_data = {
        'model': optimized_ensemble,
        'feature_cols': feature_cols,
        'cv_mae': cv_mae,
        'cv_std': cv_std,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mape': test_mape,
        'model_type': 'ML',
        'model_name': 'OptimizedStackingEnsemble',
        'num_features': len(feature_cols),
        'base_learners': ['RandomForest_Tuned', 'XGBoost_Tuned'],
        'meta_learner': 'LinearRegression'
    }
    
    output_path = models_dir / "OptimizedStackingEnsemble_model.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n💾 Saved optimized ensemble to: {output_path}")
    
    return {
        'model': optimized_ensemble,
        'test_mae': test_mae,
        'cv_mae': cv_mae,
        'improvement_vs_original': improvement if original_path.exists() else None,
        'improvement_vs_rf': rf_improvement
    }

if __name__ == "__main__":
    results = create_optimized_ensemble()
    
    print(f"\n🎯 SUMMARY")
    print("=" * 30)
    print(f"✅ Created OptimizedStackingEnsemble")
    print(f"📊 Test MAE: {results['test_mae']:,.0f}")
    print(f"🚀 Ready for dashboard integration!")