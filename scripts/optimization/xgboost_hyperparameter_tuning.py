#!/usr/bin/env python3
"""
XGBoost Hyperparameter Tuning for FYP
High-impact optimization with systematic approach
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import time
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import xgboost as xgb

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import mae

def xgboost_hyperparameter_tuning(X_train, y_train, X_test, y_test, method='grid'):
    """
    Comprehensive XGBoost hyperparameter tuning
    
    Args:
        method: 'grid' for GridSearch, 'random' for RandomizedSearch
    """
    
    print(f"\n🚀 XGBOOST HYPERPARAMETER TUNING ({method.upper()})")
    print("="*60)
    
    # Current baseline for comparison
    baseline_model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    # Quick baseline test
    print("📊 BASELINE PERFORMANCE:")
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_mae = mae(y_test, baseline_pred)
    print(f"   Current XGBoost MAE: {baseline_mae:,.0f}")
    
    # Define hyperparameter space
    if method == 'grid':
        param_grid = {
            'n_estimators': [300, 500, 700],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }
        
        # Reduced grid for faster execution (FYP time constraints)
        param_grid_fast = {
            'n_estimators': [400, 600],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.08, 0.1, 0.12],
            'subsample': [0.9, 1.0],
            'colsample_bytree': [0.9, 1.0]
        }
        
        print(f"🔧 Grid Search Parameters:")
        print(f"   Full grid combinations: {np.prod([len(v) for v in param_grid.values()]):,}")
        print(f"   Fast grid combinations: {np.prod([len(v) for v in param_grid_fast.values()]):,}")
        
        # Use fast grid for FYP (reasonable time)
        param_space = param_grid_fast
        
    else:  # random search
        param_space = {
            'n_estimators': [200, 300, 400, 500, 600, 700, 800],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 12],
            'learning_rate': [0.05, 0.08, 0.1, 0.12, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0]
        }
    
    # Time series cross-validation (critical for forecasting)
    tscv = TimeSeriesSplit(n_splits=3)  # Reduced from 5 for speed
    
    # Custom scoring function
    mae_scorer = make_scorer(mae, greater_is_better=False)
    
    # Base estimator
    xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    print(f"\n⏱️ Starting {method} search...")
    start_time = time.time()
    
    if method == 'grid':
        search = GridSearchCV(
            estimator=xgb_base,
            param_grid=param_space,
            cv=tscv,
            scoring=mae_scorer,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
    else:
        search = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=param_space,
            n_iter=50,  # Try 50 random combinations
            cv=tscv,
            scoring=mae_scorer,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True
        )
    
    # Fit the search
    search.fit(X_train, y_train)
    
    search_time = time.time() - start_time
    
    # Get best model
    best_model = search.best_estimator_
    
    # Test on holdout set
    best_pred = best_model.predict(X_test)
    best_mae = mae(y_test, best_pred)
    
    # Calculate improvement
    improvement = (baseline_mae - best_mae) / baseline_mae * 100
    
    # Results summary
    print(f"\n🏆 OPTIMIZATION RESULTS:")
    print("="*40)
    print(f"⏱️ Search time: {search_time/60:.1f} minutes")
    print(f"🔍 Combinations tested: {len(search.cv_results_['mean_test_score'])}")
    print(f"📊 Best CV score: {-search.best_score_:,.0f}")
    print(f"🎯 Best test MAE: {best_mae:,.0f}")
    print(f"📈 Improvement: {improvement:+.1f}% vs baseline")
    
    print(f"\n🔧 OPTIMAL PARAMETERS:")
    print("-" * 25)
    for param, value in search.best_params_.items():
        print(f"   {param}: {value}")
    
    return {
        'best_model': best_model,
        'best_params': search.best_params_,
        'best_score': -search.best_score_,
        'test_mae': best_mae,
        'baseline_mae': baseline_mae,
        'improvement': improvement,
        'search_time': search_time,
        'search_object': search
    }

def analyze_hyperparameter_importance(search_results):
    """Analyze which hyperparameters had the biggest impact"""
    
    print(f"\n📊 HYPERPARAMETER IMPACT ANALYSIS:")
    print("="*45)
    
    search = search_results['search_object']
    results_df = pd.DataFrame(search.cv_results_)
    
    # Get parameter columns
    param_cols = [col for col in results_df.columns if col.startswith('param_')]
    
    print(f"\n🔍 Top 5 parameter combinations:")
    print("-" * 35)
    
    # Sort by test score
    top_5 = results_df.nlargest(5, 'mean_test_score')
    
    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
        score = -row['mean_test_score']
        print(f"\n#{i}: MAE = {score:,.0f}")
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            print(f"     {param_name}: {row[param_col]}")
    
    # Parameter value impact analysis
    print(f"\n🎯 PARAMETER SENSITIVITY:")
    print("-" * 30)
    
    for param_col in param_cols:
        param_name = param_col.replace('param_', '')
        param_impact = results_df.groupby(param_col)['mean_test_score'].agg(['mean', 'std', 'count'])
        param_impact = param_impact.sort_values('mean', ascending=False)
        
        if len(param_impact) > 1:
            best_value = param_impact.index[0]
            worst_value = param_impact.index[-1]
            impact_diff = param_impact.iloc[0]['mean'] - param_impact.iloc[-1]['mean']
            
            print(f"{param_name}:")
            print(f"   Best: {best_value} (MAE: {-param_impact.iloc[0]['mean']:,.0f})")
            print(f"   Worst: {worst_value} (MAE: {-param_impact.iloc[-1]['mean']:,.0f})")
            print(f"   Impact: {impact_diff:.0f} MAE difference")

def create_fyp_optimization_report(search_results):
    """Create a comprehensive report for FYP presentation"""
    
    print(f"\n📋 FYP OPTIMIZATION REPORT")
    print("="*50)
    
    baseline_mae = search_results['baseline_mae']
    optimized_mae = search_results['test_mae']
    improvement = search_results['improvement']
    search_time = search_results['search_time']
    
    print(f"\n🎯 EXECUTIVE SUMMARY:")
    print(f"   Problem: XGBoost underperforming (3x worse than RandomForest)")
    print(f"   Solution: Systematic hyperparameter optimization")
    print(f"   Method: {search_results['search_object'].__class__.__name__} with TimeSeriesSplit")
    print(f"   Time Investment: {search_time/60:.1f} minutes")
    print(f"   Result: {improvement:+.1f}% performance improvement")
    
    print(f"\n📊 QUANTITATIVE RESULTS:")
    print(f"   Baseline MAE: {baseline_mae:,.0f}")
    print(f"   Optimized MAE: {optimized_mae:,.0f}")
    print(f"   Absolute Improvement: {baseline_mae - optimized_mae:,.0f}")
    print(f"   Relative Improvement: {improvement:.1f}%")
    
    if improvement > 20:
        verdict = "🌟 EXCELLENT - Significant improvement achieved"
    elif improvement > 10:
        verdict = "✅ GOOD - Meaningful improvement"
    elif improvement > 5:
        verdict = "🔶 MODERATE - Some improvement"
    else:
        verdict = "🔴 LIMITED - Minimal improvement"
    
    print(f"\n🏆 VERDICT: {verdict}")
    
    print(f"\n🎓 ACADEMIC VALUE:")
    print(f"   • Demonstrates systematic optimization methodology")
    print(f"   • Shows understanding of hyperparameter impact")
    print(f"   • Illustrates proper time series validation")
    print(f"   • Proves practical ML engineering skills")
    
    print(f"\n💡 KEY LEARNINGS:")
    best_params = search_results['best_params']
    print(f"   • Optimal n_estimators: {best_params.get('n_estimators', 'N/A')}")
    print(f"   • Optimal max_depth: {best_params.get('max_depth', 'N/A')}")
    print(f"   • Optimal learning_rate: {best_params.get('learning_rate', 'N/A')}")
    print(f"   • Time series CV is critical for forecasting")

def implement_in_training_pipeline(search_results):
    """Show how to implement optimized parameters in main training"""
    
    print(f"\n🔄 IMPLEMENTATION IN TRAINING PIPELINE")
    print("="*50)
    
    best_params = search_results['best_params']
    
    print(f"\n✅ Replace current XGBoost configuration in train_ml.py:")
    print(f"\n❌ CURRENT (lines ~136-143):")
    print(f"   model = xgb.XGBRegressor(")
    print(f"       n_estimators=400,")
    print(f"       max_depth=6,")
    print(f"       learning_rate=0.1,")
    print(f"       random_state=42,")
    print(f"       n_jobs=-1")
    print(f"   )")
    
    print(f"\n✅ OPTIMIZED (FYP-tuned parameters):")
    print(f"   model = xgb.XGBRegressor(")
    for param, value in best_params.items():
        print(f"       {param}={value},")
    print(f"       random_state=42,")
    print(f"       n_jobs=-1")
    print(f"   )")
    
    improvement = search_results['improvement']
    print(f"\n🎯 Expected result: {improvement:+.1f}% improvement!")

def main():
    """Main hyperparameter tuning pipeline"""
    
    print("🎓 XGBOOST HYPERPARAMETER TUNING FOR FYP")
    print("="*60)
    
    # Load data (same as main training)
    sys.path.insert(0, '.')
    from train_ml import load_processed_data, prepare_ml_features, split_data_temporal
    
    print("📂 Loading data...")
    df = load_processed_data()
    df_clean, feature_cols, target_col = prepare_ml_features(df)
    X_train, X_test, y_train, y_test, _, _ = split_data_temporal(df_clean, feature_cols, target_col)
    
    print(f"✅ Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Choose tuning method
    print(f"\n🤔 TUNING METHOD SELECTION:")
    print(f"   Grid Search: Systematic, thorough, slower (~30-45 minutes)")
    print(f"   Random Search: Efficient, good coverage, faster (~15-20 minutes)")
    
    # For FYP, recommend Random Search for time efficiency
    method = 'random'  # Change to 'grid' if you have more time
    print(f"   🎯 Selected: {method.upper()} (good for FYP time constraints)")
    
    # Run hyperparameter tuning
    results = xgboost_hyperparameter_tuning(X_train, y_train, X_test, y_test, method=method)
    
    # Analysis
    analyze_hyperparameter_importance(results)
    create_fyp_optimization_report(results)
    implement_in_training_pipeline(results)
    
    return results

if __name__ == "__main__":
    results = main()
