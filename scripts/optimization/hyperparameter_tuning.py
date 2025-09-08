#!/usr/bin/env python3
"""
Hyperparameter Tuning for RandomForest Model
Experiment with different parameter combinations to optimize performance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import time

def load_data():
    """Load the processed dataset"""
    df = pd.read_parquet('data/processed/mta_model.parquet')
    
    # Prepare features (same as in training)
    feature_cols = [c for c in df.columns if c not in ['MONTHLY_ACTUAL', 'YYYY_MM']]
    target_col = 'MONTHLY_ACTUAL'
    
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y

def build_pipeline(n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """Build RandomForest pipeline with configurable parameters"""
    
    # Identify categorical and numerical columns
    cat = ["AGENCY_NAME", "INDICATOR_NAME"]
    num = [c for c in X.columns if c not in cat]
    
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat),
         ("num", "passthrough", num)]
    )
    
    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        ))
    ])
    
    return model

def test_n_estimators():
    """Test different numbers of trees"""
    print("🌳 TESTING N_ESTIMATORS (Number of Trees)")
    print("="*50)
    
    n_estimators_options = [100, 200, 300, 400, 500, 700, 1000]
    results = []
    
    for n_est in n_estimators_options:
        print(f"\nTesting {n_est} trees...")
        start_time = time.time()
        
        model = build_pipeline(n_estimators=n_est)
        tscv = TimeSeriesSplit(n_splits=5)
        scores = -cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
        
        train_time = time.time() - start_time
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        results.append({
            'n_estimators': n_est,
            'mae': avg_score,
            'std': std_score,
            'time': train_time
        })
        
        print(f"  MAE: {avg_score:,.0f} (±{std_score:,.0f})")
        print(f"  Time: {train_time:.1f}s")
    
    # Find best
    best = min(results, key=lambda x: x['mae'])
    print(f"\n🏆 BEST: {best['n_estimators']} trees")
    print(f"  MAE: {best['mae']:,.0f}")
    print(f"  Time: {best['time']:.1f}s")
    
    return results

def test_cv_splits():
    """Test different numbers of CV splits"""
    print("\n🔀 TESTING CV SPLITS")
    print("="*50)
    
    splits_options = [3, 5, 7, 10]
    results = []
    
    for n_splits in splits_options:
        print(f"\nTesting {n_splits} CV splits...")
        start_time = time.time()
        
        model = build_pipeline(n_estimators=400)  # Use current best
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = -cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
        
        train_time = time.time() - start_time
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        results.append({
            'n_splits': n_splits,
            'mae': avg_score,
            'std': std_score,
            'time': train_time
        })
        
        print(f"  MAE: {avg_score:,.0f} (±{std_score:,.0f})")
        print(f"  Time: {train_time:.1f}s")
    
    # Find best
    best = min(results, key=lambda x: x['mae'])
    print(f"\n🏆 BEST: {best['n_splits']} splits")
    print(f"  MAE: {best['mae']:,.0f}")
    
    return results

def test_tree_parameters():
    """Test tree structure parameters"""
    print("\n🌲 TESTING TREE STRUCTURE PARAMETERS")
    print("="*50)
    
    param_combinations = [
        {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'name': 'Default'},
        {'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'name': 'Limited Depth'},
        {'max_depth': None, 'min_samples_split': 5, 'min_samples_leaf': 2, 'name': 'Conservative'},
        {'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 1, 'name': 'Balanced'},
        {'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 5, 'name': 'Very Conservative'},
    ]
    
    results = []
    
    for params in param_combinations:
        print(f"\nTesting {params['name']}...")
        start_time = time.time()
        
        model = build_pipeline(
            n_estimators=400,
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf']
        )
        
        tscv = TimeSeriesSplit(n_splits=5)
        scores = -cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
        
        train_time = time.time() - start_time
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        results.append({
            'name': params['name'],
            'mae': avg_score,
            'std': std_score,
            'time': train_time,
            'params': params
        })
        
        print(f"  MAE: {avg_score:,.0f} (±{std_score:,.0f})")
        print(f"  Time: {train_time:.1f}s")
    
    # Find best
    best = min(results, key=lambda x: x['mae'])
    print(f"\n🏆 BEST: {best['name']}")
    print(f"  MAE: {best['mae']:,.0f}")
    print(f"  Parameters: {best['params']}")
    
    return results

def compare_with_current():
    """Compare with current model performance"""
    print("\n📊 COMPARISON WITH CURRENT MODEL")
    print("="*50)
    
    # Current model
    current_model = build_pipeline(n_estimators=400)
    tscv = TimeSeriesSplit(n_splits=5)
    current_scores = -cross_val_score(current_model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
    current_mae = np.mean(current_scores)
    
    print(f"Current model MAE: {current_mae:,.0f}")
    print(f"Published MAE: 14,095")
    print(f"Difference: {abs(current_mae - 14095):,.0f}")

def main():
    """Run hyperparameter tuning experiments"""
    global X, y
    
    print("🔧 RANDOMFOREST HYPERPARAMETER TUNING")
    print("="*60)
    
    # Load data
    print("Loading data...")
    X, y = load_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Test current model
    compare_with_current()
    
    # Run experiments
    tree_results = test_n_estimators()
    cv_results = test_cv_splits()
    param_results = test_tree_parameters()
    
    print("\n" + "="*60)
    print("🎯 OPTIMIZATION SUMMARY")
    print("="*60)
    
    print("\n💡 RECOMMENDATIONS:")
    print(f"1. Best n_estimators: {min(tree_results, key=lambda x: x['mae'])['n_estimators']}")
    print(f"2. Best CV splits: {min(cv_results, key=lambda x: x['mae'])['n_splits']}")
    print(f"3. Best tree config: {min(param_results, key=lambda x: x['mae'])['name']}")
    
    current_mae = 14095  # Your published result
    best_tree_mae = min(tree_results, key=lambda x: x['mae'])['mae']
    improvement = current_mae - best_tree_mae
    
    if improvement > 0:
        print(f"\n🚀 POTENTIAL IMPROVEMENT: {improvement:,.0f} MAE reduction")
        print(f"   ({improvement/current_mae*100:.1f}% better accuracy)")
    else:
        print(f"\n✅ CURRENT PARAMETERS ARE OPTIMAL")
        print("   No significant improvement found")

if __name__ == "__main__":
    main()
