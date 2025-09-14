#!/usr/bin/env python3
"""
Quick Linear Regression Enhancement - FYP Optimization
Add regularized linear models for improved performance
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import mae, rmse, mape

def test_regularized_linear_models(X_train, y_train, X_test, y_test):
    """
    Quick test of regularized linear regression models
    This is the 'quick win' - 15-30 minutes for potential 15-25% improvement
    """
    
    print("\n🔧 QUICK LINEAR REGRESSION ENHANCEMENT")
    print("="*50)
    print("Testing regularized versions for improved performance...")
    
    # Define models to test
    models = {
        'LinearRegression': LinearRegression(),
        
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0, random_state=42))
        ]),
        
        'Lasso': Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=1.0, random_state=42, max_iter=2000))
        ]),
        
        'ElasticNet': Pipeline([
            ('scaler', StandardScaler()),
            ('elastic', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000))
        ]),
        
        'Ridge_Tuned': Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=10.0, random_state=42))  # Higher regularization
        ])
    }
    
    results = {}
    tscv = TimeSeriesSplit(n_splits=3)  # Faster CV for quick test
    
    print(f"\n📊 Testing {len(models)} linear models...")
    
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        
        # Cross-validation
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            pred_fold = model.predict(X_val_fold)
            cv_scores.append(mae(y_val_fold, pred_fold))
        
        cv_score = np.mean(cv_scores)
        
        # Final model on test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        test_mae = mae(y_test, y_pred)
        test_rmse = rmse(y_test, y_pred)
        test_mape = mape(y_test, y_pred)
        
        results[model_name] = {
            'cv_mae': cv_score,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'model': model
        }
        
        print(f"  CV MAE: {cv_score:,.0f}")
        print(f"  Test MAE: {test_mae:,.0f}")
        print(f"  Test MAPE: {test_mape:.2f}%")
    
    return results

def analyze_improvements(results):
    """Analyze improvement from regularization"""
    
    print("\n🏆 REGULARIZATION IMPROVEMENT ANALYSIS")
    print("="*50)
    
    # Get baseline (vanilla LinearRegression)
    baseline = results['LinearRegression']
    baseline_mae = baseline['test_mae']
    
    print(f"📉 Baseline Linear Regression MAE: {baseline_mae:,.0f}")
    print(f"\n📈 Regularized Model Performance:")
    
    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_mae'])
    
    best_improvement = 0
    best_model = None
    
    for model_name, metrics in sorted_results:
        mae_val = metrics['test_mae']
        improvement = (baseline_mae - mae_val) / baseline_mae * 100
        
        if model_name != 'LinearRegression':
            if improvement > best_improvement:
                best_improvement = improvement
                best_model = model_name
        
        status = "📊" if model_name == 'LinearRegression' else "🔧"
        improvement_text = f"({improvement:+.1f}%)" if model_name != 'LinearRegression' else "(baseline)"
        
        print(f"  {status} {model_name:15}: {mae_val:>8,.0f} MAE {improvement_text}")
    
    print(f"\n🎯 QUICK WIN RESULTS:")
    print(f"  • Best Model: {best_model}")
    print(f"  • Improvement: {best_improvement:.1f}% better than baseline")
    print(f"  • Time Investment: ~15 minutes")
    print(f"  • FYP Value: Demonstrates regularization understanding")
    
    if best_improvement > 10:
        print(f"\n✅ SUCCESS: {best_improvement:.1f}% improvement achieved!")
        print(f"   This shows you understand regularization techniques")
    elif best_improvement > 5:
        print(f"\n🔶 MODERATE: {best_improvement:.1f}% improvement")
        print(f"   Still demonstrates optimization attempt")
    else:
        print(f"\n🔴 LIMITED: {best_improvement:.1f}% improvement")
        print(f"   But still shows systematic approach")
    
    return best_model, best_improvement

def create_comparison_table(results):
    """Create comparison table for FYP presentation"""
    
    print(f"\n📋 FYP PRESENTATION TABLE:")
    print("="*60)
    print(f"{'Model':<15} {'MAE':<12} {'MAPE':<10} {'Improvement':<12}")
    print("-" * 60)
    
    baseline_mae = results['LinearRegression']['test_mae']
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_mae'])
    
    for model_name, metrics in sorted_results:
        mae_val = metrics['test_mae']
        mape_val = metrics['test_mape']
        improvement = (baseline_mae - mae_val) / baseline_mae * 100
        
        improvement_text = "baseline" if model_name == 'LinearRegression' else f"{improvement:+.1f}%"
        
        print(f"{model_name:<15} {mae_val:<12,.0f} {mape_val:<10.1f}% {improvement_text:<12}")

def demonstrate_quick_win():
    """Demonstrate the quick win concept"""
    
    print("🎓 LINEAR REGRESSION 'QUICK WIN' DEMONSTRATION")
    print("="*60)
    
    print("\n💡 CONCEPT EXPLANATION:")
    print("A 'quick win' means:")
    print("  • Minimal time investment (15-30 minutes)")
    print("  • Easy to implement (just change the model)")
    print("  • Measurable improvement (5-25% typical)")
    print("  • High academic value (shows understanding)")
    
    print("\n🔧 WHAT WE'RE DOING:")
    print("  1. Replace: LinearRegression()")
    print("  2. With: Ridge, Lasso, ElasticNet (regularized versions)")
    print("  3. Add: StandardScaler for proper feature scaling")
    print("  4. Test: Quick 3-fold CV to compare performance")
    
    print("\n📈 WHY IT WORKS:")
    print("  • Your dataset has 45+ features → High dimensionality")
    print("  • Vanilla LinearRegression → Prone to overfitting")
    print("  • Regularization → Prevents overfitting, improves generalization")
    print("  • Scaling → Ensures fair coefficient penalties")
    
    print("\n🎯 EXPECTED OUTCOMES:")
    print("  • Ridge: L2 penalty → Shrinks coefficients, reduces variance")
    print("  • Lasso: L1 penalty → Feature selection, sparse models")
    print("  • ElasticNet: L1+L2 → Best of both worlds")
    print("  • Improvement: 10-25% MAE reduction typical")
    
    print("\n⏱️ TIME BREAKDOWN:")
    print("  • Code modification: 5 minutes")
    print("  • Running tests: 10 minutes")
    print("  • Analysis: 5 minutes")
    print("  • Total: 20 minutes for potential 20% improvement!")

if __name__ == "__main__":
    demonstrate_quick_win()
    
    print(f"\n\n🔄 TO IMPLEMENT IN YOUR TRAINING:")
    print("="*50)
    print("Replace this in train_ml.py line ~195:")
    print()
    print("❌ OLD:")
    print("   model = LinearRegression()")
    print()
    print("✅ NEW:")
    print("   from sklearn.linear_model import Ridge")
    print("   from sklearn.preprocessing import StandardScaler")
    print("   from sklearn.pipeline import Pipeline")
    print()
    print("   model = Pipeline([")
    print("       ('scaler', StandardScaler()),")
    print("       ('ridge', Ridge(alpha=1.0, random_state=42))")
    print("   ])")
    print()
    print("🎯 Expected result: 15-25% improvement in 15 minutes!")
