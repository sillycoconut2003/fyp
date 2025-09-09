"""
Machine Learning Training Pipeline for MTA KPI Forecasting
=========================================================
Trains RandomForest, XGBoost, and Ridge Regression models using cross-series features.
Uses engineered features and temporal cross-validation for robust performance evaluation.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

# Add the src directory to the path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROCESSED
from eval import rmse, mae, mape

def load_processed_data():
    """Load the processed dataset for ML training"""
    print(f"Loading processed data from: {PROCESSED}")
    df = pd.read_parquet(PROCESSED)
    print(f"Loaded dataset with shape: {df.shape}")
    return df

def prepare_ml_features(df):
    """Prepare features for ML models"""
    print("Preparing features for ML training...")
    
    # Target variable
    target_col = "MONTHLY_ACTUAL"
    
    # Feature columns - exclude target and identifier columns
    exclude_cols = [
        "MONTHLY_ACTUAL", "YYYY_MM", "INDICATOR_SEQ", "PARENT_SEQ",
        "DESCRIPTION", "INDICATOR_UNIT", "DECIMAL_PLACES", 
        "PERIOD_YEAR", "PERIOD_MONTH", "AGENCY_NAME", "INDICATOR_NAME",
        "CATEGORY", "DESIRED_CHANGE"  # Exclude text categorical columns as they're already one-hot encoded
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove rows with missing target values
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # Handle missing values efficiently
    numeric_cols = df_clean[feature_cols].select_dtypes(include=[np.number]).columns
    non_numeric_cols = [col for col in feature_cols if col not in numeric_cols]
    
    # Fill numeric columns with median
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    # Handle non-numeric columns
    for col in non_numeric_cols:
        print(f"Warning: Converting non-numeric column: {col} ({df_clean[col].dtype})")
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")
    print(f"Clean dataset shape: {df_clean.shape}")
    
    return df_clean, feature_cols, target_col

def split_data_temporal(df, feature_cols, target_col, test_size=0.2):
    """Split data temporally for time series validation"""
    # Sort by date to ensure temporal order
    df_sorted = df.sort_values('YYYY_MM').copy()
    
    # Split point based on time
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train date range: {train_df['YYYY_MM'].min()} to {train_df['YYYY_MM'].max()}")
    print(f"Test date range: {test_df['YYYY_MM'].min()} to {test_df['YYYY_MM'].max()}")
    
    return X_train, X_test, y_train, y_test, train_df, test_df

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train RandomForest model optimized for MTA KPI prediction.
    
    Uses optimal hyperparameters discovered through extensive tuning:
    - 500 trees for stable predictions
    - max_depth=15 prevents overfitting (reduces model size from 158MB to 27MB)
    - Comprehensive cross-validation with TimeSeriesSplit
    
    Args:
        X_train, y_train: Training features and target
        X_test, y_test: Test features and target for final evaluation
        
    Returns:
        dict: Model results including trained model, metrics, and predictions
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)
    print("Configuration: 500 trees, max_depth=15, 10-fold CV")
    
    # Optimal RandomForest configuration (extensively tuned)
    model = RandomForestRegressor(
        n_estimators=500,       # Optimal tree count for performance/stability balance
        max_depth=15,           # Critical: prevents overfitting and reduces file size
        min_samples_split=5,    # Prevents overfitting on small node splits
        min_samples_leaf=2,     # Allows granular predictions while preventing noise
        max_features=1.0,       # Use all features (optimal for this dataset)
        bootstrap=True,         # Enable bootstrap sampling for robustness
        random_state=42,        # Reproducibility
        n_jobs=-1,             # Use all CPU cores
        verbose=1              # Show training progress
    )
    
    # Robust time series cross-validation
    print("\nPerforming 10-fold time series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=10)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        print(f"Fold {fold}/10: Training on {len(train_idx)} samples, validating on {len(val_idx)} samples")
        
        # Split data for current fold
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        # Train and evaluate fold
        model.fit(X_train_fold, y_train_fold)
        pred_fold = model.predict(X_val_fold)
        fold_mae = mae(y_val_fold, pred_fold)
        cv_scores.append(fold_mae)
        
        print(f"  Fold {fold} MAE: {fold_mae:.2f}")
    
    # Calculate cross-validation performance
    cv_score = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"\nCross-Validation Results:")
    print(f"  Mean MAE: {cv_score:.2f} (±{cv_std:.2f})")
    
    # Train final model on full training set
    print("\nTraining final model on complete training set...")
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_mae = mae(y_test, y_pred)
    test_rmse = rmse(y_test, y_pred)
    test_mape = mape(y_test, y_pred)
    
    # Compile comprehensive results
    results = {
        'model': model,
        'cv_mae': cv_score,
        'cv_std': cv_std,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mape': test_mape,
        'predictions': y_pred,
        'cv_scores': cv_scores
    }
    
    # Display final performance metrics
    print(f"\nFinal Model Performance:")
    print(f"  Cross-Validation MAE: {cv_score:.2f} (±{cv_std:.2f})")
    print(f"  Test MAE: {test_mae:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  Test MAPE: {test_mape:.2f}%")
    print("="*60)
    
    return results

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost model with optimized hyperparameters for MTA KPI prediction.
    
    Uses extensively tuned parameters for optimal performance:
    - 400 trees with controlled depth and learning rate
    - Regularization to prevent overfitting
    - Efficient 5-fold cross-validation
    
    Args:
        X_train, y_train: Training features and target
        X_test, y_test: Test features and target for final evaluation
        
    Returns:
        dict: Model results including trained model, metrics, and predictions
    """
    print("\n" + "="*60)
    print("TRAINING XGBOOST MODEL")
    print("="*60)
    print("Configuration: 400 trees, max_depth=12, learning_rate=0.12")
    
    # Optimized XGBoost hyperparameters
    xgb_params = {
        'n_estimators': 400,        # Optimal tree count
        'max_depth': 12,            # Increased from default 6 for better performance
        'learning_rate': 0.12,      # Optimized learning rate
        'subsample': 1.0,           # Use full sample for robustness
        'colsample_bytree': 0.8,    # Feature subsampling for regularization
        'reg_lambda': 0.5,          # L2 regularization
        'reg_alpha': 0,             # No L1 regularization (optimal for this data)
        'min_child_weight': 1,      # Minimum weight in leaf nodes
        'random_state': 42,         # Reproducibility
        'n_jobs': -1,              # Use all CPU cores
        'verbosity': 1             # Show training progress
    }
    
    # Time series cross-validation
    print("\nPerforming 5-fold time series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        print(f"Fold {fold}/5: Training on {len(train_idx)} samples, validating on {len(val_idx)} samples")
        
        # Split data for current fold
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        # Create and train XGBoost model
        model = xgb.XGBRegressor(**xgb_params)
        
        # Train and evaluate fold
        model.fit(X_train_fold, y_train_fold)
        pred_fold = model.predict(X_val_fold)
        fold_mae = mae(y_val_fold, pred_fold)
        cv_scores.append(fold_mae)
        
        print(f"  Fold {fold} MAE: {fold_mae:.2f}")
    
    # Calculate cross-validation performance
    cv_score = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"\nCross-Validation Results:")
    print(f"  Mean MAE: {cv_score:.2f} (±{cv_std:.2f})")
    
    # Train final model on full training set
    print("\nTraining final model on complete training set...")
    final_model = xgb.XGBRegressor(**xgb_params)
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = final_model.predict(X_test)
    test_mae = mae(y_test, y_pred)
    test_rmse = rmse(y_test, y_pred)
    test_mape = mape(y_test, y_pred)
    
    # Compile comprehensive results
    results = {
        'model': final_model,
        'cv_mae': cv_score,
        'cv_std': cv_std,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mape': test_mape,
        'predictions': y_pred,
        'cv_scores': cv_scores
    }
    
    # Display final performance metrics
    print(f"\nFinal Model Performance:")
    print(f"  Cross-Validation MAE: {cv_score:.2f} (±{cv_std:.2f})")
    print(f"  Test MAE: {test_mae:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  Test MAPE: {test_mape:.2f}%")
    print("="*60)
    
    return results

def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    Train Ridge Regression with automated alpha optimization.
    
    Uses StandardScaler preprocessing and cross-validation to find optimal
    regularization parameter. Provides quick baseline performance for comparison.
    
    Args:
        X_train, y_train: Training features and target
        X_test, y_test: Test features and target for final evaluation
        
    Returns:
        dict: Model results including trained model, metrics, and predictions
    """
    print("\n" + "="*60)
    print("TRAINING RIDGE REGRESSION MODEL")
    print("="*60)
    print("Configuration: StandardScaler + Ridge with alpha optimization")
    
    # Alpha values to test for regularization
    alphas = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    best_alpha = None
    best_cv_score = float('inf')
    alpha_results = []
    
    # Time series cross-validation for alpha selection
    tscv = TimeSeriesSplit(n_splits=5)
    
    print(f"\nOptimizing alpha parameter with 5-fold cross-validation:")
    print(f"{'Alpha':<10} {'CV MAE':<12} {'Status'}")
    print("-" * 35)
    
    # Test each alpha value
    for alpha in alphas:
        # Create pipeline with current alpha
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha, random_state=42))
        ])
        
        # Cross-validate current alpha
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            pred_fold = model.predict(X_val_fold)
            cv_scores.append(mae(y_val_fold, pred_fold))
        
        # Calculate average performance for this alpha
        avg_cv_score = np.mean(cv_scores)
        alpha_results.append((alpha, avg_cv_score))
        
        # Check if this is the new best alpha
        status = "← NEW BEST" if avg_cv_score < best_cv_score else ""
        if avg_cv_score < best_cv_score:
            best_cv_score = avg_cv_score
            best_alpha = alpha
        
        print(f"{alpha:<10} {avg_cv_score:<12.2f} {status}")
    
    print(f"\nOptimal alpha selected: {best_alpha} (MAE: {best_cv_score:.2f})")
    
    # Train final model with optimal alpha
    print(f"\nTraining final Ridge model with alpha={best_alpha}...")
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=best_alpha, random_state=42))
    ])
    
    # Final cross-validation for reporting
    cv_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        temp_model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=best_alpha, random_state=42))
        ])
        temp_model.fit(X_train_fold, y_train_fold)
        pred_fold = temp_model.predict(X_val_fold)
        cv_scores.append(mae(y_val_fold, pred_fold))
    
    cv_score = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # Train final model on full training set
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = final_model.predict(X_test)
    test_mae = mae(y_test, y_pred)
    test_rmse = rmse(y_test, y_pred)
    test_mape = mape(y_test, y_pred)
    
    # Compile comprehensive results
    results = {
        'model': final_model,
        'best_alpha': best_alpha,
        'cv_mae': cv_score,
        'cv_std': cv_std,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mape': test_mape,
        'predictions': y_pred,
        'alpha_results': alpha_results
    }
    
    # Display final performance metrics
    print(f"\nFinal Model Performance:")
    print(f"  Best Alpha: {best_alpha}")
    print(f"  Cross-Validation MAE: {cv_score:.2f} (±{cv_std:.2f})")
    print(f"  Test MAE: {test_mae:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  Test MAPE: {test_mape:.2f}%")
    print("="*60)
    
    return results

def save_ml_models(results_dict, feature_cols):
    """
    Save trained ML models to disk with comprehensive metadata.
    
    Args:
        results_dict: Dictionary containing model results
        feature_cols: List of feature column names
    """
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"\n" + "="*60)
    print("SAVING TRAINED MODELS")
    print("="*60)
    
    for model_name, results in results_dict.items():
        model_path = models_dir / f"{model_name}_model.pkl"
        
        # Comprehensive model metadata
        save_dict = {
            'model': results['model'],
            'feature_cols': feature_cols,
            'cv_mae': results['cv_mae'],
            'cv_std': results.get('cv_std', 0),
            'test_mae': results['test_mae'],
            'test_rmse': results['test_rmse'],
            'test_mape': results['test_mape'],
            'model_type': model_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'num_features': len(feature_cols)
        }
        
        # Add model-specific metadata
        if model_name == 'Ridge' and 'best_alpha' in results:
            save_dict['best_alpha'] = results['best_alpha']
        
        with open(model_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✓ Saved {model_name} model: {model_path}")
        print(f"  Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print("="*60)

def compare_models(results_dict):
    """
    Generate comprehensive comparison of all trained models.
    
    Args:
        results_dict: Dictionary containing results from all trained models
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create detailed comparison DataFrame
    comparison_data = {
        'Model': list(results_dict.keys()),
        'CV_MAE': [results['cv_mae'] for results in results_dict.values()],
        'Test_MAE': [results['test_mae'] for results in results_dict.values()],
        'Test_RMSE': [results['test_rmse'] for results in results_dict.values()],
        'Test_MAPE': [results['test_mape'] for results in results_dict.values()]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by test MAE (lower is better)
    comparison_df = comparison_df.sort_values('Test_MAE')
    
    # Display comprehensive comparison table
    print(f"\n{'Model':<15} {'CV MAE':<10} {'Test MAE':<10} {'Test RMSE':<12} {'Test MAPE':<10}")
    print("-" * 65)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['Model']:<15} {row['CV_MAE']:<10.2f} {row['Test_MAE']:<10.2f} "
              f"{row['Test_RMSE']:<12.2f} {row['Test_MAPE']:<10.2f}%")
    
    # Identify best model
    best_model = comparison_df.iloc[0]['Model']
    best_mae = comparison_df.iloc[0]['Test_MAE']
    
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model}")
    print(f"   Test MAE: {best_mae:.2f}")
    
    # Calculate performance differences
    if len(comparison_df) > 1:
        second_best_mae = comparison_df.iloc[1]['Test_MAE']
        improvement = ((second_best_mae - best_mae) / second_best_mae) * 100
        print(f"   Performance improvement: {improvement:.1f}% better than next best")
    
    print("="*80)
    
    return comparison_df

def main():
    """
    Main training pipeline for MTA KPI prediction models.
    
    Executes comprehensive ML training workflow:
    1. Load and prepare data
    2. Train multiple model types (RandomForest, XGBoost, Ridge)
    3. Compare performance across models
    4. Save trained models for deployment
    
    This is the optimized FYP version with extensive hyperparameter tuning.
    """
    print("="*80)
    print("MTA KPI PREDICTION - ML MODEL TRAINING PIPELINE")
    print("="*80)
    print("FYP Project: Optimized machine learning models for cross-series forecasting")
    
    try:
        # Step 1: Load processed data
        print("\n📊 Loading processed MTA data...")
        df = load_processed_data()
        print(f"Loaded dataset: {df.shape[0]} records, {df.shape[1]} columns")
        
        # Step 2: Prepare ML features
        print("\n🔧 Preparing features for machine learning...")
        df_clean, feature_cols, target_col = prepare_ml_features(df)
        print(f"Features prepared: {len(feature_cols)} features, target: {target_col}")
        
        # Step 3: Temporal data split
        print("\n📈 Performing temporal train/test split...")
        X_train, X_test, y_train, y_test, train_df, test_df = split_data_temporal(
            df_clean, feature_cols, target_col
        )
        print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
        
        # Step 4: Train models with progress tracking
        print(f"\n🚀 Training optimized ML models...")
        results = {}
        
        # Train RandomForest (current best performer)
        print(f"\n[1/3] RandomForest with optimal parameters...")
        results['RandomForest'] = train_random_forest(X_train, y_train, X_test, y_test)
        
        # Train XGBoost (competitive alternative)  
        print(f"\n[2/3] XGBoost with tuned hyperparameters...")
        results['XGBoost'] = train_xgboost(X_train, y_train, X_test, y_test)
        
        # Train Ridge Regression (baseline)
        print(f"\n[3/3] Ridge Regression with alpha optimization...")
        results['LinearRegression'] = train_linear_regression(X_train, y_train, X_test, y_test)
        
        # Step 5: Comprehensive model comparison
        print(f"\n📊 Analyzing model performance...")
        comparison_df = compare_models(results)
        
        # Step 6: Save trained models
        print(f"\n💾 Saving models for deployment...")
        save_ml_models(results, feature_cols)
        
        # Final summary
        best_model = comparison_df.iloc[0]['Model']
        best_mae = comparison_df.iloc[0]['Test_MAE']
        
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Best model: {best_model} (MAE: {best_mae:.2f})")
        print("All models saved and ready for production use.")
        print("="*80)
        
        return results, comparison_df
        
    except Exception as e:
        print(f"\n❌ ERROR during training pipeline: {e}")
        raise
    
    print("\nML Training Pipeline Completed!")
    return results, comparison_df

if __name__ == "__main__":
    results, comparison = main()