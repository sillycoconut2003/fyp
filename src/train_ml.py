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

def train_random_forest_baseline(X_train, y_train, X_test, y_test):
    """
    Train RandomForest model with DEFAULT parameters first to establish baseline.
    
    Args:
        X_train, y_train: Training features and target
        X_test, y_test: Test features and target for final evaluation
        
    Returns:
        dict: Model results including trained model, metrics, and predictions
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL (BASELINE - DEFAULT PARAMETERS)")
    print("="*60)
    print("Configuration: Default parameters, 10-fold CV")
    
    # Default RandomForest configuration
    model = RandomForestRegressor(
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
        'cv_scores': cv_scores,
        'model_type': 'RandomForest_Baseline'
    }
    
    # Display final performance metrics
    print(f"\nBaseline Model Performance:")
    print(f"  Cross-Validation MAE: {cv_score:.2f} (±{cv_std:.2f})")
    print(f"  Test MAE: {test_mae:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  Test MAPE: {test_mape:.2f}%")
    print("="*60)
    
    return results

def train_random_forest_tuned(X_train, y_train, X_test, y_test):
    """
    Train RandomForest model with OPTIMIZED hyperparameters for comparison.
    
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
    print("TRAINING RANDOM FOREST MODEL (TUNED - OPTIMIZED PARAMETERS)")
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
        'cv_scores': cv_scores,
        'model_type': 'RandomForest_Tuned'
    }
    
    # Display final performance metrics
    print(f"\nTuned Model Performance:")
    print(f"  Cross-Validation MAE: {cv_score:.2f} (±{cv_std:.2f})")
    print(f"  Test MAE: {test_mae:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  Test MAPE: {test_mape:.2f}%")
    print("="*60)
    
    return results

def train_xgboost_baseline(X_train, y_train, X_test, y_test):
    """
    Train XGBoost model with DEFAULT parameters first to establish baseline.
    
    Args:
        X_train, y_train: Training features and target
        X_test, y_test: Test features and target for final evaluation
        
    Returns:
        dict: Model results including trained model, metrics, and predictions
    """
    print("\n" + "="*60)
    print("TRAINING XGBOOST MODEL (BASELINE - DEFAULT PARAMETERS)")
    print("="*60)
    print("Configuration: Default parameters, 5-fold CV")
    
    # Default XGBoost configuration
    model = xgb.XGBRegressor(
        random_state=42,         # Reproducibility
        n_jobs=-1,              # Use all CPU cores
        verbosity=1             # Show training progress
    )
    
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
        fold_model = xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=0  # Quiet during CV
        )
        
        # Train and evaluate fold
        fold_model.fit(X_train_fold, y_train_fold)
        pred_fold = fold_model.predict(X_val_fold)
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
    final_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
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
        'cv_scores': cv_scores,
        'model_type': 'XGBoost_Baseline'
    }
    
    # Display final performance metrics
    print(f"\nBaseline Model Performance:")
    print(f"  Cross-Validation MAE: {cv_score:.2f} (±{cv_std:.2f})")
    print(f"  Test MAE: {test_mae:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  Test MAPE: {test_mape:.2f}%")
    print("="*60)
    
    return results

def train_xgboost_tuned(X_train, y_train, X_test, y_test):
    """
    Train XGBoost model with OPTIMIZED hyperparameters for comparison.
    
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
    print("TRAINING XGBOOST MODEL (TUNED - OPTIMIZED PARAMETERS)")
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
        'cv_scores': cv_scores,
        'model_type': 'XGBoost_Tuned'
    }
    
    # Display final performance metrics
    print(f"\nTuned Model Performance:")
    print(f"  Cross-Validation MAE: {cv_score:.2f} (±{cv_std:.2f})")
    print(f"  Test MAE: {test_mae:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  Test MAPE: {test_mape:.2f}%")
    print("="*60)
    
    return results

def train_linear_regression_baseline(X_train, y_train, X_test, y_test):
    """
    Train Ridge Regression with DEFAULT parameters first to establish baseline.
    
    Args:
        X_train, y_train: Training features and target
        X_test, y_test: Test features and target for final evaluation
        
    Returns:
        dict: Model results including trained model, metrics, and predictions
    """
    print("\n" + "="*60)
    print("TRAINING RIDGE REGRESSION MODEL (BASELINE - DEFAULT PARAMETERS)")
    print("="*60)
    print("Configuration: StandardScaler + Ridge with alpha=1.0")
    
    # Default Ridge configuration with preprocessing
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(random_state=42))  # Default alpha=1.0
    ])
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    print("\nPerforming 5-fold time series cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        print(f"Fold {fold}/5: Training on {len(train_idx)} samples, validating on {len(val_idx)} samples")
        
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        fold_model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(random_state=42))
        ])
        fold_model.fit(X_train_fold, y_train_fold)
        pred_fold = fold_model.predict(X_val_fold)
        cv_scores.append(mae(y_val_fold, pred_fold))
        
        print(f"  Fold {fold} MAE: {cv_scores[-1]:.2f}")
    
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
        'cv_scores': cv_scores,
        'model_type': 'Ridge_Baseline',
        'best_alpha': 1.0  # Default alpha
    }
    
    # Display final performance metrics
    print(f"\nBaseline Model Performance:")
    print(f"  Default Alpha: 1.0")
    print(f"  Cross-Validation MAE: {cv_score:.2f} (±{cv_std:.2f})")
    print(f"  Test MAE: {test_mae:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  Test MAPE: {test_mape:.2f}%")
    print("="*60)
    
    return results

def train_linear_regression_tuned(X_train, y_train, X_test, y_test):
    """
    Train Ridge Regression with OPTIMIZED alpha parameter for comparison.
    
    Uses StandardScaler preprocessing and cross-validation to find optimal
    regularization parameter. Provides optimized performance for comparison.
    
    Args:
        X_train, y_train: Training features and target
        X_test, y_test: Test features and target for final evaluation
        
    Returns:
        dict: Model results including trained model, metrics, and predictions
    """
    print("\n" + "="*60)
    print("TRAINING RIDGE REGRESSION MODEL (TUNED - OPTIMIZED ALPHA)")
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
        'alpha_results': alpha_results,
        'model_type': 'Ridge_Tuned'
    }
    
    # Display final performance metrics
    print(f"\nTuned Model Performance:")
    print(f"  Best Alpha: {best_alpha}")
    print(f"  Cross-Validation MAE: {cv_score:.2f} (±{cv_std:.2f})")
    print(f"  Test MAE: {test_mae:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  Test MAPE: {test_mape:.2f}%")
    print("="*60)
    
    return results

def compare_before_after_models(baseline_results, tuned_results):
    """
    Compare baseline (default parameters) vs tuned (optimized parameters) model performance.
    
    Args:
        baseline_results: Dictionary containing baseline model results
        tuned_results: Dictionary containing tuned model results
        
    Returns:
        pd.DataFrame: Comprehensive comparison results
    """
    print("\n" + "="*80)
    print("BEFORE vs AFTER HYPERPARAMETER TUNING COMPARISON")
    print("="*80)
    
    comparison_data = []
    
    # Map model names between baseline and tuned
    model_mapping = {
        'RandomForest_Baseline': 'RandomForest_Tuned',
        'XGBoost_Baseline': 'XGBoost_Tuned', 
        'Ridge_Baseline': 'Ridge_Tuned'
    }
    
    for baseline_key, tuned_key in model_mapping.items():
        if baseline_key in baseline_results and tuned_key in tuned_results:
            baseline = baseline_results[baseline_key]
            tuned = tuned_results[tuned_key]
            
            # Calculate improvements
            mae_improvement = baseline['test_mae'] - tuned['test_mae']
            mae_improvement_pct = (mae_improvement / baseline['test_mae']) * 100
            
            rmse_improvement = baseline['test_rmse'] - tuned['test_rmse']
            rmse_improvement_pct = (rmse_improvement / baseline['test_rmse']) * 100
            
            r2_improvement = 0  # R² not calculated in current version
            
            comparison_data.append({
                'Model': baseline_key.replace('_Baseline', ''),
                'Baseline_MAE': baseline['test_mae'],
                'Tuned_MAE': tuned['test_mae'],
                'MAE_Improvement': mae_improvement,
                'MAE_Improvement_%': mae_improvement_pct,
                'Baseline_RMSE': baseline['test_rmse'],
                'Tuned_RMSE': tuned['test_rmse'],
                'RMSE_Improvement': rmse_improvement,
                'RMSE_Improvement_%': rmse_improvement_pct,
                'Baseline_MAPE': baseline['test_mape'],
                'Tuned_MAPE': tuned['test_mape']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display detailed comparison
    print(f"\n📊 COMPREHENSIVE PERFORMANCE COMPARISON:")
    print("="*50)
    for _, row in comparison_df.iterrows():
        print(f"\n🔍 {row['Model']}:")
        print(f"   📈 MAE: {row['Baseline_MAE']:,.2f} → {row['Tuned_MAE']:,.2f}")
        print(f"   📊 Improvement: {row['MAE_Improvement']:,.2f} ({row['MAE_Improvement_%']:+.2f}%)")
        print(f"   📈 RMSE: {row['Baseline_RMSE']:,.2f} → {row['Tuned_RMSE']:,.2f}")
        print(f"   📊 Improvement: {row['RMSE_Improvement']:,.2f} ({row['RMSE_Improvement_%']:+.2f}%)")
        print(f"   📈 MAPE: {row['Baseline_MAPE']:.2f}% → {row['Tuned_MAPE']:.2f}%")
    
    # Find best improvement
    if len(comparison_df) > 0:
        best_model = comparison_df.loc[comparison_df['MAE_Improvement_%'].idxmax()]
        print(f"\n🏆 BEST IMPROVEMENT: {best_model['Model']}")
        print(f"   📈 MAE Improvement: {best_model['MAE_Improvement_%']:+.2f}%")
        print(f"   📊 Final MAE: {best_model['Tuned_MAE']:,.2f}")
    
    print("="*80)
    
    return comparison_df

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
    Main training pipeline for MTA KPI prediction models with BEFORE/AFTER comparison.
    
    Executes comprehensive ML training workflow:
    1. Load and prepare data
    2. PHASE 1: Train baseline models (default parameters)
    3. PHASE 2: Train tuned models (optimized parameters) 
    4. PHASE 3: Compare before vs after performance improvements
    5. Save tuned models for deployment
    
    This demonstrates the impact of hyperparameter tuning by training each model
    twice - once with defaults and once with optimized parameters.
    
    Returns:
        tuple: (baseline_results, tuned_results, comparison_df)
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
        
        # Step 4: Train models with BEFORE/AFTER workflow
        print(f"\n🚀 Training ML models: BASELINE → TUNING → COMPARISON...")
        
        # PHASE 1: Train baseline models (default parameters)
        print(f"\n" + "="*70)
        print("PHASE 1: BASELINE TRAINING (DEFAULT PARAMETERS)")
        print("="*70)
        baseline_results = {}
        
        print(f"\n[1/3] RandomForest baseline (default parameters)...")
        baseline_results['RandomForest_Baseline'] = train_random_forest_baseline(X_train, y_train, X_test, y_test)
        
        print(f"\n[2/3] XGBoost baseline (default parameters)...")
        baseline_results['XGBoost_Baseline'] = train_xgboost_baseline(X_train, y_train, X_test, y_test)
        
        print(f"\n[3/3] Ridge baseline (default parameters)...")
        baseline_results['Ridge_Baseline'] = train_linear_regression_baseline(X_train, y_train, X_test, y_test)
        
        # PHASE 2: Train tuned models (optimized parameters)
        print(f"\n" + "="*70)
        print("PHASE 2: HYPERPARAMETER TUNING (OPTIMIZED PARAMETERS)")
        print("="*70)
        tuned_results = {}
        
        print(f"\n[1/3] RandomForest tuned (optimized parameters)...")
        tuned_results['RandomForest_Tuned'] = train_random_forest_tuned(X_train, y_train, X_test, y_test)
        
        print(f"\n[2/3] XGBoost tuned (optimized parameters)...")
        tuned_results['XGBoost_Tuned'] = train_xgboost_tuned(X_train, y_train, X_test, y_test)
        
        print(f"\n[3/3] Ridge tuned (optimized parameters)...")
        tuned_results['Ridge_Tuned'] = train_linear_regression_tuned(X_train, y_train, X_test, y_test)
        
        # PHASE 3: Compare baseline vs tuned performance
        print(f"\n" + "="*70)
        print("PHASE 3: BEFORE vs AFTER COMPARISON")
        print("="*70)
        comparison_df = compare_before_after_models(baseline_results, tuned_results)
        
        # Step 5: Save best models (tuned versions for production)
        print(f"\n💾 Saving tuned models for deployment...")
        save_ml_models(tuned_results, feature_cols)
        
        # Final summary with improvement highlights
        if len(comparison_df) > 0:
            best_improvement = comparison_df.loc[comparison_df['MAE_Improvement_%'].idxmax()]
            best_tuned = min(tuned_results.values(), key=lambda x: x['test_mae'])
            
            print(f"\n🎉 BEFORE/AFTER TRAINING COMPLETED SUCCESSFULLY!")
            print(f"🏆 Best improvement: {best_improvement['Model']} ({best_improvement['MAE_Improvement_%']:+.2f}% MAE)")
            print(f"🥇 Best final model: {best_tuned['model_type']} (MAE: {best_tuned['test_mae']:.2f})")
            print("All tuned models saved and ready for production use.")
        else:
            print(f"\n🎉 TRAINING COMPLETED!")
        
        print("="*80)
        
        return baseline_results, tuned_results, comparison_df
    except Exception as e:
        print(f"\n❌ ERROR during training pipeline: {e}")
        raise
    
    print("\nML Training Pipeline Completed!")
    return baseline_results, tuned_results, comparison_df

if __name__ == "__main__":
    baseline_results, tuned_results, comparison = main()