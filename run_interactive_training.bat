@echo off
REM ML Training Pipeline - Baseline vs Tuned Comparison
REM This script runs the comprehensive ML training workflow:
REM 1. Train ML models (RandomForest, XGBoost, Ridge) with DEFAULT parameters first
REM 2. Analyze performance metrics (MAE, RMSE, R², MAPE, Residuals)
REM 3. Apply hyperparameter tuning for optimization
REM 4. Compare baseline vs tuned performance improvements

echo.
echo ========================================
echo  ML Training: Baseline → Tuned → Compare
echo ========================================
echo.
echo Phase 1: Train models with DEFAULT parameters (baseline)
echo Phase 2: Analyze comprehensive performance metrics
echo Phase 3: Apply hyperparameter tuning (optimization)
echo Phase 4: Compare baseline vs tuned improvements
echo.
echo Starting ML training pipeline...
echo.

REM Set environment variables
set PYTHONWARNINGS=ignore
set SKLEARN_SHOW_PROGRESS=True

REM Navigate to project directory and run ML training
cd /d "%~dp0"
python src/train_ml.py

echo.
echo ========================================
echo ML training complete! 
echo Check reports/ for results and models/ for saved models
echo Baseline → Tuned comparison methodology implemented
echo ========================================
pause