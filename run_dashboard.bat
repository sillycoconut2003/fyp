@echo off
REM FYP 2025 - MTA Analytics Dashboard Launcher
REM Optimized ML Models with Hyperparameter Tuning

echo.
echo ========================================
echo   🎓 FYP 2025 - MTA KPI Analytics
echo   🚀 Optimized ML Dashboard
echo   ✅ Ready for Presentation
echo ========================================
echo.
echo 🔧 Loading optimized models:
echo    - RandomForest: 13,637 MAE (CHAMPION - Best Performance)
echo    - XGBoost: 39,885 MAE (+192%% vs RandomForest)
echo    - LinearRegression: 130,912 MAE (+860%% vs RandomForest)
echo    - Time Series: 264 Prophet + SARIMA models
echo.
echo 📊 Starting dashboard with clean output...
echo.

REM Set environment variables to suppress ALL verbose outputs
set STREAMLIT_LOGGER_LEVEL=ERROR
set STREAMLIT_SERVER_ENABLE_STATIC_SERVING=false
set PYTHONWARNINGS=ignore
set SKLEARN_SHOW_PROGRESS=False

REM Navigate to project directory and start dashboard with output suppression
cd /d "%~dp0"
streamlit run dashboard/app.py --server.port 8501 2>nul

pause
 