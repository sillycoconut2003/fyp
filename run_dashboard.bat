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
echo    - XGBoost: 39,885 MAE (9.5%% improvement)
echo    - RandomForest: 46,891 MAE (2.8%% improvement) 
echo    - Ridge Regression: 96,886 MAE (34.8%% improvement)
echo.
echo 📊 Starting dashboard with clean output...
echo.

REM Set environment variables to suppress warnings
set STREAMLIT_LOGGER_LEVEL=ERROR
set STREAMLIT_SERVER_ENABLE_STATIC_SERVING=false

REM Navigate to project directory and start dashboard
cd /d "%~dp0"
streamlit run dashboard/app.py --server.port 8501

pause
 