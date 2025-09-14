# MTA KPI Forecasting System - Clean Project Structure# MTA KPI Forecasting System Project Structure



## Directory Organization## Directory Organization



``````

FYP PROJECT/FYP PROJECT/

├── 📊 dashboard/                # Professional Analytics Dashboard├── 📊 src/                      # Core source code

│   ├── app.py                   # Streamlit application ⭐│   ├── config.py                # Configuration settings

│   └── style.css                # Professional Apple-inspired theme ⭐│   ├── eval.py                  # Evaluation metrics

││   ├── features.py              # Feature engineering

├── 📈 data/                     # Dataset storage│   ├── preprocess.py            # Data preprocessing

│   ├── raw/                     # Original MTA data│   ├── train_ml.py              # Optimized ML training pipeline ⭐

│   ├── interim/                 # Cleaned data│   ├── train_ts.py              # Time series training pipeline ⭐

│   └── processed/               # Model-ready data ⭐│   └── utils_io.py              # I/O utilities

│       └── mta_model.parquet    # Final optimized dataset│

│├── 🎯 dashboard/                # Interactive dashboard

├── 🤖 models/                   # Production Models│   └── app.py                   # Streamlit application ⭐

│   ├── RandomForest_model.pkl   # Best performer (MAE: 13,637) ⭐│

│   ├── XGBoost_model.pkl        # Secondary model (MAE: 39,885)├── 📈 data/                     # Dataset storage

│   ├── LinearRegression_model.pkl # Baseline model (MAE: 130,912)│   ├── raw/                     # Original MTA data

│   └── time_series/             # 264 Prophet & SARIMA models│   ├── interim/                 # Cleaned data

│       ├── prophet_models.pkl   # Prophet forecasting models│   └── processed/               # Model-ready data ⭐

│       └── sarima_models.pkl    # SARIMA time series models│

│├── 🤖 models/                   # Trained models

├── 📊 src/                      # Core Source Code│   ├── RandomForest_model.pkl   # Best performer ⭐

│   ├── config.py                # Configuration settings│   ├── XGBoost_model.pkl        # Optimized ⭐

│   ├── eval.py                  # Evaluation metrics│   ├── LinearRegression_model.pkl # Baseline

│   ├── features.py              # Feature engineering (45 features)│   └── time_series/             # Prophet & SARIMA models

│   ├── preprocess.py            # Data preprocessing pipeline│

│   ├── train_ml.py              # ML training pipeline ⭐├── 🔬 scripts/                  # Analysis & utilities

│   ├── train_ts.py              # Time series training pipeline ⭐│   ├── analysis/                # Performance analysis

│   └── utils_io.py              # I/O utilities│   ├── optimization/            # Hyperparameter tuning ⭐

││   ├── data_processing/         # Data preparation

├── 🔬 scripts/                  # Analysis & Optimization│   └── visualization/           # Chart generation

│   ├── analysis/                # Performance analysis tools│

│   │   ├── analyze_model_performance.py # Model comparison ⭐├── 📊 reports/                  # Generated outputs

│   │   ├── analyze_optimal_models.py    # Optimization analysis│   ├── eda_summary.md           # Exploratory analysis

│   │   └── check_model_structure.py     # Model validation│   └── figures/                 # Performance charts ⭐

│   ├── optimization/            # Hyperparameter tuning│

│   │   ├── hyperparameter_tuning.py     # General tuning├── 📚 docs/                     # Documentation

│   │   └── xgboost_hyperparameter_tuning.py # XGBoost specific│   ├── DASHBOARD_GUIDE.md       # User instructions

│   └── visualization/           # Chart generation│   ├── DASHBOARD_USER_GUIDE.md  # Detailed guide

│       ├── create_performance_graphs.py # Performance charts ⭐│   └── MODEL_SELECTION_GUIDE.md # Model recommendations

│       └── create_training_graphs.py    # Training visualization│

│├── 📝 notebooks/                # Jupyter analysis

├── 📊 reports/                  # Performance Reports│   └── eda.ipynb               # Exploratory data analysis

│   ├── model_evaluation_final_report.txt        # Technical evaluation ⭐│

│   ├── model_evaluation_results.csv             # Performance metrics├── 📋 README.md                 # Project overview ⭐

│   ├── model_performance_overview_updated.md    # Executive summary ⭐├── 📦 requirements.txt          # Dependencies

│   └── figures/                                 # Performance visualizations ⭐├── 🚀 run_dashboard.bat         # Quick startup ⭐

│       ├── ml_model_performance.png└── ⚙️ Makefile                 # Build automation

│       ├── model_ecosystem_summary.png```

│       ├── model_evaluation_comprehensive.png

│       ├── model_selection_flow.png### **Core Implementation**

│       ├── performance_table.png- **`src/train_ml.py`** - Main ML training with optimized hyperparameters

│       └── training_iterations.png- **`dashboard/app.py`** - Interactive forecasting dashboard

│- **`models/`** - All 267 trained models

├── 📚 docs/                     # Documentation

│   ├── DASHBOARD_GUIDE.md       # Dashboard usage instructions### **Optimization Results**

│   ├── DASHBOARD_USER_GUIDE.md  # Detailed user guide- **`scripts/optimization/`** - Hyperparameter tuning methodology

│   ├── MODEL_SELECTION_GUIDE.md # Model selection methodology- **`reports/figures/`** - Performance visualization charts

│   ├── PERCENTAGE_KPI_GUIDE.md  # KPI-specific optimizations

│   └── STATISTICAL_VALIDATION_RESPONSE.md # Validation methodology### **Documentation**

│- **`docs/`** - Complete user guides and model selection

├── 📝 notebooks/                # Analysis Notebooks- **`README.md`** - Project overview and instructions

│   ├── eda.ipynb               # Exploratory data analysis

│   └── model_evaluation_methodology.ipynb # Evaluation methodology### **Quick Demo**

│- **`run_dashboard.bat`** - One-click dashboard startup

├── 🧪 tests/                    # Test Suite- **`data/processed/mta_model.parquet`** - Ready-to-use dataset

│   └── test_percentage_prediction.py # Model validation tests

│## Quick Start Commands

├── 📦 dev/                      # Development Scripts (Archived)

│   ├── fyp_optimization_strategy.py    # Optimization strategy```bash

│   ├── hyperparameter_recommendations.py # Parameter recommendations# Install dependencies

│   └── linear_regression_quick_win.py  # Linear regression enhancementspip install -r requirements.txt

│

├── 📁 archive/                  # Historical Analysis (Archived)# Launch dashboard

│   ├── architecture_optimization_log.md # Optimization history./run_dashboard.bat

│   └── eda_summary.md                   # Historical EDA summary

│# Run training pipeline

├── 📋 README.md                 # Project overview ⭐python src/train_ml.py

├── 📦 requirements.txt          # Dependencies ⭐

├── 🚀 run_dashboard.bat         # One-click dashboard startup ⭐# Generate performance analysis

├── 📄 PROJECT_STRUCTURE.md      # This filepython scripts/analysis/analyze_model_performance.py

└── ⚙️ Makefile                 # Build automation```

```

## File Usage Priority

## **Production Components** ⭐

### **High Priority**

### **Core System**1. `src/train_ml.py` - Core training implementation

1. **`dashboard/app.py`** - Professional analytics dashboard with Apple-inspired design2. `dashboard/app.py` - Interactive demonstration

2. **`src/train_ml.py`** - Optimized ML training (RandomForest best performer)3. `reports/figures/` - Performance visualizations

3. **`models/`** - 267 trained models (3 ML + 264 time series)4. `docs/` - Documentation for presentation

4. **`data/processed/mta_model.parquet`** - Clean, feature-engineered dataset

### **Medium Priority**

### **Performance Documentation**1. `scripts/optimization/` - Shows methodology

1. **`reports/model_performance_overview_updated.md`** - Executive summary with corrected metrics2. `models/` - Trained model artifacts

2. **`reports/figures/`** - Professional performance charts3. `data/processed/` - Clean dataset

3. **`docs/DASHBOARD_USER_GUIDE.md`** - Complete user instructions

### **Low Priority**

### **Key Achievements** 🎯1. `scripts/analysis/` - Development tools

- **5x Performance Improvement**: Percentage KPI prediction (MAE: 0.335 vs 1.732)2. `scripts/data_processing/` - Data preparation

- **Professional UI**: Apple-inspired dashboard with confidence intervals3. `notebooks/` - Exploratory analysis

- **Comprehensive ML Pipeline**: 45 engineered features, optimized hyperparameters

- **Statistical Validation**: Bootstrap confidence intervals for ML, native for time series---

## **Quick Start Commands**

```bash
# Launch professional dashboard
./run_dashboard.bat

# Install dependencies
pip install -r requirements.txt

# Run optimized ML training
python src/train_ml.py

# Generate performance analysis
python scripts/analysis/analyze_model_performance.py
```

## **Performance Summary**

| Model | MAE | Status | Performance vs Best |
|-------|-----|--------|-------------------|
| **RandomForest** | **13,637** | ✅ **Best Performer** | **Baseline** |
| XGBoost | 39,885 | ⚠️ Secondary | +192% |
| LinearRegression | 130,912 | 📊 Baseline | +860% |

**Total Models Trained**: 267 (3 ML + 264 Time Series)
**Dataset**: 45 engineered features, March 2017 cutoff
**Optimization**: Unified RandomForest approach, removed specialized predictors

---

*Project cleaned and organized - September 15, 2025*