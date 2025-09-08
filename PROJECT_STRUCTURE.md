# 🚇 MTA KPI Forecasting System - Project Structure

## 📁 Directory Organization

```
FYP PROJECT/
├── 📊 src/                      # Core source code
│   ├── config.py                # Configuration settings
│   ├── eval.py                  # Evaluation metrics
│   ├── features.py              # Feature engineering
│   ├── model_reg.py             # ML model definitions
│   ├── model_ts.py              # Time series models
│   ├── preprocess.py            # Data preprocessing
│   ├── train_ml.py              # ML training pipeline ⭐
│   ├── train_ts.py              # Time series training
│   └── utils_io.py              # I/O utilities
│
├── 🎯 dashboard/                # Interactive dashboard
│   └── app.py                   # Streamlit application ⭐
│
├── 📈 data/                     # Dataset storage
│   ├── raw/                     # Original MTA data
│   ├── interim/                 # Cleaned data
│   └── processed/               # Model-ready data ⭐
│
├── 🤖 models/                   # Trained models
│   ├── RandomForest_model.pkl   # Best performer ⭐
│   ├── XGBoost_model.pkl        # Optimized ⭐
│   ├── LinearRegression_model.pkl # Baseline
│   └── time_series/             # Prophet & SARIMA models
│
├── 🔬 scripts/                  # Analysis & utilities
│   ├── analysis/                # Performance analysis
│   ├── optimization/            # Hyperparameter tuning ⭐
│   ├── data_processing/         # Data preparation
│   └── visualization/           # Chart generation
│
├── 📊 reports/                  # Generated outputs
│   ├── eda_summary.md           # Exploratory analysis
│   └── figures/                 # Performance charts ⭐
│
├── 📚 docs/                     # Documentation
│   ├── DASHBOARD_GUIDE.md       # User instructions
│   ├── DASHBOARD_USER_GUIDE.md  # Detailed guide
│   └── MODEL_SELECTION_GUIDE.md # Model recommendations
│
├── 📝 notebooks/                # Jupyter analysis
│   └── eda.ipynb               # Exploratory data analysis
│
├── 📋 README.md                 # Project overview ⭐
├── 📦 requirements.txt          # Dependencies
├── 🚀 run_dashboard.bat         # Quick startup ⭐
└── ⚙️ Makefile                 # Build automation
```

## 🎯 Key Files for FYP Presentation

### **Core Implementation**
- **`src/train_ml.py`** - Main ML training with optimized hyperparameters
- **`dashboard/app.py`** - Interactive forecasting dashboard
- **`models/`** - All 267 trained models

### **Optimization Results**
- **`scripts/optimization/`** - Hyperparameter tuning methodology
- **`reports/figures/`** - Performance visualization charts

### **Documentation**
- **`docs/`** - Complete user guides and model selection
- **`README.md`** - Project overview and instructions

### **Quick Demo**
- **`run_dashboard.bat`** - One-click dashboard startup
- **`data/processed/mta_model.parquet`** - Ready-to-use dataset

## 🏆 Project Achievements

### **Performance Metrics**
- **RandomForest**: MAE 14,095 (Champion) ⭐
- **XGBoost**: MAE 39,885 (9.5% optimized) ⭐  
- **Ridge Regression**: MAE 96,886 (34.8% improved) ⭐
- **267 Total Models**: 3 ML + 264 Time Series

### **Technical Contributions**
- ✅ Systematic hyperparameter optimization
- ✅ Time series cross-validation methodology  
- ✅ Multi-scale data handling (decimals to millions)
- ✅ Professional dashboard with clean UI
- ✅ Comprehensive model selection framework

### **Academic Value**
- 🎓 Demonstrates ML engineering best practices
- 🎓 Shows optimization methodology understanding
- 🎓 Illustrates practical forecasting deployment
- 🎓 Proves systematic approach to model evaluation

## 🚀 Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
./run_dashboard.bat

# Run training pipeline
python src/train_ml.py

# Generate performance analysis
python scripts/analysis/analyze_model_performance.py
```

## 📊 File Usage Priority

### **High Priority (FYP Essential)**
1. `src/train_ml.py` - Core training implementation
2. `dashboard/app.py` - Interactive demonstration
3. `reports/figures/` - Performance visualizations
4. `docs/` - Documentation for presentation

### **Medium Priority (Supporting)**
1. `scripts/optimization/` - Shows methodology
2. `models/` - Trained model artifacts
3. `data/processed/` - Clean dataset

### **Low Priority (Background)**
1. `scripts/analysis/` - Development tools
2. `scripts/data_processing/` - Data preparation
3. `notebooks/` - Exploratory analysis

---

**🎯 Ready for FYP submission and defense presentation!**
