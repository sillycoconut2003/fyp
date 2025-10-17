# 🚇 MTA Performance Analytics Dashboard

## 📊 Overview
Advanced machine learning system for predicting MTA agency KPI performance with **99.77% R² accuracy**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🏆 Model Performance

### **Production Models**
| Model | MAE | Performance | Status |
|-------|-----|-------------|---------|
| **RandomForest** | **13,637** | 🥇 Best | Production |
| OptimizedEnsemble | 20,880 | 🥈 Good | Available |
| XGBoost | 39,885 | 🥉 Fast | Backup |
| Ridge Regression | 130,912 | 📊 Baseline | Fallback |

## � **Rigorous Model Evaluation & Statistical Validation**

**Statistical Evidence for Performance Claims**: All model performance claims are now backed by rigorous time-based cross-validation and statistical testing.

**🎯 Key Validation Results:**
- **Time-based CV**: No data leakage, expanding window validation
- **Statistical Tests**: Paired t-tests, Wilcoxon signed-rank tests, confidence intervals
- **Per-Series Analysis**: Individual KPI performance distributions (not just overall means)
- **Claim Validation**: Direct statistical testing of stated performance metrics

**📋 Complete Evaluation Framework**: [`notebooks/model_evaluation_methodology.ipynb`](notebooks/model_evaluation_methodology.ipynb)

This comprehensive evaluation addresses the critical gap between technical implementation and statistical evidence, providing the rigorous foundation needed for academic credibility.

## �🔧 **NEW: Specialized Percentage KPI Prediction**

For percentage-based KPIs (like completion rates), this project now includes a specialized prediction system that addresses fundamental issues with applying general ML models to bounded operational metrics.

**Key Features:**
- Smart percentage KPI detection
- Model-specific prediction strategies  
- Realistic bounds enforcement (93-100% for completion rates)
- Stable, deterministic results

**Supported KPIs:**
- % of Completed Trips (Bus/Transit)
- Elevator/Escalator Availability  
- On-time Performance Rates
- Other operational efficiency metrics

See [`docs/PERCENTAGE_KPI_GUIDE.md`](docs/PERCENTAGE_KPI_GUIDE.md) for complete details.

## Quickstart

```bash
# from this folder
make setup
make build_processed     # writes data/processed/mta_model.parquet
make dashboard           # opens Streamlit dashboard
```

## Project layout

```
mta-kpi/
├─ data/
│  ├─ raw/               # put MTA_Performance_Agencies.csv here
│  ├─ interim/           # cached cleaned parquet/csv
│  └─ processed/         # modelling-ready datasets
├─ notebooks/
├─ src/
│  ├─ config.py
│  ├─ utils_io.py
│  ├─ preprocess.py
│  ├─ features.py        # Build processed dataset (Step 1)
│  ├─ train_ml.py        # Optimized ML training pipeline
│  ├─ train_ts.py        # Time series training pipeline
│  └─ eval.py
├─ dashboard/
│  └─ app.py             # Streamlit dashboard
├─ reports/
│  ├─ figures/
│  └─ eda_summary.md
├─ tests/
│  └─ test_basic.py
├─ requirements.txt
├─ Makefile
└─ README.md
```

## Step 1 (Processed Dataset)

Run `make build_processed` to create **data/processed/mta_model.parquet** with:
- calendar features: `year`, `month`, `quarter`
- lag/rolling features: `m_act_lag1, lag3, lag12`, `m_act_roll3, roll6, roll12` (leakage-safe)
- one-hots: agencies, plus top-N indicators (default N=25, others mapped to `OTHER`)

> Note: The pipeline is leakage-safe: lags are computed with `shift(1)`, and rolling features are computed on **shifted** series, so the current month never uses its own value.
