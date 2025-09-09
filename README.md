# MTA KPI Analytics (VS Code Starter)

This project predicts and analyzes monthly KPIs from the **MTA_Performance_Agencies.csv** dataset with two modelling tracks:
- **Track A:** Per-series time-series (Prophet / SARIMA)
- **Track B:** Multivariate ML (RandomForest / XGBoost / Linear) using engineered features

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
