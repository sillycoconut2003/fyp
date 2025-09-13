# 🚇 MTA KPI Analytics & Forecasting Dashboard

## 🎯 Overview
This dashboard provides advanced predictive analytics for Metropolitan Transportation Authority Key Performance Indicators. It combines historical data analysis with machine learning and time series forecasting capabilities.

## 🚀 Features

### 📊 **Data Exploration**
- **5 MTA Agencies**: Bridges & Tunnels, Long Island Rail Road, MTA Bus, Metro-North Railroad, NYC Transit
- **130+ KPIs**: Performance metrics across all transportation modes
- **8+ Years of Data**: January 2009 to April 2017 (12,266 data points)

### 🤖 **Machine Learning Forecasting**
- **RandomForest**: Best overall performer (MAE: ~12K)
- **XGBoost**: Advanced gradient boosting (MAE: ~49K)
- **LinearRegression**: Baseline model (MAE: ~147K)

### 📈 **Time Series Forecasting**
- **Prophet**: Facebook's time series forecasting (individual series)
- **SARIMA**: Statistical ARIMA with seasonality (individual series)
- **132 trained models** per method (one for each agency-indicator combination)

## 🎮 How to Use

### 1. **Select Your Data**
- Choose an **Agency** from the dropdown (e.g., "NYC Transit")
- Select a **KPI** (e.g., "Total Ridership - Subways")

### 2. **Configure Forecast**
- Set **forecast horizon** (1-24 months)
- Choose **models** to compare:
  - ML models work for any series
  - Time series models only available if trained for that specific series

### 3. **Generate Predictions**
- Click **"🚀 Generate Forecast"**
- View interactive chart with:
  - Historical actual values (blue line)
  - Historical targets (green dashed line)
  - Model forecasts (colored dotted lines)
  - Confidence intervals (shaded areas for Prophet)

### 4. **Analyze Results**
- Compare model performance using MAE (Mean Absolute Error)
- Review forecast tables with monthly predictions
- Check data overview and model availability

## 📊 Model Performance Summary

| Approach | Best Model | Average MAE | Strengths |
|----------|------------|-------------|-----------|
| **Machine Learning** | RandomForest | ~12,651 | Cross-series patterns, engineered features |
| **Time Series** | Prophet | ~82,882 | Individual series characteristics, seasonality |

## 🔍 Key Insights

### **🏆 Champion: RandomForest**
- **75% better** than time series methods
- **4x improvement** over average time series performance
- Validated across 70 cross-validation splits with 95% confidence intervals
- Leverages relationships across different KPIs
- Uses engineered features (lags, rolling averages, calendar features)

### **📈 Time Series Models Excel At:**
- Series-specific patterns and seasonality
- Interpretable trend decomposition
- Individual KPI forecasting

## 💡 Tips for Best Results

1. **For overall accuracy**: Use RandomForest
2. **For series-specific insights**: Use Prophet or SARIMA
3. **For model comparison**: Generate forecasts from multiple models
4. **For seasonal patterns**: Prophet handles seasonality well
5. **For stable trends**: SARIMA works well for stationary series

## 📁 Data Sources
- **Raw Data**: MTA_Performance_Agencies.csv (>50MB)
- **Processed Data**: 58 engineered features including lags, rolling averages, and one-hot encodings
- **Models**: 267 total trained models (3 ML + 264 time series)

## 🎉 Project Achievement
This dashboard represents a complete dual-track modeling system comparing machine learning and time series approaches across 130+ transportation KPIs - a comprehensive solution for MTA performance forecasting.
