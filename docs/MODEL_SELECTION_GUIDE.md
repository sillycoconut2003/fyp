# 🤖 Model Selection Guide for MTA Dashboard

## Understanding Your Models

### **🔬 Machine Learning Models (Cross-Series)**
**Available for: ALL agency-indicator combinations**

| Model | MAE | Best Use Case | When to Choose |
|-------|-----|---------------|----------------|
| **RandomForest** | 12,651 | General-purpose forecasting | **Default choice** - Most reliable |
| **XGBoost** | 49,445 | Fast processing | When speed matters over accuracy |
| **LinearRegression** | 147,061 | Baseline comparison | Understanding linear relationships |

### **📈 Time Series Models (Individual Series)**
**Available for: Specific agency-indicator combinations with sufficient data**

| Model | Coverage | Best Use Case | When to Choose |
|-------|----------|---------------|----------------|
| **Prophet** | 132 series | Seasonal patterns, holidays | Strong seasonal trends expected |
| **SARIMA** | 132 series | Statistical modeling | Traditional time series analysis |

---

## 🎯 Model Selection Strategy

### **For Most Users: Start Here**
```
1. Always select "ML: RandomForest" (best accuracy)
2. Add "TS: Prophet" if available (seasonal insights)
3. Compare results for ensemble decision-making
```

### **Advanced Users: Model Combination Guide**

#### **High-Volume KPIs** (Ridership, Traffic)
- **Primary**: ML: RandomForest
- **Secondary**: TS: Prophet (for seasonality)
- **Why**: Cross-series patterns more important than individual trends

#### **Percentage KPIs** (On-time %, Availability)
- **Primary**: TS: Prophet (captures operational cycles)
- **Secondary**: ML: RandomForest (cross-agency benchmarking)
- **Why**: Individual operational patterns matter

#### **Safety KPIs** (Incident rates, Injuries)
- **Primary**: ML: RandomForest (rare events benefit from cross-series)
- **Secondary**: TS: SARIMA (statistical confidence)
- **Why**: Small numbers need robust statistical approach

---

## 🚨 Model Availability Rules

### **Why Some Models Don't Appear:**

#### **Time Series Models Missing**
- **Insufficient Data**: Series needs enough historical points
- **No Seasonal Pattern**: Some KPIs don't have clear seasonality
- **Training Requirements**: Prophet/SARIMA need minimum data length

#### **Machine Learning Always Available**
- **Cross-Series Training**: Uses patterns from ALL KPIs
- **Feature Engineering**: 58 engineered features enable prediction
- **Robust Design**: Works even with limited individual series data

---

## 📊 Practical Model Selection Examples

### **Example 1: NYC Transit - Elevator Availability**
**Available Models**: ML: All, TS: Prophet, TS: SARIMA
**Recommendation**: 
- Primary: TS: Prophet (operational cycles)
- Secondary: ML: RandomForest (cross-agency comparison)
**Why**: Elevator maintenance has clear operational patterns

### **Example 2: Bridges & Tunnels - Traffic Volume**
**Available Models**: ML: All, TS: Limited
**Recommendation**:
- Primary: ML: RandomForest (weather, events impact)
- Secondary: ML: XGBoost (fast updates)
**Why**: Traffic influenced by external factors, limited historical data

### **Example 3: LIRR - On-Time Performance**
**Available Models**: ML: All, TS: Prophet, TS: SARIMA
**Recommendation**:
- Primary: TS: Prophet (schedule seasonality)
- Secondary: ML: RandomForest (system-wide factors)
**Why**: Rail schedules have strong seasonal/weekly patterns

---

## 🎨 Dashboard Interpretation Guide

### **When Models Agree (Forecasts Similar)**
- **High Confidence**: Multiple approaches converge
- **Action**: Use consensus forecast for planning
- **Risk**: Low uncertainty, proceed with confidence

### **When Models Disagree (Forecasts Different)**
- **High Uncertainty**: Different methods see different patterns
- **Action**: Use ensemble average, plan conservatively
- **Risk**: High uncertainty, prepare multiple scenarios

### **Confidence Intervals (Prophet Only)**
- **Narrow Bands**: Reliable forecast, plan aggressively
- **Wide Bands**: High uncertainty, plan conservatively
- **Trend**: More important than exact values

---

## 🛠️ Troubleshooting Model Selection

### **"No Models Available"**
**Cause**: Selected combination has insufficient training data
**Solution**: 
1. Try different agency with same KPI
2. Select more common KPI (top 25 indicators)
3. Check if training completed successfully

### **"Only ML Models Available"**
**Cause**: Time series models need more historical data
**Solution**:
1. Use ML models (they're often more accurate anyway)
2. RandomForest provides excellent forecasts
3. Consider this combination for ensemble with other KPIs

### **"Poor Forecast Quality"**
**Cause**: KPI may be influenced by external factors
**Solution**:
1. Try different models for comparison
2. Use ensemble of multiple models
3. Consider operational context (strikes, weather, events)

---

## 🎯 Quick Decision Tree

```
Is this a percentage-based KPI (0-100%)?
├─ YES: Start with Prophet (if available), add RandomForest
└─ NO: Start with RandomForest, add Prophet for seasonality

Is forecast for operational planning (<6 months)?
├─ YES: Emphasize time series models
└─ NO: Emphasize machine learning models

Is accuracy critical for budget/safety?
├─ YES: Use RandomForest + Prophet ensemble
└─ NO: XGBoost for quick insights

Do you see seasonal patterns in historical data?
├─ YES: Prophet essential
└─ NO: Focus on ML models
```

Remember: **RandomForest is your safest bet** - it has 100x better accuracy than time series for cross-indicator prediction! 🎯
