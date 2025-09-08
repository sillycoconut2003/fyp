# 🚇 MTA KPI Analytics Dashboard - User Guide

## Quick Start
1. **Launch Dashboard**: Run `streamlit run dashboard/app.py` in your terminal
2. **Access**: Open http://localhost:8501 in your browser
3. **Navigate**: Use sidebar controls to explore different agencies and KPIs

---

## 🎯 Dashboard Capabilities Overview

Your dashboard provides **comprehensive MTA performance analytics** with three main functionalities:

### 1. **Historical Data Visualization**
- View 9+ years of KPI trends (2008-2017)
- Compare actual vs target performance
- Identify seasonal patterns and anomalies

### 2. **Multi-Model Forecasting**
- **Machine Learning Models**: RandomForest, XGBoost, LinearRegression
- **Time Series Models**: Prophet, SARIMA
- **Ensemble Insights**: Compare multiple forecasting approaches

### 3. **Interactive Analysis**
- Filter by agency and specific KPIs
- Customize forecast horizons (1-24 months)
- Real-time model comparison

---

## 📊 How to Use Each Feature Effectively

### **🏢 Agency Selection**
```
Available Agencies:
• NYC Transit (60.7% of data) - Subway/bus operations
• Long Island Rail Road (14.5%) - Commuter rail
• MTA Bus (12.1%) - Bus operations
• Metro-North Railroad (10.3%) - Regional rail
• Bridges and Tunnels (2.4%) - Infrastructure
```

**Best Practice**: Start with NYC Transit for most comprehensive data, then explore specialized agencies for focused analysis.

### **📈 KPI Selection**
Your dashboard covers **130+ unique indicators** including:

**Service Indicators (~79% of data)**:
- On-time performance metrics
- Completion rates by depot
- Service availability measures

**Safety Indicators (~9% of data)**:
- Accident and injury rates
- Employee safety metrics
- Equipment failure rates

**Usage Tip**: Look for patterns across similar KPIs (e.g., all depot completion rates) to identify operational best practices.

### **🔮 Forecasting Controls**

#### **Forecast Horizon**
- **1-6 months**: Short-term operational planning
- **6-12 months**: Budget and resource allocation
- **12-24 months**: Strategic planning and capacity management

#### **Model Selection**
**Machine Learning Models**:
- **RandomForest** (MAE: 12,651): Best overall accuracy, use for reliable predictions
- **XGBoost** (MAE: 49,445): Fast processing, good for real-time analysis
- **LinearRegression** (MAE: 147,061): Baseline comparison, simple interpretation

**Time Series Models**:
- **Prophet**: Excellent for seasonal patterns and holiday effects
- **SARIMA**: Statistical approach, good for stable time series

---

## 🧭 Strategic Analysis Workflows

### **Workflow 1: Performance Benchmarking**
```
1. Select NYC Transit → "Elevator Availability"
2. View historical trends (should be 85-95%)
3. Compare with "Escalator Availability" 
4. Identify performance gaps and improvement opportunities
```

### **Workflow 2: Cross-Agency Comparison**
```
1. Choose common KPI (e.g., "Employee Lost Time Rate")
2. Analyze across different agencies
3. Identify best-performing agencies
4. Extract operational insights for knowledge transfer
```

### **Workflow 3: Forecasting Analysis**
```
1. Select critical KPI with business impact
2. Generate 12-month forecasts with multiple models
3. Compare ML vs Time Series approaches
4. Use ensemble insights for robust planning
```

### **Workflow 4: Seasonal Pattern Detection**
```
1. Choose percentage-based indicators
2. View full historical timeline
3. Look for recurring seasonal dips/peaks
4. Plan resource allocation accordingly
```

---

## 💡 Key Insights You Can Extract

### **📊 Operational Performance**
- **Baseline Performance**: Most percentage KPIs cluster around 85-95%
- **Consistency**: MTA maintains stable performance across seasons
- **Variability**: Some agencies show more variation than others

### **🔍 Agency Specialization**
- **NYC Transit**: Largest operation with most diverse metrics
- **Rail Services**: LIRR/Metro-North focus on on-time performance
- **MTA Bus**: Depot-specific completion rate tracking
- **Infrastructure**: Bridges & Tunnels emphasize safety metrics

### **📈 Forecasting Reliability**
- **Machine Learning Advantage**: 100x better accuracy than time series for cross-indicator prediction
- **Model Ensemble**: Different models excel at different patterns
- **Confidence Intervals**: Prophet provides uncertainty quantification

### **⚠️ Risk Identification**
- **Zero Target Issues**: 8-13% of records have problematic targets
- **Performance Degradation**: Early warning through trend analysis
- **Outlier Detection**: Identify anomalous performance periods

---

## 🎯 Business Decision Support

### **For Operations Managers**
- **Resource Allocation**: Identify underperforming areas needing attention
- **Performance Targets**: Set realistic goals based on historical patterns
- **Seasonal Planning**: Prepare for predictable performance variations

### **For Strategic Planners**
- **Long-term Trends**: Understand multi-year performance evolution
- **Investment Priorities**: Focus resources on high-impact areas
- **Benchmarking**: Compare performance across similar operations

### **For Data Analysts**
- **Model Validation**: Compare different forecasting approaches
- **Pattern Discovery**: Identify relationships between KPIs
- **Quality Assessment**: Monitor data consistency and completeness

---

## 🔧 Advanced Features

### **Model Performance Comparison**
Your dashboard shows **Mean Absolute Error (MAE)** for each model:
- Lower MAE = Better accuracy
- Compare models for same KPI to choose best approach
- Use ensemble of top models for critical decisions

### **Confidence Intervals**
Prophet models provide uncertainty bounds:
- Wide intervals = High uncertainty, plan conservatively
- Narrow intervals = Reliable forecasts, plan aggressively
- Use for risk management and scenario planning

### **Historical Context**
- **2008-2009**: System ramp-up period, limited data
- **2009-2017**: Mature operations, stable reporting
- **Trend Analysis**: Identify long-term improvement or degradation

---

## 🚀 Power User Tips

### **1. Pattern Recognition**
- Look for KPIs that move together (correlation analysis)
- Identify leading indicators that predict other metrics
- Use cross-agency patterns for operational insights

### **2. Anomaly Detection**
- Sudden performance drops may indicate system issues
- Compare actual vs target to identify systematic problems
- Use forecasts to detect when performance deviates from predictions

### **3. Strategic Planning**
- Combine multiple KPI forecasts for comprehensive planning
- Use worst-case scenarios from confidence intervals for risk planning
- Track forecast accuracy over time to improve planning processes

### **4. Operational Excellence**
- Identify best-performing agencies/time periods for best practice extraction
- Monitor improvement trends to validate intervention effectiveness
- Use real-time forecasting for proactive management

---

## 📋 Troubleshooting

**No Models Available**: Ensure all training scripts have been run
**Empty Data**: Check that processed data exists in data/processed/
**Slow Performance**: Select specific agencies/KPIs rather than viewing all data
**Model Errors**: Verify model files exist in models/ directory

---

Your dashboard transforms 267 trained models and 9+ years of MTA data into actionable business intelligence for transportation management! 🎉
