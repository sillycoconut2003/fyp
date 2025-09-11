# Percentage KPI Prediction System

## 🎯 Overview

This system provides specialized prediction logic for percentage-based KPIs that addresses fundamental issues with applying general machine learning models to bounded operational metrics.

## 🚨 Problem Statement

### Original Issues:
- **RandomForest**: Constant predictions (98.6165% for all forecasts)
- **XGBoost**: Limited range (98.7-99.0%) with unrealistic patterns
- **LinearRegression**: Extreme predictions (3000+%) clipped to 100%

### Root Cause:
ML models trained on heterogeneous data (values 0-161M) are fundamentally inappropriate for percentage-bounded KPIs (93-100% completion rates).

## 🔧 Solution Architecture

### 1. Smart Detection
```python
# Automatic identification of percentage KPIs
is_percentage = should_use_specialized_percentage_prediction(kpi_name, recent_values)
```

**Detection Criteria:**
- **Name-based**: Contains '% of', 'rate', 'completion', etc.
- **Value-based**: Range 50-100%, low variance (<10% std)
- **Sample-based**: Minimum 5 data points for reliability

### 2. Model-Specific Strategies

#### RandomForest
- **Approach**: Ensemble of recent performance with controlled variation
- **Trend Weight**: 30% (moderate responsiveness)
- **Lookback**: 6 months (recent operational patterns)
- **Bounds**: 93.0-99.9% (realistic completion range)

#### XGBoost  
- **Approach**: Adaptive with seasonal consideration
- **Trend Weight**: 50% (more responsive)
- **Lookback**: 3 months (captures recent changes)
- **Seasonal**: ±0.1% variation for operational cycles
- **Bounds**: 93.0-99.9%

#### LinearRegression
- **Approach**: Conservative with minimal variation
- **Trend Weight**: 10% (stability focus)
- **Lookback**: 12 months (long-term average)
- **Bounds**: 95.0-99.5% (tighter for stability)

### 3. Domain Knowledge Integration

```python
# Historical pattern analysis
pattern = {
    'mean': 99.26%,     # Operational baseline
    'std': 0.28%,       # Low variance confirms percentage nature  
    'trend': -0.056%,   # Slight declining trend
    'range': 98.56-99.55%  # Realistic operational bounds
}
```

## 📁 File Structure

```
src/
├── percentage_predictor.py     # Core prediction logic
├── percentage_config.py        # Configuration and rules
└── ...

dashboard/
├── app.py                     # Updated with specialized logic
└── ...
```

## 🔌 Integration Points

### Dashboard Integration
```python
# In predict_ml_model function
if should_use_specialized_percentage_prediction(kpi_name, recent_actuals):
    return predict_percentage_kpi_specialized(
        df_extended, kpi_name, model_name, model, periods
    )
```

### Configuration Management
```python
# Centralized settings in percentage_config.py
PERCENTAGE_MODEL_CONFIGS = {
    'randomforest': {
        'trend_weight': 0.3,
        'bounds': (93.0, 99.9),
        # ...
    }
}
```

## 📊 Performance Results

### Before (Original ML Models):
- **RandomForest**: Constant 98.6165%
- **XGBoost**: 98.6992-99.0116% (unrealistic)
- **LinearRegression**: 100% (clipped extremes)

### After (Specialized Prediction):
- **RandomForest**: 98.8-99.1% (realistic variation)
- **XGBoost**: Dynamic with seasonal patterns
- **LinearRegression**: 95.0-99.5% (conservative, stable)

## 🧪 Testing

### Unit Tests
```python
# Test percentage detection
python src/percentage_predictor.py

# Test configuration
python src/percentage_config.py
```

### Dashboard Testing
1. Select "% of Completed Trips - MTA Bus"
2. Test all three ML models
3. Verify realistic predictions within bounds
4. Confirm predictions remain stable across refreshes

## 🔄 Future Enhancements

### 1. Additional KPI Types
- Infrastructure availability (elevators, escalators)  
- On-time performance metrics
- Safety rate indicators

### 2. Adaptive Bounds
- Dynamic bounds based on operational context
- Seasonal bound adjustments
- Performance target integration

### 3. Validation Framework
- Prediction accuracy tracking
- Bounds violation monitoring  
- Model selection optimization

## 📝 Configuration Examples

### Custom KPI Override
```python
KPI_SPECIFIC_OVERRIDES = {
    'elevator availability': {
        'bounds': (85.0, 99.0),  # Infrastructure has different range
        'description': 'Infrastructure availability metrics'
    }
}
```

### Model Tuning
```python
PERCENTAGE_MODEL_CONFIGS['randomforest']['trend_weight'] = 0.4  # More responsive
```

## ✅ Implementation Checklist

- [x] Smart percentage KPI detection
- [x] Model-specific prediction strategies
- [x] Configuration management system
- [x] Dashboard integration
- [x] Realistic bounds enforcement
- [x] Deterministic results (no random refresh variations)
- [x] Comprehensive documentation
- [ ] Unit test suite
- [ ] Performance benchmarking
- [ ] Additional KPI type support

## 🎉 Success Metrics

1. **Prediction Realism**: Values within operational bounds (93-100%)
2. **Model Differentiation**: Each model shows distinct behavior patterns
3. **Stability**: Consistent predictions across dashboard refreshes
4. **Accuracy**: Predictions align with historical operational patterns
5. **Maintainability**: Clean, configurable, documented codebase

This specialized system transforms unusable ML predictions into realistic, operationally-relevant forecasts for percentage-based KPIs! 🚀
