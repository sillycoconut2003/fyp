"""
This module contains configuration settings for the specialized percentage KPI
prediction system, including detection rules, model parameters, and bounds.
"""

# Percentage KPI Detection Rules
PERCENTAGE_INDICATORS = [
    '% of completed trips',
    'completion rate', 
    'availability',
    'on-time',
    'otp',  # On-Time Performance abbreviation
    'percentage',
    'rate',
    'efficiency', 
    '% of',
    'percent',
    'completion',
    'availability rate'
]

# Statistical thresholds for percentage detection
PERCENTAGE_DETECTION_THRESHOLDS = {
    'min_value': 25.0,      # Minimum value to consider percentage range (lowered for transit OTP)
    'max_value': 100.0,     # Maximum value for percentage range  
    'max_std': 15.0,        # Maximum standard deviation for low variance (increased for seasonal variation)
    'min_samples': 5        # Minimum samples needed for reliable detection
}

# Model-specific configurations for percentage KPI prediction
PERCENTAGE_MODEL_CONFIGS = {
    'randomforest': {
        'description': 'Ensemble approach with moderate variation',
        'trend_weight': 0.3,
        'noise_factor': 0.3,
        'lookback_months': 6,
        'bounds': (95.0, 100.0),
        'seasonal': False,
        'use_case': 'Best for stable operational metrics with slight variation'
    },
    
    'xgboost': {
        'description': 'Adaptive with seasonal patterns',
        'trend_weight': 0.5,
        'noise_factor': 0.4,
        'lookback_months': 3,
        'bounds': (95.0, 100.0),
        'seasonal': True,
        'use_case': 'Best for metrics with seasonal operational patterns'
    },
    
    'linearregression': {
        'description': 'Conservative with minimal variation',
        'trend_weight': 0.1,
        'noise_factor': 0.2,
        'lookback_months': 12,
        'bounds': (97.0, 100.0),
        'seasonal': False,
        'use_case': 'Best for highly stable completion rates'
    },
    
    'ridge': {
        'description': 'Regularized linear with stability',
        'trend_weight': 0.15,
        'noise_factor': 0.12,
        'lookback_months': 10,
        'bounds': (94.5, 99.6),
        'seasonal': False,
        'use_case': 'Alternative to LinearRegression with slight flexibility'
    }
}

# KPI-specific overrides for special cases
KPI_SPECIFIC_OVERRIDES = {
    '% of completed trips': {
        'bounds': (93.0, 99.9),
        'description': 'Bus/transit completion rates'
    },
    
    'elevator availability': {
        'bounds': (85.0, 99.0),
        'description': 'Infrastructure availability metrics'
    },
    
    'escalator availability': {
        'bounds': (85.0, 99.0), 
        'description': 'Infrastructure availability metrics'
    },
    
    'on-time performance': {
        'bounds': (80.0, 99.0),
        'seasonal': True,
        'description': 'Schedule adherence metrics'
    }
}

# Validation settings
VALIDATION_SETTINGS = {
    'min_forecast_months': 1,
    'max_forecast_months': 24,
    'prediction_bounds_buffer': 0.5,  # Buffer for extreme predictions
    'trend_significance_threshold': 0.1  # Minimum trend to consider significant
}

# Debug and logging settings
DEBUG_SETTINGS = {
    'show_historical_analysis': True,
    'show_prediction_range': True,
    'show_model_selection_rationale': False,
    'verbose_bounds_checking': False
}

# Export key configurations for easy access
__all__ = [
    'PERCENTAGE_INDICATORS',
    'PERCENTAGE_DETECTION_THRESHOLDS', 
    'PERCENTAGE_MODEL_CONFIGS',
    'KPI_SPECIFIC_OVERRIDES',
    'VALIDATION_SETTINGS',
    'DEBUG_SETTINGS'
]
