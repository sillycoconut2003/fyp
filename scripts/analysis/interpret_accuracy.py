#!/usr/bin/env python3
"""
Analyze MTA data context to understand model accuracy metrics
"""
import pandas as pd
import numpy as np

def analyze_data_context():
    print("📊 MTA DATA CONTEXT & ACCURACY INTERPRETATION")
    print("=" * 60)
    
    # Load the data
    df = pd.read_parquet('data/processed/mta_model.parquet')
    
    print("\n🎯 TARGET VARIABLE (MONTHLY_ACTUAL) ANALYSIS:")
    print("-" * 45)
    
    actual_values = df['MONTHLY_ACTUAL']
    
    print(f"Mean value: {actual_values.mean():,.0f}")
    print(f"Median value: {actual_values.median():,.0f}")
    print(f"Minimum value: {actual_values.min():,.0f}")
    print(f"Maximum value: {actual_values.max():,.0f}")
    print(f"Standard deviation: {actual_values.std():,.0f}")
    print(f"75th percentile: {actual_values.quantile(0.75):,.0f}")
    print(f"95th percentile: {actual_values.quantile(0.95):,.0f}")
    
    print(f"\n📈 SAMPLE ACTUAL VALUES:")
    print(actual_values.head(10).values)
    
    print(f"\n🔍 ACCURACY INTERPRETATION:")
    print("-" * 35)
    
    # Model performance interpretation
    rf_mae = 12651.07
    xgb_mae = 49445.94
    lr_mae = 147061.44
    
    mean_actual = actual_values.mean()
    
    print(f"\n🥇 RandomForest MAE: {rf_mae:,.0f}")
    print(f"   → As % of mean value: {(rf_mae/mean_actual)*100:.2f}%")
    print(f"   → Typical prediction error: ±{rf_mae:,.0f} units")
    
    print(f"\n🥈 XGBoost MAE: {xgb_mae:,.0f}")
    print(f"   → As % of mean value: {(xgb_mae/mean_actual)*100:.2f}%")
    print(f"   → Typical prediction error: ±{xgb_mae:,.0f} units")
    
    print(f"\n🥉 LinearRegression MAE: {lr_mae:,.0f}")
    print(f"   → As % of mean value: {(lr_mae/mean_actual)*100:.2f}%")
    print(f"   → Typical prediction error: ±{lr_mae:,.0f} units")
    
    print(f"\n💡 WHAT THIS MEANS:")
    print("-" * 20)
    print(f"• MTA performance values range from {actual_values.min():,.0f} to {actual_values.max():,.0f}")
    print(f"• RandomForest typically predicts within ±{rf_mae:,.0f} of actual value")
    print(f"• Given the scale of MTA data, this is {(rf_mae/mean_actual)*100:.1f}% relative error")
    
    if (rf_mae/mean_actual)*100 < 5:
        print("• ✅ This is EXCELLENT accuracy for this type of data!")
    elif (rf_mae/mean_actual)*100 < 10:
        print("• ✅ This is GOOD accuracy for this type of data!")
    else:
        print("• ⚠️ This suggests room for improvement")
        
    print(f"\n🎯 MAPE EXPLANATION:")
    print("-" * 20)
    print("• MAPE can be misleading when actual values vary widely")
    print("• Very large MAPE values often indicate division by small numbers")
    print("• Focus on MAE and RMSE for more reliable accuracy assessment")

if __name__ == "__main__":
    analyze_data_context()
