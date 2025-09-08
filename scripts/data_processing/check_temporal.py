#!/usr/bin/env python3
"""
Analyze temporal coverage of the dataset
"""
import pandas as pd
import numpy as np

def analyze_temporal_coverage():
    print("📅 TEMPORAL COVERAGE ANALYSIS")
    print("=" * 50)
    
    df = pd.read_parquet('data/processed/mta_model.parquet')
    
    print("🗓️ YEAR AND MONTH DISTRIBUTION:")
    print("-" * 35)
    
    # Overall year distribution
    year_counts = df['PERIOD_YEAR'].value_counts().sort_index()
    print("Data points per year:")
    for year, count in year_counts.items():
        print(f"  {year}: {count:,} records")
    
    print(f"\nDate range: {df['PERIOD_YEAR'].min()} to {df['PERIOD_YEAR'].max()}")
    print()
    
    # Focus on 2017 data
    print("🔍 DETAILED 2017 ANALYSIS:")
    print("-" * 25)
    
    df_2017 = df[df['PERIOD_YEAR'] == 2017]
    
    if len(df_2017) > 0:
        print(f"Total 2017 records: {len(df_2017):,}")
        
        # Month distribution in 2017
        month_counts_2017 = df_2017['PERIOD_MONTH'].value_counts().sort_index()
        print("\n2017 records by month:")
        for month, count in month_counts_2017.items():
            print(f"  Month {month}: {count:,} records")
        
        # Check which indicators have data through which months
        print(f"\n📊 INDICATOR COVERAGE IN 2017:")
        print("-" * 30)
        
        indicators_by_last_month = {}
        for indicator in df_2017['INDICATOR_NAME'].unique():
            indicator_data = df_2017[df_2017['INDICATOR_NAME'] == indicator]
            last_month = indicator_data['PERIOD_MONTH'].max()
            
            if last_month not in indicators_by_last_month:
                indicators_by_last_month[last_month] = []
            indicators_by_last_month[last_month].append(indicator)
        
        for month in sorted(indicators_by_last_month.keys()):
            indicators = indicators_by_last_month[month]
            print(f"\nIndicators ending in Month {month} ({len(indicators)} indicators):")
            for indicator in indicators[:5]:  # Show first 5
                print(f"  • {indicator}")
            if len(indicators) > 5:
                print(f"  ... and {len(indicators) - 5} more")
    
    # Check for data imbalance implications
    print(f"\n⚠️  POTENTIAL ISSUES:")
    print("-" * 20)
    
    # Check last few months of data
    recent_data = df[df['PERIOD_YEAR'] >= 2016]
    
    print("Recent data distribution:")
    recent_summary = recent_data.groupby(['PERIOD_YEAR', 'PERIOD_MONTH']).size().reset_index(name='count')
    recent_summary = recent_summary.sort_values(['PERIOD_YEAR', 'PERIOD_MONTH'])
    
    for _, row in recent_summary.tail(10).iterrows():
        print(f"  {int(row['PERIOD_YEAR'])}-{int(row['PERIOD_MONTH']):02d}: {row['count']:,} records")
    
    print(f"\n� IMPACT ON MODEL TRAINING:")
    print("-" * 25)
    
    # Check current train/test split behavior
    df_sorted = df.sort_values('YYYY_MM')
    
    # Simulate 20% test split (what ML models use)
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    print(f"With 80/20 temporal split:")
    print(f"  Training ends: {train_df['YYYY_MM'].max()}")
    print(f"  Testing starts: {test_df['YYYY_MM'].min()}")
    print(f"  Testing ends: {test_df['YYYY_MM'].max()}")
    
    # Check if 2017 data is in test set
    test_2017 = test_df[test_df['PERIOD_YEAR'] == 2017]
    if len(test_2017) > 0:
        print(f"  ⚠️  2017 data in test set: {len(test_2017)} records")
        month_dist = test_2017['PERIOD_MONTH'].value_counts().sort_index()
        print(f"  Test set 2017 months: {dict(month_dist)}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 15)
    print("1. ✅ GOOD NEWS: Inconsistency likely in test set, not affecting training much")
    print("2. Consider ending all series at March 2017 for consistency")
    print("3. Alternative: Use 2016 as consistent cutoff for both train/test")
    print("4. Monitor if this affects time series model evaluation")

if __name__ == "__main__":
    analyze_temporal_coverage()
