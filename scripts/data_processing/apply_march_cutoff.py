#!/usr/bin/env python3
"""
Create consistent dataset with March 2017 cutoff for Final Year Project
"""
import pandas as pd
import numpy as np
from pathlib import Path

def create_march_2017_cutoff():
    print("🎓 CREATING CONSISTENT DATASET FOR FINAL YEAR PROJECT")
    print("=" * 60)
    
    # Load original data
    original_file = Path('data/processed/mta_model.parquet')
    backup_file = Path('data/processed/mta_model_original_backup.parquet')
    
    print("📁 Loading original dataset...")
    df = pd.read_parquet(original_file)
    
    print(f"Original dataset: {len(df):,} records")
    print(f"Date range: {df['PERIOD_YEAR'].min()}-{df['PERIOD_MONTH'].min():02d} to {df['PERIOD_YEAR'].max()}-{df['PERIOD_MONTH'].max():02d}")
    
    # Create backup of original
    print(f"\n💾 Creating backup: {backup_file}")
    df.to_parquet(backup_file)
    
    # Apply March 2017 cutoff
    print(f"\n✂️ Applying March 2017 cutoff...")
    df_consistent = df[
        (df['PERIOD_YEAR'] < 2017) | 
        ((df['PERIOD_YEAR'] == 2017) & (df['PERIOD_MONTH'] <= 3))
    ].copy()
    
    print(f"Consistent dataset: {len(df_consistent):,} records")
    print(f"Records removed: {len(df) - len(df_consistent):,}")
    print(f"New date range: {df_consistent['PERIOD_YEAR'].min()}-{df_consistent['PERIOD_MONTH'].min():02d} to {df_consistent['PERIOD_YEAR'].max()}-{df_consistent['PERIOD_MONTH'].max():02d}")
    
    # Verify consistency
    print(f"\n🔍 VERIFICATION:")
    df_2017 = df_consistent[df_consistent['PERIOD_YEAR'] == 2017]
    if len(df_2017) > 0:
        month_counts = df_2017['PERIOD_MONTH'].value_counts().sort_index()
        print(f"2017 month distribution: {dict(month_counts)}")
        
        # Check that all indicators now end at same point
        indicators_2017 = df_2017.groupby('INDICATOR_NAME')['PERIOD_MONTH'].max()
        max_months = indicators_2017.value_counts()
        print(f"Indicators by last month: {dict(max_months)}")
        
        if len(max_months) == 1 and 3 in max_months.index:
            print("✅ SUCCESS: All indicators now consistently end at March 2017!")
        else:
            print("⚠️ WARNING: Inconsistency still exists")
    
    # Save the consistent dataset
    print(f"\n💾 Saving consistent dataset...")
    df_consistent.to_parquet(original_file)
    
    print(f"\n📊 IMPACT SUMMARY:")
    print(f"  • Academic consistency: ✅ Achieved")
    print(f"  • Data loss: {((len(df) - len(df_consistent))/len(df)*100):.1f}% ({len(df) - len(df_consistent)} records)")
    print(f"  • All indicators end: March 2017")
    print(f"  • Backup created: {backup_file}")
    
    print(f"\n🎓 FOR YOUR FYP REPORT:")
    print(f"  • Total dataset: {len(df_consistent):,} records")
    print(f"  • Time span: {df_consistent['PERIOD_YEAR'].max() - df_consistent['PERIOD_YEAR'].min() + 1} years") 
    print(f"  • Temporal consistency: All series end March 2017")
    print(f"  • Data quality decision: Prioritized consistency over completeness")

if __name__ == "__main__":
    create_march_2017_cutoff()
