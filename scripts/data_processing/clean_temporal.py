#!/usr/bin/env python3
"""
Optional: Create a cleaned dataset with consistent temporal cutoff
"""
import pandas as pd

def create_consistent_dataset():
    print("🧹 CREATING TEMPORALLY CONSISTENT DATASET")
    print("=" * 50)
    
    df = pd.read_parquet('data/processed/mta_model.parquet')
    
    print(f"Original dataset: {len(df):,} records")
    
    # Option 1: Cut all data at March 2017
    df_march_cutoff = df[
        (df['PERIOD_YEAR'] < 2017) | 
        ((df['PERIOD_YEAR'] == 2017) & (df['PERIOD_MONTH'] <= 3))
    ].copy()
    
    print(f"March 2017 cutoff: {len(df_march_cutoff):,} records ({len(df) - len(df_march_cutoff)} removed)")
    
    # Option 2: Cut all data at December 2016 
    df_2016_cutoff = df[df['PERIOD_YEAR'] <= 2016].copy()
    
    print(f"December 2016 cutoff: {len(df_2016_cutoff):,} records ({len(df) - len(df_2016_cutoff)} removed)")
    
    print(f"\n📊 COMPARISON:")
    print(f"  Current: 2009-2017 (partial)")
    print(f"  March cutoff: 2009-2017 March")  
    print(f"  2016 cutoff: 2009-2016")
    
    # Save the cleaned version if desired
    # df_march_cutoff.to_parquet('data/processed/mta_model_consistent.parquet')
    
    print(f"\n💭 DECISION GUIDE:")
    print("- If models performing well → Keep as-is")
    print("- If perfectionist → Use March 2017 cutoff") 
    print("- If want more data → Keep as-is")
    print("- If publishing research → Consider March 2017 cutoff")

if __name__ == "__main__":
    create_consistent_dataset()
