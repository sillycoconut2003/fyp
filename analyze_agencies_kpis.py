#!/usr/bin/env python3
"""
Analyze MTA agencies and their KPIs to understand data availability for multi-agency comparison
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

def analyze_agency_kpis():
    """Analyze agencies and their KPIs with data availability"""
    
    # Load the processed dataset
    data_path = Path("data/processed/mta_model.parquet")
    if not data_path.exists():
        print(f"❌ Dataset not found at {data_path}")
        return
    
    df = pd.read_parquet(data_path)
    print(f"📊 Loaded dataset: {len(df)} records")
    print(f"📅 Date range: {df['YYYY_MM'].min()} to {df['YYYY_MM'].max()}")
    print(f"🏢 Agencies: {df['AGENCY_NAME'].nunique()}")
    print(f"📈 KPIs: {df['INDICATOR_NAME'].nunique()}")
    print("="*80)
    
    # Get all agencies
    agencies = sorted(df['AGENCY_NAME'].unique())
    print(f"\n🏢 **MTA AGENCIES** ({len(agencies)} total):")
    for i, agency in enumerate(agencies, 1):
        agency_records = len(df[df['AGENCY_NAME'] == agency])
        agency_kpis = df[df['AGENCY_NAME'] == agency]['INDICATOR_NAME'].nunique()
        print(f"{i:2d}. {agency}")
        print(f"    📊 {agency_records:,} records | {agency_kpis} unique KPIs")
    
    print("\n" + "="*80)
    
    # Analyze KPIs by agency
    print("\n📈 **KPIs BY AGENCY**:")
    agency_kpis = {}
    
    for agency in agencies:
        agency_data = df[df['AGENCY_NAME'] == agency]
        kpis = {}
        
        for kpi in sorted(agency_data['INDICATOR_NAME'].unique()):
            kpi_clean = kpi.strip()
            records = len(agency_data[agency_data['INDICATOR_NAME'] == kpi])
            kpis[kpi_clean] = records
        
        agency_kpis[agency] = kpis
        
        print(f"\n🏢 **{agency}** ({len(kpis)} KPIs):")
        for kpi, count in kpis.items():
            status = "✅" if count >= 24 else "⚠️" if count >= 12 else "❌"
            print(f"   {status} {kpi} ({count} records)")
    
    print("\n" + "="*80)
    
    # Find common KPIs across agencies
    print("\n🔄 **COMMON KPIs ANALYSIS**:")
    
    # Create a mapping of KPI -> agencies that have it with sufficient data
    kpi_agencies = defaultdict(list)
    
    for agency, kpis in agency_kpis.items():
        for kpi, count in kpis.items():
            if count > 0:  # Has data
                kpi_agencies[kpi].append((agency, count))
    
    # Find KPIs available in multiple agencies
    common_kpis = {}
    for kpi, agency_list in kpi_agencies.items():
        if len(agency_list) >= 2:
            common_kpis[kpi] = agency_list
    
    print(f"\n📊 **KPIs AVAILABLE FOR MULTI-AGENCY COMPARISON** ({len(common_kpis)} total):")
    
    # Sort by number of agencies (most common first)
    sorted_common = sorted(common_kpis.items(), key=lambda x: len(x[1]), reverse=True)
    
    for kpi, agency_list in sorted_common:
        agencies_with_good_data = [(a, c) for a, c in agency_list if c >= 12]
        agencies_with_some_data = [(a, c) for a, c in agency_list if c < 12 and c > 0]
        
        print(f"\n📈 **{kpi}** ({len(agency_list)} agencies)")
        
        if agencies_with_good_data:
            print(f"   ✅ Good data (≥12 records): {len(agencies_with_good_data)} agencies")
            for agency, count in agencies_with_good_data:
                print(f"      • {agency}: {count} records")
        
        if agencies_with_some_data:
            print(f"   ⚠️ Limited data (<12 records): {len(agencies_with_some_data)} agencies")
            for agency, count in agencies_with_some_data:
                print(f"      • {agency}: {count} records")
    
    print("\n" + "="*80)
    
    # Recommend best agency pairs for comparison
    print("\n🎯 **RECOMMENDED AGENCY PAIRS FOR COMPARISON**:")
    
    best_comparisons = []
    
    for kpi, agency_list in sorted_common:
        good_agencies = [(a, c) for a, c in agency_list if c >= 24]  # At least 2 years
        if len(good_agencies) >= 2:
            # Sort by data volume
            good_agencies.sort(key=lambda x: x[1], reverse=True)
            best_comparisons.append((kpi, good_agencies))
    
    # Show top 10 recommendations
    print(f"\n🏆 **TOP 10 COMPARISON OPPORTUNITIES**:")
    for i, (kpi, agencies) in enumerate(best_comparisons[:10], 1):
        print(f"\n{i:2d}. **{kpi}**")
        print(f"    Agencies: {', '.join([a for a, c in agencies[:3]])}")
        total_records = sum(c for a, c in agencies)
        print(f"    Total data: {total_records:,} records")
        print(f"    Best pair: {agencies[0][0]} ({agencies[0][1]} records) vs {agencies[1][0]} ({agencies[1][1]} records)")
    
    print("\n" + "="*80)
    print("✅ **Analysis complete!**")
    print("💡 Use the recommendations above to select agency-KPI combinations with sufficient data.")

if __name__ == "__main__":
    analyze_agency_kpis()