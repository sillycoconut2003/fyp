#!/usr/bin/env python3
"""
Quick training status check
"""
import sys
import os
from pathlib import Path
import pickle

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_training_status():
    print("🔍 Time Series Training Status Check")
    print("=" * 50)
    
    models_dir = Path("models/time_series")
    
    if not models_dir.exists():
        print("❌ Models directory doesn't exist yet")
        print("🔄 Training is likely still in progress...")
        return
    
    # Check for saved models
    prophet_file = models_dir / "prophet_models.pkl"
    sarima_file = models_dir / "sarima_models.pkl"
    
    if prophet_file.exists():
        try:
            with open(prophet_file, 'rb') as f:
                prophet_models = pickle.load(f)
            print(f"✅ Prophet models: {len(prophet_models)} series trained")
        except:
            print("❌ Error reading Prophet models file")
    else:
        print("⏳ Prophet models not saved yet")
    
    if sarima_file.exists():
        try:
            with open(sarima_file, 'rb') as f:
                sarima_models = pickle.load(f)
            print(f"✅ SARIMA models: {len(sarima_models)} series trained")
        except:
            print("❌ Error reading SARIMA models file")
    else:
        print("⏳ SARIMA models not saved yet")
    
    # Show file sizes if they exist
    for file_path in [prophet_file, sarima_file]:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"📁 {file_path.name}: {size_mb:.1f} MB")

if __name__ == "__main__":
    check_training_status()
