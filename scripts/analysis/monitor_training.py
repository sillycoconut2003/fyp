#!/usr/bin/env python3
"""
Monitor time series training progress
"""
import time
from pathlib import Path

def monitor_training():
    models_dir = Path("models/time_series")
    
    print("🔍 Monitoring Time Series Training Progress...")
    print("📁 Checking models directory:", models_dir)
    print("=" * 50)
    
    while True:
        if models_dir.exists():
            files = list(models_dir.glob("*.pkl"))
            if files:
                print(f"📊 Found {len(files)} model files saved")
                for f in files:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"   - {f.name}: {size_mb:.1f} MB")
            else:
                print("⏳ No model files saved yet...")
        else:
            print("📁 Models directory not created yet...")
        
        print("=" * 50)
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")
