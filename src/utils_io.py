import pandas as pd
import sys
import os

# Add the src directory to the path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RAW

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, encoding='ISO-8859-1')
    # Parse YYYY_MM if present; else build from year+month
    if "YYYY_MM" in df.columns:
        # try standard YYYY-MM format; fallback to coerce
        try:
            df["YYYY_MM"] = pd.to_datetime(df["YYYY_MM"], format="%Y-%m")
        except Exception:
            df["YYYY_MM"] = pd.to_datetime(df["YYYY_MM"], errors="coerce")
    elif {"PERIOD_YEAR","PERIOD_MONTH"}.issubset(df.columns):
        df["YYYY_MM"] = pd.to_datetime(
            df["PERIOD_YEAR"].astype(str) + "-" + df["PERIOD_MONTH"].astype(str).str.zfill(2),
            format="%Y-%m")
    else:
        raise ValueError("Expected either YYYY_MM or (PERIOD_YEAR, PERIOD_MONTH) in CSV.")
    return df
