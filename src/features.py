import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the src directory to the path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import COLS, PROCESSED
from utils_io import load_raw
from preprocess import run_preprocess

def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "YYYY_MM" not in out.columns:
        raise ValueError("YYYY_MM missing. Ensure utils_io.load_raw created it.")
    out["year"] = out["YYYY_MM"].dt.year
    out["month"] = out["YYYY_MM"].dt.month
    out["quarter"] = out["YYYY_MM"].dt.quarter
    return out

def _add_lags_rolls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values([COLS["agency"], COLS["indicator"], "YYYY_MM"]).copy()
    grp = out.groupby([COLS["agency"], COLS["indicator"]], sort=False)["MONTHLY_ACTUAL"]

    out["m_act_lag1"] = grp.shift(1)
    out["m_act_lag3"] = grp.shift(3)
    out["m_act_lag12"] = grp.shift(12)

    # Leakage-safe rolling means: use shifted series so current month isn't included
    out["m_act_roll3"] = grp.shift(1).rolling(3).mean()
    out["m_act_roll6"] = grp.shift(1).rolling(6).mean()
    out["m_act_roll12"] = grp.shift(1).rolling(12).mean()

    return out

def _one_hot_agency(df: pd.DataFrame) -> pd.DataFrame:
    if COLS["agency"] not in df.columns:
        return df
    dummies = pd.get_dummies(df[COLS["agency"]], prefix="AGENCY", drop_first=False, dtype="int8")
    return pd.concat([df, dummies], axis=1)

def _one_hot_top_indicators(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if COLS["indicator"] not in df.columns:
        return df
    counts = df[COLS["indicator"]].value_counts()
    top = set(counts.head(top_n).index)
    tmp = df[COLS["indicator"]].where(df[COLS["indicator"]].isin(top), other="OTHER")
    dummies = pd.get_dummies(tmp, prefix="IND", drop_first=False, dtype="int8")
    return pd.concat([df, dummies], axis=1)

def build_processed(top_n_indicators: int = 25) -> pd.DataFrame:
    df_raw = load_raw()
    df_clean = run_preprocess(df_raw)
    df_feat = _add_calendar(df_clean)
    df_feat = _add_lags_rolls(df_feat)
    df_feat = _one_hot_agency(df_feat)
    df_feat = _one_hot_top_indicators(df_feat, top_n=top_n_indicators)

    # Drop rows that still have NA due to initial lags (keep at least 12 months history for full features)
    df_feat = df_feat.dropna(subset=["m_act_lag1","m_act_lag3","m_act_lag12","m_act_roll3","m_act_roll6","m_act_roll12"])

    # Save
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(PROCESSED, index=False)
    return df_feat

if __name__ == "__main__":
    df = build_processed(top_n_indicators=25)
    print(f"Processed dataset written to: {PROCESSED} with shape {df.shape}")
