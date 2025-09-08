import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to the path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import COLS, INTERIM, NUMS

def _ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def impute_zero_targets(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2 = _ensure_numeric(df2, ["MONTHLY_TARGET","YTD_TARGET","MONTHLY_ACTUAL","YTD_ACTUAL"])
    for col in ["MONTHLY_TARGET","YTD_TARGET"]:
        if col not in df2.columns:
            continue
        mask = (df2[col] == 0) | (df2[col].isna())
        med = (df2
               .groupby([COLS["agency"], COLS["indicator"]])[col]
               .transform(lambda s: np.nanmedian(s.replace(0, np.nan))))
        df2.loc[mask, col] = df2.loc[mask, col].fillna(med[mask])
    return df2

def maybe_log1p(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in ["MONTHLY_ACTUAL","YTD_ACTUAL"]:
        if col in df2.columns and (df2[col] >= 0).all() and (df2[col].max(skipna=True) > 1000):
            df2[col+"_log1p"] = np.log1p(df2[col])
    return df2

def run_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = impute_zero_targets(df)
    df = maybe_log1p(df)
    # FREQUENCY sometimes constant; safe to drop to reduce noise
    if "FREQUENCY" in df.columns and df["FREQUENCY"].nunique(dropna=True) <= 1:
        df = df.drop(columns=["FREQUENCY"])
    df.to_parquet(INTERIM, index=False)
    return df
