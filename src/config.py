from pathlib import Path

# Base paths
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "MTA_Performance_Agencies.csv"
INTERIM = BASE / "data" / "interim" / "mta_clean.parquet"
PROCESSED = BASE / "data" / "processed" / "mta_model.parquet"

# Canonical column names in the dataset. Adjust if your CSV differs.
COLS = {
    "id": "INDICATOR_SEQ",
    "agency": "AGENCY_NAME",
    "indicator": "INDICATOR_NAME",
    "category": "CATEGORY",
    "freq": "FREQUENCY",
    "desired": "DESIRED_CHANGE",
    "m_tgt": "MONTHLY_TARGET",
    "m_act": "MONTHLY_ACTUAL",
    "ytd_tgt": "YTD_TARGET",
    "ytd_act": "YTD_ACTUAL",
    "yr": "PERIOD_YEAR",
    "mo": "PERIOD_MONTH",
    "ym": "YYYY_MM",
}
