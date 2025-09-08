from prophet import Prophet
import pandas as pd

def fit_prophet(series_df: pd.DataFrame):
    # expects columns: YYYY_MM (datetime), MONTHLY_ACTUAL
    df = series_df.rename(columns={"YYYY_MM":"ds","MONTHLY_ACTUAL":"y"})[["ds","y"]]
    m = Prophet(weekly_seasonality=False, daily_seasonality=False)
    m.fit(df)
    return m
