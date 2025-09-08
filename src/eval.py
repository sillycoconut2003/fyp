import numpy as np
import pandas as pd

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true)-np.array(y_pred))))

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)

def target_gap(forecast: pd.Series, target: pd.Series, desired_change: pd.Series = None) -> pd.Series:
    gap = forecast - target
    if desired_change is None:
        return gap
    # If lower-is-better, invert sign so positive means "good"
    sign = desired_change.map({"Decrease": -1, "Increase": 1}).fillna(1)
    return gap * sign
