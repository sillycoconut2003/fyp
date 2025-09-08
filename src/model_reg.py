import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def build_rf_and_score(X: pd.DataFrame, y: pd.Series):
    # We keep original categorical cols too for pipeline-based OHE
    cat = [c for c in ["AGENCY_NAME","INDICATOR_NAME"] if c in X.columns]
    num = [c for c in X.columns if c not in cat]

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat),
         ("num", "passthrough", num)]
    )
    model = Pipeline([("pre", pre),
                      ("rf", RandomForestRegressor(n_estimators=400, random_state=42))])
    tscv = TimeSeriesSplit(n_splits=5)
    scores = -cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
    return model, float(np.mean(scores))
