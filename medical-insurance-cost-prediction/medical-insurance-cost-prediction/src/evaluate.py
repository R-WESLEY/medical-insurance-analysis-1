"""Evaluation helpers: RMSE, MAE and R-squared."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    """Return RMSE, MAE and R2 as a dictionary."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Evaluate a fitted model on both datasets; returns a 2-row DataFrame."""
    rows = []
    for dataset, X, y in (("Training", X_train, y_train), ("Testing", X_test, y_test)):
        metrics = regression_metrics(y, model.predict(X))
        rows.append({"Model": name, "Dataset": dataset, **metrics})
    return pd.DataFrame(rows)
