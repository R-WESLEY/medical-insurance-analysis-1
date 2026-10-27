"""Reusable prediction function.

Usage from another script (run from the project root):

    from src.predict import predict_charges
    estimate = predict_charges(age=35, sex="male", bmi=27.5, children=2,
                               smoker="no", region="northwest")

Command-line demo:

    python -m src.predict

The saved pipeline already contains every preprocessing step, so nothing is
re-implemented here: this module only validates input and calls `pipeline.predict`.
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import MODEL_PATH, validate_features

DISCLAIMER = (
    "Estimates come from a simple statistical model trained on a small public "
    "dataset. They are NOT insurance quotes, guaranteed prices or medical advice."
)


@lru_cache(maxsize=1)
def load_model(path=MODEL_PATH):
    """Load the saved pipeline (cached so repeated calls are fast)."""
    try:
        return joblib.load(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No trained model at {path}. Train one first with: python -m src.train"
        ) from exc


def predict_dataframe(data, model=None):
    """Predict charges for every row of a DataFrame with the expected columns."""
    clean = validate_features(data)
    model = model if model is not None else load_model()
    predictions = np.asarray(model.predict(clean), dtype=float)
    if not np.isfinite(predictions).all():
        raise RuntimeError("The model returned a non-finite prediction.")
    # A linear model can extrapolate below zero; charges cannot be negative.
    return np.clip(predictions, 0, None)


def predict_charges(age, sex, bmi, children, smoker, region, model=None):
    """Estimate the annual medical insurance charge for ONE person (in dollars).

    Raises ValueError for missing / invalid inputs (see validate_features).
    """
    person = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker, "region": region,
    }])
    return float(predict_dataframe(person, model=model)[0])


if __name__ == "__main__":
    examples = [
        dict(age=25, sex="female", bmi=22.0, children=0, smoker="no", region="southwest"),
        dict(age=45, sex="male", bmi=31.5, children=2, smoker="yes", region="southeast"),
        dict(age=60, sex="female", bmi=27.0, children=1, smoker="no", region="northeast"),
    ]
    for person in examples:
        print(f"{person} -> ${predict_charges(**person):,.2f}")
    print("\n" + DISCLAIMER)
