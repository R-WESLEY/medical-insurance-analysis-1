"""Feature engineering and the scikit-learn preprocessing pipeline.

Preprocessing  = making the data usable by a model (encoding, scaling).
Feature engineering = creating NEW columns that may carry useful signal
                      (BMI categories, age groups, interactions).

All engineered features here are *stateless* (they use fixed rules, never
statistics learned from data), so they cannot leak information from the test set.
Anything that learns from data (scaler, encoder) lives inside the Pipeline and is
fitted on the training data only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data_preprocessing import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS

# Fixed (domain-knowledge) bin edges. right=False -> intervals are [low, high).
BMI_BINS = [0, 18.5, 25, 30, np.inf]
BMI_LABELS = ["underweight", "normal", "overweight", "obese"]
AGE_BINS = [0, 30, 40, 50, np.inf]
AGE_LABELS = ["18-29", "30-39", "40-49", "50+"]

# Which engineered columns are numeric and which are categorical.
ENGINEERED_NUMERIC = {"smoker_bmi", "smoker_age", "smoker_obese"}
ENGINEERED_CATEGORICAL = {"bmi_category", "age_group"}
ALL_ENGINEERED = ("bmi_category", "age_group", "smoker_bmi", "smoker_age", "smoker_obese")


def add_engineered_features(X, features=ALL_ENGINEERED):
    """Return a copy of X with the requested engineered columns added.

    features : iterable of names from ALL_ENGINEERED
        bmi_category : underweight / normal / overweight / obese
        age_group    : 18-29 / 30-39 / 40-49 / 50+
        smoker_bmi   : bmi if the person smokes, else 0
        smoker_age   : age if the person smokes, else 0
        smoker_obese : 1 if smoker AND bmi >= 30, else 0
    """
    X = X.copy()
    is_smoker = (X["smoker"] == "yes").astype(int)
    if "bmi_category" in features:
        X["bmi_category"] = pd.cut(X["bmi"], BMI_BINS, labels=BMI_LABELS, right=False).astype(str)
    if "age_group" in features:
        X["age_group"] = pd.cut(X["age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    if "smoker_bmi" in features:
        X["smoker_bmi"] = is_smoker * X["bmi"]
    if "smoker_age" in features:
        X["smoker_age"] = is_smoker * X["age"]
    if "smoker_obese" in features:
        X["smoker_obese"] = is_smoker * (X["bmi"] >= 30).astype(int)
    return X


def build_preprocessor(engineered=(), scale_numeric=True):
    """Build the preprocessing Pipeline.

    Steps
    -----
    1. add_features : create the engineered columns requested in `engineered`.
    2. encode_scale : ColumnTransformer
         - numeric columns   -> StandardScaler (or passthrough if scale_numeric=False)
         - sex, smoker       -> one-hot with one dummy (binary 0/1 column)
         - region + other categorical columns -> one-hot, first level dropped

    Dropping one level of each category avoids the "dummy variable trap"
    (perfect collinearity with the intercept), which keeps OLS coefficients
    interpretable.
    """
    engineered = tuple(engineered)
    numeric_cols = NUMERICAL_COLUMNS + [c for c in engineered if c in ENGINEERED_NUMERIC]
    multi_cols = ["region"] + [c for c in engineered if c in ENGINEERED_CATEGORICAL]
    binary_cols = [c for c in CATEGORICAL_COLUMNS if c != "region"]  # sex, smoker

    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    encode_scale = ColumnTransformer(
        transformers=[
            ("numeric", numeric_step, numeric_cols),
            ("binary", OneHotEncoder(drop="if_binary"), binary_cols),
            ("multi", OneHotEncoder(drop="first"), multi_cols),
        ]
    )
    return Pipeline(
        steps=[
            ("add_features", FunctionTransformer(add_engineered_features, kw_args={"features": engineered})),
            ("encode_scale", encode_scale),
        ]
    )
