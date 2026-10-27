"""Data loading, cleaning, validation and splitting.

Everything that is *not* a modelling decision lives here, so that the notebooks,
the training script and the prediction function all use exactly the same rules.
"""
from pathlib import Path
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Project-wide constants
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "insurance.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "insurance_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "medical_insurance_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_URL = (
    "https://raw.githubusercontent.com/stedy/"
    "Machine-Learning-with-R-datasets/master/insurance.csv"
)

TARGET = "charges"
FEATURE_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]
NUMERICAL_COLUMNS = ["age", "bmi", "children"]
CATEGORICAL_COLUMNS = ["sex", "smoker", "region"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

ALLOWED_CATEGORIES = {
    "sex": {"female", "male"},
    "smoker": {"no", "yes"},
    "region": {"northeast", "northwest", "southeast", "southwest"},
}
# Impossible values -> error.
HARD_LIMITS = {"age": (0, 120), "bmi": (10, 70), "children": (0, 10)}
# Range seen in the training data -> warning only (the model would be extrapolating).
TRAINING_RANGES = {"age": (18, 64), "bmi": (15, 54), "children": (0, 5)}


# ----------------------------------------------------------------------------
# Loading and cleaning
# ----------------------------------------------------------------------------
def load_raw_data(path=RAW_DATA_PATH):
    """Read the raw CSV into a DataFrame (falls back to the GitHub URL if missing)."""
    path = Path(path)
    source = path if path.exists() else DATA_URL
    return pd.read_csv(source)


def clean_data(df):
    """Return a cleaned copy of the raw data.

    The only cleaning step needed is dropping exact duplicate rows
    (see notebook 02 for the justification). Extreme charges / BMI values are
    kept because they are legitimate observations.
    """
    return df.drop_duplicates().reset_index(drop=True)


def load_clean_data(path=RAW_DATA_PATH):
    """Load the raw data and apply `clean_data`."""
    return clean_data(load_raw_data(path))


# ----------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------
def split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Separate X / y and split into train and test sets (80/20, seed 42)."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ----------------------------------------------------------------------------
# Validation of new observations (used by predict.py and the tests)
# ----------------------------------------------------------------------------
def validate_features(data):
    """Validate input for prediction and return a clean DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Must contain the columns in FEATURE_COLUMNS (extra columns are ignored).

    Raises
    ------
    ValueError
        If columns are missing, values are missing / non-numeric, categories are
        unknown, or numbers are impossible.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Input DataFrame has no rows.")

    missing_cols = [c for c in FEATURE_COLUMNS if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = data[FEATURE_COLUMNS].copy()
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Missing values found in columns: {bad}")

    # Numeric columns: must be numbers inside plausible limits.
    for col in NUMERICAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' must be numeric.")
        low, high = HARD_LIMITS[col]
        if ((df[col] < low) | (df[col] > high)).any():
            raise ValueError(f"Column '{col}' must be between {low} and {high}.")
        if col in ("age", "children") and (df[col] % 1 != 0).any():
            raise ValueError(f"Column '{col}' must be a whole number.")
        t_low, t_high = TRAINING_RANGES[col]
        if ((df[col] < t_low) | (df[col] > t_high)).any():
            warnings.warn(
                f"'{col}' is outside the training range [{t_low}, {t_high}]; "
                "the prediction is an extrapolation and may be unreliable.",
                stacklevel=2,
            )

    # Categorical columns: normalise text, then check allowed values.
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str).str.strip().str.lower()
        unknown = set(df[col]) - ALLOWED_CATEGORIES[col]
        if unknown:
            raise ValueError(
                f"Column '{col}' has invalid values {sorted(unknown)}. "
                f"Allowed: {sorted(ALLOWED_CATEGORIES[col])}"
            )

    df["age"] = df["age"].astype(int)
    df["children"] = df["children"].astype(int)
    return df.reset_index(drop=True)
