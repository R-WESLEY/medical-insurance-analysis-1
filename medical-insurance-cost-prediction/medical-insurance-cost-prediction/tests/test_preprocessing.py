"""Basic tests for preprocessing, validation and the prediction workflow.

Run from the project root with:   python -m pytest -v
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_preprocessing import (  # noqa: E402
    FEATURE_COLUMNS, MODEL_PATH, TARGET, load_clean_data, split_data, validate_features,
)
from src.feature_engineering import add_engineered_features, build_preprocessor  # noqa: E402
from src.predict import predict_charges, predict_dataframe  # noqa: E402
from src.train import build_ols_model  # noqa: E402
from src.feature_engineering import ALL_ENGINEERED  # noqa: E402


@pytest.fixture(scope="module")
def clean_df():
    return load_clean_data()


@pytest.fixture
def valid_person():
    return dict(age=40, sex="female", bmi=28.0, children=1, smoker="no", region="northeast")


@pytest.fixture(scope="module")
def trained_model(clean_df):
    """A quickly trained model, so prediction tests do not depend on a saved file."""
    X_train, _, y_train, _ = split_data(clean_df)
    return build_ols_model(ALL_ENGINEERED).fit(X_train, y_train)


# ----------------------------- data loading / cleaning ----------------------
def test_clean_data_has_expected_columns_and_no_problems(clean_df):
    assert list(clean_df.columns) == FEATURE_COLUMNS + [TARGET]
    assert clean_df.isna().sum().sum() == 0
    assert clean_df.duplicated().sum() == 0


def test_split_is_80_20_and_reproducible(clean_df):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    assert len(X_train) + len(X_test) == len(clean_df)
    assert abs(len(X_test) / len(clean_df) - 0.2) < 0.01
    X_train2, _, _, _ = split_data(clean_df)
    pd.testing.assert_frame_equal(X_train, X_train2)
    assert TARGET not in X_train.columns


# ----------------------------- input validation -----------------------------
def test_validate_accepts_valid_input_and_normalises_text(valid_person):
    valid_person.update(sex=" Female ", smoker="NO")
    out = validate_features(pd.DataFrame([valid_person]))
    assert out.loc[0, "sex"] == "female" and out.loc[0, "smoker"] == "no"
    assert list(out.columns) == FEATURE_COLUMNS


def test_validate_rejects_missing_column(valid_person):
    valid_person.pop("bmi")
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_features(pd.DataFrame([valid_person]))


def test_validate_rejects_missing_value(valid_person):
    valid_person["bmi"] = np.nan
    with pytest.raises(ValueError, match="Missing values"):
        validate_features(pd.DataFrame([valid_person]))


@pytest.mark.parametrize("column, bad_value", [
    ("smoker", "maybe"), ("sex", "other"), ("region", "mars"),
    ("age", -5), ("age", 200), ("bmi", 0), ("children", -1), ("bmi", "heavy"),
    ("age", 30.5),
])
def test_validate_rejects_invalid_values(valid_person, column, bad_value):
    valid_person[column] = bad_value
    with pytest.raises(ValueError):
        validate_features(pd.DataFrame([valid_person]))


def test_validate_rejects_non_dataframe_and_empty():
    with pytest.raises(TypeError):
        validate_features({"age": 30})
    with pytest.raises(ValueError):
        validate_features(pd.DataFrame(columns=FEATURE_COLUMNS))


def test_out_of_training_range_warns_but_does_not_fail(valid_person):
    valid_person["age"] = 80
    with pytest.warns(UserWarning, match="outside the training range"):
        validate_features(pd.DataFrame([valid_person]))


# ----------------------------- feature engineering --------------------------
def test_engineered_features_values():
    X = pd.DataFrame([
        dict(age=29, sex="male", bmi=30.0, children=0, smoker="yes", region="northeast"),
        dict(age=50, sex="male", bmi=18.4, children=0, smoker="no", region="northeast"),
    ])
    out = add_engineered_features(X)
    assert out.loc[0, "bmi_category"] == "obese"          # 30.0 is obese ([30, inf))
    assert out.loc[1, "bmi_category"] == "underweight"
    assert out.loc[0, "age_group"] == "18-29" and out.loc[1, "age_group"] == "50+"
    assert out.loc[0, "smoker_bmi"] == 30.0 and out.loc[1, "smoker_bmi"] == 0
    assert out.loc[0, "smoker_age"] == 29 and out.loc[0, "smoker_obese"] == 1
    assert out.loc[1, "smoker_obese"] == 0
    assert list(X.columns) == FEATURE_COLUMNS            # input was not modified


def test_preprocessor_is_fitted_on_training_data_only(clean_df):
    X_train, X_test, _, _ = split_data(clean_df)
    pre = build_preprocessor(scale_numeric=True).fit(X_train)
    scaler = pre.named_steps["encode_scale"].named_transformers_["numeric"]
    np.testing.assert_allclose(scaler.mean_, X_train[["age", "bmi", "children"]].mean().values)
    out = pre.transform(X_test)
    assert out.shape[0] == len(X_test) and np.isfinite(out).all()


# ----------------------------- prediction workflow --------------------------
def test_predictions_are_finite_numbers(trained_model, valid_person):
    value = predict_charges(**valid_person, model=trained_model)
    assert isinstance(value, float) and np.isfinite(value) and value > 0


def test_predict_dataframe_handles_several_rows(trained_model, clean_df):
    preds = predict_dataframe(clean_df[FEATURE_COLUMNS].head(10), model=trained_model)
    assert preds.shape == (10,) and np.isfinite(preds).all()


def test_predict_raises_on_invalid_input(trained_model, valid_person):
    valid_person["smoker"] = "sometimes"
    with pytest.raises(ValueError):
        predict_charges(**valid_person, model=trained_model)


def test_smoker_costs_more_than_non_smoker(trained_model, valid_person):
    """Sanity check: the model should reflect the strongest pattern in the data."""
    non_smoker = predict_charges(**valid_person, model=trained_model)
    smoker = predict_charges(**{**valid_person, "smoker": "yes"}, model=trained_model)
    assert smoker > non_smoker


def test_saved_model_loads_and_predicts(valid_person):
    if not MODEL_PATH.exists():
        pytest.skip("Run `python -m src.train` first to create the saved model.")
    model = joblib.load(MODEL_PATH)
    value = predict_charges(**valid_person, model=model)
    assert np.isfinite(value)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a saved model should predict without warnings
        predict_charges(**valid_person, model=model)
