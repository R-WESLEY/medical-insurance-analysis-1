"""Model definitions, experiments, model selection and saving.

Run the whole workflow from the project root with:

    python -m src.train

What it does
------------
1. Loads and cleans the data, then makes the 80/20 split (seed 42).
2. Builds every candidate model as a Pipeline (preprocessing + estimator).
3. Tunes Ridge and SGD with cross-validation on the TRAINING data only.
4. Scores every candidate with 5-fold CV on the training data, fits it on the
   training data, and records training / testing metrics.
5. Selects the best model by lowest mean CV RMSE (the test set is NOT used to select).
6. Saves the selected pipeline with joblib and writes reports/model_comparison.csv.
"""
import warnings

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.data_preprocessing import (
    MODEL_PATH, RANDOM_STATE, REPORTS_DIR, load_clean_data, split_data,
)
from src.evaluate import evaluate_model
from src.feature_engineering import ALL_ENGINEERED, build_preprocessor

CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SGD_PARAMS = dict(max_iter=1000, tol=1e-3, random_state=RANDOM_STATE)
CV_SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R2": "r2",
}


# ----------------------------------------------------------------------------
# Model builders - every model takes the SAME raw DataFrame as input
# ----------------------------------------------------------------------------
def build_age_only_model():
    """Baseline: charges = w * age + b."""
    select_age = ColumnTransformer([("age", "passthrough", ["age"])])
    return Pipeline([("select_age", select_age), ("model", LinearRegression())])


def build_ols_model(engineered=()):
    """OLS. Scaling does not change OLS predictions, so we keep natural units
    (coefficients are then in dollars per unit of each feature)."""
    return Pipeline([
        ("preprocess", build_preprocessor(engineered, scale_numeric=False)),
        ("model", LinearRegression()),
    ])


def build_sgd_model(engineered=(), scale_numeric=True, **sgd_params):
    """SGD regression with StandardScaler on the numeric columns."""
    params = {**SGD_PARAMS, **sgd_params}
    return Pipeline([
        ("preprocess", build_preprocessor(engineered, scale_numeric=scale_numeric)),
        ("model", SGDRegressor(**params)),
    ])


def build_ridge_model(engineered=(), alpha=1.0):
    """Ridge (L2-regularised) regression on scaled numeric features."""
    return Pipeline([
        ("preprocess", build_preprocessor(engineered, scale_numeric=True)),
        ("model", Ridge(alpha=alpha)),
    ])


# ----------------------------------------------------------------------------
# Tuning (cross-validation on training data only)
# ----------------------------------------------------------------------------
def tune_ridge(X_train, y_train, engineered=ALL_ENGINEERED):
    grid = {"model__alpha": [0.01, 0.1, 1, 10, 100]}
    search = GridSearchCV(build_ridge_model(engineered), grid, cv=CV,
                          scoring=CV_SCORING["RMSE"])
    return search.fit(X_train, y_train)


def tune_sgd(X_train, y_train, engineered=ALL_ENGINEERED):
    grid = {
        "model__alpha": [1e-4, 1e-3, 1e-2],
        "model__eta0": [0.001, 0.01, 0.05],
        "model__learning_rate": ["invscaling", "adaptive"],
    }
    # max_iter=5000: the best 'adaptive' setting needed ~1,560 epochs to meet tol, so the
    # default of 1000 raised a ConvergenceWarning (investigated in notebook 06).
    search = GridSearchCV(build_sgd_model(engineered, max_iter=5000), grid, cv=CV,
                          scoring=CV_SCORING["RMSE"], n_jobs=1)
    with warnings.catch_warnings():
        # Slow-converging grid corners may still warn; we only keep the best model, and
        # notebook 06 checks that the chosen configuration converges.
        warnings.simplefilter("ignore", ConvergenceWarning)
        return search.fit(X_train, y_train)


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def get_candidates(X_train, y_train, ridge_search=None, sgd_search=None):
    """Return {name: unfitted model}. Tuned models use the best CV parameters.

    Pre-computed GridSearchCV objects can be passed in to avoid tuning twice.
    """
    ridge_search = ridge_search or tune_ridge(X_train, y_train)
    sgd_search = sgd_search or tune_sgd(X_train, y_train)
    candidates = {
        "Age-only OLS": build_age_only_model(),
        "OLS": build_ols_model(),
        "SGD": build_sgd_model(),
        "OLS (engineered)": build_ols_model(ALL_ENGINEERED),
        "SGD (engineered)": build_sgd_model(ALL_ENGINEERED),
        "Ridge (engineered, tuned)": clone(ridge_search.best_estimator_),
        "SGD (engineered, tuned)": clone(sgd_search.best_estimator_),
    }
    best_params = {
        "Ridge (engineered, tuned)": ridge_search.best_params_,
        "SGD (engineered, tuned)": sgd_search.best_params_,
    }
    return candidates, best_params


def run_experiments(candidates, X_train, y_train, X_test, y_test):
    """Cross-validate, fit and evaluate every candidate.

    Returns
    -------
    comparison : DataFrame with rows 'CV (5-fold)', 'Training' and 'Testing' per model
    fitted     : {name: model fitted on the full training set}
    cv_summary : DataFrame with the mean and spread of the 5 fold RMSE values per model
    """
    tables, fitted, cv_rows = [], {}, []
    for name, model in candidates.items():
        # Record (rather than ignore) convergence warnings raised while cross-validating.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            cv_out = cross_validate(model, X_train, y_train, cv=CV, scoring=CV_SCORING)
        n_conv_warnings = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        fold_rmse = -cv_out["test_RMSE"]
        cv_rows.append({
            "Model": name,
            "CV_RMSE_mean": fold_rmse.mean(),
            "CV_RMSE_std": fold_rmse.std(ddof=1),
            "CV_convergence_warnings": n_conv_warnings,
        })
        cv_row = pd.DataFrame([{
            "Model": name,
            "Dataset": "CV (5-fold)",
            "RMSE": fold_rmse.mean(),
            "MAE": -cv_out["test_MAE"].mean(),
            "R2": cv_out["test_R2"].mean(),
        }])
        fitted_model = clone(model).fit(X_train, y_train)
        fitted[name] = fitted_model
        tables.append(pd.concat(
            [cv_row, evaluate_model(name, fitted_model, X_train, y_train, X_test, y_test)],
            ignore_index=True))
    return pd.concat(tables, ignore_index=True), fitted, pd.DataFrame(cv_rows)


# Simplest -> most complex. Used only to break ties (see select_best_model).
SIMPLICITY_ORDER = [
    "Age-only OLS", "OLS", "OLS (engineered)", "Ridge (engineered, tuned)",
    "SGD", "SGD (engineered)", "SGD (engineered, tuned)",
]


def select_best_model(cv_summary, n_folds=5):
    """One-standard-error rule, using TRAINING-data cross-validation only.

    1. Find the model with the lowest mean CV RMSE and its standard error.
    2. Every model within one standard error of it is statistically tied.
    3. Among the tied models choose the simplest one (SIMPLICITY_ORDER).

    Rationale: CV differences smaller than the fold-to-fold noise are not evidence
    that one model is truly better, so we prefer the simpler, more interpretable one.
    """
    best = cv_summary.loc[cv_summary["CV_RMSE_mean"].idxmin()]
    threshold = best["CV_RMSE_mean"] + best["CV_RMSE_std"] / n_folds ** 0.5
    tied = cv_summary[cv_summary["CV_RMSE_mean"] <= threshold]["Model"].tolist()
    return min(tied, key=SIMPLICITY_ORDER.index), tied, threshold


def save_model(model, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def main():
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    candidates, best_params = get_candidates(X_train, y_train)
    comparison, fitted, cv_summary = run_experiments(
        candidates, X_train, y_train, X_test, y_test)
    best_name, tied, threshold = select_best_model(cv_summary)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    path = save_model(fitted[best_name])

    print(comparison.round(4).to_string(index=False))
    print("\nCV summary:\n", cv_summary.round(2).to_string(index=False))
    print("\nTuned parameters:", best_params)
    print(f"\nModels within one SE of the best CV RMSE (<= {threshold:.1f}): {tied}")
    print(f"Selected model (simplest among the tied): {best_name}")
    print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
