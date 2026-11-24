
"""Linear machine learning models for the multi-asset engine.

Models:
- Logistic Regression for direction classification.
- Ridge Regression for future return prediction.

Scaling is fitted on training data only through sklearn Pipeline.
The test dataset is never loaded.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_asset"
    / "splits"
)

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"

RESULTS_PATH = SPLIT_DIR / "linear_model_results.csv"
COEFFICIENTS_PATH = SPLIT_DIR / "linear_model_coefficients.csv"

HORIZONS = (5, 10, 20)

METADATA_COLUMNS = [
    "Date",
    "Ticker",
    "AssetName",
    "AssetClass",
    "Sector",
    "Market",
]

TARGET_COLUMNS = [
    f"target_return_{horizon}d"
    for horizon in HORIZONS
] + [
    f"target_direction_{horizon}d"
    for horizon in HORIZONS
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only training and validation datasets."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_PATH}"
        )

    train = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["Date"],
    )

    validation = pd.read_csv(
        VALIDATION_PATH,
        parse_dates=["Date"],
    )

    return train, validation


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Identify predictive features, excluding metadata and targets."""

    excluded = set(
        METADATA_COLUMNS + TARGET_COLUMNS
    )

    return [
        column
        for column in data.columns
        if column not in excluded
    ]


def validate_data(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Validate chronological order and numeric model inputs."""

    if train.empty or validation.empty:
        raise ValueError(
            "Training or validation dataset is empty."
        )

    if train["Date"].max() >= validation["Date"].min():
        raise ValueError(
            "Training and validation dates overlap."
        )

    if set(train["Ticker"].unique()) != set(
        validation["Ticker"].unique()
    ):
        raise ValueError(
            "Training and validation instruments do not match."
        )

    required_columns = feature_columns + TARGET_COLUMNS

    for name, frame in [
        ("Training", train),
        ("Validation", validation),
    ]:
        if frame[required_columns].isna().any().any():
            raise ValueError(
                f"Missing values detected in {name} data."
            )

        values = frame[required_columns].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values detected in {name} data."
            )


def evaluate_classification(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[dict], list[dict]]:
    """Train and evaluate Logistic Regression for each horizon."""

    results = []
    coefficient_results = []

    x_train = train[feature_columns]
    x_validation = validation[feature_columns]

    for horizon in HORIZONS:
        target = f"target_direction_{horizon}d"

        y_train = train[target].astype(int)
        y_validation = validation[target].astype(int)

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )

        model.fit(x_train, y_train)

        predictions = model.predict(x_validation)

        probabilities = model.predict_proba(
            x_validation
        )[:, list(model.classes_).index(1)]

        results.append(
            {
                "task": "classification",
                "horizon": horizon,
                "model": "logistic_regression",
                "accuracy": accuracy_score(
                    y_validation,
                    predictions,
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y_validation,
                    predictions,
                ),
                "precision": precision_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
                "roc_auc": roc_auc_score(
                    y_validation,
                    probabilities,
                ),
                "brier_score": brier_score_loss(
                    y_validation,
                    probabilities,
                ),
                "predicted_positive_rate": predictions.mean(),
            }
        )

        coefficients = (
            model.named_steps["model"].coef_[0]
        )

        for feature, coefficient in zip(
            feature_columns,
            coefficients,
        ):
            coefficient_results.append(
                {
                    "task": "classification",
                    "horizon": horizon,
                    "feature": feature,
                    "coefficient": coefficient,
                }
            )

        print(
            f"Logistic Regression completed: "
            f"{horizon}-day classification"
        )

    return results, coefficient_results


def evaluate_regression(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[dict], list[dict]]:
    """Train and evaluate Ridge Regression for each horizon."""

    results = []
    coefficient_results = []

    x_train = train[feature_columns]
    x_validation = validation[feature_columns]

    for horizon in HORIZONS:
        target = f"target_return_{horizon}d"

        y_train = train[target].astype(float)
        y_validation = validation[target].astype(float)

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    Ridge(alpha=1.0),
                ),
            ]
        )

        model.fit(x_train, y_train)

        predictions = model.predict(x_validation)

        results.append(
            {
                "task": "regression",
                "horizon": horizon,
                "model": "ridge_regression",
                "mae": mean_absolute_error(
                    y_validation,
                    predictions,
                ),
                "rmse": np.sqrt(
                    mean_squared_error(
                        y_validation,
                        predictions,
                    )
                ),
                "r2": r2_score(
                    y_validation,
                    predictions,
                ),
                "prediction_mean": predictions.mean(),
                "prediction_std": predictions.std(),
                "actual_mean": y_validation.mean(),
            }
        )

        coefficients = (
            model.named_steps["model"].coef_
        )

        for feature, coefficient in zip(
            feature_columns,
            coefficients,
        ):
            coefficient_results.append(
                {
                    "task": "regression",
                    "horizon": horizon,
                    "feature": feature,
                    "coefficient": coefficient,
                }
            )

        print(
            f"Ridge Regression completed: "
            f"{horizon}-day regression"
        )

    return results, coefficient_results


def main() -> None:
    """Run linear models and save validation results."""

    print("\nLINEAR MODEL EXPERIMENTS")
    print("=" * 75)

    train, validation = load_data()

    feature_columns = get_feature_columns(train)

    validate_data(
        train,
        validation,
        feature_columns,
    )

    print(f"Training observations:   {len(train):,}")
    print(f"Validation observations: {len(validation):,}")
    print(f"Predictive features:     {len(feature_columns)}")
    print(f"Instruments:             {train['Ticker'].nunique()}")

    print(
        f"Training period:         "
        f"{train['Date'].min()} to {train['Date'].max()}"
    )

    print(
        f"Validation period:       "
        f"{validation['Date'].min()} to {validation['Date'].max()}"
    )

    classification_results, classification_coefficients = (
        evaluate_classification(
            train,
            validation,
            feature_columns,
        )
    )

    regression_results, regression_coefficients = (
        evaluate_regression(
            train,
            validation,
            feature_columns,
        )
    )

    results = pd.DataFrame(
        classification_results + regression_results
    )

    coefficients = pd.DataFrame(
        classification_coefficients + regression_coefficients
    )

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    coefficients.to_csv(
        COEFFICIENTS_PATH,
        index=False,
    )

    print("\nCLASSIFICATION RESULTS")
    print("=" * 90)

    classification_table = results[
        results["task"] == "classification"
    ]

    print(
        classification_table[
            [
                "horizon",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "brier_score",
                "predicted_positive_rate",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nREGRESSION RESULTS")
    print("=" * 90)

    regression_table = results[
        results["task"] == "regression"
    ]

    print(
        regression_table[
            [
                "horizon",
                "mae",
                "rmse",
                "r2",
                "prediction_mean",
                "prediction_std",
                "actual_mean",
            ]
        ].round(5).to_string(index=False)
    )

    print("\nEXPERIMENT VALIDATION")
    print("=" * 90)
    print("Scaling: fitted on training data only")
    print("Validation: evaluated chronologically")
    print("Test data: not loaded or evaluated")

    print(f"\nResults saved: {RESULTS_PATH}")
    print(f"Coefficients saved: {COEFFICIENTS_PATH}")


if __name__ == "__main__":
    main()
