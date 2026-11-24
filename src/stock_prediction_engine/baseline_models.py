
"""Baseline classification and regression experiments.

Uses only training and validation data.
The test set is deliberately never loaded or evaluated.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


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

OUTPUT_PATH = SPLIT_DIR / "baseline_validation_results.csv"

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
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_PATH}"
        )

    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    validation = pd.read_csv(
        VALIDATION_PATH,
        parse_dates=["Date"],
    )

    return train, validation


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Identify model input columns."""

    excluded = set(METADATA_COLUMNS + TARGET_COLUMNS)

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
    """Validate input data before running experiments."""

    if train.empty or validation.empty:
        raise ValueError("Training or validation dataset is empty.")

    if train["Date"].max() >= validation["Date"].min():
        raise ValueError(
            "Training and validation dates overlap or are misordered."
        )

    if set(train["Ticker"].unique()) != set(
        validation["Ticker"].unique()
    ):
        raise ValueError(
            "Training and validation instruments do not match."
        )

    required = feature_columns + TARGET_COLUMNS

    for name, frame in [
        ("Training", train),
        ("Validation", validation),
    ]:
        if frame[required].isna().any().any():
            raise ValueError(
                f"Missing values detected in {name} dataset."
            )

        values = frame[required].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values detected in {name} dataset."
            )


def evaluate_classification(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> list[dict]:
    """Evaluate majority-class baselines for all horizons."""

    results = []

    for horizon in HORIZONS:
        target = f"target_direction_{horizon}d"

        y_train = train[target].astype(int)
        y_validation = validation[target].astype(int)

        # The DummyClassifier learns only the class distribution.
        # It does not use the predictive features.
        model = DummyClassifier(strategy="prior")

        model.fit(
            np.zeros((len(y_train), 1)),
            y_train,
        )

        predictions = model.predict(
            np.zeros((len(y_validation), 1))
        )

        probabilities = model.predict_proba(
            np.zeros((len(y_validation), 1))
        )[:, 1]

        results.append(
            {
                "task": "classification",
                "horizon": horizon,
                "model": "majority_class_prior",
                "train_positive_rate": y_train.mean(),
                "validation_positive_rate": y_validation.mean(),
                "predicted_class": int(model.classes_[
                    np.argmax(model.class_prior_)
                ]),
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
            }
        )

    return results


def evaluate_regression(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> list[dict]:
    """Evaluate simple return-prediction baselines."""

    results = []

    for horizon in HORIZONS:
        target = f"target_return_{horizon}d"

        y_train = train[target].astype(float)
        y_validation = validation[target].astype(float)

        # Global mean calculated using training data only.
        global_mean = y_train.mean()

        # Historical mean calculated separately for each ticker.
        ticker_means = train.groupby("Ticker")[target].mean()

        asset_mean_predictions = (
            validation["Ticker"]
            .map(ticker_means)
            .to_numpy(dtype=float)
        )

        if not np.isfinite(asset_mean_predictions).all():
            raise ValueError(
                f"Missing historical mean for horizon {horizon}."
            )

        predictions = {
            "zero_return": np.zeros(len(validation)),
            "global_training_mean": np.full(
                len(validation),
                global_mean,
            ),
            "per_asset_training_mean": asset_mean_predictions,
        }

        for model_name, prediction in predictions.items():
            results.append(
                {
                    "task": "regression",
                    "horizon": horizon,
                    "model": model_name,
                    "train_mean_return": global_mean,
                    "mae": mean_absolute_error(
                        y_validation,
                        prediction,
                    ),
                    "rmse": np.sqrt(
                        mean_squared_error(
                            y_validation,
                            prediction,
                        )
                    ),
                    "r2": r2_score(
                        y_validation,
                        prediction,
                    ),
                }
            )

    return results


def print_classification_results(
    results: list[dict],
) -> None:
    """Print classification baseline metrics."""

    print("\nCLASSIFICATION BASELINES")
    print("=" * 90)

    columns = [
        "horizon",
        "model",
        "train_positive_rate",
        "validation_positive_rate",
        "predicted_class",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "brier_score",
    ]

    frame = pd.DataFrame(results)

    print(
        frame[columns].round(4).to_string(index=False)
    )


def print_regression_results(
    results: list[dict],
) -> None:
    """Print regression baseline metrics."""

    print("\nREGRESSION BASELINES")
    print("=" * 90)

    columns = [
        "horizon",
        "model",
        "train_mean_return",
        "mae",
        "rmse",
        "r2",
    ]

    frame = pd.DataFrame(results)

    print(
        frame[columns].round(5).to_string(index=False)
    )


def main() -> None:
    """Run all baseline experiments and save validation results."""

    print("\nBASELINE MODEL EXPERIMENTS")
    print("=" * 90)

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
    print(f"Training instruments:    {train['Ticker'].nunique()}")

    print(
        f"Training period:         "
        f"{train['Date'].min()} to {train['Date'].max()}"
    )

    print(
        f"Validation period:       "
        f"{validation['Date'].min()} to {validation['Date'].max()}"
    )

    classification_results = evaluate_classification(
        train,
        validation,
    )

    regression_results = evaluate_regression(
        train,
        validation,
    )

    print_classification_results(
        classification_results
    )

    print_regression_results(
        regression_results
    )

    results = pd.DataFrame(
        classification_results + regression_results
    )

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nEXPERIMENT VALIDATION")
    print("=" * 90)
    print("Training data: Used")
    print("Validation data: Used")
    print("Test data: Not loaded or evaluated")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
