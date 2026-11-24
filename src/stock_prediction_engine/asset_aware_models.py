
"""Train asset-aware linear models for multi-asset stock prediction.

The experiment adds Ticker as a categorical feature to the existing
40 numerical features. It uses train and validation only.
The test set is deliberately not loaded.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------------------

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

RESULTS_PATH = SPLIT_DIR / "asset_aware_model_results.csv"
PREDICTIONS_PATH = SPLIT_DIR / "asset_aware_predictions.csv"
COEFFICIENTS_PATH = SPLIT_DIR / "asset_aware_coefficients.csv"

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
    "target_return_5d",
    "target_return_10d",
    "target_return_20d",
    "target_direction_5d",
    "target_direction_10d",
    "target_direction_20d",
]


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def load_datasets():
    """Load training and validation datasets only."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_PATH}"
        )

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    if train.empty or validation.empty:
        raise ValueError("Training or validation dataset is empty.")

    required_columns = (
        METADATA_COLUMNS
        + TARGET_COLUMNS
    )

    for column in required_columns:
        if column not in train.columns:
            raise ValueError(
                f"Missing column in training dataset: {column}"
            )

        if column not in validation.columns:
            raise ValueError(
                f"Missing column in validation dataset: {column}"
            )

    if train["Ticker"].isna().any():
        raise ValueError("Missing Ticker values in training data.")

    if validation["Ticker"].isna().any():
        raise ValueError("Missing Ticker values in validation data.")

    # Identify numerical features without including metadata or targets.
    feature_columns = [
        column
        for column in train.columns
        if column not in METADATA_COLUMNS + TARGET_COLUMNS
    ]

    if len(feature_columns) != 40:
        raise ValueError(
            "Expected 40 numerical features, "
            f"but found {len(feature_columns)}."
        )

    missing_validation_features = [
        column
        for column in feature_columns
        if column not in validation.columns
    ]

    if missing_validation_features:
        raise ValueError(
            "Missing validation features: "
            f"{missing_validation_features}"
        )

    # Check that all numerical features contain valid numeric values.
    for column in feature_columns:
        train[column] = pd.to_numeric(
            train[column],
            errors="raise",
        )

        validation[column] = pd.to_numeric(
            validation[column],
            errors="raise",
        )

    if not np.isfinite(
        train[feature_columns].to_numpy()
    ).all():
        raise ValueError(
            "Non-finite values detected in training features."
        )

    if not np.isfinite(
        validation[feature_columns].to_numpy()
    ).all():
        raise ValueError(
            "Non-finite values detected in validation features."
        )

    print("\nDATASETS LOADED")
    print("=" * 70)
    print(f"Training observations:   {len(train):,}")
    print(f"Validation observations: {len(validation):,}")
    print(f"Numerical features:      {len(feature_columns)}")
    print(f"Training tickers:        {train['Ticker'].nunique()}")
    print(f"Validation tickers:      {validation['Ticker'].nunique()}")

    print("\nTraining date range:")
    print(f"{train['Date'].min()} -> {train['Date'].max()}")

    print("\nValidation date range:")
    print(
        f"{validation['Date'].min()} -> "
        f"{validation['Date'].max()}"
    )

    return train, validation, feature_columns


# ---------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------

def build_preprocessor(numerical_features):
    """Build numerical and categorical preprocessing."""

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                numerical_features,
            ),
            (
                "ticker",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
                ["Ticker"],
            ),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def evaluate_classification(y_true, predictions, probabilities):
    """Calculate classification metrics."""

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "predicted_positive_rate": np.mean(
            predictions
        ),
    }


def evaluate_regression(y_true, predictions):
    """Calculate regression metrics."""

    return {
        "mae": mean_absolute_error(
            y_true,
            predictions,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_true,
                predictions,
            )
        ),
        "r2": r2_score(
            y_true,
            predictions,
        ),
        "prediction_mean": np.mean(predictions),
        "prediction_std": np.std(predictions),
        "actual_mean": np.mean(y_true),
    }


# ---------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------

def run_experiment(train, validation, feature_columns):
    """Train and evaluate asset-aware classification and regression."""

    results = []
    predictions_output = []
    coefficients_output = []

    numerical_features = feature_columns

    for horizon in HORIZONS:

        print("\n" + "=" * 75)
        print(f"ASSET-AWARE EXPERIMENT - {horizon} DAYS")
        print("=" * 75)

        return_target = f"target_return_{horizon}d"
        direction_target = f"target_direction_{horizon}d"

        # -------------------------------------------------------------
        # Prepare input data
        # -------------------------------------------------------------

        X_train = train[
            numerical_features + ["Ticker"]
        ].copy()

        X_validation = validation[
            numerical_features + ["Ticker"]
        ].copy()

        # -------------------------------------------------------------
        # CLASSIFICATION
        # -------------------------------------------------------------

        print("\nCLASSIFICATION")
        print("-" * 70)

        y_train = train[direction_target].astype(int)
        y_validation = validation[direction_target].astype(int)

        if set(y_train.unique()) != {0, 1}:
            raise ValueError(
                f"Unexpected training labels for {direction_target}: "
                f"{sorted(y_train.unique())}"
            )

        if not set(y_validation.unique()).issubset({0, 1}):
            raise ValueError(
                f"Unexpected validation labels for {direction_target}."
            )

        classification_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numerical_features),
                ),
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

        classification_pipeline.fit(
            X_train,
            y_train,
        )

        class_predictions = (
            classification_pipeline.predict(X_validation)
        )

        class_probabilities = (
            classification_pipeline.predict_proba(
                X_validation
            )[:, 1]
        )

        classification_metrics = evaluate_classification(
            y_validation,
            class_predictions,
            class_probabilities,
        )

        print("Validation metrics:")

        for metric, value in classification_metrics.items():
            print(f"{metric:25s}: {value:.5f}")

        results.append(
            {
                "task": "classification",
                "horizon": horizon,
                "model": "LogisticRegression_AssetAware",
                **classification_metrics,
            }
        )

        for index in range(len(validation)):
            predictions_output.append(
                {
                    "Date": validation.iloc[index]["Date"],
                    "Ticker": validation.iloc[index]["Ticker"],
                    "task": "classification",
                    "horizon": horizon,
                    "actual": int(y_validation.iloc[index]),
                    "prediction": int(class_predictions[index]),
                    "predicted_probability_up": float(
                        class_probabilities[index]
                    ),
                }
            )

        # Save classification coefficients.
        fitted_preprocessor = (
            classification_pipeline.named_steps["preprocessor"]
        )

        feature_names = (
            fitted_preprocessor.get_feature_names_out()
        )

        model_coefficients = (
            classification_pipeline.named_steps["model"].coef_[0]
        )

        for feature_name, coefficient in zip(
            feature_names,
            model_coefficients,
        ):
            coefficients_output.append(
                {
                    "task": "classification",
                    "horizon": horizon,
                    "feature": feature_name,
                    "coefficient": coefficient,
                }
            )

        # -------------------------------------------------------------
        # REGRESSION
        # -------------------------------------------------------------

        print("\nREGRESSION")
        print("-" * 70)

        y_train = train[return_target].astype(float)
        y_validation = validation[return_target].astype(float)

        regression_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numerical_features),
                ),
                (
                    "model",
                    Ridge(alpha=1.0),
                ),
            ]
        )

        regression_pipeline.fit(
            X_train,
            y_train,
        )

        regression_predictions = (
            regression_pipeline.predict(X_validation)
        )

        regression_metrics = evaluate_regression(
            y_validation,
            regression_predictions,
        )

        print("Validation metrics:")

        for metric, value in regression_metrics.items():
            print(f"{metric:25s}: {value:.5f}")

        results.append(
            {
                "task": "regression",
                "horizon": horizon,
                "model": "Ridge_AssetAware",
                **regression_metrics,
            }
        )

        for index in range(len(validation)):
            predictions_output.append(
                {
                    "Date": validation.iloc[index]["Date"],
                    "Ticker": validation.iloc[index]["Ticker"],
                    "task": "regression",
                    "horizon": horizon,
                    "actual": float(y_validation.iloc[index]),
                    "prediction": float(
                        regression_predictions[index]
                    ),
                    "predicted_probability_up": np.nan,
                }
            )

        # Save regression coefficients.
        fitted_preprocessor = (
            regression_pipeline.named_steps["preprocessor"]
        )

        feature_names = (
            fitted_preprocessor.get_feature_names_out()
        )

        model_coefficients = (
            regression_pipeline.named_steps["model"].coef_
        )

        for feature_name, coefficient in zip(
            feature_names,
            model_coefficients,
        ):
            coefficients_output.append(
                {
                    "task": "regression",
                    "horizon": horizon,
                    "feature": feature_name,
                    "coefficient": coefficient,
                }
            )

    return (
        pd.DataFrame(results),
        pd.DataFrame(predictions_output),
        pd.DataFrame(coefficients_output),
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("\nSTOCK PREDICTION ENGINE")
    print("Asset-Aware Linear Models")
    print("=" * 75)

    train, validation, feature_columns = load_datasets()

    results, predictions, coefficients = run_experiment(
        train,
        validation,
        feature_columns,
    )

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    coefficients.to_csv(
        COEFFICIENTS_PATH,
        index=False,
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT COMPLETED")
    print("=" * 75)

    print("\nRESULTS")
    print(results.to_string(index=False))

    print("\nFiles saved:")
    print(f"- {RESULTS_PATH}")
    print(f"- {PREDICTIONS_PATH}")
    print(f"- {COEFFICIENTS_PATH}")

    print("\nThe test dataset was not loaded or used.")


if __name__ == "__main__":
    main()
