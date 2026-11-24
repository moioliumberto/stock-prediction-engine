
"""Train and evaluate stock return regression models."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_prediction_engine.features import FEATURE_COLUMNS


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "AAPL_features.csv"
)

TARGET_COLUMN = "target_return_5d"
HORIZON = 5
PURGE_GAP = HORIZON
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    """Load the feature dataset and validate required columns."""

    print("Loading feature dataset...")

    if not FEATURES_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURES_DATA_PATH}"
        )

    data = pd.read_csv(
        FEATURES_DATA_PATH,
        index_col="Date",
        parse_dates=True,
    )

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("The feature dataset is empty.")

    if data[required_columns].isna().any().any():
        raise ValueError(
            "Missing values detected in features or target."
        )

    data = data.sort_index()

    print(f"Observations: {len(data)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Target: {TARGET_COLUMN}")

    return data


def create_train_validation_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create chronological training and validation sets."""

    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)

    train = data.iloc[:train_end - PURGE_GAP].copy()

    validation = data.iloc[
        train_end:validation_end - PURGE_GAP
    ].copy()

    if train.empty or validation.empty:
        raise ValueError(
            "Training or validation set is empty."
        )

    print("\nChronological split")
    print("-" * 50)

    print(f"Training observations: {len(train)}")
    print(
        f"Training period: {train.index.min().date()} "
        f"to {train.index.max().date()}"
    )

    print(f"\nValidation observations: {len(validation)}")
    print(
        f"Validation period: {validation.index.min().date()} "
        f"to {validation.index.max().date()}"
    )

    print(f"\nPurge gap: {PURGE_GAP} observations")

    print("\nTraining target statistics:")
    print(train[TARGET_COLUMN].describe())

    print("\nValidation target statistics:")
    print(validation[TARGET_COLUMN].describe())

    return train, validation


def create_models() -> dict:
    """Create regression models."""

    models = {
        "Mean Baseline": DummyRegressor(
            strategy="mean"
        ),

        "Ridge Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    Ridge(
                        alpha=1.0,
                    ),
                ),
            ]
        ),
    }

    return models


def calculate_directional_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate directional accuracy, excluding zero predictions."""

    actual_direction = np.sign(actual)
    predicted_direction = np.sign(predicted)

    nonzero_predictions = predicted_direction != 0

    if not nonzero_predictions.any():
        return float("nan")

    correct_predictions = (
        actual_direction[nonzero_predictions]
        == predicted_direction[nonzero_predictions]
    )

    return float(correct_predictions.mean())


def evaluate_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[dict, np.ndarray]:
    """Train a regression model and evaluate it."""

    print("\n" + "=" * 65)
    print(f"MODEL: {name}")
    print("=" * 65)

    print("Training model...")

    model.fit(X_train, y_train)

    print("Generating validation predictions...")

    predictions = model.predict(X_validation)

    mae = mean_absolute_error(
        y_validation,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            predictions,
        )
    )

    r2 = r2_score(
        y_validation,
        predictions,
    )

    directional_accuracy = calculate_directional_accuracy(
        y_validation.to_numpy(),
        predictions,
    )

    print("\nValidation metrics")
    print("-" * 40)

    print(f"MAE:                  {mae:.6f}")
    print(f"RMSE:                 {rmse:.6f}")
    print(f"R²:                   {r2:.4f}")

    if np.isnan(directional_accuracy):
        print("Directional Accuracy: N/A")
    else:
        print(
            f"Directional Accuracy: "
            f"{directional_accuracy:.4f}"
        )

    print("\nReturn statistics")
    print("-" * 40)
    print(
        f"Actual mean return: "
        f"{y_validation.mean():.4%}"
    )
    print(
        f"Predicted mean return: "
        f"{predictions.mean():.4%}"
    )
    print(
        f"Actual standard deviation: "
        f"{y_validation.std():.4%}"
    )
    print(
        f"Predicted standard deviation: "
        f"{np.std(predictions):.4%}"
    )

    actual_series = pd.Series(
        y_validation.to_numpy()
    )
    predicted_series = pd.Series(predictions)

    correlation = actual_series.corr(predicted_series)

    print(f"Prediction correlation: {correlation:.4f}")

    results = {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": directional_accuracy,
        "Prediction Correlation": correlation,
        "Mean Predicted Return": predictions.mean(),
    }

    return results, predictions


def evaluate_zero_return_baseline(
    y_validation: pd.Series,
) -> tuple[dict, np.ndarray]:
    """Evaluate a baseline that always predicts zero return."""

    print("\n" + "=" * 65)
    print("MODEL: Zero Return Baseline")
    print("=" * 65)

    predictions = np.zeros(len(y_validation))

    mae = mean_absolute_error(
        y_validation,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            predictions,
        )
    )

    r2 = r2_score(
        y_validation,
        predictions,
    )

    print("\nValidation metrics")
    print("-" * 40)
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²:   {r2:.4f}")

    print("\nThis baseline does not predict a direction.")

    results = {
        "Model": "Zero Return Baseline",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": float("nan"),
        "Prediction Correlation": float("nan"),
        "Mean Predicted Return": 0.0,
    }

    return results, predictions


def main() -> None:
    """Run the regression experiment."""

    data = load_dataset()

    train, validation = create_train_validation_split(data)

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN].astype(float)

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN].astype(float)

    models = create_models()

    results = []
    predictions_by_model = {}

    for name, model in models.items():
        result, predictions = evaluate_model(
            name=name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
        )

        results.append(result)
        predictions_by_model[name] = predictions

    zero_result, zero_predictions = (
        evaluate_zero_return_baseline(y_validation)
    )

    results.append(zero_result)
    predictions_by_model["Zero Return Baseline"] = (
        zero_predictions
    )

    results_df = pd.DataFrame(results)

    print("\n\n" + "=" * 90)
    print("FINAL REGRESSION COMPARISON")
    print("=" * 90)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "MAE": "{:.6f}".format,
                "RMSE": "{:.6f}".format,
                "R2": "{:.4f}".format,
                "Directional Accuracy": (
                    lambda value: (
                        "N/A"
                        if pd.isna(value)
                        else f"{value:.4f}"
                    )
                ),
                "Prediction Correlation": (
                    lambda value: (
                        "N/A"
                        if pd.isna(value)
                        else f"{value:.4f}"
                    )
                ),
                "Mean Predicted Return": (
                    "{:.4%}".format
                ),
            },
        )
    )

    print("\nSample predictions: Ridge Regression")
    print("-" * 65)

    sample = pd.DataFrame(
        {
            "Actual Return": y_validation,
            "Predicted Return": predictions_by_model[
                "Ridge Regression"
            ],
        },
        index=validation.index,
    )

    print(sample.head(10).to_string())

    print("\nRegression experiment completed.")
    print("The Test Set has not been used.")


if __name__ == "__main__":
    main()
