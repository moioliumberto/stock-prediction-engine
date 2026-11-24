
"""Compare Ridge Regression regularization using time-series CV."""

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
from sklearn.model_selection import TimeSeriesSplit
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

N_SPLITS = 5
ALPHA_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]


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

    return data.sort_index()


def create_training_set(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Select the chronological training portion only."""

    train_end = int(len(data) * 0.70)

    # Remove the final five observations before the next split.
    train = data.iloc[:train_end - PURGE_GAP].copy()

    if train.empty:
        raise ValueError("Training set is empty.")

    print("\nTraining dataset")
    print("-" * 50)
    print(f"Observations: {len(train)}")
    print(
        f"Period: {train.index.min().date()} "
        f"to {train.index.max().date()}"
    )
    print(f"Target: {TARGET_COLUMN}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    return train


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """Calculate regression and directional metrics."""

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    actual_direction = np.sign(actual)
    predicted_direction = np.sign(predicted)

    nonzero_predictions = predicted_direction != 0

    if nonzero_predictions.any():
        directional_accuracy = np.mean(
            actual_direction[nonzero_predictions]
            == predicted_direction[nonzero_predictions]
        )
    else:
        directional_accuracy = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": directional_accuracy,
    }


def create_ridge_model(alpha: float) -> Pipeline:
    """Create a Ridge regression pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                Ridge(alpha=alpha),
            ),
        ]
    )


def run_cross_validation(
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate Ridge and baseline models across time-series folds."""

    X = train[FEATURE_COLUMNS]
    y = train[TARGET_COLUMN].astype(float)

    time_series_split = TimeSeriesSplit(
        n_splits=N_SPLITS,
        gap=PURGE_GAP,
    )

    results = []

    print("\nCross-validation configuration")
    print("-" * 50)
    print(f"Folds: {N_SPLITS}")
    print(f"Purge gap: {PURGE_GAP}")
    print(f"Alpha values: {ALPHA_VALUES}")
    print(
        f"Ridge configurations: {len(ALPHA_VALUES)}"
    )

    for fold, (train_indices, validation_indices) in enumerate(
        time_series_split.split(X),
        start=1,
    ):
        X_fold_train = X.iloc[train_indices]
        y_fold_train = y.iloc[train_indices]

        X_fold_validation = X.iloc[validation_indices]
        y_fold_validation = y.iloc[validation_indices]

        actual = y_fold_validation.to_numpy()

        print("\n" + "=" * 65)
        print(f"FOLD {fold}")
        print("=" * 65)

        print(
            f"Training observations: {len(X_fold_train)}"
        )
        print(
            f"Validation observations: "
            f"{len(X_fold_validation)}"
        )
        print(
            f"Training period: "
            f"{X_fold_train.index.min().date()} "
            f"to {X_fold_train.index.max().date()}"
        )
        print(
            f"Validation period: "
            f"{X_fold_validation.index.min().date()} "
            f"to {X_fold_validation.index.max().date()}"
        )

        # Baseline 1: predict the training fold mean.
        mean_model = DummyRegressor(strategy="mean")

        mean_model.fit(
            X_fold_train,
            y_fold_train,
        )

        mean_predictions = mean_model.predict(
            X_fold_validation
        )

        mean_metrics = calculate_metrics(
            actual,
            mean_predictions,
        )

        results.append(
            {
                "Fold": fold,
                "Model": "Mean Baseline",
                **mean_metrics,
            }
        )

        # Baseline 2: always predict zero return.
        zero_predictions = np.zeros(
            len(X_fold_validation)
        )

        zero_metrics = calculate_metrics(
            actual,
            zero_predictions,
        )

        results.append(
            {
                "Fold": fold,
                "Model": "Zero Baseline",
                **zero_metrics,
            }
        )

        # Ridge models with different alpha values.
        for alpha in ALPHA_VALUES:
            model = create_ridge_model(alpha)

            model.fit(
                X_fold_train,
                y_fold_train,
            )

            predictions = model.predict(
                X_fold_validation
            )

            metrics = calculate_metrics(
                actual,
                predictions,
            )

            results.append(
                {
                    "Fold": fold,
                    "Model": f"Ridge alpha={alpha}",
                    **metrics,
                }
            )

            print(
                f"Ridge alpha={alpha:<6} "
                f"MAE={metrics['MAE']:.5f} "
                f"RMSE={metrics['RMSE']:.5f} "
                f"R2={metrics['R2']:.4f}"
            )

        print(
            f"Mean Baseline: "
            f"MAE={mean_metrics['MAE']:.5f} "
            f"RMSE={mean_metrics['RMSE']:.5f} "
            f"R2={mean_metrics['R2']:.4f}"
        )

        print(
            f"Zero Baseline: "
            f"MAE={zero_metrics['MAE']:.5f} "
            f"RMSE={zero_metrics['RMSE']:.5f} "
            f"R2={zero_metrics['R2']:.4f}"
        )

    return pd.DataFrame(results)


def summarize_results(
    results: pd.DataFrame,
) -> None:
    """Print mean and standard deviation across folds."""

    summary = (
        results
        .groupby("Model")
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            Directional_mean=(
                "Directional Accuracy",
                "mean",
            ),
            Directional_std=(
                "Directional Accuracy",
                "std",
            ),
        )
        .reset_index()
    )

    print("\n\n" + "=" * 100)
    print("REGRESSION CROSS-VALIDATION SUMMARY")
    print("=" * 100)

    print(
        summary.to_string(
            index=False,
            formatters={
                "MAE_mean": "{:.5f}".format,
                "MAE_std": "{:.5f}".format,
                "RMSE_mean": "{:.5f}".format,
                "RMSE_std": "{:.5f}".format,
                "R2_mean": "{:.4f}".format,
                "R2_std": "{:.4f}".format,
                "Directional_mean": (
                    lambda value: (
                        "N/A"
                        if pd.isna(value)
                        else f"{value:.4f}"
                    )
                ),
                "Directional_std": (
                    lambda value: (
                        "N/A"
                        if pd.isna(value)
                        else f"{value:.4f}"
                    )
                ),
            },
        )
    )

    ridge_summary = summary[
        summary["Model"].str.startswith("Ridge")
    ].copy()

    best_ridge = ridge_summary.sort_values(
        by="MAE_mean",
        ascending=True,
    ).iloc[0]

    print("\nBest Ridge configuration by mean MAE")
    print("-" * 50)
    print(f"Model: {best_ridge['Model']}")
    print(f"Mean MAE: {best_ridge['MAE_mean']:.5f}")
    print(f"Mean RMSE: {best_ridge['RMSE_mean']:.5f}")
    print(f"Mean R²: {best_ridge['R2_mean']:.4f}")

    print(
        "\nNote: the best configuration is selected using "
        "Training Set cross-validation only."
    )


def main() -> None:
    """Run the Ridge regression cross-validation experiment."""

    data = load_dataset()

    train = create_training_set(data)

    results = run_cross_validation(train)

    summarize_results(results)

    print("\nRegression cross-validation completed.")
    print("The Validation Set and Test Set were not used.")


if __name__ == "__main__":
    main()
