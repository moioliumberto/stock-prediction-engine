
"""Train and evaluate a Random Forest regression model."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit

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
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    """Load and validate the feature dataset."""

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
        raise ValueError("The dataset is empty.")

    if data[required_columns].isna().any().any():
        raise ValueError("Missing values detected.")

    return data.sort_index()


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

    print(f"\nPurge gap: {PURGE_GAP}")

    print("\nTraining target statistics:")
    print(train[TARGET_COLUMN].describe())

    print("\nValidation target statistics:")
    print(validation[TARGET_COLUMN].describe())

    return train, validation


def create_model() -> RandomForestRegressor:
    """Create a controlled Random Forest regressor."""

    return RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=20,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def calculate_directional_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate the percentage of correctly predicted directions."""

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


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """Calculate regression metrics."""

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted),
        "Directional Accuracy": calculate_directional_accuracy(
            actual,
            predicted,
        ),
    }


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Evaluate Random Forest and baselines using time-series CV."""

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        gap=PURGE_GAP,
    )

    results = []

    print("\nRegression Cross-Validation")
    print("-" * 50)
    print(f"Folds: {N_SPLITS}")
    print(f"Purge gap: {PURGE_GAP}")

    for fold, (train_idx, validation_idx) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]

        X_fold_validation = X.iloc[validation_idx]
        y_fold_validation = y.iloc[validation_idx]

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

        # Mean baseline
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

        # Zero return baseline
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

        # Random Forest
        model = create_model()

        model.fit(
            X_fold_train,
            y_fold_train,
        )

        predictions = model.predict(
            X_fold_validation
        )

        rf_metrics = calculate_metrics(
            actual,
            predictions,
        )

        results.append(
            {
                "Fold": fold,
                "Model": "Random Forest",
                **rf_metrics,
            }
        )

        print(
            f"Random Forest: "
            f"MAE={rf_metrics['MAE']:.5f} "
            f"RMSE={rf_metrics['RMSE']:.5f} "
            f"R2={rf_metrics['R2']:.4f}"
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


def summarize_cv(results: pd.DataFrame) -> None:
    """Summarize cross-validation results."""

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
    print("RANDOM FOREST REGRESSION CROSS-VALIDATION SUMMARY")
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


def evaluate_validation(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> None:
    """Train on the full training set and evaluate validation."""

    print("\n" + "=" * 65)
    print("FINAL RANDOM FOREST VALIDATION")
    print("=" * 65)

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)
    actual = y_validation.to_numpy()

    metrics = calculate_metrics(
        actual,
        predictions,
    )

    print("\nValidation metrics")
    print("-" * 40)

    for name, value in metrics.items():
        if np.isnan(value):
            print(f"{name}: N/A")
        elif name == "Directional Accuracy":
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value:.6f}")

    print("\nReturn statistics")
    print("-" * 40)
    print(f"Actual mean return: {actual.mean():.4%}")
    print(f"Predicted mean return: {predictions.mean():.4%}")
    print(f"Actual standard deviation: {actual.std():.4%}")
    print(
        f"Predicted standard deviation: "
        f"{predictions.std():.4%}"
    )

    correlation = pd.Series(actual).corr(
        pd.Series(predictions)
    )

    print(f"Prediction correlation: {correlation:.4f}")

    print("\nFeature Importance")
    print("-" * 50)

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(ascending=False)

    print(importance.to_string())

    print("\nSample predictions")
    print("-" * 65)

    sample = pd.DataFrame(
        {
            "Actual Return": actual,
            "Predicted Return": predictions,
        },
        index=X_validation.index,
    )

    print(sample.head(10).to_string())


def main() -> None:
    """Run the Random Forest regression experiment."""

    data = load_dataset()

    print(f"Observations: {len(data)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Target: {TARGET_COLUMN}")

    train, validation = create_train_validation_split(data)

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN].astype(float)

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN].astype(float)

    cv_results = run_cross_validation(
        X_train,
        y_train,
    )

    summarize_cv(cv_results)

    model = create_model()

    evaluate_validation(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    print("\nRandom Forest regression completed.")
    print("The Test Set has not been used.")


if __name__ == "__main__":
    main()
