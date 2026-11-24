
"""Evaluate HistGradientBoostingRegressor for 5-day stock return prediction."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit

from stock_prediction_engine.features import FEATURE_COLUMNS


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "AAPL_features.csv"

TARGET_COLUMN = "target_return_5d"

N_SPLITS = 5
PURGE_GAP = 5

RANDOM_STATE = 42


# ---------------------------------------------------------
# Data loading and chronological split
# ---------------------------------------------------------

def load_data():
    """Load the feature dataset and create chronological splits."""

    data = pd.read_csv(
        FEATURES_FILE,
        index_col="Date",
        parse_dates=True,
    )

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = data.dropna(subset=required_columns)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)

    X_train = X.iloc[:train_end - PURGE_GAP]
    y_train = y.iloc[:train_end - PURGE_GAP]

    X_validation = X.iloc[train_end:validation_end]
    y_validation = y.iloc[train_end:validation_end]

    print("\nDATASET")
    print("-" * 60)
    print(f"Total observations: {len(data)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Target: {TARGET_COLUMN}")

    print("\nTRAINING SET")
    print(f"Observations: {len(X_train)}")
    print(f"Period: {X_train.index.min().date()} to "
          f"{X_train.index.max().date()}")

    print("\nVALIDATION SET")
    print(f"Observations: {len(X_validation)}")
    print(f"Period: {X_validation.index.min().date()} to "
          f"{X_validation.index.max().date()}")

    print(f"\nPurge gap: {PURGE_GAP} observations")

    return X_train, y_train, X_validation, y_validation


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(y_true, predictions):
    """Calculate regression and directional prediction metrics."""

    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
        "Directional Accuracy": np.mean(
            np.sign(y_true) == np.sign(predictions)
        ),
    }


def print_metrics(name, metrics):
    """Display model metrics."""

    print(f"\n{name}")
    print("-" * 60)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


# ---------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------

def run_cross_validation(X_train, y_train):
    """Evaluate models using chronological time-series cross-validation."""

    time_series_cv = TimeSeriesSplit(
        n_splits=N_SPLITS,
        gap=PURGE_GAP,
    )

    models = {
        "Mean Baseline": DummyRegressor(strategy="mean"),

        "Zero Baseline": DummyRegressor(strategy="constant", constant=0),

        "Gradient Boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    print("\nCROSS-VALIDATION")
    print("=" * 60)
    print(f"Folds: {N_SPLITS}")
    print(f"Purge gap: {PURGE_GAP}")

    for model_name, model in models.items():

        fold_metrics = []

        for fold, (train_index, validation_index) in enumerate(
            time_series_cv.split(X_train),
            start=1,
        ):

            X_fold_train = X_train.iloc[train_index]
            y_fold_train = y_train.iloc[train_index]

            X_fold_validation = X_train.iloc[validation_index]
            y_fold_validation = y_train.iloc[validation_index]

            model.fit(X_fold_train, y_fold_train)

            predictions = model.predict(X_fold_validation)

            metrics = calculate_metrics(
                y_fold_validation,
                predictions,
            )

            fold_metrics.append(metrics)

            if model_name == "Gradient Boosting":
                print(
                    f"Fold {fold}: "
                    f"MAE={metrics['MAE']:.4f}, "
                    f"RMSE={metrics['RMSE']:.4f}, "
                    f"R2={metrics['R2']:.4f}, "
                    f"DA={metrics['Directional Accuracy']:.4f}"
                )

        results.append({
            "Model": model_name,
            "MAE": np.mean([m["MAE"] for m in fold_metrics]),
            "MAE Std": np.std([m["MAE"] for m in fold_metrics]),
            "RMSE": np.mean([m["RMSE"] for m in fold_metrics]),
            "RMSE Std": np.std([m["RMSE"] for m in fold_metrics]),
            "R2": np.mean([m["R2"] for m in fold_metrics]),
            "R2 Std": np.std([m["R2"] for m in fold_metrics]),
            "Directional Accuracy": np.mean(
                [m["Directional Accuracy"] for m in fold_metrics]
            ),
            "DA Std": np.std(
                [m["Directional Accuracy"] for m in fold_metrics]
            ),
        })

    results_df = pd.DataFrame(results).set_index("Model")

    print("\nCROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(results_df.round(4).to_string())

    return results_df


# ---------------------------------------------------------
# Validation evaluation
# ---------------------------------------------------------

def evaluate_validation(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """Fit the model on training data and evaluate on validation data."""

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    metrics = calculate_metrics(
        y_validation,
        predictions,
    )

    print("\nVALIDATION RESULTS")
    print("=" * 60)

    print_metrics("Gradient Boosting Regression", metrics)

    print("\nPREDICTION ANALYSIS")
    print("-" * 60)

    print(f"Actual mean: {y_validation.mean():.4%}")
    print(f"Predicted mean: {predictions.mean():.4%}")

    print(f"Actual standard deviation: {y_validation.std():.4%}")
    print(f"Predicted standard deviation: {predictions.std():.4%}")

    correlation = np.corrcoef(
        y_validation,
        predictions,
    )[0, 1]

    print(f"Prediction correlation: {correlation:.4f}")

    comparison = pd.DataFrame({
        "Actual Return": y_validation,
        "Predicted Return": predictions,
    })

    print("\nFIRST 10 PREDICTIONS")
    print("-" * 60)
    print(comparison.head(10).to_string())

    return model, metrics


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    X_train, y_train, X_validation, y_validation = load_data()

    run_cross_validation(X_train, y_train)

    evaluate_validation(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )


if __name__ == "__main__":
    main()
