
"""Compare stock direction prediction across 5, 10 and 20 days."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_prediction_engine.features import FEATURE_COLUMNS


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TICKER = "AAPL"

HORIZONS = (5, 10, 20)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_feature_dates(index):
    """Normalize feature dates to timezone-naive New York time."""

    dates = pd.DatetimeIndex(pd.to_datetime(index))

    if dates.tz is not None:
        dates = (
            dates.tz_convert("America/New_York")
            .tz_localize(None)
        )

    return dates


def normalize_source_dates(index):
    """Normalize source dates with mixed UTC offsets."""

    dates = pd.to_datetime(index, utc=True)

    return (
        dates.tz_convert("America/New_York")
        .tz_localize(None)
    )


# ============================================================
# TARGET ALIGNMENT AUDIT
# ============================================================

def check_target_alignment(ticker=TICKER):
    """Verify all future-return targets against source prices."""

    ticker = ticker.upper()

    source_path = (
        PROCESSED_DATA_DIR / f"{ticker}_processed.csv"
    )

    feature_path = (
        PROCESSED_DATA_DIR / f"{ticker}_features.csv"
    )

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    source = pd.read_csv(
        source_path,
        index_col="Date",
    )

    source.index = normalize_source_dates(source.index)
    source.index.name = "Date"
    source = source.sort_index()

    features = pd.read_csv(
        feature_path,
        index_col="Date",
    )

    features.index = normalize_feature_dates(
        features.index
    )
    features.index.name = "Date"
    features = features.sort_index()

    print("\n" + "=" * 65)
    print("MULTI-HORIZON TEMPORAL ALIGNMENT AUDIT")
    print("=" * 65)

    for horizon in HORIZONS:

        return_column = f"target_return_{horizon}d"
        target_column = f"target_{horizon}d"

        expected_return = (
            source["Adj Close"].shift(-horizon)
            / source["Adj Close"]
            - 1
        )

        audit = features[
            [return_column, target_column]
        ].join(
            expected_return.rename("expected_return"),
            how="left",
        ).dropna()

        error = (
            audit[return_column]
            - audit["expected_return"]
        ).abs()

        expected_labels = (
            audit["expected_return"] > 0
        ).astype(int)

        labels_match = (
            audit[target_column].astype(int)
            == expected_labels
        )

        print(f"\nHorizon: {horizon} trading sessions")
        print(f"Rows checked: {len(audit)}")
        print(f"Maximum return error: {error.max():.12f}")
        print(f"Labels matching: {labels_match.mean():.2%}")

        if error.max() > 1e-9:
            raise ValueError(
                f"{horizon}-day return alignment failed."
            )

        if not labels_match.all():
            raise ValueError(
                f"{horizon}-day classification alignment failed."
            )

    print("\nAll target alignments verified successfully.")


# ============================================================
# LOAD FEATURE DATASET
# ============================================================

def load_feature_dataset(ticker=TICKER):
    """Load and validate the feature dataset."""

    file_path = (
        PROCESSED_DATA_DIR / f"{ticker.upper()}_features.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    data = pd.read_csv(
        file_path,
        index_col="Date",
    )

    data.index = normalize_feature_dates(data.index)
    data.index.name = "Date"
    data = data.sort_index()

    target_columns = []

    for horizon in HORIZONS:
        target_columns.extend([
            f"target_return_{horizon}d",
            f"target_{horizon}d",
        ])

    required_columns = FEATURE_COLUMNS + target_columns

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data[required_columns].isna().any().any():
        raise ValueError(
            "Missing values detected in the feature dataset."
        )

    if data.index.has_duplicates:
        raise ValueError("Duplicate dates detected.")

    return data


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def split_dataset(data, purge_gap):
    """Create chronological train, validation and test datasets."""

    if not np.isclose(
        TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO,
        1.0,
    ):
        raise ValueError(
            "Split ratios must sum to 1."
        )

    n = len(data)

    train_end = int(n * TRAIN_RATIO)

    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train = data.iloc[
        :train_end - purge_gap
    ].copy()

    validation = data.iloc[
        train_end:validation_end - purge_gap
    ].copy()

    test = data.iloc[
        validation_end:
    ].copy()

    return train, validation, test


def print_dataset_summary(
    train,
    validation,
    test,
    target_column,
    horizon,
    purge_gap,
):
    """Print split dates, sizes and target distributions."""

    datasets = {
        "TRAIN": train,
        "VALIDATION": validation,
        "TEST": test,
    }

    print("\n" + "=" * 65)
    print(f"{horizon}-DAY CHRONOLOGICAL SPLIT")
    print("=" * 65)

    for name, dataset in datasets.items():

        print(f"\n{name}")
        print(f"Observations: {len(dataset)}")
        print(f"Start: {dataset.index.min()}")
        print(f"End:   {dataset.index.max()}")

        print("\nTarget distribution:")

        print(
            dataset[target_column]
            .value_counts(normalize=True)
            .sort_index()
            .rename(index={0: "Down", 1: "Up"})
            .round(4)
        )

    print(f"\nPurge gap: {purge_gap} observations")


# ============================================================
# MODEL FACTORY
# ============================================================

def create_logistic_regression():
    """Create a fresh Logistic Regression pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def create_dummy_classifier():
    """Create a majority-class baseline."""

    return DummyClassifier(
        strategy="most_frequent"
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X, y):
    """Calculate classification metrics."""

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, predictions),

        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
    }

    return metrics, predictions, probabilities


def print_evaluation(
    name,
    metrics,
    y_true,
    predictions,
):
    """Print metrics and confusion matrix."""

    print(f"\n{name}")

    for metric_name, value in metrics.items():
        print(f"{metric_name:12s}: {value:.4f}")

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        )
    )


# ============================================================
# TIME SERIES CROSS-VALIDATION
# ============================================================

def run_time_series_cross_validation(
    X_train,
    y_train,
    horizon,
):
    """Run expanding-window CV with a horizon-specific purge gap."""

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        gap=horizon,
    )

    results = []

    print("\n" + "=" * 65)
    print(f"{horizon}-DAY TIME SERIES CROSS-VALIDATION")
    print("=" * 65)

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(X_train),
        start=1,
    ):

        X_fold_train = X_train.iloc[train_indices]
        y_fold_train = y_train.iloc[train_indices]

        X_fold_validation = X_train.iloc[validation_indices]
        y_fold_validation = y_train.iloc[validation_indices]

        print(f"\nFold {fold}")

        print(
            f"Training: {X_fold_train.index.min()} "
            f"to {X_fold_train.index.max()}"
        )

        print(
            f"Validation: {X_fold_validation.index.min()} "
            f"to {X_fold_validation.index.max()}"
        )

        models = {
            "Dummy Classifier": create_dummy_classifier(),
            "Logistic Regression": create_logistic_regression(),
        }

        for model_name, model in models.items():

            if model_name == "Dummy Classifier":
                X_fit = np.zeros((len(y_fold_train), 1))
                X_evaluate = np.zeros(
                    (len(y_fold_validation), 1)
                )
            else:
                X_fit = X_fold_train
                X_evaluate = X_fold_validation

            model.fit(
                X_fit,
                y_fold_train,
            )

            metrics, _, _ = evaluate_model(
                model,
                X_evaluate,
                y_fold_validation,
            )

            results.append({
                "horizon": horizon,
                "fold": fold,
                "model": model_name,
                **metrics,
            })

            print(
                f"{model_name}: "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )

    results_df = pd.DataFrame(results)

    summary = (
        results_df
        .groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
        )
    )

    print("\nCross-validation summary:")
    print(summary.round(4))

    return results_df, summary


# ============================================================
# FINAL VALIDATION
# ============================================================

def evaluate_on_validation(
    train,
    validation,
    target_column,
    horizon,
):
    """Train final models on Training and evaluate on Validation."""

    X_train = train[FEATURE_COLUMNS]
    y_train = train[target_column].astype(int)

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[target_column].astype(int)

    models = {
        "Dummy Classifier": create_dummy_classifier(),
        "Logistic Regression": create_logistic_regression(),
    }

    results = {}

    print("\n" + "=" * 65)
    print(f"{horizon}-DAY FINAL VALIDATION")
    print("=" * 65)

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        if model_name == "Dummy Classifier":
            X_fit = np.zeros((len(y_train), 1))
            X_evaluate = np.zeros(
                (len(y_validation), 1)
            )
        else:
            X_fit = X_train
            X_evaluate = X_validation

        model.fit(
            X_fit,
            y_train,
        )

        train_metrics, train_predictions, _ = evaluate_model(
            model,
            X_fit,
            y_train,
        )

        validation_metrics, validation_predictions, probabilities = (
            evaluate_model(
                model,
                X_evaluate,
                y_validation,
            )
        )

        print_evaluation(
            f"{model_name} - TRAINING",
            train_metrics,
            y_train,
            train_predictions,
        )

        print_evaluation(
            f"{model_name} - VALIDATION",
            validation_metrics,
            y_validation,
            validation_predictions,
        )

        results[model_name] = validation_metrics

        if model_name == "Logistic Regression":
            probability_summary = (
                pd.Series(probabilities)
                .describe(
                    percentiles=[
                        0.05,
                        0.25,
                        0.50,
                        0.75,
                        0.95,
                    ]
                )
                .round(4)
            )

            print("\nPredicted probability distribution:")
            print(probability_summary)

            print("\nPredicted class distribution:")
            print(
                pd.Series(validation_predictions)
                .value_counts(normalize=True)
                .sort_index()
                .rename(index={0: "Down", 1: "Up"})
                .round(4)
            )

    comparison = pd.DataFrame(results).T

    print("\nValidation comparison:")
    print(comparison.round(4))

    return comparison


# ============================================================
# MULTI-HORIZON EXPERIMENT
# ============================================================

def run_horizon_experiment(data, horizon):
    """Run CV and validation for one prediction horizon."""

    target_column = f"target_{horizon}d"

    purge_gap = horizon

    print("\n\n" + "#" * 70)
    print(f"# EXPERIMENT: {horizon}-DAY PREDICTION")
    print("#" * 70)

    train, validation, test = split_dataset(
        data,
        purge_gap=purge_gap,
    )

    print_dataset_summary(
        train,
        validation,
        test,
        target_column,
        horizon,
        purge_gap,
    )

    X_train = train[FEATURE_COLUMNS]
    y_train = train[target_column].astype(int)

    cv_results, cv_summary = (
        run_time_series_cross_validation(
            X_train,
            y_train,
            horizon,
        )
    )

    validation_comparison = evaluate_on_validation(
        train,
        validation,
        target_column,
        horizon,
    )

    return {
        "horizon": horizon,
        "cv_results": cv_results,
        "cv_summary": cv_summary,
        "validation_comparison": validation_comparison,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_training_pipeline():
    """Compare 5-, 10- and 20-session prediction horizons."""

    print("\nSTOCK PREDICTION ENGINE")
    print("Multi-Horizon Prediction Experiment")
    print("=" * 65)

    # 1. Verify all target calculations.
    check_target_alignment(TICKER)

    # 2. Load the common feature dataset.
    data = load_feature_dataset(TICKER)

    print(f"\nDataset observations: {len(data)}")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print(f"Prediction horizons: {HORIZONS}")

    # 3. Run each horizon independently.
    experiment_results = []

    for horizon in HORIZONS:

        result = run_horizon_experiment(
            data,
            horizon,
        )

        experiment_results.append(result)

    # 4. Build cross-validation comparison.
    cv_rows = []

    for result in experiment_results:

        horizon = result["horizon"]
        summary = result["cv_summary"]

        for model_name, row in summary.iterrows():

            cv_rows.append({
                "horizon": horizon,
                "model": model_name,
                **row.to_dict(),
            })

    cv_comparison = pd.DataFrame(cv_rows)

    logistic_cv = (
        cv_comparison[
            cv_comparison["model"] == "Logistic Regression"
        ]
        .set_index("horizon")
        .sort_index()
    )

    print("\n" + "=" * 70)
    print("FINAL CROSS-VALIDATION COMPARISON")
    print("=" * 70)

    print(
        logistic_cv[
            [
                "accuracy_mean",
                "accuracy_std",
                "f1_mean",
                "roc_auc_mean",
                "roc_auc_std",
            ]
        ].round(4)
    )

    # 5. Build Validation comparison.
    validation_rows = []

    for result in experiment_results:

        horizon = result["horizon"]
        comparison = result["validation_comparison"]

        for model_name, row in comparison.iterrows():

            validation_rows.append({
                "horizon": horizon,
                "model": model_name,
                **row.to_dict(),
            })

    validation_comparison = pd.DataFrame(
        validation_rows
    )

    logistic_validation = (
        validation_comparison[
            validation_comparison["model"] == "Logistic Regression"
        ]
        .set_index("horizon")
        .sort_index()
    )

    print("\n" + "=" * 70)
    print("FINAL VALIDATION COMPARISON")
    print("=" * 70)

    print(
        logistic_validation[
            [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        ].round(4)
    )

    print("\n" + "=" * 70)
    print("MULTI-HORIZON EXPERIMENT COMPLETED")
    print("=" * 70)

    print("All three horizons were evaluated.")
    print("The Test Set has not been used.")
    print("Validation was not used for cross-validation or tuning.")

    return {
        "experiments": experiment_results,
        "cv_comparison": cv_comparison,
        "validation_comparison": validation_comparison,
    }


if __name__ == "__main__":
    run_training_pipeline()
