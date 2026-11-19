
"""Analyze feature correlation and perform time-series feature ablation."""

from itertools import combinations
from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_prediction_engine.features import FEATURE_COLUMNS


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "AAPL_features.csv"

HORIZON = 5
TARGET_COLUMN = f"target_{HORIZON}d"

TRAIN_RATIO = 0.70
PURGE_GAP = HORIZON

N_SPLITS = 5
RANDOM_STATE = 42

CORRELATION_THRESHOLD = 0.80


# ---------------------------------------------------------
# FEATURE GROUPS
# ---------------------------------------------------------

FEATURE_GROUPS = {
    "Returns and momentum": [
        "return_1d",
        "return_5d",
        "return_20d",
    ],
    "Trend": [
        "price_to_sma_5",
        "price_to_sma_20",
        "price_to_sma_50",
    ],
    "Volatility": [
        "volatility_10d",
        "volatility_20d",
    ],
    "RSI and MACD": [
        "rsi_14",
        "macd_pct",
        "macd_signal_pct",
        "macd_hist_pct",
    ],
    "Volume": [
        "volume_change",
    ],
}


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_training_data():
    """Load data and isolate the chronological training period."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(
        DATA_PATH,
        index_col="Date",
        parse_dates=True,
    )

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data.dropna(subset=required_columns).sort_index()

    train_end = int(len(data) * TRAIN_RATIO)

    train = data.iloc[:train_end - PURGE_GAP].copy()

    return train


# ---------------------------------------------------------
# FEATURE CORRELATION
# ---------------------------------------------------------

def analyze_feature_correlation(train):
    """Identify strongly correlated feature pairs."""

    correlation_matrix = train[FEATURE_COLUMNS].corr()

    print("\n" + "=" * 70)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 70)

    pairs = []

    for feature_a, feature_b in combinations(FEATURE_COLUMNS, 2):
        correlation = correlation_matrix.loc[feature_a, feature_b]

        if abs(correlation) >= CORRELATION_THRESHOLD:
            pairs.append({
                "feature_1": feature_a,
                "feature_2": feature_b,
                "correlation": correlation,
            })

    if pairs:
        results = pd.DataFrame(pairs).sort_values(
            "correlation",
            key=lambda values: values.abs(),
            ascending=False,
        )

        print(
            f"\nFeature pairs with |correlation| >= "
            f"{CORRELATION_THRESHOLD}:"
        )

        print(
            results.to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )
    else:
        print(
            f"\nNo feature pairs with absolute correlation "
            f">= {CORRELATION_THRESHOLD}."
        )

    print("\nComplete feature correlation matrix:")
    print(
        correlation_matrix.to_string(
            float_format=lambda value: f"{value:.2f}",
        )
    )

    return correlation_matrix


# ---------------------------------------------------------
# MODEL FACTORIES
# ---------------------------------------------------------

def create_logistic_model():
    """Create a standardized Logistic Regression pipeline."""

    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def create_dummy_model():
    """Create a majority-class benchmark."""

    return DummyClassifier(strategy="most_frequent")


# ---------------------------------------------------------
# TIME SERIES CROSS-VALIDATION
# ---------------------------------------------------------

def evaluate_feature_set(train, feature_columns, model_name):
    """Evaluate a feature set using chronological cross-validation."""

    X = train[feature_columns]
    y = train[TARGET_COLUMN]

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        gap=PURGE_GAP,
    )

    fold_results = []

    for fold, (train_idx, validation_idx) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_validation = X.iloc[validation_idx]

        y_train = y.iloc[train_idx]
        y_validation = y.iloc[validation_idx]

        if model_name == "Dummy Classifier":
            model = create_dummy_model()
        else:
            model = create_logistic_model()

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)
        probabilities = model.predict_proba(X_validation)[:, 1]

        fold_results.append({
            "fold": fold,
            "accuracy": accuracy_score(
                y_validation,
                predictions,
            ),
            "roc_auc": roc_auc_score(
                y_validation,
                probabilities,
            ),
        })

    results = pd.DataFrame(fold_results)

    return {
        "accuracy_mean": results["accuracy"].mean(),
        "accuracy_std": results["accuracy"].std(),
        "roc_auc_mean": results["roc_auc"].mean(),
        "roc_auc_std": results["roc_auc"].std(),
    }


# ---------------------------------------------------------
# FEATURE ABLATION
# ---------------------------------------------------------

def analyze_feature_ablation(train):
    """Compare the full model against models without feature groups."""

    experiments = {
        "All features": FEATURE_COLUMNS.copy(),
    }

    for group_name, group_features in FEATURE_GROUPS.items():
        experiments[f"Without {group_name}"] = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in group_features
        ]

    print("\n" + "=" * 70)
    print("FEATURE ABLATION - TIME SERIES CROSS-VALIDATION")
    print("=" * 70)

    print(f"\nTraining observations: {len(train)}")
    print(f"Cross-validation folds: {N_SPLITS}")
    print(f"Purge gap: {PURGE_GAP}")
    print("Validation and Test Sets are not used.")

    results = []

    # Dummy benchmark
    dummy_result = evaluate_feature_set(
        train,
        FEATURE_COLUMNS,
        "Dummy Classifier",
    )

    results.append({
        "experiment": "Dummy Classifier",
        "features": len(FEATURE_COLUMNS),
        **dummy_result,
    })

    # Logistic Regression experiments
    for experiment_name, selected_features in experiments.items():

        print(f"\nEvaluating: {experiment_name}")
        print(f"Number of features: {len(selected_features)}")

        result = evaluate_feature_set(
            train,
            selected_features,
            "Logistic Regression",
        )

        results.append({
            "experiment": experiment_name,
            "features": len(selected_features),
            **result,
        })

        print(
            f"Mean ROC-AUC: {result['roc_auc_mean']:.4f} "
            f"+/- {result['roc_auc_std']:.4f}"
        )

    summary = pd.DataFrame(results)

    summary = summary.sort_values(
        "roc_auc_mean",
        ascending=False,
    )

    print("\n" + "=" * 70)
    print("FEATURE ABLATION SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return summary


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    """Run feature correlation and ablation analysis."""

    print("\n" + "=" * 70)
    print("STOCK PREDICTION ENGINE")
    print("FEATURE CORRELATION & ABLATION ANALYSIS")
    print("=" * 70)

    train = load_training_data()

    print(f"\nPrediction horizon: {HORIZON} trading sessions")
    print(f"Training observations: {len(train)}")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")

    analyze_feature_correlation(train)

    analyze_feature_ablation(train)

    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nOriginal dataset was not modified.")
    print("Validation Set was not used.")
    print("Test Set was not used.")


if __name__ == "__main__":
    main()
