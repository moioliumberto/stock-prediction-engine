
"""Diagnose Logistic Regression and analyze feature behavior."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
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
VALIDATION_RATIO = 0.15
PURGE_GAP = HORIZON

RANDOM_STATE = 42


# ---------------------------------------------------------
# DATA LOADING AND TEMPORAL SPLIT
# ---------------------------------------------------------

def load_dataset():
    """Load the feature dataset and validate required columns."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(
        DATA_PATH,
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
            f"Missing columns in dataset: {missing_columns}"
        )

    data = data.dropna(subset=required_columns).copy()
    data = data.sort_index()

    return data


def chronological_split(data):
    """Create chronological training and validation datasets."""

    train_end = int(len(data) * TRAIN_RATIO)
    validation_end = int(
        len(data) * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train = data.iloc[:train_end - PURGE_GAP].copy()

    validation = data.iloc[
        train_end:validation_end - PURGE_GAP
    ].copy()

    return train, validation


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def create_model():
    """Create the Logistic Regression pipeline."""

    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# ---------------------------------------------------------
# FEATURE COEFFICIENT ANALYSIS
# ---------------------------------------------------------

def analyze_coefficients(model):
    """Analyze standardized Logistic Regression coefficients."""

    classifier = model.named_steps["logistic_regression"]

    coefficients = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "coefficient": classifier.coef_[0],
    })

    coefficients["absolute_coefficient"] = (
        coefficients["coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    print("\n" + "=" * 65)
    print("FEATURE COEFFICIENT ANALYSIS")
    print("=" * 65)

    print(
        coefficients.to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print("\nInterpretation:")
    print("Positive coefficient: increases the estimated probability of Up.")
    print("Negative coefficient: decreases the estimated probability of Up.")
    print("Larger absolute value: stronger contribution within this model.")

    return coefficients


# ---------------------------------------------------------
# FEATURE-TARGET CORRELATION
# ---------------------------------------------------------

def analyze_correlations(train, validation):
    """Compare feature-target correlations across periods."""

    train_correlations = train[
        FEATURE_COLUMNS + [TARGET_COLUMN]
    ].corr()[TARGET_COLUMN].drop(TARGET_COLUMN)

    validation_correlations = validation[
        FEATURE_COLUMNS + [TARGET_COLUMN]
    ].corr()[TARGET_COLUMN].drop(TARGET_COLUMN)

    comparison = pd.DataFrame({
        "train_correlation": train_correlations,
        "validation_correlation": validation_correlations,
    })

    comparison["absolute_train"] = (
        comparison["train_correlation"].abs()
    )

    comparison["absolute_validation"] = (
        comparison["validation_correlation"].abs()
    )

    comparison["sign_changed"] = (
        np.sign(comparison["train_correlation"])
        != np.sign(comparison["validation_correlation"])
    )

    comparison = comparison.sort_values(
        "absolute_train",
        ascending=False,
    )

    print("\n" + "=" * 65)
    print("FEATURE-TARGET CORRELATION")
    print("=" * 65)

    print(
        comparison.to_string(
            float_format=lambda value: f"{value:.4f}",
        )
    )

    changed_count = comparison["sign_changed"].sum()

    print(
        f"\nFeatures with correlation sign changes: "
        f"{changed_count}/{len(comparison)}"
    )

    return comparison


# ---------------------------------------------------------
# DISTRIBUTION SHIFT
# ---------------------------------------------------------

def analyze_distribution_shift(train, validation):
    """Compare feature distributions between training and validation."""

    train_stats = train[FEATURE_COLUMNS].agg(["mean", "std"]).T
    validation_stats = validation[FEATURE_COLUMNS].agg(
        ["mean", "std"]
    ).T

    comparison = pd.DataFrame({
        "train_mean": train_stats["mean"],
        "validation_mean": validation_stats["mean"],
        "train_std": train_stats["std"],
        "validation_std": validation_stats["std"],
    })

    comparison["mean_difference"] = (
        comparison["validation_mean"]
        - comparison["train_mean"]
    )

    comparison["std_ratio"] = (
        comparison["validation_std"]
        / comparison["train_std"].replace(0, np.nan)
    )

    comparison["standardized_mean_difference"] = (
        comparison["mean_difference"]
        / comparison["train_std"].replace(0, np.nan)
    )

    comparison = comparison.sort_values(
        "standardized_mean_difference",
        key=lambda values: values.abs(),
        ascending=False,
    )

    print("\n" + "=" * 65)
    print("FEATURE DISTRIBUTION SHIFT")
    print("=" * 65)

    print(
        comparison.to_string(
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return comparison


# ---------------------------------------------------------
# PREDICTION DIAGNOSTICS
# ---------------------------------------------------------

def analyze_predictions(model, validation):
    """Analyze validation predictions and probability distribution."""

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN]

    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = model.predict(X_validation)

    probability_series = pd.Series(probabilities)

    print("\n" + "=" * 65)
    print("VALIDATION PREDICTION DIAGNOSTICS")
    print("=" * 65)

    print("\nProbability distribution:")
    print(
        probability_series.describe(
            percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
        ).to_string()
    )

    print("\nPredicted class distribution:")
    print(
        pd.Series(predictions)
        .value_counts(normalize=True)
        .rename(index={0: "Down", 1: "Up"})
        .to_string()
    )

    print("\nActual class distribution:")
    print(
        y_validation.value_counts(normalize=True)
        .rename(index={0: "Down", 1: "Up"})
        .to_string()
    )

    print("\nValidation metrics:")
    print(f"Accuracy: {accuracy_score(y_validation, predictions):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_validation, probabilities):.4f}")

    print("\nProbability threshold analysis:")

    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60]:
        threshold_predictions = (
            probabilities >= threshold
        ).astype(int)

        predicted_up = threshold_predictions.mean()

        print(
            f"Threshold {threshold:.2f} | "
            f"Predicted Up: {predicted_up:.2%}"
        )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    """Run the complete model diagnostics experiment."""

    print("\n" + "=" * 65)
    print("STOCK PREDICTION ENGINE")
    print("MODEL DIAGNOSTICS & FEATURE ANALYSIS")
    print("=" * 65)

    data = load_dataset()

    train, validation = chronological_split(data)

    print(f"\nPrediction horizon: {HORIZON} trading sessions")
    print(f"Total observations: {len(data)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    print("\nTraining dataset:")
    print(f"Observations: {len(train)}")
    print(f"Start: {train.index.min()}")
    print(f"End:   {train.index.max()}")

    print("\nValidation dataset:")
    print(f"Observations: {len(validation)}")
    print(f"Start: {validation.index.min()}")
    print(f"End:   {validation.index.max()}")

    print(f"\nPurge gap: {PURGE_GAP} observations")

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    model = create_model()

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    analyze_coefficients(model)

    analyze_correlations(train, validation)

    analyze_distribution_shift(train, validation)

    analyze_predictions(model, validation)

    print("\n" + "=" * 65)
    print("MODEL DIAGNOSTICS COMPLETED")
    print("=" * 65)

    print("\nNo changes were made to the original dataset.")
    print("The Test Set was not used.")


if __name__ == "__main__":
    main()
