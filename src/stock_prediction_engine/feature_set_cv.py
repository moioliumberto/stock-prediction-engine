
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_asset"
)

MODELS_DIR = DATA_DIR / "models"
TRAIN_FILE = DATA_DIR / "splits" / "train.csv"

N_SPLITS = 5
PURGE_GAP = 20

RANDOM_STATE = 42
N_ESTIMATORS = 200
MIN_SAMPLES_LEAF = 10
N_JOBS = 1

HORIZONS = [5, 10, 20]


# ============================================================
# FEATURE CONFIGURATIONS
# ============================================================

FULL_FEATURES = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",

    "price_to_sma_10",
    "price_to_sma_20",
    "price_to_sma_50",
    "price_to_sma_100",
    "price_to_sma_200",
    "price_to_ema_12",
    "price_to_ema_26",
    "sma_20_to_sma_50",
    "sma_50_to_sma_200",

    "volatility_10d",
    "volatility_20d",
    "volatility_60d",
    "volatility_ratio_10_60",
    "atr_pct",

    "rsi_14",
    "macd_pct",
    "macd_signal_pct",
    "macd_hist_pct",

    "volume_ratio_20",
    "obv_change_20",

    "market_spy_return_1d",
    "market_spy_return_5d",
    "market_spy_return_20d",
    "market_qqq_return_5d",
    "market_qqq_return_20d",
    "market_iwm_return_20d",
    "market_tlt_return_20d",
    "market_gld_return_20d",
    "market_uso_return_20d",
    "market_spy_volatility_20d",

    "relative_strength_spy_5d",
    "relative_strength_spy_20d",
    "relative_strength_spy_60d",
    "correlation_spy_60d",
    "beta_spy_60d",
]


CORE_FEATURES = [
    "price_to_sma_100",
    "sma_50_to_sma_200",
    "volatility_10d",
    "volatility_60d",
    "market_qqq_return_5d",
    "market_qqq_return_20d",
    "correlation_spy_60d",
    "beta_spy_60d",
]


REDUCED_FEATURES = CORE_FEATURES + [
    "obv_change_20",
    "market_gld_return_20d",
    "relative_strength_spy_5d",
    "return_60d",
]


BALANCED_FEATURES = CORE_FEATURES + [
    "return_60d",
    "price_to_sma_50",
    "atr_pct",
    "rsi_14",
    "obv_change_20",
    "market_gld_return_20d",
    "market_spy_volatility_20d",
    "relative_strength_spy_5d",
]


FEATURE_CONFIGURATIONS = {
    "full_40": FULL_FEATURES,
    "core_8": CORE_FEATURES,
    "reduced_12": REDUCED_FEATURES,
    "balanced_16": BALANCED_FEATURES,
}


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_configurations():
    """Validate feature definitions and configuration sizes."""

    expected_sizes = {
        "full_40": 40,
        "core_8": 8,
        "reduced_12": 12,
        "balanced_16": 16,
    }

    for name, features in FEATURE_CONFIGURATIONS.items():

        if len(features) != expected_sizes[name]:
            raise ValueError(
                f"{name}: expected {expected_sizes[name]} "
                f"features, found {len(features)}."
            )

        if len(set(features)) != len(features):
            raise ValueError(
                f"Duplicate features found in {name}."
            )

        missing_from_full = set(features) - set(FULL_FEATURES)

        if missing_from_full:
            raise ValueError(
                f"Unknown features in {name}: "
                f"{sorted(missing_from_full)}"
            )

    if not set(CORE_FEATURES).issubset(REDUCED_FEATURES):
        raise ValueError(
            "Reduced configuration must contain Core features."
        )

    if not set(CORE_FEATURES).issubset(BALANCED_FEATURES):
        raise ValueError(
            "Balanced configuration must contain Core features."
        )


def load_training_data():
    """Load and validate the Training Set only."""

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_FILE}"
        )

    data = pd.read_csv(TRAIN_FILE)

    data["Date"] = pd.to_datetime(data["Date"])

    required_columns = {
        "Date",
        "Ticker",
        *FULL_FEATURES,
    }

    # Classification targets use target_direction_{horizon}d.
    # Regression targets use target_return_{horizon}d.

    for horizon in HORIZONS:
        required_columns.add(
            f"target_direction_{horizon}d"
        )
        required_columns.add(
            f"target_return_{horizon}d"
        )

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns in training dataset: "
            f"{sorted(missing)}"
        )

    if data[list(required_columns)].isna().any().any():
        raise ValueError(
            "Missing values found in required training columns."
        )

    if data.duplicated(["Date", "Ticker"]).any():
        raise ValueError(
            "Duplicate Date/Ticker observations found."
        )

    data = data.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    return data


# ============================================================
# TEMPORAL FOLDS
# ============================================================

def create_temporal_folds(data):
    """Create chronological folds with a 20-date purge gap."""

    unique_dates = np.array(
        sorted(data["Date"].unique())
    )

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    folds = []

    for fold_number, (train_idx, val_idx) in enumerate(
        splitter.split(unique_dates),
        start=1,
    ):

        original_train_dates = unique_dates[train_idx]
        val_dates = unique_dates[val_idx]

        if len(original_train_dates) <= PURGE_GAP:
            raise ValueError(
                f"Fold {fold_number}: insufficient dates "
                f"for purge gap."
            )

        # Exclude the last 20 training dates.
        train_dates = original_train_dates[:-PURGE_GAP]

        fold_train = data[
            data["Date"].isin(train_dates)
        ].copy()

        fold_val = data[
            data["Date"].isin(val_dates)
        ].copy()

        if fold_train.empty or fold_val.empty:
            raise ValueError(
                f"Fold {fold_number} is empty."
            )

        if fold_train["Date"].max() >= fold_val["Date"].min():
            raise ValueError(
                f"Temporal overlap detected in fold {fold_number}."
            )

        folds.append(
            {
                "fold": fold_number,
                "train": fold_train,
                "validation": fold_val,
            }
        )

    return folds


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_preprocessor():
    """One-hot encode the ticker and pass numerical features through."""

    return ColumnTransformer(
        transformers=[
            (
                "ticker",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
                ["Ticker"],
            )
        ],
        remainder="passthrough",
    )


def build_classifier():
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    )


def build_regressor():
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    max_features=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def evaluate_classification(
    train,
    validation,
    features,
    horizon,
    fold_number,
    configuration,
):
    """Train and evaluate a classification model."""

    # Correct target name from the actual dataset.
    target = f"target_direction_{horizon}d"

    X_train = train[["Ticker"] + features]
    X_val = validation[["Ticker"] + features]

    y_train = train[target].astype(int)
    y_val = validation[target].astype(int)

    model = build_classifier()

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)

    classes = list(
        model.named_steps["model"].classes_
    )

    if 1 not in classes:
        raise ValueError(
            f"Positive class 1 not found in fold {fold_number}."
        )

    positive_class_index = classes.index(1)

    predictions_proba = probabilities[:, positive_class_index]
    predictions = (
        predictions_proba >= 0.5
    ).astype(int)

    auc = (
        roc_auc_score(y_val, predictions_proba)
        if y_val.nunique() == 2
        else np.nan
    )

    model_results = {
        "configuration": configuration,
        "task": "classification",
        "horizon": horizon,
        "fold": fold_number,
        "n_features": len(features),
        "accuracy": accuracy_score(y_val, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_val,
            predictions,
        ),
        "roc_auc": auc,
        "brier_score": brier_score_loss(
            y_val,
            predictions_proba,
        ),
        "predicted_up_rate": predictions.mean(),
        "actual_up_rate": y_val.mean(),
    }

    # Baseline: training-set class prior.

    baseline = DummyClassifier(
        strategy="prior"
    )

    baseline.fit(
        np.zeros((len(y_train), 1)),
        y_train,
    )

    baseline_classes = list(baseline.classes_)

    baseline_positive_index = baseline_classes.index(1)

    baseline_proba = baseline.predict_proba(
        np.zeros((len(y_val), 1))
    )[:, baseline_positive_index]

    baseline_predictions = (
        baseline_proba >= 0.5
    ).astype(int)

    baseline_auc = (
        roc_auc_score(y_val, baseline_proba)
        if y_val.nunique() == 2
        else np.nan
    )

    baseline_results = {
        "configuration": "baseline_prior",
        "task": "classification",
        "horizon": horizon,
        "fold": fold_number,
        "n_features": 0,
        "accuracy": accuracy_score(
            y_val,
            baseline_predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_val,
            baseline_predictions,
        ),
        "roc_auc": baseline_auc,
        "brier_score": brier_score_loss(
            y_val,
            baseline_proba,
        ),
        "predicted_up_rate": baseline_predictions.mean(),
        "actual_up_rate": y_val.mean(),
    }

    return [
        model_results,
        baseline_results,
    ]


# ============================================================
# REGRESSION
# ============================================================

def evaluate_regression(
    train,
    validation,
    features,
    horizon,
    fold_number,
    configuration,
):
    """Train and evaluate a regression model."""

    target = f"target_return_{horizon}d"

    X_train = train[["Ticker"] + features]
    X_val = validation[["Ticker"] + features]

    y_train = train[target].astype(float)
    y_val = validation[target].astype(float)

    model = build_regressor()

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    model_results = {
        "configuration": configuration,
        "task": "regression",
        "horizon": horizon,
        "fold": fold_number,
        "n_features": len(features),
        "mae": mean_absolute_error(
            y_val,
            predictions,
        ),
        "rmse": np.sqrt(
            mean_squared_error(y_val, predictions)
        ),
        "r2": r2_score(y_val, predictions),
        "prediction_mean": predictions.mean(),
        "actual_mean": y_val.mean(),
    }

    # Baseline: historical mean return for each ticker,
    # calculated using the training fold only.

    asset_means = train.groupby("Ticker")[target].mean()

    baseline_predictions = (
        validation["Ticker"]
        .map(asset_means)
        .fillna(y_train.mean())
        .to_numpy()
    )

    baseline_results = {
        "configuration": "baseline_asset_mean",
        "task": "regression",
        "horizon": horizon,
        "fold": fold_number,
        "n_features": 0,
        "mae": mean_absolute_error(
            y_val,
            baseline_predictions,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_val,
                baseline_predictions,
            )
        ),
        "r2": r2_score(
            y_val,
            baseline_predictions,
        ),
        "prediction_mean": baseline_predictions.mean(),
        "actual_mean": y_val.mean(),
    }

    return [
        model_results,
        baseline_results,
    ]


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(data):
    """Run all configurations across temporal folds."""

    folds = create_temporal_folds(data)

    results = []

    print("\nTEMPORAL CROSS-VALIDATION")
    print("=" * 70)

    for fold in folds:

        fold_number = fold["fold"]
        train = fold["train"]
        validation = fold["validation"]

        print(f"\nFold {fold_number}/{N_SPLITS}")

        print(
            f"Training: {train['Date'].min().date()} "
            f"-> {train['Date'].max().date()} "
            f"({train['Date'].nunique()} dates)"
        )

        print(
            f"Validation: {validation['Date'].min().date()} "
            f"-> {validation['Date'].max().date()} "
            f"({validation['Date'].nunique()} dates)"
        )

        for horizon in HORIZONS:

            for configuration, features in (
                FEATURE_CONFIGURATIONS.items()
            ):

                print(
                    f"  {configuration} | "
                    f"{horizon}d | "
                    f"{len(features)} features"
                )

                classification_results = (
                    evaluate_classification(
                        train,
                        validation,
                        features,
                        horizon,
                        fold_number,
                        configuration,
                    )
                )

                regression_results = (
                    evaluate_regression(
                        train,
                        validation,
                        features,
                        horizon,
                        fold_number,
                        configuration,
                    )
                )

                results.extend(classification_results)
                results.extend(regression_results)

    return pd.DataFrame(results)


# ============================================================
# SUMMARY
# ============================================================

def create_summary(results):
    """Calculate mean and standard deviation across folds."""

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "brier_score",
        "predicted_up_rate",
        "actual_up_rate",
        "mae",
        "rmse",
        "r2",
        "prediction_mean",
        "actual_mean",
    ]

    available_metrics = [
        column
        for column in metric_columns
        if column in results.columns
    ]

    grouped = results.groupby(
        [
            "configuration",
            "task",
            "horizon",
            "n_features",
        ]
    )[available_metrics]

    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std().add_suffix("_std")

    return pd.concat(
        [means, stds],
        axis=1,
    ).reset_index()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("FEATURE SET COMPARISON")
    print("=" * 70)

    validate_configurations()

    for name, features in FEATURE_CONFIGURATIONS.items():
        print(f"{name}: {len(features)} features")

    data = load_training_data()

    print("\nTraining data loaded.")
    print(f"Observations: {len(data):,}")
    print(f"Assets: {data['Ticker'].nunique()}")
    print(f"Dates: {data['Date'].nunique():,}")

    print("\nIMPORTANT:")
    print("Only train.csv is used.")
    print("Validation and Test are not loaded.")

    results = run_experiment(data)

    summary = create_summary(results)

    results_path = (
        MODELS_DIR / "feature_set_cv_results.csv"
    )

    summary_path = (
        MODELS_DIR / "feature_set_cv_summary.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    display_columns = [
        "configuration",
        "task",
        "horizon",
        "n_features",
    ]

    for column in [
        "roc_auc_mean",
        "balanced_accuracy_mean",
        "brier_score_mean",
        "mae_mean",
        "rmse_mean",
        "r2_mean",
    ]:
        if column in summary.columns:
            display_columns.append(column)

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(results_path)
    print(summary_path)

    print("\nFeature set comparison completed.")


if __name__ == "__main__":
    main()
