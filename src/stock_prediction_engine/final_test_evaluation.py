
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

SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = DATA_DIR / "models"

TRAIN_FILE = SPLITS_DIR / "train.csv"
VALIDATION_FILE = SPLITS_DIR / "validation.csv"
TEST_FILE = SPLITS_DIR / "test.csv"

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
# VALIDATION
# ============================================================

def validate_configurations():
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

        if not set(features).issubset(FULL_FEATURES):
            raise ValueError(
                f"Unknown feature found in {name}."
            )


def load_datasets():

    for path in [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {path}"
            )

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)
    test = pd.read_csv(TEST_FILE)

    for dataset in [train, validation, test]:
        dataset["Date"] = pd.to_datetime(dataset["Date"])

    return train, validation, test


def validate_datasets(train, validation, test):

    required_columns = {
        "Date",
        "Ticker",
        *FULL_FEATURES,
    }

    for horizon in HORIZONS:
        required_columns.add(
            f"target_direction_{horizon}d"
        )
        required_columns.add(
            f"target_return_{horizon}d"
        )

    for name, dataset in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        missing = required_columns - set(dataset.columns)

        if missing:
            raise ValueError(
                f"{name}: missing columns {sorted(missing)}"
            )

        if dataset[list(required_columns)].isna().any().any():
            raise ValueError(
                f"{name}: missing values found."
            )

        if dataset.duplicated(["Date", "Ticker"]).any():
            raise ValueError(
                f"{name}: duplicate Date/Ticker observations."
            )

    if train["Date"].max() >= validation["Date"].min():
        raise ValueError(
            "Training and Validation dates overlap."
        )

    if validation["Date"].max() >= test["Date"].min():
        raise ValueError(
            "Validation and Test dates overlap."
        )

    # The last training/validation observations must be
    # separated from the next split by the existing purge gap.

    print("\nDATASET AUDIT")
    print("=" * 70)

    for name, dataset in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        print(
            f"{name}: {len(dataset):,} rows | "
            f"{dataset['Date'].nunique()} dates | "
            f"{dataset['Ticker'].nunique()} assets | "
            f"{dataset['Date'].min().date()} -> "
            f"{dataset['Date'].max().date()}"
        )

    print("\nDate overlap checks passed.")


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_preprocessor():

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
    test,
    features,
    horizon,
    configuration,
):

    target = f"target_direction_{horizon}d"

    X_train = train[["Ticker"] + features]
    X_test = test[["Ticker"] + features]

    y_train = train[target].astype(int)
    y_test = test[target].astype(int)

    model = build_classifier()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)

    classes = list(model.named_steps["model"].classes_)

    if 1 not in classes:
        raise ValueError(
            "Positive class 1 not found in training data."
        )

    positive_index = classes.index(1)

    predicted_probabilities = probabilities[:, positive_index]
    predictions = (
        predicted_probabilities >= 0.5
    ).astype(int)

    auc = (
        roc_auc_score(y_test, predicted_probabilities)
        if y_test.nunique() == 2
        else np.nan
    )

    metrics = {
        "configuration": configuration,
        "task": "classification",
        "horizon": horizon,
        "n_features": len(features),
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test,
            predictions,
        ),
        "roc_auc": auc,
        "brier_score": brier_score_loss(
            y_test,
            predicted_probabilities,
        ),
        "predicted_up_rate": predictions.mean(),
        "actual_up_rate": y_test.mean(),
    }

    predictions_data = test[
        ["Date", "Ticker"]
    ].copy()

    predictions_data["configuration"] = configuration
    predictions_data["task"] = "classification"
    predictions_data["horizon"] = horizon
    predictions_data["actual"] = y_test.to_numpy()
    predictions_data["prediction"] = predictions
    predictions_data["probability_up"] = predicted_probabilities

    # Baseline: training-set class prior.

    baseline = DummyClassifier(strategy="prior")

    baseline.fit(
        np.zeros((len(y_train), 1)),
        y_train,
    )

    baseline_classes = list(baseline.classes_)
    baseline_positive_index = baseline_classes.index(1)

    baseline_probabilities = baseline.predict_proba(
        np.zeros((len(y_test), 1))
    )[:, baseline_positive_index]

    baseline_predictions = (
        baseline_probabilities >= 0.5
    ).astype(int)

    baseline_metrics = {
        "configuration": "baseline_prior",
        "task": "classification",
        "horizon": horizon,
        "n_features": 0,
        "accuracy": accuracy_score(
            y_test,
            baseline_predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_test,
            baseline_predictions,
        ),
        "roc_auc": (
            roc_auc_score(y_test, baseline_probabilities)
            if y_test.nunique() == 2
            else np.nan
        ),
        "brier_score": brier_score_loss(
            y_test,
            baseline_probabilities,
        ),
        "predicted_up_rate": baseline_predictions.mean(),
        "actual_up_rate": y_test.mean(),
    }

    baseline_predictions_data = test[
        ["Date", "Ticker"]
    ].copy()

    baseline_predictions_data["configuration"] = "baseline_prior"
    baseline_predictions_data["task"] = "classification"
    baseline_predictions_data["horizon"] = horizon
    baseline_predictions_data["actual"] = y_test.to_numpy()
    baseline_predictions_data["prediction"] = baseline_predictions
    baseline_predictions_data["probability_up"] = baseline_probabilities

    return (
        [metrics, baseline_metrics],
        [predictions_data, baseline_predictions_data],
    )


# ============================================================
# REGRESSION
# ============================================================

def evaluate_regression(
    train,
    test,
    features,
    horizon,
    configuration,
):

    target = f"target_return_{horizon}d"

    X_train = train[["Ticker"] + features]
    X_test = test[["Ticker"] + features]

    y_train = train[target].astype(float)
    y_test = test[target].astype(float)

    model = build_regressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "configuration": configuration,
        "task": "regression",
        "horizon": horizon,
        "n_features": len(features),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "r2": r2_score(y_test, predictions),
        "prediction_mean": predictions.mean(),
        "actual_mean": y_test.mean(),
    }

    predictions_data = test[
        ["Date", "Ticker"]
    ].copy()

    predictions_data["configuration"] = configuration
    predictions_data["task"] = "regression"
    predictions_data["horizon"] = horizon
    predictions_data["actual"] = y_test.to_numpy()
    predictions_data["prediction"] = predictions

    # Baseline: historical mean return per ticker.
    # Calculated exclusively from Training + Validation.

    asset_means = train.groupby("Ticker")[target].mean()

    baseline_predictions = (
        test["Ticker"]
        .map(asset_means)
        .fillna(y_train.mean())
        .to_numpy()
    )

    baseline_metrics = {
        "configuration": "baseline_asset_mean",
        "task": "regression",
        "horizon": horizon,
        "n_features": 0,
        "mae": mean_absolute_error(
            y_test,
            baseline_predictions,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_test,
                baseline_predictions,
            )
        ),
        "r2": r2_score(
            y_test,
            baseline_predictions,
        ),
        "prediction_mean": baseline_predictions.mean(),
        "actual_mean": y_test.mean(),
    }

    baseline_predictions_data = test[
        ["Date", "Ticker"]
    ].copy()

    baseline_predictions_data["configuration"] = (
        "baseline_asset_mean"
    )
    baseline_predictions_data["task"] = "regression"
    baseline_predictions_data["horizon"] = horizon
    baseline_predictions_data["actual"] = y_test.to_numpy()
    baseline_predictions_data["prediction"] = baseline_predictions

    return (
        [metrics, baseline_metrics],
        [predictions_data, baseline_predictions_data],
    )


# ============================================================
# PER-ASSET ANALYSIS
# ============================================================

def create_asset_metrics(predictions):

    records = []

    for (configuration, task, horizon, ticker), group in (
        predictions.groupby(
            ["configuration", "task", "horizon", "Ticker"]
        )
    ):

        actual = group["actual"].to_numpy()
        predicted = group["prediction"].to_numpy()

        row = {
            "configuration": configuration,
            "task": task,
            "horizon": horizon,
            "ticker": ticker,
            "observations": len(group),
        }

        if task == "classification":

            probabilities = group["probability_up"].to_numpy()

            row["accuracy"] = accuracy_score(
                actual,
                predicted,
            )

            row["balanced_accuracy"] = balanced_accuracy_score(
                actual,
                predicted,
            )

            row["roc_auc"] = (
                roc_auc_score(actual, probabilities)
                if len(np.unique(actual)) == 2
                else np.nan
            )

            row["brier_score"] = brier_score_loss(
                actual,
                probabilities,
            )

        else:

            row["mae"] = mean_absolute_error(
                actual,
                predicted,
            )

            row["rmse"] = np.sqrt(
                mean_squared_error(actual, predicted)
            )

            row["r2"] = (
                r2_score(actual, predicted)
                if len(actual) > 1
                else np.nan
            )

        records.append(row)

    return pd.DataFrame(records)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("FINAL OUT-OF-SAMPLE TEST")
    print("=" * 70)

    validate_configurations()

    train, validation, test = load_datasets()

    validate_datasets(
        train,
        validation,
        test,
    )

    # Final model fitting uses all pre-test observations.
    # The existing purge gap before the Test Set is preserved.

    development_data = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    development_data = development_data.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    if development_data["Date"].max() >= test["Date"].min():
        raise ValueError(
            "Development data overlaps with Test."
        )

    print("\nFINAL TRAINING DATA")
    print("=" * 70)
    print(f"Rows: {len(development_data):,}")
    print(f"Assets: {development_data['Ticker'].nunique()}")
    print(f"Dates: {development_data['Date'].nunique():,}")
    print(
        f"Period: {development_data['Date'].min().date()} "
        f"-> {development_data['Date'].max().date()}"
    )

    print("\nFINAL TEST DATA")
    print("=" * 70)
    print(f"Rows: {len(test):,}")
    print(f"Assets: {test['Ticker'].nunique()}")
    print(f"Dates: {test['Date'].nunique():,}")
    print(
        f"Period: {test['Date'].min().date()} "
        f"-> {test['Date'].max().date()}"
    )

    print("\nIMPORTANT:")
    print("Feature configurations are frozen.")
    print("No hyperparameter tuning will be performed.")
    print("Test results must not be used to retune the models.")

    all_metrics = []
    all_predictions = []

    for horizon in HORIZONS:

        print("\n" + "=" * 70)
        print(f"HORIZON: {horizon} DAYS")
        print("=" * 70)

        for configuration, features in (
            FEATURE_CONFIGURATIONS.items()
        ):

            print(
                f"\nEvaluating {configuration} "
                f"({len(features)} features)"
            )

            classification_metrics, classification_predictions = (
                evaluate_classification(
                    development_data,
                    test,
                    features,
                    horizon,
                    configuration,
                )
            )

            regression_metrics, regression_predictions = (
                evaluate_regression(
                    development_data,
                    test,
                    features,
                    horizon,
                    configuration,
                )
            )

            all_metrics.extend(classification_metrics)
            all_metrics.extend(regression_metrics)

            all_predictions.extend(classification_predictions)
            all_predictions.extend(regression_predictions)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    asset_metrics = create_asset_metrics(
        predictions_df
    )

    metrics_path = (
        MODELS_DIR / "final_test_results.csv"
    )

    predictions_path = (
        MODELS_DIR / "final_test_predictions.csv"
    )

    asset_metrics_path = (
        MODELS_DIR / "final_test_asset_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    asset_metrics.to_csv(
        asset_metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(metrics_path)
    print(predictions_path)
    print(asset_metrics_path)

    print("\nFinal out-of-sample evaluation completed.")


if __name__ == "__main__":
    main()
