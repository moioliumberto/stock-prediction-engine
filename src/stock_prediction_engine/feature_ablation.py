
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
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
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_asset"
    / "splits"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_asset"
    / "models"
)

HORIZONS = (5, 10, 20)

RANDOM_STATE = 42
N_ESTIMATORS = 300
MIN_SAMPLES_LEAF = 10

# Use one worker to avoid the parallelism warning
# encountered in the previous experiment.
N_JOBS = 1


# ============================================================
# FEATURE GROUPS
# ============================================================

FEATURE_GROUPS = {

    "momentum": [
        "return_1d",
        "return_5d",
        "return_10d",
        "return_20d",
        "return_60d",
    ],

    "trend": [
        "price_to_sma_10",
        "price_to_sma_20",
        "price_to_sma_50",
        "price_to_sma_100",
        "price_to_sma_200",
        "price_to_ema_12",
        "price_to_ema_26",
        "sma_20_to_sma_50",
        "sma_50_to_sma_200",
    ],

    "volatility": [
        "volatility_10d",
        "volatility_20d",
        "volatility_60d",
        "volatility_ratio_10_60",
        "atr_pct",
    ],

    "oscillators": [
        "rsi_14",
        "macd_pct",
        "macd_signal_pct",
        "macd_hist_pct",
    ],

    "volume": [
        "volume_ratio_20",
        "obv_change_20",
    ],

    "market_context": [
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
    ],

    "relative_strength": [
        "relative_strength_spy_5d",
        "relative_strength_spy_20d",
        "relative_strength_spy_60d",
        "correlation_spy_60d",
        "beta_spy_60d",
    ],
}


NUMERIC_FEATURES = [
    feature
    for group in FEATURE_GROUPS.values()
    for feature in group
]


# ============================================================
# REDUCED FEATURE SET
# ============================================================

# This is a manually selected candidate dataset.
# Selection is based on the previous redundancy and
# permutation analyses, not on the test dataset.

REDUCED_FEATURES = [

    # Momentum
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",

    # Trend
    "price_to_sma_20",
    "price_to_sma_50",
    "price_to_sma_100",
    "price_to_sma_200",
    "sma_20_to_sma_50",
    "sma_50_to_sma_200",

    # Volatility
    "volatility_10d",
    "volatility_20d",
    "volatility_60d",
    "volatility_ratio_10_60",

    # Oscillators
    "rsi_14",
    "macd_pct",
    "macd_hist_pct",

    # Volume
    "volume_ratio_20",
    "obv_change_20",

    # Market context
    "market_spy_return_1d",
    "market_qqq_return_5d",
    "market_qqq_return_20d",
    "market_iwm_return_20d",
    "market_tlt_return_20d",
    "market_gld_return_20d",
    "market_uso_return_20d",
    "market_spy_volatility_20d",

    # Relative strength and sensitivity
    "relative_strength_spy_5d",
    "relative_strength_spy_20d",
    "relative_strength_spy_60d",
    "correlation_spy_60d",
    "beta_spy_60d",
]


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

def build_feature_configurations():

    configurations = {
        "full": NUMERIC_FEATURES.copy(),
        "reduced": REDUCED_FEATURES.copy(),
    }

    # Leave-one-group-out experiments.
    # Each configuration retains every feature except
    # the selected family.

    for group_name, group_features in FEATURE_GROUPS.items():

        configurations[f"without_{group_name}"] = [
            feature
            for feature in NUMERIC_FEATURES
            if feature not in group_features
        ]

    return configurations


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    train_path = SPLIT_DIR / "train.csv"
    validation_path = SPLIT_DIR / "validation.csv"

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    if not validation_path.exists():
        raise FileNotFoundError(validation_path)

    train = pd.read_csv(
        train_path,
        parse_dates=["Date"],
    )

    validation = pd.read_csv(
        validation_path,
        parse_dates=["Date"],
    )

    return train, validation


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data(train, validation):

    required_columns = (
        ["Ticker"]
        + NUMERIC_FEATURES
    )

    for dataset_name, dataset in [
        ("Training", train),
        ("Validation", validation),
    ]:

        missing = [
            column
            for column in required_columns
            if column not in dataset.columns
        ]

        if missing:
            raise ValueError(
                f"{dataset_name} missing columns: {missing}"
            )

        if dataset[required_columns].isna().any().any():
            raise ValueError(
                f"{dataset_name} contains missing features."
            )

        values = dataset[
            NUMERIC_FEATURES
        ].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{dataset_name} contains infinite values."
            )

    print("\nData validation completed.")

    print(
        f"Training observations: {len(train):,}"
    )

    print(
        f"Validation observations: {len(validation):,}"
    )

    print(
        f"Training instruments: {train['Ticker'].nunique()}"
    )

    print(
        f"Validation instruments: {validation['Ticker'].nunique()}"
    )


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_preprocessor(features):

    return ColumnTransformer(
        transformers=[
            (
                "ticker",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
                ["Ticker"],
            ),
            (
                "numeric",
                "passthrough",
                features,
            ),
        ],
        remainder="drop",
    )


def build_classifier(features):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(features),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_features="sqrt",
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    class_weight=None,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    )


def build_regressor(features):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(features),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_features=1.0,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    )


# ============================================================
# CLASSIFICATION EVALUATION
# ============================================================

def evaluate_classification(
    y_true,
    predictions,
    probabilities,
):

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),

        "predicted_up_rate": np.mean(
            predictions
        ),

        "actual_up_rate": np.mean(
            y_true
        ),
    }


# ============================================================
# REGRESSION EVALUATION
# ============================================================

def evaluate_regression(
    y_true,
    predictions,
):

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

        "prediction_mean": np.mean(
            predictions
        ),

        "actual_mean": np.mean(
            y_true
        ),
    }


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_experiment(
    configuration_name,
    features,
    horizon,
    train,
    validation,
):

    X_train = train[
        ["Ticker"] + features
    ]

    X_validation = validation[
        ["Ticker"] + features
    ]

    y_train_class = train[
        f"target_direction_{horizon}d"
    ].astype(int)

    y_validation_class = validation[
        f"target_direction_{horizon}d"
    ].astype(int)

    y_train_reg = train[
        f"target_return_{horizon}d"
    ].astype(float)

    y_validation_reg = validation[
        f"target_return_{horizon}d"
    ].astype(float)

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    classifier = build_classifier(features)

    classifier.fit(
        X_train,
        y_train_class,
    )

    class_predictions = classifier.predict(
        X_validation
    )

    class_probabilities = classifier.predict_proba(
        X_validation
    )[:, 1]

    class_metrics = evaluate_classification(
        y_validation_class,
        class_predictions,
        class_probabilities,
    )

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    regressor = build_regressor(features)

    regressor.fit(
        X_train,
        y_train_reg,
    )

    reg_predictions = regressor.predict(
        X_validation
    )

    reg_metrics = evaluate_regression(
        y_validation_reg,
        reg_predictions,
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    common = {
        "configuration": configuration_name,
        "horizon": horizon,
        "n_features": len(features),
    }

    results = [
        {
            **common,
            "task": "classification",
            **class_metrics,
        },
        {
            **common,
            "task": "regression",
            **reg_metrics,
        },
    ]

    return results


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("FEATURE ABLATION EXPERIMENT")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train, validation = load_data()

    validate_data(
        train,
        validation,
    )

    configurations = build_feature_configurations()

    print("\n" + "=" * 70)
    print("FEATURE CONFIGURATIONS")
    print("=" * 70)

    for name, features in configurations.items():

        print(
            f"{name:30s} {len(features):2d} features"
        )

    total_models = (
        len(configurations)
        * len(HORIZONS)
        * 2
    )

    print(
        f"\nTotal models to train: {total_models}"
    )

    print(
        f"Trees per model: {N_ESTIMATORS}"
    )

    print(
        "\nThe test dataset will not be loaded."
    )

    all_results = []

    # --------------------------------------------------------
    # Run all configurations
    # --------------------------------------------------------

    for configuration_name, features in configurations.items():

        print("\n" + "=" * 70)

        print(
            f"CONFIGURATION: {configuration_name}"
        )

        print(
            f"Number of features: {len(features)}"
        )

        print("=" * 70)

        for horizon in HORIZONS:

            print(
                f"\nRunning horizon: {horizon} days"
            )

            results = run_experiment(
                configuration_name=configuration_name,
                features=features,
                horizon=horizon,
                train=train,
                validation=validation,
            )

            all_results.extend(results)

            for result in results:

                print(
                    f"{result['task']:15s} "
                    f"completed"
                )

                if result["task"] == "classification":

                    print(
                        f"  ROC-AUC: "
                        f"{result['roc_auc']:.4f}"
                    )

                    print(
                        f"  Balanced Accuracy: "
                        f"{result['balanced_accuracy']:.4f}"
                    )

                else:

                    print(
                        f"  MAE: "
                        f"{result['mae']:.5f}"
                    )

                    print(
                        f"  R²: "
                        f"{result['r2']:.4f}"
                    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        OUTPUT_DIR
        / "feature_ablation_results.csv"
    )

    features_path = (
        OUTPUT_DIR
        / "feature_ablation_feature_sets.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # Save the exact feature composition
    # of every experiment for reproducibility.

    feature_records = []

    for configuration_name, features in configurations.items():

        for feature in features:

            feature_records.append(
                {
                    "configuration": configuration_name,
                    "feature": feature,
                }
            )

    feature_sets_df = pd.DataFrame(
        feature_records
    )

    feature_sets_df.to_csv(
        features_path,
        index=False,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE ABLATION COMPLETED")
    print("=" * 70)

    for task in [
        "classification",
        "regression",
    ]:

        for horizon in HORIZONS:

            subset = results_df[
                (results_df["task"] == task)
                & (results_df["horizon"] == horizon)
            ].copy()

            print(
                f"\n=== {task.upper()} - "
                f"{horizon} DAYS ==="
            )

            if task == "classification":

                subset = subset.sort_values(
                    "roc_auc",
                    ascending=False,
                )

                columns = [
                    "configuration",
                    "n_features",
                    "roc_auc",
                    "balanced_accuracy",
                    "brier_score",
                    "predicted_up_rate",
                ]

            else:

                subset = subset.sort_values(
                    "mae",
                    ascending=True,
                )

                columns = [
                    "configuration",
                    "n_features",
                    "mae",
                    "rmse",
                    "r2",
                ]

            print(
                subset[columns].to_string(
                    index=False
                )
            )

    print("\nFiles saved:")

    print(results_path)
    print(features_path)

    print(
        "\nIMPORTANT: The test dataset was never loaded."
    )


if __name__ == "__main__":
    main()
