
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
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


NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = ["Ticker"]


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load training and validation data. Test is never loaded."""

    train_path = SPLIT_DIR / "train.csv"
    validation_path = SPLIT_DIR / "validation.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation file not found: {validation_path}"
        )

    train = pd.read_csv(train_path, parse_dates=["Date"])
    validation = pd.read_csv(
        validation_path,
        parse_dates=["Date"],
    )

    return train, validation


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_features(train, validation):
    """Prepare numerical and categorical model inputs."""

    required_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    missing_train = [
        column
        for column in required_columns
        if column not in train.columns
    ]

    missing_validation = [
        column
        for column in required_columns
        if column not in validation.columns
    ]

    if missing_train:
        raise ValueError(
            f"Missing training features: {missing_train}"
        )

    if missing_validation:
        raise ValueError(
            f"Missing validation features: {missing_validation}"
        )

    X_train = train[required_columns].copy()
    X_validation = validation[required_columns].copy()

    if X_train.isna().any().any():
        raise ValueError("Missing values found in training features.")

    if X_validation.isna().any().any():
        raise ValueError("Missing values found in validation features.")

    if not np.isfinite(
        X_train[NUMERIC_FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError("Infinite values found in training features.")

    if not np.isfinite(
        X_validation[NUMERIC_FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite values found in validation features."
        )

    return X_train, X_validation


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor():
    """
    Encode ticker as a categorical variable.

    Numeric features are passed through unchanged because
    Random Forest does not require feature standardization.
    """

    return ColumnTransformer(
        transformers=[
            (
                "ticker",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def evaluate_classification(
    y_true,
    y_pred,
    y_probability,
    horizon,
):
    """Calculate classification metrics."""

    return {
        "task": "classification",
        "horizon": horizon,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
        "brier_score": brier_score_loss(
            y_true,
            y_probability,
        ),
        "predicted_up_rate": np.mean(y_pred),
        "actual_up_rate": np.mean(y_true),
    }


def train_classification_model(
    X_train,
    X_validation,
    y_train,
    y_validation,
    horizon,
):
    """Train and evaluate a Random Forest classifier."""

    preprocessor = build_preprocessor()

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_validation)

    probabilities = pipeline.predict_proba(
        X_validation
    )[:, 1]

    metrics = evaluate_classification(
        y_validation,
        predictions,
        probabilities,
        horizon,
    )

    return pipeline, metrics, predictions, probabilities


# ============================================================
# REGRESSION
# ============================================================

def train_regression_model(
    X_train,
    X_validation,
    y_train,
    y_validation,
    horizon,
):
    """Train and evaluate a Random Forest regressor."""

    preprocessor = build_preprocessor()

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_features=1.0,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_validation)

    metrics = {
        "task": "regression",
        "horizon": horizon,
        "mae": mean_absolute_error(
            y_validation,
            predictions,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_validation,
                predictions,
            )
        ),
        "r2": r2_score(
            y_validation,
            predictions,
        ),
        "prediction_mean": np.mean(predictions),
        "prediction_std": np.std(predictions),
        "actual_mean": np.mean(y_validation),
    }

    return pipeline, metrics, predictions


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def extract_feature_importance(pipeline, task, horizon):
    """Extract feature importance from the fitted Random Forest."""

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
            "task": task,
            "horizon": horizon,
        }
    )

    return importance.sort_values(
        "importance",
        ascending=False,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("RANDOM FOREST - MULTI-ASSET EXPERIMENT")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train, validation = load_data()

    print(f"\nTraining observations: {len(train):,}")
    print(f"Validation observations: {len(validation):,}")

    print(
        f"Training date range: "
        f"{train['Date'].min().date()} - "
        f"{train['Date'].max().date()}"
    )

    print(
        f"Validation date range: "
        f"{validation['Date'].min().date()} - "
        f"{validation['Date'].max().date()}"
    )

    print(
        f"\nTraining instruments: "
        f"{train['Ticker'].nunique()}"
    )

    print(
        f"Validation instruments: "
        f"{validation['Ticker'].nunique()}"
    )

    X_train, X_validation = prepare_features(
        train,
        validation,
    )

    print(
        f"\nNumerical features: {len(NUMERIC_FEATURES)}"
    )

    print(
        f"Categorical features: {len(CATEGORICAL_FEATURES)}"
    )

    print(
        f"Random Forest trees: {N_ESTIMATORS}"
    )

    print(
        f"Minimum samples per leaf: {MIN_SAMPLES_LEAF}"
    )

    all_metrics = []
    all_predictions = []
    all_importances = []

    # --------------------------------------------------------
    # Run each prediction horizon
    # --------------------------------------------------------

    for horizon in HORIZONS:

        print("\n" + "=" * 70)
        print(f"HORIZON: {horizon} DAYS")
        print("=" * 70)

        classification_target = (
            f"target_direction_{horizon}d"
        )

        regression_target = (
            f"target_return_{horizon}d"
        )

        for target in [
            classification_target,
            regression_target,
        ]:
            if target not in train.columns:
                raise ValueError(
                    f"Target not found in training data: {target}"
                )

            if target not in validation.columns:
                raise ValueError(
                    f"Target not found in validation data: {target}"
                )

        y_train_class = train[
            classification_target
        ].astype(int)

        y_validation_class = validation[
            classification_target
        ].astype(int)

        y_train_reg = train[
            regression_target
        ].astype(float)

        y_validation_reg = validation[
            regression_target
        ].astype(float)

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        print("\nTraining classification model...")

        classifier, class_metrics, class_predictions, probabilities = (
            train_classification_model(
                X_train,
                X_validation,
                y_train_class,
                y_validation_class,
                horizon,
            )
        )

        all_metrics.append(class_metrics)

        class_importance = extract_feature_importance(
            classifier,
            "classification",
            horizon,
        )

        all_importances.append(class_importance)

        print("\nClassification results:")

        for metric, value in class_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                print(f"{metric}: {value:.4f}")

        # ----------------------------------------------------
        # Regression
        # ----------------------------------------------------

        print("\nTraining regression model...")

        regressor, reg_metrics, reg_predictions = (
            train_regression_model(
                X_train,
                X_validation,
                y_train_reg,
                y_validation_reg,
                horizon,
            )
        )

        all_metrics.append(reg_metrics)

        reg_importance = extract_feature_importance(
            regressor,
            "regression",
            horizon,
        )

        all_importances.append(reg_importance)

        print("\nRegression results:")

        for metric, value in reg_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                print(f"{metric}: {value:.4f}")

        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        predictions_df = validation[
            [
                "Date",
                "Ticker",
                "AssetName",
                "AssetClass",
            ]
        ].copy()

        predictions_df["horizon"] = horizon

        predictions_df["actual_direction"] = (
            y_validation_class.to_numpy()
        )

        predictions_df["predicted_direction"] = (
            class_predictions
        )

        predictions_df["probability_up"] = (
            probabilities
        )

        predictions_df["actual_return"] = (
            y_validation_reg.to_numpy()
        )

        predictions_df["predicted_return"] = (
            reg_predictions
        )

        all_predictions.append(predictions_df)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_metrics)

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    importance_df = pd.concat(
        all_importances,
        ignore_index=True,
    )

    results_path = (
        OUTPUT_DIR
        / "random_forest_multi_asset_results.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / "random_forest_multi_asset_predictions.csv"
    )

    importance_path = (
        OUTPUT_DIR
        / "random_forest_multi_asset_feature_importance.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    print("\nResults:")
    print(results_df.to_string(index=False))

    print("\nFiles saved:")

    print(results_path)
    print(predictions_path)
    print(importance_path)

    print(
        "\nIMPORTANT: The test dataset was not loaded or used."
    )


if __name__ == "__main__":
    main()
