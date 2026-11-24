
"""Feature optimization through purged temporal cross-validation.

Uses only the training partition.
Validation and test partitions are never loaded.

Evaluates Random Forest classification and regression models
across expanding chronological folds, with a 20-date purge gap.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_asset"
    / "splits"
    / "train.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_asset"
    / "models"
)

NUMERIC_FEATURES = [
    # Momentum
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",

    # Relative trend
    "price_to_sma_10",
    "price_to_sma_20",
    "price_to_sma_50",
    "price_to_sma_100",
    "price_to_sma_200",
    "price_to_ema_12",
    "price_to_ema_26",
    "sma_20_to_sma_50",
    "sma_50_to_sma_200",

    # Volatility
    "volatility_10d",
    "volatility_20d",
    "volatility_60d",
    "volatility_ratio_10_60",
    "atr_pct",

    # Oscillators
    "rsi_14",
    "macd_pct",
    "macd_signal_pct",
    "macd_hist_pct",

    # Volume
    "volume_ratio_20",
    "obv_change_20",

    # Market context
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

    # Relative strength and market sensitivity
    "relative_strength_spy_5d",
    "relative_strength_spy_20d",
    "relative_strength_spy_60d",
    "correlation_spy_60d",
    "beta_spy_60d",
]

HORIZONS = (5, 10, 20)

N_SPLITS = 5
PURGE_DAYS = 20

N_ESTIMATORS = 200
MIN_SAMPLES_LEAF = 10
RANDOM_STATE = 42
N_JOBS = 1


def load_training_data() -> pd.DataFrame:
    """Load only the training partition and validate its columns."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    data = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["Date"],
    )

    required_columns = (
        ["Date", "Ticker"]
        + NUMERIC_FEATURES
        + [
            f"target_direction_{horizon}d"
            for horizon in HORIZONS
        ]
        + [
            f"target_return_{horizon}d"
            for horizon in HORIZONS
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    if data.duplicated(["Date", "Ticker"]).any():
        raise ValueError(
            "Duplicate Date-Ticker combinations detected."
        )

    if data[NUMERIC_FEATURES].isna().any().any():
        raise ValueError(
            "Missing values detected in model features."
        )

    if not np.isfinite(
        data[NUMERIC_FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite values detected in model features."
        )

    data = data.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    print("\nTRAINING DATA")
    print("=" * 65)
    print(f"Observations: {len(data):,}")
    print(f"Instruments: {data['Ticker'].nunique()}")
    print(f"Unique dates: {data['Date'].nunique()}")
    print(f"Features: {len(NUMERIC_FEATURES)}")
    print(
        f"Date range: {data['Date'].min()} "
        f"to {data['Date'].max()}"
    )

    return data


def create_temporal_folds(
    data: pd.DataFrame,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create expanding chronological folds with a purge gap."""

    dates = pd.DatetimeIndex(
        sorted(data["Date"].unique())
    )

    if len(dates) <= N_SPLITS + PURGE_DAYS:
        raise ValueError(
            "Insufficient dates for temporal cross-validation."
        )

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
    )

    folds = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(dates),
        start=1,
    ):
        if len(train_indices) <= PURGE_DAYS:
            raise ValueError(
                f"Fold {fold_number}: insufficient training dates "
                "after applying purge."
            )

        purged_train_indices = train_indices[:-PURGE_DAYS]

        train_dates = dates[purged_train_indices]
        validation_dates = dates[validation_indices]

        train_fold = data[
            data["Date"].isin(train_dates)
        ].copy()

        validation_fold = data[
            data["Date"].isin(validation_dates)
        ].copy()

        if train_fold.empty or validation_fold.empty:
            raise ValueError(
                f"Fold {fold_number} is empty."
            )

        if train_fold["Date"].max() >= validation_fold["Date"].min():
            raise ValueError(
                f"Fold {fold_number}: chronological separation failed."
            )

        folds.append(
            (train_fold, validation_fold)
        )

        print(
            f"Fold {fold_number}: "
            f"train {train_fold['Date'].min().date()} "
            f"to {train_fold['Date'].max().date()} | "
            f"validation {validation_fold['Date'].min().date()} "
            f"to {validation_fold['Date'].max().date()} | "
            f"{len(train_fold):,} train rows | "
            f"{len(validation_fold):,} validation rows"
        )

    return folds


def evaluate_classification(
    model: RandomForestClassifier,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict:
    """Calculate classification metrics for one fold."""

    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results = {
        "accuracy": accuracy_score(
            y_validation,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_validation,
            predictions,
        ),
        "brier_score": brier_score_loss(
            y_validation,
            probabilities,
        ),
        "predicted_up_rate": predictions.mean(),
        "actual_up_rate": y_validation.mean(),
    }

    if y_validation.nunique() == 2:
        results["roc_auc"] = roc_auc_score(
            y_validation,
            probabilities,
        )
    else:
        results["roc_auc"] = np.nan

    return results


def evaluate_regression(
    model: RandomForestRegressor,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict:
    """Calculate regression metrics for one fold."""

    predictions = model.predict(X_validation)

    return {
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
        "prediction_mean": predictions.mean(),
        "prediction_std": predictions.std(),
        "actual_mean": y_validation.mean(),
    }


def run_optimization_analysis(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train models and collect fold-level feature importances."""

    importance_rows = []
    metric_rows = []

    total_models = len(folds) * len(HORIZONS) * 2
    completed_models = 0

    for horizon in HORIZONS:
        classification_target = f"target_direction_{horizon}d"
        regression_target = f"target_return_{horizon}d"

        print(f"\nHORIZON: {horizon} DAYS")
        print("=" * 65)

        for fold_number, (train_fold, validation_fold) in enumerate(
            folds,
            start=1,
        ):
            X_train = train_fold[NUMERIC_FEATURES]
            X_validation = validation_fold[NUMERIC_FEATURES]

            # Classification
            y_train_class = train_fold[
                classification_target
            ].astype(int)

            y_validation_class = validation_fold[
                classification_target
            ].astype(int)

            if not set(y_train_class.unique()).issubset({0, 1}):
                raise ValueError(
                    f"Unexpected classification labels in "
                    f"{classification_target}."
                )

            if y_train_class.nunique() != 2:
                raise ValueError(
                    f"Training fold {fold_number} has only one "
                    f"class for {classification_target}."
                )

            classifier = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            )

            classifier.fit(
                X_train,
                y_train_class,
            )

            class_metrics = evaluate_classification(
                classifier,
                X_validation,
                y_validation_class,
            )

            metric_rows.append(
                {
                    "model_type": "classification",
                    "horizon": horizon,
                    "fold": fold_number,
                    **class_metrics,
                }
            )

            for feature, importance in zip(
                NUMERIC_FEATURES,
                classifier.feature_importances_,
            ):
                importance_rows.append(
                    {
                        "model_type": "classification",
                        "horizon": horizon,
                        "fold": fold_number,
                        "feature": feature,
                        "importance": importance,
                    }
                )

            completed_models += 1

            print(
                f"Classification {horizon}d | "
                f"Fold {fold_number} | "
                f"AUC: {class_metrics['roc_auc']:.4f} | "
                f"BA: {class_metrics['balanced_accuracy']:.4f} | "
                f"Model {completed_models}/{total_models}"
            )

            # Regression
            y_train_reg = train_fold[regression_target]
            y_validation_reg = validation_fold[regression_target]

            regressor = RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                max_features=1.0,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            )

            regressor.fit(
                X_train,
                y_train_reg,
            )

            reg_metrics = evaluate_regression(
                regressor,
                X_validation,
                y_validation_reg,
            )

            metric_rows.append(
                {
                    "model_type": "regression",
                    "horizon": horizon,
                    "fold": fold_number,
                    **reg_metrics,
                }
            )

            for feature, importance in zip(
                NUMERIC_FEATURES,
                regressor.feature_importances_,
            ):
                importance_rows.append(
                    {
                        "model_type": "regression",
                        "horizon": horizon,
                        "fold": fold_number,
                        "feature": feature,
                        "importance": importance,
                    }
                )

            completed_models += 1

            print(
                f"Regression {horizon}d | "
                f"Fold {fold_number} | "
                f"MAE: {reg_metrics['mae']:.5f} | "
                f"R2: {reg_metrics['r2']:.4f} | "
                f"Model {completed_models}/{total_models}"
            )

    return (
        pd.DataFrame(importance_rows),
        pd.DataFrame(metric_rows),
    )


def summarize_feature_importance(
    importance_data: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate importance and measure Top 10 stability across folds."""

    # Rank features separately within each fold.
    importance_data = importance_data.copy()

    importance_data["fold_rank"] = (
        importance_data
        .groupby(
            ["model_type", "horizon", "fold"]
        )["importance"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    # Calculate the frequency with which each feature enters Top 10.
    top_10_frequency = (
        importance_data
        .assign(
            in_top_10=lambda frame: frame["fold_rank"] <= 10
        )
        .groupby(
            ["model_type", "horizon", "feature"]
        )["in_top_10"]
        .mean()
        .rename("top_10_frequency")
        .reset_index()
    )

    # Aggregate importance across folds.
    summary = (
        importance_data
        .groupby(
            ["model_type", "horizon", "feature"]
        )["importance"]
        .agg(
            mean_importance="mean",
            std_importance="std",
            min_importance="min",
            max_importance="max",
        )
        .reset_index()
    )

    summary["coefficient_of_variation"] = (
        summary["std_importance"]
        / summary["mean_importance"].replace(0, np.nan)
    )

    summary["importance_rank"] = (
        summary
        .groupby(
            ["model_type", "horizon"]
        )["mean_importance"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    summary = summary.merge(
        top_10_frequency,
        on=["model_type", "horizon", "feature"],
        how="left",
    )

    return summary.sort_values(
        ["model_type", "horizon", "importance_rank"]
    ).reset_index(drop=True)


def save_results(
    importance_data: pd.DataFrame,
    importance_summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """Save all optimization reports."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_path = (
        OUTPUT_DIR / "feature_optimization_fold_importance.csv"
    )

    summary_path = (
        OUTPUT_DIR / "feature_optimization_summary.csv"
    )

    metrics_path = (
        OUTPUT_DIR / "feature_optimization_cv_metrics.csv"
    )

    importance_data.to_csv(
        importance_path,
        index=False,
    )

    importance_summary.to_csv(
        summary_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    print("\nSAVED REPORTS")
    print("=" * 65)
    print(importance_path)
    print(summary_path)
    print(metrics_path)


def print_summary(
    importance_summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """Print compact summaries of CV performance and feature rankings."""

    print("\nCROSS-VALIDATION PERFORMANCE")
    print("=" * 65)

    metric_columns = {
        "classification": [
            "roc_auc",
            "balanced_accuracy",
            "brier_score",
            "predicted_up_rate",
            "actual_up_rate",
        ],
        "regression": [
            "mae",
            "rmse",
            "r2",
        ],
    }

    for model_type, columns in metric_columns.items():
        print(f"\n{model_type.upper()}")

        available_columns = [
            column
            for column in columns
            if column in metrics.columns
        ]

        summary = (
            metrics[metrics["model_type"] == model_type]
            .groupby("horizon")[available_columns]
            .mean()
            .round(4)
        )

        print(summary.to_string())

    print("\nTOP 10 FEATURES BY MEAN IMPORTANCE")
    print("=" * 65)

    for model_type in ("classification", "regression"):
        for horizon in HORIZONS:
            subset = importance_summary[
                (importance_summary["model_type"] == model_type)
                & (importance_summary["horizon"] == horizon)
            ].head(10)

            print(
                f"\n{model_type.upper()} | {horizon}-DAY HORIZON"
            )

            print(
                subset[
                    [
                        "feature",
                        "mean_importance",
                        "std_importance",
                        "top_10_frequency",
                        "importance_rank",
                    ]
                ].round(5).to_string(index=False)
            )


def main() -> None:
    """Run the complete feature optimization analysis."""

    print("\nFEATURE OPTIMIZATION")
    print("=" * 65)
    print("Data source: TRAIN ONLY")
    print(f"Temporal folds: {N_SPLITS}")
    print(f"Purge gap: {PURGE_DAYS} trading days")
    print(f"Features: {len(NUMERIC_FEATURES)}")
    print(f"Horizons: {HORIZONS}")
    print(
        "Models: Random Forest Classifier and "
        "Random Forest Regressor"
    )

    data = load_training_data()

    print("\nCREATING TEMPORAL FOLDS")
    print("=" * 65)

    folds = create_temporal_folds(data)

    importance_data, metrics = run_optimization_analysis(
        folds
    )

    importance_summary = summarize_feature_importance(
        importance_data
    )

    save_results(
        importance_data,
        importance_summary,
        metrics,
    )

    print_summary(
        importance_summary,
        metrics,
    )

    print("\nFEATURE OPTIMIZATION ANALYSIS COMPLETED")
    print("Validation and test datasets were not loaded.")


if __name__ == "__main__":
    main()
