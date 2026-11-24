
"""Temporal permutation importance across expanding CV folds.

Uses only the training partition.
Never loads validation.csv or test.csv.

Each feature is circularly shifted across dates while preserving
the cross-sectional values of all instruments belonging to a date.

Classification importance:
    Baseline ROC-AUC - Permuted ROC-AUC

Regression importance:
    Permuted MAE - Baseline MAE

Positive importance means that permutation worsened performance.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
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

N_REPEATS = 3
RANDOM_STATE = 42
N_JOBS = 1


def load_training_data() -> pd.DataFrame:
    """Load and validate only the training partition."""

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
            f"target_direction_{h}d"
            for h in HORIZONS
        ]
        + [
            f"target_return_{h}d"
            for h in HORIZONS
        ]
    )

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
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
    print(f"Dates: {data['Date'].nunique()}")
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
                f"Insufficient training dates in fold {fold_number}."
            )

        train_dates = dates[
            train_indices[:-PURGE_DAYS]
        ]

        validation_dates = dates[
            validation_indices
        ]

        train_fold = data[
            data["Date"].isin(train_dates)
        ].copy()

        validation_fold = data[
            data["Date"].isin(validation_dates)
        ].copy()

        if train_fold.empty or validation_fold.empty:
            raise ValueError(
                f"Empty dataset in fold {fold_number}."
            )

        if train_fold["Date"].max() >= validation_fold["Date"].min():
            raise ValueError(
                f"Chronological separation failed in fold {fold_number}."
            )

        folds.append(
            (train_fold, validation_fold)
        )

        print(
            f"Fold {fold_number}: "
            f"{len(train_fold):,} training rows | "
            f"{len(validation_fold):,} validation rows | "
            f"purge {PURGE_DAYS} dates"
        )

    return folds


def validate_cross_sectional_structure(
    data: pd.DataFrame,
) -> tuple[list, list]:
    """Ensure every date contains the same instruments."""

    tickers = sorted(
        data["Ticker"].unique().tolist()
    )

    dates = sorted(
        data["Date"].unique().tolist()
    )

    counts = data.groupby("Date")["Ticker"].nunique()

    if not (counts == len(tickers)).all():
        raise ValueError(
            "Incomplete instrument coverage detected."
        )

    expected_rows = len(dates) * len(tickers)

    if len(data) != expected_rows:
        raise ValueError(
            "Unexpected number of rows for date-instrument grid."
        )

    return dates, tickers


def circular_date_shift(
    data: pd.DataFrame,
    feature: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shift one feature across dates, preserving cross-sectional groups."""

    dates, tickers = validate_cross_sectional_structure(
        data
    )

    n_dates = len(dates)
    n_tickers = len(tickers)

    if n_dates < 2:
        raise ValueError(
            "At least two dates are required for permutation."
        )

    ordered = data.sort_values(
        ["Date", "Ticker"]
    )

    values = ordered[feature].to_numpy(
        dtype=float
    )

    matrix = values.reshape(
        n_dates,
        n_tickers,
    )

    # A non-zero shift avoids retaining the original date alignment.
    shift = int(
        rng.integers(1, n_dates)
    )

    shifted_matrix = np.roll(
        matrix,
        shift=shift,
        axis=0,
    )

    return shifted_matrix.reshape(-1)


def classification_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict:
    """Calculate classification metrics from probabilities."""

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "roc_auc": (
            roc_auc_score(y_true, probabilities)
            if y_true.nunique() == 2
            else np.nan
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
    }


def regression_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict:
    """Calculate regression metrics."""

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
    }


def append_importance_results(
    rows: list,
    model_type: str,
    horizon: int,
    fold_number: int,
    feature: str,
    repeat: int,
    baseline_metrics: dict,
    permuted_metrics: dict,
) -> None:
    """Save metric changes caused by permutation."""

    if model_type == "classification":
        importance_definitions = {
            "roc_auc_drop": (
                baseline_metrics["roc_auc"]
                - permuted_metrics["roc_auc"]
            ),
            "balanced_accuracy_drop": (
                baseline_metrics["balanced_accuracy"]
                - permuted_metrics["balanced_accuracy"]
            ),
            "brier_increase": (
                permuted_metrics["brier_score"]
                - baseline_metrics["brier_score"]
            ),
        }
    else:
        importance_definitions = {
            "mae_increase": (
                permuted_metrics["mae"]
                - baseline_metrics["mae"]
            ),
            "rmse_increase": (
                permuted_metrics["rmse"]
                - baseline_metrics["rmse"]
            ),
            "r2_drop": (
                baseline_metrics["r2"]
                - permuted_metrics["r2"]
            ),
        }

    for metric_name, importance in importance_definitions.items():
        rows.append(
            {
                "model_type": model_type,
                "horizon": horizon,
                "fold": fold_number,
                "feature": feature,
                "repeat": repeat,
                "metric": metric_name,
                "baseline_value": baseline_metrics[
                    metric_name.replace("_drop", "")
                    .replace("_increase", "")
                ] if metric_name in ("roc_auc_drop", "mae_increase") else np.nan,
                "permutation_importance": importance,
            }
        )


def run_classification_permutation(
    train_fold: pd.DataFrame,
    validation_fold: pd.DataFrame,
    horizon: int,
    fold_number: int,
    rows: list,
) -> None:
    """Train classifier and measure temporal permutation importance."""

    target = f"target_direction_{horizon}d"

    X_train = train_fold[NUMERIC_FEATURES]
    X_validation = validation_fold[NUMERIC_FEATURES]

    y_train = train_fold[target].astype(int)
    y_validation = validation_fold[target].astype(int)

    if y_train.nunique() != 2:
        raise ValueError(
            f"Fold {fold_number}: classification training "
            "target contains only one class."
        )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    model.fit(
        X_train,
        y_train,
    )

    baseline_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    baseline_metrics = classification_metrics(
        y_validation,
        baseline_probabilities,
    )

    print(
        f"\nClassification baseline | "
        f"{horizon}d | Fold {fold_number} | "
        f"AUC: {baseline_metrics['roc_auc']:.4f}"
    )

    rng = np.random.default_rng(
        RANDOM_STATE + horizon * 100 + fold_number
    )

    for feature_number, feature in enumerate(
        NUMERIC_FEATURES,
        start=1,
    ):
        for repeat in range(1, N_REPEATS + 1):
            X_permuted = X_validation.copy()

            X_permuted[feature] = circular_date_shift(
                validation_fold,
                feature,
                rng,
            )

            permuted_probabilities = model.predict_proba(
                X_permuted
            )[:, 1]

            permuted_metrics = classification_metrics(
                y_validation,
                permuted_probabilities,
            )

            append_importance_results(
                rows,
                "classification",
                horizon,
                fold_number,
                feature,
                repeat,
                baseline_metrics,
                permuted_metrics,
            )

        print(
            f"Classification {horizon}d | Fold {fold_number} | "
            f"Feature {feature_number}/{len(NUMERIC_FEATURES)}: "
            f"{feature}"
        )


def run_regression_permutation(
    train_fold: pd.DataFrame,
    validation_fold: pd.DataFrame,
    horizon: int,
    fold_number: int,
    rows: list,
) -> None:
    """Train regressor and measure temporal permutation importance."""

    target = f"target_return_{horizon}d"

    X_train = train_fold[NUMERIC_FEATURES]
    X_validation = validation_fold[NUMERIC_FEATURES]

    y_train = train_fold[target]
    y_validation = validation_fold[target]

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    model.fit(
        X_train,
        y_train,
    )

    baseline_predictions = model.predict(
        X_validation
    )

    baseline_metrics = regression_metrics(
        y_validation,
        baseline_predictions,
    )

    print(
        f"\nRegression baseline | "
        f"{horizon}d | Fold {fold_number} | "
        f"MAE: {baseline_metrics['mae']:.5f}"
    )

    rng = np.random.default_rng(
        RANDOM_STATE + horizon * 1000 + fold_number
    )

    for feature_number, feature in enumerate(
        NUMERIC_FEATURES,
        start=1,
    ):
        for repeat in range(1, N_REPEATS + 1):
            X_permuted = X_validation.copy()

            X_permuted[feature] = circular_date_shift(
                validation_fold,
                feature,
                rng,
            )

            permuted_predictions = model.predict(
                X_permuted
            )

            permuted_metrics = regression_metrics(
                y_validation,
                permuted_predictions,
            )

            append_importance_results(
                rows,
                "regression",
                horizon,
                fold_number,
                feature,
                repeat,
                baseline_metrics,
                permuted_metrics,
            )

        print(
            f"Regression {horizon}d | Fold {fold_number} | "
            f"Feature {feature_number}/{len(NUMERIC_FEATURES)}: "
            f"{feature}"
        )


def summarize_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate permutation importance across repeats and folds."""

    summary = (
        results
        .groupby(
            [
                "model_type",
                "horizon",
                "metric",
                "feature",
            ]
        )["permutation_importance"]
        .agg(
            mean_importance="mean",
            std_importance="std",
            min_importance="min",
            max_importance="max",
        )
        .reset_index()
    )

    summary["positive_frequency"] = (
        results
        .assign(
            positive=lambda frame: (
                frame["permutation_importance"] > 0
            )
        )
        .groupby(
            [
                "model_type",
                "horizon",
                "metric",
                "feature",
            ]
        )["positive"]
        .mean()
        .to_numpy()
    )

    summary["importance_rank"] = (
        summary
        .groupby(
            ["model_type", "horizon", "metric"]
        )["mean_importance"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    return summary.sort_values(
        [
            "model_type",
            "horizon",
            "metric",
            "importance_rank",
        ]
    ).reset_index(drop=True)


def save_results(
    results: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Save detailed and aggregated reports."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        OUTPUT_DIR
        / "temporal_permutation_cv_results.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "temporal_permutation_cv_summary.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\nSAVED REPORTS")
    print("=" * 65)
    print(results_path)
    print(summary_path)


def print_summary(
    summary: pd.DataFrame,
) -> None:
    """Print the leading features for each model and horizon."""

    print("\nTOP 15 PERMUTATION FEATURES")
    print("=" * 65)

    main_metrics = {
        "classification": "roc_auc_drop",
        "regression": "mae_increase",
    }

    for model_type, metric in main_metrics.items():
        for horizon in HORIZONS:
            subset = summary[
                (summary["model_type"] == model_type)
                & (summary["horizon"] == horizon)
                & (summary["metric"] == metric)
            ].head(15)

            print(
                f"\n{model_type.upper()} | "
                f"{horizon}-DAY | {metric}"
            )

            print(
                subset[
                    [
                        "feature",
                        "mean_importance",
                        "std_importance",
                        "positive_frequency",
                        "importance_rank",
                    ]
                ].round(5).to_string(index=False)
            )


def main() -> None:
    """Execute temporal permutation importance across all folds."""

    print("\nTEMPORAL PERMUTATION IMPORTANCE")
    print("=" * 65)
    print("Data source: TRAIN ONLY")
    print(f"Temporal folds: {N_SPLITS}")
    print(f"Purge gap: {PURGE_DAYS} trading days")
    print(f"Features: {len(NUMERIC_FEATURES)}")
    print(f"Horizons: {HORIZONS}")
    print(f"Permutation repeats: {N_REPEATS}")
    print("Permutation method: Circular shift across dates")
    print("Cross-sectional instrument groups: Preserved")

    data = load_training_data()

    print("\nCREATING TEMPORAL FOLDS")
    print("=" * 65)

    folds = create_temporal_folds(data)

    rows = []

    total_experiments = (
        len(folds)
        * len(HORIZONS)
        * 2
    )

    completed = 0

    for horizon in HORIZONS:
        print(f"\nHORIZON: {horizon} DAYS")
        print("=" * 65)

        for fold_number, (train_fold, validation_fold) in enumerate(
            folds,
            start=1,
        ):
            print(
                f"\nEXPERIMENT {completed + 1}/{total_experiments}"
            )

            # Validate the date/ticker grid before permutation.
            validate_cross_sectional_structure(
                validation_fold
            )

            run_classification_permutation(
                train_fold,
                validation_fold,
                horizon,
                fold_number,
                rows,
            )

            run_regression_permutation(
                train_fold,
                validation_fold,
                horizon,
                fold_number,
                rows,
            )

            completed += 1

    results = pd.DataFrame(rows)

    summary = summarize_results(
        results
    )

    save_results(
        results,
        summary,
    )

    print_summary(
        summary
    )

    print("\nTEMPORAL PERMUTATION ANALYSIS COMPLETED")
    print("Only training data and internal temporal folds were used.")
    print("Validation and test partitions were not loaded.")


if __name__ == "__main__":
    main()
