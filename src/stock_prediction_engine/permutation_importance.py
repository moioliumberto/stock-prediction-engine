
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
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

N_ESTIMATORS = 300
MIN_SAMPLES_LEAF = 10

N_REPEATS = 3
RANDOM_STATE = 42


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
    """
    Load training and validation datasets.

    The test dataset is deliberately never loaded.
    """

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
# FEATURE PREPARATION
# ============================================================

def prepare_features(train, validation):

    required_features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    for column in required_features:

        if column not in train.columns:
            raise ValueError(
                f"Missing training feature: {column}"
            )

        if column not in validation.columns:
            raise ValueError(
                f"Missing validation feature: {column}"
            )

    X_train = train[required_features].copy()

    X_validation = validation[
        required_features
    ].copy()

    if X_train.isna().any().any():
        raise ValueError(
            "Missing values in training features."
        )

    if X_validation.isna().any().any():
        raise ValueError(
            "Missing values in validation features."
        )

    if not np.isfinite(
        X_train[NUMERIC_FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite values in training features."
        )

    if not np.isfinite(
        X_validation[NUMERIC_FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite values in validation features."
        )

    return X_train, X_validation


# ============================================================
# PREPROCESSING
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
# MODEL BUILDERS
# ============================================================

def build_classifier():

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_features="sqrt",
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    class_weight=None,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_regressor():

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_features=1.0,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ============================================================
# TEMPORAL CIRCULAR-SHIFT PERMUTATION
# ============================================================

def prepare_temporal_index(validation):

    """
    Build a date-based index.

    Every date must contain the same set of instruments.
    """

    dates = pd.Index(
        sorted(validation["Date"].unique())
    )

    tickers = set(
        validation["Ticker"].unique()
    )

    expected_rows = len(dates) * len(tickers)

    if len(validation) != expected_rows:
        raise ValueError(
            "Validation dataset does not contain "
            "a complete date/ticker grid."
        )

    if validation.duplicated(
        subset=["Date", "Ticker"]
    ).any():
        raise ValueError(
            "Duplicate Date-Ticker combinations found."
        )

    date_positions = {
        date: position
        for position, date in enumerate(dates)
    }

    return dates, date_positions


def circular_shift_feature(
    X_validation,
    validation,
    feature,
    offset,
    dates,
    date_positions,
):
    """
    Circularly shift one feature across dates.

    All instruments move together by the same date offset.

    This preserves:
    - The feature distribution.
    - The temporal sequence, apart from the circular boundary.
    - The cross-sectional relationship between instruments
      on the shifted dates.

    It breaks the original alignment between the feature
    and the validation targets.
    """

    shifted = X_validation.copy()

    date_indexes = np.array(
        [
            date_positions[date]
            for date in validation["Date"]
        ]
    )

    source_date_indexes = (
        date_indexes + offset
    ) % len(dates)

    source_dates = dates.take(
        source_date_indexes
    )

    original_index = pd.MultiIndex.from_arrays(
        [
            validation["Date"].to_numpy(),
            validation["Ticker"].to_numpy(),
        ],
        names=["Date", "Ticker"],
    )

    source_index = pd.MultiIndex.from_arrays(
        [
            source_dates,
            validation["Ticker"].to_numpy(),
        ],
        names=["Date", "Ticker"],
    )

    lookup = pd.Series(
        X_validation[feature].to_numpy(),
        index=original_index,
    )

    shifted[feature] = (
        lookup.reindex(source_index).to_numpy()
    )

    return shifted


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def calculate_permutation_importance(
    pipeline,
    X_validation,
    validation,
    target,
    task,
    horizon,
    dates,
    date_positions,
    rng,
):
    """
    Measure the performance deterioration caused by
    temporally shifting each feature.

    Classification:
        Score = ROC-AUC

    Regression:
        Score = negative MAE

    Importance = baseline score - permuted score.

    Positive importance means that the permutation
    deteriorated model performance.
    """

    y_true = validation[target].to_numpy()

    # --------------------------------------------------------
    # Baseline performance
    # --------------------------------------------------------

    if task == "classification":

        baseline_probability = pipeline.predict_proba(
            X_validation
        )[:, 1]

        baseline_score = roc_auc_score(
            y_true,
            baseline_probability,
        )

    else:

        baseline_predictions = pipeline.predict(
            X_validation
        )

        baseline_score = -mean_absolute_error(
            y_true,
            baseline_predictions,
        )

    results = []

    # --------------------------------------------------------
    # Evaluate every feature
    # --------------------------------------------------------

    for feature_number, feature in enumerate(
        NUMERIC_FEATURES,
        start=1,
    ):

        permuted_scores = []

        for repeat in range(N_REPEATS):

            # Random non-zero circular shift.
            offset = int(
                rng.integers(
                    1,
                    len(dates),
                )
            )

            X_permuted = circular_shift_feature(
                X_validation=X_validation,
                validation=validation,
                feature=feature,
                offset=offset,
                dates=dates,
                date_positions=date_positions,
            )

            if task == "classification":

                probabilities = pipeline.predict_proba(
                    X_permuted
                )[:, 1]

                score = roc_auc_score(
                    y_true,
                    probabilities,
                )

            else:

                predictions = pipeline.predict(
                    X_permuted
                )

                score = -mean_absolute_error(
                    y_true,
                    predictions,
                )

            permuted_scores.append(score)

        mean_permuted_score = float(
            np.mean(permuted_scores)
        )

        importance = (
            baseline_score
            - mean_permuted_score
        )

        results.append(
            {
                "task": task,
                "horizon": horizon,
                "feature": feature,
                "baseline_score": baseline_score,
                "permuted_score_mean": mean_permuted_score,
                "permuted_score_std": np.std(
                    permuted_scores,
                    ddof=1,
                ),
                "importance": importance,
                "repeats": N_REPEATS,
            }
        )

        print(
            f"  [{feature_number:02d}/"
            f"{len(NUMERIC_FEATURES)}] "
            f"{feature}: "
            f"importance={importance:.6f}"
        )

    return results


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("TEMPORAL PERMUTATION IMPORTANCE")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train, validation = load_data()

    print(
        f"\nTraining observations: {len(train):,}"
    )

    print(
        f"Validation observations: {len(validation):,}"
    )

    print(
        f"Training instruments: "
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

    dates, date_positions = prepare_temporal_index(
        validation
    )

    print(
        f"\nValidation dates: {len(dates):,}"
    )

    print(
        f"Features to evaluate: {len(NUMERIC_FEATURES)}"
    )

    print(
        f"Permutation repetitions: {N_REPEATS}"
    )

    print(
        "\nThe test dataset will not be loaded."
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    all_results = []

    # --------------------------------------------------------
    # Classification and regression, all horizons
    # --------------------------------------------------------

    for horizon in HORIZONS:

        print("\n" + "=" * 70)
        print(
            f"HORIZON: {horizon} DAYS"
        )
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
                    f"Missing target: {target}"
                )

            if target not in validation.columns:
                raise ValueError(
                    f"Missing target: {target}"
                )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        print(
            "\nTraining Random Forest Classifier..."
        )

        classifier = build_classifier()

        classifier.fit(
            X_train,
            train[classification_target].astype(int),
        )

        print(
            "\nClassification permutation importance:"
        )

        classification_results = (
            calculate_permutation_importance(
                pipeline=classifier,
                X_validation=X_validation,
                validation=validation,
                target=classification_target,
                task="classification",
                horizon=horizon,
                dates=dates,
                date_positions=date_positions,
                rng=rng,
            )
        )

        all_results.extend(
            classification_results
        )

        # ----------------------------------------------------
        # Regression
        # ----------------------------------------------------

        print(
            "\nTraining Random Forest Regressor..."
        )

        regressor = build_regressor()

        regressor.fit(
            X_train,
            train[regression_target].astype(float),
        )

        print(
            "\nRegression permutation importance:"
        )

        regression_results = (
            calculate_permutation_importance(
                pipeline=regressor,
                X_validation=X_validation,
                validation=validation,
                target=regression_target,
                task="regression",
                horizon=horizon,
                dates=dates,
                date_positions=date_positions,
                rng=rng,
            )
        )

        all_results.extend(
            regression_results
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df.sort_values(
        [
            "task",
            "horizon",
            "importance",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    output_path = (
        OUTPUT_DIR
        / "permutation_importance_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("PERMUTATION IMPORTANCE COMPLETED")
    print("=" * 70)

    for task in [
        "classification",
        "regression",
    ]:

        for horizon in HORIZONS:

            subset = results_df[
                (results_df["task"] == task)
                & (results_df["horizon"] == horizon)
            ]

            print(
                f"\n=== {task.upper()} - "
                f"{horizon} DAYS ==="
            )

            print(
                subset[
                    [
                        "feature",
                        "importance",
                        "permuted_score_std",
                    ]
                ]
                .head(15)
                .to_string(index=False)
            )

    print(
        f"\nResults saved to:\n{output_path}"
    )

    print(
        "\nIMPORTANT: The test dataset was never loaded."
    )


if __name__ == "__main__":
    main()
