
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

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

CORRELATION_THRESHOLD = 0.85

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


# ============================================================
# DATA LOADING
# ============================================================

def load_training_data():
    """
    Load only the training dataset.

    Validation and test data are not used for feature selection.
    """

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    data = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["Date"],
    )

    return data


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data(data):

    required_columns = (
        ["Date", "Ticker"]
        + NUMERIC_FEATURES
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if data.duplicated(
        subset=["Date", "Ticker"]
    ).any():
        raise ValueError(
            "Duplicate Date-Ticker combinations found."
        )

    if data[NUMERIC_FEATURES].isna().any().any():
        raise ValueError(
            "Missing values found in numerical features."
        )

    values = data[NUMERIC_FEATURES].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Infinite values found in numerical features."
        )

    print("\nData validation completed.")

    print(
        f"Observations: {len(data):,}"
    )

    print(
        f"Instruments: {data['Ticker'].nunique()}"
    )

    print(
        f"Numerical features: {len(NUMERIC_FEATURES)}"
    )

    print(
        f"Date range: {data['Date'].min().date()} - "
        f"{data['Date'].max().date()}"
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def calculate_correlations_by_asset(data):

    """
    Calculate Pearson and Spearman correlations separately
    for each ticker.

    The final results will aggregate the correlations
    across instruments using the median.
    """

    pearson_records = []
    spearman_records = []

    assets = sorted(
        data["Ticker"].unique()
    )

    feature_pairs = list(
        combinations(NUMERIC_FEATURES, 2)
    )

    print(
        f"\nAnalyzing {len(feature_pairs):,} feature pairs "
        f"across {len(assets)} instruments..."
    )

    for ticker in assets:

        print(
            f"Processing {ticker}..."
        )

        asset_data = (
            data[data["Ticker"] == ticker]
            .sort_values("Date")
        )

        pearson_matrix = (
            asset_data[NUMERIC_FEATURES]
            .corr(method="pearson")
        )

        spearman_matrix = (
            asset_data[NUMERIC_FEATURES]
            .corr(method="spearman")
        )

        for feature_a, feature_b in feature_pairs:

            pearson_value = pearson_matrix.loc[
                feature_a,
                feature_b,
            ]

            spearman_value = spearman_matrix.loc[
                feature_a,
                feature_b,
            ]

            pearson_records.append(
                {
                    "ticker": ticker,
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "correlation": pearson_value,
                }
            )

            spearman_records.append(
                {
                    "ticker": ticker,
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "correlation": spearman_value,
                }
            )

    return (
        pd.DataFrame(pearson_records),
        pd.DataFrame(spearman_records),
    )


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_correlations(
    pearson_records,
    spearman_records,
):

    """
    Aggregate correlations across instruments.

    Signed medians show the general direction.
    Absolute medians show the strength of association.
    """

    pearson_summary = (
        pearson_records
        .groupby(
            ["feature_a", "feature_b"]
        )
        .agg(
            pearson_median=(
                "correlation",
                "median",
            ),
            pearson_abs_median=(
                "correlation",
                lambda values: values.abs().median(),
            ),
            pearson_min=(
                "correlation",
                "min",
            ),
            pearson_max=(
                "correlation",
                "max",
            ),
            pearson_assets_above_threshold=(
                "correlation",
                lambda values: (
                    values.abs() >= CORRELATION_THRESHOLD
                ).sum(),
            ),
        )
        .reset_index()
    )

    spearman_summary = (
        spearman_records
        .groupby(
            ["feature_a", "feature_b"]
        )
        .agg(
            spearman_median=(
                "correlation",
                "median",
            ),
            spearman_abs_median=(
                "correlation",
                lambda values: values.abs().median(),
            ),
            spearman_min=(
                "correlation",
                "min",
            ),
            spearman_max=(
                "correlation",
                "max",
            ),
            spearman_assets_above_threshold=(
                "correlation",
                lambda values: (
                    values.abs() >= CORRELATION_THRESHOLD
                ).sum(),
            ),
        )
        .reset_index()
    )

    results = pearson_summary.merge(
        spearman_summary,
        on=[
            "feature_a",
            "feature_b",
        ],
        how="inner",
    )

    results["pearson_redundant"] = (
        results["pearson_abs_median"]
        >= CORRELATION_THRESHOLD
    )

    results["spearman_redundant"] = (
        results["spearman_abs_median"]
        >= CORRELATION_THRESHOLD
    )

    results["potentially_redundant"] = (
        results["pearson_redundant"]
        | results["spearman_redundant"]
    )

    results["max_abs_median_correlation"] = (
        results[
            [
                "pearson_abs_median",
                "spearman_abs_median",
            ]
        ].max(axis=1)
    )

    results = results.sort_values(
        "max_abs_median_correlation",
        ascending=False,
    )

    return results


# ============================================================
# CORRELATION MATRICES
# ============================================================

def build_median_correlation_matrix(
    correlation_records,
):

    """
    Build a symmetric matrix using the median signed
    correlation across instruments.
    """

    matrix = pd.DataFrame(
        np.eye(len(NUMERIC_FEATURES)),
        index=NUMERIC_FEATURES,
        columns=NUMERIC_FEATURES,
    )

    for _, row in correlation_records.iterrows():

        feature_a = row["feature_a"]
        feature_b = row["feature_b"]

        value = (
            correlation_records[
                (
                    correlation_records["feature_a"]
                    == feature_a
                )
                &
                (
                    correlation_records["feature_b"]
                    == feature_b
                )
            ]["correlation"]
            .median()
        )

        matrix.loc[feature_a, feature_b] = value
        matrix.loc[feature_b, feature_a] = value

    return matrix


# ============================================================
# REPORTING
# ============================================================

def print_report(results):

    print("\n" + "=" * 70)
    print("FEATURE REDUNDANCY REPORT")
    print("=" * 70)

    print(
        f"\nCorrelation threshold: "
        f"{CORRELATION_THRESHOLD}"
    )

    print(
        f"\nTotal feature pairs: {len(results):,}"
    )

    redundant = results[
        results["potentially_redundant"]
    ]

    print(
        f"Potentially redundant pairs: "
        f"{len(redundant):,}"
    )

    print(
        "\n=== TOP 30 CORRELATED FEATURE PAIRS ==="
    )

    print(
        results[
            [
                "feature_a",
                "feature_b",
                "pearson_median",
                "spearman_median",
                "pearson_assets_above_threshold",
                "spearman_assets_above_threshold",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print(
        "\n=== POTENTIALLY REDUNDANT FEATURES ==="
    )

    if redundant.empty:

        print(
            "No feature pairs exceeded the threshold."
        )

    else:

        print(
            redundant[
                [
                    "feature_a",
                    "feature_b",
                    "pearson_abs_median",
                    "spearman_abs_median",
                    "pearson_assets_above_threshold",
                    "spearman_assets_above_threshold",
                ]
            ]
            .to_string(index=False)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FEATURE REDUNDANCY ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_training_data()

    validate_data(data)

    (
        pearson_records,
        spearman_records,
    ) = calculate_correlations_by_asset(data)

    results = aggregate_correlations(
        pearson_records,
        spearman_records,
    )

    pearson_matrix = build_median_correlation_matrix(
        pearson_records
    )

    spearman_matrix = build_median_correlation_matrix(
        spearman_records
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    pairs_path = (
        OUTPUT_DIR
        / "feature_redundancy_pairs.csv"
    )

    pearson_path = (
        OUTPUT_DIR
        / "feature_correlation_pearson.csv"
    )

    spearman_path = (
        OUTPUT_DIR
        / "feature_correlation_spearman.csv"
    )

    results.to_csv(
        pairs_path,
        index=False,
    )

    pearson_matrix.to_csv(
        pearson_path,
    )

    spearman_matrix.to_csv(
        spearman_path,
    )

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print_report(results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nFiles saved:")

    print(pairs_path)
    print(pearson_path)
    print(spearman_path)

    print(
        "\nIMPORTANT: Only the training dataset was used."
    )


if __name__ == "__main__":
    main()
