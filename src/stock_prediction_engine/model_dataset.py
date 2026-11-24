
"""Prepare a curated dataset for machine learning experiments.

This module selects model inputs from the engineered multi-asset dataset.
It does not normalize or standardize features.

Scaling, when required, must be fitted on the training set only.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_asset"
    / "features_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_asset"
)

OUTPUT_PATH = OUTPUT_DIR / "model_ready_dataset.csv"


# Selected predictive features.
# Raw-scale and unstable features are deliberately excluded.
MODEL_FEATURES = [
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

    # Momentum indicators
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

    # Relative strength
    "relative_strength_spy_5d",
    "relative_strength_spy_20d",
    "relative_strength_spy_60d",

    # Market sensitivity
    "correlation_spy_60d",
    "beta_spy_60d",
]


TARGET_COLUMNS = [
    "target_return_5d",
    "target_return_10d",
    "target_return_20d",
    "target_direction_5d",
    "target_direction_10d",
    "target_direction_20d",
]


METADATA_COLUMNS = [
    "Date",
    "Ticker",
    "AssetName",
    "AssetClass",
    "Sector",
    "Market",
]


def load_feature_dataset() -> pd.DataFrame:
    """Load and validate the engineered feature dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}"
        )

    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"],
    )

    required_columns = (
        METADATA_COLUMNS
        + MODEL_FEATURES
        + TARGET_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    return data


def validate_model_dataset(data: pd.DataFrame) -> None:
    """Validate the selected model inputs and targets."""

    if data.empty:
        raise ValueError("The model dataset is empty.")

    if data.duplicated(["Date", "Ticker"]).any():
        raise ValueError(
            "Duplicate Date-Ticker combinations detected."
        )

    numeric_columns = MODEL_FEATURES + TARGET_COLUMNS

    missing_values = data[numeric_columns].isna().sum()

    if missing_values.any():
        problematic = missing_values[missing_values > 0]

        raise ValueError(
            "Missing values detected in model inputs or targets:\n"
            f"{problematic}"
        )

    numeric_data = data[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(numeric_data).all():
        raise ValueError(
            "Infinite or non-finite values detected."
        )

    for horizon in (5, 10, 20):
        target = f"target_direction_{horizon}d"

        invalid_classes = set(data[target].unique()) - {0, 1}

        if invalid_classes:
            raise ValueError(
                f"Unexpected classes in {target}: {invalid_classes}"
            )

        expected = (
            data[f"target_return_{horizon}d"] > 0
        ).astype(int)

        mismatches = (
            data[target].astype(int) != expected
        ).sum()

        if mismatches:
            raise ValueError(
                f"Target inconsistency detected for {horizon} days: "
                f"{mismatches} rows"
            )


def build_model_dataset() -> pd.DataFrame:
    """Create and save the curated model-ready dataset."""

    data = load_feature_dataset()

    selected_columns = (
        METADATA_COLUMNS
        + MODEL_FEATURES
        + TARGET_COLUMNS
    )

    model_data = data[selected_columns].copy()

    model_data = model_data.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    validate_model_dataset(model_data)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nMODEL-READY DATASET SUMMARY")
    print("=" * 65)

    print(f"Observations:       {len(model_data):,}")
    print(f"Instruments:        {model_data['Ticker'].nunique()}")
    print(f"Predictive features: {len(MODEL_FEATURES)}")
    print(f"Targets:            {len(TARGET_COLUMNS)}")
    print(
        f"Date range:         "
        f"{model_data['Date'].min()} to "
        f"{model_data['Date'].max()}"
    )

    print("\nOBSERVATIONS PER INSTRUMENT")
    print("=" * 65)
    print(model_data.groupby("Ticker").size())

    print("\nCLASS DISTRIBUTIONS")
    print("=" * 65)

    for horizon in (5, 10, 20):
        target = f"target_direction_{horizon}d"

        print(f"\n{target}")
        print(
            model_data[target]
            .value_counts(normalize=True)
            .sort_index()
        )

    print("\nVALIDATION")
    print("=" * 65)
    print("Missing values: 0")
    print("Infinite values: 0")
    print("Duplicate Date-Ticker pairs: 0")
    print("Target consistency: OK")

    print("\nOUTPUT")
    print("=" * 65)
    print(OUTPUT_PATH)

    return model_data


if __name__ == "__main__":
    build_model_dataset()
