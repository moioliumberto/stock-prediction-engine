
"""Feature engineering for the Stock Prediction Engine."""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PRICE_COLUMN = "Adj Close"
VOLUME_COLUMN = "Volume"

TRADING_DAYS_PER_YEAR = 252
MARKET_TIMEZONE = "America/New_York"

RSI_PERIOD = 14

SHORT_EMA_PERIOD = 12
LONG_EMA_PERIOD = 26
SIGNAL_EMA_PERIOD = 9


# ============================================================
# 2. FEATURE AND TARGET DEFINITIONS
# ============================================================

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "price_to_sma_5",
    "price_to_sma_20",
    "price_to_sma_50",
    "volatility_10d",
    "volatility_20d",
    "rsi_14",
    "macd_pct",
    "macd_signal_pct",
    "macd_hist_pct",
    "volume_change",
]

TARGET_COLUMNS = [
    "target_return_1d",
    "target_return_5d",
    "target_1d",
    "target_5d",
]


# ============================================================
# 3. DATA LOADING AND VALIDATION
# ============================================================

def load_feature_source(ticker: str) -> pd.DataFrame:
    """Load and validate processed historical market data."""

    ticker = ticker.upper()

    input_path = (
        PROCESSED_DATA_DIR / f"{ticker}_processed.csv"
    )

    print(f"\nLoading processed data for {ticker}...")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {input_path}. "
            "Run data_processing.py first."
        )

    data = pd.read_csv(
        input_path,
        index_col="Date",
    )

    if data.empty:
        raise ValueError(
            "The processed dataset is empty."
        )

    required_columns = [
        PRICE_COLUMN,
        VOLUME_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Normalize dates to the US market timezone.
    # Yahoo Finance daily data can contain both EST and EDT
    # offsets because of daylight saving time.

    try:
        data.index = (
            pd.to_datetime(data.index, utc=True)
            .tz_convert(MARKET_TIMEZONE)
            .tz_localize(None)
        )
    except (ValueError, TypeError) as error:
        raise ValueError(
            "Unable to parse and normalize dataset dates."
        ) from error

    data.index.name = "Date"

    # Ensure chronological order.
    data = data.sort_index().copy()

    if data.index.has_duplicates:
        raise ValueError(
            "Duplicate dates detected in the dataset."
        )

    # Convert relevant columns to numeric values.
    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if data[required_columns].isna().any().any():
        raise ValueError(
            "Missing or non-numeric values detected "
            "in price or volume columns."
        )

    if not np.isfinite(
        data[required_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite values detected in source data."
        )

    if (data[PRICE_COLUMN] <= 0).any():
        raise ValueError(
            "Adjusted closing prices must be positive."
        )

    if (data[VOLUME_COLUMN] < 0).any():
        raise ValueError(
            "Trading volume cannot be negative."
        )

    print(f"Source observations: {len(data)}")

    print(
        f"Date range: {data.index.min().date()} "
        f"to {data.index.max().date()}"
    )

    return data


# ============================================================
# 4. TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(
    price: pd.Series,
    period: int = RSI_PERIOD,
) -> pd.Series:
    """Calculate RSI using Wilder-style exponential smoothing."""

    price_change = price.diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    relative_strength = average_gain.div(
        average_loss.where(average_loss != 0)
    )

    rsi = 100 - (
        100 / (1 + relative_strength)
    )

    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100,
    )

    rsi = rsi.mask(
        (average_loss == 0) & (average_gain == 0),
        50,
    )

    return rsi


def calculate_macd(
    price: pd.Series,
) -> pd.DataFrame:
    """Calculate normalized MACD, signal line and histogram."""

    short_ema = price.ewm(
        span=SHORT_EMA_PERIOD,
        min_periods=SHORT_EMA_PERIOD,
        adjust=False,
    ).mean()

    long_ema = price.ewm(
        span=LONG_EMA_PERIOD,
        min_periods=LONG_EMA_PERIOD,
        adjust=False,
    ).mean()

    macd_line = short_ema - long_ema

    signal_line = macd_line.ewm(
        span=SIGNAL_EMA_PERIOD,
        min_periods=SIGNAL_EMA_PERIOD,
        adjust=False,
    ).mean()

    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd_pct": macd_line / price,
            "macd_signal_pct": signal_line / price,
            "macd_hist_pct": histogram / price,
        },
        index=price.index,
    )


# ============================================================
# 5. HISTORICAL FEATURE ENGINEERING
# ============================================================

def create_historical_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate historical features using past and current data."""

    data = data.copy()

    price = data[PRICE_COLUMN]
    volume = data[VOLUME_COLUMN]

    # 5.1 Historical returns

    data["return_1d"] = price.pct_change(
        periods=1,
        fill_method=None,
    )

    data["return_5d"] = price.pct_change(
        periods=5,
        fill_method=None,
    )

    data["return_20d"] = price.pct_change(
        periods=20,
        fill_method=None,
    )

    # 5.2 Simple Moving Averages

    data["sma_5"] = price.rolling(
        window=5,
        min_periods=5,
    ).mean()

    data["sma_20"] = price.rolling(
        window=20,
        min_periods=20,
    ).mean()

    data["sma_50"] = price.rolling(
        window=50,
        min_periods=50,
    ).mean()

    # Normalized distance from moving averages.

    data["price_to_sma_5"] = (
        price / data["sma_5"]
    ) - 1

    data["price_to_sma_20"] = (
        price / data["sma_20"]
    ) - 1

    data["price_to_sma_50"] = (
        price / data["sma_50"]
    ) - 1

    # 5.3 Historical volatility

    daily_returns = data["return_1d"]

    data["volatility_10d"] = (
        daily_returns
        .rolling(window=10, min_periods=10)
        .std()
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    data["volatility_20d"] = (
        daily_returns
        .rolling(window=20, min_periods=20)
        .std()
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    # 5.4 Relative Strength Index

    data["rsi_14"] = calculate_rsi(
        price,
        period=RSI_PERIOD,
    )

    # 5.5 MACD

    macd_features = calculate_macd(price)

    data = data.join(macd_features)

    # 5.6 Volume variation

    data["volume_change"] = volume.pct_change(
        periods=1,
        fill_method=None,
    )

    return data


# ============================================================
# 6. FUTURE TARGET GENERATION
# ============================================================

def create_targets(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create future-return and future-direction targets."""

    data = data.copy()

    price = data[PRICE_COLUMN]

    # Future returns.

    future_return_1d = (
        price.shift(-1) / price
    ) - 1

    future_return_5d = (
        price.shift(-5) / price
    ) - 1

    data["target_return_1d"] = future_return_1d
    data["target_return_5d"] = future_return_5d

    # Classification:
    # 1 = positive future return.
    # 0 = zero or negative future return.
    # Missing future observations remain missing.

    data["target_1d"] = (
        future_return_1d > 0
    ).astype(float).where(
        future_return_1d.notna()
    )

    data["target_5d"] = (
        future_return_5d > 0
    ).astype(float).where(
        future_return_5d.notna()
    )

    return data


# ============================================================
# 7. FINAL DATASET PREPARATION
# ============================================================

def prepare_feature_dataset(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Remove unusable observations and validate the final dataset."""

    data = data.copy()

    required_output_columns = (
        FEATURE_COLUMNS + TARGET_COLUMNS
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    original_rows = len(data)

    data = data.dropna(
        subset=required_output_columns
    ).copy()

    removed_rows = original_rows - len(data)

    if data.empty:
        raise ValueError(
            "No usable observations remain after "
            "feature engineering."
        )

    data["target_1d"] = data["target_1d"].astype(int)
    data["target_5d"] = data["target_5d"].astype(int)

    if not data["rsi_14"].between(0, 100).all():
        raise ValueError(
            "Invalid RSI values detected."
        )

    if not np.isfinite(
        data[FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite values detected in features."
        )

    print("\nDataset preparation completed.")
    print(f"Original rows: {original_rows}")
    print(f"Removed rows: {removed_rows}")
    print(f"Usable rows: {len(data)}")

    return data


# ============================================================
# 8. MAIN FEATURE ENGINEERING PIPELINE
# ============================================================

def create_features(
    ticker: str = "AAPL",
) -> pd.DataFrame:
    """Execute the complete feature engineering pipeline."""

    ticker = ticker.upper()

    print("\n" + "=" * 60)
    print(f"FEATURE ENGINEERING: {ticker}")
    print("=" * 60)

    # Step 1: Load and validate source data.
    data = load_feature_source(ticker)

    # Step 2: Calculate historical features.
    data = create_historical_features(data)

    # Step 3: Generate future prediction targets.
    data = create_targets(data)

    # Step 4: Prepare the final dataset.
    data = prepare_feature_dataset(data)

    # Step 5: Save the dataset.
    output_path = (
        PROCESSED_DATA_DIR / f"{ticker}_features.csv"
    )

    data.to_csv(output_path)

    # Summary.

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)

    print(f"Ticker: {ticker}")
    print(f"Observations: {len(data)}")
    print(f"Total columns: {len(data.columns)}")
    print(f"ML features: {len(FEATURE_COLUMNS)}")
    print(f"Prediction targets: {len(TARGET_COLUMNS)}")

    print(
        f"Date range: {data.index.min().date()} "
        f"to {data.index.max().date()}"
    )

    print(f"\nDataset saved to:\n{output_path}")

    print("\nTarget distribution - 1 trading session:")

    print(
        data["target_1d"]
        .value_counts(normalize=True)
        .sort_index()
        .rename(
            index={
                0: "Down or unchanged",
                1: "Up",
            }
        )
    )

    print("\nTarget distribution - 5 trading sessions:")

    print(
        data["target_5d"]
        .value_counts(normalize=True)
        .sort_index()
        .rename(
            index={
                0: "Down or unchanged",
                1: "Up",
            }
        )
    )

    print("\nFeature columns:")

    for feature in FEATURE_COLUMNS:
        print(f" - {feature}")

    print("\nTarget columns:")

    for target in TARGET_COLUMNS:
        print(f" - {target}")

    return data


# ============================================================
# 9. SCRIPT EXECUTION
# ============================================================

if __name__ == "__main__":

    df = create_features("AAPL")

    print("\nFirst 5 observations:")
    print(df.head())

    print("\nDataset information:")
    df.info()

    print("\nFeature statistics:")
    print(df[FEATURE_COLUMNS].describe())
