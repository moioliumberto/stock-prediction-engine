
"""Create technical indicators and multi-horizon prediction targets."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_HORIZONS = (1, 5, 10, 20)


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

TARGET_RETURN_COLUMNS = [
    f"target_return_{h}d" for h in TARGET_HORIZONS
]

TARGET_CLASS_COLUMNS = [
    f"target_{h}d" for h in TARGET_HORIZONS
]

TARGET_COLUMNS = TARGET_RETURN_COLUMNS + TARGET_CLASS_COLUMNS


def normalize_source_dates(index):
    """Normalize source dates to timezone-naive New York time."""

    dates = pd.to_datetime(index, utc=True)

    return (
        dates.tz_convert("America/New_York")
        .tz_localize(None)
    )


def calculate_rsi(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate RSI using Wilder-style exponential smoothing."""

    delta = prices.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

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

    relative_strength = average_gain / average_loss

    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.mask(
        (average_gain == 0) & (average_loss == 0),
        50,
    )

    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100,
    )

    return rsi


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create historical features and future prediction targets."""

    data = data.copy()

    if "Adj Close" not in data.columns:
        raise ValueError("Missing Adj Close column.")

    if "Volume" not in data.columns:
        raise ValueError("Missing Volume column.")

    price = data["Adj Close"].astype(float)
    volume = data["Volume"].astype(float)

    # 1. Historical returns
    data["return_1d"] = price.pct_change(1)
    data["return_5d"] = price.pct_change(5)
    data["return_20d"] = price.pct_change(20)

    # 2. Moving averages
    sma_5 = price.rolling(window=5).mean()
    sma_20 = price.rolling(window=20).mean()
    sma_50 = price.rolling(window=50).mean()

    data["price_to_sma_5"] = price / sma_5 - 1
    data["price_to_sma_20"] = price / sma_20 - 1
    data["price_to_sma_50"] = price / sma_50 - 1

    # 3. Historical volatility
    daily_returns = price.pct_change()

    data["volatility_10d"] = (
        daily_returns.rolling(window=10).std()
        * np.sqrt(252)
    )

    data["volatility_20d"] = (
        daily_returns.rolling(window=20).std()
        * np.sqrt(252)
    )

    # 4. RSI
    data["rsi_14"] = calculate_rsi(price, period=14)

    # 5. MACD
    ema_12 = price.ewm(span=12, adjust=False).mean()
    ema_26 = price.ewm(span=26, adjust=False).mean()

    macd = ema_12 - ema_26

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
    ).mean()

    macd_histogram = macd - macd_signal

    data["macd_pct"] = macd / price
    data["macd_signal_pct"] = macd_signal / price
    data["macd_hist_pct"] = macd_histogram / price

    # 6. Volume
    data["volume_change"] = volume.pct_change()

    # 7. Future returns and classification targets
    for horizon in TARGET_HORIZONS:

        future_return = (
            price.shift(-horizon) / price - 1
        )

        return_column = f"target_return_{horizon}d"
        class_column = f"target_{horizon}d"

        data[return_column] = future_return

        data[class_column] = (
            future_return > 0
        ).astype(float)

        data.loc[
            future_return.isna(),
            class_column,
        ] = np.nan

    # 8. Remove unavailable features and targets
    required_columns = FEATURE_COLUMNS + TARGET_COLUMNS

    data = data.dropna(
        subset=required_columns
    ).copy()

    for column in TARGET_CLASS_COLUMNS:
        data[column] = data[column].astype(int)

    return data


def build_feature_dataset(
    ticker: str = "AAPL",
) -> pd.DataFrame:
    """Build and save the multi-horizon feature dataset."""

    ticker = ticker.upper()

    input_path = (
        PROCESSED_DATA_DIR / f"{ticker}_processed.csv"
    )

    output_path = (
        PROCESSED_DATA_DIR / f"{ticker}_features.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    print("\n" + "=" * 60)
    print(f"FEATURE ENGINEERING: {ticker}")
    print("=" * 60)

    source = pd.read_csv(
        input_path,
        index_col="Date",
    )

    source.index = normalize_source_dates(source.index)
    source.index.name = "Date"
    source = source.sort_index()

    if source.index.has_duplicates:
        raise ValueError("Duplicate source dates detected.")

    print(f"Source observations: {len(source)}")

    data = create_features(source)

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(output_path)

    print("\nFeature engineering completed.")
    print(f"Final observations: {len(data)}")
    print(f"Total columns: {len(data.columns)}")
    print(f"ML features: {len(FEATURE_COLUMNS)}")

    print("\nPrediction targets:")

    for horizon in TARGET_HORIZONS:
        print(
            f"  {horizon} days: "
            f"target_return_{horizon}d, "
            f"target_{horizon}d"
        )

    print(f"\nStart date: {data.index.min()}")
    print(f"End date:   {data.index.max()}")

    print("\nTarget distributions:")

    for horizon in TARGET_HORIZONS:
        column = f"target_{horizon}d"

        distribution = (
            data[column]
            .value_counts(normalize=True)
            .sort_index()
            .rename(index={0: "Down", 1: "Up"})
        )

        print(f"\n{horizon}-day target:")
        print(distribution.round(4))

    print(f"\nSaved to: {output_path}")

    return data


if __name__ == "__main__":
    build_feature_dataset()
