
"""Feature engineering for the multi-asset prediction engine."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "multi_asset" / "market_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "multi_asset"
OUTPUT_PATH = OUTPUT_DIR / "features_dataset.csv"

TARGET_HORIZONS = (5, 10, 20)
PRICE_COLUMN = "Adj Close"


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using exponentially smoothed gains and losses."""

    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gains = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_losses = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    relative_strength = average_gains / average_losses
    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.mask(average_losses == 0, 100)
    rsi = rsi.mask(
        (average_gains == 0) & (average_losses == 0),
        50,
    )

    return rsi


def calculate_adjusted_ohlc(data: pd.DataFrame) -> pd.DataFrame:
    """Adjust OHLC prices using the Adj Close / Close adjustment factor."""

    adjustment_factor = data["Adj Close"] / data["Close"]

    adjusted = pd.DataFrame(index=data.index)
    adjusted["adjusted_open"] = data["Open"] * adjustment_factor
    adjusted["adjusted_high"] = data["High"] * adjustment_factor
    adjusted["adjusted_low"] = data["Low"] * adjustment_factor

    return adjusted


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range using adjusted OHLC prices."""

    previous_close = data[PRICE_COLUMN].shift(1)

    true_range_components = pd.concat(
        [
            data["adjusted_high"] - data["adjusted_low"],
            (data["adjusted_high"] - previous_close).abs(),
            (data["adjusted_low"] - previous_close).abs(),
        ],
        axis=1,
    )

    true_range = true_range_components.max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()


def calculate_asset_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical features for one instrument."""

    data = data.sort_values("Date").copy()

    prices = data[PRICE_COLUMN]
    returns_1d = prices.pct_change(fill_method=None)

    # 1. Momentum
    for period in (1, 5, 10, 20, 60):
        data[f"return_{period}d"] = prices.pct_change(
            periods=period,
            fill_method=None,
        )

    # 2. Trend
    for period in (10, 20, 50, 100, 200):
        moving_average = prices.rolling(
            window=period,
            min_periods=period,
        ).mean()

        data[f"sma_{period}"] = moving_average
        data[f"price_to_sma_{period}"] = prices / moving_average - 1

    for period in (12, 26):
        exponential_average = prices.ewm(
            span=period,
            adjust=False,
            min_periods=period,
        ).mean()

        data[f"ema_{period}"] = exponential_average
        data[f"price_to_ema_{period}"] = (
            prices / exponential_average - 1
        )

    data["sma_20_to_sma_50"] = (
        data["sma_20"] / data["sma_50"] - 1
    )

    data["sma_50_to_sma_200"] = (
        data["sma_50"] / data["sma_200"] - 1
    )

    # 3. Volatility
    for period in (10, 20, 60):
        data[f"volatility_{period}d"] = (
            returns_1d.rolling(
                window=period,
                min_periods=period,
            ).std()
            * np.sqrt(252)
        )

    data["volatility_ratio_10_60"] = (
        data["volatility_10d"] / data["volatility_60d"]
    )

    # 4. Adjusted OHLC and ATR
    adjusted_ohlc = calculate_adjusted_ohlc(data)
    data = pd.concat([data, adjusted_ohlc], axis=1)

    data["atr_14"] = calculate_atr(data, period=14)
    data["atr_pct"] = data["atr_14"] / prices

    # 5. RSI and MACD
    data["rsi_14"] = calculate_rsi(prices, period=14)

    ema_12 = prices.ewm(
        span=12,
        adjust=False,
        min_periods=12,
    ).mean()

    ema_26 = prices.ewm(
        span=26,
        adjust=False,
        min_periods=26,
    ).mean()

    macd = ema_12 - ema_26

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    data["macd_pct"] = macd / prices
    data["macd_signal_pct"] = macd_signal / prices
    data["macd_hist_pct"] = (macd - macd_signal) / prices

    # 6. Volume
    volume_average_20 = data["Volume"].rolling(
        window=20,
        min_periods=20,
    ).mean()

    data["volume_ratio_20"] = data["Volume"] / volume_average_20

    data["volume_change_1d"] = data["Volume"].pct_change(
        fill_method=None
    )

    price_direction = np.sign(prices.diff())

    data["obv"] = (
        price_direction * data["Volume"]
    ).fillna(0).cumsum()

    data["obv_change_20"] = (
        data["obv"].diff(20) / (volume_average_20 * 20)
    )

    # 7. Future targets
    for horizon in TARGET_HORIZONS:
        future_price = prices.shift(-horizon)
        future_return = future_price / prices - 1

        data[f"target_return_{horizon}d"] = future_return

        data[f"target_direction_{horizon}d"] = np.where(
            future_return.notna(),
            (future_return > 0).astype(int),
            np.nan,
        )

    return data


def calculate_market_context(
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate market-wide features using information available at each date."""

    prices = market_data.pivot(
        index="Date",
        columns="Ticker",
        values=PRICE_COLUMN,
    ).sort_index()

    market_returns = prices.pct_change(fill_method=None)

    context = pd.DataFrame(index=prices.index)

    market_periods = {
        "SPY": (1, 5, 20),
        "QQQ": (5, 20),
        "IWM": (20,),
        "TLT": (20,),
        "GLD": (20,),
        "USO": (20,),
    }

    for ticker, periods in market_periods.items():
        if ticker not in prices.columns:
            continue

        for period in periods:
            context[f"market_{ticker.lower()}_return_{period}d"] = (
                prices[ticker].pct_change(
                    periods=period,
                    fill_method=None,
                )
            )

    if "SPY" in market_returns.columns:
        context["market_spy_volatility_20d"] = (
            market_returns["SPY"]
            .rolling(window=20, min_periods=20)
            .std()
            * np.sqrt(252)
        )

    return context.reset_index()


def add_relative_strength_and_beta(
    asset_data: pd.DataFrame,
    market_context: pd.DataFrame,
) -> pd.DataFrame:
    """Add relative performance, rolling correlation and beta versus SPY."""

    # Merge ALL market context features into each instrument.
    data = asset_data.merge(
        market_context,
        on="Date",
        how="left",
        validate="many_to_one",
    )

    spy_returns = data["market_spy_return_1d"]
    asset_returns = data["return_1d"]

    # Relative strength: asset return minus SPY return.
    for period in (5, 20, 60):
        spy_period_return = (
            (1 + spy_returns)
            .rolling(window=period, min_periods=period)
            .apply(np.prod, raw=True)
            - 1
        )

        data[f"relative_strength_spy_{period}d"] = (
            data[f"return_{period}d"] - spy_period_return
        )

    # Rolling correlation and beta versus SPY.
    data["correlation_spy_60d"] = (
        asset_returns.rolling(
            window=60,
            min_periods=40,
        ).corr(spy_returns)
    )

    market_variance = spy_returns.rolling(
        window=60,
        min_periods=40,
    ).var()

    market_covariance = asset_returns.rolling(
        window=60,
        min_periods=40,
    ).cov(spy_returns)

    data["beta_spy_60d"] = market_covariance / market_variance

    return data


def build_feature_dataset() -> pd.DataFrame:
    """Build and save the complete multi-asset feature dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"],
    )

    required_columns = {
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data = data.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    if data.duplicated(["Date", "Ticker"]).any():
        raise ValueError(
            "Duplicate Date-Ticker combinations detected."
        )

    if "SPY" not in data["Ticker"].unique():
        raise ValueError(
            "SPY is required for market context calculations."
        )

    market_context = calculate_market_context(data)

    processed_assets = []

    for ticker, asset_data in data.groupby("Ticker", sort=True):
        asset_data = calculate_asset_features(asset_data)

        asset_data = add_relative_strength_and_beta(
            asset_data,
            market_context,
        )

        processed_assets.append(asset_data)

        print(
            f"Features calculated: {ticker} "
            f"({len(asset_data)} observations)"
        )

    features = pd.concat(
        processed_assets,
        ignore_index=True,
    )

    # Remove temporary adjusted OHLC columns.
    features = features.drop(
        columns=[
            "adjusted_open",
            "adjusted_high",
            "adjusted_low",
        ],
        errors="ignore",
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    target_columns = [
        f"target_return_{horizon}d"
        for horizon in TARGET_HORIZONS
    ] + [
        f"target_direction_{horizon}d"
        for horizon in TARGET_HORIZONS
    ]

    excluded_columns = {
        "Date",
        "Ticker",
        "AssetName",
        "AssetClass",
        "Sector",
        "Market",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
        "Capital Gains",
        *target_columns,
    }

    feature_columns = [
        column
        for column in features.columns
        if column not in excluded_columns
    ]

    required_for_model = feature_columns + target_columns

    rows_before = len(features)

    features = features.dropna(
        subset=required_for_model
    ).copy()

    rows_removed = rows_before - len(features)

    features = features.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nFEATURE ENGINEERING SUMMARY")
    print("=" * 65)
    print(f"Input observations:       {rows_before:,}")
    print(f"Output observations:      {len(features):,}")
    print(f"Removed observations:     {rows_removed:,}")
    print(f"Number of instruments:    {features['Ticker'].nunique()}")
    print(f"Number of features:       {len(feature_columns)}")
    print(f"Number of targets:        {len(target_columns)}")
    print(
        f"Date range:               "
        f"{features['Date'].min()} to {features['Date'].max()}"
    )
    print(f"Output file:              {OUTPUT_PATH}")

    print("\nOBSERVATIONS PER INSTRUMENT")
    print("=" * 65)
    print(features.groupby("Ticker").size())

    print("\nMISSING VALUES IN MODEL INPUTS AND TARGETS")
    print("=" * 65)
    print(features[required_for_model].isna().sum().sum())

    print("\nFEATURE COLUMNS")
    print("=" * 65)
    for column in feature_columns:
        print(column)

    print("\nTARGET COLUMNS")
    print("=" * 65)
    for column in target_columns:
        print(column)

    return features


if __name__ == "__main__":
    build_feature_dataset()
