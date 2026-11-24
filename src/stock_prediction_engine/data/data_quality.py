
"""Reusable quality checks for historical market data."""

import pandas as pd


REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]


def validate_market_data(
    data: pd.DataFrame,
    ticker: str,
) -> dict:
    """Validate a historical market DataFrame and return a report."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    missing_values_total = 0
    duplicate_dates = 0
    invalid_ohlc_rows = 0
    nonpositive_price_rows = 0
    negative_volume_rows = 0

    date_ordered = False

    if not data.empty:
        duplicate_dates = int(data.index.duplicated().sum())
        date_ordered = bool(data.index.is_monotonic_increasing)

    available_required_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column in data.columns
    ]

    if available_required_columns:
        missing_values_total = int(
            data[available_required_columns].isna().sum().sum()
        )

    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    if all(column in data.columns for column in price_columns):
        invalid_ohlc = (
            (data["High"] < data["Low"])
            | (data["High"] < data["Open"])
            | (data["High"] < data["Close"])
            | (data["Low"] > data["Open"])
            | (data["Low"] > data["Close"])
        )

        invalid_ohlc_rows = int(invalid_ohlc.sum())

    available_price_columns = [
        column
        for column in ["Open", "High", "Low", "Close", "Adj Close"]
        if column in data.columns
    ]

    if available_price_columns:
        nonpositive_prices = (
            data[available_price_columns] <= 0
        ).any(axis=1)

        nonpositive_price_rows = int(nonpositive_prices.sum())

    if "Volume" in data.columns:
        negative_volume_rows = int(
            (data["Volume"] < 0).sum()
        )

    is_valid = (
        not data.empty
        and not missing_columns
        and missing_values_total == 0
        and duplicate_dates == 0
        and invalid_ohlc_rows == 0
        and nonpositive_price_rows == 0
        and negative_volume_rows == 0
        and date_ordered
    )

    return {
        "ticker": ticker,
        "is_valid": is_valid,
        "rows": len(data),
        "start_date": (
            str(data.index.min().date())
            if not data.empty else None
        ),
        "end_date": (
            str(data.index.max().date())
            if not data.empty else None
        ),
        "missing_columns": ", ".join(missing_columns),
        "missing_values": missing_values_total,
        "duplicate_dates": duplicate_dates,
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "nonpositive_price_rows": nonpositive_price_rows,
        "negative_volume_rows": negative_volume_rows,
        "date_ordered": date_ordered,
    }
