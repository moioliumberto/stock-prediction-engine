
"""Load, validate, and save historical stock market data."""

from pathlib import Path

import pandas as pd


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# Required columns for the initial dataset
REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]


def load_and_validate_data(
    ticker: str = "AAPL",
) -> pd.DataFrame:
    """
    Load historical stock data and perform quality checks.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Validated pandas DataFrame.
    """

    ticker = ticker.upper()

    file_path = RAW_DATA_DIR / f"{ticker}_historical.csv"

    print(f"\nLoading data for {ticker}...")

    # 1. Check that the source file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Historical data file not found: {file_path}"
        )

    # 2. Load the CSV file
    data = pd.read_csv(
        file_path,
        index_col="Date",
        parse_dates=True,
    )

    if data.empty:
        raise ValueError("The dataset is empty.")

    # 3. Check required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 4. Check missing values
    missing_values = data[REQUIRED_COLUMNS].isna().sum()

    print("\nMissing values:")
    print(missing_values)

    # 5. Check duplicated dates
    duplicate_dates = data.index.duplicated().sum()

    print(f"\nDuplicated dates: {duplicate_dates}")

    if duplicate_dates > 0:
        raise ValueError(
            "Duplicate dates detected. "
            "Please investigate the source data."
        )

    # 6. Check OHLC price consistency
    invalid_prices = (
        (data["High"] < data["Low"])
        | (data["High"] < data["Open"])
        | (data["High"] < data["Close"])
        | (data["Low"] > data["Open"])
        | (data["Low"] > data["Close"])
    )

    invalid_price_count = invalid_prices.sum()

    print(f"Invalid OHLC rows: {invalid_price_count}")

    if invalid_price_count > 0:
        raise ValueError(
            "Invalid OHLC price relationships detected."
        )

    # 7. Check negative prices and volumes
    negative_values = (
        (data[["Open", "High", "Low", "Close", "Adj Close"]] < 0)
        .any(axis=1)
        | (data["Volume"] < 0)
    )

    negative_count = negative_values.sum()

    print(f"Rows with negative values: {negative_count}")

    if negative_count > 0:
        raise ValueError(
            "Negative prices or volumes detected."
        )

    # 8. Check missing values in required columns
    if missing_values.sum() > 0:
        raise ValueError(
            "Missing values detected in required columns."
        )

    # 9. Save validated data
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_DATA_DIR / f"{ticker}_processed.csv"
    )

    data.to_csv(output_path)

    print("\nData validation completed successfully.")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Processed file saved to: {output_path}")

    return data


if __name__ == "__main__":
    df = load_and_validate_data()

    print("\nFirst 5 rows of processed data:")
    print(df.head())

    print("\nDataset summary:")
    df.info()
