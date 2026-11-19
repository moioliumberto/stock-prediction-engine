
"""Download historical stock market data using Yahoo Finance."""

from pathlib import Path

import pandas as pd
import yfinance as yf


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory where raw market data will be stored
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_stock_data(
    ticker: str = "AAPL",
    period: str = "10y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download historical market data for a stock.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL).
        period: Historical period (e.g. 10y).
        interval: Data frequency (e.g. 1d).

    Returns:
        DataFrame containing historical market data.
    """

    print(f"Downloading historical data for {ticker}...")

    stock = yf.Ticker(ticker)

    data = stock.history(
        period=period,
        interval=interval,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError(
            f"No historical data found for ticker: {ticker}"
        )

    data.index.name = "Date"

    data = data.dropna(subset=["Close"])

    data = data[~data.index.duplicated(keep="first")]

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_path = RAW_DATA_DIR / f"{ticker.upper()}_historical.csv"

    data.to_csv(file_path)

    print(f"Download completed: {len(data)} rows")
    print(f"File saved to: {file_path}")

    return data


if __name__ == "__main__":
    df = download_stock_data()

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())
