
"""Download historical data for the configured multi-asset universe."""

from pathlib import Path

import pandas as pd
import yfinance as yf

from stock_prediction_engine.data.universe import (
    ASSET_UNIVERSE,
    TICKERS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = (
    PROJECT_ROOT / "data" / "raw" / "multi_asset"
)

DEFAULT_PERIOD = "10y"
DEFAULT_INTERVAL = "1d"


def download_ticker_history(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Download historical data for one ticker."""

    ticker = ticker.upper()

    print(f"\nDownloading {ticker}...")

    stock = yf.Ticker(ticker)

    data = stock.history(
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=True,
    )

    if data.empty:
        raise ValueError(
            f"No historical data returned for {ticker}."
        )

    # Normalize the index to timezone-naive local market dates.
    data.index = pd.DatetimeIndex(data.index)

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data.index.name = "Date"

    # Keep observations in chronological order.
    data = data.sort_index()

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RAW_DATA_DIR / f"{ticker}_historical.csv"
    )

    data.to_csv(output_path)

    print(f"Downloaded observations: {len(data)}")
    print(f"Saved to: {output_path}")

    return data


def download_universe(
    tickers: tuple = TICKERS,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Download historical data for every ticker in the universe."""

    results = []

    print("\nMULTI-ASSET DATA COLLECTION")
    print("=" * 65)
    print(f"Instruments: {len(tickers)}")
    print(f"Period: {period}")
    print(f"Interval: {interval}")

    for ticker in tickers:

        try:
            data = download_ticker_history(
                ticker=ticker,
                period=period,
                interval=interval,
            )

            results.append({
                "ticker": ticker,
                "status": "SUCCESS",
                "rows": len(data),
                "start_date": str(data.index.min().date()),
                "end_date": str(data.index.max().date()),
                "error": "",
            })

        except Exception as error:
            print(f"ERROR downloading {ticker}: {error}")

            results.append({
                "ticker": ticker,
                "status": "FAILED",
                "rows": 0,
                "start_date": None,
                "end_date": None,
                "error": str(error),
            })

    results_df = pd.DataFrame(results)

    print("\nCOLLECTION SUMMARY")
    print("=" * 65)
    print(results_df.to_string(index=False))

    successful = int(
        (results_df["status"] == "SUCCESS").sum()
    )

    failed = int(
        (results_df["status"] == "FAILED").sum()
    )

    print("\nFINAL STATUS")
    print(f"Successful downloads: {successful}")
    print(f"Failed downloads: {failed}")
    print(f"Total instruments: {len(results_df)}")

    return results_df


def main():
    """Run the initial multi-asset collection."""

    results = download_universe()

    if len(results) != len(ASSET_UNIVERSE):
        raise RuntimeError(
            "The collection did not process the full universe."
        )


if __name__ == "__main__":
    main()
