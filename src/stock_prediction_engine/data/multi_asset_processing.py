
"""Validate and combine historical data for multiple assets."""

from pathlib import Path

import pandas as pd

from stock_prediction_engine.data.data_quality import (
    validate_market_data,
)
from stock_prediction_engine.data.universe import (
    ASSET_UNIVERSE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = (
    PROJECT_ROOT / "data" / "raw" / "multi_asset"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed" / "multi_asset"
)

MARKET_DATA_DIR = (
    PROJECT_ROOT / "data" / "multi_asset"
)


def load_ticker_data(ticker: str) -> pd.DataFrame:
    """Load one ticker's raw historical CSV."""

    file_path = (
        RAW_DATA_DIR / f"{ticker}_historical.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {file_path}"
        )

    data = pd.read_csv(
        file_path,
        index_col="Date",
        parse_dates=True,
    )

    data.index = pd.DatetimeIndex(data.index)

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data.index.name = "Date"

    return data


def process_universe() -> pd.DataFrame:
    """Validate, process, and combine all available instruments."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MARKET_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_frames = []
    quality_reports = []

    print("\nMULTI-ASSET DATA PROCESSING")
    print("=" * 65)

    for asset in ASSET_UNIVERSE:

        ticker = asset.ticker

        print(f"\nProcessing {ticker}...")

        try:
            data = load_ticker_data(ticker)

            report = validate_market_data(
                data=data,
                ticker=ticker,
            )

            report["status"] = (
                "VALID" if report["is_valid"] else "INVALID"
            )

            quality_reports.append(report)

            if not report["is_valid"]:
                print(
                    f"Data quality validation failed for {ticker}."
                )
                continue

            # Save the validated ticker dataset separately.
            processed_path = (
                PROCESSED_DATA_DIR
                / f"{ticker}_processed.csv"
            )

            data.to_csv(processed_path)

            # Add instrument metadata to the long-format dataset.
            asset_data = data.reset_index()

            asset_data["Ticker"] = ticker
            asset_data["AssetName"] = asset.name
            asset_data["AssetClass"] = asset.asset_class
            asset_data["Sector"] = asset.sector
            asset_data["Market"] = asset.market

            processed_frames.append(asset_data)

            print(f"Validation successful: {ticker}")
            print(f"Observations: {len(data)}")

        except Exception as error:
            print(f"ERROR processing {ticker}: {error}")

            quality_reports.append({
                "ticker": ticker,
                "is_valid": False,
                "rows": 0,
                "start_date": None,
                "end_date": None,
                "missing_columns": "",
                "missing_values": None,
                "duplicate_dates": None,
                "invalid_ohlc_rows": None,
                "nonpositive_price_rows": None,
                "negative_volume_rows": None,
                "date_ordered": False,
                "status": "ERROR",
                "error": str(error),
            })

    # Save the data quality report.
    quality_df = pd.DataFrame(quality_reports)

    quality_path = (
        PROCESSED_DATA_DIR / "data_quality_report.csv"
    )

    quality_df.to_csv(
        quality_path,
        index=False,
    )

    print("\nDATA QUALITY REPORT")
    print("=" * 65)
    print(quality_df.to_string(index=False))

    if not processed_frames:
        raise RuntimeError(
            "No valid instruments available for the multi-asset dataset."
        )

    # Combine all validated instruments in long format.
    market_dataset = pd.concat(
        processed_frames,
        ignore_index=True,
    )

    market_dataset = market_dataset.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    market_path = (
        MARKET_DATA_DIR / "market_dataset.csv"
    )

    market_dataset.to_csv(
        market_path,
        index=False,
    )

    print("\nMULTI-ASSET DATASET")
    print("=" * 65)
    print(f"Total observations: {len(market_dataset)}")
    print(
        f"Valid instruments: "
        f"{market_dataset['Ticker'].nunique()}"
    )
    print(
        f"Date range: "
        f"{market_dataset['Date'].min()} to "
        f"{market_dataset['Date'].max()}"
    )
    print(f"Saved to: {market_path}")

    return market_dataset


if __name__ == "__main__":
    process_universe()
