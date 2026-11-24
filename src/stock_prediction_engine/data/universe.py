
"""Central configuration of the initial multi-asset universe."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    """Metadata describing a financial instrument."""

    ticker: str
    name: str
    asset_class: str
    sector: str
    market: str = "USA"


ASSET_UNIVERSE = (
    Asset("AAPL", "Apple", "Equity", "Technology"),
    Asset("MSFT", "Microsoft", "Equity", "Technology"),
    Asset("NVDA", "NVIDIA", "Equity", "Semiconductors"),
    Asset("AMZN", "Amazon", "Equity", "Consumer Discretionary"),
    Asset("GOOGL", "Alphabet", "Equity", "Communication Services"),
    Asset("META", "Meta Platforms", "Equity", "Communication Services"),
    Asset("JPM", "JPMorgan Chase", "Equity", "Financials"),
    Asset("XOM", "Exxon Mobil", "Equity", "Energy"),

    Asset("SPY", "SPDR S&P 500 ETF", "Equity ETF", "US Large Cap"),
    Asset("QQQ", "Invesco QQQ ETF", "Equity ETF", "US Technology"),
    Asset("IWM", "iShares Russell 2000 ETF", "Equity ETF", "US Small Cap"),
    Asset("EFA", "iShares MSCI EAFE ETF", "Equity ETF", "International Equity"),

    Asset("TLT", "iShares 20+ Year Treasury Bond ETF", "Bond ETF", "US Treasury"),
    Asset("GLD", "SPDR Gold Shares", "Commodity ETF", "Gold"),
    Asset("USO", "United States Oil Fund", "Commodity ETF", "Oil"),
)


TICKERS = tuple(asset.ticker for asset in ASSET_UNIVERSE)


def get_asset_metadata(ticker: str) -> Asset:
    """Return metadata for a ticker."""

    ticker = ticker.upper()

    for asset in ASSET_UNIVERSE:
        if asset.ticker == ticker:
            return asset

    raise ValueError(f"Ticker not found in universe: {ticker}")


def print_universe() -> None:
    """Display the configured asset universe."""

    print("\nCONFIGURED ASSET UNIVERSE")
    print("=" * 65)

    for asset in ASSET_UNIVERSE:
        print(
            f"{asset.ticker:<8} "
            f"{asset.name:<32} "
            f"{asset.asset_class:<15} "
            f"{asset.sector}"
        )

    print("-" * 65)
    print(f"Total instruments: {len(ASSET_UNIVERSE)}")


if __name__ == "__main__":
    print_universe()
