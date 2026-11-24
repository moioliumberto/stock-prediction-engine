
"""Chronological train, validation and test splitting.

All instruments sharing a date are assigned to the same partition.
A 20-trading-day purge gap prevents forward targets from crossing
partition boundaries.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_asset"
    / "model_ready_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_asset"
    / "splits"
)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

PURGE_DAYS = 20


def load_dataset() -> pd.DataFrame:
    """Load and validate the model-ready dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Model-ready dataset not found: {INPUT_PATH}"
        )

    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"],
    )

    if data.duplicated(["Date", "Ticker"]).any():
        raise ValueError(
            "Duplicate Date-Ticker combinations detected."
        )

    return data.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)


def split_by_date(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the dataset chronologically using shared date boundaries."""

    ratios_sum = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if abs(ratios_sum - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.")

    dates = pd.DatetimeIndex(
        sorted(data["Date"].unique())
    )

    total_dates = len(dates)

    train_boundary = int(total_dates * TRAIN_RATIO)

    validation_boundary = int(
        total_dates * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    # Purge the last 20 dates before each boundary.
    train_end = train_boundary - PURGE_DAYS
    validation_end = validation_boundary - PURGE_DAYS

    if train_end <= 0:
        raise ValueError("Insufficient dates for training.")

    if validation_end <= train_boundary:
        raise ValueError("Insufficient dates for validation.")

    if validation_boundary >= total_dates:
        raise ValueError("Insufficient dates for testing.")

    train_dates = dates[:train_end]

    validation_dates = dates[
        train_boundary:validation_end
    ]

    test_dates = dates[validation_boundary:]

    splits = {
        "train": data[data["Date"].isin(train_dates)].copy(),
        "validation": data[
            data["Date"].isin(validation_dates)
        ].copy(),
        "test": data[data["Date"].isin(test_dates)].copy(),
    }

    return splits


def validate_splits(
    splits: dict[str, pd.DataFrame],
) -> None:
    """Check date separation, coverage and instrument consistency."""

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    train_dates = set(train["Date"].unique())
    validation_dates = set(validation["Date"].unique())
    test_dates = set(test["Date"].unique())

    if train_dates & validation_dates:
        raise ValueError("Train and validation dates overlap.")

    if train_dates & test_dates:
        raise ValueError("Train and test dates overlap.")

    if validation_dates & test_dates:
        raise ValueError("Validation and test dates overlap.")

    if not (
        max(train_dates)
        < min(validation_dates)
        < max(validation_dates)
        < min(test_dates)
    ):
        raise ValueError("Chronological ordering is invalid.")

    instruments = set(train["Ticker"].unique())

    if set(validation["Ticker"].unique()) != instruments:
        raise ValueError(
            "Validation instruments differ from training."
        )

    if set(test["Ticker"].unique()) != instruments:
        raise ValueError(
            "Test instruments differ from training."
        )

    # Every date must contain the same instruments.
    for name, frame in splits.items():
        counts = frame.groupby("Date")["Ticker"].nunique()

        if not (counts == len(instruments)).all():
            raise ValueError(
                f"Incomplete instrument coverage in {name}."
            )


def save_splits(
    splits: dict[str, pd.DataFrame],
) -> None:
    """Save the three datasets and a split summary."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    for name, frame in splits.items():
        output_path = OUTPUT_DIR / f"{name}.csv"

        frame.to_csv(
            output_path,
            index=False,
        )

        summary_rows.append(
            {
                "split": name,
                "observations": len(frame),
                "instruments": frame["Ticker"].nunique(),
                "unique_dates": frame["Date"].nunique(),
                "start_date": frame["Date"].min(),
                "end_date": frame["Date"].max(),
            }
        )

        print(f"\n{name.upper()}")
        print("-" * 55)
        print(f"Observations: {len(frame):,}")
        print(f"Instruments: {frame['Ticker'].nunique()}")
        print(f"Unique dates: {frame['Date'].nunique()}")
        print(f"Start: {frame['Date'].min()}")
        print(f"End:   {frame['Date'].max()}")
        print(f"Saved: {output_path}")

    summary = pd.DataFrame(summary_rows)

    summary_path = OUTPUT_DIR / "split_summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(f"\nSplit summary saved: {summary_path}")


def main() -> None:
    """Execute the complete chronological splitting process."""

    data = load_dataset()

    print("\nTEMPORAL DATA SPLITTING")
    print("=" * 65)
    print(f"Total observations: {len(data):,}")
    print(f"Unique dates: {data['Date'].nunique()}")
    print(f"Configured purge period: {PURGE_DAYS} trading days")

    splits = split_by_date(data)

    validate_splits(splits)

    save_splits(splits)

    print("\nSPLITTING VALIDATION")
    print("=" * 65)
    print("Date overlap: None")
    print("Chronological order: Valid")
    print("Instrument consistency: Valid")
    print("Date-level grouping: Valid")
    print(f"Purge period: {PURGE_DAYS} trading days")

    print("\nTEMPORAL SPLITTING COMPLETED")


if __name__ == "__main__":
    main()
