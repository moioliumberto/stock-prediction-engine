
"""Train and evaluate a Random Forest classification model."""

from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from stock_prediction_engine.features import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "AAPL_features.csv"
)

TARGET_COLUMN = "target_5d"
HORIZON = 5
PURGE_GAP = HORIZON

N_SPLITS = 5
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    """Load and validate the feature dataset."""

    print("Loading feature dataset...")

    if not FEATURES_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURES_DATA_PATH}"
        )

    data = pd.read_csv(
        FEATURES_DATA_PATH,
        index_col="Date",
        parse_dates=True,
    )

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("The dataset is empty.")

    if data[required_columns].isna().any().any():
        raise ValueError("Missing values detected.")

    return data.sort_index()


def create_train_validation_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create chronological training and validation sets."""

    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)

    train = data.iloc[:train_end - PURGE_GAP].copy()

    validation = data.iloc[
        train_end:validation_end - PURGE_GAP
    ].copy()

    print("\nChronological split")
    print("-" * 50)

    print(f"Training observations: {len(train)}")
    print(
        f"Training period: {train.index.min().date()} "
        f"to {train.index.max().date()}"
    )

    print(f"\nValidation observations: {len(validation)}")
    print(
        f"Validation period: {validation.index.min().date()} "
        f"to {validation.index.max().date()}"
    )

    print(f"\nPurge gap: {PURGE_GAP}")

    print("\nTraining target distribution:")
    print(train[TARGET_COLUMN].value_counts(normalize=True))

    print("\nValidation target distribution:")
    print(
        validation[TARGET_COLUMN].value_counts(normalize=True)
    )

    return train, validation


def create_model(
    n_estimators: int = 300,
    max_depth: int | None = 5,
    min_samples_leaf: int = 20,
) -> RandomForestClassifier:
    """Create a Random Forest classifier."""

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def calculate_metrics(
    y_true: pd.Series,
    predictions,
    probabilities,
) -> dict:
    """Calculate classification metrics."""

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_true, probabilities
        ),
    }


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Evaluate Random Forest using chronological cross-validation."""

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        gap=PURGE_GAP,
    )

    results = []

    print("\nRandom Forest Cross-Validation")
    print("-" * 50)
    print(f"Folds: {N_SPLITS}")
    print(f"Purge gap: {PURGE_GAP}")

    for fold, (train_idx, validation_idx) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_validation = X.iloc[validation_idx]
        y_validation = y.iloc[validation_idx]

        print(f"\nFold {fold}")
        print(f"Training observations: {len(X_train)}")
        print(
            f"Validation observations: {len(X_validation)}"
        )

        # Dummy baseline
        dummy = DummyClassifier(
            strategy="most_frequent"
        )

        dummy.fit(X_train, y_train)

        dummy_predictions = dummy.predict(X_validation)
        dummy_probabilities = dummy.predict_proba(
            X_validation
        )[:, 1]

        dummy_metrics = calculate_metrics(
            y_validation,
            dummy_predictions,
            dummy_probabilities,
        )

        results.append(
            {
                "Fold": fold,
                "Model": "Dummy",
                **dummy_metrics,
            }
        )

        # Random Forest
        model = create_model()

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)
        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        metrics = calculate_metrics(
            y_validation,
            predictions,
            probabilities,
        )

        results.append(
            {
                "Fold": fold,
                "Model": "Random Forest",
                **metrics,
            }
        )

        print(
            f"Random Forest ROC-AUC: "
            f"{metrics['ROC-AUC']:.4f}"
        )
        print(
            f"Dummy ROC-AUC: "
            f"{dummy_metrics['ROC-AUC']:.4f}"
        )

    return pd.DataFrame(results)


def summarize_cv(results: pd.DataFrame) -> None:
    """Summarize cross-validation metrics."""

    summary = (
        results
        .groupby("Model")
        .agg(
            Accuracy_mean=("Accuracy", "mean"),
            Accuracy_std=("Accuracy", "std"),
            F1_mean=("F1", "mean"),
            F1_std=("F1", "std"),
            ROC_AUC_mean=("ROC-AUC", "mean"),
            ROC_AUC_std=("ROC-AUC", "std"),
        )
        .reset_index()
    )

    print("\nCross-Validation Summary")
    print("=" * 75)

    print(
        summary.to_string(
            index=False,
            formatters={
                "Accuracy_mean": "{:.4f}".format,
                "Accuracy_std": "{:.4f}".format,
                "F1_mean": "{:.4f}".format,
                "F1_std": "{:.4f}".format,
                "ROC_AUC_mean": "{:.4f}".format,
                "ROC_AUC_std": "{:.4f}".format,
            },
        )
    )


def evaluate_validation(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> None:
    """Train on the full training set and evaluate validation."""

    print("\nFinal Random Forest Validation")
    print("=" * 65)

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)
    probabilities = model.predict_proba(X_validation)[:, 1]

    metrics = calculate_metrics(
        y_validation,
        predictions,
        probabilities,
    )

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nConfusion Matrix")
    print("Rows = actual | Columns = predicted")
    print("Class 0 = Down | Class 1 = Up")

    print(
        confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        )
    )

    print("\nPredicted class distribution:")
    print(
        pd.Series(predictions)
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nProbability statistics:")
    print(f"Mean: {probabilities.mean():.4f}")
    print(f"Std: {probabilities.std():.4f}")
    print(f"Min: {probabilities.min():.4f}")
    print(f"Max: {probabilities.max():.4f}")

    print("\nFeature Importance")
    print("-" * 50)

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(ascending=False)

    print(importance.to_string())


def main() -> None:
    """Run the complete Random Forest experiment."""

    data = load_dataset()

    print(f"Observations: {len(data)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Target: {TARGET_COLUMN}")

    train, validation = create_train_validation_split(data)

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN].astype(int)

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN].astype(int)

    cv_results = run_cross_validation(
        X_train,
        y_train,
    )

    summarize_cv(cv_results)

    model = create_model()

    evaluate_validation(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    print("\nRandom Forest classification completed.")
    print("The Test Set has not been used.")


if __name__ == "__main__":
    main()
