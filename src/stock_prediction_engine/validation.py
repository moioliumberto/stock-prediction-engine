
"""Evaluate selected classification models on the validation set."""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_prediction_engine.features import FEATURE_COLUMNS


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "AAPL_features.csv"
)

TARGET_COLUMN = "target_5d"
HORIZON = 5
PURGE_GAP = HORIZON
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    """Load the feature dataset and validate required columns."""

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
        raise ValueError("The feature dataset is empty.")

    if data[required_columns].isna().any().any():
        raise ValueError(
            "Missing values detected in features or target."
        )

    data = data.sort_index()

    print(f"Observations: {len(data)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Target: {TARGET_COLUMN}")

    return data


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

    if train.empty or validation.empty:
        raise ValueError(
            "Training or validation set is empty."
        )

    print("\nChronological split")
    print("-" * 50)

    print(
        f"Training observations: {len(train)}"
    )
    print(
        f"Training period: {train.index.min().date()} "
        f"to {train.index.max().date()}"
    )

    print(
        f"\nValidation observations: {len(validation)}"
    )
    print(
        f"Validation period: {validation.index.min().date()} "
        f"to {validation.index.max().date()}"
    )

    print(f"\nPurge gap: {PURGE_GAP} observations")

    print("\nTraining target distribution:")
    print(
        train[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .rename("proportion")
    )

    print("\nValidation target distribution:")
    print(
        validation[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .rename("proportion")
    )

    return train, validation


def create_models() -> dict[str, Pipeline]:
    """Create the three models selected for validation."""

    models = {
        "Dummy Classifier": Pipeline(
            steps=[
                (
                    "model",
                    DummyClassifier(
                        strategy="most_frequent"
                    ),
                )
            ]
        ),

        "Logistic Regression L2": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        l1_ratio=0.0,
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "Logistic Regression Elastic Net": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.01,
                        l1_ratio=0.25,
                        solver="saga",
                        max_iter=20000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_model(
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict:
    """Train a model and evaluate it on validation data."""

    print("\n" + "=" * 65)
    print(f"MODEL: {name}")
    print("=" * 65)

    print("Training model...")

    model.fit(X_train, y_train)

    print("Generating validation predictions...")

    predictions = model.predict(X_validation)
    probabilities = model.predict_proba(X_validation)[:, 1]

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1],
    )

    print("\nValidation metrics")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion matrix")
    print("Rows = actual class | Columns = predicted class")
    print("Class 0 = Down | Class 1 = Up")
    print(matrix)

    prediction_distribution = (
        pd.Series(predictions)
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nPredicted class distribution:")
    print(
        prediction_distribution.rename("proportion")
    )

    print("\nPredicted probability distribution:")
    print(f"Mean:  {probabilities.mean():.4f}")
    print(f"Std:   {probabilities.std():.4f}")
    print(f"Min:   {probabilities.min():.4f}")
    print(f"Max:   {probabilities.max():.4f}")

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "Predicted Up %": (
            predictions == 1
        ).mean() * 100,
    }


def main() -> None:
    """Run the validation experiment."""

    data = load_dataset()

    train, validation = create_train_validation_split(data)

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN].astype(int)

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN].astype(int)

    models = create_models()

    results = []

    for name, model in models.items():
        result = evaluate_model(
            name=name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n\n" + "=" * 75)
    print("FINAL VALIDATION COMPARISON")
    print("=" * 75)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format,
                "ROC-AUC": "{:.4f}".format,
                "Predicted Up %": "{:.2f}%".format,
            },
        )
    )

    print("\nValidation completed.")
    print("The Test Set has not been used.")


if __name__ == "__main__":
    main()
