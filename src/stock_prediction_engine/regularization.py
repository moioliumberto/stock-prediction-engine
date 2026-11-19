
"""Compare L1, L2 and Elastic Net regularization using time-series CV."""

from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_prediction_engine.features import FEATURE_COLUMNS


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "AAPL_features.csv"

HORIZON = 5
TARGET_COLUMN = f"target_{HORIZON}d"

TRAIN_RATIO = 0.70
PURGE_GAP = HORIZON

N_SPLITS = 5
RANDOM_STATE = 42
MAX_ITER = 20000

C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
ELASTIC_NET_L1_RATIOS = [0.25, 0.50, 0.75]


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_training_data():
    """Load the dataset and isolate the training period."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(
        DATA_PATH,
        index_col="Date",
        parse_dates=True,
    )

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data.dropna(subset=required_columns).sort_index()

    train_end = int(len(data) * TRAIN_RATIO)

    train = data.iloc[:train_end - PURGE_GAP].copy()

    return train


# ---------------------------------------------------------
# EXPERIMENT CONFIGURATIONS
# ---------------------------------------------------------

def build_experiments():
    """Create the regularization configurations to compare."""

    experiments = []

    # L2 regularization
    for c_value in C_VALUES:
        experiments.append({
            "name": "L2",
            "penalty": "l2",
            "C": c_value,
            "l1_ratio": None,
            "solver": "lbfgs",
        })

    # L1 regularization
    for c_value in C_VALUES:
        experiments.append({
            "name": "L1",
            "penalty": "l1",
            "C": c_value,
            "l1_ratio": None,
            "solver": "liblinear",
        })

    # Elastic Net regularization
    for c_value in C_VALUES:
        for l1_ratio in ELASTIC_NET_L1_RATIOS:
            experiments.append({
                "name": "Elastic Net",
                "penalty": "elasticnet",
                "C": c_value,
                "l1_ratio": l1_ratio,
                "solver": "saga",
            })

    return experiments


# ---------------------------------------------------------
# MODEL FACTORY
# ---------------------------------------------------------

def create_logistic_model(experiment):
    """Create a standardized Logistic Regression pipeline."""

    parameters = {
        "penalty": experiment["penalty"],
        "C": experiment["C"],
        "solver": experiment["solver"],
        "max_iter": MAX_ITER,
        "random_state": RANDOM_STATE,
    }

    if experiment["penalty"] == "elasticnet":
        parameters["l1_ratio"] = experiment["l1_ratio"]

    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(**parameters),
        ),
    ])


# ---------------------------------------------------------
# CROSS-VALIDATION
# ---------------------------------------------------------

def evaluate_experiment(train, experiment, splitter):
    """Evaluate one configuration over all chronological folds."""

    X = train[FEATURE_COLUMNS]
    y = train[TARGET_COLUMN]

    fold_results = []
    zero_coefficients = []

    for fold, (train_idx, validation_idx) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_validation = X.iloc[validation_idx]

        y_train = y.iloc[train_idx]
        y_validation = y.iloc[validation_idx]

        model = create_logistic_model(experiment)

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)
        probabilities = model.predict_proba(X_validation)[:, 1]

        coefficients = (
            model.named_steps["model"].coef_[0]
        )

        zero_count = (
            abs(coefficients) < 1e-8
        ).sum()

        zero_coefficients.append(zero_count)

        fold_results.append({
            "fold": fold,
            "accuracy": accuracy_score(
                y_validation,
                predictions,
            ),
            "roc_auc": roc_auc_score(
                y_validation,
                probabilities,
            ),
            "zero_coefficients": zero_count,
        })

    results = pd.DataFrame(fold_results)

    return {
        "accuracy_mean": results["accuracy"].mean(),
        "accuracy_std": results["accuracy"].std(),
        "roc_auc_mean": results["roc_auc"].mean(),
        "roc_auc_std": results["roc_auc"].std(),
        "zero_coefficients_mean": sum(zero_coefficients) / len(
            zero_coefficients
        ),
        "fold_results": results,
    }


# ---------------------------------------------------------
# DUMMY BENCHMARK
# ---------------------------------------------------------

def evaluate_dummy(train, splitter):
    """Evaluate the majority-class benchmark."""

    X = train[FEATURE_COLUMNS]
    y = train[TARGET_COLUMN]

    fold_results = []

    for fold, (train_idx, validation_idx) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_validation = X.iloc[validation_idx]

        y_train = y.iloc[train_idx]
        y_validation = y.iloc[validation_idx]

        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)
        probabilities = model.predict_proba(X_validation)[:, 1]

        fold_results.append({
            "fold": fold,
            "accuracy": accuracy_score(
                y_validation,
                predictions,
            ),
            "roc_auc": roc_auc_score(
                y_validation,
                probabilities,
            ),
        })

    results = pd.DataFrame(fold_results)

    return {
        "accuracy_mean": results["accuracy"].mean(),
        "accuracy_std": results["accuracy"].std(),
        "roc_auc_mean": results["roc_auc"].mean(),
        "roc_auc_std": results["roc_auc"].std(),
    }


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------

def main():
    """Run the complete regularization experiment."""

    print("\n" + "=" * 70)
    print("STOCK PREDICTION ENGINE")
    print("REGULARIZATION EXPERIMENT")
    print("=" * 70)

    train = load_training_data()

    print(f"\nPrediction horizon: {HORIZON} trading sessions")
    print(f"Training observations: {len(train)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Cross-validation folds: {N_SPLITS}")
    print(f"Purge gap: {PURGE_GAP}")

    print("\nRegularization configurations:")
    print(f"L2 C values: {C_VALUES}")
    print(f"L1 C values: {C_VALUES}")
    print(f"Elastic Net C values: {C_VALUES}")
    print(f"Elastic Net L1 ratios: {ELASTIC_NET_L1_RATIOS}")

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        gap=PURGE_GAP,
    )

    # Dummy benchmark
    print("\n" + "=" * 70)
    print("DUMMY CLASSIFIER BENCHMARK")
    print("=" * 70)

    dummy_results = evaluate_dummy(train, splitter)

    print(
        f"Accuracy: {dummy_results['accuracy_mean']:.4f} "
        f"+/- {dummy_results['accuracy_std']:.4f}"
    )

    print(
        f"ROC-AUC:  {dummy_results['roc_auc_mean']:.4f} "
        f"+/- {dummy_results['roc_auc_std']:.4f}"
    )

    # Logistic Regression experiments
    experiments = build_experiments()

    all_results = []

    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION REGULARIZATION")
    print("=" * 70)

    for index, experiment in enumerate(experiments, start=1):

        description = (
            f"{experiment['name']} | "
            f"C={experiment['C']}"
        )

        if experiment["l1_ratio"] is not None:
            description += (
                f" | l1_ratio={experiment['l1_ratio']}"
            )

        print(
            f"\n[{index}/{len(experiments)}] "
            f"Evaluating {description}"
        )

        result = evaluate_experiment(
            train,
            experiment,
            splitter,
        )

        row = {
            "method": experiment["name"],
            "C": experiment["C"],
            "l1_ratio": experiment["l1_ratio"],
            "accuracy_mean": result["accuracy_mean"],
            "accuracy_std": result["accuracy_std"],
            "roc_auc_mean": result["roc_auc_mean"],
            "roc_auc_std": result["roc_auc_std"],
            "zero_coefficients_mean": (
                result["zero_coefficients_mean"]
            ),
        }

        all_results.append(row)

        print(
            f"ROC-AUC: {result['roc_auc_mean']:.4f} "
            f"+/- {result['roc_auc_std']:.4f}"
        )

        print(
            f"Accuracy: {result['accuracy_mean']:.4f} "
            f"+/- {result['accuracy_std']:.4f}"
        )

        print(
            "Average zero coefficients: "
            f"{result['zero_coefficients_mean']:.1f}"
        )

    # Summary
    summary = pd.DataFrame(all_results)

    summary = summary.sort_values(
        "roc_auc_mean",
        ascending=False,
    )

    print("\n" + "=" * 70)
    print("REGULARIZATION RESULTS")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
                if pd.notna(value)
                else "N/A"
            ),
        )
    )

    print("\n" + "=" * 70)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 70)

    print(
        summary.head(5).to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
                if pd.notna(value)
                else "N/A"
            ),
        )
    )

    print("\n" + "=" * 70)
    print("REGULARIZATION EXPERIMENT COMPLETED")
    print("=" * 70)

    print("\nDummy benchmark:")
    print(f"ROC-AUC: {dummy_results['roc_auc_mean']:.4f}")

    print("\nValidation Set was not used.")
    print("Test Set was not used.")
    print("Original dataset was not modified.")

    print(
        "\nImportant: the best configuration is only a "
        "cross-validation candidate, not yet a validated model."
    )


if __name__ == "__main__":
    main()
