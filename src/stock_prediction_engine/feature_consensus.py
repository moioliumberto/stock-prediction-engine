
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_asset"
    / "models"
)


# ============================================================
# FEATURE GROUPS
# ============================================================

FEATURE_GROUPS = {
    "momentum": [
        "return_1d",
        "return_5d",
        "return_10d",
        "return_20d",
        "return_60d",
    ],
    "trend": [
        "price_to_sma_10",
        "price_to_sma_20",
        "price_to_sma_50",
        "price_to_sma_100",
        "price_to_sma_200",
        "price_to_ema_12",
        "price_to_ema_26",
        "sma_20_to_sma_50",
        "sma_50_to_sma_200",
    ],
    "volatility": [
        "volatility_10d",
        "volatility_20d",
        "volatility_60d",
        "volatility_ratio_10_60",
        "atr_pct",
    ],
    "oscillators": [
        "rsi_14",
        "macd_pct",
        "macd_signal_pct",
        "macd_hist_pct",
    ],
    "volume": [
        "volume_ratio_20",
        "obv_change_20",
    ],
    "market_context": [
        "market_spy_return_1d",
        "market_spy_return_5d",
        "market_spy_return_20d",
        "market_qqq_return_5d",
        "market_qqq_return_20d",
        "market_iwm_return_20d",
        "market_tlt_return_20d",
        "market_gld_return_20d",
        "market_uso_return_20d",
        "market_spy_volatility_20d",
    ],
    "relative_strength": [
        "relative_strength_spy_5d",
        "relative_strength_spy_20d",
        "relative_strength_spy_60d",
        "correlation_spy_60d",
        "beta_spy_60d",
    ],
}

ALL_FEATURES = [
    feature
    for features in FEATURE_GROUPS.values()
    for feature in features
]

HORIZONS = [5, 10, 20]

REDUNDANCY_THRESHOLD = 0.85
TOP_N = 10


# ============================================================
# FILE LOADING
# ============================================================

def load_report(filename: str) -> pd.DataFrame:
    """Load a report from the models directory."""

    path = MODELS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required report not found: {path}"
        )

    return pd.read_csv(path)


def validate_feature_groups() -> None:
    """Check that the feature definitions are consistent."""

    if len(ALL_FEATURES) != 40:
        raise ValueError(
            f"Expected 40 features, found {len(ALL_FEATURES)}."
        )

    if len(set(ALL_FEATURES)) != len(ALL_FEATURES):
        raise ValueError(
            "Duplicate features found in FEATURE_GROUPS."
        )


# ============================================================
# TEMPORAL PERMUTATION CV
# ============================================================

def summarize_permutation_metric(
    summary: pd.DataFrame,
    model_type: str,
    metric: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Summarize one permutation metric for each feature.

    Importance ranks are calculated separately for each horizon.
    A feature is considered top-ranked when its rank is <= TOP_N.
    """

    subset = summary[
        (summary["model_type"] == model_type)
        & (summary["metric"] == metric)
    ].copy()

    columns = [
        "feature",
        f"{prefix}_mean_importance",
        f"{prefix}_std_importance",
        f"{prefix}_positive_frequency",
        f"{prefix}_mean_rank",
        f"{prefix}_top10_horizons",
    ]

    if subset.empty:
        print(
            f"WARNING: No data found for "
            f"{model_type} / {metric}"
        )

        return pd.DataFrame(columns=columns)

    required_columns = {
        "feature",
        "horizon",
        "mean_importance",
        "std_importance",
        "positive_frequency",
        "importance_rank",
    }

    missing = required_columns - set(subset.columns)

    if missing:
        raise ValueError(
            f"Missing columns in temporal summary: {missing}"
        )

    subset["is_top10"] = (
        subset["importance_rank"] <= TOP_N
    ).astype(int)

    result = (
        subset.groupby("feature")
        .agg(
            mean_importance=("mean_importance", "mean"),
            std_importance=("std_importance", "mean"),
            positive_frequency=("positive_frequency", "mean"),
            mean_rank=("importance_rank", "mean"),
            top10_horizons=("is_top10", "sum"),
        )
        .reset_index()
    )

    result = result.rename(
        columns={
            "mean_importance": f"{prefix}_mean_importance",
            "std_importance": f"{prefix}_std_importance",
            "positive_frequency": f"{prefix}_positive_frequency",
            "mean_rank": f"{prefix}_mean_rank",
            "top10_horizons": f"{prefix}_top10_horizons",
        }
    )

    return result


# ============================================================
# FEATURE REDUNDANCY
# ============================================================

def build_redundancy_summary(
    redundancy: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate redundancy evidence for each feature."""

    required_columns = {
        "feature_a",
        "feature_b",
        "max_abs_median_correlation",
    }

    missing = required_columns - set(redundancy.columns)

    if missing:
        raise ValueError(
            f"Missing columns in redundancy report: {missing}"
        )

    pairs = redundancy[
        redundancy["max_abs_median_correlation"]
        >= REDUNDANCY_THRESHOLD
    ].copy()

    records = []

    for feature in ALL_FEATURES:
        related = pairs[
            (pairs["feature_a"] == feature)
            | (pairs["feature_b"] == feature)
        ]

        if related.empty:
            records.append(
                {
                    "feature": feature,
                    "redundant_pair_count": 0,
                    "max_abs_correlation": 0.0,
                    "strongest_redundant_feature": "",
                    "is_redundant": False,
                }
            )
            continue

        strongest_index = (
            related["max_abs_median_correlation"].idxmax()
        )

        strongest_pair = related.loc[strongest_index]

        counterpart = (
            strongest_pair["feature_b"]
            if strongest_pair["feature_a"] == feature
            else strongest_pair["feature_a"]
        )

        records.append(
            {
                "feature": feature,
                "redundant_pair_count": len(related),
                "max_abs_correlation": (
                    strongest_pair["max_abs_median_correlation"]
                ),
                "strongest_redundant_feature": counterpart,
                "is_redundant": True,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# FEATURE ABLATION
# ============================================================

def build_ablation_report(
    ablation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare each 'without group' configuration against 'full'.

    Classification:
        delta_auc = AUC_without_group - AUC_full

        Negative value:
            Removing the group reduced AUC.
            The group appears useful.

        Positive value:
            Removing the group improved AUC.

    Regression:
        delta_mae = MAE_without_group - MAE_full

        Positive value:
            Removing the group increased MAE.
            The group appears useful.

        Negative value:
            Removing the group improved MAE.
    """

    records = []

    for task in ["classification", "regression"]:
        task_data = ablation[
            ablation["task"] == task
        ]

        for horizon in HORIZONS:
            horizon_data = task_data[
                task_data["horizon"] == horizon
            ]

            full = horizon_data[
                horizon_data["configuration"] == "full"
            ]

            if full.empty:
                continue

            full_row = full.iloc[0]

            for group in FEATURE_GROUPS:
                configuration = f"without_{group}"

                removed = horizon_data[
                    horizon_data["configuration"]
                    == configuration
                ]

                if removed.empty:
                    continue

                removed_row = removed.iloc[0]

                if task == "classification":
                    full_score = full_row["roc_auc"]
                    removed_score = removed_row["roc_auc"]

                    delta = removed_score - full_score
                    metric_name = "delta_roc_auc"

                else:
                    full_score = full_row["mae"]
                    removed_score = removed_row["mae"]

                    delta = removed_score - full_score
                    metric_name = "delta_mae"

                records.append(
                    {
                        "task": task,
                        "horizon": horizon,
                        "feature_group": group,
                        "full_score": full_score,
                        "without_group_score": removed_score,
                        metric_name: delta,
                    }
                )

    report = pd.DataFrame(records)

    if report.empty:
        return report

    return report


def summarize_group_ablation(
    ablation_report: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate ablation evidence across horizons."""

    if ablation_report.empty:
        return pd.DataFrame()

    records = []

    for group in FEATURE_GROUPS:
        row = {"feature_group": group}

        classification = ablation_report[
            (ablation_report["task"] == "classification")
            & (ablation_report["feature_group"] == group)
        ]

        regression = ablation_report[
            (ablation_report["task"] == "regression")
            & (ablation_report["feature_group"] == group)
        ]

        if not classification.empty:
            row["mean_delta_roc_auc"] = (
                classification["delta_roc_auc"].mean()
            )

            row["classification_horizons_improved_by_removal"] = (
                classification["delta_roc_auc"].gt(0).sum()
            )

        else:
            row["mean_delta_roc_auc"] = float("nan")
            row["classification_horizons_improved_by_removal"] = 0

        if not regression.empty:
            row["mean_delta_mae"] = (
                regression["delta_mae"].mean()
            )

            row["regression_horizons_improved_by_removal"] = (
                regression["delta_mae"].lt(0).sum()
            )

        else:
            row["mean_delta_mae"] = float("nan")
            row["regression_horizons_improved_by_removal"] = 0

        records.append(row)

    return pd.DataFrame(records)


# ============================================================
# CONSENSUS MATRIX
# ============================================================

def build_consensus_matrix(
    temporal_summary: pd.DataFrame,
    redundancy: pd.DataFrame,
    group_ablation: pd.DataFrame,
) -> pd.DataFrame:

    matrix = pd.DataFrame(
        {"feature": ALL_FEATURES}
    )

    matrix["feature_group"] = matrix["feature"].map(
        {
            feature: group
            for group, features in FEATURE_GROUPS.items()
            for feature in features
        }
    )

    # Classification: ROC-AUC
    classification_auc = summarize_permutation_metric(
        temporal_summary,
        "classification",
        "roc_auc_drop",
        "classification_auc",
    )

    matrix = matrix.merge(
        classification_auc,
        on="feature",
        how="left",
    )

    # Classification: Balanced Accuracy
    classification_balanced = summarize_permutation_metric(
        temporal_summary,
        "classification",
        "balanced_accuracy_drop",
        "classification_balanced",
    )

    matrix = matrix.merge(
        classification_balanced,
        on="feature",
        how="left",
    )

    # Regression: MAE
    regression_mae = summarize_permutation_metric(
        temporal_summary,
        "regression",
        "mae_increase",
        "regression_mae",
    )

    matrix = matrix.merge(
        regression_mae,
        on="feature",
        how="left",
    )

    # Redundancy
    matrix = matrix.merge(
        redundancy,
        on="feature",
        how="left",
    )

    # Group-level ablation evidence
    if not group_ablation.empty:
        matrix = matrix.merge(
            group_ablation,
            left_on="feature_group",
            right_on="feature_group",
            how="left",
        )

    # --------------------------------------------------------
    # CLASSIFICATION SUPPORT
    # --------------------------------------------------------

    matrix["classification_support"] = (
        (matrix["classification_auc_top10_horizons"] >= 2)
        & (
            matrix["classification_auc_positive_frequency"]
            >= 0.60
        )
    )

    matrix["classification_experimental"] = (
        (
            matrix["classification_auc_top10_horizons"] >= 1
        )
        | (
            matrix["classification_auc_positive_frequency"]
            >= 0.60
        )
    ) & ~matrix["classification_support"]

    # --------------------------------------------------------
    # REGRESSION SUPPORT
    # --------------------------------------------------------

    matrix["regression_support"] = (
        (matrix["regression_mae_top10_horizons"] >= 2)
        & (
            matrix["regression_mae_positive_frequency"]
            >= 0.60
        )
    )

    # --------------------------------------------------------
    # FINAL CATEGORIZATION
    # --------------------------------------------------------

    def assign_category(row):
        if row["classification_support"]:
            return "Core"

        if row["classification_experimental"]:
            return "Experimental"

        if row["regression_support"]:
            return "Experimental"

        if row["is_redundant"]:
            return "Redundant"

        return "Weak"

    matrix["category"] = matrix.apply(
        assign_category,
        axis=1,
    )

    # Keep redundancy separate from the main category:
    # a Core feature can also be highly correlated with another.

    matrix["evidence_note"] = matrix.apply(
        lambda row: (
            "Strong classification evidence across horizons"
            if row["category"] == "Core"
            else
            "Promising but not sufficiently consistent"
            if row["category"] == "Experimental"
            else
            "High correlation with one or more features"
            if row["category"] == "Redundant"
            else
            "Limited evidence of predictive contribution"
        ),
        axis=1,
    )

    return matrix


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("FEATURE CONSENSUS ANALYSIS")
    print("=" * 70)

    validate_feature_groups()

    print(f"\nFeatures defined: {len(ALL_FEATURES)}")
    print(f"Feature groups: {len(FEATURE_GROUPS)}")

    # Load input reports
    temporal_summary = load_report(
        "temporal_permutation_cv_summary.csv"
    )

    redundancy = load_report(
        "feature_redundancy_pairs.csv"
    )

    ablation = load_report(
        "feature_ablation_results.csv"
    )

    print("\nInput reports loaded successfully.")

    # Group-level ablation
    ablation_detail = build_ablation_report(ablation)

    ablation_summary = summarize_group_ablation(
        ablation_detail
    )

    # Feature-level redundancy
    redundancy_summary = build_redundancy_summary(
        redundancy
    )

    # Build consensus matrix
    matrix = build_consensus_matrix(
        temporal_summary,
        redundancy_summary,
        ablation_summary,
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    matrix_path = MODELS_DIR / "feature_consensus_matrix.csv"

    report_columns = [
        "feature",
        "feature_group",
        "category",
        "classification_support",
        "classification_auc_top10_horizons",
        "classification_auc_positive_frequency",
        "classification_auc_mean_importance",
        "regression_support",
        "regression_mae_top10_horizons",
        "regression_mae_positive_frequency",
        "regression_mae_mean_importance",
        "is_redundant",
        "redundant_pair_count",
        "max_abs_correlation",
        "strongest_redundant_feature",
        "mean_delta_roc_auc",
        "mean_delta_mae",
        "evidence_note",
    ]

    available_columns = [
        column
        for column in report_columns
        if column in matrix.columns
    ]

    report = matrix[available_columns].copy()

    report_path = MODELS_DIR / "feature_consensus_report.csv"

    ablation_detail_path = (
        MODELS_DIR / "feature_group_ablation_detail.csv"
    )

    ablation_summary_path = (
        MODELS_DIR / "feature_group_ablation_summary.csv"
    )

    matrix.to_csv(matrix_path, index=False)
    report.to_csv(report_path, index=False)

    ablation_detail.to_csv(
        ablation_detail_path,
        index=False,
    )

    ablation_summary.to_csv(
        ablation_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE CATEGORY SUMMARY")
    print("=" * 70)

    category_counts = (
        matrix["category"]
        .value_counts()
        .reindex(
            ["Core", "Experimental", "Redundant", "Weak"],
            fill_value=0,
        )
    )

    print(category_counts.to_string())

    print("\n" + "=" * 70)
    print("FEATURES BY CATEGORY")
    print("=" * 70)

    for category in [
        "Core",
        "Experimental",
        "Redundant",
        "Weak",
    ]:
        features = matrix.loc[
            matrix["category"] == category,
            "feature",
        ].tolist()

        print(f"\n{category} ({len(features)}):")

        for feature in features:
            print(f"  - {feature}")

    print("\n" + "=" * 70)
    print("FEATURE GROUP ABLATION")
    print("=" * 70)

    if not ablation_summary.empty:
        print(
            ablation_summary.to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            )
        )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(matrix_path)
    print(report_path)
    print(ablation_detail_path)
    print(ablation_summary_path)

    print("\nFeature consensus analysis completed.")


if __name__ == "__main__":
    main()
