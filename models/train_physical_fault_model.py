
import os
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# CYBERSNIFF - PHYSICAL FAULT BASELINE CLASSIFIER
# Evaluation: hold out complete bearing IDs.
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    ROOT, "data", "processed",
    "paderborn_real_features.csv"
)

MODEL_DIR = os.path.join(ROOT, "models", "artifacts")
RESULTS_DIR = os.path.join(ROOT, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_FILE = os.path.join(
    MODEL_DIR, "physical_fault_random_forest.joblib"
)

FEATURES_FILE = os.path.join(
    MODEL_DIR, "physical_fault_features.json"
)

METRICS_FILE = os.path.join(
    RESULTS_DIR, "physical_fault_cv_metrics.csv"
)

PREDICTIONS_FILE = os.path.join(
    RESULTS_DIR, "physical_fault_cv_predictions.csv"
)

IMPORTANCE_FILE = os.path.join(
    RESULTS_DIR, "physical_fault_feature_importance.csv"
)

# Each fold tests on two entirely unseen bearing IDs:
# one healthy and one faulty.
FOLDS = [
    {"test": ["K001", "KA03"]},
    {"test": ["K002", "KA04"]},
    {"test": ["K003", "KA05"]},
]

LABELS = ["Healthy", "Physical_Fault"]

RANDOM_STATE = 42


def make_model():
    """Create a reproducible Random Forest classifier."""

    return RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def main():

    print("=" * 70)
    print("CYBERSNIFF - PHYSICAL FAULT CLASSIFICATION")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required = {
        "source_file",
        "bearing_id",
        "label",
        "damage_type",
    }

    missing_columns = required - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    if df[list(required)].isna().any().any():
        raise ValueError(
            "Missing values found in required metadata."
        )

    if not set(df["label"].unique()).issubset(set(LABELS)):
        raise ValueError(
            f"Unexpected labels: {df['label'].unique()}"
        )

    # Never use identifiers or target-related metadata as features.
    excluded = {
        "source_file",
        "bearing_id",
        "label",
        "damage_type",
    }

    feature_columns = [
        col for col in df.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found.")

    X = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    if X.isna().any().any():
        raise ValueError(
            "Numeric features contain missing or infinite values. "
            "Investigate them before training."
        )

    y = df["label"]
    bearing_ids = df["bearing_id"].astype(str)

    print("\nDataset shape:", df.shape)
    print("Number of numeric features:", len(feature_columns))
    print("\nClass distribution:")
    print(y.value_counts().to_string())

    all_predictions = []
    fold_metrics = []

    # --------------------------------------------------------
    # GROUPED EVALUATION
    # --------------------------------------------------------

    for fold_number, fold in enumerate(FOLDS, start=1):

        test_bearings = set(fold["test"])

        unknown = test_bearings - set(bearing_ids.unique())

        if unknown:
            raise ValueError(
                f"Unknown test bearing IDs: {unknown}"
            )

        test_mask = bearing_ids.isin(test_bearings)
        train_mask = ~test_mask

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]

        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]

        train_bearings = set(bearing_ids.loc[train_mask])
        actual_test_bearings = set(
            bearing_ids.loc[test_mask]
        )

        # Verify that no bearing is shared across train and test.
        assert train_bearings.isdisjoint(actual_test_bearings)

        if set(y_train.unique()) != set(LABELS):
            raise ValueError(
                f"Fold {fold_number}: training data lacks a class."
            )

        if set(y_test.unique()) != set(LABELS):
            raise ValueError(
                f"Fold {fold_number}: test data lacks a class."
            )

        model = make_model()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        balanced_acc = balanced_accuracy_score(
            y_test, predictions
        )
        precision = precision_score(
            y_test, predictions,
            pos_label="Physical_Fault",
            zero_division=0,
        )
        recall = recall_score(
            y_test, predictions,
            pos_label="Physical_Fault",
            zero_division=0,
        )
        f1 = f1_score(
            y_test, predictions,
            pos_label="Physical_Fault",
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test, predictions, labels=LABELS
        )

        print("\n" + "-" * 70)
        print(f"FOLD {fold_number}")
        print("Held-out bearings:", sorted(test_bearings))
        print("Training bearings:", sorted(train_bearings))
        print("Training recordings:", len(X_train))
        print("Test recordings:", len(X_test))
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Balanced accuracy: {balanced_acc:.4f}")
        print(f"Fault precision:   {precision:.4f}")
        print(f"Fault recall:      {recall:.4f}")
        print(f"Fault F1-score:    {f1:.4f}")

        print("\nConfusion matrix")
        print("Rows = actual; columns = predicted")
        print(pd.DataFrame(
            matrix,
            index=[f"Actual_{label}" for label in LABELS],
            columns=[f"Predicted_{label}" for label in LABELS],
        ).to_string())

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            labels=LABELS,
            zero_division=0,
        ))

        fold_metrics.append({
            "fold": fold_number,
            "test_bearings": ",".join(sorted(test_bearings)),
            "train_recordings": len(X_train),
            "test_recordings": len(X_test),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "fault_precision": precision,
            "fault_recall": recall,
            "fault_f1": f1,
        })

        for idx, prediction in zip(
            X_test.index, predictions
        ):
            all_predictions.append({
                "fold": fold_number,
                "source_file": df.loc[idx, "source_file"],
                "bearing_id": df.loc[idx, "bearing_id"],
                "actual_label": y.loc[idx],
                "predicted_label": prediction,
            })

    # --------------------------------------------------------
    # SAVE CROSS-VALIDATION RESULTS
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(fold_metrics)
    predictions_df = pd.DataFrame(all_predictions)

    metrics_df.to_csv(METRICS_FILE, index=False)
    predictions_df.to_csv(PREDICTIONS_FILE, index=False)

    print("\n" + "=" * 70)
    print("GROUPED EVALUATION SUMMARY")
    print("=" * 70)

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "fault_precision",
        "fault_recall",
        "fault_f1",
    ]

    print("\nMean across folds:")
    print(metrics_df[metric_columns].mean().round(4).to_string())

    # Every row should have exactly one out-of-fold prediction.
    if len(predictions_df) != len(df):
        raise RuntimeError(
            "Not every dataset row received exactly one "
            "out-of-fold prediction."
        )

    if predictions_df["source_file"].duplicated().any():
        raise RuntimeError(
            "Duplicate out-of-fold predictions detected."
        )

    print("\nCombined out-of-fold classification report:")
    print(classification_report(
        predictions_df["actual_label"],
        predictions_df["predicted_label"],
        labels=LABELS,
        zero_division=0,
    ))

    combined_matrix = confusion_matrix(
        predictions_df["actual_label"],
        predictions_df["predicted_label"],
        labels=LABELS,
    )

    print("\nCombined out-of-fold confusion matrix:")
    print(pd.DataFrame(
        combined_matrix,
        index=[f"Actual_{label}" for label in LABELS],
        columns=[f"Predicted_{label}" for label in LABELS],
    ).to_string())

    # --------------------------------------------------------
    # FIT FINAL MODEL ON ALL DATA FOR FUTURE DEMONSTRATIONS
    # IMPORTANT: This model is not the evaluation model.
    # Use grouped CV metrics above when reporting performance.
    # --------------------------------------------------------

    final_model = make_model()
    final_model.fit(X, y)

    joblib.dump(final_model, MODEL_FILE)

    with open(FEATURES_FILE, "w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(IMPORTANCE_FILE, index=False)

    print("\nTop 10 feature importances from the final model:")
    print(importance_df.head(10).to_string(index=False))

    print("\nSaved files:")
    print("Model:", MODEL_FILE)
    print("Features:", FEATURES_FILE)
    print("Fold metrics:", METRICS_FILE)
    print("Out-of-fold predictions:", PREDICTIONS_FILE)
    print("Feature importances:", IMPORTANCE_FILE)

    print("\nTRAINING AND EVALUATION COMPLETED!")


if __name__ == "__main__":
    main()
