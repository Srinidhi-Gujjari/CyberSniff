import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "paderborn_real_features.csv"
ARTIFACT = ROOT / "models" / "artifacts"
OUT = ROOT / "models" / "outputs"

FEATURES = [
    "vibration_1_peak_to_peak",
    "vibration_1_min",
    "vibration_1_max",
    "vibration_1_kurtosis",
]

FOLDS = [
    ["K001", "KA03"],
    ["K002", "KA04"],
    ["K003", "KA05"],
]

df = pd.read_csv(DATA)
X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = df["label"]

template = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        l2_regularization=1.0,
        random_state=42,
    ),
)

all_predictions = []
fold_results = []

for fold_no, held_out in enumerate(FOLDS, start=1):
    train_mask = ~df["bearing_id"].isin(held_out)
    test_mask = df["bearing_id"].isin(held_out)

    model = clone(template)
    model.fit(X.loc[train_mask], y.loc[train_mask])
    pred = model.predict(X.loc[test_mask])
    actual = y.loc[test_mask]

    accuracy = accuracy_score(actual, pred)
    balanced = balanced_accuracy_score(actual, pred)
    fault_recall = recall_score(
        actual, pred, pos_label="Physical_Fault", zero_division=0
    )

    fold_results.append({
        "fold": fold_no,
        "held_out_bearings": ",".join(held_out),
        "samples": int(test_mask.sum()),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "fault_recall": fault_recall,
    })

    fold_df = df.loc[test_mask, ["source_file", "bearing_id"]].copy()
    fold_df["actual_label"] = actual.to_numpy()
    fold_df["predicted_label"] = pred
    fold_df["fold"] = fold_no
    all_predictions.append(fold_df)

    print(
        f"Fold {fold_no} | {held_out} | "
        f"accuracy={accuracy:.3f} | balanced_accuracy={balanced:.3f} | "
        f"fault_recall={fault_recall:.3f}"
    )

results = pd.DataFrame(fold_results)
predictions = pd.concat(all_predictions, ignore_index=True)

results.to_csv(OUT / "hgb_4_features_grouped_cv_metrics.csv", index=False)
predictions.to_csv(OUT / "hgb_4_features_grouped_cv_predictions.csv", index=False)

print("\nFold results:")
print(results.to_string(index=False))
print("\nMean fold metrics:")
print(results[["accuracy", "balanced_accuracy", "fault_recall"]].mean().round(4))
print("\nCombined out-of-fold report:")
print(classification_report(
    predictions["actual_label"],
    predictions["predicted_label"],
    labels=["Healthy", "Physical_Fault"],
    zero_division=0,
))
print("\nCombined confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(
    predictions["actual_label"],
    predictions["predicted_label"],
    labels=["Healthy", "Physical_Fault"],
))
print("\nExperimental four-feature evaluation complete.")
print("No existing model artifacts or eight-feature results were modified.")
print("Results saved to:", OUT)
print("WARNING: Bearing identity and class are confounded; not production validated.")
