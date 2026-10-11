import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "paderborn_real_features.csv"
ARTIFACT = ROOT / "models" / "artifacts"
OUT = ROOT / "models" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_FILE = ARTIFACT / "physical_fault_histgradientboosting_8_features.joblib"
FEATURE_FILE = ARTIFACT / "physical_fault_histgradientboosting_8_features.json"

FOLDS = [
    ["K001", "KA03"],
    ["K002", "KA04"],
    ["K003", "KA05"],
]

df = pd.read_csv(DATA)
features = json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
template = joblib.load(MODEL_FILE)

X = df[features].apply(pd.to_numeric, errors="coerce")
y = df["label"]
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

results.to_csv(OUT / "hgb_8_features_grouped_cv_metrics.csv", index=False)
predictions.to_csv(OUT / "hgb_8_features_grouped_cv_predictions.csv", index=False)

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
print("\nSaved grouped evaluation results to:", OUT)
