
import os
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score
)
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_real_features.csv"
)
OUTPUT_DIR = os.path.join(ROOT, "models", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
metadata = ["source_file", "bearing_id", "label", "damage_type"]
feature_cols = df.drop(columns=metadata, errors="ignore") \
                 .select_dtypes(include="number").columns.tolist()

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "HistGradientBoosting": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.1,
            l2_regularization=1.0,
            random_state=42
        )
    )
}

records = []

for held_out in sorted(df["bearing_id"].unique()):
    train_df = df[df["bearing_id"] != held_out].copy()
    test_df = df[df["bearing_id"] == held_out].copy()

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["label"]
    y_test = test_df["label"]

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        fault_mask = y_test == "Physical_Fault"
        fault_recall = (
            recall_score(
                y_test[fault_mask],
                pred[fault_mask],
                pos_label="Physical_Fault",
                zero_division=0
            )
            if fault_mask.any() else float("nan")
        )

        records.append({
            "model": name,
            "held_out_bearing": held_out,
            "samples": len(test_df),
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "fault_recall": fault_recall
        })

        print(
            f"{name:22s} | held out: {held_out:4s} | "
            f"accuracy: {accuracy_score(y_test, pred):.3f} | "
            f"balanced accuracy: "
            f"{balanced_accuracy_score(y_test, pred):.3f} | "
            f"fault recall: {fault_recall:.3f}"
        )

results = pd.DataFrame(records)
out_path = os.path.join(
    OUTPUT_DIR, "model_comparison_by_bearing.csv"
)
results.to_csv(out_path, index=False)

print("\nMEAN SCORES ACROSS HELD-OUT BEARINGS")
print(
    results.groupby("model")[
        ["accuracy", "balanced_accuracy", "fault_recall"]
    ].mean().round(3).to_string()
)

print(f"\nSaved results: {out_path}")
