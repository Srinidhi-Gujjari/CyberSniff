
import os
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_real_features.csv"
)
OUTPUT_DIR = os.path.join(ROOT, "models", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
metadata = ["source_file", "bearing_id", "label", "damage_type"]
features = df.drop(columns=metadata, errors="ignore") \
             .select_dtypes(include="number").columns.tolist()

records = []

for held_out in sorted(df["bearing_id"].unique()):
    train = df[df["bearing_id"] != held_out]
    test = df[df["bearing_id"] == held_out]

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        )
    )

    model.fit(train[features], train["label"])
    pred = model.predict(test[features])

    fault_rows = test["label"] == "Physical_Fault"
    fault_recall = (
        (pred[fault_rows] == "Physical_Fault").mean()
        if fault_rows.any() else float("nan")
    )

    records.append({
        "held_out_bearing": held_out,
        "samples": len(test),
        "accuracy": accuracy_score(test["label"], pred),
        "balanced_accuracy": balanced_accuracy_score(
            test["label"], pred
        ),
        "fault_recall": fault_recall
    })

results = pd.DataFrame(records)
print("\nSTANDARDIZED LOGISTIC REGRESSION")
print(results.round(3).to_string(index=False))

print("\nMean scores across held-out bearings:")
print(results[["accuracy", "balanced_accuracy", "fault_recall"]]
      .mean().round(3))

results.to_csv(
    os.path.join(OUTPUT_DIR, "scaled_logistic_results.csv"),
    index=False
)
print("\nResults saved to models/outputs/scaled_logistic_results.csv")
