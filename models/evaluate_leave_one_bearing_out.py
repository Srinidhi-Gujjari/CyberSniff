
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_real_features.csv"
)
OUTPUT_DIR = os.path.join(
    ROOT, "models", "outputs", "leave_one_bearing_out"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

metadata = ["source_file", "bearing_id", "label", "damage_type"]
feature_cols = df.drop(columns=metadata, errors="ignore") \
                 .select_dtypes(include="number").columns.tolist()

all_predictions = []

for held_out in sorted(df["bearing_id"].unique()):
    train_df = df[df["bearing_id"] != held_out].copy()
    test_df = df[df["bearing_id"] == held_out].copy()

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, train_df["label"])

    predictions = model.predict(X_test)

    result = test_df[
        ["source_file", "bearing_id", "label"]
    ].copy()
    result["predicted_label"] = predictions
    result["held_out_bearing"] = held_out
    all_predictions.append(result)

    print("\n" + "=" * 60)
    print(f"HELD-OUT BEARING: {held_out}")
    print("True label counts:")
    print(test_df["label"].value_counts().to_string())
    print("Predicted label counts:")
    print(pd.Series(predictions).value_counts().to_string())
    print(f"Accuracy: {accuracy_score(test_df['label'], predictions):.2%}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(test_df['label'], predictions):.2%}"
    )
    print(
        classification_report(
            test_df["label"],
            predictions,
            labels=["Healthy", "Physical_Fault"],
            zero_division=0
        )
    )
    print("Confusion matrix (Healthy, Physical_Fault):")
    print(confusion_matrix(
        test_df["label"],
        predictions,
        labels=["Healthy", "Physical_Fault"]
    ))

results = pd.concat(all_predictions, ignore_index=True)
results.to_csv(
    os.path.join(OUTPUT_DIR, "all_predictions.csv"),
    index=False
)

print("\nOVERALL HELD-OUT-BEARING SUMMARY")
summary = results.groupby("held_out_bearing").apply(
    lambda g: pd.Series({
        "samples": len(g),
        "accuracy": accuracy_score(g["label"], g["predicted_label"]),
        "balanced_accuracy": balanced_accuracy_score(
            g["label"], g["predicted_label"]
        )
    }),
    include_groups=False
)
print(summary.round(3).to_string())

print(f"\nPredictions saved to: {OUTPUT_DIR}")
