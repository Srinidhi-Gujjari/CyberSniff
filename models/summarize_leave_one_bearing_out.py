
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(
    ROOT,
    "models", "outputs", "leave_one_bearing_out",
    "all_predictions.csv"
)

df = pd.read_csv(PATH)

labels = ["Healthy", "Physical_Fault"]

print("CYBERSNIFF: POOLED HELD-OUT-BEARING RESULTS")
print("=" * 55)

print(f"Total samples: {len(df)}")
print(f"Accuracy: {accuracy_score(df['label'], df['predicted_label']):.2%}")
print(
    "Balanced accuracy: "
    f"{balanced_accuracy_score(df['label'], df['predicted_label']):.2%}"
)

print("\nClassification report:")
print(classification_report(
    df["label"],
    df["predicted_label"],
    labels=labels,
    zero_division=0
))

print("Confusion matrix (actual rows, predicted columns):")
print(pd.DataFrame(
    confusion_matrix(df["label"], df["predicted_label"], labels=labels),
    index=labels,
    columns=labels
))

print("\nRecall by actual class:")
for label in labels:
    subset = df[df["label"] == label]
    recall = (subset["predicted_label"] == label).mean()
    print(f"{label}: {recall:.2%} ({len(subset)} samples)")

print("\nRecall by bearing:")
for bearing, group in df.groupby("held_out_bearing"):
    actual_faults = group[group["label"] == "Physical_Fault"]
    if len(actual_faults):
        recall = (
            actual_faults["predicted_label"] == "Physical_Fault"
        ).mean()
        print(f"{bearing}: {recall:.2%} fault recall")
