
import os
import pandas as pd
from sklearn.metrics import recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(
    ROOT, "models", "outputs", "unseen_bearing_diagnostics.csv"
)

df = pd.read_csv(PATH)

df["condition"] = df["source_file"].str.extract(
    r"^([^_]+_[^_]+_[^_]+)"
)

print("\nFAULT DETECTION BY OPERATING CONDITION")
print("=" * 65)

for condition, group in sorted(df.groupby("condition")):
    fault_rows = group[group["label"] == "Physical_Fault"]
    healthy_rows = group[group["label"] == "Healthy"]

    fault_recall = (
        recall_score(
            fault_rows["label"],
            fault_rows["predicted_label"],
            pos_label="Physical_Fault",
            zero_division=0
        )
        if len(fault_rows) else float("nan")
    )

    healthy_recall = (
        (healthy_rows["predicted_label"] == "Healthy").mean()
        if len(healthy_rows) else float("nan")
    )

    print(f"\nCondition: {condition}")
    print(f"Total samples: {len(group)}")
    print(f"Fault samples: {len(fault_rows)}")
    print(f"Fault recall: {fault_recall:.2%}")
    print(f"Healthy recall: {healthy_recall:.2%}")

print("\nPREDICTIONS BY BEARING AND CONDITION")
print("=" * 65)

print(pd.crosstab(
    [df["bearing_id"], df["condition"]],
    df["predicted_label"]
).to_string())
