
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(
    ROOT, "models", "outputs", "model_comparison_by_bearing.csv"
)

df = pd.read_csv(PATH)

print("\nMODEL PERFORMANCE FOR EACH HELD-OUT BEARING")
print("=" * 75)

for model_name, group in df.groupby("model"):
    print(f"\n{model_name}")
    print(
        group[
            ["held_out_bearing", "accuracy",
             "balanced_accuracy", "fault_recall"]
        ].round(3).to_string(index=False)
    )

print("\nWORST-CASE BEARING PERFORMANCE")
print("=" * 75)

summary = df.groupby("model").agg(
    mean_balanced_accuracy=("balanced_accuracy", "mean"),
    worst_bearing_balanced_accuracy=("balanced_accuracy", "min"),
    mean_fault_recall=("fault_recall", "mean"),
    worst_bearing_fault_recall=("fault_recall", "min")
)

print(summary.round(3).to_string())
