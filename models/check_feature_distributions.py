
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_real_features.csv"
)

df = pd.read_csv(PATH)

features = [
    "force_std",
    "vibration_1_kurtosis",
    "force_kurtosis",
    "temp_2_bearing_module_mean",
    "speed_peak_to_peak"
]

features = [f for f in features if f in df.columns]

print("\nFEATURE MEDIANS BY BEARING")
print("=" * 65)

print(
    df.groupby(["bearing_id", "label"])[features]
      .median()
      .round(3)
      .to_string()
)

print("\nFEATURE MEDIANS BY LABEL AND OPERATING CONDITION")
print("=" * 65)

df["condition"] = df["source_file"].str.extract(
    r"^([^_]+_[^_]+_[^_]+)"
)

print(
    df.groupby(["condition", "label"])[features]
      .median()
      .round(3)
      .to_string()
)

print("\nSaved nothing; this is a read-only diagnostic.")
