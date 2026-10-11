
import os
import pandas as pd

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CSV_FILE = os.path.join(
    ROOT,
    "data",
    "processed",
    "paderborn_real_features.csv"
)

df = pd.read_csv(CSV_FILE)

print("Before correction:")
print(df.isna().sum()[df.isna().sum() > 0])

# Healthy samples have no damage subtype.
healthy_mask = df["label"] == "Healthy"

df.loc[healthy_mask, "damage_type"] = "Not_Applicable"

# Do not silently fill missing damage types for faulty samples.
faulty_mask = df["label"] == "Physical_Fault"

if df.loc[faulty_mask, "damage_type"].isna().any():
    raise ValueError(
        "Some Physical_Fault samples have missing damage_type."
    )

# Save the corrected dataset.
df.to_csv(CSV_FILE, index=False)

print("\nAfter correction:")
print("Missing values:", int(df.isna().sum().sum()))
print("\nDamage type distribution:")
print(df["damage_type"].value_counts(dropna=False))

print("\nCorrected dataset saved to:")
print(CSV_FILE)
