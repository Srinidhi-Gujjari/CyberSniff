
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

print("=" * 60)
print("CYBERSNIFF - MISSING VALUE INVESTIGATION")
print("=" * 60)

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("No missing values found.")
else:
    print("\nColumns containing missing values:")
    print(missing.to_string())

    print("\nTotal missing cells:", int(missing.sum()))

    print("\nMissing values by column and class:")
    for column in missing.index:
        print(f"\nColumn: {column}")
        print(df.groupby("label", dropna=False)[column]
              .agg(["count", "size"])
              .assign(missing=lambda x: x["size"] - x["count"]))

    print("\nFirst rows containing missing values:")
    print(df[df.isna().any(axis=1)].head(10).to_string(index=False))

print("\nInvestigation complete.")
