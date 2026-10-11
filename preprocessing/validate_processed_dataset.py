
import os
import re
import numpy as np
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

EXPECTED_BEARINGS = {
    "K001", "K002", "K003",
    "KA03", "KA04", "KA05"
}

EXPECTED_CONDITIONS = {
    "N09_M07_F10",
    "N15_M01_F10",
    "N15_M07_F04",
    "N15_M07_F10"
}

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"Processed CSV not found: {CSV_FILE}"
    )

df = pd.read_csv(CSV_FILE)

print("=" * 65)
print("CYBERSNIFF - PROCESSED DATASET VALIDATION")
print("=" * 65)

# 1. Basic structure
print("\n1. DATASET SHAPE")
print(df.shape)

required_columns = {
    "source_file",
    "bearing_id",
    "label",
    "damage_type"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )

# 2. Recording counts
print("\n2. RECORDINGS PER BEARING")
print(df["bearing_id"].value_counts().sort_index())

print("\n3. CLASS DISTRIBUTION")
print(df["label"].value_counts())

# 3. Duplicate filenames
duplicate_names = df[
    df["source_file"].duplicated(keep=False)
]

print("\n4. DUPLICATE SOURCE FILENAMES")
if duplicate_names.empty:
    print("No duplicate source filenames.")
else:
    print(duplicate_names["source_file"].to_list())

# 4. Extract operating condition from filename
pattern = re.compile(
    r"^(N\d{2}_M\d{2}_F\d{2})_"
    r"(KA\d{2}|K\d{3})_(\d+)\.mat$",
    re.IGNORECASE
)

parsed = df["source_file"].astype(str).str.upper().str.extract(
    pattern
)

df["condition"] = parsed[0]
df["filename_bearing"] = parsed[1]
df["measurement_number"] = parsed[2]

print("\n5. FILENAMES THAT COULD NOT BE PARSED")
bad_names = df[df["condition"].isna()]

if bad_names.empty:
    print("All source filenames parsed successfully.")
else:
    print(bad_names["source_file"].to_list())

print("\n6. RECORDINGS PER BEARING AND CONDITION")
counts = pd.crosstab(
    df["bearing_id"],
    df["condition"]
).reindex(
    index=sorted(EXPECTED_BEARINGS),
    columns=sorted(EXPECTED_CONDITIONS),
    fill_value=0
)

print(counts.to_string())

print("\n7. LABEL CONSISTENCY")
mismatch = df[
    df["bearing_id"].astype(str).str.upper()
    != df["filename_bearing"].astype(str)
]

if mismatch.empty:
    print("Bearing IDs match filenames.")
else:
    print("Bearing ID mismatches:")
    print(mismatch[
        ["source_file", "bearing_id", "filename_bearing"]
    ].to_string(index=False))

# 5. Numerical data quality
numeric = df.select_dtypes(include=[np.number])

missing_values = int(df.isna().sum().sum())
infinite_values = int(
    np.isinf(numeric.to_numpy()).sum()
)

print("\n8. DATA QUALITY")
print("Total missing values:", missing_values)
print("Infinite numerical values:", infinite_values)

# 6. Expected inventory checks
print("\n9. EXPECTED INVENTORY CHECKS")

print("Total rows == 480:", len(df) == 480)
print("Six expected bearings:",
      set(df["bearing_id"].unique()) == EXPECTED_BEARINGS)
print("Four expected conditions:",
      set(df["condition"].dropna().unique()) == EXPECTED_CONDITIONS)
print("No duplicate filenames:", duplicate_names.empty)
print("No unparsed filenames:", bad_names.empty)
print("No missing values:", missing_values == 0)
print("No infinite values:", infinite_values == 0)
print("Exactly 20 per bearing-condition:",
      bool((counts == 20).all().all()))

print("\nValidation complete.")
