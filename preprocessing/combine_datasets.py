import pandas as pd
import os


# ============================================================
# CYBERSNIFF - DATASET COMBINATION
# ============================================================
#
# Combines:
#
# 1. Normal sensor data
# 2. Physical fault data
# 3. Cyber-induced anomaly data
#
# Output:
#
# combined_sensor_data.csv
#
# ============================================================


# ============================================================
# 1. FIND PROJECT ROOT
# ============================================================

project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# 2. DEFINE DATA PATHS
# ============================================================

raw_directory = os.path.join(
    project_root,
    "data",
    "raw"
)


processed_directory = os.path.join(
    project_root,
    "data",
    "processed"
)


# Create processed directory if it doesn't exist

os.makedirs(
    processed_directory,
    exist_ok=True
)


# ============================================================
# 3. INPUT FILES
# ============================================================

normal_file = os.path.join(
    raw_directory,
    "normal_sensor_data.csv"
)

fault_file = os.path.join(
    raw_directory,
    "physical_fault_data.csv"
)

fdi_file = os.path.join(
    raw_directory,
    "fdi_sensor_data.csv"
)


# ============================================================
# 4. OUTPUT FILE
# ============================================================

output_file = os.path.join(
    processed_directory,
    "combined_sensor_data.csv"
)


# ============================================================
# 5. LOAD DATASETS
# ============================================================

print("=" * 60)
print("CYBERSNIFF - DATASET COMBINATION")
print("=" * 60)

print("\nLoading datasets...")


normal_data = pd.read_csv(
    normal_file
)

fault_data = pd.read_csv(
    fault_file
)

fdi_data = pd.read_csv(
    fdi_file
)


print("\nDatasets loaded successfully!")

print(
    f"\nNormal samples: {len(normal_data)}"
)

print(
    f"Physical fault samples: {len(fault_data)}"
)

print(
    f"FDI samples: {len(fdi_data)}"
)


# ============================================================
# 6. CHECK COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("COLUMN CHECK")
print("=" * 60)

print("\nNormal columns:")
print(normal_data.columns.tolist())

print("\nPhysical fault columns:")
print(fault_data.columns.tolist())

print("\nFDI columns:")
print(fdi_data.columns.tolist())


# ============================================================
# 7. COMBINE DATASETS
# ============================================================

print("\n" + "=" * 60)
print("COMBINING DATASETS")
print("=" * 60)


combined_data = pd.concat(
    [
        normal_data,
        fault_data,
        fdi_data
    ],
    ignore_index=True
)


# ============================================================
# 8. SHUFFLE DATA
# ============================================================
#
# Important:
# If we keep all Normal rows first, then all Fault rows,
# then all Cyber rows, the ML model may learn the ordering
# instead of the actual sensor patterns.
#
# Therefore we shuffle the rows.
#
# random_state=42 makes the result reproducible.
# ============================================================

combined_data = combined_data.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# ============================================================
# 9. SAVE COMBINED DATASET
# ============================================================

combined_data.to_csv(
    output_file,
    index=False
)


print(
    f"\nCombined dataset saved to:\n"
    f"{output_file}"
)


# ============================================================
# 10. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("COMBINED DATASET INFORMATION")
print("=" * 60)

print(
    f"\nTotal samples: "
    f"{len(combined_data)}"
)

print(
    f"Total columns: "
    f"{len(combined_data.columns)}"
)


print("\nColumns:")

for column in combined_data.columns:
    print(
        f"- {column}"
    )


# ============================================================
# 11. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("LABEL DISTRIBUTION")
print("=" * 60)

print(
    combined_data["label"].value_counts()
)


# ============================================================
# 12. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

missing_values = combined_data.isnull().sum()

print(
    missing_values
)


# ============================================================
# 13. DISPLAY SAMPLE DATA
# ============================================================

print("\n" + "=" * 60)
print("RANDOM SAMPLE")
print("=" * 60)

print(
    combined_data.sample(
        10,
        random_state=42
    )
)


# ============================================================
# 14. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("DATASET COMBINATION COMPLETED!")
print("=" * 60)

print(
    "\nOutput file:"
)

print(
    output_file
)

print(
    "\nNext step:"
)

print(
    "Feature preprocessing and preparation for ML."
)