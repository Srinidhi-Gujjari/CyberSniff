import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ============================================================
# CYBERSNIFF - DATA PREPROCESSING
# ============================================================
#
# Input:
#     data/processed/combined_sensor_data.csv
#
# Output:
#     X_train.csv
#     X_test.csv
#     y_train.csv
#     y_test.csv
#     scaler.npy
#     label_mapping.csv
#
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


input_file = os.path.join(
    project_root,
    "data",
    "processed",
    "combined_sensor_data.csv"
)


output_directory = os.path.join(
    project_root,
    "data",
    "processed"
)


os.makedirs(
    output_directory,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("CYBERSNIFF - DATA PREPROCESSING")
print("=" * 60)

print("\nLoading combined dataset...")

data = pd.read_csv(
    input_file
)

print(
    f"Dataset loaded successfully!"
)

print(
    f"Total rows: {len(data)}"
)

print(
    f"Total columns: {len(data.columns)}"
)


# ============================================================
# 3. DISPLAY DATASET
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREVIEW")
print("=" * 60)

print(
    data.head()
)


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

missing_values = data.isnull().sum()

print(
    missing_values
)


# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

duplicate_count = data.duplicated().sum()

print(
    f"Duplicate rows: {duplicate_count}"
)


if duplicate_count > 0:

    data = data.drop_duplicates()

    print(
        f"Removed {duplicate_count} duplicate rows."
    )

else:

    print(
        "No duplicate rows found."
    )


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================
#
# Features = sensor measurements
#
# Target = condition of the system
#
# ============================================================

print("\n" + "=" * 60)
print("SEPARATING FEATURES AND TARGET")
print("=" * 60)


# Remove timestamp because ML model should not
# directly learn from the date/time string.

feature_columns = [
    "water_level",
    "flow_rate",
    "pressure",
    "temperature",
    "pump_current",
    "vibration",
    "pump_status"
]


X = data[
    feature_columns
].copy()


y = data[
    "label"
].copy()


print("\nFeatures:")

for feature in feature_columns:

    print(
        f"- {feature}"
    )


print(
    f"\nNumber of features: {X.shape[1]}"
)


print(
    f"Number of samples: {X.shape[0]}"
)


# ============================================================
# 7. CHECK TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(
    y.value_counts()
)


# ============================================================
# 8. ENCODE TARGET LABELS
# ============================================================
#
# Machine learning models work better with numerical
# target values.
#
# Example:
#
# Normal                 -> 0
# Physical_Fault         -> 1
# Cyber_Induced_Anomaly  -> 2
#
# ============================================================

print("\n" + "=" * 60)
print("ENCODING LABELS")
print("=" * 60)


label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)


print("\nLabel mapping:")

for index, label in enumerate(
    label_encoder.classes_
):

    print(
        f"{label} -> {index}"
    )


# ============================================================
# 9. SAVE LABEL MAPPING
# ============================================================

label_mapping = pd.DataFrame({

    "encoded_value": range(
        len(label_encoder.classes_)
    ),

    "original_label": label_encoder.classes_

})


label_mapping_file = os.path.join(
    output_directory,
    "label_mapping.csv"
)


label_mapping.to_csv(
    label_mapping_file,
    index=False
)


print(
    f"\nLabel mapping saved to:"
)

print(
    label_mapping_file
)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================
#
# 80% -> Training
# 20% -> Testing
#
# stratify=y_encoded ensures each class is represented
# proportionally in both datasets.
#
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ============================================================
# 11. FEATURE SCALING
# ============================================================
#
# Our sensors have different numerical ranges.
#
# Example:
#
# Water level     ≈ 70
# Flow rate       ≈ 1.8
# Pressure        ≈ 4.2
# Temperature     ≈ 32
# Vibration      ≈ 0.12
#
# StandardScaler transforms them approximately to:
#
# mean = 0
# std  = 1
#
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


print(
    "Feature scaling completed."
)


# ============================================================
# 12. CONVERT BACK TO DATAFRAME
# ============================================================

X_train_scaled = pd.DataFrame(

    X_train_scaled,

    columns=feature_columns
)


X_test_scaled = pd.DataFrame(

    X_test_scaled,

    columns=feature_columns
)


# ============================================================
# 13. SAVE PROCESSED DATA
# ============================================================

X_train_file = os.path.join(
    output_directory,
    "X_train.csv"
)


X_test_file = os.path.join(
    output_directory,
    "X_test.csv"
)


y_train_file = os.path.join(
    output_directory,
    "y_train.csv"
)


y_test_file = os.path.join(
    output_directory,
    "y_test.csv"
)


# Save X

X_train_scaled.to_csv(
    X_train_file,
    index=False
)


X_test_scaled.to_csv(
    X_test_file,
    index=False
)


# Save y

pd.DataFrame({
    "label": y_train
}).to_csv(
    y_train_file,
    index=False
)


pd.DataFrame({
    "label": y_test
}).to_csv(
    y_test_file,
    index=False
)


# ============================================================
# 14. FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)


print("\nGenerated files:")

print(
    "1. X_train.csv"
)

print(
    "2. X_test.csv"
)

print(
    "3. y_train.csv"
)

print(
    "4. y_test.csv"
)

print(
    "5. label_mapping.csv"
)


# ============================================================
# 15. SHAPE CHECK
# ============================================================

print("\n" + "=" * 60)
print("FINAL SHAPES")
print("=" * 60)


print(
    f"X_train: {X_train_scaled.shape}"
)

print(
    f"X_test:  {X_test_scaled.shape}"
)

print(
    f"y_train: {len(y_train)}"
)

print(
    f"y_test:  {len(y_test)}"
)


# ============================================================
# 16. SAMPLE PROCESSED DATA
# ============================================================

print("\n" + "=" * 60)
print("PROCESSED FEATURE SAMPLE")
print("=" * 60)

print(
    X_train_scaled.head()
)


print("\n" + "=" * 60)
print("PREPROCESSING PIPELINE FINISHED!")
print("=" * 60)

print(
    "\nThe data is now ready for ML model training."
)