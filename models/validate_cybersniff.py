
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REAL_PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_real_features.csv"
)
CYBER_PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_cyber_simulation.csv"
)

print("=" * 65)
print("CYBERSNIFF - VALIDATION AND LEAKAGE CHECK")
print("=" * 65)

# 1. Load source datasets
real = pd.read_csv(REAL_PATH)
cyber = pd.read_csv(CYBER_PATH)

# Keep normal and simulated FDI/bias samples; normal attack_type is NaN.
cyber = cyber[
    cyber["attack_type"].isna()
    | cyber["attack_type"].isin([
        "False_Data_Injection",
        "Sensor_Bias"
    ])
].copy()

real["label"] = real["label"].map({
    "Healthy": "Normal",
    "Physical_Fault": "Physical_Fault"
})

real["group_id"] = "real:" + real["source_file"].astype(str)
cyber["group_id"] = "cyber:" + cyber["source_file"].astype(str)

df = pd.concat([real, cyber], ignore_index=True)

metadata = [
    "source_file", "bearing_id", "label",
    "attack_type", "sample_id", "group_id"
]

X = df.drop(columns=metadata, errors="ignore")
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Remove columns with no usable values.
X = X.dropna(axis=1, how="all")
y = df["label"]
groups = df["group_id"]

print("\nDataset shape:", df.shape)
print("\nLabels:")
print(y.value_counts())

print("\nSource bearing distribution:")
print(pd.crosstab(df["bearing_id"], y))

print("\nAttack types:")
print(df["attack_type"].value_counts(dropna=False))

# 2. Check exact duplicate feature rows across labels.
check = X.copy()
check["label"] = y.to_numpy()

duplicate_conflicts = (
    check.groupby(list(X.columns), dropna=False)["label"]
    .nunique()
    .gt(1)
    .sum()
)

print(
    "\nDistinct feature patterns associated with multiple labels:",
    duplicate_conflicts
)

# 3. Grouped split by source file.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=7
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Fit imputation only on training data.
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())

# 4. Train an independent validation model.
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=7,
    n_jobs=-1
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nValidation accuracy:", round(
    accuracy_score(y_test, pred) * 100, 2
), "%")

print("\nClassification report:")
print(classification_report(
    y_test,
    pred,
    zero_division=0
))

print("\nConfusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(
    y_test,
    pred,
    labels=model.classes_
))
print("\nClass order:", list(model.classes_))

print("\nValidation complete.")
print(
    "Note: grouping prevents source-file variants crossing the split, "
    "but does not establish generalization to new bearing IDs or "
    "real cyberattacks."
)
