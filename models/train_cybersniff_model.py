
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REAL_PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_real_features.csv"
)
CYBER_PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_cyber_simulation.csv"
)
OUTPUT_DIR = os.path.join(ROOT, "models", "outputs", "cybersniff")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 1. Load and label real measurements
# --------------------------------------------------

real = pd.read_csv(REAL_PATH)

real["label"] = real["label"].map({
    "Healthy": "Normal",
    "Physical_Fault": "Physical_Fault"
})

real["group_id"] = "real:" + real["source_file"].astype(str)

# --------------------------------------------------
# 2. Load simulated cyber scenarios
# --------------------------------------------------

cyber = pd.read_csv(CYBER_PATH)

# Exclude the replay-style shuffle in this first experiment.
cyber = cyber[
    cyber["attack_type"] != "Replay_Style_Feature_Shuffle"
].copy()

cyber["group_id"] = "cyber:" + cyber["source_file"].astype(str)

# --------------------------------------------------
# 3. Combine datasets
# --------------------------------------------------

df = pd.concat([real, cyber], ignore_index=True)

metadata = [
    "source_file",
    "bearing_id",
    "label",
    "attack_type",
    "sample_id",
    "group_id"
]

feature_columns = [
    col for col in df.columns if col not in metadata
]

X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Remove features that are entirely missing.
X = X.dropna(axis=1, how="all")
feature_columns = X.columns.tolist()

y = df["label"]
groups = df["group_id"]

# --------------------------------------------------
# 4. Grouped train/test split
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Fit imputation values on training data only.
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

print("=" * 65)
print("CYBERSNIFF - THREE-CLASS EXPERIMENT")
print("=" * 65)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print("\nTraining class distribution:")
print(y_train.value_counts())
print("\nTesting class distribution:")
print(y_test.value_counts())

# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# --------------------------------------------------
# 6. Evaluate
# --------------------------------------------------

print("\nAccuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=["Normal", "Physical_Fault", "Cyber_Induced_Anomaly"],
    zero_division=0
))

labels = [
    "Normal",
    "Physical_Fault",
    "Cyber_Induced_Anomaly"
]

cm = confusion_matrix(y_test, predictions, labels=labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)
disp.plot(values_format="d", xticks_rotation=25)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# 7. Save model and feature metadata
# --------------------------------------------------

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "imputation_medians": medians,
        "classes": model.classes_.tolist()
    },
    os.path.join(OUTPUT_DIR, "cybersniff_model.joblib")
)

print("\nSaved confusion matrix and model to:")
print(OUTPUT_DIR)
print("\nTHREE-CLASS EXPERIMENT COMPLETED!")
print("Remember: this is a synthetic cyber-simulation baseline.")
