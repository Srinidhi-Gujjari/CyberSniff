
import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REAL_PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_real_features.csv"
)
CYBER_PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_cyber_simulation.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT, "models", "outputs", "model_selection"
)
CANDIDATE_DIR = os.path.join(OUTPUT_DIR, "candidates")

os.makedirs(CANDIDATE_DIR, exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42

EXCLUDED_COLUMNS = {
    "label",
    "source_file",
    "bearing_id",
    "attack_type",
    "sample_id",
    "group_id",
    "scenario",
    "class",
    "target",
}


# ============================================================
# 2. LOAD AND PREPARE THE DATA
# ============================================================

print("\nLoading datasets...")

real = pd.read_csv(REAL_PATH)
cyber = pd.read_csv(CYBER_PATH)

required_real = {"label", "source_file"}
missing_real = required_real - set(real.columns)

if missing_real:
    raise ValueError(
        f"Real dataset is missing required columns: {missing_real}"
    )

# Match the three requested classes.
real["label"] = real["label"].replace({
    "Healthy": "Normal",
})

# Give real and simulated groups separate identities.
real["_split_group"] = (
    "real:" + real["source_file"].fillna("unknown").astype(str)
)

# Match the existing CyberSniff experiment by excluding this scenario.
if "attack_type" in cyber.columns:
    cyber = cyber[
        cyber["attack_type"] != "Replay_Style_Feature_Shuffle"
    ].copy()

# Existing cyber examples represent simulated cyber anomalies.
cyber["label"] = "Cyber_Induced_Anomaly"

if "group_id" in cyber.columns:
    cyber["_split_group"] = (
        "cyber:" + cyber["group_id"].fillna("unknown").astype(str)
    )
elif "source_file" in cyber.columns:
    cyber["_split_group"] = (
        "cyber:" + cyber["source_file"].fillna("unknown").astype(str)
    )
else:
    raise ValueError(
        "Cyber dataset has neither group_id nor source_file. "
        "We need to inspect its columns before splitting safely."
    )

# Only use numeric features that exist in BOTH datasets.
real_numeric = set(
    real.select_dtypes(include=[np.number]).columns
)
cyber_numeric = set(
    cyber.select_dtypes(include=[np.number]).columns
)

feature_columns = sorted(
    (real_numeric & cyber_numeric) - EXCLUDED_COLUMNS
)

if not feature_columns:
    raise ValueError(
        "No common numeric features found in the two datasets."
    )

# Build the combined dataset with identical feature columns.
real_part = real[feature_columns].copy()
real_part["label"] = real["label"].values
real_part["_split_group"] = real["_split_group"].values

cyber_part = cyber[feature_columns].copy()
cyber_part["label"] = cyber["label"].values
cyber_part["_split_group"] = cyber["_split_group"].values

data = pd.concat(
    [real_part, cyber_part],
    ignore_index=True
)

# Convert infinities to missing values; imputers handle missing values.
data[feature_columns] = data[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

# Remove rows with no label or no group identifier.
data = data.dropna(subset=["label", "_split_group"]).copy()
data["_split_group"] = data["_split_group"].astype(str)

expected_classes = {
    "Normal",
    "Physical_Fault",
    "Cyber_Induced_Anomaly",
}

actual_classes = set(data["label"].unique())
missing_classes = expected_classes - actual_classes

if missing_classes:
    raise ValueError(
        f"Missing expected classes: {missing_classes}. "
        f"Classes found: {sorted(actual_classes)}"
    )

# Remove columns that contain no usable values.
feature_columns = [
    col for col in feature_columns
    if data[col].notna().any()
]

if not feature_columns:
    raise ValueError("All candidate features are empty.")

X = data[feature_columns].copy()
y = data["label"].copy()
groups = data["_split_group"].copy()

print(f"Rows: {len(data)}")
print(f"Features: {len(feature_columns)}")
print("\nClass counts:")
print(y.value_counts().to_string())
print(f"\nDistinct split groups: {groups.nunique()}")


# ============================================================
# 3. MAKE ONE GROUPED TRAIN/TEST SPLIT
# ============================================================

print("\nCreating a shared grouped train/test split...")

splitter = GroupShuffleSplit(
    n_splits=200,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

split = None

for train_idx, test_idx in splitter.split(X, y, groups):
    train_classes = set(y.iloc[train_idx])
    test_classes = set(y.iloc[test_idx])

    if (
        train_classes == expected_classes
        and test_classes == expected_classes
    ):
        split = (train_idx, test_idx)
        break

if split is None:
    raise ValueError(
        "Could not find a grouped split containing all three classes "
        "in both training and test sets. Inspect group counts and labels; "
        "do not silently switch to a random row split."
    )

train_idx, test_idx = split

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

if train_groups & test_groups:
    raise RuntimeError("Group leakage detected between train and test.")

print(f"Training rows: {len(train_idx)}")
print(f"Test rows: {len(test_idx)}")
print("Train/test groups are separate.")
print("\nTest-set class counts:")
print(y_test.value_counts().to_string())


# ============================================================
# 4. DEFINE FIVE DIFFERENT ML MODELS
# ============================================================

models = {
    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),

    "ExtraTrees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),

    "HistGradientBoosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_iter=150,
            random_state=RANDOM_STATE,
        )),
    ]),

    "SVM_RBF": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            class_weight="balanced",
        )),
    ]),

    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ]),
}


# ============================================================
# 5. TRAIN AND EVALUATE ALL MODELS ON THE SAME TEST SET
# ============================================================

results = []
fitted_models = {}

print("\nComparing models...")

for model_name, pipeline in models.items():
    print(f"\nTraining {model_name}...")

    fitted = clone(pipeline)
    fitted.fit(X_train, y_train)

    predictions = fitted.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    })

    fitted_models[model_name] = fitted

    candidate_path = os.path.join(
        CANDIDATE_DIR, f"{model_name}.joblib"
    )

    joblib.dump({
        "model": fitted,
        "features": feature_columns,
        "classes": list(fitted.classes_),
        "model_name": model_name,
    }, candidate_path)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


# ============================================================
# 6. SELECT THE HIGHEST-ACCURACY MODEL
# ============================================================

results_df = pd.DataFrame(results)

# Accuracy is the primary selection criterion.
# Macro F1 breaks ties.
results_df = results_df.sort_values(
    ["accuracy", "macro_f1"],
    ascending=[False, False],
).reset_index(drop=True)

comparison_path = os.path.join(
    OUTPUT_DIR, "model_comparison.csv"
)
results_df.to_csv(comparison_path, index=False)

winner_name = results_df.iloc[0]["model"]
winner_accuracy = float(results_df.iloc[0]["accuracy"])

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False, float_format="%.4f"))

print(f"\nSelected model: {winner_name}")
print(f"Held-out accuracy: {winner_accuracy:.4f}")

print("\nClassification report for the selected model:")
test_predictions = fitted_models[winner_name].predict(X_test)

print(classification_report(
    y_test,
    test_predictions,
    labels=[
        "Normal",
        "Physical_Fault",
        "Cyber_Induced_Anomaly",
    ],
    zero_division=0,
))


# ============================================================
# 7. REFIT WINNER ON ALL AVAILABLE DATA AND SAVE IT
# ============================================================

print("\nRefitting the selected model on all available data...")

final_model = clone(models[winner_name])
final_model.fit(X, y)

best_model_path = os.path.join(
    OUTPUT_DIR, "best_model.joblib"
)

joblib.dump({
    "model": final_model,
    "features": feature_columns,
    "classes": list(final_model.classes_),
    "model_name": winner_name,
    "held_out_accuracy": winner_accuracy,
    "selection_metric": "accuracy",
    "comparison_file": comparison_path,
}, best_model_path)

metadata_path = os.path.join(
    OUTPUT_DIR, "best_model_metadata.json"
)

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump({
        "selected_model": winner_name,
        "held_out_accuracy": winner_accuracy,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "classes": list(final_model.classes_),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "split_method": "GroupShuffleSplit",
        "warning": (
            "Cyber examples are simulated. "
            "Held-out accuracy does not prove generalization "
            "to unseen bearings or real cyberattacks."
        ),
    }, file, indent=2)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Best model: {best_model_path}")
print(f"Comparison: {comparison_path}")
print(f"Metadata: {metadata_path}")
print("\nAll five models used the same held-out test rows.")
print("The saved best_model.joblib was refit on all available rows.")