
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_real_features.csv"
)

OUTPUT_DIR = os.path.join(ROOT, "models", "outputs", "unseen_bearings")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Hold out entire bearing IDs for evaluation.
# Repeat with other holdout groups later.
test_bearings = ["K003", "KA05"]

train_df = df[~df["bearing_id"].isin(test_bearings)].copy()
test_df = df[df["bearing_id"].isin(test_bearings)].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Training or testing split is empty.")

metadata = [
    "source_file",
    "bearing_id",
    "label",
    "damage_type"
]

X_train = train_df.drop(columns=metadata, errors="ignore")
X_test = test_df.drop(columns=metadata, errors="ignore")

# Keep only numeric features available in both sets.
X_train = X_train.select_dtypes(include="number")
X_test = X_test.reindex(columns=X_train.columns)

y_train = train_df["label"]
y_test = test_df["label"]

# Learn imputation values from training data only.
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("=" * 60)
print("CYBERSNIFF - UNSEEN BEARING EVALUATION")
print("=" * 60)

print("\nTraining bearing IDs:")
print(sorted(train_df["bearing_id"].unique()))

print("\nHeld-out bearing IDs:")
print(sorted(test_df["bearing_id"].unique()))

print("\nTraining label counts:")
print(y_train.value_counts())

print("\nTest label counts:")
print(y_test.value_counts())

print("\nAccuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
))
print("Class order:", list(model.classes_))

print("\nEvaluation complete.")
print(
    "Interpret results cautiously: the outcome depends on the "
    "verified bearing labels and operating conditions."
)
