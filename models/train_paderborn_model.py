
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "paderborn_real_features.csv"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load the real dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("CYBERSNIFF - PADERBORN RANDOM FOREST MODEL")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())

# --------------------------------------------------
# 3. Prepare features and target
# --------------------------------------------------

# Metadata must not be used as model features.
metadata = ["source_file", "bearing_id", "label"]

X = df.drop(columns=metadata)
y = df["label"]

# Replace infinite values and handle any missing features.
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(X.median())

# --------------------------------------------------
# 4. Split into training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 5. Train Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 6. Evaluate the model
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# --------------------------------------------------
# 7. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot(values_format="d")
plt.title("CyberSniff - Paderborn Classification")
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()

# --------------------------------------------------
# 8. Feature importance
# --------------------------------------------------

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv"
)

importance.to_csv(importance_path, index=False)

print("\nTop 10 important features:")
print(importance.head(10).to_string(index=False))

# Plot the top 10 features.
top_features = importance.head(10).sort_values("importance")

plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Feature Importance")
plt.title("Top 10 Sensor Features - CyberSniff")
plt.tight_layout()

feature_plot_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.png"
)

plt.savefig(feature_plot_path, dpi=300)
plt.close()

# --------------------------------------------------
# 9. Save the trained model
# --------------------------------------------------

import joblib

model_path = os.path.join(OUTPUT_DIR, "paderborn_random_forest.joblib")
joblib.dump(model, model_path)

print("\nSaved model:", model_path)
print("Saved confusion matrix:", cm_path)
print("Saved feature importance:", importance_path)
print("Saved feature plot:", feature_plot_path)

print("\nTRAINING COMPLETED!")
