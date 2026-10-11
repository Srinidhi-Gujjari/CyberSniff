
import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_real_features.csv"
)

MODEL_PATH = os.path.join(
    ROOT, "models", "outputs", "paderborn_random_forest.joblib"
)

OUTPUT_DIR = os.path.join(ROOT, "models", "outputs", "shap")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load data and trained model
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

metadata = ["source_file", "bearing_id", "label"]
X = df.drop(columns=metadata)
y = df["label"]

X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(X.median())

# Reproduce the original train/test split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Model classes:", model.classes_)
print("Test samples available:", len(X_test))

# --------------------------------------------------
# 3. Calculate SHAP values
# --------------------------------------------------

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Select the output corresponding to Outer_Race_Fault.
class_index = list(model.classes_).index("Outer_Race_Fault")

values = shap_values.values

# Multiclass/binary SHAP output may have a class dimension.
if values.ndim == 3:
    fault_values = values[:, :, class_index]
    base_value = shap_values.base_values[:, class_index]
else:
    fault_values = values
    base_value = shap_values.base_values

# --------------------------------------------------
# 4. Global feature importance
# --------------------------------------------------

plt.figure()
shap.summary_plot(
    fault_values,
    X_test,
    show=False,
    max_display=15
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "shap_summary.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

plt.figure()
shap.summary_plot(
    fault_values,
    X_test,
    plot_type="bar",
    show=False,
    max_display=15
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "shap_global_importance.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# --------------------------------------------------
# 5. Explain one test prediction
# --------------------------------------------------

sample_index = 0

one_explanation = shap.Explanation(
    values=fault_values[sample_index],
    base_values=base_value[sample_index],
    data=X_test.iloc[sample_index].values,
    feature_names=X_test.columns.tolist()
)

plt.figure()
shap.plots.waterfall(one_explanation, max_display=12, show=False)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "shap_local_explanation.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# --------------------------------------------------
# 6. Save SHAP scores for analysis
# --------------------------------------------------

importance = pd.DataFrame({
    "feature": X_test.columns,
    "mean_absolute_shap": abs(fault_values).mean(axis=0)
}).sort_values("mean_absolute_shap", ascending=False)

importance.to_csv(
    os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"),
    index=False
)

print("\nTop 10 features by mean absolute SHAP value:")
print(importance.head(10).to_string(index=False))

print("\nSHAP files saved to:", OUTPUT_DIR)
print("SHAP EXPLANATION COMPLETED!")
