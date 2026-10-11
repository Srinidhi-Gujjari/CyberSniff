
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    ROOT, "data", "processed", "paderborn_real_features.csv"
)

df = pd.read_csv(DATA_PATH)

test_bearings = ["K003", "KA05"]
metadata = ["source_file", "bearing_id", "label", "damage_type"]

train_df = df[~df["bearing_id"].isin(test_bearings)].copy()
test_df = df[df["bearing_id"].isin(test_bearings)].copy()

X_train = train_df.drop(columns=metadata, errors="ignore")
X_test = test_df.drop(columns=metadata, errors="ignore")

X_train = X_train.select_dtypes(include="number")
X_test = X_test.reindex(columns=X_train.columns)

medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, train_df["label"])

test_df["predicted_label"] = model.predict(X_test)
probabilities = model.predict_proba(X_test)

for index, class_name in enumerate(model.classes_):
    test_df[f"prob_{class_name}"] = probabilities[:, index]

print("\n1. ACTUAL LABELS BY BEARING")
print(pd.crosstab(test_df["bearing_id"], test_df["label"]))

print("\n2. PREDICTIONS BY BEARING")
print(pd.crosstab(
    test_df["bearing_id"],
    test_df["predicted_label"]
))

print("\n3. MEAN PREDICTED CLASS PROBABILITIES")
prob_cols = [f"prob_{c}" for c in model.classes_]
print(test_df.groupby("bearing_id")[prob_cols].mean().round(3))

print("\n4. FEATURE MEANS BY ACTUAL LABEL")
feature_data = df.groupby("label")[X_train.columns].mean().T
feature_data["absolute_difference"] = (
    feature_data.iloc[:, 0] - feature_data.iloc[:, 1]
).abs()

print(
    feature_data.sort_values(
        "absolute_difference", ascending=False
    ).head(15).round(3)
)

output_path = os.path.join(
    ROOT, "models", "outputs", "unseen_bearing_diagnostics.csv"
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
test_df.to_csv(output_path, index=False)

print(f"\nDetailed predictions saved to: {output_path}")
