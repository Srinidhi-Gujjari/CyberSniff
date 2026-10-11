import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "paderborn_real_features.csv"
OUT = ROOT / "models" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "vibration_1_kurtosis",
    "vibration_1_peak_to_peak",
    "vibration_1_min",
    "force_std",
    "vibration_1_mean",
    "vibration_1_max",
    "force_kurtosis",
    "vibration_1_std",
]

df = pd.read_csv(DATA)
required = FEATURES + ["label", "bearing_id"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = df["label"]

if y.isna().any() or set(y.unique()) != {"Healthy", "Physical_Fault"}:
    raise ValueError("Expected Healthy and Physical_Fault labels.")

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        l2_regularization=1.0,
        random_state=42,
    ),
)
model.fit(X, y)

model_path = OUT / "physical_fault_histgradientboosting_8_features.joblib"
features_path = OUT / "physical_fault_histgradientboosting_8_features.json"

joblib.dump(model, model_path)
features_path.write_text(json.dumps(FEATURES, indent=2), encoding="utf-8")

print("Saved separate experimental model:", model_path)
print("Saved feature list:", features_path)
print("Features:", len(FEATURES))
print("Training rows:", len(df))
print("Training labels:")
print(y.value_counts().to_string())
print("Bearings:", sorted(df["bearing_id"].unique()))
print("WARNING: This artifact is experimental, not production-validated.")
