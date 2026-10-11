import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "paderborn_real_features.csv"
ARTIFACT = ROOT / "models" / "artifacts"

FEATURES = [
    "vibration_1_peak_to_peak",
    "vibration_1_min",
    "vibration_1_max",
    "vibration_1_kurtosis",
]

df = pd.read_csv(DATA)
X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = df["label"]

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

model_path = ARTIFACT / "physical_fault_histgradientboosting_4_features.joblib"
metadata_path = ARTIFACT / "physical_fault_histgradientboosting_4_features.json"

joblib.dump(model, model_path)

metadata = {
    "model_type": "HistGradientBoostingClassifier",
    "features": FEATURES,
    "target": "label",
    "classes": [str(c) for c in model.classes_],
    "training_rows": int(len(df)),
    "training_data": "data/processed/paderborn_real_features.csv",
    "training_scope": "All available rows; not a held-out evaluation model",
    "sklearn_version": sklearn.__version__,
    "validation_warning": (
        "Bearing identity and class are confounded. "
        "Grouped cross-validation results are experimental, "
        "and production generalization is not established."
    ),
}

metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("Four-feature experimental model trained.")
print("Model:", model_path)
print("Metadata:", metadata_path)
print("Features:", FEATURES)
print("Training rows:", len(df))
print("Classes:", list(model.classes_))
print("WARNING: This artifact was fitted on all rows; "
      "it is not a held-out evaluation model.")
