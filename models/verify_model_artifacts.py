import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "models" / "artifacts"
DATA = ROOT / "data" / "processed" / "paderborn_real_features.csv"

df = pd.read_csv(DATA)

candidates = [
    ("Random Forest", "physical_fault_random_forest.joblib",
     "physical_fault_features.json", None),
    ("HGB 8-feature", "physical_fault_histgradientboosting_8_features.joblib",
     "physical_fault_histgradientboosting_8_features.json", None),
    ("HGB 4-feature", "physical_fault_histgradientboosting_4_features.joblib",
     "physical_fault_histgradientboosting_4_features.json", None),
]

for name, model_file, metadata_file, _ in candidates:
    print(f"\n--- {name} ---")
    model_path = ARTIFACTS / model_file
    metadata_path = ARTIFACTS / metadata_file

    if not model_path.exists():
        print(f"MISSING MODEL: {model_path.name}")
        continue
    if not metadata_path.exists():
        print(f"MISSING METADATA: {metadata_path.name}")
        continue

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if isinstance(metadata, list):
        features = metadata
    else:
        features = metadata.get("features", [])

    missing = [f for f in features if f not in df.columns]
    print("Artifact loaded:", type(model).__name__)
    print("Feature count in metadata:", len(features))
    print("Missing dataset features:", missing)
    print("Classes:", list(getattr(model, "classes_", [])))

    if missing or not features:
        print("CHECK FAILED: metadata/features need review")
        continue

    X = df[features].apply(pd.to_numeric, errors="coerce").head(5)
    probabilities = model.predict_proba(X)
    print("5-row probability output shape:", probabilities.shape)
    print("Smoke test: PASS")
