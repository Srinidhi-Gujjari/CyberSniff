
from pathlib import Path
import json
import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# Locate the CyberSniff project directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "outputs" / "model_selection"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "best_model_metadata.json"


# Load the model metadata.
if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

with METADATA_PATH.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

FEATURES = metadata["features"]
CLASSES = metadata["classes"]

if len(FEATURES) != metadata["feature_count"]:
    raise ValueError("Feature count does not match metadata.")


# Load the already-trained model.
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")


saved_artifact = joblib.load(MODEL_PATH)

# The saved file is a dictionary containing the trained pipeline
# and its associated metadata.
if isinstance(saved_artifact, dict) and "model" in saved_artifact:
    model = saved_artifact["model"]
else:
    model = saved_artifact

# Check that the loaded object supports predictions.
if not hasattr(model, "predict"):
    raise TypeError(
        "The saved model artifact does not contain a usable prediction model."
    )

# Verify the artifact's feature order matches the metadata.
artifact_features = (
    saved_artifact.get("features")
    if isinstance(saved_artifact, dict)
    else None
)

if artifact_features is not None and artifact_features != FEATURES:
    raise ValueError(
        "Feature order in the model artifact does not match the metadata."
    )



# Create the FastAPI application.

# Create the FastAPI application.
app = FastAPI(
    title="CyberSniff API",
    description="Predicts classes from sensor features using the trained model.",
    version="1.0.0",
)

# Allow the local frontend to communicate with the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# Define the prediction request format.
class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="All 49 expected sensor features and their numeric values.",
    )


@app.get("/")
def home():
    return {
        "message": "Welcome to the CyberSniff API",
        "health": "/health",
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": metadata["selected_model"],
        "feature_count": len(FEATURES),
        "classes": CLASSES,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    supplied_features = request.features

    # Find missing or unexpected feature names.
    missing = [name for name in FEATURES if name not in supplied_features]
    extra = [name for name in supplied_features if name not in FEATURES]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Feature names do not match the trained model.",
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    # Reject NaN and infinity.
    invalid = [
        name
        for name in FEATURES
        if not math.isfinite(supplied_features[name])
    ]

    if invalid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Feature values must be finite numbers.",
                "invalid_features": invalid,
            },
        )

    # Arrange the 49 features in the exact training order.
    input_data = pd.DataFrame(
        [[supplied_features[name] for name in FEATURES]],
        columns=FEATURES,
    )

    try:
        prediction = model.predict(input_data)[0]

        
        result = {
            "prediction": str(prediction),
            "model_name": metadata["selected_model"],
            "available_classes": CLASSES,
        }


        # Return class probabilities when supported.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            result["probabilities"] = {
                str(label): float(probability)
                for label, probability in zip(
                    model.classes_, probabilities
                )
            }

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The model could not generate a prediction.",
        ) from exc
