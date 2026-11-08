
from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "outputs" / "model_selection"

MODEL_PATH = MODEL_DIR / "best_model.joblib"

METADATA_PATH = MODEL_DIR / "best_model_metadata.json"

# Representative samples from the training dataset.
# This file must contain the same 49 model features.
BACKGROUND_PATH = MODEL_DIR / "shap_background.csv"


# ============================================================
# 2. LOAD MODEL METADATA
# ============================================================

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata not found: {METADATA_PATH}"
    )

with METADATA_PATH.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

FEATURES = metadata["features"]
CLASSES = metadata["classes"]

if len(FEATURES) != metadata["feature_count"]:
    raise ValueError(
        "Feature count does not match metadata."
    )


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

saved_artifact = joblib.load(MODEL_PATH)

if isinstance(saved_artifact, dict) and "model" in saved_artifact:
    model = saved_artifact["model"]
else:
    model = saved_artifact

if not hasattr(model, "predict"):
    raise TypeError(
        "Saved artifact does not contain a usable prediction model."
    )

artifact_features = (
    saved_artifact.get("features")
    if isinstance(saved_artifact, dict)
    else None
)

if artifact_features is not None and artifact_features != FEATURES:
    raise ValueError(
        "Feature order in model artifact does not match metadata."
    )

MODEL_NAME = metadata.get(
    "selected_model",
    type(model).__name__
)

MODEL_CLASSES = list(
    getattr(model, "classes_", CLASSES)
)

print("CyberSniff model loaded:", MODEL_NAME)
print("Model classes:", MODEL_CLASSES)
print("Number of features:", len(FEATURES))


# ============================================================
# 4. LOAD SHAP BACKGROUND DATA
# ============================================================

background_data = None
explainer = None
EXPLANATION_OUTPUT_TYPE = None

if BACKGROUND_PATH.exists():

    print("Loading SHAP background dataset...")

    background_data = pd.read_csv(BACKGROUND_PATH)

    missing_background_features = [
        name for name in FEATURES
        if name not in background_data.columns
    ]

    if missing_background_features:
        raise ValueError(
            "SHAP background dataset is missing features: "
            + str(missing_background_features)
        )

    background_data = background_data[FEATURES].copy()

    background_data = background_data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    background_data = background_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Use a small representative background dataset
    # to control explanation computation time.
    background_data = background_data.dropna()

    if len(background_data) > 25:
        background_data = background_data.sample(
            n=25,
            random_state=42
        )

    if len(background_data) > 0:

        # Use probabilities when supported by the model.
        if hasattr(model, "predict_proba"):
            EXPLANATION_OUTPUT_TYPE = "probability"

            def shap_prediction_function(data):
                frame = pd.DataFrame(
                    data,
                    columns=FEATURES
                )
                return model.predict_proba(frame)

        # Otherwise explain the model's decision scores.
        elif hasattr(model, "decision_function"):
            EXPLANATION_OUTPUT_TYPE = "decision_score"

            def shap_prediction_function(data):
                frame = pd.DataFrame(
                    data,
                    columns=FEATURES
                )
                return model.decision_function(frame)

        else:
            shap_prediction_function = None

        if shap_prediction_function is not None:

            masker = shap.maskers.Independent(
                background_data
            )

            explainer = shap.Explainer(
                shap_prediction_function,
                masker,
                algorithm="permutation"
            )

            print("SHAP explainer initialized.")
            print(
                "Explanation output:",
                EXPLANATION_OUTPUT_TYPE
            )

        else:
            print(
                "SHAP unavailable: model has neither "
                "predict_proba nor decision_function."
            )

    else:
        print(
            "SHAP unavailable: background dataset has no valid rows."
        )

else:
    print(
        "SHAP background file not found. "
        "Predictions will work without local explanations."
    )


# ============================================================
# 5. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="CyberSniff API",
    description=(
        "Industrial anomaly classification with "
        "SHAP-based explainability."
    ),
    version="2.0.0",
)


# ============================================================
# 6. CORS CONFIGURATION
# ============================================================

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


# ============================================================
# 7. REQUEST FORMAT
# ============================================================

class PredictionRequest(BaseModel):

    features: dict[str, float] = Field(
        ...,
        description=(
            "All 49 expected sensor features "
            "and their numeric values."
        )
    )


# ============================================================
# 8. PREPARE INPUT FEATURES
# ============================================================

def prepare_input(supplied_features):

    missing = [
        name for name in FEATURES
        if name not in supplied_features
    ]

    extra = [
        name for name in supplied_features
        if name not in FEATURES
    ]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Feature names do not match "
                    "the trained model."
                ),
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    invalid = [
        name for name in FEATURES
        if not math.isfinite(supplied_features[name])
    ]

    if invalid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Feature values must be finite numbers."
                ),
                "invalid_features": invalid,
            },
        )

    input_data = pd.DataFrame(
        [[supplied_features[name] for name in FEATURES]],
        columns=FEATURES,
    )

    return input_data


# ============================================================
# 9. GENERATE LOCAL SHAP EXPLANATION
# ============================================================

def generate_local_explanation(input_data, prediction):

    if explainer is None:
        return None

    try:

        # Generate SHAP values for this individual prediction.
        shap_result = explainer(
            input_data,
            max_evals=(2 * len(FEATURES)) + 1
        )

        values = np.asarray(shap_result.values)
        base_values = np.asarray(shap_result.base_values)

        class_index = MODEL_CLASSES.index(prediction)

        # ----------------------------------------------------
        # Handle multiclass SHAP output:
        # (samples, features, classes)
        # ----------------------------------------------------

        if values.ndim == 3:

            class_values = values[0, :, class_index]

            if base_values.ndim >= 2:
                base_value = float(
                    base_values[0, class_index]
                )
            else:
                base_value = float(
                    base_values[class_index]
                )

        # ----------------------------------------------------
        # Handle single-output SHAP:
        # (samples, features)
        # ----------------------------------------------------

        elif values.ndim == 2:

            class_values = values[0]

            if base_values.size == 1:
                base_value = float(base_values.reshape(-1)[0])
            else:
                base_value = float(
                    base_values.reshape(-1)[class_index]
                )

            # Binary decision_function explains the margin
            # for classes_[1]. Reverse the sign when explaining
            # classes_[0].
            if (
                EXPLANATION_OUTPUT_TYPE == "decision_score"
                and len(MODEL_CLASSES) == 2
                and class_index == 0
            ):
                class_values = -class_values
                base_value = -base_value

        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {values.shape}"
            )

        # ----------------------------------------------------
        # Prepare feature contributions
        # ----------------------------------------------------

        feature_results = []

        for index, feature_name in enumerate(FEATURES):

            feature_results.append({
                "name": feature_name,
                "value": float(
                    input_data.iloc[0, index]
                ),
                "shap_value": float(
                    class_values[index]
                ),
            })

        # Sort by absolute contribution.
        feature_results.sort(
            key=lambda item: abs(item["shap_value"]),
            reverse=True
        )

        return {
            "class_name": str(prediction),
            "output_type": EXPLANATION_OUTPUT_TYPE,
            "base_value": base_value,
            "features": feature_results[:10],
        }

    except Exception as exc:

        print("SHAP explanation error:", str(exc))

        return None


# ============================================================
# 10. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Welcome to the CyberSniff API",
        "health": "/health",
        "documentation": "/docs",
    }


# ============================================================
# 11. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "feature_count": len(FEATURES),
        "classes": CLASSES,
        "shap_available": explainer is not None,
    }


# ============================================================
# 12. PREDICTION + LOCAL SHAP ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    supplied_features = request.features

    # Prepare and validate input.
    input_data = prepare_input(supplied_features)

    try:

        # ----------------------------------------------------
        # A. MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]

        result = {
            "prediction": str(prediction),
            "model_name": MODEL_NAME,
            "available_classes": CLASSES,
        }

        # ----------------------------------------------------
        # B. CLASS PROBABILITIES
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            result["probabilities"] = {
                str(label): float(probability)
                for label, probability in zip(
                    MODEL_CLASSES,
                    probabilities
                )
            }

        # ----------------------------------------------------
        # C. LOCAL SHAP EXPLANATION
        # ----------------------------------------------------

        shap_explanation = generate_local_explanation(
            input_data,
            prediction
        )

        result["shap_explanation"] = (
            {
                "features": [],
                "available": False,
                "message": (
                    "Local SHAP explanation is unavailable."
                ),
            }
            if shap_explanation is None
            else {
                **shap_explanation,
                "available": True,
            }
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:

        print("Prediction error:", str(exc))

        raise HTTPException(
            status_code=500,
            detail=(
                "The model could not generate "
                "the prediction."
            ),
        ) from exc
