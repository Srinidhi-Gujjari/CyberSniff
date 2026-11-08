import os
import joblib
import shap
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    ROOT,
    "models",
    "outputs",
    "cybersniff",
    "cybersniff_model.joblib"
)

# Load trained model bundle
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_columns = bundle["feature_columns"]
medians = bundle["imputation_medians"]

classes = list(model.classes_)

# Load background data for local SHAP explanations.
# This file should contain representative, preprocessed
# training samples with the same 49 feature columns.
BACKGROUND_PATH = os.path.join(
    ROOT,
    "models",
    "outputs",
    "cybersniff",
    "shap_background.csv"
)

background = pd.read_csv(BACKGROUND_PATH)[feature_columns]

background = background.apply(
    pd.to_numeric,
    errors="coerce"
)

background = background.replace(
    [np.inf, -np.inf],
    np.nan
)

background = background.fillna(medians)

explainer = shap.TreeExplainer(model)


def prepare_features(input_data):

    # input_data must already contain the 49 engineered features.
    X = pd.DataFrame([input_data])

    missing = [
        col for col in feature_columns
        if col not in X.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required model features: {missing}"
        )

    X = X[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(medians)

    return X


def explain_prediction(input_data):

    X = prepare_features(input_data)

    prediction = model.predict(X)[0]

    shap_output = explainer(X)

    values = shap_output.values

    class_index = classes.index(prediction)

    if isinstance(values, list):
        class_values = np.asarray(
            values[class_index]
        )[0]

    elif values.ndim == 3:
        class_values = values[0, :, class_index]

    else:
        class_values = values[0]

    explanation = []

    for i, feature in enumerate(feature_columns):

        explanation.append({
            "name": feature,
            "value": float(X.iloc[0, i]),
            "shap_value": float(class_values[i])
        })

    explanation.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True
    )

    return {
        "prediction": str(prediction),
        "model": "CyberSniff trained model",
        "explanation": explanation[:10]
    }