
# CyberSniff Backend

FastAPI backend for the CyberSniff sensor classification project.

## Model

The API loads the existing trained model from:

`models/outputs/model_selection/best_model.joblib`

The model metadata is read from:

`models/outputs/model_selection/best_model_metadata.json`

The model expects 49 sensor features in the exact order recorded in its metadata.

## API Endpoints

- `GET /` — welcome message
- `GET /health` — API and model status
- `POST /predict` — classify a sensor feature record

## Setup

From the CyberSniff project root, activate the project environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```powershell
python -m pip install -r .\backend\requirements.txt
```

## Run the API

```powershell
Set-Location .\backend
python -m uvicorn app.main:app --reload
```

Open the interactive API documentation at:

http://127.0.0.1:8000/docs

## Run Tests

From the `backend` directory:

```powershell
python -m pytest -v
```

## Important limitation

The model metadata reports a held-out accuracy of approximately 88.7%. Cyber examples were simulated, and this result does not establish generalization to unseen bearings or real cyberattacks.
