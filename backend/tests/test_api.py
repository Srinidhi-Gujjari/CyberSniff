
from fastapi.testclient import TestClient

from app.main import app, FEATURES, CLASSES


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()
    assert data["message"] == "Welcome to the CyberSniff API"
    assert data["health"] == "/health"
    assert data["documentation"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["feature_count"] == 49
    assert data["model_name"] == "SVM_RBF"
    assert data["classes"] == CLASSES


def test_prediction_with_all_features():
    # Synthetic values for an API smoke test only.
    # This does not represent a real sensor observation.
    sample_features = {name: 0.0 for name in FEATURES}

    response = client.post(
        "/predict",
        json={"features": sample_features},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in CLASSES
    assert data["model_name"] == "SVM_RBF"
    assert response.json()["available_classes"] == CLASSES


def test_prediction_rejects_missing_features():
    response = client.post(
        "/predict",
        json={"features": {"force_mean": 0.0}},
    )

    assert response.status_code == 422

    detail = response.json()["detail"]
    assert "missing_features" in detail
    assert len(detail["missing_features"]) > 0


def test_prediction_rejects_unexpected_features():
    sample_features = {name: 0.0 for name in FEATURES}
    sample_features["unexpected_feature"] = 1.0

    response = client.post(
        "/predict",
        json={"features": sample_features},
    )

    assert response.status_code == 422

    detail = response.json()["detail"]
    assert "unexpected_feature" in detail["unexpected_features"]



def test_prediction_rejects_non_finite_values():
    sample_features = {name: 0.0 for name in FEATURES}
    sample_features["force_mean"] = "NaN"

    response = client.post(
        "/predict",
        json={"features": sample_features},
    )

    assert response.status_code == 422


def test_cors_allows_local_frontend():
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
