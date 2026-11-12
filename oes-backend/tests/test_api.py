import pytest
from fastapi.testclient import TestClient
from oes_api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_catalogue_marks_only_ols_as_implemented():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    models = response.json()["models"]
    assert next(m for m in models if m["id"] == "ols")["status"] == "implemented"
    assert all(m["status"] == "planned" for m in models if m["id"] != "ols")

def test_ols_endpoint():
    response = client.post("/api/v1/estimate/ols", json={
        "data": {
            "y": [2, 5, 8, 11, 14],
            "x": [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]]
        },
        "variable_names": ["const", "trend"],
        "covariance": "HC1"
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["coefficients"][0]["estimate"] == pytest.approx(2.0)
    assert payload["coefficients"][1]["estimate"] == pytest.approx(3.0)
