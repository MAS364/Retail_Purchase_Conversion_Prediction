"""
test_api.py — Tests for the FastAPI prediction API.

Run with:  pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

VALID_SESSION = {
    "price": 150.0, "total_time_spent": 120,
    "session_length": 6, "interaction_count": 6,
    "view_count": 3, "click_count": 2,
    "wishlist_count": 0, "add_to_cart_count": 1,
    "avg_time_per_interaction": 20.0,
    "cart_to_view_ratio": 0.33, "click_to_view_ratio": 0.67,
    "has_cart_action": 1, "has_wishlist_action": 0,
    "hour": 14, "day_of_week": 2, "month": 3, "is_weekend": 0,
    "category": "electronics", "brand": "apple",
    "channel": "web", "device_type": "desktop",
    "region": "uk", "traffic_source": "organic",
}


class TestHealthEndpoints:

    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert isinstance(body["models_loaded"], list)

    def test_models(self):
        r = client.get("/models")
        assert r.status_code == 200
        assert "available_models" in r.json()

    def test_root_redirects_to_docs(self):
        r = client.get("/", follow_redirects=False)
        assert r.status_code == 307
        assert "/docs" in r.headers["location"]

    def test_openapi_schema(self):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        assert "paths" in r.json()


class TestPredictEndpoint:

    def test_valid_prediction(self):
        r = client.post("/predict", json=VALID_SESSION)
        assert r.status_code == 200
        body = r.json()
        assert body["prediction"] in (0, 1)
        assert body["label"] in ("purchase", "no_purchase")
        assert body["model"] == "mlp"

    def test_select_model(self):
        r = client.post("/predict?model=decision_tree", json=VALID_SESSION)
        assert r.status_code == 200
        assert r.json()["model"] == "decision_tree"

    def test_invalid_model(self):
        r = client.post("/predict?model=nonexistent", json=VALID_SESSION)
        assert r.status_code == 400

    def test_missing_feature(self):
        bad = {k: v for k, v in VALID_SESSION.items() if k != "price"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422  # Pydantic validation error

    def test_invalid_type(self):
        bad = {**VALID_SESSION, "price": "not_a_number"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_out_of_range(self):
        bad = {**VALID_SESSION, "hour": 25}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


class TestBatchEndpoint:

    def test_batch_prediction(self):
        r = client.post("/predict/batch", json={
            "sessions": [VALID_SESSION, VALID_SESSION],
        })
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2
        assert len(body["predictions"]) == 2

    def test_batch_with_model(self):
        r = client.post("/predict/batch", json={
            "sessions": [VALID_SESSION],
            "model": "naive_bayes",
        })
        assert r.status_code == 200
        assert r.json()["predictions"][0]["model"] == "naive_bayes"

    def test_empty_batch(self):
        r = client.post("/predict/batch", json={"sessions": []})
        assert r.status_code == 422  # min_length validation
