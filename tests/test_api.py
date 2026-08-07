from conftest import VALID_PAYLOAD


def test_health_is_public(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_requires_api_key(api_client):
    resp = api_client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 401


def test_predict_rejects_wrong_api_key(api_client):
    resp = api_client.post(
        "/predict", json=VALID_PAYLOAD, headers={"X-API-Key": "wrong-key"}
    )
    assert resp.status_code == 401


def test_predict_returns_prediction_with_valid_key(api_client):
    resp = api_client.post(
        "/predict", json=VALID_PAYLOAD, headers={"X-API-Key": "test-api-key"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"is_fraud", "fraud_probability", "model_version"}
    assert body["model_version"] == "test-version"


def test_predict_rejects_missing_field(api_client):
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "V14"}
    resp = api_client.post(
        "/predict", json=incomplete, headers={"X-API-Key": "test-api-key"}
    )
    assert resp.status_code == 422


def test_predict_rejects_negative_amount(api_client):
    bad_payload = {**VALID_PAYLOAD, "Amount": -5.0}
    resp = api_client.post(
        "/predict", json=bad_payload, headers={"X-API-Key": "test-api-key"}
    )
    assert resp.status_code == 422


def test_model_status_is_public(api_client):
    resp = api_client.get("/model/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "test-version"


def test_predict_explain_requires_api_key(api_client):
    resp = api_client.post("/predict/explain", json=VALID_PAYLOAD)
    assert resp.status_code == 401


def test_predict_explain_returns_top_features(api_client):
    resp = api_client.post(
        "/predict/explain", json=VALID_PAYLOAD, headers={"X-API-Key": "test-api-key"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "top_features" in body
    assert len(body["top_features"]) <= 10


def test_model_drift_requires_api_key(api_client):
    resp = api_client.get("/model/drift")
    assert resp.status_code == 401


def test_model_drift_with_no_recent_predictions(api_client):
    resp = api_client.get("/model/drift", headers={"X-API-Key": "test-api-key"})
    assert resp.status_code == 200
    assert resp.json()["sample_size"] == 0