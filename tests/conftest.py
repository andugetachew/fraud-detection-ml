import sys
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "training"))
sys.path.append(str(ROOT / "serving"))

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

VALID_PAYLOAD = {col: 0.0 for col in FEATURE_COLUMNS}
VALID_PAYLOAD.update({"Amount": 149.62, "V3": 2.54, "V14": -0.31})


class FakeModel:
    """Stands in for the trained XGBoost model — fixed, predictable output
    so endpoint tests check wiring/status codes, not model accuracy (that's
    already covered by the real end-to-end tests against the live API)."""

    def predict_proba(self, X):
        return np.array([[0.99, 0.01]])


class FakeScaler:
    def transform(self, X):
        return X.values


class FakeVersion:
    version = "test-version"
    artifact_path = "unused"
    metrics = json.dumps({"precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0})
    reference_data_path = "unused"
    created_at = "2026-01-01T00:00:00"


class FakeExplanation:
    def __init__(self, n_features):
        self.values = np.zeros((1, n_features))


class FakeExplainer:
    def __init__(self, model):
        pass

    def __call__(self, row):
        return FakeExplanation(row.shape[1])


@pytest.fixture
def api_client(monkeypatch):
    """Builds a TestClient for the real FastAPI app, with the DB, model
    artifact, Celery broker, and SHAP explainer all mocked at the module
    boundary — so this exercises the actual routing/auth/validation logic
    in main.py and predict.py without needing live Neon, Redis, or a
    trained model file."""
    import joblib
    import shap
    import pandas
    import registry
    from prometheus_client import REGISTRY

    monkeypatch.setattr(joblib, "load", lambda path: {"model": FakeModel(), "scaler": FakeScaler()})
    monkeypatch.setattr(registry, "get_active_version", lambda: FakeVersion())
    monkeypatch.setattr(registry, "log_prediction", lambda *a, **k: None)
    monkeypatch.setattr(registry, "get_recent_predictions", lambda *a, **k: [])
    monkeypatch.setattr(shap, "TreeExplainer", FakeExplainer)
    monkeypatch.setattr(pandas, "read_csv", lambda path: pandas.DataFrame())
    monkeypatch.setenv("API_KEY", "test-api-key")

    # main.py re-registers Prometheus metrics on every import. The default
    # registry is a process-wide global, so without clearing it here, the
    # second test's reimport collides with the first test's already-
    # registered collectors (DuplicateTimeseries).
    REGISTRY._collector_to_names.clear()
    REGISTRY._names_to_collectors.clear()

    # Force a fresh import so each module picks up the patches above,
    # rather than reusing another test's already-constructed ModelStore.
    for mod_name in ["config", "auth", "tasks", "predict", "main"]:
        sys.modules.pop(mod_name, None)

    import main as main_module
    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as client:
        yield client