import auth
import pytest
from fastapi import HTTPException


def test_rejects_when_no_key_configured(monkeypatch):
    monkeypatch.setattr(auth, "API_KEY", None)
    with pytest.raises(HTTPException) as exc:
        auth.verify_api_key(x_api_key="anything")
    assert exc.value.status_code == 503


def test_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(auth, "API_KEY", "correct-key")
    with pytest.raises(HTTPException) as exc:
        auth.verify_api_key(x_api_key="wrong-key")
    assert exc.value.status_code == 401


def test_rejects_missing_key(monkeypatch):
    monkeypatch.setattr(auth, "API_KEY", "correct-key")
    with pytest.raises(HTTPException) as exc:
        auth.verify_api_key(x_api_key=None)
    assert exc.value.status_code == 401


def test_accepts_correct_key(monkeypatch):
    monkeypatch.setattr(auth, "API_KEY", "correct-key")
    auth.verify_api_key(x_api_key="correct-key")  # should not raise