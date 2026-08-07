"""
Regression test for a real bug: CELERY_TASK_ALWAYS_EAGER was accidentally
set to `not DEBUG` instead of `DEBUG`. This silently made task execution
synchronous in the exact opposite case intended — DEBUG=true (local
convenience) ran tasks for real via the broker, while DEBUG=false
(meant to enable real async via the worker) made everything run eager/
inline instead, causing every /predict call to block on a synchronous
Neon round-trip. No test caught it because config.py values are only
ever exercised at import time — this test exercises the actual formula.
"""
import importlib


def test_debug_true_means_eager(monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    import config
    importlib.reload(config)
    assert config.CELERY_TASK_ALWAYS_EAGER is True


def test_debug_false_means_not_eager(monkeypatch):
    monkeypatch.setenv("DEBUG", "false")
    import config
    importlib.reload(config)
    assert config.CELERY_TASK_ALWAYS_EAGER is False