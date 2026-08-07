import json

import pandas as pd

from celery_app import app
from train import main as run_training
from registry import get_active_version, get_recent_predictions, log_prediction
from drift import compute_drift
from logging_config import get_logger

logger = get_logger(__name__)


@app.task(name="tasks.train_task")
def train_task(activate: bool = False) -> dict:
    """Runs the training pipeline as a background job instead of blocking
    whatever triggers it (a cron hit, a future 'retrain' API call, etc.)."""
    return run_training(activate=activate)


@app.task(name="tasks.log_prediction_task")
def log_prediction_task(model_version: str, features: dict, fraud_probability: float, is_fraud: bool):
    """Writes the prediction log row to Neon. Runs on the worker, off the
    request path — a load test showed /predict blocking on this DB
    round-trip was the actual bottleneck (low CPU, high tail latency —
    classic I/O wait), not model inference itself."""
    log_prediction(model_version, features, fraud_probability, is_fraud)


@app.task(name="tasks.check_drift_task")
def check_drift_task() -> dict:
    """Compares recent live prediction inputs against the active model's
    reference sample via Evidently. Logs a warning if drift is flagged —
    this is the check that tells you "retrain" is actually warranted,
    rather than retraining blindly on a schedule."""
    active = get_active_version()
    if active is None:
        logger.warning("No active model version — skipping drift check")
        return {"skipped": True}

    reference_df = pd.read_csv(active.reference_data_path)
    recent = get_recent_predictions(active.version, limit=500)
    current_df = pd.DataFrame([json.loads(r.features) for r in recent])

    result = compute_drift(reference_df, current_df)
    if result["is_drifting"]:
        logger.warning(
            "Drift detected",
            extra={"extra_fields": {"version": active.version, **result}},
        )
    else:
        logger.info(
            "No drift detected",
            extra={"extra_fields": {"version": active.version, **result}},
        )
    return result



app.conf.beat_schedule = {
    "check-drift-hourly": {
        "task": "tasks.check_drift_task",
        "schedule": 60 * 60,
    },
}

