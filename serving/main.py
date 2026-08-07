import json
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from schemas import (
    TransactionInput,
    PredictionResponse,
    ModelStatusResponse,
    ExplanationResponse,
)
from predict import store
from auth import verify_api_key

sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
from registry import get_recent_predictions  # noqa: E402
from drift import compute_drift  # noqa: E402
from logging_config import get_logger  # noqa: E402
from celery_app import app as celery_app  # noqa: E402

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Fraud Detection API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Standard request count/latency/status-code metrics, auto-instrumented.
# /metrics is left unauthenticated — a Prometheus scraper hitting it needs
# network-level access control (e.g. internal-only ingress), not an API key
# meant for the business endpoints.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# Business-level metric beyond generic HTTP instrumentation — lets you graph
# fraud rate over time, not just request volume.
PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total", "Total predictions made, by outcome", ["is_fraud"]
)


@app.on_event("startup")
def log_startup():
    logger.info("API starting up", extra={"extra_fields": {"active_version": store.version}})
    # Establishes the Redis connection now, once per process at boot — without
    # this, the FIRST .delay() call on each of the 4 worker processes pays
    # this connection-setup cost inline during a real user's request instead.
    try:
        with celery_app.connection_or_acquire() as conn:
            conn.ensure_connection(max_retries=3)
        logger.info("Celery broker connection warmed up")
    except Exception as e:
        logger.warning("Celery broker warm-up failed", extra={"extra_fields": {"error": str(e)}})


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = round((time.monotonic() - start) * 1000, 2)
    logger.info(
        "request handled",
        extra={"extra_fields": {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        }},
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/status", response_model=ModelStatusResponse)
def model_status():
    return ModelStatusResponse(
        version=store.version,
        metrics=store.metrics,
        created_at=store.created_at,
    )


@app.post("/model/reload", dependencies=[Depends(verify_api_key)])
def model_reload():
    """Call this after activating a new version via evaluate.py --activate,
    so serving picks it up without restarting the container."""
    try:
        store.reload()
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"reloaded": True, "version": store.version}


@app.get("/model/drift", dependencies=[Depends(verify_api_key)])
def model_drift(limit: int = 500):
    reference_df = pd.read_csv(store.reference_data_path)
    recent = get_recent_predictions(store.version, limit=limit)
    current_df = pd.DataFrame([json.loads(r.features) for r in recent])
    return compute_drift(reference_df, current_df)


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
def predict(request: Request, transaction: TransactionInput):
    is_fraud, probability = store.predict(transaction.model_dump())
    PREDICTIONS_TOTAL.labels(is_fraud=str(is_fraud)).inc()
    logger.info(
        "prediction made",
        extra={"extra_fields": {
            "model_version": store.version,
            "is_fraud": is_fraud,
            "fraud_probability": round(probability, 4),
        }},
    )
    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(probability, 4),
        model_version=store.version,
    )


@app.post(
    "/predict/explain",
    response_model=ExplanationResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("10/minute")  # SHAP is heavier than a plain predict — tighter limit
def predict_explain(request: Request, transaction: TransactionInput):
    features = transaction.model_dump()
    is_fraud, probability = store.predict(features)
    top_features = store.explain(features)
    logger.info(
        "explanation generated",
        extra={"extra_fields": {"model_version": store.version, "is_fraud": is_fraud}},
    )
    return ExplanationResponse(
        is_fraud=is_fraud,
        fraud_probability=round(probability, 4),
        model_version=store.version,
        top_features=top_features,
    )