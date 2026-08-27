# Fraud Detection ML Platform

![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen) ![Tests](https://img.shields.io/badge/tests-42%20passed-brightgreen) ![Python](https://img.shields.io/badge/python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-enabled-009688) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon-blue) ![Redis](https://img.shields.io/badge/Redis-7-red) ![Celery](https://img.shields.io/badge/Celery-enabled-green) ![Kafka](https://img.shields.io/badge/Kafka-enabled-black) ![Docker](https://img.shields.io/badge/Docker-enabled-blue) ![License](https://img.shields.io/badge/license-MIT-yellow)


[![CI](https://github.com/andugetachew/fraud-detection-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/andugetachew/fraud-detection-ml/actions)
[![Run in Postman](https://run.pstmn.io/button.svg)](https://github.com/andugetachew/fraud-detection

## 🌐 Live

| | URL |
|--|--|
| **API** | https://fraud-detection-ml-api-l8s7.onrender.com |
| **Docs** | https://fraud-detection-ml-api-l8s7.onrender.com/docs |
| **Health** | https://fraud-detection-ml-api-l8s7.onrender.com/health |

- 42 automated tests
- 95% code coverage
- Unit + integration test layers, including live FastAPI endpoint tests (auth, rate limiting, predictions) and Kafka streaming consumer/producer tests
- 0 failed tests

Production-ready fraud detection system built with XGBoost, FastAPI, Celery, Redis, Docker, and Prometheus, featuring a custom FastAPI model-serving layer with versioned model loading, prediction and explanation endpoints, health/status monitoring, drift detection, and hot model reload.

## Architecture

Dataset
    │
    ▼
Training Pipeline
    │
    ▼
Model Registry
    │
    ▼
FastAPI Inference API
    │
    ├── SHAP Explainability
    ├── Prometheus Metrics
    ├── Authentication
    ├── Rate Limiting
    └── Drift Monitoring (Evidently)

## Features

- Production-ready FastAPI inference API
- XGBoost fraud detection model
- SQLite / PostgreSQL model registry
- Celery background jobs
- Redis task queue
- SHAP explainability
- Evidently drift detection
- Prometheus metrics
- Structured JSON logging
- API key authentication
- Rate limiting
- Docker Compose deployment
- Automated tests (95.6% coverage)

## Setup

```
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env       # optional — defaults to local SQLite if skipped
```

## Get the dataset

Download `creditcard.csv` from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place it at `data/creditcard.csv`.

## Train a model

```
cd training
python train.py
```

This trains an XGBoost classifier, evaluates it on a holdout set, saves
the artifact to `artifacts/`, and registers the version in the
`model_versions` table (SQLite locally, Neon Postgres in prod via
`DATABASE_URL`). It does NOT auto-activate the version — review the
printed metrics first.

## Review and activate a version

```
python evaluate.py
python evaluate.py --activate v20260728_143000
```

Only the active version is picked up by the serving layer (added next).

## Docker + Celery + Redis

Training now also runs as a background Celery task instead of only as a
blocking script — same async pattern as your other projects.

```
docker compose up --build
```

This starts `redis`, a `worker`, and a `beat` scheduler (idle by default —
no schedule is registered until you uncomment one in `training/tasks.py`).

To trigger training as a background task instead of `python train.py` directly:

```python
from tasks import train_task
train_task.delay(activate=False)
```

With `DEBUG=true` (the default), `CELERY_TASK_ALWAYS_EAGER=True`, so tasks
run synchronously in-process — no worker needed for local testing. Set
`DEBUG=false` in `.env` to use the real worker/broker.

## Streaming Ingestion (Kafka)

In addition to the REST `/predict` endpoint, transactions can be scored as a
live stream — closer to how production fraud systems actually work, where
transactions arrive as events rather than one-by-one API calls.

- **`streaming/producer.py`** replays `data/creditcard.csv` onto a Kafka
  `transactions` topic, simulating live transaction arrivals (the fraud
  label is dropped from the payload — a real ingestion pipeline wouldn't
  know it yet).
- **`streaming/consumer.py`** consumes that topic and scores each
  transaction using the *same* `ModelStore` the REST API uses
  (`serving/predict.py`) — one scoring path, no duplicated model logic
  between the two ingestion methods.

## Serving (FastAPI)

Runs as its own container (`api` service) alongside `worker`/`beat`.

```
docker compose up --build
```

Endpoints (once a model version is activated — see above):

- `POST /predict` — body matches the dataset schema (`Time`, `V1`..`V28`, `Amount`); returns `is_fraud`, `fraud_probability`, `model_version`
- `GET /model/status` — currently active version + its metrics
- `POST /model/reload` — reloads the active model in whichever worker
  process handles the request. Since `api` runs 4 uvicorn worker
  processes (each with its own in-memory model copy) for real
  concurrency, a single `/model/reload` call does NOT guarantee every
  process picks up the new version — after activating a new model,
  `docker compose restart api` is the reliable way to make every worker
  process load it
- `GET /health`

Model and scaler are saved together in the same artifact file — the
scaler fitted during training on `Time`/`Amount` is reused at inference
time rather than being refit, which would silently corrupt predictions.

## Load testing

`load_testing/locustfile.py` — simulates concurrent traffic against
`/predict`, `/predict/explain`, and `/health` with realistic, jittered
payloads.

```
pip install -r load_testing/requirements-load.txt
set LOAD_TEST_API_KEY=<your-key>
locust -f load_testing/locustfile.py --host http://localhost:8088
```

Open http://localhost:8089 to set concurrency and watch live RPS/latency,
or run headless for a fixed duration:
```
locust -f load_testing/locustfile.py --host http://localhost:8088 --users 50 --spawn-rate 5 --run-time 2m --headless
```

`/predict` is rate-limited to 20/min per IP — since Locust runs from one
machine (one IP), you'll see 429s once that's hit at real concurrency.
That's the rate limiter working correctly, not a failure; the test
treats 429 as a pass so it doesn't pollute the error rate, while still
surfacing real throughput/latency for traffic under the limit.

**Real results (20 concurrent users):** 15.9 RPS sustained, 0% hard
failures. `/predict` median latency was fast (50ms) but p95/p99 were
much worse (3.6s / 4.3s) — investigated with `docker stats` during a
run: CPU was barely used (~1.5%), ruling out compute as the cause.
Combined with `/health` staying fast on the same container, this pointed
to I/O wait, not CPU: `/predict` was synchronously writing to Neon
(over the public internet) *inside* the request path before responding.
Fixed by moving that write to a Celery background task
(`tasks.log_prediction_task`, fired via `.delay()`).

Getting that fix to actually take effect surfaced a second, real bug:
`training/config.py` had `CELERY_TASK_ALWAYS_EAGER = not DEBUG` — the
exact opposite of the intended `DEBUG=true` → eager (local convenience,
no worker needed) relationship documented above. `DEBUG=false` was
silently forcing tasks to run eagerly/inline instead of asynchronously,
so the "async" fix was still executing the Neon write synchronously the
whole time. Fixed to `CELERY_TASK_ALWAYS_EAGER = DEBUG`, with a
regression test (`tests/test_config.py`) locking in the correct
relationship going forward.

**Re-verified after the fix, under sustained load (20 users, 12,675
total requests):** `/predict` p95 56ms / p99 65ms, `/predict/explain`
p95 71ms / p99 90ms, 16.1 RPS sustained, 0% failures. Down from p95
11,000ms before the fix — roughly a 190x reduction in tail latency.

## Observability (Prometheus)

`GET /metrics` (unauthenticated — a scraper needs network-level access
control, not an API key meant for business endpoints) exposes standard
HTTP metrics (request count, latency histograms, status codes per route)
via `prometheus-fastapi-instrumentator`, plus a custom business metric:
`fraud_predictions_total{is_fraud="true"|"false"}` — lets you graph
fraud rate over time, not just request volume.

## Docker hardening

Multi-stage build: build tools (`gcc`) only exist in the build stage and
never ship in the final image. Runs as a non-root user (`appuser`) —
fixes the `SecurityWarning: running as superuser` Celery showed earlier
when everything ran as root.

## Explainability (SHAP)

`POST /predict/explain` returns the same prediction as `/predict`, plus
the top 10 features driving that specific decision (SHAP value + the
feature's actual value), via `shap.TreeExplainer` — fast for tree
models since it doesn't need a background dataset like `KernelExplainer`
would. Rate-limited tighter than `/predict` (10/min vs 20/min) since SHAP
computation is heavier per request.

## Auth & rate limiting

- API key required (`X-API-Key` header) on `/predict`, `/model/reload`,
  `/model/drift`. Fails closed — if `API_KEY` isn't set on the server,
  those endpoints reject everything rather than silently running open.
  `/health` and `/model/status` stay public.
- `/predict` is rate-limited to 20 requests/minute per client IP (via
  `slowapi`) — protects a real-cost inference endpoint from abuse.

Generate a key:
```
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
Set it as `API_KEY` in `.env`, then include it on requests:
```
curl -X POST http://localhost:8088/predict -H "X-API-Key: <key>" -H "Content-Type: application/json" -d "..."
```

## Data validation

`training/data_validation.py` validates the raw dataset (via Pandera)
immediately after loading, before it reaches feature engineering or
training — column types, non-negative `Time`/`Amount`, `Class` in
{0, 1}, no missing/extra columns, no nulls. This is separate from
`serving/schemas.py`, which validates individual API requests — this
validates the whole training dataset's shape and quality at ingestion.

## Structured logging

Every service (`api`, `worker`, `beat`) logs JSON to stdout via
`training/logging_config.py` — one log line per event, machine-parseable
by any real log aggregator (CloudWatch, Loki, Datadog, etc.), not free
text. `evaluate.py` stays on plain `print()` deliberately — it's a
human-facing CLI tool, not a service.

`api` also logs every request (method, path, status, duration) via
middleware, and every prediction outcome (version, is_fraud, probability).

## Monitoring

Every `/predict` call is logged (features + output) to the
`prediction_logs` table. Drift is checked with **Evidently**
(`DataDriftPreset`), which runs a proper statistical test per column
(KS-test, PSI, etc. — auto-selected by column type/size) comparing a
saved reference sample of real training rows against recent live
prediction inputs. This replaced an earlier hand-rolled z-score check —
maintaining two drift implementations side by side wasn't worth it once
a recognized library did the job better.

- `GET /model/drift` — on-demand check via the API. Returns
  `insufficient_data: true` (no drift computed) when fewer than 30 recent
  predictions exist — statistical tests on tiny samples produce false
  positives (a single point vs. a distribution nearly always "fails"),
  which isn't real drift, just a sample-size artifact.
- `beat` runs `tasks.check_drift_task` automatically every hour and logs
  a warning if drift is flagged (`docker compose logs beat`)
- Retraining is NOT triggered automatically on drift — that stays a
  manual decision (`tasks.train_task` / `train.py`), since auto-retraining
  on a drift signal without human review is how you silently ship a
  worse model.

## Design notes

- `scale_pos_weight` handles the ~0.17% class imbalance in this dataset
  directly in XGBoost rather than resampling (SMOTE etc.) — simpler and
  works well for this dataset size.
- The registry is one table, no MLflow server — matches the scale of
  this project. Serving reads `is_active=True` to know which artifact
  to load.
- Each model version stores a CSV sample (`REFERENCE_SAMPLE_SIZE=5000`
  rows) of raw training features — Evidently needs real reference rows
  to run its statistical tests, not just summary stats.

---

## 📄 License

MIT License

---

  ---

**Andualem Getachew**
[![GitHub](https://img.shields.io/badge/GitHub-andugetachew-black?logo=github)](https://github.com/andugetachew)
[![Email](https://img.shields.io/badge/Email-andugeta41%40gmail.com-red?logo=gmail)](mailto:andugeta41@gmail.com)